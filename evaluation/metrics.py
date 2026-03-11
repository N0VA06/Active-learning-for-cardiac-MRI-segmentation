from __future__ import annotations
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _one_hot(indices: torch.Tensor, C: int) -> torch.Tensor:
    """(B,1,H,W) int64 → (B,C,H,W) float32."""
    B, _, H, W = indices.shape
    oh = torch.zeros(B, C, H, W, device=indices.device, dtype=torch.float32)
    oh.scatter_(1, indices.long(), 1.0)
    return oh

def _pred_oh(logits: torch.Tensor, C: int) -> torch.Tensor:
    return _one_hot(torch.argmax(logits, 1, keepdim=True), C)

def _lbl_oh(labels: torch.Tensor, C: int) -> torch.Tensor:
    return _one_hot(labels.unsqueeze(1), C)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model:      nn.Module,
    loader:     DataLoader,
    cfg:        dict,
    device:     torch.device,
    save_preds: bool = False,
    preds_dir:  Optional[str] = None,
) -> Dict:
    from pathlib import Path

    model.eval()
    C   = cfg["model"]["out_channels"]
    inc = cfg["evaluation"].get("include_background", False)
    hdp = cfg["evaluation"].get("hausdorff_percentile", 95)

    dice_m = DiceMetric(include_background=inc, reduction="mean_batch")
    hd_m   = HausdorffDistanceMetric(
        include_background=inc, percentile=hdp, reduction="mean_batch"
    )

    all_iou: List[float] = []
    all_prec: List[float] = []
    all_rec:  List[float] = []

    if save_preds and preds_dir:
        Path(preds_dir).mkdir(parents=True, exist_ok=True)

    for batch in loader:
        images    = batch["image"].to(device)
        labels    = batch["label"].to(device)
        slice_ids = batch.get("slice_id", [])

        logits  = model(images).float()
        pred_oh = _pred_oh(logits, C)
        lbl_oh  = _lbl_oh(labels, C)

        dice_m(y_pred=pred_oh, y=lbl_oh)

        # Suppress the "class 0 all zeros" MONAI warning — it is expected
        # behaviour when background is excluded and the model is still training.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*prediction of class.*all 0.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=".*nan/inf.*",
            )
            try:
                hd_m(y_pred=pred_oh.contiguous(), y=lbl_oh.contiguous())
            except Exception:
                pass

        start = 0 if inc else 1
        for c in range(start, C):
            inter = (pred_oh[:, c] * lbl_oh[:, c]).sum(dim=[1, 2]).float()
            union = (pred_oh[:, c] + lbl_oh[:, c]).clamp(0, 1).sum(dim=[1, 2]).float()
            all_iou.append((inter / (union + 1e-8)).mean().item())

        for c in range(start, C):
            tp = (pred_oh[:, c] * lbl_oh[:, c]).sum().float()
            fp = (pred_oh[:, c] * (1 - lbl_oh[:, c])).sum().float()
            fn = ((1 - pred_oh[:, c]) * lbl_oh[:, c]).sum().float()
            all_prec.append((tp / (tp + fp + 1e-8)).item())
            all_rec.append((tp / (tp + fn + 1e-8)).item())

        if save_preds and preds_dir and slice_ids:
            preds_np = torch.argmax(logits, 1).cpu().numpy()
            for sid, p in zip(slice_ids, preds_np):
                np.save(Path(preds_dir) / f"{sid}_pred.npy", p.astype(np.uint8))

    dice_arr = dice_m.aggregate().cpu().numpy()
    dice_m.reset()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hd_arr = hd_m.aggregate().cpu().numpy()
        hd_m.reset()
        mean_hd95 = float(np.nanmean(hd_arr))
    except Exception:
        mean_hd95 = float("nan")

    return {
        "mean_dice":      float(np.nanmean(dice_arr)),
        "mean_hd95":      mean_hd95,
        "mean_iou":       float(np.nanmean(all_iou))  if all_iou  else 0.0,
        "mean_precision": float(np.nanmean(all_prec)) if all_prec else 0.0,
        "mean_recall":    float(np.nanmean(all_rec))  if all_rec  else 0.0,
        "class_dice":     dice_arr.tolist(),
    }
