from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import evaluate_model
from visualization.segmentation_visualizer import save_segmentation_comparison


def load_best_checkpoint(
    model:      nn.Module,
    models_dir: str | Path,
    cycle:      Optional[int] = None,
    device:     torch.device  = torch.device("cpu"),
) -> nn.Module:
    models_dir = Path(models_dir)
    ckpts = sorted(models_dir.glob("cycle_*_best.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {models_dir}")
    ckpt_path = (
        models_dir / f"cycle_{cycle:02d}_best.pth"
        if cycle is not None else ckpts[-1]
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def run_inference(
    model:     nn.Module,
    loader:    DataLoader,
    cfg:       dict,
    device:    torch.device,
    preds_dir: str | Path,
    plots_dir: Optional[str | Path] = None,
    max_vis:   int = 10,
) -> Dict:
    preds_dir = Path(preds_dir)
    preds_dir.mkdir(parents=True, exist_ok=True)
    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    vis_count = 0

    for batch in loader:
        images    = batch["image"].to(device)
        labels    = batch["label"]
        slice_ids = batch.get("slice_id",
                              [f"slice_{i}" for i in range(images.shape[0])])

        logits = model(images).float()
        preds  = torch.argmax(logits, 1)   # (B,H,W)

        for sid, img_t, lbl_t, pred_t in zip(
            slice_ids, images.cpu(), labels.cpu(), preds.cpu()
        ):
            np.save(preds_dir / f"{sid}_pred.npy",
                    pred_t.numpy().astype(np.uint8))

            if plots_dir is not None and vis_count < max_vis:
                save_segmentation_comparison(
                    image=img_t.numpy()[0],
                    gt_mask=lbl_t.numpy(),
                    pred_mask=pred_t.numpy(),
                    slice_id=sid,
                    save_path=plots_dir / f"{sid}_comparison.png",
                )
                vis_count += 1

    return evaluate_model(
        model=model, loader=loader, cfg=cfg, device=device,
        save_preds=True, preds_dir=str(preds_dir),
    )
