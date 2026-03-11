from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader

from monai.metrics import DiceMetric
from training.loss import SegmentationLoss


# ──────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch one-hot
# ──────────────────────────────────────────────────────────────────────────────

def _one_hot(indices: torch.Tensor, C: int) -> torch.Tensor:
    """(B,1,H,W) int64 → (B,C,H,W) float32 via scatter_."""
    B, _, H, W = indices.shape
    oh = torch.zeros(B, C, H, W, device=indices.device, dtype=torch.float32)
    oh.scatter_(1, indices.long(), 1.0)
    return oh


def _batch_dice(logits: torch.Tensor, labels: torch.Tensor, C: int) -> float:
    """Mean foreground Dice for one batch."""
    pred_oh  = _one_hot(torch.argmax(logits, 1, keepdim=True), C)
    label_oh = _one_hot(labels.unsqueeze(1), C)
    dm = DiceMetric(include_background=False, reduction="mean")
    dm(y_pred=pred_oh, y=label_oh)
    val = dm.aggregate().item()
    dm.reset()
    return val


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warmup → cosine anneal
# ──────────────────────────────────────────────────────────────────────────────

def _build_scheduler(optimizer, warmup_epochs: int, total_epochs: int, lr_min: float, base_lr: float):
    warmup = LambdaLR(
        optimizer,
        lr_lambda=lambda e: (e + 1) / max(warmup_epochs, 1)
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max   = max(total_epochs - warmup_epochs, 1),
        eta_min = lr_min,
    )
    return SequentialLR(
        optimizer,
        schedulers  = [warmup, cosine],
        milestones  = [warmup_epochs],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_round(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    loss_fn:      SegmentationLoss,
    cfg:          dict,
    device:       torch.device,
    al_cycle:     int,
    save_dir:     str | Path,
) -> Tuple[float, Dict[str, List[float]]]:

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t_cfg        = cfg["training"]
    num_epochs   = t_cfg["epochs"]
    base_lr      = float(t_cfg["lr"])
    lr_min       = float(t_cfg.get("lr_min", 1e-6))
    weight_decay = float(t_cfg.get("weight_decay", 1e-4))
    grad_clip    = float(t_cfg.get("grad_clip", 1.0))
    warmup_ep    = int(t_cfg.get("warmup_epochs", 5))
    use_amp      = bool(t_cfg.get("amp", True)) and device.type == "cuda"
    C            = cfg["model"]["out_channels"]

    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = _build_scheduler(optimizer, warmup_ep, num_epochs, lr_min, base_lr)
    scaler    = GradScaler('cuda', enabled=use_amp)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
    }
    best_val_dice  = -1.0
    best_ckpt_path = save_dir / f"cycle_{al_cycle:02d}_best.pth"

    for epoch in range(1, num_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t0 = time.time()
        tr_loss, tr_dice, nb = 0.0, 0.0, 0

        for batch in train_loader:
            images = batch["image"].to(device)   # (B,1,H,W)
            labels = batch["label"].to(device)   # (B,H,W)

            optimizer.zero_grad()
            with autocast('cuda', enabled=use_amp):
                logits = model(images)
                loss   = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                tr_loss += loss.item()
                tr_dice += _batch_dice(logits.float().detach(), labels, C)
            nb += 1

        scheduler.step()
        avg_tr_loss = tr_loss / max(nb, 1)
        avg_tr_dice = tr_dice / max(nb, 1)
        current_lr  = scheduler.get_last_lr()[0]

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        vl_loss, vl_dice, nv = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                with autocast('cuda', enabled=use_amp):
                    logits = model(images)
                vl_loss += loss_fn(logits.float(), labels).item()
                vl_dice += _batch_dice(logits.float(), labels, C)
                nv += 1

        avg_vl_loss = vl_loss / max(nv, 1)
        avg_vl_dice = vl_dice / max(nv, 1)

        history["train_loss"].append(avg_tr_loss)
        history["val_loss"].append(avg_vl_loss)
        history["train_dice"].append(avg_tr_dice)
        history["val_dice"].append(avg_vl_dice)

        print(
            f"  Ep {epoch:>3}/{num_epochs} | "
            f"Loss {avg_tr_loss:.4f}/{avg_vl_loss:.4f} | "
            f"Dice {avg_tr_dice:.4f}/{avg_vl_dice:.4f} | "
            f"lr={current_lr:.2e} | {time.time()-t0:.1f}s"
        )

        if avg_vl_dice > best_val_dice:
            best_val_dice = avg_vl_dice
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_dice":   best_val_dice,
                "al_cycle":   al_cycle,
            }, best_ckpt_path)

    print(f"  [Cycle {al_cycle}] Best val Dice: {best_val_dice:.4f} → {best_ckpt_path}")

    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return best_val_dice, history