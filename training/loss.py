from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, TverskyLoss


class FocalLoss(nn.Module):
    """Per-pixel focal loss for multi-class segmentation."""

    def __init__(self, gamma: float = 2.0,
                 class_weights: list | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "weight",
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights else None,
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Move class weights to the same device as logits (handles CPU/CUDA)
        weight = self.weight.to(logits.device) if self.weight is not None else None
        # logits: (B,C,H,W)  labels: (B,H,W)
        ce = F.cross_entropy(logits, labels, weight=weight,
                             reduction="none")                      # (B,H,W)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class SegmentationLoss(nn.Module):

    def __init__(
        self,
        loss_type:      str   = "dice_focal",
        dice_weight:    float = 0.6,
        focal_weight:   float = 0.4,
        focal_gamma:    float = 2.0,
        tversky_alpha:  float = 0.3,   # FP weight  (<0.5 → penalise FP less)
        tversky_beta:   float = 0.7,   # FN weight  (>0.5 → penalise FN more)
        class_weights:  list | None = None,
    ) -> None:
        super().__init__()
        self.loss_type    = loss_type
        self.dice_w       = dice_weight
        self.focal_w      = focal_weight

        if loss_type == "dice_focal":
            self.tversky = TverskyLoss(
                include_background = False,
                to_onehot_y        = True,
                softmax            = True,
                alpha              = tversky_alpha,
                beta               = tversky_beta,
                reduction          = "mean",
            )
            self.focal = FocalLoss(gamma=focal_gamma,
                                   class_weights=class_weights)
        else:
            # Fallback: original DiceCE
            self.dicece = DiceCELoss(
                include_background = False,
                to_onehot_y        = True,
                softmax            = True,
                reduction          = "mean",
            )

    def forward(
        self,
        logits: torch.Tensor,   # (B, C, H, W)
        labels: torch.Tensor,   # (B, H, W) int64
    ) -> torch.Tensor:
        if self.loss_type == "dice_focal":
            tversky_loss = self.tversky(logits, labels.unsqueeze(1).float())
            focal_loss   = self.focal(logits, labels)
            return self.dice_w * tversky_loss + self.focal_w * focal_loss
        else:
            return self.dicece(logits, labels.unsqueeze(1).float())


def build_loss(cfg: dict) -> SegmentationLoss:
    lc = cfg.get("loss", {})
    return SegmentationLoss(
        loss_type     = lc.get("type",           "dice_focal"),
        dice_weight   = lc.get("dice_weight",    0.6),
        focal_weight  = lc.get("focal_weight",   0.4),
        focal_gamma   = lc.get("focal_gamma",    2.0),
        tversky_alpha = lc.get("tversky_alpha",  0.3),
        tversky_beta  = lc.get("tversky_beta",   0.7),
        class_weights = lc.get("class_weights",  [0.1, 1.0, 1.0, 1.0]),
    )