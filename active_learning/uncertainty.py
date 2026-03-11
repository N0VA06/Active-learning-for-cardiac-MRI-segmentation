from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def predictive_entropy(prob_mean: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy of mean MC probs.  (B,C,H,W) → (B,H,W)."""
    return -(prob_mean * torch.log(prob_mean + eps)).sum(dim=1)


def mutual_information(prob_stack: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """BALD.  (T,B,C,H,W) → (B,H,W)."""
    mean_p = prob_stack.mean(0)
    H_mean = predictive_entropy(mean_p, eps)
    H_each = -(prob_stack * torch.log(prob_stack + eps)).sum(2)   # (T,B,H,W)
    return H_mean - H_each.mean(0)


@torch.no_grad()
def mc_dropout_inference(
    model:  nn.Module,
    loader: DataLoader,
    T:      int,
    device: torch.device,
    method: str = "entropy",
) -> Dict[str, float]:
    model.eval()
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout()
    else:
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    scores: Dict[str, float] = {}

    for batch in loader:
        images    = batch["image"].to(device)
        slice_ids = batch["slice_id"]

        samples: List[torch.Tensor] = []
        for _ in range(T):
            probs = F.softmax(model(images).float(), dim=1)
            samples.append(probs.cpu())

        stack = torch.stack(samples, 0)       # (T,B,C,H,W)
        mean  = stack.mean(0)                 # (B,C,H,W)

        unc = (predictive_entropy(mean) if method == "entropy"
               else mutual_information(stack))

        for sid, s in zip(slice_ids, unc.mean(dim=[1, 2]).tolist()):
            scores[sid] = s

    if hasattr(model, "disable_mc_dropout"):
        model.disable_mc_dropout()

    return scores
