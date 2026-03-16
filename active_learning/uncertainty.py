"""
active_learning/uncertainty.py
===============================
Uncertainty estimation strategies for active learning query selection.

Functions
---------
predictive_entropy            – Shannon entropy of MC mean probabilities
mutual_information            – BALD / JSD across a stack of distributions
mc_dropout_inference          – Original MC-Dropout (final exit only)
combined_mc_exit_uncertainty  – α·MC + β·exit-disagreement  (additive)
multiplicative_ud_uncertainty – U × D  (uncertainty × diversity)  ← NEW

Multiplicative U×D Rationale
-----------------------------
Additive combination (α·U + β·D) has a fundamental flaw: a sample with
zero uncertainty but high diversity, or high uncertainty but zero novelty,
can still accumulate a non-trivial score and be selected.

Multiplicative scoring enforces a logical AND:
    score(x) = U_norm(x) × D_norm(x)

    • U = 0  →  score = 0   (confident; useless regardless of novelty)
    • D = 0  →  score = 0   (redundant; useless regardless of difficulty)
    • U > 0  AND  D > 0  →  score > 0   (uncertain AND novel → worth labeling)

Diversity is measured in the model's own feature space — the global-average-
pooled output of the deepest hooked decoder layer (available from
EarlyExitWrapper's hooks at zero extra cost).  For plain UNets the softmax
output serves as a C-dimensional proxy feature.

Patient-level scores
--------------------
Both U and D are averaged across a patient's slices before multiplication
so that one outlier slice does not dominate the patient's score.

Diversity per patient
---------------------
    D(slice) = 1 - max_{s ∈ labeled_pool} cosine_sim( feat(slice), feat(s) )
             = cosine distance to the nearest labeled neighbour

Global min-max normalisation is applied to U and D independently before
multiplication so neither dominates due to scale.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Core uncertainty primitives
# ──────────────────────────────────────────────────────────────────────────────

def predictive_entropy(prob_mean: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy of mean MC probability distribution.
    (B, C, H, W) → (B, H, W)
    """
    return -(prob_mean * torch.log(prob_mean + eps)).sum(dim=1)


def mutual_information(prob_stack: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """BALD / JSD across K distributions.
    (K, B, C, H, W) → (B, H, W)
    """
    mean_p = prob_stack.mean(0)
    H_mean = predictive_entropy(mean_p, eps)
    H_each = -(prob_stack * torch.log(prob_stack + eps)).sum(2)   # (K, B, H, W)
    return H_mean - H_each.mean(0)


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction helper
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _extract_features(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    One deterministic forward pass; returns {slice_id: L2-normalised feature vec}.

    Feature source (priority):
      1. EarlyExitWrapper: GAP of deepest captured decoder feature
      2. Fallback: mean softmax probability vector over spatial dims
    """
    has_exits = hasattr(model, "set_exit_mode")
    model.eval()

    if has_exits:
        model.set_exit_mode(True)

    feats: Dict[str, torch.Tensor] = {}

    for batch in loader:
        images    = batch["image"].to(device)
        slice_ids = batch["slice_id"]

        if has_exits:
            final_logits, _ = model(images)
            raw = model._captured[0]
            feat = (raw.mean(dim=[2, 3]).cpu() if raw is not None
                    else F.softmax(final_logits.float(), dim=1).mean(dim=[2, 3]).cpu())
        else:
            logits = model(images)
            feat   = F.softmax(logits.float(), dim=1).mean(dim=[2, 3]).cpu()

        feat = F.normalize(feat, p=2, dim=1)   # L2-normalise → cosine space

        for sid, fv in zip(slice_ids, feat):
            feats[sid] = fv

    if has_exits:
        model.set_exit_mode(False)

    return feats


# ──────────────────────────────────────────────────────────────────────────────
# Original MC-Dropout inference (unchanged – backward compatible)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def mc_dropout_inference(
    model:  nn.Module,
    loader: DataLoader,
    T:      int,
    device: torch.device,
    method: str = "entropy",
) -> Dict[str, float]:
    """Standard MC-Dropout over the final model exit."""
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
        samples   = []
        for _ in range(T):
            probs = F.softmax(model(images).float(), dim=1)
            samples.append(probs.cpu())

        stack = torch.stack(samples, 0)
        mean  = stack.mean(0)
        unc   = (predictive_entropy(mean) if method == "entropy"
                 else mutual_information(stack))

        for sid, s in zip(slice_ids, unc.mean(dim=[1, 2]).tolist()):
            scores[sid] = s

    if hasattr(model, "disable_mc_dropout"):
        model.disable_mc_dropout()

    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Combined MC + Exit disagreement (additive) – kept for ablation
# ──────────────────────────────────────────────────────────────────────────────

def combined_mc_exit_uncertainty(
    model:  nn.Module,
    loader: DataLoader,
    T:      int,
    device: torch.device,
    alpha:  float = 0.5,
    beta:   float = 0.5,
    method: str   = "entropy",
) -> Dict[str, float]:
    """α·U_mc + β·U_exit_disagree  (additive, both min-max normalised)."""
    has_exits = hasattr(model, "set_exit_mode")
    raw_mc:   Dict[str, float] = {}
    raw_exit: Dict[str, float] = {}

    model.eval()
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout()
    else:
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    with torch.no_grad():
        for batch in loader:
            images    = batch["image"].to(device)
            slice_ids = batch["slice_id"]
            samples   = []
            for _ in range(T):
                probs = F.softmax(model(images).float(), dim=1)
                samples.append(probs.cpu())
            stack = torch.stack(samples, 0)
            mean  = stack.mean(0)
            unc   = (predictive_entropy(mean) if method == "entropy"
                     else mutual_information(stack))
            for sid, s in zip(slice_ids, unc.mean(dim=[1, 2]).tolist()):
                raw_mc[sid] = s

    if has_exits:
        model.eval()
        model.set_exit_mode(True)
        with torch.no_grad():
            for batch in loader:
                images    = batch["image"].to(device)
                slice_ids = batch["slice_id"]
                final_logits, exit_list = model(images)
                all_p = [F.softmax(final_logits.float(), dim=1).cpu()]
                all_p += [F.softmax(el.float(), dim=1).cpu() for el in exit_list]
                dis = (mutual_information(torch.stack(all_p, 0))
                       if len(all_p) >= 2
                       else torch.zeros(all_p[0].shape[0], *all_p[0].shape[2:]))
                for sid, s in zip(slice_ids, dis.mean(dim=[1, 2]).tolist()):
                    raw_exit[sid] = s
        model.set_exit_mode(False)
    else:
        raw_exit = {k: 0.0 for k in raw_mc}

    if hasattr(model, "disable_mc_dropout"):
        model.disable_mc_dropout()

    sids  = list(raw_mc.keys())
    mc_v  = torch.tensor([raw_mc[s]   for s in sids], dtype=torch.float32)
    ex_v  = torch.tensor([raw_exit[s] for s in sids], dtype=torch.float32)

    def _mm(t):
        lo, hi = t.min(), t.max()
        return (t - lo) / (hi - lo + 1e-8)

    comb   = alpha * _mm(mc_v) + beta * _mm(ex_v)
    scores = {sid: float(comb[i]) for i, sid in enumerate(sids)}

    print(f"  [Uncertainty] MC μ={_mm(mc_v).mean():.4f}  "
          f"Exit μ={_mm(ex_v).mean():.4f}  α={alpha}  β={beta}")
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Multiplicative U × D  (NEW – recommended)
# ──────────────────────────────────────────────────────────────────────────────

def multiplicative_ud_uncertainty(
    model:             nn.Module,
    unlabeled_loader:  DataLoader,
    labeled_loader:    DataLoader,
    T:                 int,
    device:            torch.device,
    method:            str = "entropy",
) -> Dict[str, float]:
    """
    score(slice) = U_norm(slice) × D_norm(slice)

    U  = combined MC-Dropout + exit-disagreement uncertainty
         (plain MC entropy if model has no early exits)
    D  = cosine distance to the nearest slice in the labeled pool's
         feature space  (1 - max cosine similarity to any labeled slice)

    Both U and D are min-max normalised across the full unlabeled pool
    before multiplication.

    Parameters
    ----------
    unlabeled_loader : DataLoader over the unlabeled pool
    labeled_loader   : DataLoader over the current labeled pool
    T                : MC-Dropout forward passes for U estimation
    method           : 'entropy' | 'bald'

    Returns
    -------
    scores : {slice_id: U_norm × D_norm}
    """
    has_exits = hasattr(model, "set_exit_mode")

    # ── Pass 1: MC uncertainty ────────────────────────────────────────────
    print(f"  [U×D] Pass 1 – MC uncertainty  (T={T}) ...")

    raw_mc:   Dict[str, float] = {}
    raw_exit: Dict[str, float] = {}

    model.eval()
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout()
    else:
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    with torch.no_grad():
        for batch in unlabeled_loader:
            images    = batch["image"].to(device)
            slice_ids = batch["slice_id"]
            samples   = []
            for _ in range(T):
                probs = F.softmax(model(images).float(), dim=1)
                samples.append(probs.cpu())
            stack = torch.stack(samples, 0)
            mean  = stack.mean(0)
            unc   = (predictive_entropy(mean) if method == "entropy"
                     else mutual_information(stack))
            for sid, s in zip(slice_ids, unc.mean(dim=[1, 2]).tolist()):
                raw_mc[sid] = s

    if hasattr(model, "disable_mc_dropout"):
        model.disable_mc_dropout()

    # Exit disagreement (free if exits present)
    if has_exits:
        model.eval()
        model.set_exit_mode(True)
        with torch.no_grad():
            for batch in unlabeled_loader:
                images    = batch["image"].to(device)
                slice_ids = batch["slice_id"]
                final_logits, exit_list = model(images)
                all_p = [F.softmax(final_logits.float(), dim=1).cpu()]
                all_p += [F.softmax(el.float(), dim=1).cpu() for el in exit_list]
                dis = (mutual_information(torch.stack(all_p, 0))
                       if len(all_p) >= 2
                       else torch.zeros(all_p[0].shape[0], *all_p[0].shape[2:]))
                for sid, s in zip(slice_ids, dis.mean(dim=[1, 2]).tolist()):
                    raw_exit[sid] = s
        model.set_exit_mode(False)
    else:
        raw_exit = {k: 0.0 for k in raw_mc}

    # ── Pass 2: Feature extraction for diversity ──────────────────────────
    print("  [U×D] Pass 2 – feature extraction (unlabeled + labeled) ...")

    unl_feats = _extract_features(model, unlabeled_loader, device)
    lbl_feats = _extract_features(model, labeled_loader,   device)

    # ── Compute U and D, then multiply ───────────────────────────────────
    sids  = list(raw_mc.keys())
    mc_v  = torch.tensor([raw_mc[s]   for s in sids], dtype=torch.float32)
    ex_v  = torch.tensor([raw_exit[s] for s in sids], dtype=torch.float32)

    def _mm(t: torch.Tensor) -> torch.Tensor:
        lo, hi = t.min(), t.max()
        return (t - lo) / (hi - lo + 1e-8)

    # Combined U: average of normalised MC and exit signals
    u_norm = (_mm(mc_v) + _mm(ex_v)) / 2.0 if has_exits else _mm(mc_v)

    # Diversity: cosine distance to nearest labeled slice
    lbl_mat = torch.stack(list(lbl_feats.values()), dim=0)  # (L, F)

    d_vals: List[float] = []
    for sid in sids:
        fv   = unl_feats[sid].unsqueeze(0)           # (1, F)
        sim  = (fv * lbl_mat).sum(dim=1)             # (L,) cosine similarity
        dist = 1.0 - sim.max().item()                # distance to nearest
        d_vals.append(max(dist, 0.0))

    d_vec  = torch.tensor(d_vals, dtype=torch.float32)
    d_norm = _mm(d_vec)

    # U × D
    combined = u_norm * d_norm

    scores = {sid: float(combined[i]) for i, sid in enumerate(sids)}

    print(
        f"  [U×D] U  μ={u_norm.mean():.4f}  σ={u_norm.std():.4f}\n"
        f"  [U×D] D  μ={d_norm.mean():.4f}  σ={d_norm.std():.4f}\n"
        f"  [U×D] U×D μ={combined.mean():.4f}  "
        f"top-1={combined.max():.4f}  bottom-1={combined.min():.4f}"
    )
    return scores