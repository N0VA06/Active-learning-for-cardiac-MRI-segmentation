"""
active_learning/query_strategy.py
==================================
Query strategies for patient-level active learning selection.

Strategies
----------
EntropyQuery             – top-K by mean slice uncertainty score (any signal)
RandomQuery              – random patient baseline
MultiplicativeUDQuery    – top-K by U×D score  ← NEW recommended default

``build_query_strategy`` dispatch:
    "mul"      → MultiplicativeUDQuery
    "combined" → EntropyQuery  (scores already computed externally)
    "entropy"  → EntropyQuery
    "bald"     → EntropyQuery
    anything   → RandomQuery
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class QueryStrategy(ABC):
    @abstractmethod
    def select(
        self,
        unlabeled_ids:      List[str],
        uncertainty_scores: Dict[str, float],
        k:                  int,
    ) -> Tuple[List[str], List[str]]: ...


# ──────────────────────────────────────────────────────────────────────────────
# Entropy / Top-K  (works for any pre-computed scalar score)
# ──────────────────────────────────────────────────────────────────────────────

class EntropyQuery(QueryStrategy):
    """Select k patients with the highest mean slice score."""

    def select(self, unlabeled_ids, uncertainty_scores, k):
        p2s: Dict[str, List[str]] = {}
        for sid in unlabeled_ids:
            p2s.setdefault(sid.split("_frame")[0], []).append(sid)

        p_scores = {
            p: sum(uncertainty_scores.get(s, 0.0) for s in sl) / max(len(sl), 1)
            for p, sl in p2s.items()
        }
        ranked = sorted(p_scores, key=p_scores.__getitem__, reverse=True)
        sel_p  = ranked[:k]
        sel_s  = [s for p in sel_p for s in p2s[p]]
        rem    = [s for s in unlabeled_ids if s not in set(sel_s)]

        print(
            f"  [Query] {len(sel_p)} patient(s) ({len(sel_s)} slices)  |  "
            "top-3: " + ", ".join(f"{p}={p_scores[p]:.4f}" for p in sel_p[:3])
        )
        return sel_s, rem


# ──────────────────────────────────────────────────────────────────────────────
# Multiplicative U×D  (NEW)
# ──────────────────────────────────────────────────────────────────────────────

class MultiplicativeUDQuery(EntropyQuery):
    """
    Select k patients by their U×D score (uncertainty × diversity).

    The U×D scores are computed externally by
    ``multiplicative_ud_uncertainty`` and passed in via ``uncertainty_scores``.
    Selection logic is identical to ``EntropyQuery`` (mean score per patient,
    then top-K) — this class exists as a distinct symbol for logging clarity.

    Why multiplicative is strictly better than additive
    ---------------------------------------------------
    Additive:      score = α·U + β·D
      Problem: a confident (U≈0) but highly diverse sample still scores ~β·D
               and can be selected — wasting a labeling budget on a sample
               the model doesn't need.

    Multiplicative: score = U × D
      Enforces AND logic: both signals must be non-zero to score.
      • Confident + diverse   → 0   (don't waste label on easy sample)
      • Uncertain + redundant → 0   (don't re-label what you already have)
      • Uncertain + diverse   → high score  (exactly what AL should pick)
    """

    def select(self, unlabeled_ids, uncertainty_scores, k):
        print("  [Query] Strategy: Multiplicative U×D")
        return super().select(unlabeled_ids, uncertainty_scores, k)


# ──────────────────────────────────────────────────────────────────────────────
# Random baseline
# ──────────────────────────────────────────────────────────────────────────────

class RandomQuery(QueryStrategy):
    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def select(self, unlabeled_ids, uncertainty_scores, k):
        p2s: Dict[str, List[str]] = {}
        for sid in unlabeled_ids:
            p2s.setdefault(sid.split("_frame")[0], []).append(sid)
        pats = list(p2s.keys())
        self._rng.shuffle(pats)
        sel = [s for p in pats[:k] for s in p2s[p]]
        rem = [s for s in unlabeled_ids if s not in set(sel)]
        print(f"  [Query] Strategy: Random  |  {len(pats[:k])} patient(s)")
        return sel, rem


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_query_strategy(cfg: dict) -> QueryStrategy:
    method = cfg.get("active_learning", {}).get("uncertainty", "mul")

    if method == "mul":
        return MultiplicativeUDQuery()
    if method in ("entropy", "bald", "combined"):
        return EntropyQuery()
    return RandomQuery(seed=cfg["training"]["seed"])