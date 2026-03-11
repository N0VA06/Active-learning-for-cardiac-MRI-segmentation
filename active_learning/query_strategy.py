from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class QueryStrategy(ABC):
    @abstractmethod
    def select(self, unlabeled_ids: List[str],
               uncertainty_scores: Dict[str, float],
               k: int) -> Tuple[List[str], List[str]]: ...


class EntropyQuery(QueryStrategy):
    """Top-K patients by mean slice entropy."""

    def select(self, unlabeled_ids, uncertainty_scores, k):
        p2s: Dict[str, List[str]] = {}
        for sid in unlabeled_ids:
            p2s.setdefault(sid.split("_frame")[0], []).append(sid)

        p_scores = {
            p: sum(uncertainty_scores.get(s, 0.0) for s in sl) / max(len(sl), 1)
            for p, sl in p2s.items()
        }
        ranked  = sorted(p_scores, key=p_scores.__getitem__, reverse=True)
        sel_p   = ranked[:k]
        sel_s   = [s for p in sel_p for s in p2s[p]]
        rem     = [s for s in unlabeled_ids if s not in set(sel_s)]
        print(
            f"  [Query] {len(sel_p)} patient(s) ({len(sel_s)} slices) | "
            "top: " + ", ".join(f"{p}={p_scores[p]:.4f}" for p in sel_p[:3])
        )
        return sel_s, rem


class RandomQuery(QueryStrategy):
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def select(self, unlabeled_ids, uncertainty_scores, k):
        p2s: Dict[str, List[str]] = {}
        for sid in unlabeled_ids:
            p2s.setdefault(sid.split("_frame")[0], []).append(sid)
        pats = list(p2s.keys())
        self._rng.shuffle(pats)
        sel = [s for p in pats[:k] for s in p2s[p]]
        rem = [s for s in unlabeled_ids if s not in set(sel)]
        return sel, rem


def build_query_strategy(cfg: dict) -> QueryStrategy:
    method = cfg.get("active_learning", {}).get("uncertainty", "entropy")
    if method in ("entropy", "bald"):
        return EntropyQuery()
    return RandomQuery(seed=cfg["training"]["seed"])
