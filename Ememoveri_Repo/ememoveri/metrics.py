"""
Standard information-retrieval metrics, used by the E5-M-style retrieval
quality evaluation (and reusable for a real E5 run once ground-truth
relevant-claim labels exist per task).

Deliberately dependency-free (no numpy/sklearn) so ememoveri's core stays
zero-dependency; matplotlib/numpy are only required for `ememoveri.figures`.
"""

from __future__ import annotations

import math
from typing import Sequence


def precision_at_k(ranked_relevance: Sequence[bool], k: int) -> float:
    """ranked_relevance[i] = True if the i-th ranked item is relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = ranked_relevance[:k]
    if not top_k:
        return 0.0
    return sum(top_k) / len(top_k)


def recall_at_k(ranked_relevance: Sequence[bool], k: int, n_relevant_total: int) -> float:
    if n_relevant_total <= 0:
        return float("nan")
    top_k = ranked_relevance[:k]
    return sum(top_k) / n_relevant_total


def reciprocal_rank(ranked_relevance: Sequence[bool]) -> float:
    for i, rel in enumerate(ranked_relevance, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked_gains: Sequence[float], k: int) -> float:
    dcg = 0.0
    for i, gain in enumerate(ranked_gains[:k], start=1):
        dcg += gain / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_gains: Sequence[float], k: int) -> float:
    ideal = sorted(ranked_gains, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_gains, k) / idcg


def mean(values: Sequence[float]) -> float:
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)
