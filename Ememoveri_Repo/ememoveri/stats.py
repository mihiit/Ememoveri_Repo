"""Small, dependency-light statistics for paired benchmark comparisons."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    """Mean paired difference (a-b) with a 95% percentile bootstrap CI over tasks."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact(a: Sequence[float], b: Sequence[float]) -> Tuple[int, int, float]:
    """Exact two-sided McNemar on paired binary outcomes. Returns (b01, b10, p)."""
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return n01, n10, 1.0
    k = min(n10, n01)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n01, n10, min(1.0, p)


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for r, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - r) * p))
        out[k] = running
    return out


def risk_coverage(conf: Sequence[float], correct: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Selective-prediction curve: answer the most confident fraction first."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf, kind="stable")
    c = correct[order]
    cov = np.arange(1, len(c) + 1) / len(c)
    acc = np.cumsum(c) / np.arange(1, len(c) + 1)
    return cov, acc


def aurc(conf, correct) -> float:
    """Area under the risk-coverage curve (risk = 1-accuracy); lower is better."""
    cov, acc = risk_coverage(conf, correct)
    return float(np.mean(1.0 - acc))


def auroc(score: Sequence[float], label: Sequence[float]) -> float:
    s, y = np.asarray(score, float), np.asarray(label) > 0.5
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(prob: Sequence[float], correct: Sequence[float], bins: int = 10) -> float:
    p, y = np.asarray(prob, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def brier(prob, correct) -> float:
    p, y = np.asarray(prob, float), np.asarray(correct, float)
    return float(np.mean((p - y) ** 2))


def signflip_p(d: Sequence[float], n_mc: int = 200000, seed: int = 0) -> float:
    """Two-sided exact sign-flip permutation test for the mean of paired differences d (unit = repository).
    Exact enumeration for n <= 16, Monte Carlo otherwise."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    obs = abs(d.mean())
    if n <= 16:
        signs = (((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1).astype(float)
        stats = np.abs((signs * d).mean(1))
        return float((stats >= obs - 1e-12).mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_mc, n))
    return float(((np.abs((signs * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n_mc + 1))


def cohens_dz(d: Sequence[float]) -> float:
    """Standardised paired effect size: mean difference over the standard deviation of the differences."""
    d = np.asarray(d, dtype=float)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return float(d.mean() / sd) if sd > 0 else float("inf") if d.mean() != 0 else 0.0
