"""Evidence-level retrieval (EmemoVeri-2).

The original scorer ranks a claim by the similarity between the query and a five-subject summary of the claim.  EmemoVeri-2
scores a claim from its *evidence ledger*: every evidence item (a commit that touched the file, carried across renames) is
compared with the query on its own, and the item's contribution is weighted by what the ledger knows about it --
its strength (bulk commits tell less about any single file), how many files its commit touched (dilution), its type (TEST evidence), its age, and whether a later
revert cancelled it.  Claim similarity, a change-frequency prior and a recency prior are blended in; trust and abstention
are removed because they carried no signal (see the paper).

All functions here are pure numpy over the flat ledger exported by ``ExtMemory.ledger()``; a task at time tau sees only the
prefix of the ledger that existed at tau (claims with index < n_claims, commits with id < nc), so no future evidence is used.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class EvidenceConfig:
    agg: str = "max"                     # how item scores combine into a claim score: max | sum | noisyor
    strength_exp: float = 0.0            # item weight ~ strength ** strength_exp  (0 = ignore strength)
    dilution_exp: float = 0.0            # item weight ~ (files touched by the commit) ** -dilution_exp  (0 = ignore)
    test_bonus: float = 0.0              # item weight x (1 + test_bonus) for TEST evidence
    half_life_e: Optional[float] = None  # days; item weight x 2 ** (-age / half_life_e); None = no decay
    revert_cancel: bool = False          # items of commits later reverted get weight 0
    # blend
    w_sim: float = 1.0                   # claim similarity (hashed TF-IDF over the claim text)
    w_ev: float = 1.0                    # evidence-level score
    l_pop: float = 0.0                   # change-frequency prior  n / (n + 5)
    l_rec: float = 0.0                   # recency prior 2 ** (-age_days / H)
    H: float = 90.0

    def as_dict(self) -> Dict:
        return asdict(self)


def item_scores(L: Dict[str, np.ndarray], t, cfg: EvidenceConfig, cs: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-claim evidence score for task ``t`` (length = number of claims existing at the task)."""
    n = len(t.F.sup)
    cs = t.cs if cs is None else cs
    out = np.zeros(n, dtype=np.float64)
    end = int(L["ptr"][n])
    if end == 0 or cs is None or len(cs) == 0:
        return out
    com = L["commit"][:end]
    valid = com < t.nc
    c = np.minimum(com, len(cs) - 1)
    w = np.ones(end, dtype=np.float64)
    if cfg.strength_exp:
        w *= np.power(L["c_strength"][c].astype(np.float64), cfg.strength_exp)
    if cfg.dilution_exp:
        w *= np.power(np.maximum(L["c_nfiles"][c], 1).astype(np.float64), -cfg.dilution_exp)
    if cfg.test_bonus:
        w *= 1.0 + cfg.test_bonus * L["c_test"][c]
    if cfg.half_life_e:
        age = np.maximum(t.clock - L["c_clock"][c], 0.0) / 86400.0
        w *= np.power(2.0, -age / cfg.half_life_e)
    if cfg.revert_cancel:
        w = np.where(t.ning >= L["c_cancel"][c], 0.0, w)
    x = np.where(valid, w * cs[c], 0.0)
    lens = np.diff(L["ptr"][: n + 1])
    has = lens > 0
    if not has.any():
        return out
    starts = L["ptr"][:n][has]
    if cfg.agg == "max":
        out[has] = np.maximum.reduceat(x, starts)
    elif cfg.agg == "sum":
        out[has] = np.add.reduceat(x, starts)
    elif cfg.agg == "noisyor":
        out[has] = 1.0 - np.exp(np.add.reduceat(np.log1p(-np.clip(x, 0.0, 0.999)), starts))
    else:
        raise ValueError(cfg.agg)
    return out


def claim_scores(L: Dict[str, np.ndarray], t, cfg: EvidenceConfig, sim_key: str = "tfidf", cs: Optional[np.ndarray] = None) -> np.ndarray:
    """Full EmemoVeri-2 score per claim (before masking to live local claims)."""
    s = cfg.w_sim * t.sims[sim_key].astype(np.float64)
    if cfg.w_ev:
        s = s + cfg.w_ev * item_scores(L, t, cfg, cs)
    if cfg.l_pop:
        s = s + cfg.l_pop * (t.n_touch / (t.n_touch + 5.0))
    if cfg.l_rec:
        s = s + cfg.l_rec * np.power(2.0, -t.F.age_days / cfg.H)
    return s


def rank(L, t, cfg: EvidenceConfig, sim_key: str = "tfidf", cs: Optional[np.ndarray] = None, k: int = 20) -> np.ndarray:
    """Indices of the top-k live, same-repository claims (ties broken by recency, as everywhere else)."""
    s = claim_scores(L, t, cfg, sim_key, cs)
    idx = np.flatnonzero(t.local & ~t.F.sup)
    return idx[np.lexsort((-t.F.updated[idx], -s[idx]))[:k]]
