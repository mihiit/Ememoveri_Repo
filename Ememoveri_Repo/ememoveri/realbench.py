"""
Real-repository benchmark: chronological replay of git history.

Task (objective, no LLM, no heuristic labels)
---------------------------------------------
"Memory-guided change localisation." At time t a developer/agent receives a
natural-language task description (the cleaned subject line of a *future*
commit C). Memory has been built **only** from commits that precede C. The
memory mechanism must return a ranked list of source files. Ground truth is
the set of source files that C actually modified -- read from git, not from
any classifier.

Why this tests EmemoVeri's claims on real data
----------------------------------------------
* Memory claims are file-level engineering facts whose state evolves from
  structural git events (add / modify / delete / rename / revert), so the
  state machine, supersession, and contradiction paths are exercised by real
  history rather than synthetic scenarios.
* Dead paths (deleted / renamed files) are objectively identifiable, so
  stale-knowledge errors are measurable without manual labels.
* Several repositories are replayed on one global clock, so cross-repository
  contamination is measurable: other repositories' files are never correct.

Leakage discipline
------------------
Memory is updated only *after* a task commit has been scored. The text index
(IDF statistics included) is computed from pre-cutoff claims only. A "leak
canary" scores the same task *after* ingesting C; it must inflate accuracy,
which demonstrates the harness would detect a leaking pipeline. A "shuffled
query" control must collapse accuracy towards chance.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

from .mining import Commit, DEFAULT_REPOS, RepoConfig, clean_subject, is_noise
from .retrieval import RetrievalWeights
from .schema import Evidence, EvidenceRelation as R, EvidenceType as T, MemoryClaim, MemoryState
from .state_engine import StateEvolutionEngine
from .trust import TrustModel

N_FEATURES = 2 ** 18
_TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b"


def _prep(s: str) -> str:
    return re.sub(r"[_./\\\-]+", " ", s).lower()


_HV = HashingVectorizer(n_features=N_FEATURES, alternate_sign=False, norm=None, ngram_range=(1, 2),
                        stop_words="english", token_pattern=_TOKEN, preprocessor=_prep)


# ---------------------------------------------------------------------------
# Task selection (structure only; pass 1)
# ---------------------------------------------------------------------------

def eligible_task_indices(commits: Sequence[Commit], cfg: RepoConfig, warmup_frac: float = 0.35,
                          min_words: int = 4, max_files: int = 4) -> Tuple[List[int], dict]:
    """Indices of commits usable as tasks. Mirrors `Memory` path bookkeeping."""
    live: set = set()
    touching = [i for i, c in enumerate(commits) if any(cfg.is_source(fc.path) or
                (fc.old_path and cfg.is_source(fc.old_path)) for fc in c.changes)]
    warm_cut = touching[int(len(touching) * warmup_frac)] if touching else 0
    out: List[int] = []
    stats = {"touching": len(touching), "cold_file_excluded": 0, "too_many_files": 0,
             "structural_excluded": 0, "noise_or_short": 0}
    for i, c in enumerate(commits):
        src = [fc for fc in c.changes if cfg.is_source(fc.path) or (fc.old_path and cfg.is_source(fc.old_path))]
        if i >= warm_cut and src:
            words = len(clean_subject(c.subject).split())
            if is_noise(c) or c.reverts or words < min_words:
                stats["noise_or_short"] += 1
            elif any(fc.status != "M" for fc in src):
                stats["structural_excluded"] += 1
            elif len(src) > max_files:
                stats["too_many_files"] += 1
            elif not all(fc.path in live for fc in src):
                stats["cold_file_excluded"] += 1
            else:
                out.append(i)
        for fc in src:
            if fc.status == "A":
                live.add(fc.path)
            elif fc.status == "M":
                live.add(fc.path)
            elif fc.status == "D":
                live.discard(fc.path)
            elif fc.status == "R":
                if fc.old_path and cfg.is_source(fc.old_path):
                    live.discard(fc.old_path)
                if cfg.is_source(fc.path):
                    live.add(fc.path)
    return out, stats


# ---------------------------------------------------------------------------
# Memory built from structural events
# ---------------------------------------------------------------------------

class Memory:
    """File-level claims evolved by the real EmemoVeri state/trust engines."""

    def __init__(self, repo_cfgs: Dict[str, RepoConfig], recent_k: int = 5, bulk_files: int = 3,
                 bulk_strength: float = 0.3):
        self.repo_cfgs = repo_cfgs
        self.repo_ids = {r: i for i, r in enumerate(repo_cfgs)}
        self.engine = StateEvolutionEngine()
        self.tm = TrustModel()
        self.recent_k, self.bulk_files, self.bulk_strength = recent_k, bulk_files, bulk_strength
        self.claims: List[MemoryClaim] = []
        self.repo_of: List[int] = []
        self.path_of: List[str] = []
        self.live: Dict[Tuple[int, str], int] = {}
        self.sum_signed: List[float] = []
        self.ev_total: List[float] = []
        self.trust_seq: List[float] = []
        self.updated: List[float] = []
        self.superseded: List[bool] = []
        self.subjects: List[deque] = []
        self.rows_idx: List[Optional[np.ndarray]] = []
        self.rows_dat: List[Optional[np.ndarray]] = []
        self.dirty: set = set()
        self.df = np.zeros(N_FEATURES, dtype=np.int32)
        self.max_evidence_clock = 0.0

    # -- claim lifecycle ---------------------------------------------------
    def _new_claim(self, repo: str, path: str, commit: Commit, inherit: Optional[int] = None) -> int:
        i = len(self.claims)
        c = MemoryClaim(statement=path, scope=repo)
        c.created_at = c.updated_at = commit.dt
        self.claims.append(c)
        self.repo_of.append(self.repo_ids[repo])
        self.path_of.append(path)
        self.sum_signed.append(0.0)
        self.ev_total.append(0.0)
        self.trust_seq.append(c.trust)
        self.updated.append(commit.clock)
        self.superseded.append(False)
        self.subjects.append(deque(maxlen=self.recent_k))
        self.rows_idx.append(None)
        self.rows_dat.append(None)
        self.live[(self.repo_ids[repo], path)] = i
        if inherit is not None:  # rename: history follows the file
            old = self.claims[inherit]
            c.evidence = list(old.evidence)
            self.subjects[i].extend(self.subjects[inherit])
            c.trust = self.tm.sequential_update_all(c.evidence)
            c.state = self.engine._resolve_state(c)
            self.sum_signed[i] = sum(e.signed_weight for e in c.evidence if e.relation != R.SUPERSEDES)
            self.ev_total[i] = sum(abs(e.signed_weight) for e in c.evidence)
            self.trust_seq[i] = c.trust
        self.dirty.add(i)
        return i

    def _apply(self, i: int, ev: Evidence) -> None:
        c = self.claims[i]
        self.engine.apply_evidence(c, ev)
        c.trust = self.tm.sequential_update(c.trust, ev)
        if ev.relation != R.SUPERSEDES:
            self.sum_signed[i] += ev.signed_weight
        self.ev_total[i] += abs(ev.signed_weight)
        self.trust_seq[i] = c.trust
        self.updated[i] = max(self.updated[i], ev.timestamp.timestamp())
        self.superseded[i] = c.state == MemoryState.SUPERSEDED
        self.max_evidence_clock = max(self.max_evidence_clock, ev.timestamp.timestamp())

    # -- commit ingestion --------------------------------------------------
    def ingest(self, commit: Commit) -> None:
        cfg = self.repo_cfgs[commit.repo]
        rid = self.repo_ids[commit.repo]
        src = [fc for fc in commit.changes if cfg.is_source(fc.path) or (fc.old_path and cfg.is_source(fc.old_path))]
        if not src:
            return
        n_src = sum(1 for fc in src if cfg.is_source(fc.path))
        strength = 1.0 if n_src <= self.bulk_files else self.bulk_strength
        etype = T.TEST if any(cfg.is_test(fc.path) for fc in commit.changes) else T.COMMIT
        dt = commit.dt
        cleaned = clean_subject(commit.subject) if not is_noise(commit) else ""
        is_revert = commit.reverts is not None

        def ev(rel, meta=None):
            return Evidence(etype, rel, "", dt, source_id=commit.sha, strength=strength, metadata=meta or {})

        def note(i):
            if cleaned:
                self.subjects[i].append(cleaned)
            self.dirty.add(i)

        for fc in src:
            new_src, old_src = cfg.is_source(fc.path), bool(fc.old_path and cfg.is_source(fc.old_path))
            if fc.status in ("A", "M") and new_src:
                i = self.live.get((rid, fc.path))
                if i is None:
                    i = self._new_claim(commit.repo, fc.path, commit)
                    self._apply(i, ev(R.CORROBORATES))
                elif is_revert:
                    self._apply(i, ev(R.CONTRADICTS))
                else:
                    self._apply(i, ev(R.REFINES if fc.status == "M" else R.CORROBORATES))
                note(i)
            elif fc.status == "D" and new_src:
                i = self.live.pop((rid, fc.path), None)
                if i is not None:
                    self._apply(i, ev(R.SUPERSEDES))
            elif fc.status == "R":
                old_i = self.live.get((rid, fc.old_path)) if old_src else None
                clash = self.live.get((rid, fc.path)) if new_src else None
                if clash is not None:
                    self.live.pop((rid, fc.path))
                    self._apply(clash, ev(R.SUPERSEDES))
                if old_i is not None:
                    self.live.pop((rid, fc.old_path))
                if new_src:
                    new_i = self._new_claim(commit.repo, fc.path, commit, inherit=old_i)
                    if old_i is not None:
                        self._apply(old_i, ev(R.SUPERSEDES, {"superseding_claim_id": self.claims[new_i].claim_id}))
                    self._apply(new_i, ev(R.CORROBORATES))
                    note(new_i)
                elif old_i is not None:
                    self._apply(old_i, ev(R.SUPERSEDES))

    # -- text index ----------------------------------------------------------
    def _refresh(self) -> None:
        if not self.dirty:
            return
        idxs = sorted(self.dirty)
        texts = [os.path.splitext(self.path_of[i])[0] + " " + " ".join(self.subjects[i]) for i in idxs]
        M = _HV.transform(texts).tocsr()
        M.data = 1.0 + np.log(M.data)
        for j, i in enumerate(idxs):
            s, e = M.indptr[j], M.indptr[j + 1]
            new_idx, new_dat = M.indices[s:e].copy(), M.data[s:e].copy()
            if self.rows_idx[i] is not None:
                self.df[self.rows_idx[i]] -= 1
            self.df[new_idx] += 1
            self.rows_idx[i], self.rows_dat[i] = new_idx, new_dat
        self.dirty.clear()

    def similarities(self, query: str) -> np.ndarray:
        """Cosine similarity of the query to every claim; IDF from pre-cutoff claims only."""
        self._refresh()
        n = len(self.claims)
        lens = np.fromiter((len(r) for r in self.rows_idx), dtype=np.int64, count=n)
        indptr = np.concatenate([[0], np.cumsum(lens)])
        indices = np.concatenate(self.rows_idx)
        data = np.concatenate(self.rows_dat)
        idf = np.log((n + 1.0) / (self.df + 1.0)) + 1.0
        w = data * idf[indices]
        sq = np.add.reduceat(w * w, indptr[:-1].clip(max=len(w) - 1)) if len(w) else np.zeros(n)
        norms = np.sqrt(np.where(lens > 0, sq, 1.0))
        q = _HV.transform([query]).tocsr()
        if q.nnz == 0:
            return np.zeros(n)
        qd = np.zeros(N_FEATURES)
        qd[q.indices] = (1.0 + np.log(q.data)) * idf[q.indices]
        qn = np.linalg.norm(qd)
        if qn == 0:
            return np.zeros(n)
        qd /= qn
        X = sp.csr_matrix((w, indices, indptr), shape=(n, N_FEATURES))
        sims = (X @ qd) / np.maximum(norms, 1e-12)
        return np.clip(sims, 0.0, 1.0)

    def features(self, now_clock: float) -> "Features":
        return Features(
            trust_seq=np.array(self.trust_seq), sum_signed=np.array(self.sum_signed),
            ev_total=np.array(self.ev_total), age_days=np.maximum(0.0, now_clock - np.array(self.updated)) / 86400.0,
            sup=np.array(self.superseded, dtype=bool), repo=np.array(self.repo_of), now=now_clock,
            updated=np.array(self.updated))


# ---------------------------------------------------------------------------
# Vectorised scoring (verified equal to RetrievalScorer in tests)
# ---------------------------------------------------------------------------

@dataclass
class Features:
    trust_seq: np.ndarray
    sum_signed: np.ndarray
    ev_total: np.ndarray
    age_days: np.ndarray
    sup: np.ndarray
    repo: np.ndarray
    now: float
    updated: np.ndarray

    def trust(self, mode: str, lr: float = 0.35, prior: float = 0.5) -> np.ndarray:
        if mode == "seq":
            return self.trust_seq
        if mode == "agg":
            return np.clip(prior + lr * self.sum_signed, 0.0, 1.0)
        if mode == "logit":
            z = math.log(prior / (1 - prior)) + lr * self.sum_signed
            return 1.0 / (1.0 + np.exp(-z))
        raise ValueError(mode)


@dataclass(frozen=True)
class Cond:
    """One retrieval condition = a weight vector + policy + flags."""

    name: str
    w_sim: float = 0.45
    w_trust: float = 0.25
    w_ev: float = 0.20
    w_scope: float = 0.10
    w_stale: float = 0.10
    half_life: float = 90.0
    trust_mode: str = "seq"
    policy: str = "local_first"      # local_first | local_only | open | similarity_gate
    state_aware: bool = True
    sat: float = 5.0
    gate: float = 0.80
    margin: float = 0.15
    kind: str = "score"              # score | recency | popularity | random

    def weights(self) -> RetrievalWeights:
        return RetrievalWeights(self.w_sim, self.w_trust, self.w_ev, self.w_scope, self.w_stale)


def score_vec(c: Cond, sim: np.ndarray, F: Features, local: np.ndarray) -> np.ndarray:
    ev = F.ev_total / (F.ev_total + c.sat)
    stale = 1.0 - np.power(2.0, -F.age_days / c.half_life)
    return (c.w_sim * sim + c.w_trust * F.trust(c.trust_mode) + c.w_ev * ev
            + c.w_scope * local.astype(float) - c.w_stale * stale)


TOPN = 20


def rank_claims(c: Cond, sim: np.ndarray, F: Features, local: np.ndarray, rng: Optional[random.Random] = None):
    """Return (indices, scores) of the top-N eligible claims, best first."""
    mask = ~F.sup if c.state_aware else np.ones(len(sim), bool)
    if c.kind == "recency":
        mask = mask & local
        score = F.updated.copy()
    elif c.kind == "popularity":
        mask = mask & local
        score = F.ev_total.copy()
    elif c.kind == "random":
        mask = mask & local
        score = np.array([rng.random() for _ in range(len(sim))])
    else:
        if c.policy == "local_only":
            mask = mask & local
        elif c.policy == "similarity_gate":
            mask = mask & (local | (sim >= c.gate))
        score = score_vec(c, sim, F, local)
        if c.policy == "local_first":
            lm = mask & local
            best = score[lm].max() if lm.any() else -np.inf
            mask = mask & (local | (score >= best + c.margin))
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return idx, np.array([])
    sc = score[idx]
    # deterministic tie-break: higher score, then more recently updated
    order = np.lexsort((-F.updated[idx], -sc))[:TOPN]
    return idx[order], sc[order]


# ---------------------------------------------------------------------------
# Per-task metrics
# ---------------------------------------------------------------------------

def task_metrics(top_idx: np.ndarray, top_score: np.ndarray, mem: Memory, F: Features, gold: set,
                 q_repo: int) -> dict:
    seen, keys, cl = set(), [], []
    for i in top_idx:
        k = (int(mem.repo_of[i]), mem.path_of[i])
        if k in seen:
            continue
        seen.add(k)
        keys.append(k)
        cl.append(int(i))
    hit = [k in gold for k in keys]
    top5 = keys[:5]
    out = {
        "hit1": float(any(hit[:1])), "hit3": float(any(hit[:3])), "hit5": float(any(hit[:5])),
        "rr": next((1.0 / (r + 1) for r, h in enumerate(hit) if h), 0.0),
        "rec5": len([k for k in top5 if k in gold]) / len(gold),
        "stale5": (sum(bool(F.sup[i]) for i in cl[:5]) / len(top5)) if top5 else 0.0,
        "contam5": (sum(k[0] != q_repo for k in top5) / len(top5)) if top5 else 0.0,
    }
    if len(top_score) >= 2:
        out["gap"] = float(top_score[0] - top_score[1])
    else:
        out["gap"] = float("inf") if len(top_score) == 1 else 0.0
    out["top_score"] = float(top_score[0]) if len(top_score) else 0.0
    out["top1_correct"] = float(bool(hit and hit[0]))
    out["top1_trust"] = float(F.trust_seq[cl[0]]) if cl else 0.0
    return out


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def _renorm(**kw) -> dict:
    pos = {k: v for k, v in kw.items() if k != "w_stale"}
    tot = sum(pos.values())
    return {**{k: v / tot for k, v in pos.items()}, "w_stale": kw["w_stale"]}


def named_conditions() -> List[Cond]:
    d = Cond("EMEMOVERI")
    sim_only = dict(w_sim=1.0, w_trust=0.0, w_ev=0.0, w_scope=0.0, w_stale=0.0)
    conds = [
        Cond("RANDOM_LOCAL", kind="random", state_aware=True),
        Cond("RECENCY", kind="recency"),
        Cond("POPULARITY", kind="popularity"),
        Cond("PLAIN_RAG", policy="open", state_aware=False, **sim_only),
        Cond("NAIVE_MEMORY", policy="local_only", state_aware=False, **sim_only),
        Cond("SNAPSHOT_RAG", policy="local_only", state_aware=True, **sim_only),
        d,
        # ablations of the default system
        Cond("ABL_no_trust", **_renorm(w_sim=0.45, w_trust=0.0, w_ev=0.20, w_scope=0.10, w_stale=0.10)),
        Cond("ABL_no_evidence", **_renorm(w_sim=0.45, w_trust=0.25, w_ev=0.0, w_scope=0.10, w_stale=0.10)),
        Cond("ABL_no_staleness", w_stale=0.0),
        Cond("ABL_state_blind", state_aware=False),
        Cond("ABL_policy_open", policy="open"),
        Cond("ABL_policy_local_only", policy="local_only"),
        Cond("ABL_policy_sim_gate", policy="similarity_gate"),
        Cond("ABL_trust_aggclamp", trust_mode="agg"),
        Cond("ABL_trust_logit", trust_mode="logit"),
        Cond("ABL_halflife_30d", half_life=30.0),
        Cond("ABL_halflife_365d", half_life=365.0),
    ]
    return conds


def tuning_grid() -> List[Cond]:
    grid = []
    for w_sim in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        for w_trust in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            w_ev = round(0.9 - w_sim - w_trust, 10)
            if w_ev < -1e-9:
                continue
            for w_stale in (0.0, 0.05, 0.1, 0.2):
                for hl in (30.0, 90.0, 365.0):
                    if w_stale == 0.0 and hl != 90.0:
                        continue
                    for tm in ("seq", "logit"):
                        grid.append(Cond(f"grid", w_sim=w_sim, w_trust=w_trust, w_ev=max(0.0, w_ev), w_scope=0.1,
                                         w_stale=w_stale, half_life=hl, trust_mode=tm))
    return grid


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    repos: Tuple[str, ...] = ("fastapi", "yt-dlp", "browser-use")
    warmup_frac: float = 0.35
    max_tasks_per_repo: int = 400
    dev_frac: float = 0.30
    seed: int = 7
    leak_canary: bool = True
    shuffle_control: bool = True


def _sample_evenly(xs: List[int], k: int) -> List[int]:
    if len(xs) <= k:
        return xs
    step = len(xs) / k
    return [xs[int(i * step)] for i in range(k)]


def run_benchmark(commits_by_repo: Dict[str, List[Commit]], cfg: BenchConfig,
                  log: Callable[[str], None] = print,
                  repo_configs: Optional[Dict[str, RepoConfig]] = None) -> dict:
    t_start = time.time()
    repo_cfgs = {r: (repo_configs or DEFAULT_REPOS)[r] for r in cfg.repos}
    # ---- pass 1: select tasks ------------------------------------------------
    selection: Dict[Tuple[str, int], dict] = {}
    selection_stats = {}
    for r in cfg.repos:
        cs = commits_by_repo[r]
        elig, st = eligible_task_indices(cs, repo_cfgs[r], cfg.warmup_frac)
        chosen = _sample_evenly(elig, cfg.max_tasks_per_repo)
        st.update({"eligible": len(elig), "selected": len(chosen)})
        selection_stats[r] = st
        for i in chosen:
            selection[(r, i)] = {}
        log(f"[select] {r}: {st}")
    task_clocks = sorted(commits_by_repo[r][i].clock for (r, i) in selection)
    split_clock = task_clocks[int(len(task_clocks) * cfg.dev_frac)]
    # ---- global chronological stream ------------------------------------------
    stream = []
    for ri, r in enumerate(cfg.repos):
        for i, c in enumerate(commits_by_repo[r]):
            stream.append((c.clock, ri, i, c))
    stream.sort(key=lambda t: (t[0], t[1], t[2]))
    task_list = [(r, i) for (r, i) in selection]
    rng_ctrl = random.Random(cfg.seed)
    perm = task_list[:]
    rng_ctrl.shuffle(perm)
    shuffled_query = {task_list[k]: commits_by_repo[perm[k][0]][perm[k][1]].subject for k in range(len(task_list))}

    mem = Memory(repo_cfgs)
    conds = named_conditions()
    grid = tuning_grid()
    grid_dev = np.zeros((len(grid), 2))   # sum MRR, sum hit5 on dev
    n_dev = 0
    tuned: Optional[Cond] = None
    tuned_info: dict = {}
    rng = random.Random(cfg.seed)
    rows: List[dict] = []
    repo_ids = mem.repo_ids
    approx_leak_checks = 0

    def eval_cond(c: Cond, sim, F, local, gold, q_repo):
        idx, sc = rank_claims(c, sim, F, local, rng)
        return task_metrics(idx, sc, mem, F, gold, q_repo)

    for n_done, (clock, ri, i, commit) in enumerate(stream):
        key = (commit.repo, i)
        if key in selection:
            q_repo = repo_ids[commit.repo]
            now = commit.clock
            # --- leakage assertion: memory holds nothing from now or later ---
            assert mem.max_evidence_clock <= now, "memory contains evidence from the future"
            mcfg = repo_cfgs[commit.repo]
            gold = {(q_repo, fc.path) for fc in commit.changes if mcfg.is_source(fc.path)}
            for k in gold:
                assert k in mem.live, f"gold file without a live claim: {k}"
            query = clean_subject(commit.subject)
            sim = mem.similarities(query)
            F = mem.features(now)
            local = F.repo == q_repo
            split = "dev" if commit.clock < split_clock else "test"
            row = {"repo": commit.repo, "sha": commit.sha, "split": split, "clock": now, "n_gold": len(gold),
                   "n_claims": len(mem.claims), "n_dead": int(F.sup.sum()), "m": {}}
            # tuning on the dev period only
            if split == "dev":
                for gi, g in enumerate(grid):
                    idx, sc = rank_claims(g, sim, F, local)
                    m = task_metrics(idx, sc, mem, F, gold, q_repo)
                    grid_dev[gi, 0] += m["rr"]
                    grid_dev[gi, 1] += m["hit5"]
                n_dev += 1
            elif tuned is None:
                best = max(range(len(grid)), key=lambda gi: (round(grid_dev[gi, 0], 9), grid_dev[gi, 1]))
                g = grid[best]
                tuned = Cond("EMEMOVERI_TUNED", g.w_sim, g.w_trust, g.w_ev, g.w_scope, g.w_stale, g.half_life,
                             g.trust_mode)
                tuned_info = {"w_sim": g.w_sim, "w_trust": g.w_trust, "w_ev": g.w_ev, "w_scope": g.w_scope,
                              "w_stale": g.w_stale, "half_life": g.half_life, "trust_mode": g.trust_mode,
                              "dev_mrr": grid_dev[best, 0] / max(n_dev, 1),
                              "dev_hit5": grid_dev[best, 1] / max(n_dev, 1), "n_dev_tasks": n_dev,
                              "grid_size": len(grid)}
                log(f"[tune] selected on {n_dev} dev tasks: {tuned_info}")
            active = conds + ([tuned] if tuned is not None else [])
            for c in active:
                row["m"][c.name] = eval_cond(c, sim, F, local, gold, q_repo)
            if cfg.shuffle_control:
                sq = clean_subject(shuffled_query[key])
                ssim = mem.similarities(sq)
                for c in (next(x for x in conds if x.name == "SNAPSHOT_RAG"), next(x for x in conds if x.name == "EMEMOVERI")):
                    row["m"][f"SHUFFLED::{c.name}"] = eval_cond(c, ssim, F, local, gold, q_repo)
            rows.append(row)
            # --- ingest the task commit, then the leak canary ---------------
            mem.ingest(commit)
            if cfg.leak_canary:
                sim2 = mem.similarities(query)
                F2 = mem.features(now)
                for c in (next(x for x in conds if x.name == "SNAPSHOT_RAG"), next(x for x in conds if x.name == "EMEMOVERI")):
                    row["m"][f"LEAKED::{c.name}"] = eval_cond(c, sim2, F2, local, gold, q_repo)
        else:
            mem.ingest(commit)
        if n_done % 10000 == 0:
            log(f"[replay] {n_done}/{len(stream)} events, {len(rows)} tasks scored, {len(mem.claims)} claims, "
                f"{time.time() - t_start:.0f}s")

    return {"config": cfg.__dict__, "selection_stats": selection_stats, "split_clock": split_clock,
            "tuned": tuned_info, "rows": rows, "n_claims_final": len(mem.claims),
            "runtime_s": time.time() - t_start}
