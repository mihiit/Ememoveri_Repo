"""
Extensions to the real-repository replay (v0.3).

One chronological replay collects, for every benchmark task, everything needed to score *any* ranking
condition offline: the claim features at the task time, the query-claim similarity under three text
backends (hashed TF-IDF, BM25, static word-vector average), similarities for degraded queries, and the
gold set.  It also records lifecycle events (deletions, renames, re-creations, reverts) and evaluates them
before and after the event is ingested.

Nothing here changes the published memory mechanism: `ExtMemory` only adds bookkeeping (raw change
counts, optional rename carry-over switch, resurrection counter) and alternative similarity functions.
"""
from __future__ import annotations

import math
import os
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .mining import Commit, DEFAULT_REPOS, RepoConfig, clean_subject, is_noise
from .realbench import (Cond, Features, Memory, N_FEATURES, _HV, _TOKEN, _prep, _sample_evenly,
                        eligible_task_indices, rank_claims)
from .schema import EvidenceRelation as R, MemoryState

_TOK = re.compile(_TOKEN)


# ---------------------------------------------------------------------------
# Static word vectors (spaCy en_core_web_md, loaded lazily; vectors only)
# ---------------------------------------------------------------------------

class WordVectors:
    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_md", exclude=["parser", "ner", "tagger", "lemmatizer",
                                                          "attribute_ruler", "senter"])
        self.vocab = self.nlp.vocab
        self.dim = self.vocab.vectors.shape[1]
        self._cache = lru_cache(maxsize=None)(self._lookup)

    def _lookup(self, w: str):
        lex = self.vocab[w]
        return np.asarray(lex.vector, dtype=np.float32) if lex.has_vector else None

    def embed(self, text: str) -> np.ndarray:
        toks = [t for t in _TOK.findall(_prep(text)) if t not in ENGLISH_STOP_WORDS]
        vs = [v for v in (self._cache(t) for t in toks) if v is not None]
        if not vs:
            return np.zeros(self.dim, dtype=np.float32)
        m = np.mean(vs, axis=0)
        n = np.linalg.norm(m)
        return (m / n).astype(np.float32) if n > 0 else np.zeros(self.dim, dtype=np.float32)


# ---------------------------------------------------------------------------
# Memory with bookkeeping and alternative similarities
# ---------------------------------------------------------------------------

class ExtMemory(Memory):
    def __init__(self, repo_cfgs, carry: bool = True, wv: Optional[WordVectors] = None, **kw):
        super().__init__(repo_cfgs, **kw)
        self.carry = carry
        self.wv = wv
        self.n_touch: List[int] = []
        self.emb: List[Optional[np.ndarray]] = []
        self.resurrected = 0
        self.ever_dead: set = set()
        self.recreated = 0
        # per-claim commit history (ids of commits with a subject), commit subject vectors, files per commit
        self.hist: List[List[int]] = []
        self.c_idx: List[np.ndarray] = []
        self.c_tf: List[np.ndarray] = []
        self.c_claims: List[List[int]] = []
        self.c_df = np.zeros(N_FEATURES)
        self._cur: Optional[Commit] = None
        self._cur_id: Optional[int] = None
        # evidence-level ledger (v0.6): per hist entry the relation code; per commit clock, strength, TEST flag, cancellation
        self.hist_rel: List[List[int]] = []
        self.c_clock: List[float] = []
        self.c_strength: List[float] = []
        self.c_test: List[int] = []
        self.c_cancel: List[float] = []          # ingest index at which a later revert cancelled this commit's evidence
        self.c_by_sha: Dict[str, int] = {}
        self.c_subject: List[str] = []
        self.n_ing = 0
        self._last_cs = np.zeros(0, dtype=np.float32)

    def _new_claim(self, repo, path, commit, inherit=None):
        if not self.carry:
            inherit = None
        i = super()._new_claim(repo, path, commit, inherit=inherit)
        self.n_touch.append(sum(1 for e in self.claims[i].evidence if e.relation != R.SUPERSEDES) if inherit is not None else 0)
        self.emb.append(None)
        self.hist.append(list(self.hist[inherit]) if inherit is not None else [])
        self.hist_rel.append(list(self.hist_rel[inherit]) if inherit is not None else [])
        return i

    def ingest(self, commit):
        self._cur, self._cur_id = commit, None
        self.n_ing += 1
        super().ingest(commit)
        if commit.reverts:
            ref = commit.reverts.lower()
            for sha, cid in self.c_by_sha.items():
                if sha.startswith(ref) and self.c_cancel[cid] == float("inf"):
                    self.c_cancel[cid] = float(self.n_ing)      # the reverted commit's evidence no longer counts
                    break
        self._cur = None

    def _commit_id(self) -> int:
        """Lazily register the commit being ingested (only commits that touch tracked files and carry a subject)."""
        if self._cur_id is None:
            c = self._cur
            cleaned = clean_subject(c.subject) if (c is not None and not is_noise(c)) else ""
            if not cleaned:
                self._cur_id = -1
            else:
                v = _HV.transform([cleaned]).tocsr()
                self._cur_id = len(self.c_idx)
                self.c_idx.append(v.indices.astype(np.int32))
                self.c_tf.append(v.data.astype(np.float32))
                self.c_claims.append([])
                self.c_df[v.indices] += 1
                self.c_clock.append(float(c.clock)); self.c_strength.append(1.0); self.c_test.append(0)
                self.c_cancel.append(float("inf")); self.c_by_sha[c.sha] = self._cur_id; self.c_subject.append(cleaned)
        return self._cur_id

    def _apply(self, i, ev):
        was = self.superseded[i]
        super()._apply(i, ev)
        if was and not self.superseded[i]:
            self.resurrected += 1
        if ev.relation != R.SUPERSEDES:
            self.n_touch[i] += 1
            cid = self._commit_id()
            if cid >= 0 and (not self.hist[i] or self.hist[i][-1] != cid):
                self.hist[i].append(cid)
                self.hist_rel[i].append({R.CORROBORATES: 0, R.REFINES: 1, R.CONTRADICTS: 2}.get(ev.relation, 0))
                self.c_claims[cid].append(i)
                self.c_strength[cid] = float(ev.strength)
                self.c_test[cid] = int(ev.evidence_type.name == "TEST")

    def _claim_text(self, i: int) -> str:
        return os.path.splitext(self.path_of[i])[0] + " " + " ".join(self.subjects[i])

    def _refresh(self):
        d = set(self.dirty)
        super()._refresh()
        if self.wv is not None:
            for i in d:
                self.emb[i] = self.wv.embed(self._claim_text(i))

    def ledger(self) -> Dict[str, np.ndarray]:
        """Final evidence ledger as flat arrays (claims only grow, so any earlier time is a prefix: entries with
        commit id < nc at that time, claims with index < n_claims at that time)."""
        n = len(self.hist)
        ptr = np.concatenate([[0], np.cumsum([len(h) for h in self.hist])]).astype(np.int64)
        flat = np.concatenate([np.asarray(h, dtype=np.int64) for h in self.hist]) if ptr[-1] else np.zeros(0, np.int64)
        rel = np.concatenate([np.asarray(h, dtype=np.int8) for h in self.hist_rel]) if ptr[-1] else np.zeros(0, np.int8)
        owner = np.repeat(np.arange(n), np.diff(ptr))
        return {"ptr": ptr, "commit": flat, "rel": rel, "owner": owner,
                "c_clock": np.asarray(self.c_clock, dtype=np.float64), "c_strength": np.asarray(self.c_strength, dtype=np.float32),
                "c_test": np.asarray(self.c_test, dtype=np.int8), "c_cancel": np.asarray(self.c_cancel, dtype=np.float64),
                "c_nfiles": np.asarray([len(c) for c in self.c_claims], dtype=np.int32),
                "c_sha": np.asarray(list(self.c_by_sha.keys()), dtype="U40")[np.argsort(list(self.c_by_sha.values()))] if self.c_by_sha else np.zeros(0, "U40"),
                "c_subject": np.asarray(self.c_subject, dtype=object)}

    # -- BM25 over the same hashed uni+bigram terms --------------------------------
    def bm25(self, query: str, k1: float = 1.2, b: float = 0.75) -> np.ndarray:
        self._refresh()
        n = len(self.claims)
        lens = np.fromiter((len(r) for r in self.rows_idx), dtype=np.int64, count=n)
        indptr = np.concatenate([[0], np.cumsum(lens)])
        indices = np.concatenate(self.rows_idx)
        tf = np.rint(np.exp(np.concatenate(self.rows_dat) - 1.0))
        q = _HV.transform([query]).tocsr()
        if q.nnz == 0 or len(tf) == 0:
            return np.zeros(n)
        dl = np.zeros(n)
        nz = lens > 0
        dl[nz] = np.add.reduceat(tf, indptr[:-1][nz])
        avgdl = max(dl.mean(), 1e-9)
        X = sp.csr_matrix((tf, indices, indptr), shape=(n, N_FEATURES)).tocsc()
        qi = q.indices
        df = self.df[qi].astype(float)
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        sub = X[:, qi].toarray()
        denom = sub + k1 * (1.0 - b + b * dl / avgdl)[:, None]
        s = ((sub * (k1 + 1.0)) / np.maximum(denom, 1e-12) * idf[None, :]).sum(1)
        m = s.max()
        return s / m if m > 0 else s

    # -- similar-history and co-change channels (v0.5) ----------------------------------
    def extra(self, query: str, sim: np.ndarray, mask: np.ndarray, bulk: int = 10) -> Dict[str, np.ndarray]:
        """knn_max / knn_sum: cosine between the query and the subjects of past commits touching each claim.
        cochange: share of the top-1 claim's past non-bulk commits in which each claim also changed."""
        n = len(self.claims)
        out = {k: np.zeros(n, dtype=np.float32) for k in ("knn_max", "knn_sum", "cochange")}
        nc = len(self.c_idx)
        self._last_cs = np.zeros(nc, dtype=np.float32)
        q = _HV.transform([query]).tocsr()
        if nc and q.nnz:
            lens = np.fromiter((len(a) for a in self.c_idx), dtype=np.int64, count=nc)
            indptr = np.concatenate([[0], np.cumsum(lens)])
            indices = np.concatenate(self.c_idx)
            data = np.concatenate(self.c_tf)
            idf = np.log((nc + 1.0) / (self.c_df + 1.0)) + 1.0
            w = (1.0 + np.log(np.maximum(data, 1.0))) * idf[indices]
            sq = np.add.reduceat(w * w, indptr[:-1].clip(max=len(w) - 1)) if len(w) else np.zeros(nc)
            norms = np.sqrt(np.where(lens > 0, sq, 1.0))
            qd = np.zeros(N_FEATURES)
            qd[q.indices] = (1.0 + np.log(q.data)) * idf[q.indices]
            qn = np.linalg.norm(qd)
            if qn > 0:
                qd /= qn
                X = sp.csr_matrix((w, indices, indptr), shape=(nc, N_FEATURES))
                cs = np.clip((X @ qd) / np.maximum(norms, 1e-12), 0.0, 1.0)
                hl = np.fromiter((len(h) for h in self.hist), dtype=np.int64, count=n)
                has = hl > 0
                if has.any():
                    flat = np.concatenate([h for h in self.hist if h]).astype(np.int64)
                    starts = np.concatenate([[0], np.cumsum(hl[has])[:-1]])
                    vals = cs[flat]
                    out["knn_max"][has] = np.maximum.reduceat(vals, starts)
                    out["knn_sum"][has] = np.add.reduceat(vals, starts)
                self._last_cs = cs.astype(np.float32)
        cand = np.flatnonzero(mask)
        if cand.size and self.hist[int(cand[np.argmax(sim[cand])])]:
            seed = int(cand[np.argmax(sim[cand])])
            seed_commits = [c for c in self.hist[seed] if len(self.c_claims[c]) <= bulk]
            if seed_commits:
                cnt = np.zeros(n, dtype=np.float32)
                for c in seed_commits:
                    for j in self.c_claims[c]:
                        cnt[j] += 1.0
                cnt[seed] = 0.0
                out["cochange"] = cnt / len(seed_commits)
        return out

    # -- static word-vector cosine, min-max normalised per query --------------------
    def dense(self, query: str) -> np.ndarray:
        self._refresh()
        n = len(self.claims)
        qv = self.wv.embed(query)
        if not qv.any():
            return np.zeros(n)
        M = np.vstack([e if e is not None else np.zeros(self.wv.dim, np.float32) for e in self.emb])
        c = M @ qv
        lo, hi = float(c.min()), float(c.max())
        return (c - lo) / (hi - lo) if hi > lo else np.zeros(n)


# ---------------------------------------------------------------------------
# Query degradation
# ---------------------------------------------------------------------------

_TRAILER = re.compile(r"(?im)^(signed-off-by|co-authored-by|reviewed-by|reported-by|fixes|closes|resolves|see|cc|change-id|this reverts commit)\b.*$")
_URL = re.compile(r"https?://\S+")


def body_text(commit: Commit, limit: int = 600) -> str:
    """Commit message body without trailers and URLs (developer's own explanation), truncated."""
    b = _URL.sub(" ", _TRAILER.sub(" ", commit.body or ""))
    return re.sub(r"\s+", " ", b).strip()[:limit]


def degrade(query: str, p: float, key: str) -> str:
    words = query.split()
    if len(words) <= 1 or p <= 0:
        return query
    rng = random.Random(f"{key}:{p}")
    keep = [w for w in words if rng.random() >= p]
    if not keep:
        keep = [rng.choice(words)]
    return " ".join(keep)


# ---------------------------------------------------------------------------
# Metrics (superset of realbench.task_metrics)
# ---------------------------------------------------------------------------

METRICS = ["hit1", "hit3", "hit5", "rr", "rec5", "prec5", "ndcg5", "dead5", "contam5"]


def metrics_vec(mem: Memory, F: Features, idx: np.ndarray, gold: set, q_repo: int) -> np.ndarray:
    seen, keys, cl = set(), [], []
    for i in idx:
        k = (int(mem.repo_of[i]), mem.path_of[i])
        if k in seen:
            continue
        seen.add(k)
        keys.append(k)
        cl.append(int(i))
    hit = [k in gold for k in keys]
    top5 = keys[:5]
    dcg = sum(1.0 / math.log2(r + 2) for r, h in enumerate(hit[:5]) if h)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(gold), 5)))
    return np.array([
        float(any(hit[:1])), float(any(hit[:3])), float(any(hit[:5])),
        next((1.0 / (r + 1) for r, h in enumerate(hit) if h), 0.0),
        len([k for k in top5 if k in gold]) / len(gold),
        sum(hit[:5]) / 5.0,
        dcg / idcg if idcg > 0 else 0.0,
        (sum(bool(F.sup[i]) for i in cl[:5]) / len(top5)) if top5 else 0.0,
        (sum(k[0] != q_repo for k in top5) / len(top5)) if top5 else 0.0,
    ])


def rank_generic(score: np.ndarray, mask: np.ndarray, updated: np.ndarray, topn: int = 20) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return idx
    order = np.lexsort((-updated[idx], -score[idx]))[:topn]
    return idx[order]


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

@dataclass
class TaskRec:
    repo: str
    sha: str
    split: str
    clock: float
    month: str
    gold: set
    q_repo: int
    F: Features
    n_touch: np.ndarray
    local: np.ndarray
    sims: Dict[str, np.ndarray]
    cs: Optional[np.ndarray] = None          # cosine between the query and every registered commit subject (len = nc)
    nc: int = 0                              # commits registered when the task was scored
    ning: int = 0                            # ingest counter when the task was scored (revert cancellation)
    cs_b: Optional[np.ndarray] = None        # same for the subject+body query
    has_body: bool = False
    cs_deg: Optional[Dict[float, np.ndarray]] = None   # same for the word-dropout queries (non-slim collections)


DEGRADE_LEVELS = (0.3, 0.5, 0.7)
TUNED_FIXED = Cond("EMEMOVERI_TUNED", 0.8, 0.0, 0.1, 0.1, 0.05, 365.0, "seq")


def collect(commits_by_repo: Dict[str, List[Commit]], repos: Sequence[str], wv: Optional[WordVectors],
            warmup_frac: float = 0.35, max_tasks_per_repo: int = 400, dev_frac: float = 0.30,
            log=print, max_events_per_type: int = 400, slim: bool = False, track_events: bool = True, carry: bool = True) -> dict:
    """slim: keep only what the repository-level analyses use (TF-IDF, body-query, similar-history and co-change channels as
    float32; no BM25 / word-vector / degraded-query channels) -- the stored values are identical to what ``--slim`` produced
    after the fact, but peak memory is much lower.  track_events=False skips the lifecycle-event machinery."""
    import datetime as _dt
    repo_cfgs = {r: DEFAULT_REPOS[r] for r in repos}
    selection = set()
    warm_idx = {}
    for r in repos:
        cs = commits_by_repo[r]
        elig, _ = eligible_task_indices(cs, repo_cfgs[r], warmup_frac)
        for i in _sample_evenly(elig, max_tasks_per_repo):
            selection.add((r, i))
        touching = [i for i, c in enumerate(cs) if any(repo_cfgs[r].is_source(fc.path) or
                    (fc.old_path and repo_cfgs[r].is_source(fc.old_path)) for fc in c.changes)]
        warm_idx[r] = touching[int(len(touching) * warmup_frac)]
    clocks = sorted(commits_by_repo[r][i].clock for (r, i) in selection)
    split_clock = clocks[int(len(clocks) * dev_frac)]
    stream = sorted(((c.clock, ri, i, c) for ri, r in enumerate(repos) for i, c in enumerate(commits_by_repo[r])),
                    key=lambda t: (t[0], t[1], t[2]))

    mem = ExtMemory(repo_cfgs, carry=carry, wv=wv)
    mem_nc = ExtMemory(repo_cfgs, carry=False, wv=None) if track_events else None
    snap = Cond("SNAP", 1.0, 0, 0, 0, 0, policy="local_only", state_aware=True)
    naive = Cond("NAIVE", 1.0, 0, 0, 0, 0, policy="local_only", state_aware=False)
    tasks: List[TaskRec] = []
    events: List[dict] = []
    counts = {"delete": 0, "rename": 0, "recreate": 0, "revert_source_commits": 0, "revert_claims_touched": 0}
    per_type = {"delete": 0, "rename": 0}

    for n_done, (clock, ri, i, commit) in enumerate(stream):
        r = commit.repo
        cfg = repo_cfgs[r]
        rid = mem.repo_ids[r]
        key = (r, i)
        if key in selection:
            assert mem.max_evidence_clock <= clock, "memory contains evidence from the future"
            gold = {(rid, fc.path) for fc in commit.changes if cfg.is_source(fc.path)}
            q = clean_subject(commit.subject)
            F = mem.features(clock)
            sims = {"tfidf": mem.similarities(q)}
            sims.update(mem.extra(q, sims["tfidf"], (F.repo == rid) & ~F.sup))
            cs_main = mem._last_cs.copy()
            cs_deg = None
            body = body_text(commit)
            has_body = len(body) >= 40
            cs_body = None
            if has_body:
                qb = q + " " + body
                sims["tfidf_b"] = mem.similarities(qb)
                mem.extra(qb, sims["tfidf_b"], (F.repo == rid) & ~F.sup)
                cs_body = mem._last_cs.copy()
            if not slim:
                sims["bm25"] = mem.bm25(q)
                if wv is not None:
                    sims["dense"] = mem.dense(q)
                cs_deg = {}
                for p in DEGRADE_LEVELS:
                    dq = degrade(q, p, commit.sha)
                    sims[f"tfidf@{p}"] = mem.similarities(dq)
                    mem.extra(dq, sims[f"tfidf@{p}"], (F.repo == rid) & ~F.sup)
                    cs_deg[p] = mem._last_cs.copy()
            sims = {k: v.astype(np.float64 if (k == "tfidf" and not slim) else np.float32) for k, v in sims.items()}
            month = _dt.datetime.fromtimestamp(clock, tz=_dt.timezone.utc).strftime("%Y-%m")
            tasks.append(TaskRec(r, commit.sha, "dev" if clock < split_clock else "test", clock, f"{r}:{month}",
                                 gold, rid, F, np.array(mem.n_touch, dtype=np.float64), F.repo == rid, sims,
                                 cs=cs_main, nc=len(mem.c_idx), ning=mem.n_ing, cs_b=cs_body, has_body=has_body, cs_deg=cs_deg))
        # ---- lifecycle events (post warm-up) ------------------------------------------
        src = [fc for fc in commit.changes if cfg.is_source(fc.path) or (fc.old_path and cfg.is_source(fc.old_path))]
        if track_events and i >= warm_idx[r] and src:
            if commit.reverts is not None:
                counts["revert_source_commits"] += 1
                counts["revert_claims_touched"] += sum(1 for fc in src if mem.live.get((rid, fc.path)) is not None)
            for fc in src:
                ev = None
                if fc.status == "D" and cfg.is_source(fc.path):
                    old = fc.path
                    ev = "delete"
                elif fc.status == "R" and fc.old_path and cfg.is_source(fc.old_path):
                    old = fc.old_path
                    ev = "rename"
                elif fc.status == "A" and cfg.is_source(fc.path) and (rid, fc.path) not in mem.live \
                        and (rid, fc.path) in mem.ever_dead:
                    counts["recreate"] += 1
                if ev is None:
                    continue
                ci = mem.live.get((rid, old))
                if ci is None or not mem.subjects[ci] or per_type[ev] >= max_events_per_type:
                    continue
                counts[ev] += 1
                per_type[ev] += 1
                q = mem.subjects[ci][-1]
                pre = {}
                Fb = mem.features(clock)
                sb = mem.similarities(q)
                loc = Fb.repo == rid
                for nm, c in (("naive", naive), ("snapshot", snap), ("tuned", TUNED_FIXED)):
                    idx, _ = rank_claims(c, sb, Fb, loc)
                    keys5 = []
                    for j in idx:
                        k = (int(mem.repo_of[j]), mem.path_of[j])
                        if k not in keys5:
                            keys5.append(k)
                    pre[nm] = (rid, old) in keys5[:5]
                events.append({"type": ev, "repo": r, "sha": commit.sha, "old": old, "new": fc.path if ev == "rename" else None,
                               "clock": clock, "pre": pre, "_q": q, "_rid": rid})
                # stash memories' states for post-eval after ingest (below)
        # ---- ingest on both memories -------------------------------------------------------
        pending = [e for e in events if e.get("sha") == commit.sha and "post" not in e]
        for fc in src:
            if fc.status == "D" and cfg.is_source(fc.path):
                mem.ever_dead.add((rid, fc.path))
            elif fc.status == "R" and fc.old_path and cfg.is_source(fc.old_path):
                mem.ever_dead.add((rid, fc.old_path))
        mem.ingest(commit)
        if track_events:
            mem_nc.ingest(commit)
        for e in pending:
            q, rid_e = e.pop("_q"), e.pop("_rid")
            post = {}
            for nm, M in (("carry", mem), ("nocarry", mem_nc)):
                Fa = M.features(clock)
                sa = M.similarities(q)
                loc = Fa.repo == rid_e
                for cn, c in (("naive", naive), ("snapshot", snap), ("tuned", TUNED_FIXED)):
                    if nm == "nocarry" and cn != "snapshot":
                        continue
                    idx, _ = rank_claims(c, sa, Fa, loc)
                    keys5 = []
                    for j in idx:
                        k = (int(M.repo_of[j]), M.path_of[j])
                        if k not in keys5:
                            keys5.append(k)
                    keys5 = keys5[:5]
                    post[f"{nm}:{cn}"] = {"old_in_top5": (rid_e, e["old"]) in keys5,
                                          "new_in_top5": (rid_e, e["new"]) in keys5 if e["new"] else None}
            e["post"] = post
        if n_done % 10000 == 0:
            log(f"[ext-replay] {n_done}/{len(stream)} events, {len(tasks)} tasks, {len(mem.claims)} claims")

    for e in events:
        e.pop("_q", None)
        e.pop("_rid", None)
    return {"tasks": tasks, "ledger": mem.ledger(), "path_of": list(mem.path_of), "repo_of": list(mem.repo_of), "events": events, "counts": counts, "split_clock": split_clock,
            "resurrected": mem.resurrected, "n_superseded": int(sum(mem.superseded))}


# ---------------------------------------------------------------------------
# External tasks (v0.4): queries that are not commit subjects, e.g. real issue text.
# The memory seen by a task contains exactly the commits with clock <= task cutoff (the issue's creation time);
# gold files are those modified by the later fix, restricted to files that already have a live claim.
# ---------------------------------------------------------------------------

def collect_external(commits_by_repo, repos, ext_tasks, wv=None, log=print):
    """ext_tasks: list of dicts {id, repo, cutoff (unix s), queries: {name: text}, gold: [paths], sha}.
    Returns {"tasks": [TaskRec], "path_of", "repo_of", "dropped_gold": int}.  TaskRec.sims has one entry per
    (backend, query name), e.g. 'tfidf:full'; TaskRec.sha holds the task id."""
    import datetime as _dt
    repo_cfgs = {r: DEFAULT_REPOS[r] for r in repos}
    stream = sorted(((c.clock, ri, i, c) for ri, r in enumerate(repos) for i, c in enumerate(commits_by_repo[r])),
                    key=lambda t: (t[0], t[1], t[2]))
    pending = sorted(ext_tasks, key=lambda t: t["cutoff"])
    mem = ExtMemory(repo_cfgs, carry=True, wv=wv)
    out, dropped, p = [], 0, 0

    def evaluate(t):
        nonlocal dropped
        assert mem.max_evidence_clock <= t["cutoff"], "memory contains evidence from after the issue was opened"
        rid = mem.repo_ids[t["repo"]]
        gold = {(rid, g) for g in t["gold"] if mem.live.get((rid, g)) is not None}
        dropped += len(t["gold"]) - len(gold)
        if not gold:
            return
        F = mem.features(t["cutoff"])
        sims = {}
        cs_ext = None
        for qn, q in t["queries"].items():
            sims[f"tfidf:{qn}"] = mem.similarities(q).astype(np.float32)
            for k, v in mem.extra(q, sims[f"tfidf:{qn}"], (F.repo == rid) & ~F.sup).items():
                sims[f"{k}:{qn}"] = v
            if qn == "full":
                cs_ext = mem._last_cs.copy()
            sims[f"bm25:{qn}"] = mem.bm25(q).astype(np.float32)
            if wv is not None:
                sims[f"dense:{qn}"] = mem.dense(q).astype(np.float32)
        month = _dt.datetime.fromtimestamp(t["cutoff"], tz=_dt.timezone.utc).strftime("%Y-%m")
        out.append(TaskRec(t["repo"], t["id"], "test", t["cutoff"], f"{t['repo']}:{month}", gold, rid, F,
                           np.array(mem.n_touch, dtype=np.float64), F.repo == rid, sims,
                           cs=cs_ext, nc=len(mem.c_idx), ning=mem.n_ing))

    for clock, ri, i, commit in stream:
        while p < len(pending) and pending[p]["cutoff"] < clock:
            evaluate(pending[p]); p += 1
        mem.ingest(commit)
    while p < len(pending):
        evaluate(pending[p]); p += 1
    return {"tasks": out, "ledger": mem.ledger(), "path_of": list(mem.path_of), "repo_of": list(mem.repo_of), "dropped_gold": dropped}


class DenseEvidenceIndex:
    """Static-word-vector embeddings of every registered commit subject; ``cs_for(task)`` is the cosine between the task's
    own (future) commit subject and the subjects registered when the task was scored."""

    def __init__(self, ledger: Dict[str, np.ndarray], wv: WordVectors):
        self.E = np.vstack([wv.embed(t) if t else np.zeros(wv.dim, np.float32) for t in ledger["c_subject"]]).astype(np.float32) \
            if len(ledger["c_subject"]) else np.zeros((0, wv.dim), np.float32)
        self.wv = wv
        self.cid = {s: i for i, s in enumerate(ledger["c_sha"])}

    def cs_for(self, task) -> np.ndarray:
        i = self.cid.get(task.sha)
        if i is None or task.nc == 0:
            return np.zeros(task.nc, dtype=np.float32)
        return np.clip(self.E[: task.nc] @ self.E[i], 0.0, 1.0).astype(np.float32)
