"""v0.4 confirmatory analysis on held-out repositories (protocol: docs/PROTOCOL_V04.md).
All systems are frozen: weights come from results/real_ext/summary_ext.json (selected on the original dev split)
or from the published EmemoVeri conditions.  Inference unit: repository."""
import json, pickle, sys, time
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, ".")
from ememoveri.realbench import Cond, rank_claims
from ememoveri.realbench_ext import METRICS, metrics_vec

src, frozen_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
KEY = sys.argv[4] if len(sys.argv) > 4 else "tfidf"   # e.g. "tfidf:full" for issue-text tasks
D = pickle.load(open(src, "rb"))
FZ = json.load(open(frozen_path))
tasks = D["tasks"]
shim = SimpleNamespace(path_of=D["path_of"], repo_of=D["repo_of"])
IDX = {m: k for k, m in enumerate(METRICS)}
repos = sorted({t.repo for t in tasks})
rng = np.random.default_rng(2026)
print(len(tasks), len(repos), flush=True)


def top_idx(t, cfg, state=True):
    s = t.sims[KEY].astype(np.float64)
    F = t.F
    if cfg.get("lp"):
        s = s + cfg["lp"] * (t.n_touch / (t.n_touch + 5.0))
    if cfg.get("le"):
        s = s + cfg["le"] * (F.ev_total / (F.ev_total + 5.0))
    if cfg.get("lt"):
        s = s + cfg["lt"] * F.trust_seq
    if cfg.get("lr"):
        s = s + cfg["lr"] * np.power(2.0, -F.age_days / cfg["H"])
    idx = np.flatnonzero(t.local & (~F.sup if state else True))
    return idx[np.lexsort((-F.updated[idx], -s[idx]))[:20]]


def run_cfg(cfg, state=True):
    return np.array([metrics_vec(shim, t.F, top_idx(t, cfg, state), t.gold, t.q_repo) for t in tasks])


def run_cond(c):
    return np.array([metrics_vec(shim, t.F, rank_claims(c, t.sims[KEY].astype(np.float64), t.F, t.local)[0], t.gold, t.q_repo)
                     for t in tasks])


SIM = dict(w_sim=1.0, w_trust=0.0, w_ev=0.0, w_scope=0.0, w_stale=0.0)
S = {}
S["Random"] = None
S["Recency"] = run_cond(Cond("RECENCY", kind="recency"))
S["Popularity"] = run_cond(Cond("POPULARITY", kind="popularity"))
S["Naive memory"] = run_cond(Cond("NAIVE", policy="local_only", state_aware=False, **SIM))
S["Snapshot RAG"] = run_cond(Cond("SNAPSHOT", policy="local_only", state_aware=True, **SIM))
S["EmemoVeri-D"] = run_cond(Cond("EV_DEFAULT"))
S["EmemoVeri-T"] = run_cond(Cond("EV_TUNED", 0.8, 0.0, 0.1, 0.1, 0.05, 365.0, "seq"))
del S["Random"]
strong = {r["name"]: r["cfg"] for r in FZ["strong"]}
for nm in ("Sim+Recency", "Sim+Popularity", "Sim+Recency+Popularity", "Sim+State+Recency", "Sim+State+Popularity", "Sim+State+Recency+Popularity"):
    S[nm] = run_cfg(strong[nm], state=nm.startswith("Sim+State"))
lad = {r["name"]: r["cfg"] for r in FZ["ladder"]}
LAD = ["R1 + state filter", "R2 + frequency prior", "R3 + evidence", "R4 + trust", "R5 + recency"]
S["R0 similarity (dead files kept)"] = S["Naive memory"]
for nm in LAD:
    S[nm] = run_cfg(lad[nm], state=True)
print("systems done", flush=True)

rep = np.array([t.repo for t in tasks])
month = np.array([t.month for t in tasks])


def per_repo(A, met):
    return np.array([A[rep == r, IDX[met]].mean() for r in repos])


def repo_boot(d, B=10000):
    """d: per-repository differences; resample repositories."""
    n = len(d)
    k = rng.integers(0, n, (B, n))
    m = d[k].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


R = {"repos": repos, "n_tasks": {r: int((rep == r).sum()) for r in repos}, "n_total": len(tasks), "counts": D.get("counts"),
     "resurrected": D.get("resurrected"), "systems": {}, "per_repo": {}}
for nm, A in S.items():
    R["systems"][nm] = {m: float(A[:, IDX[m]].mean()) for m in METRICS}
    R["systems"][nm].update({f"macro_{m}": float(per_repo(A, m).mean()) for m in ("hit1", "hit5", "rr", "ndcg5")})
    R["per_repo"][nm] = {m: dict(zip(repos, map(float, per_repo(A, m)))) for m in ("hit5", "rr")}

CONTR = {
    "H1 EmemoVeri-T - Snapshot": ("EmemoVeri-T", "Snapshot RAG"),
    "H2 EmemoVeri-T - Sim+State+Rec+Pop": ("EmemoVeri-T", "Sim+State+Recency+Popularity"),
    "H3 EmemoVeri-D - Snapshot": ("EmemoVeri-D", "Snapshot RAG"),
    "Sim+State+Rec+Pop - Snapshot": ("Sim+State+Recency+Popularity", "Snapshot RAG"),
    "Snapshot - Naive": ("Snapshot RAG", "Naive memory"),
    "EmemoVeri-T - EmemoVeri-D": ("EmemoVeri-T", "EmemoVeri-D"),
    "R5 - Sim+State+Rec+Pop": ("R5 + recency", "Sim+State+Recency+Popularity"),
}
for a, b in zip(["R0 similarity (dead files kept)"] + LAD[:-1], LAD):
    CONTR[f"ladder {b}"] = (b, a)
R["contrasts"] = {}
for nm, (a, b) in CONTR.items():
    row = {}
    for met in ("hit5", "rr", "ndcg5"):
        d = per_repo(S[a], met) - per_repo(S[b], met)
        row[met] = {"macro": repo_boot(d), "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()), "ties": int((np.abs(d) <= 1e-9).sum())}
    # pooled task-level and month-block bootstrap for reference
    dt = S[a][:, IDX["hit5"]] - S[b][:, IDX["hit5"]]
    n = len(dt)
    ii = rng.integers(0, n, (4000, n))
    row["pooled_task"] = [float(dt.mean()), float(np.percentile(dt[ii].mean(1), 2.5)), float(np.percentile(dt[ii].mean(1), 97.5))]
    u, inv = np.unique(rep + ":" + month, return_inverse=True)
    sm = np.bincount(inv, weights=dt, minlength=len(u)); cn = np.bincount(inv, minlength=len(u)).astype(float)
    kk = rng.integers(0, len(u), (4000, len(u)))
    mm = sm[kk].sum(1) / cn[kk].sum(1)
    row["pooled_month"] = [float(dt.mean()), float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5)), int(len(u))]
    R["contrasts"][nm] = row
    print(nm, [round(x, 3) for x in row["hit5"]["macro"]], row["hit5"]["wins"], row["hit5"]["losses"], flush=True)

# ---- lifecycle on held-out repositories (deletions, renames) ---------------------------------------
ev = D.get("events", [])
life = {}
for typ in (("delete", "rename") if ev else ()):
    es = [e for e in ev if e["type"] == typ and "post" in e]
    entry = {"n": len(es)}
    for sy in ("naive", "snapshot", "tuned"):
        entry[f"pre_hit_{sy}"] = float(np.mean([e["pre"][sy] for e in es]))
        entry[f"post_old_in_top5_{sy}"] = float(np.mean([e["post"][f"carry:{sy}"]["old_in_top5"] for e in es]))
    if typ == "rename":
        for k in ("carry:naive", "carry:snapshot", "carry:tuned", "nocarry:snapshot"):
            entry[f"post_new_in_top5_{k}"] = float(np.mean([e["post"][k]["new_in_top5"] for e in es]))
        pre = [e for e in es if e["pre"]["snapshot"]]
        entry["n_pre_hit_snapshot"] = len(pre)
        entry["continuity_given_pre_hit_carry"] = float(np.mean([e["post"]["carry:snapshot"]["new_in_top5"] for e in pre])) if pre else None
        entry["continuity_given_pre_hit_nocarry"] = float(np.mean([e["post"]["nocarry:snapshot"]["new_in_top5"] for e in pre])) if pre else None
    life[typ] = entry
R["lifecycle"] = life
json.dump(R, open(out_path, "w"), indent=1)
print("wrote", out_path)
