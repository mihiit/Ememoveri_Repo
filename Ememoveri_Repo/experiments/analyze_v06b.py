"""v0.6 supplementary analyses for EmemoVeri-2 (frozen): (a) query degradation on the dev-test split, (b) contradiction handling
through revert cancellation on every repository set.

  python experiments/analyze_v06b.py collected_dev.pkl collected_heldout.pkl collected_multilang.pkl collected_fresh.pkl \
      results/real_ext/summary_ext.json results/real_ext/summary_v05.json results/real_ext/ev2_config.json results/real_ext/ev2_dev.json out.json"""
import json, pickle, sys
from dataclasses import replace
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, ".")
from ememoveri.evidence_retrieval import EvidenceConfig, rank as ev_rank
from ememoveri.realbench import Cond, rank_claims
from ememoveri.realbench_ext import METRICS, metrics_vec
from ememoveri.stats import signflip_p

dev_p, held_p, ml_p, fresh_p, ext_p, v05_p, ev2_p, ev2dev_p, out_p = sys.argv[1:10]
EXT, V05, EV2D = json.load(open(ext_p)), json.load(open(v05_p)), json.load(open(ev2dev_p))
raw = json.load(open(ev2_p))
EV2 = EvidenceConfig(**{k: v for k, v in raw.items() if k in EvidenceConfig.__dataclass_fields__})
PLAIN = EvidenceConfig(**EV2D["cfg_plain"])
IDX = {m: k for k, m in enumerate(METRICS)}
rng = np.random.default_rng(606)
SIM = dict(w_sim=1.0, w_trust=0.0, w_ev=0.0, w_scope=0.0, w_stale=0.0)
C_SNAP, C_EVD, C_EVT = Cond("S", policy="local_only", state_aware=True, **SIM), Cond("D"), Cond("T", 0.8, 0.0, 0.1, 0.1, 0.05, 365.0, "seq")
SSRP = {r["name"]: r["cfg"] for r in EXT["strong"]}["Sim+State+Recency+Popularity"]
B1 = V05["configs"]["B1 similar-history + sim + priors"]


def boot(d, B=10000):
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def ssrp_idx(t, c):
    F = t.F; s = c.get("ws", 1.0) * t.sims["tfidf"].astype(np.float64)
    if c.get("wm"): s = s + c["wm"] * t.sims["knn_max"]
    if c.get("lp"): s = s + c["lp"] * (t.n_touch / (t.n_touch + 5.0))
    if c.get("lr"): s = s + c["lr"] * np.power(2.0, -F.age_days / c["H"])
    idx = np.flatnonzero(t.local & ~F.sup)
    return idx[np.lexsort((-F.updated[idx], -s[idx]))[:20]]


R = {}
# ---------------------------------------------------------------- (a) degraded queries ------------------------------------------
D = pickle.load(open(dev_p, "rb")); L = D["ledger"]; shim = SimpleNamespace(path_of=D["path_of"], repo_of=D["repo_of"])
te = [t for t in D["tasks"] if t.split == "test"]
mv = lambda t, idx: metrics_vec(shim, t.F, idx, t.gold, t.q_repo)
R["degrade"] = {}
for lvl in (0.0, 0.3, 0.5, 0.7):
    key = "tfidf" if lvl == 0 else f"tfidf@{lvl}"
    row, per = {}, {}
    for nm in ("Snapshot RAG", "EmemoVeri-T", "Sim+State+Rec+Pop", "EmemoVeri-2 plain", "EmemoVeri-2"):
        A = np.zeros((len(te), len(METRICS)))
        for j, t in enumerate(te):
            sim = t.sims[key].astype(np.float64)
            cs = t.cs if lvl == 0 else t.cs_deg[lvl]
            if nm == "Snapshot RAG": idx = rank_claims(C_SNAP, sim, t.F, t.local)[0]
            elif nm == "EmemoVeri-T": idx = rank_claims(C_EVT, sim, t.F, t.local)[0]
            elif nm == "Sim+State+Rec+Pop":
                tt = SimpleNamespace(**{**t.__dict__, "sims": {**t.sims, "tfidf": sim}}); idx = ssrp_idx(tt, SSRP)
            else: idx = ev_rank(L, t, EV2 if nm == "EmemoVeri-2" else PLAIN, key, cs)
            A[j] = mv(t, idx)
        per[nm] = A; row[nm] = {m: float(A[:, IDX[m]].mean()) for m in ("hit1", "hit5", "rr", "ndcg5")}
    for nm in ("Snapshot RAG", "EmemoVeri-T", "Sim+State+Rec+Pop", "EmemoVeri-2 plain"):
        row[nm]["ev2_minus"] = boot(per["EmemoVeri-2"][:, IDX["hit5"]] - per[nm][:, IDX["hit5"]])
    R["degrade"][str(lvl)] = row
    print("degrade", lvl, {k: round(v["hit5"], 3) for k, v in row.items()}, flush=True)
del D

# ---------------------------------------------------------------- (b) contradiction through revert cancellation ------------
CANCEL = replace(EV2, revert_cancel=True)
R["revert"] = {}
for tag, path in (("dev", dev_p), ("heldout12", held_p), ("multilang8", ml_p), ("fresh16", fresh_p)):
    D = pickle.load(open(path, "rb")); L = D["ledger"]; shim = SimpleNamespace(path_of=D["path_of"], repo_of=D["repo_of"])
    mv = lambda t, idx: metrics_vec(shim, t.F, idx, t.gold, t.q_repo)
    tasks = D["tasks"]
    n_cancelled = int(np.isfinite(L["c_cancel"]).sum())
    exposed, a_off, a_on = [], [], []
    for t in tasks:
        n = len(t.F.sup)
        end = int(L["ptr"][n])
        com = L["commit"][:end]
        # the task is *exposed* if any live local claim holds evidence from a commit that a revert has cancelled by then
        live = t.local & ~t.F.sup
        owner = L["owner"][:end]
        hit = (com < t.nc) & (t.ning >= L["c_cancel"][np.minimum(com, len(L["c_cancel"]) - 1)]) & live[owner]
        if not hit.any():
            continue
        r_off, r_on = ev_rank(L, t, EV2), ev_rank(L, t, CANCEL)
        exposed.append(t)
        a_off.append(mv(t, r_off)); a_on.append(mv(t, r_on))
    a_off, a_on = np.array(a_off).reshape(-1, len(METRICS)), np.array(a_on).reshape(-1, len(METRICS))
    row = {"n_tasks": len(tasks), "cancelled_commits": n_cancelled, "exposed": len(exposed)}
    if len(exposed):
        changed = (a_off != a_on).any(1)
        d = a_on[:, IDX["hit5"]] - a_off[:, IDX["hit5"]]
        better, worse = int((d > 0).sum()), int((d < 0).sum())
        from scipy.stats import binomtest
        row.update({"ranking_changed": int(changed.sum()), "hit5_off": float(a_off[:, IDX["hit5"]].mean()), "hit5_on": float(a_on[:, IDX["hit5"]].mean()),
                    "mrr_off": float(a_off[:, IDX["rr"]].mean()), "mrr_on": float(a_on[:, IDX["rr"]].mean()), "better": better, "worse": worse,
                    "p_sign": float(binomtest(better, better + worse, 0.5).pvalue) if better + worse else 1.0, "delta_hit5": boot(d) if len(d) > 1 else [float(d.mean())] * 3})
    R["revert"][tag] = row
    print("revert", tag, row, flush=True)
    del D
json.dump(R, open(out_p, "w"), indent=1, default=float)
print("wrote", out_p)
