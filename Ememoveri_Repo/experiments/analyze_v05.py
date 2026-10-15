"""v0.5 analyses (protocol: docs/PROTOCOL_V05.md).

  python experiments/analyze_v05.py collected_dev.pkl collected_heldout.pkl collected_multilang.pkl \
         results/real_ext/summary_ext.json results/real_ext/summary_v05.json results/real_ext/ltr_models.pkl results/real_ext/task_flags.json

1. Tune the new baselines (similar-history, co-change, all-channel linear, learned rankers) on the ORIGINAL dev split only.
2. Evaluate them, together with the frozen v0.4 systems, on the dev-test split (task-level inference), the 12 Python held-out
   repositories and the 8 multi-language repositories (repository-level inference: bootstrap over repositories, exact
   sign-flip test, Holm correction within each contrast family, Cohen's d_z)."""
import json, pickle, sys, time
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, ".")
from ememoveri.realbench import Cond, rank_claims
from ememoveri.realbench_ext import METRICS, metrics_vec
from ememoveri.stats import holm, signflip_p, cohens_dz

dev_p, held_p, ml_p, frozen_p, out_p, ltr_p, flags_p = sys.argv[1:8]
FLAGS = json.load(open(flags_p))
routine = lambda t: FLAGS[t.sha]["dup"] or FLAGS[t.sha]["pattern"]
FZ = json.load(open(frozen_p))
IDX = {m: k for k, m in enumerate(METRICS)}
rng = np.random.default_rng(5)
T0 = time.time()
SIM = dict(w_sim=1.0, w_trust=0.0, w_ev=0.0, w_scope=0.0, w_stale=0.0)
C_SNAP = Cond("SNAP", policy="local_only", state_aware=True, **SIM)
C_EVD = Cond("EV_DEFAULT")
C_EVT = Cond("EV_TUNED", 0.8, 0.0, 0.1, 0.1, 0.05, 365.0, "seq")
SSRP = {r["name"]: r["cfg"] for r in FZ["strong"]}["Sim+State+Recency+Popularity"]


class Data:
    def __init__(self, path):
        D = pickle.load(open(path, "rb"))
        self.tasks = D["tasks"]
        self.shim = SimpleNamespace(path_of=D["path_of"], repo_of=D["repo_of"])
        self.repos = sorted({t.repo for t in self.tasks})


# ------------------------------------------------------------------ scoring ----
def score(t, cfg):
    F = t.F
    s = cfg.get("ws", 1.0) * t.sims["tfidf"].astype(np.float64)
    if cfg.get("wm"):
        s = s + cfg["wm"] * t.sims["knn_max"]
    if cfg.get("wsum"):
        s = s + cfg["wsum"] * t.sims["knn_sum"]
    if cfg.get("wc"):
        s = s + cfg["wc"] * t.sims["cochange"]
    if cfg.get("lp"):
        s = s + cfg["lp"] * (t.n_touch / (t.n_touch + 5.0))
    if cfg.get("le"):
        s = s + cfg["le"] * (F.ev_total / (F.ev_total + 5.0))
    if cfg.get("lt"):
        s = s + cfg["lt"] * F.trust_seq
    if cfg.get("lr"):
        s = s + cfg["lr"] * np.power(2.0, -F.age_days / cfg["H"])
    return s


def top_cfg(t, cfg):
    mask = t.local & ~t.F.sup
    s = score(t, cfg)
    idx = np.flatnonzero(mask)
    if cfg.get("seed"):                                    # co-change recommendation: the seed file comes first
        sim = t.sims["tfidf"][idx]
        s = s.copy(); s[idx[int(np.argmax(sim))]] += 100.0
    return idx[np.lexsort((-t.F.updated[idx], -s[idx]))[:20]]


def eval_cfgs(D, tasks, cfgs):
    A = np.zeros((len(cfgs), len(tasks), len(METRICS)))
    for j, t in enumerate(tasks):
        for g, cfg in enumerate(cfgs):
            A[g, j] = metrics_vec(D.shim, t.F, top_cfg(t, cfg), t.gold, t.q_repo)
    return A


def eval_cond(D, tasks, cond):
    return np.array([metrics_vec(D.shim, t.F, rank_claims(cond, t.sims["tfidf"].astype(np.float64), t.F, t.local)[0], t.gold, t.q_repo)
                     for t in tasks])


def pick(A):
    m = A.mean(1)
    return max(range(A.shape[0]), key=lambda g: (round(m[g, IDX["rr"]], 9), m[g, IDX["hit5"]]))


# ------------------------------------------------------------------ learned rankers ----
def cand_features(t, k=50):
    F = t.F
    mask = t.local & ~F.sup
    idx = np.flatnonzero(mask)
    sim = t.sims["tfidf"].astype(np.float64)
    order = idx[np.lexsort((-F.updated[idx], -sim[idx]))[:k]]
    s = sim[order]
    X = np.column_stack([
        s, s - s.max() if len(s) else s, np.arange(len(order)) / float(k),
        t.sims["knn_max"][order], t.sims["knn_sum"][order], t.sims["cochange"][order],
        t.n_touch[order] / (t.n_touch[order] + 5.0), F.ev_total[order] / (F.ev_total[order] + 5.0), F.trust_seq[order],
        np.power(2.0, -F.age_days[order] / 30.0), np.power(2.0, -F.age_days[order] / 365.0), np.log1p(F.age_days[order])])
    return order, X


def ltr_train(D, tasks):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xs, ys = [], []
    for t in tasks:
        order, X = cand_features(t)
        if len(order) == 0:
            continue
        keys = [(int(D.shim.repo_of[i]), D.shim.path_of[i]) for i in order]
        Xs.append(X); ys.append(np.array([k in t.gold for k in keys], dtype=int))
    X, y = np.vstack(Xs), np.concatenate(ys)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)
    gb = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1, random_state=0).fit(X, y)
    return {"logistic": lr, "gbt": gb, "n_train_tasks": len(Xs), "n_pos": int(y.sum()), "n_rows": int(len(y))}


def eval_ltr(D, tasks, model):
    A = np.zeros((len(tasks), len(METRICS)))
    for j, t in enumerate(tasks):
        order, X = cand_features(t)
        if len(order) == 0:
            continue
        p = model.predict_proba(X)[:, 1]
        o = order[np.lexsort((-t.F.updated[order], -p))]
        A[j] = metrics_vec(D.shim, t.F, o, t.gold, t.q_repo)
    return A


# ------------------------------------------------------------------ 1. tune on dev ----
dev = Data(dev_p)
dev_tr = [t for t in dev.tasks if t.split == "dev"]
dev_te = [t for t in dev.tasks if t.split == "test"]
print("dev tasks", len(dev_tr), "test", len(dev_te), flush=True)
LAMP = (0.0, 0.05, 0.15, 0.4)
REC = [(0.0, 90.0), (0.05, 30.0), (0.05, 365.0)]
G = {
    "B1 similar-history (alone)": [dict(ws=0.0, wm=1.0)],
    "B1 similar-history + sim + priors": [dict(ws=ws, wm=1.0, wsum=wsum, lp=lp, lr=lr, H=h)
                                          for ws in (0.0, 0.5, 1.0, 2.0) for wsum in (0.0, 0.1, 0.3) for lp in LAMP for lr, h in REC],
    "B2 co-change (alone)": [dict(ws=1.0, wc=5.0, seed=True)],
    "B2 co-change + sim": [dict(ws=1.0, wc=wc) for wc in (0.05, 0.15, 0.4, 1.0)],
    "All channels + priors (linear)": [dict(ws=1.0, wm=wm, wsum=wsum, wc=wc, lp=lp, lr=lr, H=h)
                                       for wm in (0.0, 0.5, 1.0) for wsum in (0.0, 0.1) for wc in (0.0, 0.15, 0.4) for lp in LAMP for lr, h in REC],
}
FROZEN = {"Snapshot RAG": None}
CFG = {}
for name, grid in G.items():
    if len(grid) == 1:
        CFG[name] = grid[0]
    else:
        A = eval_cfgs(dev, dev_tr, grid)
        CFG[name] = grid[pick(A)]
    print("tuned", name, CFG[name], f"{time.time()-T0:.0f}s", flush=True)
models = ltr_train(dev, dev_tr)
pickle.dump(models, open(ltr_p, "wb"))
print("ltr trained", models["n_train_tasks"], models["n_pos"], models["n_rows"], flush=True)

SYSTEMS = ["Snapshot RAG", "EmemoVeri-D", "EmemoVeri-T", "Sim+State+Rec+Pop", *G.keys(), "Learned ranker (logistic)", "Learned ranker (GBT)"]


def run_all(D, tasks):
    S = {"Snapshot RAG": eval_cond(D, tasks, C_SNAP), "EmemoVeri-D": eval_cond(D, tasks, C_EVD), "EmemoVeri-T": eval_cond(D, tasks, C_EVT),
         "Sim+State+Rec+Pop": eval_cfgs(D, tasks, [SSRP])[0]}
    for name in G:
        S[name] = eval_cfgs(D, tasks, [CFG[name]])[0]
    S["Learned ranker (logistic)"] = eval_ltr(D, tasks, models["logistic"])
    S["Learned ranker (GBT)"] = eval_ltr(D, tasks, models["gbt"])
    return S


R = {"configs": CFG, "ltr": {k: models[k] for k in ("n_train_tasks", "n_pos", "n_rows")}, "systems": SYSTEMS}

# ------------------------------------------------------------------ 2a. dev-test (task-level inference) ----
S = run_all(dev, dev_te)
month = np.array([t.month for t in dev_te])
filec = np.array([f"{t.repo}:{sorted(t.gold)[0][1]}" for t in dev_te])


def boot_task(d, B=10000):
    i = rng.integers(0, len(d), (B, len(d)))
    m = d[i].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_cluster(d, lab, B=10000):
    u, inv = np.unique(lab, return_inverse=True)
    sm = np.bincount(inv, weights=d, minlength=len(u)); cn = np.bincount(inv, minlength=len(u)).astype(float)
    k = rng.integers(0, len(u), (B, len(u)))
    m = sm[k].sum(1) / cn[k].sum(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


R["dev_test"] = {"n": len(dev_te), "systems": {}}
for name, A in S.items():
    row = {m: float(A[:, IDX[m]].mean()) for m in METRICS}
    if name != "EmemoVeri-T":
        d = S["EmemoVeri-T"][:, IDX["hit5"]] - A[:, IDX["hit5"]]
        row["evt_minus"] = {"task": boot_task(d), "month": boot_cluster(d, month), "file": boot_cluster(d, filec)}
    R["dev_test"]["systems"][name] = row
    print("dev-test", f"{name:36s} hit5 {row['hit5']:.3f} mrr {row['rr']:.3f}", flush=True)

# ------------------------------------------------------------------ 2b. repository-level sets ----
CONTRASTS = [("H1 EmemoVeri-T - Snapshot RAG", "EmemoVeri-T", "Snapshot RAG"),
             ("H2 EmemoVeri-T - Sim+State+Rec+Pop", "EmemoVeri-T", "Sim+State+Rec+Pop"),
             ("H3 EmemoVeri-D - Snapshot RAG", "EmemoVeri-D", "Snapshot RAG"),
             ("H4 EmemoVeri-T - similar-history (tuned)", "EmemoVeri-T", "B1 similar-history + sim + priors"),
             ("H4 EmemoVeri-T - co-change + sim (tuned)", "EmemoVeri-T", "B2 co-change + sim"),
             ("H4 EmemoVeri-T - all channels (linear)", "EmemoVeri-T", "All channels + priors (linear)"),
             ("H4 EmemoVeri-T - learned ranker (logistic)", "EmemoVeri-T", "Learned ranker (logistic)"),
             ("H4 EmemoVeri-T - learned ranker (GBT)", "EmemoVeri-T", "Learned ranker (GBT)")]


def repo_boot(d, B=10000):
    k = rng.integers(0, len(d), (B, len(d)))
    m = d[k].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def analyze_set(D, tasks, S_all, tag):
    keep = np.array([not routine(t) for t in D.tasks]) if tag.endswith("_novel") else np.ones(len(D.tasks), bool)
    S = {k: v[keep] for k, v in S_all.items()}
    tasks = [t for t, k in zip(D.tasks, keep) if k]
    rep = np.array([t.repo for t in tasks])
    per = lambda A, met: np.array([A[rep == r, IDX[met]].mean() for r in D.repos])
    out = {"repos": D.repos, "n_tasks": {r: int((rep == r).sum()) for r in D.repos}, "n_total": len(tasks), "systems": {}, "per_repo_hit5": {}, "contrasts": {}}
    for name, A in S.items():
        out["systems"][name] = {**{m: float(A[:, IDX[m]].mean()) for m in METRICS},
                                **{f"macro_{m}": float(per(A, m).mean()) for m in ("hit1", "hit5", "rr", "ndcg5")}}
        out["per_repo_hit5"][name] = dict(zip(D.repos, map(float, per(A, "hit5"))))
    pv = {}
    for lab, a_, b_ in CONTRASTS:
        row = {}
        for met in ("hit5", "rr", "ndcg5"):
            d = per(S[a_], met) - per(S[b_], met)
            row[met] = {"macro": repo_boot(d), "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()),
                        "p_signflip": signflip_p(d), "dz": cohens_dz(d)}
        out["contrasts"][lab] = row
        pv[lab] = row["hit5"]["p_signflip"]
    for lab, p in holm(pv).items():
        out["contrasts"][lab]["hit5"]["p_holm"] = p
    R[tag] = out
    print(tag, out["n_total"], {s_: round(out["systems"][s_]["macro_hit5"], 3) for s_ in SYSTEMS}, flush=True)
    for lab, row in out["contrasts"].items():
        h = row["hit5"]
        print(f"  {lab:46s} {h['macro'][0]:+.3f} [{h['macro'][1]:+.3f},{h['macro'][2]:+.3f}] W/L {h['wins']}/{h['losses']} p={h['p_signflip']:.4f} holm={h['p_holm']:.4f} dz={h['dz']:.2f}", flush=True)


def lifecycle(D_path):
    ev = pickle.load(open(D_path, "rb")).get("events", [])
    life = {}
    for typ in ("delete", "rename"):
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
    return life


for tag, path in (("heldout12", held_p), ("multilang8", ml_p)):
    D = Data(path)
    S_all = run_all(D, D.tasks)
    analyze_set(D, D.tasks, S_all, tag)
    analyze_set(D, D.tasks, S_all, tag + "_novel")
    if tag == "multilang8":
        R["multilang8_lifecycle"] = lifecycle(path)

# dev-test, non-routine tasks
keep = np.array([not routine(t) for t in dev_te])
S_all = run_all(dev, dev_te)
R["dev_test_novel"] = {"n": int(keep.sum()), "systems": {}}
for name, A in S_all.items():
    R["dev_test_novel"]["systems"][name] = {m: float(A[keep][:, IDX[m]].mean()) for m in METRICS}
    if name != "EmemoVeri-T":
        d = (S_all["EmemoVeri-T"][:, IDX["hit5"]] - A[:, IDX["hit5"]])[keep]
        R["dev_test_novel"]["systems"][name]["evt_minus"] = {"task": boot_task(d)}
print("dev-test novel", int(keep.sum()), {k: round(v["hit5"], 3) for k, v in R["dev_test_novel"]["systems"].items()})

json.dump(R, open(out_p, "w"), indent=1, default=float)
print("wrote", out_p, f"{time.time()-T0:.0f}s")
