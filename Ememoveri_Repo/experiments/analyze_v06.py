"""v0.6: EmemoVeri-2 against the frozen comparators on the fresh confirmatory set (docs/PROTOCOL_V06.md) and, post hoc, on the
sets used in v0.4/v0.5.

  python experiments/analyze_v06.py collected_dev.pkl collected_heldout.pkl collected_multilang.pkl collected_fresh.pkl \
      results/real_ext/summary_ext.json results/real_ext/summary_v05.json results/real_ext/ltr_models.pkl \
      results/real_ext/ev2_config.json results/real_ext/ev2_dev.json results/real_ext/task_flags.json results/real_ext/summary_v06.json"""
import json, pickle, sys, time
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, ".")
from ememoveri.evidence_retrieval import EvidenceConfig, rank as ev_rank
from ememoveri.realbench import Cond, rank_claims
from ememoveri.realbench_ext import METRICS, metrics_vec
from ememoveri.stats import holm, signflip_p, cohens_dz

(dev_p, held_p, ml_p, fresh_p, ext_p, v05_p, ltr_p, ev2_p, ev2dev_p, flags_p, out_p) = sys.argv[1:12]
EXT, V05, EV2D = json.load(open(ext_p)), json.load(open(v05_p)), json.load(open(ev2dev_p))
FLAGS = json.load(open(flags_p))
routine = lambda t: FLAGS[t.sha]["dup"] or FLAGS[t.sha]["pattern"]
raw = json.load(open(ev2_p))
CFG_EV2 = EvidenceConfig(**{k: v for k, v in raw.items() if k in EvidenceConfig.__dataclass_fields__})
CFG_PLAIN = EvidenceConfig(**EV2D["cfg_plain"])
MODELS = pickle.load(open(ltr_p, "rb"))
IDX = {m: k for k, m in enumerate(METRICS)}
rng = np.random.default_rng(66)
SIM = dict(w_sim=1.0, w_trust=0.0, w_ev=0.0, w_scope=0.0, w_stale=0.0)
C_SNAP, C_EVD = Cond("S", policy="local_only", state_aware=True, **SIM), Cond("D")
C_EVT = Cond("T", 0.8, 0.0, 0.1, 0.1, 0.05, 365.0, "seq")
SSRP = {r["name"]: r["cfg"] for r in EXT["strong"]}["Sim+State+Recency+Popularity"]
B1, ALL = V05["configs"]["B1 similar-history + sim + priors"], V05["configs"]["All channels + priors (linear)"]
SYSTEMS = ["Snapshot RAG", "EmemoVeri-D", "EmemoVeri-T", "Sim+State+Rec+Pop", "Similar-history (v0.5)", "All channels (linear)",
           "Learned ranker (logistic)", "Learned ranker (GBT)", "EmemoVeri-2 plain", "EmemoVeri-2"]


class Data:
    def __init__(self, path):
        D = pickle.load(open(path, "rb"))
        self.tasks, self.L = D["tasks"], D["ledger"]
        self.shim = SimpleNamespace(path_of=D["path_of"], repo_of=D["repo_of"])
        self.repos = sorted({t.repo for t in self.tasks})


def score_cfg(t, c):
    F, s = t.F, c.get("ws", 1.0) * t.sims["tfidf"].astype(np.float64)
    for k, ch in (("wm", "knn_max"), ("wsum", "knn_sum"), ("wc", "cochange")):
        if c.get(k):
            s = s + c[k] * t.sims[ch]
    if c.get("lp"): s = s + c["lp"] * (t.n_touch / (t.n_touch + 5.0))
    if c.get("le"): s = s + c["le"] * (F.ev_total / (F.ev_total + 5.0))
    if c.get("lt"): s = s + c["lt"] * F.trust_seq
    if c.get("lr"): s = s + c["lr"] * np.power(2.0, -F.age_days / c["H"])
    return s


def top_cfg(t, c):
    idx = np.flatnonzero(t.local & ~t.F.sup); s = score_cfg(t, c)
    return idx[np.lexsort((-t.F.updated[idx], -s[idx]))[:20]]


def cand_features(t, k=50):
    F = t.F; idx = np.flatnonzero(t.local & ~F.sup); sim = t.sims["tfidf"].astype(np.float64)
    order = idx[np.lexsort((-F.updated[idx], -sim[idx]))[:k]]; s = sim[order]
    X = np.column_stack([s, s - s.max() if len(s) else s, np.arange(len(order)) / float(k), t.sims["knn_max"][order], t.sims["knn_sum"][order],
                         t.sims["cochange"][order], t.n_touch[order] / (t.n_touch[order] + 5.0), F.ev_total[order] / (F.ev_total[order] + 5.0),
                         F.trust_seq[order], np.power(2.0, -F.age_days[order] / 30.0), np.power(2.0, -F.age_days[order] / 365.0), np.log1p(F.age_days[order])])
    return order, X


def run_all(D, tasks, key_sim="tfidf", cs_key=None):
    out = {}
    mv = lambda t, idx: metrics_vec(D.shim, t.F, idx, t.gold, t.q_repo)
    out["Snapshot RAG"] = np.array([mv(t, rank_claims(C_SNAP, t.sims[key_sim].astype(np.float64), t.F, t.local)[0]) for t in tasks])
    if key_sim == "tfidf":
        out["EmemoVeri-D"] = np.array([mv(t, rank_claims(C_EVD, t.sims["tfidf"].astype(np.float64), t.F, t.local)[0]) for t in tasks])
        out["EmemoVeri-T"] = np.array([mv(t, rank_claims(C_EVT, t.sims["tfidf"].astype(np.float64), t.F, t.local)[0]) for t in tasks])
        out["Sim+State+Rec+Pop"] = np.array([mv(t, top_cfg(t, SSRP)) for t in tasks])
        out["Similar-history (v0.5)"] = np.array([mv(t, top_cfg(t, B1)) for t in tasks])
        out["All channels (linear)"] = np.array([mv(t, top_cfg(t, ALL)) for t in tasks])
        for nm, key in (("Learned ranker (logistic)", "logistic"), ("Learned ranker (GBT)", "gbt")):
            A = np.zeros((len(tasks), len(METRICS)))
            for j, t in enumerate(tasks):
                order, X = cand_features(t)
                if len(order):
                    p = MODELS[key].predict_proba(X)[:, 1]
                    A[j] = mv(t, order[np.lexsort((-t.F.updated[order], -p))])
            out[nm] = A
    for nm, cfg in (("EmemoVeri-2 plain", CFG_PLAIN), ("EmemoVeri-2", CFG_EV2)):
        out[nm] = np.array([mv(t, ev_rank(D.L, t, cfg, key_sim, (t.cs_b if cs_key == "b" else t.cs))) for t in tasks])
    return out


def macro(A, rep, repos, met):
    return np.array([A[rep == r, IDX[met]].mean() for r in repos])


def repo_boot(d, B=10000):
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def task_boot(d, B=10000):
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(S, rep, repos, name_filter=None):
    out = {"systems": {}, "per_repo_hit5": {}}
    for nm, A in S.items():
        row = {m: float(A[:, IDX[m]].mean()) for m in METRICS}
        row.update({f"macro_{m}": float(macro(A, rep, repos, m).mean()) for m in ("hit1", "hit3", "hit5", "rr", "ndcg5", "rec5", "prec5")})
        row["macro_all5"] = float(np.mean([(A[rep == r, IDX["rec5"]] >= 1 - 1e-9).mean() for r in repos]))
        out["systems"][nm] = row
        out["per_repo_hit5"][nm] = dict(zip(repos, map(float, macro(A, rep, repos, "hit5"))))
    return out


FAMILY = [("H1 EV2 - Snapshot", "Snapshot RAG"), ("H2 EV2 - EmemoVeri-T", "EmemoVeri-T"), ("H3 EV2 - EV2 plain", "EmemoVeri-2 plain"),
          ("H4 EV2 - Sim+State+Rec+Pop", "Sim+State+Rec+Pop"), ("H5a EV2 - all channels (linear)", "All channels (linear)"),
          ("H5b EV2 - learned ranker (GBT)", "Learned ranker (GBT)")]
EXTRA = [("EV2 - similar-history (v0.5)", "Similar-history (v0.5)"), ("EV2 - EmemoVeri-D", "EmemoVeri-D"), ("EV2 - learned ranker (logistic)", "Learned ranker (logistic)")]


def contrasts(S, rep, repos):
    out, pv = {}, {}
    for lab, other in FAMILY + EXTRA:
        row = {}
        for met in ("hit5", "rr", "ndcg5"):
            d = macro(S["EmemoVeri-2"], rep, repos, met) - macro(S[other], rep, repos, met)
            row[met] = {"macro": repo_boot(d), "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()), "p_signflip": signflip_p(d), "dz": cohens_dz(d)}
        out[lab] = row
        if (lab, other) in FAMILY:
            pv[lab] = row["hit5"]["p_signflip"]
    for lab, p in holm(pv).items():
        out[lab]["hit5"]["p_holm"] = p
    return out


R = {"ev2_config": CFG_EV2.as_dict(), "plain_config": CFG_PLAIN.as_dict()}
T0 = time.time()
# ---- dev-test (task-level) -------------------------------------------------------------------------------
Dd = Data(dev_p)
te = [t for t in Dd.tasks if t.split == "test"]
S = run_all(Dd, te)
mon = np.array([t.month for t in te]); fil = np.array([f"{t.repo}:{sorted(t.gold)[0][1]}" for t in te])
R["dev_test"] = {"n": len(te), "systems": {nm: {m: float(A[:, IDX[m]].mean()) for m in METRICS} for nm, A in S.items()}, "ev2_minus": {}}
for nm in SYSTEMS[:-1]:
    d = S["EmemoVeri-2"][:, IDX["hit5"]] - S[nm][:, IDX["hit5"]]
    u, inv = np.unique(mon, return_inverse=True); sm = np.bincount(inv, weights=d); cn = np.bincount(inv).astype(float)
    k = rng.integers(0, len(u), (10000, len(u)))
    R["dev_test"]["ev2_minus"][nm] = {"task": task_boot(d), "month": [float(d.mean()), float(np.percentile(sm[k].sum(1) / cn[k].sum(1), 2.5)), float(np.percentile(sm[k].sum(1) / cn[k].sum(1), 97.5))]}
print("dev-test", {k: round(v["hit5"], 3) for k, v in R["dev_test"]["systems"].items()}, f"{time.time()-T0:.0f}s", flush=True)
hb = [t for t in te if t.has_body]
Sb = run_all(Dd, hb, "tfidf_b", "b"); Ss = run_all(Dd, hb)
R["dev_test_body"] = {"n": len(hb), "subject": {k: float(Ss[k][:, IDX["hit5"]].mean()) for k in ("Snapshot RAG", "EmemoVeri-2 plain", "EmemoVeri-2")},
                      "subject_plus_body": {k: float(Sb[k][:, IDX["hit5"]].mean()) for k in ("Snapshot RAG", "EmemoVeri-2 plain", "EmemoVeri-2")}}
print("dev-test body", R["dev_test_body"], flush=True)
del Dd

# ---- repository-level sets ------------------------------------------------------------------------------
PY10 = {"starlette", "uvicorn", "click", "jinja", "werkzeug", "attrs", "tornado", "marshmallow", "tqdm", "rq"}
for tag, path, status in (("fresh16", fresh_p, "confirmatory"), ("heldout12", held_p, "post-hoc"), ("multilang8", ml_p, "post-hoc")):
    D = Data(path)
    S = run_all(D, D.tasks)
    rep = np.array([t.repo for t in D.tasks])
    block = {"status": status, "repos": D.repos, "n_tasks": {r: int((rep == r).sum()) for r in D.repos}, "n_total": len(D.tasks)}
    block.update(summarize(S, rep, D.repos))
    block["contrasts"] = contrasts(S, rep, D.repos)
    keep = np.array([not routine(t) for t in D.tasks])
    Sn = {k: v[keep] for k, v in S.items()}; repn = rep[keep]
    nov = {"n_total": int(keep.sum())}
    nov.update(summarize(Sn, repn, D.repos)); nov["contrasts"] = contrasts(Sn, repn, D.repos)
    block["novel"] = nov
    if tag == "fresh16":
        py = [r for r in D.repos if r in PY10]; ot = [r for r in D.repos if r not in PY10]
        for nm, rs in (("python10", py), ("other6", ot)):
            m = np.isin(rep, rs); sub = {k: v[m] for k, v in S.items()}
            b = {"repos": rs, "n_total": int(m.sum())}; b.update(summarize(sub, rep[m], rs)); b["contrasts"] = contrasts(sub, rep[m], rs)
            block[nm] = b
        hbt = [t for t in D.tasks if t.has_body]
        Sb2 = run_all(D, hbt, "tfidf_b", "b"); Ss2 = run_all(D, hbt)
        repb = np.array([t.repo for t in hbt])
        block["body"] = {"n": len(hbt), "subject": {k: float(Ss2[k][:, IDX["hit5"]].mean()) for k in ("Snapshot RAG", "EmemoVeri-2 plain", "EmemoVeri-2")},
                         "subject_plus_body": {k: float(Sb2[k][:, IDX["hit5"]].mean()) for k in ("Snapshot RAG", "EmemoVeri-2 plain", "EmemoVeri-2")}}
    R[tag] = block
    print(tag, status, block["n_total"], {k: round(v["macro_hit5"], 3) for k, v in block["systems"].items()}, f"{time.time()-T0:.0f}s", flush=True)
    for lab, row in block["contrasts"].items():
        h = row["hit5"]
        print(f"   {lab:36s} {h['macro'][0]:+.3f} [{h['macro'][1]:+.3f},{h['macro'][2]:+.3f}] W/L {h['wins']}/{h['losses']} p={h['p_signflip']:.4f} holm={h.get('p_holm', float('nan')):.4f} dz={h['dz']:.2f}", flush=True)
    del D, S
json.dump(R, open(out_p, "w"), indent=1, default=float)
print("wrote", out_p, f"{time.time()-T0:.0f}s")
