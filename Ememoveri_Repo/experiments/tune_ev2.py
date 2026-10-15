"""Tune EmemoVeri-2 on the ORIGINAL dev split only, then report the dev-test split and per-element ablations.

  python experiments/tune_ev2.py collected_dev.pkl results/real_ext/summary_v05.json results/real_ext/ev2_config.json results/real_ext/ev2_dev.json

Stage 1 chooses the evidence-level options (aggregator, strength weighting, TEST bonus, evidence decay, revert cancellation) at a
fixed blend; stage 2 tunes the blend (claim similarity, frequency, recency) with the SAME grid for the evidence-aware system and
for a plain system whose evidence-aware options are forced off, so the comparison gives both the same tuning budget."""
import itertools, json, pickle, sys, time
from dataclasses import replace
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, ".")
from ememoveri.evidence_retrieval import EvidenceConfig, item_scores, rank
from ememoveri.realbench_ext import METRICS, metrics_vec, DenseEvidenceIndex, WordVectors

dev_p, v05_p, cfg_out, res_out = sys.argv[1:5]
D = pickle.load(open(dev_p, "rb"))
L = D["ledger"]
shim = SimpleNamespace(path_of=D["path_of"], repo_of=D["repo_of"])
IDX = {m: k for k, m in enumerate(METRICS)}
dev = [t for t in D["tasks"] if t.split == "dev"]
test = [t for t in D["tasks"] if t.split == "test"]
rng = np.random.default_rng(6)
T0 = time.time()


DENSE = DenseEvidenceIndex(L, WordVectors())
_dense_cache = {}


def dense_cs(t):
    if t.sha not in _dense_cache:
        _dense_cache[t.sha] = DENSE.cs_for(t)
    return _dense_cache[t.sha]


def evaluate(tasks, cfg, sim_key="tfidf", cs_key=None):
    """cs_key: None (TF-IDF evidence cosines) | 'b' (subject+body query) | 'dense' | ('mix', lam)."""
    out = np.zeros((len(tasks), len(METRICS)))
    for j, t in enumerate(tasks):
        if cs_key == "b" and t.cs_b is not None:
            cs = t.cs_b
        elif cs_key == "dense":
            cs = dense_cs(t)
        elif isinstance(cs_key, tuple):
            cs = t.cs + cs_key[1] * dense_cs(t)
        else:
            cs = t.cs
        out[j] = metrics_vec(shim, t.F, rank(L, t, cfg, sim_key, cs), t.gold, t.q_repo)
    return out


def score(A):
    m = A.mean(0)
    return (round(float(m[IDX["rr"]]), 9), float(m[IDX["hit5"]]))


BASE = EvidenceConfig(w_sim=1.0, w_ev=1.0, l_pop=0.15, l_rec=0.05, H=30.0)
print("dev tasks", len(dev), "test", len(test), flush=True)

# ---- stage 1: evidence-level options -------------------------------------------------------------------
STAGE1 = [dict(agg=a, strength_exp=s, dilution_exp=dl, test_bonus=b, half_life_e=h, revert_cancel=c)
          for a in ("max", "sum", "noisyor") for s in (0.0, 1.0, 2.0) for dl in (0.0, 0.5, 1.0) for b in (0.0, 0.5) for h in (None, 365.0) for c in (False, True)]
res1 = []
for o in STAGE1:
    A = evaluate(dev, replace(BASE, **o))
    res1.append((score(A), o, float(A.mean(0)[IDX["hit5"]])))
res1.sort(key=lambda r: r[0], reverse=True)
print("stage1 top5:", [(r[0], r[1]) for r in res1[:5]], f"{time.time()-T0:.0f}s", flush=True)
plain1 = [r for r in res1 if r[1]["strength_exp"] == 0 and r[1]["dilution_exp"] == 0 and r[1]["test_bonus"] == 0 and r[1]["half_life_e"] is None and not r[1]["revert_cancel"]]
best_aware, best_plain = res1[0][1], plain1[0][1]
print("best aware", best_aware, "| best plain", best_plain, flush=True)

# ---- stage 2: blend, same grid for both ---------------------------------------------------------------------
BLEND = [dict(w_sim=ws, l_pop=lp, l_rec=lr, H=h) for ws in (0.5, 1.0, 2.0) for lp in (0.0, 0.05, 0.15, 0.4) for lr, h in ((0.0, 90.0), (0.05, 30.0), (0.05, 365.0), (0.15, 30.0))]


def tune_blend(opts):
    best = None
    for b in BLEND:
        cfg = replace(BASE, **opts, **b)
        sc = score(evaluate(dev, cfg))
        if best is None or sc > best[0]:
            best = (sc, cfg)
    return best[1], best[0]


cfg_aware, sc_a = tune_blend(best_aware)
cfg_plain, sc_p = tune_blend(best_plain)
print("EV2 (aware)", cfg_aware.as_dict(), sc_a, "\nplain      ", cfg_plain.as_dict(), sc_p, f"{time.time()-T0:.0f}s", flush=True)

# ---- stage 3: evidence-level dense (static word vectors) mixed into the item cosines ------------------------------
res3 = {}
for key in (None, ("mix", 0.25), ("mix", 0.5), ("mix", 1.0), "dense"):
    res3[str(key)] = score(evaluate(dev, cfg_aware, "tfidf", key))
print("stage3 dense mixing (dev):", res3, flush=True)
best3 = max(res3, key=lambda k: res3[k])
MIX = {"None": None, "('mix', 0.25)": ("mix", 0.25), "('mix', 0.5)": ("mix", 0.5), "('mix', 1.0)": ("mix", 1.0), "dense": "dense"}[best3]
print("selected evidence cosine:", best3, flush=True)

# ---- dev-test + per-element ablation (each element switched off from the tuned EV2) ------------------------------------
month = np.array([t.month for t in test]); filec = np.array([f"{t.repo}:{sorted(t.gold)[0][1]}" for t in test])


def boot(d, B=10000):
    i = rng.integers(0, len(d), (B, len(d))); m = d[i].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_cluster(d, lab, B=10000):
    u, inv = np.unique(lab, return_inverse=True)
    sm = np.bincount(inv, weights=d, minlength=len(u)); cn = np.bincount(inv, minlength=len(u)).astype(float)
    k = rng.integers(0, len(u), (B, len(u))); m = sm[k].sum(1) / cn[k].sum(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


A_aware, A_plain = evaluate(test, cfg_aware), evaluate(test, cfg_plain)
R = {"stage1": [{"opts": r[1], "dev_mrr": r[0][0], "dev_hit5": r[2]} for r in res1[:15]], "cfg_aware": cfg_aware.as_dict(), "cfg_plain": cfg_plain.as_dict(),
     "dev_scores": {"aware": sc_a, "plain": sc_p}, "test": {"aware": {m: float(A_aware[:, IDX[m]].mean()) for m in METRICS},
                                                               "plain": {m: float(A_plain[:, IDX[m]].mean()) for m in METRICS}}}
d = A_aware[:, IDX["hit5"]] - A_plain[:, IDX["hit5"]]
R["test"]["aware_minus_plain"] = {"task": boot(d), "month": boot_cluster(d, month), "file": boot_cluster(d, filec)}
R["stage3_dev"] = res3; R["selected_cs"] = best3
R["test_cs"] = {}
for key in (None, ("mix", 0.25), ("mix", 0.5), ("mix", 1.0), "dense"):
    A = evaluate(test, cfg_aware, "tfidf", key)
    R["test_cs"][str(key)] = {m: float(A[:, IDX[m]].mean()) for m in ("hit1", "hit5", "rr", "ndcg5")}
    print("test cs", key, round(R["test_cs"][str(key)]["hit5"], 3), flush=True)
A_sel = evaluate(test, cfg_aware, "tfidf", MIX)
d = A_sel[:, IDX["hit5"]] - A_aware[:, IDX["hit5"]]
R["test"]["selected_minus_tfidf_cs"] = {"task": boot(d), "month": boot_cluster(d, month)}
ABL = {"- strength weighting": dict(strength_exp=0.0), "- dilution weighting": dict(dilution_exp=0.0),
       "+ TEST bonus 0.5": dict(test_bonus=0.5), "+ evidence decay (365 d)": dict(half_life_e=365.0),
       "+ revert cancellation": dict(revert_cancel=True), "aggregator: noisy-OR": dict(agg="noisyor"), "aggregator: sum": dict(agg="sum"),
       "- all evidence-aware weighting": dict(strength_exp=0.0, dilution_exp=0.0, test_bonus=0.0, half_life_e=None, revert_cancel=False)}
R["ablation"] = {}
for name, o in ABL.items():
    A = evaluate(test, replace(cfg_aware, **o))
    d = A_aware[:, IDX["hit5"]] - A[:, IDX["hit5"]]
    R["ablation"][name] = {"hit5": float(A[:, IDX["hit5"]].mean()), "mrr": float(A[:, IDX["rr"]].mean()), "aware_minus": boot(d), "aware_minus_month": boot_cluster(d, month)}
    print("ablate", name, round(R["ablation"][name]["hit5"], 3), [round(x, 3) for x in R["ablation"][name]["aware_minus"]], flush=True)
# body-query variant on tasks with a commit body
hb = [t for t in test if t.has_body]
R["body"] = {"n": len(hb)}
if hb:
    for nm, cfg in (("aware", cfg_aware), ("plain", cfg_plain)):
        Ab = evaluate(hb, cfg, "tfidf_b", "b"); As = evaluate(hb, cfg, "tfidf", None)
        R["body"][nm] = {"subject_only": float(As[:, IDX["hit5"]].mean()), "subject_plus_body": float(Ab[:, IDX["hit5"]].mean())}
print("body", R["body"])
print("TEST aware", round(A_aware[:, IDX["hit5"]].mean(), 4), "plain", round(A_plain[:, IDX["hit5"]].mean(), 4), R["test"]["aware_minus_plain"]["task"], flush=True)
json.dump(cfg_aware.as_dict() | {"evidence_cosine": best3, "__note": "EmemoVeri-2, tuned on the original dev split only"}, open(cfg_out, "w"), indent=1)
json.dump(R, open(res_out, "w"), indent=1)
print("wrote", cfg_out, res_out, f"{time.time()-T0:.0f}s")
