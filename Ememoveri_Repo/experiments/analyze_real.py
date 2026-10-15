"""Aggregate raw.json -> summary.json / summary.md / figures. All paper numbers come from here."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ememoveri.stats import paired_bootstrap, mcnemar_exact, holm, risk_coverage, aurc, auroc, ece, brier

out = sys.argv[1] if len(sys.argv) > 1 else "results/real"
raw = json.load(open(os.path.join(out, "raw.json")))
rows = raw["rows"]
test = [r for r in rows if r["split"] == "test"]
dev = [r for r in rows if r["split"] == "dev"]
METRICS = ["hit1", "hit3", "hit5", "rr", "rec5", "stale5", "contam5"]

def col(rs, cond, m):
    return np.array([r["m"][cond][m] for r in rs if cond in r["m"]], float)

def agg(rs, cond):
    d = {m: float(col(rs, cond, m).mean()) for m in METRICS}
    d["n"] = int(len(col(rs, cond, "hit1")))
    return d

conds = [c for c in test[0]["m"] if not c.startswith(("SHUFFLED", "LEAKED"))]
S = {"n_dev": len(dev), "n_test": len(test), "tuned": raw["tuned"], "selection": raw["selection_stats"],
     "repo_meta": raw["repo_meta"], "n_claims_final": raw["n_claims_final"]}
S["test"] = {c: agg(test, c) for c in conds}
S["dev"] = {c: agg(dev, c) for c in conds if c in dev[0]["m"]}
S["controls"] = {c: agg(test, c) for c in test[0]["m"] if c.startswith(("SHUFFLED", "LEAKED"))}
S["per_repo"] = {repo: {c: agg([r for r in test if r["repo"] == repo], c) for c in
                        ["POPULARITY", "RECENCY", "PLAIN_RAG", "SNAPSHOT_RAG", "EMEMOVERI", "EMEMOVERI_TUNED"]}
                 for repo in sorted({r["repo"] for r in test})}
S["per_repo_n"] = {repo: sum(1 for r in test if r["repo"] == repo) for repo in S["per_repo"]}

# ---- paired comparisons (primary) ------------------------------------------------
comps = []
for a, b in [("EMEMOVERI", "PLAIN_RAG"), ("EMEMOVERI", "NAIVE_MEMORY"), ("EMEMOVERI", "SNAPSHOT_RAG"),
             ("EMEMOVERI", "POPULARITY"), ("EMEMOVERI_TUNED", "SNAPSHOT_RAG"), ("EMEMOVERI_TUNED", "EMEMOVERI"),
             ("EMEMOVERI_TUNED", "POPULARITY"), ("SNAPSHOT_RAG", "PLAIN_RAG")]:
    for m in ["hit1", "hit5", "rr", "stale5"]:
        x, y = col(test, a, m), col(test, b, m)
        diff, lo, hi = paired_bootstrap(x, y)
        rec = {"a": a, "b": b, "metric": m, "diff": diff, "ci_lo": lo, "ci_hi": hi}
        if m in ("hit1", "hit5"):
            n01, n10, p = mcnemar_exact(x, y)
            rec.update({"a_only": n10, "b_only": n01, "p": p})
        comps.append(rec)
adj = holm({f"{c['a']}|{c['b']}|{c['metric']}": c["p"] for c in comps if "p" in c})
for c in comps:
    if "p" in c:
        c["p_holm"] = adj[f"{c['a']}|{c['b']}|{c['metric']}"]
S["comparisons"] = comps

# ---- ablations: delta vs default ---------------------------------------------------
abl = {}
for c in conds:
    if c.startswith("ABL_"):
        diff, lo, hi = paired_bootstrap(col(test, c, "hit5"), col(test, "EMEMOVERI", "hit5"))
        d2, lo2, hi2 = paired_bootstrap(col(test, c, "rr"), col(test, "EMEMOVERI", "rr"))
        abl[c] = {"d_hit5": diff, "hit5_lo": lo, "hit5_hi": hi, "d_rr": d2, "rr_lo": lo2, "rr_hi": hi2}
S["ablations"] = abl

# ---- selective prediction / calibration (top-1 answer) --------------------------------
sel = {}
for c in ["EMEMOVERI", "EMEMOVERI_TUNED", "SNAPSHOT_RAG"]:
    corr = col(test, c, "top1_correct")
    gap = np.minimum(col(test, c, "gap"), 10.0)
    sc = col(test, c, "top_score")
    tr = col(test, c, "top1_trust")
    d = {"acc_all": float(corr.mean()), "aurc_gap": aurc(gap, corr), "aurc_score": aurc(sc, corr),
         "aurc_trust": aurc(tr, corr), "aurc_random": float(1 - corr.mean()),
         "auroc_gap": auroc(gap, corr), "auroc_score": auroc(sc, corr), "auroc_trust": auroc(tr, corr)}
    for thr in (0.05, 0.10, 0.20):
        ans = gap >= thr
        d[f"thr{thr}"] = {"coverage": float(ans.mean()),
                          "acc_answered": float(corr[ans].mean()) if ans.any() else float("nan")}
    cov, acc = risk_coverage(gap, corr)
    d["curve_gap"] = {"cov": cov[::10].tolist(), "acc": acc[::10].tolist()}
    cov, acc = risk_coverage(sc, corr)
    d["curve_score"] = {"cov": cov[::10].tolist(), "acc": acc[::10].tolist()}
    if c != "SNAPSHOT_RAG":
        d["trust_ece"] = ece(tr, corr)
        d["trust_brier"] = brier(tr, corr)
        d["trust_mean"] = float(tr.mean())
    sel[c] = d
S["selective"] = sel

# ---- state-machine exposure: how often do structural events actually fire? ---------
S["dead_claim_share_mean"] = float(np.mean([r["n_dead"] / max(r["n_claims"], 1) for r in test]))
S["tasks_with_dead_in_plain_top5"] = float((col(test, "PLAIN_RAG", "stale5") > 0).mean())
json.dump(S, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)

# ---- markdown summary ------------------------------------------------------------------
L = [f"# Real-repository benchmark summary (test n={len(test)}, dev n={len(dev)})\n",
     f"Tuned on dev only: `{json.dumps(raw['tuned'], default=float)}`\n",
     "| condition | Hit@1 | Hit@3 | Hit@5 | MRR | Recall@5 | Stale@5 | Contam@5 |", "|---|---|---|---|---|---|---|---|"]
for c in conds:
    a = S["test"][c]
    L.append(f"| {c} | {a['hit1']:.3f} | {a['hit3']:.3f} | {a['hit5']:.3f} | {a['rr']:.3f} | {a['rec5']:.3f} | {a['stale5']:.3f} | {a['contam5']:.3f} |")
L += ["", "## Controls", "| control | Hit@1 | Hit@5 | MRR |", "|---|---|---|---|"]
for c, a in S["controls"].items():
    L.append(f"| {c} | {a['hit1']:.3f} | {a['hit5']:.3f} | {a['rr']:.3f} |")
L += ["", "## Per repository (test)", "| repo (n) | cond | Hit@1 | Hit@5 | MRR |", "|---|---|---|---|---|"]
for repo, d in S["per_repo"].items():
    for c, a in d.items():
        L.append(f"| {repo} ({S['per_repo_n'][repo]}) | {c} | {a['hit1']:.3f} | {a['hit5']:.3f} | {a['rr']:.3f} |")
L += ["", "## Paired comparisons (test)", "| A | B | metric | diff | 95% CI | p (McNemar) | p (Holm) |", "|---|---|---|---|---|---|---|"]
for c in comps:
    p = f"{c['p']:.4f}" if "p" in c else "-"
    ph = f"{c['p_holm']:.4f}" if "p_holm" in c else "-"
    L.append(f"| {c['a']} | {c['b']} | {c['metric']} | {c['diff']:+.3f} | [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] | {p} | {ph} |")
L += ["", "## Ablations (delta vs EMEMOVERI default)", "| ablation | dHit@5 [95% CI] | dMRR [95% CI] |", "|---|---|---|"]
for c, a in abl.items():
    L.append(f"| {c} | {a['d_hit5']:+.3f} [{a['hit5_lo']:+.3f}, {a['hit5_hi']:+.3f}] | {a['d_rr']:+.3f} [{a['rr_lo']:+.3f}, {a['rr_hi']:+.3f}] |")
L += ["", "## Selective prediction (top-1)", "| system | acc(all) | AURC gap | AURC score | AURC trust | AUROC gap | AUROC score | AUROC trust | gap>=0.20 coverage / acc |", "|---|---|---|---|---|---|---|---|---|"]
for c, d in sel.items():
    t = d["thr0.2"]
    L.append(f"| {c} | {d['acc_all']:.3f} | {d['aurc_gap']:.3f} | {d['aurc_score']:.3f} | {d['aurc_trust']:.3f} | {d['auroc_gap']:.3f} | {d['auroc_score']:.3f} | {d['auroc_trust']:.3f} | {t['coverage']:.3f} / {t['acc_answered']:.3f} |")
for c in ("EMEMOVERI", "EMEMOVERI_TUNED"):
    L.append(f"\n{c} trust calibration on top-1: ECE={sel[c]['trust_ece']:.3f}, Brier={sel[c]['trust_brier']:.3f}, mean trust={sel[c]['trust_mean']:.3f}")
open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
