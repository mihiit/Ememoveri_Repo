"""Figures for the real-repository benchmark (reads raw.json / summary.json)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ememoveri.stats import risk_coverage

out = sys.argv[1] if len(sys.argv) > 1 else "results/real"
raw = json.load(open(os.path.join(out, "raw.json"))); S = json.load(open(os.path.join(out, "summary.json")))
test = [r for r in raw["rows"] if r["split"] == "test"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
LAB = {"RECENCY": "Recency", "POPULARITY": "Popularity", "PLAIN_RAG": "Plain RAG\n(all repos)", "NAIVE_MEMORY": "Naive\nmemory",
       "SNAPSHOT_RAG": "Snapshot\nRAG", "EMEMOVERI": "EmemoVeri\n(default)", "EMEMOVERI_TUNED": "EmemoVeri\n(dev-tuned)"}
COL = {"RECENCY": "#b0b0b0", "POPULARITY": "#b0b0b0", "PLAIN_RAG": "#8fb8de", "NAIVE_MEMORY": "#8fb8de", "SNAPSHOT_RAG": "#4a7fb5",
       "EMEMOVERI": "#e0a030", "EMEMOVERI_TUNED": "#c05a1a"}

def ci(cond, m, rs=test, n=4000):
    x = np.array([r["m"][cond][m] for r in rs], float)
    rng = np.random.default_rng(0)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

# 1 main
names = list(LAB)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, m, t in zip(axs, ["hit5", "rr"], ["Hit@5", "MRR"]):
    v = [ci(c, m) for c in names]
    ax.bar(range(len(names)), [a for a, _, _ in v], color=[COL[c] for c in names],
           yerr=[[a - l for a, l, _ in v], [h - a for a, _, h in v]], capsize=2, linewidth=0)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([LAB[c] for c in names], fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel(t); ax.set_ylim(0, 0.9)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig_main.pdf")); fig.savefig(os.path.join(out, "fig_main.png")); plt.close(fig)

# 2 per repo
repos = sorted({r["repo"] for r in test}); sel = ["POPULARITY", "PLAIN_RAG", "SNAPSHOT_RAG", "EMEMOVERI", "EMEMOVERI_TUNED"]
fig, ax = plt.subplots(figsize=(6.2, 2.6)); w = 0.16
for j, c in enumerate(sel):
    vals = [ci(c, "hit5", [r for r in test if r["repo"] == rp]) for rp in repos]
    ax.bar(np.arange(len(repos)) + (j - 2) * w, [a for a, _, _ in vals], w, label=LAB[c].replace("\n", " "), color=COL[c],
           yerr=[[a - l for a, l, _ in vals], [h - a for a, _, h in vals]], capsize=1.5, linewidth=0)
ax.set_xticks(range(len(repos))); ax.set_xticklabels([f"{rp} (n={S['per_repo_n'][rp]})" for rp in repos])
ax.set_ylabel("Hit@5"); ax.set_ylim(0, 1.05); ax.legend(fontsize=6.5, ncol=3, loc="upper center", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig_repo.pdf")); fig.savefig(os.path.join(out, "fig_repo.png")); plt.close(fig)

# 3 ablations
ab = S["ablations"]; keys = [k for k in ab if k not in ("ABL_policy_open", "ABL_policy_local_only", "ABL_policy_sim_gate")]
keys.sort(key=lambda k: ab[k]["d_hit5"])
fig, ax = plt.subplots(figsize=(5.4, 3.0))
y = np.arange(len(keys)); d = [ab[k]["d_hit5"] for k in keys]
ax.barh(y, d, xerr=[[ab[k]["d_hit5"] - ab[k]["hit5_lo"] for k in keys], [ab[k]["hit5_hi"] - ab[k]["d_hit5"] for k in keys]],
        color=["#c05a1a" if v < 0 else "#4a7fb5" for v in d], capsize=2, linewidth=0)
ax.axvline(0, color="k", lw=0.6); ax.set_yticks(y); ax.set_yticklabels([k.replace("ABL_", "") for k in keys], fontsize=7)
ax.set_xlabel("Change in Hit@5 vs default EmemoVeri (95% CI)")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig_ablation.pdf")); fig.savefig(os.path.join(out, "fig_ablation.png")); plt.close(fig)

# 4 selective prediction
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for c, col, lab in [("SNAPSHOT_RAG", COL["SNAPSHOT_RAG"], "Snapshot RAG (sim. gap)"), ("EMEMOVERI", COL["EMEMOVERI"], "EmemoVeri default (score gap)"),
                    ("EMEMOVERI_TUNED", COL["EMEMOVERI_TUNED"], "EmemoVeri tuned (score gap)")]:
    corr = np.array([r["m"][c]["top1_correct"] for r in test]); gap = np.minimum(np.array([r["m"][c]["gap"] for r in test], float), 10)
    cov, acc = risk_coverage(gap, corr); ax.plot(cov, acc, color=col, label=lab, lw=1.4)
ax.set_xlabel("Coverage (fraction answered)"); ax.set_ylabel("Top-1 accuracy on answered"); ax.set_ylim(0, 1.02)
ax.legend(fontsize=6.5, frameon=False); fig.tight_layout()
fig.savefig(os.path.join(out, "fig_selective.pdf")); fig.savefig(os.path.join(out, "fig_selective.png")); plt.close(fig)

# 5 stale + contamination
fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.4)); sc = ["PLAIN_RAG", "NAIVE_MEMORY", "SNAPSHOT_RAG", "EMEMOVERI"]
for ax, m, t in zip(axs, ["stale5", "contam5"], ["Dead paths in top-5", "Other-repo files in top-5"]):
    v = [ci(c, m) for c in sc]
    ax.bar(range(4), [a for a, _, _ in v], color=[COL[c] for c in sc], linewidth=0, yerr=[[a - l for a, l, _ in v], [h - a for a, _, h in v]], capsize=2)
    ax.set_xticks(range(4)); ax.set_xticklabels([LAB[c] for c in sc], fontsize=6.5, rotation=45, ha="right"); ax.set_ylabel(t, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig_hazards.pdf")); fig.savefig(os.path.join(out, "fig_hazards.png")); plt.close(fig)
print("figures written")
