"""Tables and figures for the v0.6 paper (EmemoVeri-2).
Usage: python paper/make_v06.py results/real_ext <outdir>   (reads summary_v06.json, summary_v06b.json, ev2_dev.json, summary_ext.json)"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root, out = sys.argv[1], sys.argv[2]
os.makedirs(f"{out}/tables", exist_ok=True); os.makedirs(f"{out}/figures", exist_ok=True)
R, B, E, X = (json.load(open(f"{root}/{n}.json")) for n in ("summary_v06", "summary_v06b", "ev2_dev", "summary_ext"))
rng = np.random.default_rng(7)
P = lambda x: f"{x:.3f}"
M = lambda x: ("$-$" if x < -0.0005 else "") + f"{abs(x):.3f}" if abs(x) >= 0.0005 else "0.000"
CI = lambda t: f"{M(t[0])} [{M(t[1])}, {M(t[2])}]"
PV = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
NAMES = {"Snapshot RAG": "Snapshot RAG", "EmemoVeri-D": "\\sys{} blueprint", "EmemoVeri-T": "\\sys{} original, tuned", "Sim+State+Rec+Pop": "Sim+State+Rec+Pop",
         "Similar-history (v0.5)": "Similar-history + priors", "All channels (linear)": "All channels, linear", "Learned ranker (logistic)": "Learned ranker, logistic",
         "Learned ranker (GBT)": "Learned ranker, GBT", "EmemoVeri-2 plain": "\\sys{}-2 without evidence weights", "EmemoVeri-2": "\\sys{}-2"}
ORDER = list(NAMES)
F, H, ML, DT = R["fresh16"], R["heldout12"], R["multilang8"], R["dev_test"]["systems"]


def w(name, s):
    open(f"{out}/tables/{name}.tex", "w").write(s)


# ---- Table: all systems, all sets ---------------------------------------------------------------------------------
rows = []
for k in ORDER:
    f = F["systems"][k]
    rows.append(f"{NAMES[k]} & {P(f['macro_hit1'])} & {P(f['macro_hit5'])} & {P(f['macro_rr'])} & {P(f['macro_ndcg5'])} & {P(f['macro_rec5'])} & {P(f['macro_all5'])} "
                f"& {P(DT[k]['hit5'])} & {P(H['systems'][k]['macro_hit5'])} & {P(ML['systems'][k]['macro_hit5'])} \\\\")
    if k in ("Snapshot RAG", "EmemoVeri-T", "Learned ranker (GBT)"):
        rows.append("\\midrule")
w("v06_systems", "\\begin{tabular}{lcccccc|ccc}\n\\toprule\n & \\multicolumn{6}{c|}{Fresh 16 repositories (confirmatory; macro-average)} & Dev-test & Held-out 12 & Multi-lang.\\ 8 \\\\\n"
  "System (all frozen) & Hit@1 & Hit@5 & MRR & NDCG@5 & Rec.@5 & All@5 & Hit@5 & Hit@5 & Hit@5 \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---- Table: confirmatory contrasts -----------------------------------------------------------------------------------------
LAB = {"H1 EV2 - Snapshot": "H1: \\sys{}-2 $-$ Snapshot RAG", "H2 EV2 - EmemoVeri-T": "H2: \\sys{}-2 $-$ original, tuned", "H3 EV2 - EV2 plain": "H3: \\sys{}-2 $-$ without evidence weights",
       "H4 EV2 - Sim+State+Rec+Pop": "H4: \\sys{}-2 $-$ Sim+State+Rec+Pop", "H5a EV2 - all channels (linear)": "H5a: \\sys{}-2 $-$ all channels, linear",
       "H5b EV2 - learned ranker (GBT)": "H5b: \\sys{}-2 $-$ learned ranker, GBT", "EV2 - similar-history (v0.5)": "\\sys{}-2 $-$ similar-history + priors",
       "EV2 - EmemoVeri-D": "\\sys{}-2 $-$ blueprint", "EV2 - learned ranker (logistic)": "\\sys{}-2 $-$ learned ranker, logistic"}
rows = []
for k, lab in LAB.items():
    c = F["contrasts"][k]; h = c["hit5"]
    py, ot = F["python10"]["contrasts"][k]["hit5"], F["other6"]["contrasts"][k]["hit5"]
    holm = PV(h["p_holm"]) if "p_holm" in h else "--"
    rows.append(f"{lab} & {CI(h['macro'])} & {h['wins']}/{h['losses']} & {PV(h['p_signflip'])} & {holm} & {h['dz']:.2f} & {M(c['rr']['macro'][0])} & {M(py['macro'][0])} & {M(ot['macro'][0])} \\\\")
    if k == "H5b EV2 - learned ranker (GBT)":
        rows.append("\\midrule")
w("v06_contrasts", "\\begin{tabular}{lccccc|ccc}\n\\toprule\nContrast (fresh 16) & $\\Delta$Hit@5 [95\\% CI, repo bootstrap] & W/L & $p$ & Holm $p$ & $d_z$ & $\\Delta$MRR & $\\Delta$Hit@5 Py-10 & $\\Delta$Hit@5 other-6 \\\\\n\\midrule\n"
  + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---- Table: post-hoc sets ------------------------------------------------------------------------------------------------------
rows = []
for k, lab in LAB.items():
    if k.startswith("H5") or k.startswith("EV2 - learned") or k.startswith("EV2 - Em"):
        continue
    a, b = H["contrasts"][k]["hit5"], ML["contrasts"][k]["hit5"]
    rows.append(f"{lab} & {CI(a['macro'])} & {a['wins']}/{a['losses']} & {CI(b['macro'])} & {b['wins']}/{b['losses']} \\\\")
w("v06_posthoc", "\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{Held-out Python (12)} & \\multicolumn{2}{c}{Multi-language (8)} \\\\\nContrast & $\\Delta$Hit@5 [95\\% CI] & W/L & $\\Delta$Hit@5 [95\\% CI] & W/L \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---- Table: EV2 ablation on dev-test ------------------------------------------------------------------------------------------
rows = [f"\\sys{{}}-2 (tuned on dev) & {P(E['test']['aware']['hit5'])} & {P(E['test']['aware']['rr'])} & -- \\\\"]
for name, v in E["ablation"].items():
    if name in ("- dilution weighting", "+ revert cancellation"):
        continue
    nm = name.replace("- all evidence-aware weighting", "$-$ all evidence-aware weights").replace("- strength weighting", "$-$ strength weighting").replace("+ ", "+ ")
    rows.append(f"{nm} & {P(v['hit5'])} & {P(v['mrr'])} & {CI(v['aware_minus'])} \\\\")
rows.append(f"No evidence weights, re-tuned & {P(E['test']['plain']['hit5'])} & {P(E['test']['plain']['rr'])} & {CI(E['test']['aware_minus_plain']['task'])} \\\\")
w("v06_ablation", "\\begin{tabular}{lccc}\n\\toprule\nVariant (dev-test, $n{=}768$) & Hit@5 & MRR & \\sys{}-2 $-$ variant, Hit@5 [95\\% CI] \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---- Table: out-of-sample component analysis of the frozen EmemoVeri-2 ------------------------------------------------------------
AB = json.load(open(f"{root}/ablate_ev2.json"))
ROWS = [("- evidence-item term", "no evidence-item term (claim similarity + priors)"), ("- state filter", "no state filter (dead files rankable)"),
        ("- claim similarity", "no claim-similarity term"), ("- recency prior", "no recency prior"), ("- strength weighting", "no strength weighting"),
        ("- rename carry-over", "no rename carry-over"), ("- frequency prior", "no frequency prior"), ("- both priors", "neither prior")]
rows = [f"\\sys{{}}-2 (full, frozen) & {P(AB['fresh16']['variants']['EmemoVeri-2 (full)']['macro_hit5'])} & -- & -- & -- & -- & -- \\\\", "\\midrule"]
for k, lab in ROWS:
    f = AB["fresh16"]["variants"][k]; fm = f["full_minus"]
    def other(tag):
        v = AB.get(tag, {}).get("variants", {}).get(k)
        return CI(v["full_minus"]["macro"]) if v else "--"
    rows.append(f"{lab} & {P(f['macro_hit5'])} & {CI(fm['macro'])} & {fm['wins']}/{fm['losses']} & {PV(fm['p_holm'])} & {other('heldout12')} & {other('multilang8')} \\\\")
w("v06_component", "\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{4}{c}{Fresh 16 (confirmatory set; post-hoc analysis)} & Held-out 12 & Multi-lang.\\ 8 \\\\\nVariant (one ingredient removed) & Hit@5 & $\\Delta$Hit@5 [95\\% CI] & W/L & Holm $p$ & $\\Delta$Hit@5 [95\\% CI] & $\\Delta$Hit@5 [95\\% CI] \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---- narrow variants for single-column formats (LNCS) -----------------------------------------------------------------------------------
SH = {"Snapshot RAG": "Snapshot RAG", "EmemoVeri-D": "\\sys{}-1, blueprint", "EmemoVeri-T": "\\sys{}-1, tuned", "Sim+State+Rec+Pop": "Sim+State+Rec+Pop",
      "Similar-history (v0.5)": "Similar-history+priors", "All channels (linear)": "All channels, linear", "Learned ranker (logistic)": "Ranker, logistic",
      "Learned ranker (GBT)": "Ranker, GBT", "EmemoVeri-2 plain": "\\sys{}-2, no evid.\\ weights", "EmemoVeri-2": "\\sys{}-2"}
rows = []
for k in ORDER:
    f = F["systems"][k]
    rows.append(f"{SH[k]} & {P(f['macro_hit1'])} & {P(f['macro_hit5'])} & {P(f['macro_rr'])} & {P(f['macro_ndcg5'])} & {P(f['macro_all5'])} & {P(DT[k]['hit5'])} & {P(H['systems'][k]['macro_hit5'])} & {P(ML['systems'][k]['macro_hit5'])} \\\\")
    if k in ("Snapshot RAG", "EmemoVeri-T", "Learned ranker (GBT)"):
        rows.append("\\midrule")
w("v06_systems_n", "\\begin{tabular}{l@{\\hspace{4pt}}ccccc|ccc}\n\\toprule\n & \\multicolumn{5}{c|}{Fresh 16 (macro-average)} & \\multicolumn{3}{c}{Hit@5 on earlier sets} \\\\\nSystem (frozen) & Hit@1 & Hit@5 & MRR & NDCG & All@5 & Dev-test & Held-out & Multi. \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
rows = []
for k, lab in LAB.items():
    c = F["contrasts"][k]; h = c["hit5"]
    holm = PV(h["p_holm"]) if "p_holm" in h else "--"
    rows.append(f"{lab} & {CI(h['macro'])} & {h['wins']}/{h['losses']} & {holm} & {h['dz']:.2f} & {M(c['rr']['macro'][0])} \\\\")
    if k == "H5b EV2 - learned ranker (GBT)":
        rows.append("\\midrule")
w("v06_contrasts_n", "\\begin{tabular}{l@{\\hspace{4pt}}cccrc}\n\\toprule\nContrast (fresh 16) & $\\Delta$Hit@5 [95\\% CI] & W/L & Holm $p$ & $d_z$ & $\\Delta$MRR \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
rows = [f"\\sys{{}}-2 (full, frozen) & {P(AB['fresh16']['variants']['EmemoVeri-2 (full)']['macro_hit5'])} & -- & -- & -- & -- & -- \\\\", "\\midrule"]
for k, lab in ROWS:
    f = AB["fresh16"]["variants"][k]; fm = f["full_minus"]
    def oth(tag):
        v = AB.get(tag, {}).get("variants", {}).get(k)
        return M(v["full_minus"]["macro"][0]) if v else "--"
    rows.append(f"{lab} & {P(f['macro_hit5'])} & {CI(fm['macro'])} & {fm['wins']}/{fm['losses']} & {PV(fm['p_holm'])} & {oth('heldout12')} & {oth('multilang8')} \\\\")
w("v06_component_n", "\\begin{tabular}{l@{\\hspace{3pt}}ccccrr}\n\\toprule\n & \\multicolumn{4}{c}{Fresh 16} & \\multicolumn{2}{c}{$\\Delta$Hit@5, other sets} \\\\\nVariant (ingredient removed) & Hit@5 & $\\Delta$Hit@5 [95\\% CI] & W/L & Holm $p$ & Held-out & Multi. \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---- figures (vector PDF + 300-dpi PNG; sized for a 4.4-5.2 inch column, fonts >= 7 pt after scaling) ---------------------------------------------
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "pdf.fonttype": 42})
HD = f"{out}/figures_hd"; os.makedirs(HD, exist_ok=True)


def save(fig, name):
    fig.savefig(f"{out}/figures/{name}.pdf"); fig.savefig(f"{HD}/{name}.png", dpi=300); plt.close(fig)


fig = plt.figure(figsize=(5.2, 5.0))
ax0 = fig.add_axes([0.0, 0.865, 1.0, 0.13]); ax0.axis("off"); ax0.set_xlim(-1, 100.5); ax0.set_ylim(0, 30)
for t, x in [("Real git\nhistory", 0), ("Evidence\nledger\nper file", 20.5), ("Score:\nsimilarity,\nbest item,\npriors", 41), ("Future\ncommit\nsubject", 61.5), ("Top-5 files\nvs. git\nanswer", 82)]:
    ax0.add_patch(plt.Rectangle((x, 1), 17.8, 28, fc="#eef3fa", ec="#34495e", lw=0.9)); ax0.text(x + 8.9, 15, t, ha="center", va="center", fontsize=7.2)
for x in (17.9, 38.4, 58.9, 79.4):
    ax0.annotate("", xy=(x + 2.9, 15), xytext=(x + 0.1, 15), arrowprops=dict(arrowstyle="->", lw=1.0))
reps = F["repos"]; base = np.array([F["per_repo_hit5"]["Snapshot RAG"][r] for r in reps])
SYS = [("EmemoVeri-D", "original design, blueprint", "#e0a030"), ("EmemoVeri-T", "original design, tuned", "#c0572a"), ("Sim+State+Rec+Pop", "Sim+State+Rec+Pop", "#2e7d32"),
       ("Learned ranker (logistic)", "learned ranker, logistic", "#6a3d9a"), ("Learned ranker (GBT)", "learned ranker, GBT", "#8e5bb5"), ("EmemoVeri-2", "EmemoVeri-2", "#1f4e9c")]
ax = fig.add_axes([0.36, 0.605, 0.60, 0.20])
for yi, (k, lab, col) in enumerate(SYS[::-1]):
    d = np.array([F["per_repo_hit5"][k][r] for r in reps]) - base
    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    ax.errorbar(d.mean(), yi, xerr=[[d.mean() - np.percentile(bs, 2.5)], [np.percentile(bs, 97.5) - d.mean()]], fmt="o", color=col, ms=5.5, capsize=3, lw=1.3)
ax.axvline(0, color="k", lw=0.8); ax.set_yticks(range(len(SYS))); ax.set_yticklabels([x[1] for x in SYS[::-1]], fontsize=7.8)
ax.set_xlabel("Hit@5 gain over snapshot RAG\n(95% CI over the 16 fresh repositories)", fontsize=7.6); ax.set_xlim(-0.01, 0.17); ax.tick_params(labelsize=7.8)
order = sorted(reps, key=lambda r: F["per_repo_hit5"]["Snapshot RAG"][r]); y = np.arange(len(order))
ax2 = fig.add_axes([0.36, 0.075, 0.60, 0.40])
for k, col, mk, lab in (("Snapshot RAG", "#4a7ebb", "o", "snapshot RAG"), ("EmemoVeri-T", "#c0572a", "D", "original, tuned"), ("EmemoVeri-2", "#1f4e9c", "s", "EmemoVeri-2")):
    ax2.scatter([F["per_repo_hit5"][k][r] for r in order], y, c=col, marker=mk, s=22, label=lab, zorder=3)
ax2.hlines(y, [F["per_repo_hit5"]["Snapshot RAG"][r] for r in order], [F["per_repo_hit5"]["EmemoVeri-2"][r] for r in order], color="#cccccc", lw=1.0, zorder=1)
ax2.set_yticks(y); ax2.set_yticklabels([f"{r} ({F['n_tasks'][r]})" for r in order], fontsize=7.4); ax2.set_xlabel("Hit@5 per repository (number of tasks)", fontsize=7.6); ax2.tick_params(axis="x", labelsize=7.8)
ax2.legend(fontsize=7.4, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0))
fig.text(0.005, 0.82, "(a)", fontsize=9, fontweight="bold"); fig.text(0.005, 0.50, "(b)", fontsize=9, fontweight="bold")
save(fig, "fig_overview")

fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.5))
ax = axs[0]; lv = ["0.0", "0.3", "0.5", "0.7"]; xs = [float(l) for l in lv]
for key, lab, col, ls in (("Snapshot RAG", "snapshot RAG", "#4a7ebb", "-"), ("EmemoVeri-T", "original, tuned", "#c0572a", "-"), ("Sim+State+Rec+Pop", "Sim+State+Rec+Pop", "#2e7d32", "--"),
                          ("EmemoVeri-2 plain", "EmemoVeri-2, no evid. weights", "#8e8e8e", "--"), ("EmemoVeri-2", "EmemoVeri-2", "#1f4e9c", "-")):
    ax.plot(xs, [B["degrade"][l][key]["hit5"] for l in lv], ls, color=col, marker="o", ms=4, lw=1.6, label=lab)
ax.set_xlabel("Fraction of query words dropped", fontsize=8); ax.set_ylabel("Hit@5", fontsize=8); ax.set_xticks(xs); ax.tick_params(labelsize=8)
ax.set_ylim(0.26, 0.83); ax.legend(fontsize=6.6, frameon=False, loc="lower left"); ax.set_title("(a) Degraded queries (dev-test)", fontsize=8.4)
ax = axs[1]; hls = [7, 30, 90, 180, 365, 730, 3650]
for key, lab, col in (("blueprint", "blueprint weights", "#e0a030"), ("tuned", "tuned weights", "#c0572a")):
    rows_ = X["stale_sweep"][key]; basev = [r for r in rows_ if r["half_life"] is None][0]["test"]["hit5"]
    ax.plot(hls, [[r for r in rows_ if r["half_life"] == h][0]["test"]["hit5"] for h in hls], marker="o", ms=4, lw=1.6, color=col, label=lab); ax.axhline(basev, color=col, ls=":", lw=1.2)
ax.set_xscale("log"); ax.set_xlabel("Staleness half-life (days)", fontsize=8); ax.set_ylabel("Hit@5", fontsize=8); ax.tick_params(labelsize=8)
ax.legend(fontsize=7, frameon=False, loc="lower right"); ax.set_title("(b) Original design: staleness", fontsize=8.4)
fig.tight_layout(pad=0.5, w_pad=1.2); save(fig, "fig_sweeps2")
print("ok")
