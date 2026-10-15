"""Tables and the overview figure for the v0.5 analyses.
Usage: python paper/make_v05.py results/real_ext/summary_v05.json <outdir>   (writes outdir/tables/v05_*.tex, outdir/figures/fig_overview.pdf)"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open(sys.argv[1])); out = sys.argv[2]
os.makedirs(f"{out}/tables", exist_ok=True); os.makedirs(f"{out}/figures", exist_ok=True)
rng = np.random.default_rng(3)
P = lambda x: f"{x:.3f}"
M = lambda x: ("$-$" if x < -0.0005 else "") + f"{abs(x):.3f}" if abs(x) >= 0.0005 else "0.000"
PV = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
ORDER = [("Snapshot RAG", "Snapshot RAG"), ("EmemoVeri-D", "\\sys{}-D (blueprint)"), ("EmemoVeri-T", "\\sys{}-T (tuned)"),
         ("Sim+State+Rec+Pop", "Sim+State+Rec+Pop"), ("B2 co-change + sim", "Co-change + sim"),
         ("B1 similar-history (alone)", "Similar-history (alone)"), ("B1 similar-history + sim + priors", "\\sys{}-E (evidence-level)"),
         ("All channels + priors (linear)", "All channels, linear"), ("Learned ranker (logistic)", "Learned ranker, logistic"),
         ("Learned ranker (GBT)", "Learned ranker, GBT")]
DT, H, ML = R["dev_test"]["systems"], R["heldout12"], R["multilang8"]

rows = []
for k, lab in ORDER:
    rows.append(f"{lab} & {P(DT[k]['hit5'])} & {P(DT[k]['rr'])} & {P(H['systems'][k]['macro_hit5'])} & {P(H['systems'][k]['macro_rr'])} "
                f"& {P(ML['systems'][k]['macro_hit5'])} & {P(ML['systems'][k]['macro_rr'])} \\\\")
    if k in ("EmemoVeri-T", "B2 co-change + sim"):
        rows.append("\\midrule")
open(f"{out}/tables/v05_systems.tex", "w").write(
    "\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{2}{c}{Dev-test (3 repos, $n{=}768$)} & \\multicolumn{2}{c}{Held-out Python (12)} & \\multicolumn{2}{c}{Multi-language (8)} \\\\\n"
    "System (frozen) & Hit@5 & MRR & Hit@5 & MRR & Hit@5 & MRR \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

short = {"H1 EmemoVeri-T - Snapshot RAG": "H1: \\sys{}-T $-$ Snapshot", "H2 EmemoVeri-T - Sim+State+Rec+Pop": "H2: \\sys{}-T $-$ Sim+State+Rec+Pop",
         "H3 EmemoVeri-D - Snapshot RAG": "H3: \\sys{}-D $-$ Snapshot",
         "H4 EmemoVeri-T - similar-history (tuned)": "H4: \\sys{}-T $-$ \\sys{}-E", "H4 EmemoVeri-T - co-change + sim (tuned)": "H4: \\sys{}-T $-$ co-change+sim",
         "H4 EmemoVeri-T - all channels (linear)": "H4: \\sys{}-T $-$ all channels", "H4 EmemoVeri-T - learned ranker (logistic)": "H4: \\sys{}-T $-$ logistic ranker",
         "H4 EmemoVeri-T - learned ranker (GBT)": "H4: \\sys{}-T $-$ GBT ranker"}
rows = []
for tag, name in (("heldout12", "Held-out Python (12 repositories)"), ("multilang8", "Multi-language (8 repositories)")):
    rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{name}}}}} \\\\")
    for lab, v in R[tag]["contrasts"].items():
        h = v["hit5"]; m = h["macro"]
        rows.append(f"{short[lab]} & {M(m[0])} [{M(m[1])}, {M(m[2])}] & {h['wins']}/{h['losses']} & {PV(h['p_signflip'])} & {PV(h['p_holm'])} & {h['dz']:.2f} \\\\")
    if tag == "heldout12":
        rows.append("\\midrule")
open(f"{out}/tables/v05_contrasts.tex", "w").write(
    "\\begin{tabular}{lccccc}\n\\toprule\nHit@5 contrast & $\\Delta$ [95\\% CI, repository bootstrap] & W/L & $p$ (sign-flip) & Holm $p$ & $d_z$ \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

LANG = {"express": "JS", "axios": "JS", "gin": "Go", "cobra": "Go", "ripgrep": "Rust", "serde": "Rust", "jackson-databind": "Java", "commons-lang": "Java"}
rows = []
for r in sorted(ML["repos"], key=lambda r: (LANG[r], r)):
    s, t, e = (ML["per_repo_hit5"][k][r] for k in ("Snapshot RAG", "EmemoVeri-T", "B1 similar-history + sim + priors"))
    rows.append(f"{r} ({LANG[r]}, {ML['n_tasks'][r]}) & {P(s)} & {P(t)} & {P(e)} & {M(t - e)} \\\\")
open(f"{out}/tables/v05_ml_repos.tex", "w").write(
    "\\begin{tabular}{lcccc}\n\\toprule\nRepository (language, tasks) & Snapshot & \\sys{}-T & \\sys{}-E & \\sys{}-T $-$ \\sys{}-E \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---- overview figure: pipeline + gains over snapshot RAG ----------------------------------
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(7.1, 3.1))
ax0 = fig.add_axes([0.0, 0.70, 1.0, 0.28]); ax0.axis("off"); ax0.set_xlim(0, 100); ax0.set_ylim(0, 30)
boxes = [("Real git history,\none chronological clock", 0), ("Claims, evidence and state\nfrom git events", 20.5), ("Ranking: similarity, state,\nhistory priors, evidence", 41),
         ("Future task:\ncommit subject as query", 61.5), ("Top-5 files vs.\ngit reference set", 82)]
for t, x in boxes:
    ax0.add_patch(plt.Rectangle((x, 5), 17.8, 20, fc="#eef3fa", ec="#34495e", lw=0.8)); ax0.text(x + 8.9, 15, t, ha="center", va="center", fontsize=6)
for x in (17.9, 38.4, 58.9, 79.4):
    ax0.annotate("", xy=(x + 2.9, 15), xytext=(x + 0.1, 15), arrowprops=dict(arrowstyle="->", lw=0.8))
SYS = [("EmemoVeri-D", "EmemoVeri-D (blueprint)", "#e0a030"), ("EmemoVeri-T", "EmemoVeri-T (tuned)", "#c0572a"), ("Sim+State+Rec+Pop", "Sim+State+Rec+Pop", "#2e7d32"),
       ("B2 co-change + sim", "Co-change + sim", "#999999"), ("B1 similar-history + sim + priors", "EmemoVeri-E (evidence-level)", "#1f4e9c"),
       ("Learned ranker (GBT)", "Learned ranker (GBT)", "#6a3d9a")]
for pi, (tag, ttl) in enumerate((("heldout12", "12 held-out Python repositories"), ("multilang8", "8 JS / Go / Rust / Java repositories"))):
    ax = fig.add_axes([0.20 + pi * 0.40, 0.14, 0.34, 0.50])
    reps = R[tag]["repos"]; base = np.array([R[tag]["per_repo_hit5"]["Snapshot RAG"][r] for r in reps])
    for yi, (k, lab, col) in enumerate(SYS[::-1]):
        d = np.array([R[tag]["per_repo_hit5"][k][r] for r in reps]) - base
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        ax.errorbar(d.mean(), yi, xerr=[[d.mean() - np.percentile(bs, 2.5)], [np.percentile(bs, 97.5) - d.mean()]], fmt="o", color=col, ms=4, capsize=2, lw=1)
    ax.axvline(0, color="k", lw=0.6); ax.set_yticks(range(len(SYS)))
    ax.set_yticklabels([s[1] for s in SYS[::-1]] if pi == 0 else [], fontsize=6.2)
    ax.set_title(ttl, fontsize=6.6); ax.set_xlim(-0.02, 0.19); ax.set_xlabel("Hit@5 gain over snapshot RAG (95% CI over repositories)", fontsize=5.8)
fig.savefig(f"{out}/figures/fig_overview.pdf"); plt.close(fig)
print("ok")
