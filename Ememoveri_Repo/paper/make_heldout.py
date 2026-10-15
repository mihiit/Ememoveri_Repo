"""Tables/figure for the v0.4 held-out confirmation. Usage: python paper/make_heldout.py results/real_ext/summary_heldout.json <outdir>"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open(sys.argv[1])); out = sys.argv[2]
os.makedirs(f"{out}/tables", exist_ok=True); os.makedirs(f"{out}/figures", exist_ok=True)
M = lambda x: ("$-$" if x < -0.0005 else "") + f"{abs(x):.3f}" if abs(x) >= 0.0005 else "0.000"
CI = lambda t: f"{M(t[0])} [{M(t[1])}, {M(t[2])}]"
P = lambda x: f"{x:.3f}"
S = R["systems"]

rows = []
for nm, lab in (("Recency", "Recency prior"), ("Popularity", "Popularity prior"), ("Naive memory", "Naive memory"), ("Snapshot RAG", "Snapshot RAG"),
                ("EmemoVeri-D", "\\sys{}-D (blueprint)"), ("EmemoVeri-T", "\\sys{}-T (tuned)"),
                ("Sim+State+Recency+Popularity", "Sim+State+Rec+Pop")):
    v = S[nm]
    rows.append(f"{lab} & {P(v['macro_hit1'])} & {P(v['macro_hit5'])} & {P(v['macro_rr'])} & {P(v['macro_ndcg5'])} & {P(v['dead5'])} \\\\")
open(f"{out}/tables/heldout_sys.tex", "w").write("\\begin{tabular}{lccccc}\n\\toprule\nSystem (frozen) & Hit@1 & Hit@5 & MRR & NDCG@5 & Dead@5 \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

C = R["contrasts"]
spec = [("H1: \\sys{}-T $-$ Snapshot", "H1 EmemoVeri-T - Snapshot"), ("H2: \\sys{}-T $-$ Sim+State+Rec+Pop", "H2 EmemoVeri-T - Sim+State+Rec+Pop"),
        ("H3: \\sys{}-D $-$ Snapshot", "H3 EmemoVeri-D - Snapshot"), ("Ladder: + state filter", "ladder R1 + state filter"),
        ("Ladder: + frequency prior", "ladder R2 + frequency prior"), ("Ladder: + evidence", "ladder R3 + evidence"),
        ("Ladder: + trust", "ladder R4 + trust"), ("Ladder: + recency", "ladder R5 + recency")]
rows = []
for lab, k in spec:
    c = C[k]; h = c["hit5"]
    rows.append(f"{lab} & {CI(h['macro'])} & {h['wins']}/{h['losses']} & {CI(c['rr']['macro'])} & {CI(c['pooled_task'][:3]).split(' ',1)[1]} \\\\")
    if lab.startswith("H3"):
        rows.append("\\midrule")
open(f"{out}/tables/heldout_contrasts.tex", "w").write("\\begin{tabular}{lcccc}\n\\toprule\nContrast & $\\Delta$Hit@5 [95\\% CI] & W/L & $\\Delta$MRR [95\\% CI] & Task-boot.\\ CI \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
repos = sorted(R["repos"], key=lambda r: R["per_repo"]["Snapshot RAG"]["hit5"][r])
y = np.arange(len(repos))
fig, ax = plt.subplots(figsize=(4.9, 2.6))
for nm, col, mk, lab in (("Snapshot RAG", "#4a7ebb", "o", "Snapshot RAG"), ("Sim+State+Recency+Popularity", "#2e7d32", "s", "Sim+State+Rec+Pop"),
                         ("EmemoVeri-T", "#c0572a", "D", "EmemoVeri-T"), ("EmemoVeri-D", "#e0a030", "^", "EmemoVeri-D"), ("Popularity", "#999999", ".", "Popularity")):
    ax.scatter([R["per_repo"][nm]["hit5"][r] for r in repos], y, c=col, marker=mk, s=16, label=lab, zorder=3)
ax.hlines(y, [R["per_repo"]["Snapshot RAG"]["hit5"][r] for r in repos], [R["per_repo"]["EmemoVeri-T"]["hit5"][r] for r in repos], color="#cccccc", lw=0.8, zorder=1)
ax.set_yticks(y); ax.set_yticklabels([f"{r} ({R['n_tasks'][r]})" for r in repos], fontsize=6)
ax.set_xlabel("Hit@5 (frozen systems, held-out repositories)"); ax.legend(fontsize=5.5, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.45, 1.0), columnspacing=0.8, handletextpad=0.2)
fig.tight_layout(pad=0.3); fig.savefig(f"{out}/figures/fig_heldout.pdf"); plt.close(fig)
print("ok")
