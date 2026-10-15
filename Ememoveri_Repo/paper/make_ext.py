"""Generate tables and figures for the v0.3 additions from results/real_ext/summary_ext.json.
Usage: python paper/make_ext.py results/real_ext/summary_ext.json <outdir>   (writes outdir/tables/*.tex, outdir/figures/*.pdf)"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open(sys.argv[1]))
out = sys.argv[2]
os.makedirs(f"{out}/tables", exist_ok=True)
os.makedirs(f"{out}/figures", exist_ok=True)
M = lambda x: ("$-$" if x < -0.0005 else "") + f"{abs(x):.3f}" if abs(x) >= 0.0005 else "0.000"
CI = lambda t: f"{M(t[0])} [{M(t[1])}, {M(t[2])}]"
P = lambda x: f"{x:.3f}"


def write(name, s):
    open(f"{out}/tables/{name}.tex", "w").write(s)


# ---- ladder ------------------------------------------------------------------------
rows = []
steps = {s["to"]: s for s in R["ladder_steps"]}
for r in R["ladder"]:
    d = CI(steps[r["name"]]["d_hit5"]) if r["name"] in steps else "--"
    rows.append(f"{r['name']} & {P(r['hit1'])} & {P(r['hit3'])} & {P(r['hit5'])} & {P(r['rr'])} & {P(r['ndcg5'])} & {P(r['dead5'])} & {d} \\\\")
ev = R["published_test"]["EV_TUNED"]
rows.append("\\midrule")
rows.append(f"\\sys{{}} dev-tuned (reference) & {P(ev['hit1'])} & {P(ev['hit3'])} & {P(ev['hit5'])} & {P(ev['rr'])} & {P(ev['ndcg5'])} & {P(ev['dead5'])} & -- \\\\")
write("ladder", "\\begin{tabular}{lcccccc l}\n\\toprule\nRung & Hit@1 & Hit@3 & Hit@5 & MRR & NDCG@5 & Dead@5 & $\\Delta$Hit@5 [95\\% CI] \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---- strong baselines ---------------------------------------------------------------
S = {r["name"]: r for r in R["strong"]}
pub = R["published_test"]
LV = R["ladder_full_vs_strong"]
CB = R["cluster_boot"]
order = [("Snapshot RAG", pub["SNAPSHOT"], CB["EV_TUNED-SNAPSHOT:hit5"]),
         ("Sim+Rec", S["Sim+Recency"], LV["EV_TUNED-Sim+Recency"]),
         ("Sim+Pop", S["Sim+Popularity"], LV["EV_TUNED-Sim+Popularity"]),
         ("Sim+Rec+Pop", S["Sim+Recency+Popularity"], LV["EV_TUNED-Sim+Recency+Popularity"]),
         ("Sim+State+Rec", S["Sim+State+Recency"], LV["EV_TUNED-Sim+State+Recency"]),
         ("Sim+State+Pop", S["Sim+State+Popularity"], LV["EV_TUNED-Sim+State+Popularity"]),
         ("Sim+State+Rec+Pop", S["Sim+State+Recency+Popularity"], LV["EV_TUNED-Sim+State+Recency+Popularity"]),
         ("\\sys{}-D (blueprint)", pub["EV_DEFAULT"], CB["EV_TUNED-EV_DEFAULT:hit5"])]
rows = []
for nm, r, c in order:
    rows.append(f"{nm} & {P(r['hit5'])} & {P(r['rr'])} & {P(r['ndcg5'])} & {CI(c['task'][:3])} & {CI(c['file'][:3])} \\\\")
rows.append("\\midrule")
rows.append(f"\\sys{{}}-T (dev-tuned) & {P(ev['hit5'])} & {P(ev['rr'])} & {P(ev['ndcg5'])} & -- & -- \\\\")
write("strong", "\\begin{tabular}{lcccll}\n\\toprule\nSystem & Hit@5 & MRR & NDCG@5 & \\multicolumn{2}{c}{\\sys{}-T minus system, Hit@5 [95\\% CI]} \\\\\n"
      " & & & & task bootstrap & file-cluster bootstrap \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---- paired comparisons under three resampling schemes ---------------------------------
pv = {"EV_DEFAULT-SNAPSHOT": "0.112", "EV_TUNED-SNAPSHOT": "$<$0.0001", "EV_TUNED-EV_DEFAULT": "0.0003"}
lab = {"EV_DEFAULT-SNAPSHOT": "\\sys{}-D $-$ Snapshot RAG", "EV_TUNED-SNAPSHOT": "\\sys{}-T $-$ Snapshot RAG",
       "EV_TUNED-EV_DEFAULT": "\\sys{}-T $-$ \\sys{}-D"}
rows = []
for k in lab:
    c = CB[f"{k}:hit5"]
    rows.append(f"{lab[k]} & {CI(c['task'][:3])} & {CI(c['month'][:3])} & {CI(c['file'][:3])} & {pv[k]} \\\\")
c = LV["Sim+State+Recency+Popularity-SNAPSHOT"]
rows.append(f"Sim+State+Rec+Pop $-$ Snapshot RAG & {CI(c['task'][:3])} & {CI(c['month'][:3])} & {CI(c['file'][:3])} & -- \\\\")
write("cmp", "\\begin{tabular}{lcccc}\n\\toprule\nHit@5 difference & task & repo$\\times$month blocks & file clusters & Holm $p$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---- backends ---------------------------------------------------------------------------
nm = {"tfidf": "Hashed TF-IDF (main)", "bm25": "BM25", "dense": "Static word vectors"}
rows = []
for be, v in R["backends"].items():
    rows.append(f"{nm[be]} & {P(v['snapshot']['hit5'])} & {P(v['full']['hit5'])} & {CI(v['full-snapshot']['task'])} & {P(v['recpop']['hit5'])} & {CI(v['recpop-snapshot']['task'])} & {CI(v['full-recpop']['task'])} \\\\")
write("backends", "\\begin{tabular}{lcclclc}\n\\toprule\nBackend & Snap. & Full & $\\Delta$ vs.\\ Snap. & SSRP & $\\Delta$ vs.\\ Snap. & Full$-$SSRP \\\\\n\\midrule\n" + "\n".join(rows).replace("\\\\\n", "\\\\\n") + "\n\\bottomrule\n\\end{tabular}")

# ---- LORO -------------------------------------------------------------------------------
rows = []
for h, v in R["loro"].items():
    rows.append(f"{h} ({v['n']}) & {P(v['snapshot']['hit5'])} & {P(v['default']['hit5'])} & {P(v['loro_tuned']['hit5'])} & {P(v['loro_recpop']['hit5'])} & {CI(v['tuned-snapshot']['task'])} & {CI(v['tuned-recpop']['task'])} \\\\")
write("loro", "\\begin{tabular}{lccccll}\n\\toprule\nHeld-out ($n$) & Snap. & \\sys{}-D & \\sys{}-LORO & RP-LORO & LORO$-$Snap. & LORO$-$RP \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---- lifecycle ----------------------------------------------------------------------------
L = R["lifecycle"]
d, rn = L["delete"], L["rename"]
pc = lambda x: f"{100 * x:.1f}\\%"
rows = [f"Deleted file in top-5 before / after deletion ($n={d['n']}$) & {pc(d['pre_hit_naive'])} / {pc(d['post_old_in_top5_naive'])} & {pc(d['pre_hit_snapshot'])} / {pc(d['post_old_in_top5_snapshot'])} & {pc(d['pre_hit_tuned'])} / {pc(d['post_old_in_top5_tuned'])} & -- \\\\",
        f"Old path in top-5 after rename ($n={rn['n']}$) & {pc(rn['post_old_in_top5_naive'])} & {pc(rn['post_old_in_top5_snapshot'])} & {pc(rn['post_old_in_top5_tuned'])} & -- \\\\",
        f"New path in top-5 after rename & {pc(rn['post_new_in_top5_carry:naive'])} & {pc(rn['post_new_in_top5_carry:snapshot'])} & {pc(rn['post_new_in_top5_carry:tuned'])} & {pc(rn['post_new_in_top5_nocarry:snapshot'])} \\\\",
        f"\\quad given old path was in top-5 before ($n={rn['n_pre_hit_snapshot']}$) & -- & {pc(rn['continuity_given_pre_hit_carry'])} & -- & {pc(rn['continuity_given_pre_hit_nocarry'])} \\\\"]
write("life", "\\begin{tabular}{lcccc}\n\\toprule\n & Naive memory & Snapshot RAG & \\sys{}-T & Snapshot, no rename carry-over \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---------------------------------- figures ------------------------------------------------
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
# overview: pipeline + ingredient effects
fig = plt.figure(figsize=(4.9, 3.5))
ax0 = fig.add_axes([0.0, 0.62, 1.0, 0.36]); ax0.axis("off"); ax0.set_xlim(0, 100); ax0.set_ylim(0, 30)
boxes = [("Real git\nhistory,\none clock", 0), ("Claims, evidence\nand state from\ngit events", 20.5),
         ("Ranking:\nsimilarity, state,\nhistory priors", 41), ("Future task:\ncommit subject\nas query", 61.5), ("Top-5 files vs.\ngit reference\nset", 82)]
for t, x in boxes:
    ax0.add_patch(plt.Rectangle((x, 6), 17.8, 18, fc="#eef3fa", ec="#34495e", lw=0.8))
    ax0.text(x + 8.9, 15, t, ha="center", va="center", fontsize=5.6)
for x in (17.9, 38.4, 58.9, 79.4):
    ax0.annotate("", xy=(x + 2.9, 15), xytext=(x + 0.1, 15), arrowprops=dict(arrowstyle="->", lw=0.8))
ax0.text(50, 1.2, "task is scored before its own commit is ingested (leak canary + shuffled-query control)", ha="center", fontsize=5.6, style="italic")
ax = fig.add_axes([0.33, 0.13, 0.64, 0.40])
S_ = R["ladder_steps"]
labels = ["+ state\n(snapshot)", "+ frequency\nprior", "+ evidence", "+ trust", "+ staleness /\nrecency"]
vals = [s["d_hit5"][0] for s in S_]; lo = [s["d_hit5"][1] for s in S_]; hi = [s["d_hit5"][2] for s in S_]
labels.append("EmemoVeri-T vs.\nSim+State+Rec+Pop")
c = LV["EV_TUNED-Sim+State+Recency+Popularity"]["task"]
vals.append(c[0]); lo.append(c[1]); hi.append(c[2])
y = np.arange(len(vals))[::-1]
cols = ["#4a7ebb"] * 2 + ["#9aa9bb"] * 3 + ["#c0392b"]
ax.barh(y, vals, color=cols, height=0.62)
ax.errorbar(vals, y, xerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="none", ecolor="k", lw=0.8, capsize=2)
ax.axvline(0, color="k", lw=0.6)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6)
ax.set_xlabel("Change in Hit@5 (95% CI)")
ax.set_title("What each added ingredient buys (cumulative ladder; last bar: tuned EmemoVeri vs. best prior baseline)", fontsize=5.8, loc="right", x=1.0)
fig.savefig(f"{out}/figures/fig_overview.pdf"); plt.close(fig)

# sweeps: degradation + staleness
fig, axs = plt.subplots(1, 2, figsize=(4.9, 1.95))
ax = axs[0]
lv = ["0.0", "0.3", "0.5", "0.7"]; xs = [float(l) for l in lv]
for key, nm_, col, ls in (("snapshot", "Snapshot RAG", "#4a7ebb", "-"), ("default", "EmemoVeri-D", "#e0a030", "-"),
                          ("tuned", "EmemoVeri-T", "#c0572a", "-"), ("recpop", "Sim+State+Rec+Pop", "#2e7d32", "--")):
    ax.plot(xs, [R["degrade"][l][key]["hit5"] for l in lv], ls, color=col, marker="o", ms=3, lw=1.2, label=nm_)
ax.set_xlabel("Fraction of query words dropped"); ax.set_ylabel("Hit@5"); ax.set_xticks(xs)
ax.legend(fontsize=5.2, frameon=False, loc="lower left")
ax.set_title("(a) Degraded queries", fontsize=7)
ax = axs[1]
hls = [7, 30, 90, 180, 365, 730, 3650]
for key, nm_, col in (("blueprint", "blueprint weights ($w_s$=0.10)", "#e0a030"), ("tuned", "tuned weights ($w_s$=0.05)", "#c0572a")):
    rows_ = R["stale_sweep"][key]
    base = [r for r in rows_ if r["half_life"] is None][0]["test"]["hit5"]
    ax.plot(hls, [[r for r in rows_ if r["half_life"] == h][0]["test"]["hit5"] for h in hls], marker="o", ms=3, lw=1.2, color=col, label=nm_)
    ax.axhline(base, color=col, ls=":", lw=0.9)
ax.set_xscale("log"); ax.set_xlabel("Staleness half-life (days)"); ax.set_ylabel("Hit@5")
ax.legend(fontsize=5.2, frameon=False, loc="lower right"); ax.set_title("(b) Staleness half-life sweep", fontsize=7)
fig.tight_layout(pad=0.4)
fig.savefig(f"{out}/figures/fig_sweeps.pdf"); plt.close(fig)
print("ok")
