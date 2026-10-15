"""Generate LaTeX tables + number macros from results/*/summary.json (no hand-typed numbers)."""
import json, os, shutil
R = os.path.join(os.path.dirname(__file__), "..", "results")
T = os.path.join(os.path.dirname(__file__), "tables"); os.makedirs(T, exist_ok=True)
S = json.load(open(f"{R}/real/summary.json"))
for f in ["fig_main", "fig_repo", "fig_ablation", "fig_selective", "fig_hazards"]:
    shutil.copy(f"{R}/real/{f}.pdf", os.path.join(os.path.dirname(__file__), "figures", f"{f}.pdf"))
t = S["test"]
N = {}
def mac(name, val): N[name] = val
f3 = lambda x: f"{x:.3f}"; f2 = lambda x: f"{x:.2f}"; sg = lambda x: f"{x:+.3f}".replace("-", "$-$")
LAB = {"RANDOM_LOCAL": "Random (local)", "RECENCY": "Recency", "POPULARITY": "Popularity", "PLAIN_RAG": "Plain RAG (all repos)",
       "NAIVE_MEMORY": "Naive memory (local)", "SNAPSHOT_RAG": "Snapshot RAG", "EMEMOVERI": "EmemoVeri (default weights)",
       "EMEMOVERI_TUNED": "EmemoVeri (dev-tuned)"}
# ---- main table
L = ["\\begin{tabular}{lcccccc}", "\\toprule", "Condition & Hit@1 & Hit@3 & Hit@5 & MRR & Dead@5 & Other-repo@5 \\\\", "\\midrule"]
for c in LAB:
    a = t[c]
    L.append(f"{LAB[c]} & {f3(a['hit1'])} & {f3(a['hit3'])} & {f3(a['hit5'])} & {f3(a['rr'])} & {f3(a['stale5'])} & {f3(a['contam5'])} \\\\")
    if c in ("POPULARITY", "NAIVE_MEMORY", "SNAPSHOT_RAG"): L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}"]; open(f"{T}/main.tex", "w").write("\n".join(L))
# ---- per repo
L = ["\\begin{tabular}{llccc}", "\\toprule", "Repository ($n$) & Condition & Hit@1 & Hit@5 & MRR \\\\", "\\midrule"]
for rp in sorted(S["per_repo"]):
    first = True
    for c in ["POPULARITY", "PLAIN_RAG", "SNAPSHOT_RAG", "EMEMOVERI", "EMEMOVERI_TUNED"]:
        a = S["per_repo"][rp][c]; head = f"{rp} ({S['per_repo_n'][rp]})" if first else ""
        L.append(f"{head} & {LAB[c]} & {f3(a['hit1'])} & {f3(a['hit5'])} & {f3(a['rr'])} \\\\"); first = False
    L.append("\\midrule")
L[-1] = "\\bottomrule"; L.append("\\end{tabular}"); open(f"{T}/repo.tex", "w").write("\n".join(L))
# ---- comparisons
def comp(a, b, m):
    return next(c for c in S["comparisons"] if c["a"] == a and c["b"] == b and c["metric"] == m)
def fp(p):
    return "$<$0.0001" if p < 1e-4 else f"{p:.4f}"
L = ["\\begin{tabular}{llcccc}", "\\toprule", "A & B & Metric & $\\Delta$ (A$-$B) [95\\% CI] & A-only / B-only & $p_{\\text{Holm}}$ \\\\", "\\midrule"]
pairs = [("EMEMOVERI", "PLAIN_RAG"), ("EMEMOVERI", "NAIVE_MEMORY"), ("EMEMOVERI", "SNAPSHOT_RAG"), ("EMEMOVERI_TUNED", "SNAPSHOT_RAG"), ("EMEMOVERI_TUNED", "EMEMOVERI"), ("SNAPSHOT_RAG", "PLAIN_RAG")]
for a, b in pairs:
    for m, mn in (("hit1", "Hit@1"), ("hit5", "Hit@5")):
        c = comp(a, b, m)
        L.append(f"{LAB[a]} & {LAB[b]} & {mn} & {sg(c['diff'])} [{sg(c['ci_lo'])}, {sg(c['ci_hi'])}] & {c['a_only']} / {c['b_only']} & {fp(c['p_holm'])} \\\\")
    L.append("\\midrule")
L[-1] = "\\bottomrule"; L.append("\\end{tabular}"); open(f"{T}/comparisons.tex", "w").write("\n".join(L))
# ---- ablations
AN = {"ABL_no_trust": "No trust term", "ABL_no_evidence": "No evidence term", "ABL_no_staleness": "No staleness term",
      "ABL_state_blind": "State-blind (keep SUPERSEDED)", "ABL_policy_open": "Cross-repo: open", "ABL_policy_local_only": "Cross-repo: local only",
      "ABL_policy_sim_gate": "Cross-repo: similarity gate", "ABL_trust_aggclamp": "Trust: pooled, clamped", "ABL_trust_logit": "Trust: pooled, logit",
      "ABL_halflife_30d": "Staleness half-life 30 d", "ABL_halflife_365d": "Staleness half-life 365 d"}
L = ["\\begin{tabular}{lcc}", "\\toprule", "Variant of default EmemoVeri & $\\Delta$Hit@5 [95\\% CI] & $\\Delta$MRR [95\\% CI] \\\\", "\\midrule"]
for k, name in AN.items():
    a = S["ablations"][k]
    L.append(f"{name} & {sg(a['d_hit5'])} [{sg(a['hit5_lo'])}, {sg(a['hit5_hi'])}] & {sg(a['d_rr'])} [{sg(a['rr_lo'])}, {sg(a['rr_hi'])}] \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; open(f"{T}/ablation.tex", "w").write("\n".join(L))
# ---- data table
M = S["repo_meta"]; Sel = S["selection"]
L = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "Repository & Commits & Source-touching & Eligible & Selected & Cold-file excl. & Clock anomalies \\\\", "\\midrule"]
for rp in ("fastapi", "yt-dlp", "browser-use"):
    L.append(f"{rp} & {M[rp]['commits']:,} & {Sel[rp]['touching']:,} & {Sel[rp]['eligible']:,} & {Sel[rp]['selected']} & {Sel[rp]['cold_file_excluded']} & {M[rp]['timestamp_anomalies']:,} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; open(f"{T}/data.tex", "w").write("\n".join(L))
# ---- selective
L = ["\\begin{tabular}{lcccc}", "\\toprule", "System & Top-1 acc. & AURC (gap) $\\downarrow$ & AUROC (gap) & AUROC (trust) \\\\", "\\midrule"]
for c in ("SNAPSHOT_RAG", "EMEMOVERI", "EMEMOVERI_TUNED"):
    d = S["selective"][c]; L.append(f"{LAB[c]} & {f3(d['acc_all'])} & {f3(d['aurc_gap'])} & {f3(d['auroc_gap'])} & {f3(d['auroc_trust'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; open(f"{T}/selective.tex", "w").write("\n".join(L))
# ---- robustness
L = ["\\begin{tabular}{lccc}", "\\toprule", "Setting (test $n$) & Default vs.\\ Snapshot RAG & Tuned vs.\\ Snapshot RAG & Tuned trust weight \\\\", "\\midrule"]
for d, name in (("real", "Main: warm-up 0.35, 400/repo"), ("real_warm20", "Warm-up 0.20"), ("real_warm50", "Warm-up 0.50"), ("real_n150", "150 tasks/repo")):
    s = json.load(open(f"{R}/{d}/summary.json"))
    c1 = next(c for c in s["comparisons"] if c["a"] == "EMEMOVERI" and c["b"] == "SNAPSHOT_RAG" and c["metric"] == "hit5")
    c2 = next(c for c in s["comparisons"] if c["a"] == "EMEMOVERI_TUNED" and c["b"] == "SNAPSHOT_RAG" and c["metric"] == "hit5")
    L.append(f"{name} ({s['n_test']}) & {sg(c1['diff'])} [{sg(c1['ci_lo'])}, {sg(c1['ci_hi'])}] & {sg(c2['diff'])} [{sg(c2['ci_lo'])}, {sg(c2['ci_hi'])}] & {s['tuned']['w_trust']:.1f} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; open(f"{T}/robustness.tex", "w").write("\n".join(L))
# ---- macros
tu = S["tuned"]; ct = S["controls"]; se = S["selective"]
mac("nDev", S["n_dev"]); mac("nTest", S["n_test"]); mac("nTasks", S["n_dev"] + S["n_test"])
mac("tunedSim", f"{tu['w_sim']:.1f}"); mac("tunedTrust", f"{tu['w_trust']:.1f}"); mac("tunedEv", f"{tu['w_ev']:.1f}")
mac("tunedStale", f"{tu['w_stale']:.2f}"); mac("tunedHL", f"{tu['half_life']:.0f}"); mac("devMRR", f3(tu["dev_mrr"])); mac("devHit", f3(tu["dev_hit5"]))
mac("gridSize", tu["grid_size"])
for c, n in (("SNAPSHOT_RAG", "Snap"), ("EMEMOVERI", "Emv"), ("EMEMOVERI_TUNED", "Tuned"), ("PLAIN_RAG", "Plain"), ("NAIVE_MEMORY", "Naive"), ("POPULARITY", "Pop"), ("RECENCY", "Rec")):
    mac(f"hitFive{n}", f3(t[c]["hit5"])); mac(f"hitOne{n}", f3(t[c]["hit1"])); mac(f"mrr{n}", f3(t[c]["rr"]))
mac("deadPlain", f3(t["PLAIN_RAG"]["stale5"])); mac("deadNaive", f3(t["NAIVE_MEMORY"]["stale5"])); mac("deadBlind", f3(t["ABL_state_blind"]["stale5"]))
mac("contamPlain", f3(t["PLAIN_RAG"]["contam5"])); mac("deadShare", f"{100*S['dead_claim_share_mean']:.0f}")
mac("plainDeadTasks", f"{100*S['tasks_with_dead_in_plain_top5']:.0f}")
mac("shufSnapFive", f3(ct["SHUFFLED::SNAPSHOT_RAG"]["hit5"])); mac("shufEmvFive", f3(ct["SHUFFLED::EMEMOVERI"]["hit5"]))
mac("leakSnapOne", f3(ct["LEAKED::SNAPSHOT_RAG"]["hit1"])); mac("leakEmvOne", f3(ct["LEAKED::EMEMOVERI"]["hit1"]))
mac("trustMean", f3(se["EMEMOVERI"]["trust_mean"])); mac("trustECE", f3(se["EMEMOVERI"]["trust_ece"]))
mac("aurocGapEmv", f3(se["EMEMOVERI"]["auroc_gap"])); mac("aurocGapSnap", f3(se["SNAPSHOT_RAG"]["auroc_gap"])); mac("aurocGapTuned", f3(se["EMEMOVERI_TUNED"]["auroc_gap"]))
mac("aurocTrustEmv", f3(se["EMEMOVERI"]["auroc_trust"]))
for k, nm in (("0.2", "Twenty"), ("0.1", "Ten"), ("0.05", "Five")):
    for c, n in (("EMEMOVERI", "Emv"), ("EMEMOVERI_TUNED", "Tuned"), ("SNAPSHOT_RAG", "Snap")):
        d = se[c][f"thr{k}"]; mac(f"cov{nm}{n}", f"{100*d['coverage']:.1f}"); mac(f"acc{nm}{n}", f"{100*d['acc_answered']:.1f}")
c = comp("EMEMOVERI_TUNED", "SNAPSHOT_RAG", "hit5"); mac("dTunedSnapFive", f3(c["diff"])); mac("dTunedSnapLo", f3(c["ci_lo"])); mac("dTunedSnapHi", f3(c["ci_hi"]))
mac("tunedOnly", c["a_only"]); mac("snapOnly", c["b_only"])
c = comp("EMEMOVERI", "SNAPSHOT_RAG", "hit5"); mac("dEmvSnapFive", f3(c["diff"])); mac("dEmvSnapLo", f3(c["ci_lo"])); mac("dEmvSnapHi", f3(c["ci_hi"])); mac("pEmvSnap", f3(c["p_holm"]))
c = comp("EMEMOVERI", "PLAIN_RAG", "hit5"); mac("dEmvPlainFive", f3(c["diff"])); mac("dEmvPlainLo", f3(c["ci_lo"])); mac("dEmvPlainHi", f3(c["ci_hi"]))
c = comp("EMEMOVERI_TUNED", "EMEMOVERI", "hit5"); mac("dTunedEmvFive", f3(c["diff"])); mac("dTunedEmvLo", f3(c["ci_lo"])); mac("dTunedEmvHi", f3(c["ci_hi"]))
c = comp("EMEMOVERI_TUNED", "SNAPSHOT_RAG", "hit1"); mac("dTunedSnapOne", f3(c["diff"]))
c = comp("SNAPSHOT_RAG", "PLAIN_RAG", "hit5"); mac("dSnapPlainFive", f3(c["diff"]))
for k, n in (("ABL_no_trust", "NoTrust"), ("ABL_no_staleness", "NoStale")):
    mac(f"abl{n}", sg(S["ablations"][k]["d_hit5"]))
for rp, n in (("fastapi", "Fast"), ("yt-dlp", "Ytdlp"), ("browser-use", "Bu")):
    for c2, cn in (("SNAPSHOT_RAG", "Snap"), ("EMEMOVERI", "Emv"), ("EMEMOVERI_TUNED", "Tuned"), ("POPULARITY", "Pop"), ("PLAIN_RAG", "Plain")):
        mac(f"rep{n}{cn}", f3(S["per_repo"][rp][c2]["hit5"]))
mac("nClaims", S["n_claims_final"])
open(os.path.join(os.path.dirname(__file__), "numbers.tex"), "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in N.items()) + "\n")
print("tables + numbers.tex written:", len(N), "macros")
# ---- structural event counts
MS = json.load(open(f"{R}/real/mining_stats.json"))
L = ["\\begin{tabular}{lrrrrr}", "\\toprule", "Repository & Adds & Modifications & Deletions & Renames & Reverts (touching source) \\\\", "\\midrule"]
for rp in ("fastapi", "yt-dlp", "browser-use"):
    m = MS[rp]
    L.append(f"{rp} & {m['source_adds']:,} & {m['source_mods']:,} & {m['source_deletes']:,} & {m['source_renames']:,} & {m['revert_commits']} ({m['revert_commits_touching_source']}) \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; open(f"{T}/events.tex", "w").write("\n".join(L))
