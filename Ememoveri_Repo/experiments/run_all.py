"""
Master experiment runner for EmemoVeri's mechanism-layer evaluation
(E1-E4.8 plus the synthetic E5-H/M/N/O/U/V/W mechanism-stress extensions).

Runs every experiment, writes raw JSON results to results/data/, generates
every figure to results/figures/, and writes a single consolidated
results/SUMMARY.md with the actual numbers from this run (never hand-typed
placeholders) -- this is the "update numbers" entry point: re-run this
script and every number in SUMMARY.md is regenerated from scratch.

SCOPE NOTE (read this before citing numbers from this script anywhere):
This script exercises the memory MECHANISM (state evolution, trust,
retrieval scoring, conflict resolution) under controlled/synthetic
conditions. It does NOT run a coding agent against real repositories -- the
real-repository E5 benchmark (task construction, LLM agent evaluation,
multi-model/multi-repository comparison, statistics on coding-task success)
requires separate infrastructure described in docs/E5_INTEGRATION_GUIDE.md
and is not something any script in this repository can honestly produce
without that infrastructure.

Run:  python -m experiments.run_all
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "SUMMARY.md")

MODULES = [
    ("E1", "experiments.e1_state_transitions", "State-transition validity (9 scenarios)"),
    ("E2", "experiments.e2_order_sensitivity", "Evidence-order sensitivity vs order-invariant aggregation"),
    ("E3", "experiments.e3_staleness_crossover", "Staleness-aware retrieval crossover"),
    ("E4", "experiments.e4_contamination_sweep", "Cross-repository contamination sensitivity/mitigation"),
    ("E5-E*", "experiments.e5e_baseline_comparison", "Four-way baseline comparison: NO_MEMORY/PLAIN_RAG/NAIVE_MEMORY/EMEMOVERI (mechanism-layer simulation)"),
    ("E5-H*", "experiments.e5h_contradiction_stress", "Contradiction stress (mechanism-layer simulation)"),
    ("E5-M*", "experiments.e5m_retrieval_quality", "Retrieval quality: P@K/R@K/MRR/NDCG (mechanism-layer simulation)"),
    ("E5-N*", "experiments.e5n_abstention", "Abstention coverage/reliability sweep (mechanism-layer simulation)"),
    ("E5-O*", "experiments.e5o_trust_calibration", "Trust calibration: ECE/Brier (mechanism-layer simulation)"),
    ("E5-U*", "experiments.e5u_memory_corruption", "Memory corruption robustness (mechanism-layer simulation)"),
    ("E5-V*", "experiments.e5v_memory_poisoning", "Adversarial memory poisoning (mechanism-layer simulation)"),
    ("E5-W*", "experiments.e5w_long_horizon", "Long-horizon memory evolution (mechanism-layer simulation)"),
]


def run_module(module_name: str):
    mod = __import__(module_name, fromlist=["main"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = mod.main()
    return result, buf.getvalue()


def generate_figures():
    from ememoveri import figures as fig

    generated = [fig.fig_state_transition_diagram()]

    # Fig 3: E2 order sensitivity -- rebuild a small labeled set directly
    # (the e2 script itself prints aggregate stats; here we want per-scenario
    # bars, so we recompute a handful of illustrative scenarios directly).
    import random
    from ememoveri.trust import TrustModel
    from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    rng = random.Random(99)
    model = TrustModel(learning_rate=0.6)
    seq_ranges, agg_ranges, labels = [], [], []
    import itertools
    for i in range(6):
        events = [
            Evidence(rng.choice([EvidenceType.COMMIT, EvidenceType.TEST, EvidenceType.ISSUE]),
                     rng.choice([EvidenceRelation.CORROBORATES, EvidenceRelation.CONTRADICTS, EvidenceRelation.REFINES]),
                     "e", _dt.now(_tz.utc) + _td(seconds=j), strength=round(rng.uniform(0.3, 1.5), 2))
            for j in range(4)
        ]
        perms = list(itertools.permutations(events))
        seq_vals = [model.sequential_update_all(p) for p in perms]
        agg_vals = [model.aggregated_update(p) for p in perms]
        seq_ranges.append(TrustModel.dispersion(seq_vals)["range"])
        agg_ranges.append(TrustModel.dispersion(agg_vals)["range"])
        labels.append(f"scenario {i+1}")
    generated.append(fig.fig_e2_order_sensitivity(seq_ranges, agg_ranges, labels))

    # Fig 4: E3 staleness crossover -- rebuild the grid rows directly.
    from ememoveri.schema import MemoryClaim
    from ememoveri.retrieval import RetrievalScorer
    scorer = RetrievalScorer()
    rows = []
    for age in [10, 30, 60, 90, 180, 365]:
        for stale_trust, fresh_trust in [(0.9, 0.3), (0.8, 0.4), (0.7, 0.5), (0.6, 0.6)]:
            stale = MemoryClaim(statement="s", scope="repoA", trust=stale_trust)
            stale.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "e", _dt.now(_tz.utc))]
            stale.updated_at = _dt.now(_tz.utc) - _td(days=age)
            fresh = MemoryClaim(statement="s", scope="repoA", trust=fresh_trust)
            fresh.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "e", _dt.now(_tz.utc))]
            fresh.updated_at = _dt.now(_tz.utc)
            s_stale = scorer.score(stale, 0.75, "repoA")
            s_fresh = scorer.score(fresh, 0.75, "repoA")
            winner = "fresh" if s_fresh > s_stale else "stale"
            rows.append({"age": age, "stale_trust": stale_trust, "fresh_trust": fresh_trust,
                         "winner": winner, "gap": abs(s_fresh - s_stale)})
    generated.append(fig.fig_e3_staleness_crossover(rows))

    # Fig 5: E4 contamination curves -- read the JSON we just computed inline.
    from ememoveri.store import CrossRepoPolicy
    policy_rates = {}
    for mode_label, base_mode, kwargs in [
        ("local_only", "local_only", {}),
        ("open", "open", {}),
        ("similarity_gate_0.7", "similarity_gate", {"similarity_threshold": 0.7}),
        ("similarity_gate_0.9", "similarity_gate", {"similarity_threshold": 0.9}),
        ("local_first_m0.10", "local_first", {"override_margin": 0.10}),
        ("local_first_m0.20", "local_first", {"override_margin": 0.20}),
    ]:
        policy = CrossRepoPolicy(mode=base_mode, **kwargs)
        contaminated, total = 0, 0
        for sim in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for trust in [0.3, 0.5, 0.7, 0.9]:
                local = MemoryClaim(statement="s", scope="repoA", trust=0.5)
                local.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "e", _dt.now(_tz.utc))]
                cross = MemoryClaim(statement="s", scope="repoB", trust=trust)
                cross.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "e", _dt.now(_tz.utc))]
                local_scored = [(local, 0.6, scorer.score(local, 0.6, "repoA"))]
                cross_scored = [(cross, sim, scorer.score(cross, sim, "repoA"))]
                ranked = policy.filter_and_rank(local_scored, cross_scored)
                total += 1
                if ranked and ranked[0][0] is cross:
                    contaminated += 1
        policy_rates[mode_label] = 100 * contaminated / total
    generated.append(fig.fig_e4_contamination_curves(policy_rates))

    # E5-N abstention figure from saved JSON
    with open(os.path.join(DATA_DIR, "e5n_abstention.json")) as f:
        n_data = json.load(f)
    thresholds = [row["threshold"] for row in n_data["sweep"]]
    coverages = [row["coverage"] for row in n_data["sweep"]]
    correctness = [row["correctness_given_resolved"] if row["correctness_given_resolved"] == row["correctness_given_resolved"] else 0.0
                   for row in n_data["sweep"]]
    generated.append(fig.fig_abstention_coverage_reliability(thresholds, coverages, correctness))

    # E5-O calibration figure from saved JSON
    with open(os.path.join(DATA_DIR, "e5o_trust_calibration.json")) as f:
        o_data = json.load(f)
    generated.append(fig.fig_trust_calibration(o_data["bin_confidences"], o_data["bin_accuracies"], o_data["bin_counts"]))

    # E5-U corruption figure from saved JSON
    with open(os.path.join(DATA_DIR, "e5u_memory_corruption.json")) as f:
        u_data = json.load(f)
    generated.append(fig.fig_memory_corruption_degradation(
        u_data["corruption_levels"], u_data["ememoveri_correct_claim_wins_rate"], u_data["naive_correct_claim_wins_rate"]))

    # E5-W trajectory figure from saved JSON
    with open(os.path.join(DATA_DIR, "e5w_long_horizon.json")) as f:
        w_data = json.load(f)
    generated.append(fig.fig_long_horizon_trust_trajectory(w_data["steps"], w_data["trust_values"], w_data["state_labels"]))

    # E5-E four-way baseline comparison figure from saved JSON
    with open(os.path.join(DATA_DIR, "e5e_baseline_comparison.json")) as f:
        e_data = json.load(f)
    generated.append(fig.fig_four_way_baseline_comparison(e_data["conditions"]))

    return generated


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    all_results = {}
    all_logs = {}

    for exp_id, module_name, description in MODULES:
        print(f"Running {exp_id}: {description} ...")
        result, log = run_module(module_name)
        all_results[exp_id] = result
        all_logs[exp_id] = log

    print("Generating figures ...")
    figure_paths = generate_figures()

    # --- write consolidated summary ---
    lines = []
    lines.append("# EmemoVeri Mechanism-Layer Results Summary")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")
    lines.append(
        "**Scope note:** entries marked `*` (E5-H/M/N/O/U/V/W) are mechanism-layer "
        "synthetic stress tests of the memory engine itself, NOT results from a real "
        "coding agent evaluated on real repositories. E1-E4 reproduce the controlled "
        "mechanism-validation experiments from the project blueprint. The real-repository "
        "E5 benchmark (task construction, agent evaluation, multi-model comparison, "
        "statistics on coding-task success) is not run by this script -- see "
        "`docs/E5_INTEGRATION_GUIDE.md`."
    )
    lines.append("")

    for exp_id, module_name, description in MODULES:
        lines.append(f"## {exp_id} — {description}")
        lines.append("")
        lines.append("```")
        lines.append(all_logs[exp_id].strip())
        lines.append("```")
        lines.append("")

    lines.append("## Generated figures")
    lines.append("")
    for path in figure_paths:
        rel = os.path.relpath(path, RESULTS_DIR)
        lines.append(f"- `{rel}`")
    lines.append("")

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"\nAll experiments complete. Summary written to {SUMMARY_PATH}")
    print(f"Figures written to {os.path.join(RESULTS_DIR, 'figures')}")
    print(f"Raw JSON written to {DATA_DIR}")


if __name__ == "__main__":
    main()
