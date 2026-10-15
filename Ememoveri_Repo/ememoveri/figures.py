"""
Figure generation for EmemoVeri mechanism-layer experiments.

All figures are generated from data actually produced by the experiment
scripts in `experiments/` -- nothing here is a mockup or placeholder image.
Uses matplotlib's non-interactive Agg backend so it runs headless in CI or a
container with no display.

Output: results/figures/*.png (300 DPI, paper-ready).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, name: str):
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_state_transition_diagram():
    """Fig. 2 — Memory state-transition diagram (schematic, drawn from the
    actual MemoryState enum + StateEvolutionEngine transition rules, not a
    generic illustration)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = {
        "PROVISIONAL": (0.5, 0.85),
        "CORROBORATED": (0.15, 0.5),
        "REFINED": (0.5, 0.5),
        "CONTESTED": (0.85, 0.5),
        "SUPERSEDED": (0.5, 0.12),
    }
    colors = {
        "PROVISIONAL": "#9aa5b1",
        "CORROBORATED": "#2e7d32",
        "REFINED": "#1565c0",
        "CONTESTED": "#e65100",
        "SUPERSEDED": "#616161",
    }
    for name, (x, y) in positions.items():
        ax.add_patch(mpatches.FancyBboxPatch((x - 0.09, y - 0.045), 0.18, 0.09,
                                              boxstyle="round,pad=0.01",
                                              linewidth=1.5, edgecolor="black",
                                              facecolor=colors[name], alpha=0.85))
        ax.text(x, y, name, ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    edges = [
        ("PROVISIONAL", "CORROBORATED", "net support > threshold"),
        ("PROVISIONAL", "REFINED", "+refines, no contradiction"),
        ("PROVISIONAL", "CONTESTED", "conflicting/negative evidence"),
        ("CORROBORATED", "CONTESTED", "new contradiction"),
        ("CONTESTED", "CORROBORATED", "dominant new corroboration"),
        ("CORROBORATED", "SUPERSEDED", "SUPERSEDES evidence"),
        ("REFINED", "SUPERSEDED", "SUPERSEDES evidence"),
        ("CONTESTED", "SUPERSEDED", "SUPERSEDES evidence"),
    ]
    for src, dst, label in edges:
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2,
                                     connectionstyle="arc3,rad=0.15"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("EmemoVeri Memory State-Transition Lattice", fontsize=13, fontweight="bold")
    return _save(fig, "fig2_state_transition_diagram")


def fig_e2_order_sensitivity(sequential_ranges, aggregated_ranges, labels=None):
    """Fig. 3 — E2 permutation sensitivity: sequential vs order-invariant."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n = len(sequential_ranges)
    x = list(range(n))
    labels = labels or [f"s{i+1}" for i in x]
    width = 0.38
    ax.bar([i - width / 2 for i in x], sequential_ranges, width=width, label="Sequential (order-sensitive)", color="#c0392b")
    ax.bar([i + width / 2 for i in x], aggregated_ranges, width=width, label="Aggregated (order-invariant)", color="#2980b9")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Trust range across permutations")
    ax.set_title("E2: Evidence-Order Sensitivity vs Order-Invariant Aggregation")
    ax.legend()
    return _save(fig, "fig3_e2_order_sensitivity")


def fig_e3_staleness_crossover(rows):
    """Fig. 4 — E3/E3.5 staleness crossover surface.

    rows: list of dicts with keys age, stale_trust, fresh_trust, winner, gap
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ages = sorted(set(r["age"] for r in rows))
    trust_pairs = sorted(set((r["stale_trust"], r["fresh_trust"]) for r in rows))
    for stale_t, fresh_t in trust_pairs:
        sub = [r for r in rows if r["stale_trust"] == stale_t and r["fresh_trust"] == fresh_t]
        sub.sort(key=lambda r: r["age"])
        signed_gap = [r["gap"] if r["winner"] == "fresh" else -r["gap"] for r in sub]
        ax.plot([r["age"] for r in sub], signed_gap, marker="o",
                 label=f"stale_trust={stale_t:.2f}, fresh_trust={fresh_t:.2f}")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Age of stale claim (days)")
    ax.set_ylabel("Score gap (fresh − stale); positive = fresh wins")
    ax.set_title("E3/E3.5: Staleness–Trust Crossover Surface")
    ax.legend(fontsize=7)
    return _save(fig, "fig4_e3_staleness_crossover")


def fig_e4_contamination_curves(policy_rates: dict):
    """Fig. 5 — E4.6/E4.7 contamination sensitivity across policy modes.

    policy_rates: {policy_name: contamination_rate_percent}
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(policy_rates.keys())
    rates = [policy_rates[n] for n in names]
    colors = ["#7f8c8d" if "local_only" in n else
              "#c0392b" if n == "open" else
              "#f39c12" if "similarity_gate" in n else "#27ae60" for n in names]
    ax.bar(names, rates, color=colors)
    ax.set_ylabel("Contamination rate (%)")
    ax.set_title("E4.6/E4.7: Cross-Repository Contamination by Mitigation Policy")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    for i, r in enumerate(rates):
        ax.text(i, r + 1, f"{r:.1f}%", ha="center", fontsize=8)
    return _save(fig, "fig5_e4_contamination_curves")


def fig_abstention_coverage_reliability(thresholds, coverages, correctness_given_resolved):
    """E5-N: coverage-reliability trade-off as the abstention gap threshold varies."""
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(thresholds, coverages, marker="o", color="#2980b9", label="Coverage")
    ax1.set_xlabel("Abstention gap threshold")
    ax1.set_ylabel("Coverage (fraction resolved)", color="#2980b9")
    ax1.tick_params(axis="y", labelcolor="#2980b9")

    ax2 = ax1.twinx()
    ax2.plot(thresholds, correctness_given_resolved, marker="s", color="#c0392b", label="Correctness | resolved")
    ax2.set_ylabel("Correctness given resolved", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    fig.suptitle("E5-N: Coverage–Reliability Trade-off Under Abstention")
    return _save(fig, "fig_e5n_abstention_tradeoff")


def fig_trust_calibration(bin_confidences, bin_accuracies, bin_counts):
    """E5-O: reliability diagram for trust-as-confidence calibration."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.bar(bin_confidences, bin_accuracies, width=0.08, edgecolor="black",
           color="#2980b9", alpha=0.75, label="Observed")
    ax.set_xlabel("Mean predicted trust (confidence)")
    ax.set_ylabel("Empirical correctness")
    ax.set_title("E5-O: Trust Calibration (Reliability Diagram)")
    ax.legend()
    return _save(fig, "fig_e5o_trust_calibration")


def fig_memory_corruption_degradation(corruption_levels, ememoveri_acc, naive_acc):
    """E5-U: graceful-degradation curve under injected memory corruption."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(corruption_levels, ememoveri_acc, marker="o", label="EmemoVeri", color="#27ae60")
    ax.plot(corruption_levels, naive_acc, marker="s", label="Naive memory", color="#c0392b")
    ax.set_xlabel("Fraction of corrupted evidence")
    ax.set_ylabel("Correct-claim-wins rate")
    ax.set_title("E5-U: Degradation Under Injected Memory Corruption")
    ax.legend()
    return _save(fig, "fig_e5u_corruption_degradation")


def fig_long_horizon_trust_trajectory(steps, trust_values, state_labels):
    """E5-W: trust trajectory across a full lifecycle sequence
    (corroborate -> refine -> contradict -> supersede)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, trust_values, marker="o", color="#1565c0")
    for x, y, label in zip(steps, trust_values, state_labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, rotation=20)
    ax.set_xlabel("Evidence step")
    ax.set_ylabel("Trust")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("E5-W: Long-Horizon Memory Evolution (Trust Trajectory)")
    return _save(fig, "fig_e5w_long_horizon_trajectory")


def fig_four_way_baseline_comparison(summary: dict):
    """E5-E: four-way baseline comparison (NO_MEMORY / PLAIN_RAG /
    NAIVE_MEMORY / EMEMOVERI), showing correct-rate, correct-given-resolved,
    contamination-rate, and coverage side by side."""
    conditions = list(summary.keys())
    correct = [summary[c]["correct_rate"] for c in conditions]
    correct_given_resolved = [
        summary[c]["correct_rate_given_resolved"] if summary[c]["correct_rate_given_resolved"] == summary[c]["correct_rate_given_resolved"] else 0.0
        for c in conditions
    ]
    contamination = [summary[c]["contamination_rate"] for c in conditions]
    coverage = [summary[c]["coverage"] for c in conditions]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(conditions)))
    width = 0.2
    ax.bar([i - 1.5 * width for i in x], correct, width=width, label="Correct rate", color="#2980b9")
    ax.bar([i - 0.5 * width for i in x], correct_given_resolved, width=width, label="Correct | resolved", color="#27ae60")
    ax.bar([i + 0.5 * width for i in x], contamination, width=width, label="Contamination rate", color="#c0392b")
    ax.bar([i + 1.5 * width for i in x], coverage, width=width, label="Coverage", color="#8e44ad")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("E5-E: Four-Way Memory Condition Comparison (mechanism-layer)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    return _save(fig, "fig_e5e_baseline_comparison")
