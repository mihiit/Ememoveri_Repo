"""
E5-E (mechanism-layer simulation) — Four-way baseline comparison.

SCOPE NOTE: the real E5-E compares NO_MEMORY / NAIVE_MEMORY / EMEMOVERI (the
blueprint's three conditions) on real coding-task success. This script adds
a fourth condition, PLAIN_RAG (see ememoveri/baselines.py for why it's a
distinct axis from NAIVE_MEMORY), and evaluates all four on the SAME
mechanism-level proxy task used elsewhere in this repo: given a pool of one
"correct" claim and several distractors (some same-repo-but-outdated, some
cross-repo-and-irrelevant, some cross-repo-and-superficially-similar), which
condition selects the correct claim most often, and how does each condition
behave under cross-repository contamination pressure specifically?

This does NOT run an LLM or touch a real repository. It is a controlled
comparison of the retrieval/resolution mechanisms alone, which is a
necessary (not sufficient) building block for the real E5-E study described
in docs/E5_INTEGRATION_GUIDE.md.

Run:  python -m experiments.e5e_baseline_comparison
"""

import json
import os
import random
from datetime import datetime, timedelta, timezone

from ememoveri.baselines import ALL_CONDITIONS, run_condition
from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType, MemoryClaim

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "data", "e5e_baseline_comparison.json")


def make_scenario(rng: random.Random):
    """One retrieval scenario: a correct local claim, an outdated local
    claim (stale, lower trust), and an irrelevant cross-repo claim tuned to
    have high similarity (a contamination risk). Ground truth: the correct
    claim should be selected."""
    correct = MemoryClaim(statement="correct", scope="repoA", trust=round(rng.uniform(0.6, 0.9), 2))
    correct.evidence = [Evidence(EvidenceType.TEST, EvidenceRelation.CORROBORATES, "e", datetime.now(timezone.utc))]
    correct.updated_at = datetime.now(timezone.utc) - timedelta(days=rng.uniform(0, 15))

    outdated_local = MemoryClaim(statement="outdated", scope="repoA", trust=round(rng.uniform(0.2, 0.5), 2))
    outdated_local.evidence = [Evidence(EvidenceType.DOCUMENTATION, EvidenceRelation.CORROBORATES, "e", datetime.now(timezone.utc))]
    outdated_local.updated_at = datetime.now(timezone.utc) - timedelta(days=rng.uniform(60, 400))

    contaminant = MemoryClaim(statement="cross-repo distractor", scope="repoB", trust=round(rng.uniform(0.5, 0.95), 2))
    contaminant.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "e", datetime.now(timezone.utc))]
    contaminant.updated_at = datetime.now(timezone.utc) - timedelta(days=rng.uniform(0, 30))

    sim_correct = round(rng.uniform(0.55, 0.85), 2)
    sim_outdated = round(rng.uniform(0.5, 0.85), 2)
    sim_contaminant = round(rng.uniform(0.6, 0.97), 2)  # deliberately allowed to be high

    candidates = [(correct, sim_correct), (outdated_local, sim_outdated), (contaminant, sim_contaminant)]
    return candidates, correct.claim_id, contaminant.claim_id


def main(n_scenarios: int = 500, seed: int = 17):
    rng = random.Random(seed)
    scenarios = [make_scenario(rng) for _ in range(n_scenarios)]

    stats = {c.value: {"correct": 0, "contaminated": 0, "abstained": 0, "no_answer": 0} for c in ALL_CONDITIONS}

    for candidates, correct_id, contaminant_id in scenarios:
        for condition in ALL_CONDITIONS:
            result = run_condition(candidates, "repoA", condition)
            s = stats[condition.value]
            if result.abstained:
                s["abstained"] += 1
            elif result.winner is None:
                s["no_answer"] += 1
            elif result.winner.claim_id == correct_id:
                s["correct"] += 1
            elif result.winner.claim_id == contaminant_id:
                s["contaminated"] += 1
            else:
                pass  # picked the outdated-local distractor -- neither correct nor contaminated

    n = n_scenarios
    summary = {}
    for cond_name, s in stats.items():
        n_resolved = n - s["abstained"] - s["no_answer"]
        summary[cond_name] = {
            "correct_rate": s["correct"] / n,
            "contamination_rate": s["contaminated"] / n,
            "abstention_rate": s["abstained"] / n,
            "no_answer_rate": s["no_answer"] / n,
            "coverage": n_resolved / n,
            "correct_rate_given_resolved": (s["correct"] / n_resolved) if n_resolved else float("nan"),
        }

    results = {"n_scenarios": n_scenarios, "conditions": summary}
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"{'condition':14s} {'correct':10s} {'correct|resolved':17s} {'contaminated':13s} {'abstained':10s} {'coverage'}")
    for cond_name, row in summary.items():
        print(f"{cond_name:14s} {row['correct_rate']:<10.3f} {row['correct_rate_given_resolved']:<17.3f} "
              f"{row['contamination_rate']:<13.3f} {row['abstention_rate']:<10.3f} {row['coverage']:.3f}")
    print(f"\nSaved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
