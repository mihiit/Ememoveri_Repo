"""
E5-N (mechanism-layer simulation) — Abstention coverage/reliability sweep.

Sweeps the ConflictResolver's gap_threshold and measures, on synthetic
two-claim conflict scenarios with a known ground-truth winner, how coverage
(fraction resolved) trades off against correctness-given-resolved and
abstention precision.

Run:  python -m experiments.e5n_abstention
"""

import json
import os
import random
from datetime import datetime, timedelta, timezone

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType, MemoryClaim
from ememoveri.retrieval import RetrievalScorer
from ememoveri.conflict import ConflictResolver, ResolutionOutcome

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "data", "e5n_abstention.json")


def make_scenario(rng: random.Random):
    a = MemoryClaim(statement="a", scope="repoA", trust=round(rng.uniform(0.2, 0.95), 2))
    b = MemoryClaim(statement="b", scope="repoA", trust=round(rng.uniform(0.2, 0.95), 2))
    for c in (a, b):
        c.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "e", datetime.now(timezone.utc))]
        c.updated_at = datetime.now(timezone.utc) - timedelta(days=rng.uniform(0, 200))
    sim_a, sim_b = round(rng.uniform(0.5, 0.95), 2), round(rng.uniform(0.5, 0.95), 2)
    ground_truth_winner = a.claim_id if a.trust >= b.trust else b.claim_id
    is_ambiguous = abs(a.trust - b.trust) < 0.05 and abs(sim_a - sim_b) < 0.05
    return a, b, sim_a, sim_b, ground_truth_winner, is_ambiguous


def run_threshold(scorer, resolver, scenarios):
    n_resolved = n_correct = n_abstained = n_correct_abstentions = 0
    for a, b, sim_a, sim_b, gt, ambiguous in scenarios:
        score_a = scorer.score(a, sim_a, "repoA")
        score_b = scorer.score(b, sim_b, "repoA")
        resolution = resolver.resolve([(a, score_a), (b, score_b)])
        if resolution.outcome == ResolutionOutcome.ABSTAINED:
            n_abstained += 1
            if ambiguous:
                n_correct_abstentions += 1
        else:
            n_resolved += 1
            if resolution.winner.claim_id == gt:
                n_correct += 1
    n = len(scenarios)
    return {
        "coverage": n_resolved / n,
        "correctness_given_resolved": n_correct / n_resolved if n_resolved else float("nan"),
        "abstention_precision": n_correct_abstentions / n_abstained if n_abstained else float("nan"),
    }


def main(n_scenarios: int = 400, seed: int = 5):
    rng = random.Random(seed)
    scenarios = [make_scenario(rng) for _ in range(n_scenarios)]
    scorer = RetrievalScorer()

    thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    sweep = []
    for t in thresholds:
        resolver = ConflictResolver(gap_threshold=t)
        stats = run_threshold(scorer, resolver, scenarios)
        sweep.append({"threshold": t, **stats})

    results = {"n_scenarios": n_scenarios, "sweep": sweep}
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"{'threshold':10s} {'coverage':10s} {'correct|resolved':18s} {'abst_precision':15s}")
    for row in sweep:
        print(f"{row['threshold']:<10.2f} {row['coverage']:<10.3f} {row['correctness_given_resolved']:<18.3f} {row['abstention_precision']:<15}")
    print(f"\nSaved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
