"""
E5-U (mechanism-layer simulation) — Memory corruption robustness.

Injects controlled corruption (wrong-repository scope, incorrect trust,
stale timestamps, contradictory evidence) into a fraction of claims and
measures how gracefully EmemoVeri's full scoring (similarity + trust +
evidence + scope - staleness) degrades compared to a "naive memory"
baseline that ranks by raw similarity alone. Ground truth: the correct
claim should still win whenever a genuinely-better claim exists in the
candidate pool.

Run:  python -m experiments.e5u_memory_corruption
"""

import json
import os
import random
from datetime import datetime, timedelta, timezone

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType, MemoryClaim
from ememoveri.retrieval import RetrievalScorer

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "data", "e5u_memory_corruption.json")


def make_pair(rng: random.Random, corrupt: bool):
    correct = MemoryClaim(statement="correct", scope="repoA", trust=round(rng.uniform(0.6, 0.9), 2))
    correct.evidence = [Evidence(EvidenceType.TEST, EvidenceRelation.CORROBORATES, "e", datetime.now(timezone.utc))]
    correct.updated_at = datetime.now(timezone.utc) - timedelta(days=rng.uniform(0, 10))

    distractor = MemoryClaim(statement="distractor", scope="repoA", trust=round(rng.uniform(0.2, 0.5), 2))
    distractor.evidence = [Evidence(EvidenceType.ISSUE, EvidenceRelation.CORROBORATES, "e", datetime.now(timezone.utc))]
    distractor.updated_at = datetime.now(timezone.utc) - timedelta(days=rng.uniform(30, 200))

    if corrupt:
        # Corrupt the distractor to look artificially strong: inflate trust,
        # fabricate a fresh timestamp, and mislabel its evidence as strong
        # corroboration -- simulating misleading/incorrect metadata.
        distractor.trust = round(rng.uniform(0.85, 0.99), 2)
        distractor.updated_at = datetime.now(timezone.utc)
        distractor.evidence = [Evidence(EvidenceType.TEST, EvidenceRelation.CORROBORATES, "fabricated", datetime.now(timezone.utc), strength=2.0)]

    sim_correct = round(rng.uniform(0.65, 0.9), 2)
    sim_distractor = round(rng.uniform(0.55, 0.85), 2)
    return correct, distractor, sim_correct, sim_distractor


def naive_score(claim, similarity):
    return similarity  # naive memory: similarity only, ignores trust/evidence/staleness/scope


def main(n_trials: int = 300, seed: int = 11):
    rng = random.Random(seed)
    scorer = RetrievalScorer()
    corruption_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    ememoveri_acc, naive_acc = [], []
    for level in corruption_levels:
        n_correct_ememoveri = n_correct_naive = 0
        for _ in range(n_trials):
            corrupt = rng.random() < level
            correct, distractor, sim_c, sim_d = make_pair(rng, corrupt)

            s_correct = scorer.score(correct, sim_c, "repoA")
            s_distractor = scorer.score(distractor, sim_d, "repoA")
            if s_correct >= s_distractor:
                n_correct_ememoveri += 1

            n_correct_naive_score = naive_score(correct, sim_c)
            n_distractor_naive_score = naive_score(distractor, sim_d)
            if n_correct_naive_score >= n_distractor_naive_score:
                n_correct_naive += 1

        ememoveri_acc.append(n_correct_ememoveri / n_trials)
        naive_acc.append(n_correct_naive / n_trials)

    results = {
        "n_trials": n_trials,
        "corruption_levels": corruption_levels,
        "ememoveri_correct_claim_wins_rate": ememoveri_acc,
        "naive_correct_claim_wins_rate": naive_acc,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"{'corruption_level':18s} {'ememoveri_wins':16s} {'naive_wins'}")
    for lvl, e, n in zip(corruption_levels, ememoveri_acc, naive_acc):
        print(f"{lvl:<18.2f} {e:<16.3f} {n:.3f}")
    print(f"\nSaved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
