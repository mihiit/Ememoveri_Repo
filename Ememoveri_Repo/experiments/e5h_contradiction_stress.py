"""
E5-H (mechanism-layer simulation) — Contradiction stress test.

IMPORTANT SCOPE NOTE: the full E5-H in the blueprint evaluates a live coding
agent's behavior on real repository tasks. This script does NOT do that (no
LLM, no repository, no patch generation) -- it stress-tests the underlying
memory mechanism itself (state engine + retrieval + conflict resolver) under
synthetic contradiction scenarios, which is a necessary but not sufficient
precursor. Results here characterize the mechanism; they are not a claim
about end-to-end coding-agent reliability.

Run:  python -m experiments.e5h_contradiction_stress
"""

import json
import os
import random
from datetime import datetime, timedelta, timezone

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType, MemoryClaim
from ememoveri.state_engine import StateEvolutionEngine
from ememoveri.retrieval import RetrievalScorer
from ememoveri.conflict import ConflictResolver, ResolutionOutcome
from ememoveri.trust import TrustModel

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "data", "e5h_contradiction_stress.json")


def _t(offset=0):
    return datetime.now(timezone.utc) + timedelta(seconds=offset)


def make_scenario(rng: random.Random):
    """Two competing claims about the same scope: one 'correct' (backed by
    stronger/fresher evidence), one 'outdated/contradictory'. Ground truth
    ('correct_wins') is defined by construction, letting us measure whether
    the mechanism's resolution matches the constructed ground truth."""
    correct = MemoryClaim(statement="correct claim", scope="repoA")
    outdated = MemoryClaim(statement="outdated claim", scope="repoA")

    engine = StateEvolutionEngine()
    trust_model = TrustModel()
    correct_strength = round(rng.uniform(0.5, 1.5), 2)
    outdated_strength = round(rng.uniform(0.3, 1.2), 2)

    engine.apply_evidence(correct, Evidence(EvidenceType.TEST, EvidenceRelation.CORROBORATES,
                                             "test confirms", _t(), strength=correct_strength))
    engine.apply_evidence(outdated, Evidence(EvidenceType.DOCUMENTATION, EvidenceRelation.CORROBORATES,
                                              "old doc claims", _t(-1), strength=outdated_strength))
    # The outdated claim gets contradicted by the correct claim's evidence type (simulating
    # a newer commit/test invalidating the old documentation).
    engine.apply_evidence(outdated, Evidence(EvidenceType.COMMIT, EvidenceRelation.CONTRADICTS,
                                              "recent commit disputes old doc", _t(), strength=correct_strength))

    # Trust must actually reflect accumulated evidence -- without this, both
    # claims keep the default prior trust (0.5) and retrieval scores barely
    # differ regardless of the underlying state, artificially inflating the
    # abstention rate. Using the order-invariant aggregated update here
    # (rather than sequential) isolates the contradiction-handling property
    # under test from the separate order-sensitivity property (E2).
    correct.trust = trust_model.aggregated_update(correct.evidence)
    outdated.trust = trust_model.aggregated_update(outdated.evidence)

    correct_sim = round(rng.uniform(0.6, 0.95), 2)
    outdated_sim = round(rng.uniform(0.55, 0.9), 2)
    correct_age_days = round(rng.uniform(0, 5), 1)
    outdated_age_days = round(rng.uniform(20, 400), 1)
    correct.updated_at = datetime.now(timezone.utc) - timedelta(days=correct_age_days)
    outdated.updated_at = datetime.now(timezone.utc) - timedelta(days=outdated_age_days)

    # Ground truth: the "correct" claim should win because it is fresher AND
    # has net-positive support after the contradiction, by construction.
    ground_truth_correct_claim_id = correct.claim_id
    is_genuinely_ambiguous = abs(correct_sim - outdated_sim) < 0.05 and abs(correct_strength - outdated_strength) < 0.1

    return correct, outdated, correct_sim, outdated_sim, ground_truth_correct_claim_id, is_genuinely_ambiguous


def main(n_scenarios: int = 500, seed: int = 7):
    rng = random.Random(seed)
    scorer = RetrievalScorer()
    resolver = ConflictResolver(gap_threshold=0.20)

    n_detected_contradiction = 0
    n_correct_resolution = 0
    n_inappropriate_commitment = 0
    n_abstained = 0
    n_correct_abstentions = 0
    n_resolved = 0

    for _ in range(n_scenarios):
        correct, outdated, correct_sim, outdated_sim, gt_id, ambiguous = make_scenario(rng)

        if outdated.state.value == "CONTESTED":
            n_detected_contradiction += 1

        score_correct = scorer.score(correct, correct_sim, "repoA")
        score_outdated = scorer.score(outdated, outdated_sim, "repoA")
        resolution = resolver.resolve([(correct, score_correct), (outdated, score_outdated)])

        if resolution.outcome == ResolutionOutcome.ABSTAINED:
            n_abstained += 1
            if ambiguous:
                n_correct_abstentions += 1
        else:
            n_resolved += 1
            if resolution.winner.claim_id == gt_id:
                n_correct_resolution += 1
            else:
                n_inappropriate_commitment += 1

    results = {
        "n_scenarios": n_scenarios,
        "contradiction_detection_rate": n_detected_contradiction / n_scenarios,
        "resolution_correct_rate_given_resolved": n_correct_resolution / n_resolved if n_resolved else float("nan"),
        "inappropriate_commitment_rate_given_resolved": n_inappropriate_commitment / n_resolved if n_resolved else float("nan"),
        "abstention_rate": n_abstained / n_scenarios,
        "abstention_precision": n_correct_abstentions / n_abstained if n_abstained else float("nan"),
        "coverage": n_resolved / n_scenarios,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("E5-H (mechanism-layer contradiction stress) results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
