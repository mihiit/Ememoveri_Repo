"""
E5-O (mechanism-layer simulation) — Trust calibration.

SCOPE NOTE: real calibration requires ground-truth task-success labels tied
to trust scores from an actual agent run. Here we generate synthetic claims
via the TrustModel's aggregated update and pair each claim's resulting trust
with a synthetically generated correctness label whose probability is a
NOISY function of trust (not equal to trust) -- this tests whether ECE/Brier
correctly detect a realistic degree of miscalibration, and gives a reference
implementation to re-run once real labels exist.

Run:  python -m experiments.e5o_trust_calibration
"""

import json
import os
import random
from datetime import datetime, timezone

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType
from ememoveri.trust import TrustModel

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "data", "e5o_trust_calibration.json")


def make_evidence(rng: random.Random):
    n = rng.randint(1, 5)
    items = []
    for _ in range(n):
        relation = rng.choice([EvidenceRelation.CORROBORATES, EvidenceRelation.CONTRADICTS, EvidenceRelation.REFINES])
        etype = rng.choice([EvidenceType.COMMIT, EvidenceType.TEST, EvidenceType.ISSUE, EvidenceType.DOCUMENTATION])
        items.append(Evidence(etype, relation, "e", datetime.now(timezone.utc), strength=round(rng.uniform(0.3, 1.5), 2)))
    return items


def synthetic_correctness_label(rng: random.Random, trust: float, miscalibration_noise: float = 0.15) -> bool:
    """A deliberately IMPERFECT proxy: true probability of correctness is
    trust shifted by bounded noise, so the resulting calibration is neither
    perfect nor meaningless -- exercising the ECE/Brier metrics realistically."""
    p_correct = min(1.0, max(0.0, trust + rng.uniform(-miscalibration_noise, miscalibration_noise)))
    return rng.random() < p_correct


def main(n_claims: int = 1000, seed: int = 3, n_bins: int = 10):
    rng = random.Random(seed)
    model = TrustModel()
    confidences, correctness = [], []
    for _ in range(n_claims):
        evidence_items = make_evidence(rng)
        trust = model.aggregated_update(evidence_items)
        confidences.append(trust)
        correctness.append(synthetic_correctness_label(rng, trust))

    ece = TrustModel.expected_calibration_error(confidences, correctness, n_bins=n_bins)
    brier = TrustModel.brier_score(confidences, correctness)

    # bin-level data for the reliability diagram
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, correctness):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, y))
    bin_confidences, bin_accuracies, bin_counts = [], [], []
    for b in bins:
        if not b:
            continue
        bin_confidences.append(sum(c for c, _ in b) / len(b))
        bin_accuracies.append(sum(1.0 for _, y in b if y) / len(b))
        bin_counts.append(len(b))

    results = {
        "n_claims": n_claims,
        "n_bins": n_bins,
        "expected_calibration_error": ece,
        "brier_score": brier,
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"E5-O (mechanism-layer trust calibration): ECE={ece:.4f}, Brier={brier:.4f}")
    print(f"Saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
