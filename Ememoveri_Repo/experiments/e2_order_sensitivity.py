"""
Reproduces the E2 evidence-order-sensitivity experiment structure: for a
configurable number of random mixed-evidence scenarios, compute trust under
every permutation of the evidence sequence (sequential update) and compare
its dispersion (range, stdev) against the order-invariant aggregated update.

The blueprint's full-scale run was 720 permutations x 100 scenarios = 72,000
permutations. This script defaults to a smaller N for quick local runs; pass
--full for a closer approximation (still bounded by permutation blowup —
scenarios use <= 6 evidence items so exhaustive permutation stays tractable).

Run:  python -m experiments.e2_order_sensitivity [--scenarios N] [--full]
"""

import argparse
import itertools
import random
from datetime import datetime, timedelta, timezone

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType
from ememoveri.trust import TrustModel

RELATIONS = [EvidenceRelation.CORROBORATES, EvidenceRelation.CONTRADICTS, EvidenceRelation.REFINES]
TYPES = [EvidenceType.COMMIT, EvidenceType.TEST, EvidenceType.ISSUE, EvidenceType.DOCUMENTATION]


def _t(i):
    return datetime.now(timezone.utc) + timedelta(seconds=i)


def random_scenario(rng: random.Random, n_events: int = 4):
    return [
        Evidence(
            evidence_type=rng.choice(TYPES),
            relation=rng.choice(RELATIONS),
            content=f"evidence_{i}",
            timestamp=_t(i),
            strength=round(rng.uniform(0.3, 1.5), 2),
        )
        for i in range(n_events)
    ]


def evaluate_scenario(model: TrustModel, evidence_items):
    perms = list(itertools.permutations(evidence_items))
    seq_trusts = [model.sequential_update_all(p) for p in perms]
    agg_trusts = [model.aggregated_update(p) for p in perms]
    return {
        "n_permutations": len(perms),
        "sequential": TrustModel.dispersion(seq_trusts),
        "aggregated": TrustModel.dispersion(agg_trusts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=25)
    parser.add_argument("--events", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--full", action="store_true", help="run 100 scenarios (matches blueprint scale)")
    args = parser.parse_args()

    n_scenarios = 100 if args.full else args.scenarios
    rng = random.Random(args.seed)
    model = TrustModel(learning_rate=0.6)

    seq_ranges, agg_ranges = [], []
    total_permutations = 0
    for i in range(n_scenarios):
        scenario = random_scenario(rng, n_events=args.events)
        result = evaluate_scenario(model, scenario)
        total_permutations += result["n_permutations"]
        seq_ranges.append(result["sequential"]["range"])
        agg_ranges.append(result["aggregated"]["range"])

    print(f"Scenarios: {n_scenarios} | Events/scenario: {args.events} | Total permutations evaluated: {total_permutations}")
    print(f"Sequential update  -- mean trust-range across scenarios: {sum(seq_ranges)/len(seq_ranges):.6f}")
    print(f"Aggregated update  -- mean trust-range across scenarios: {sum(agg_ranges)/len(agg_ranges):.6f}")
    print()
    print("Interpretation: a non-zero sequential range with a ~zero aggregated")
    print("range demonstrates that the sequential mechanism is order-sensitive")
    print("and that pooled aggregation removes that sensitivity (RQ3).")


if __name__ == "__main__":
    main()
