"""
E5-V (mechanism-layer simulation) — Adversarial memory poisoning.

Extends E4.6's contamination sensitivity sweep with an explicitly
adversarial framing: an attacker-controlled cross-repository claim is tuned
to maximize similarity, trust, and freshness simultaneously (the worst case
for retrieval scope-awareness) and we measure how often it still wins
against a genuine, but weaker-signaled, local claim -- across the
CrossRepoPolicy mitigation modes.

Run:  python -m experiments.e5v_memory_poisoning
"""

import json
import os

from datetime import datetime, timedelta, timezone

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType, MemoryClaim
from ememoveri.retrieval import RetrievalScorer
from ememoveri.store import CrossRepoPolicy

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "data", "e5v_memory_poisoning.json")


def make_local_claim(trust=0.5, age_days=30):
    c = MemoryClaim(statement="local, genuine", scope="repoA", trust=trust)
    c.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "e", datetime.now(timezone.utc))]
    c.updated_at = datetime.now(timezone.utc) - timedelta(days=age_days)
    return c


def make_adversarial_claim(trust=0.99, age_days=0):
    """Worst-case adversarial cross-repo claim: maximal trust, maximal
    freshness, and (by construction) maximal similarity is supplied at the
    scoring call site -- this represents an attacker who fully controls the
    injected claim's metadata."""
    c = MemoryClaim(statement="cross-repo, adversarial", scope="repoB", trust=trust)
    c.evidence = [Evidence(EvidenceType.TEST, EvidenceRelation.CORROBORATES, "fabricated strong evidence",
                            datetime.now(timezone.utc), strength=2.0)]
    c.updated_at = datetime.now(timezone.utc) - timedelta(days=age_days)
    return c


def main():
    scorer = RetrievalScorer()
    local = make_local_claim()
    adversarial = make_adversarial_claim()

    similarities = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    policies = {
        "open": CrossRepoPolicy(mode="open"),
        "local_only": CrossRepoPolicy(mode="local_only"),
        "similarity_gate_0.9": CrossRepoPolicy(mode="similarity_gate", similarity_threshold=0.9),
        "local_first_m0.15": CrossRepoPolicy(mode="local_first", override_margin=0.15),
        "local_first_m0.30": CrossRepoPolicy(mode="local_first", override_margin=0.30),
    }

    local_sim_fixed = 0.6
    rows = {}
    for policy_name, policy in policies.items():
        wins = []
        for adv_sim in similarities:
            local_scored = [(local, local_sim_fixed, scorer.score(local, local_sim_fixed, "repoA"))]
            adv_scored = [(adversarial, adv_sim, scorer.score(adversarial, adv_sim, "repoA"))]
            ranked = policy.filter_and_rank(local_scored, adv_scored)
            adversarial_wins = bool(ranked) and ranked[0][0] is adversarial
            wins.append(adversarial_wins)
        rows[policy_name] = {
            "similarities": similarities,
            "adversarial_wins": wins,
            "adversarial_win_rate": sum(wins) / len(wins),
        }

    results = {"local_similarity_fixed": local_sim_fixed, "policies": rows}
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("E5-V (mechanism-layer adversarial poisoning) — worst-case adversarial win rate by policy:")
    for name, row in rows.items():
        print(f"  {name:20s}: {100*row['adversarial_win_rate']:.1f}% adversarial win rate across similarity sweep")
    print(f"\nSaved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
