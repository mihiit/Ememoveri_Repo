"""
Reproduces the E4.5/E4.6 cross-repository contamination sensitivity study:
sweep local-claim (similarity, trust) against irrelevant-cross-repo-claim
(similarity, trust) and measure how often the irrelevant cross-repo claim
outranks the local claim, under each CrossRepoPolicy mode.

This produces a contamination-rate surface (E4.6) rather than a single
number, and separately compares policy modes (E4.7) to show that a
similarity/evidence gate reduces *eligibility* without necessarily reducing
the contamination rate *among eligible items* (the E4.8 null finding).

Run:  python -m experiments.e4_contamination_sweep
"""

from datetime import datetime, timedelta, timezone

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType, MemoryClaim
from ememoveri.retrieval import RetrievalScorer
from ememoveri.store import CrossRepoPolicy

SIMILARITIES = [0.5, 0.6, 0.7, 0.8, 0.9]
TRUSTS = [0.3, 0.5, 0.7, 0.9]


def make_claim(scope, trust):
    c = MemoryClaim(statement="s", scope=scope, trust=trust)
    c.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "e", datetime.now(timezone.utc))]
    return c


def contamination_rate(scorer: RetrievalScorer, policy: CrossRepoPolicy) -> float:
    local_sim, local_trust = 0.6, 0.5  # fixed "typical" local claim
    contaminated = 0
    total = 0
    for cross_sim in SIMILARITIES:
        for cross_trust in TRUSTS:
            local = make_claim("repoA", local_trust)
            cross = make_claim("repoB", cross_trust)  # irrelevant: different repo, unrelated content
            local_scored = [(local, local_sim, scorer.score(local, local_sim, "repoA"))]
            cross_scored = [(cross, cross_sim, scorer.score(cross, cross_sim, "repoA"))]
            ranked = policy.filter_and_rank(local_scored, cross_scored)
            total += 1
            if ranked and ranked[0][0] is cross:
                contaminated += 1
    return contaminated / total, total


def main():
    scorer = RetrievalScorer()
    print(f"{'policy_mode':20s} {'contamination_rate':20s} {'n_grid_cells'}")
    print("-" * 60)
    for mode, kwargs in [
        ("local_only", {}),
        ("open", {}),
        ("similarity_gate_0.7", {"similarity_threshold": 0.7}),
        ("similarity_gate_0.9", {"similarity_threshold": 0.9}),
        ("local_first_m0.10", {"override_margin": 0.10}),
        ("local_first_m0.20", {"override_margin": 0.20}),
    ]:
        base_mode = "similarity_gate" if mode.startswith("similarity_gate") else (
            "local_first" if mode.startswith("local_first") else mode
        )
        policy = CrossRepoPolicy(mode=base_mode, **kwargs)
        rate, n = contamination_rate(scorer, policy)
        print(f"{mode:20s} {100*rate:18.2f}% {n}")

    print()
    print("Note: 'open' establishes the unmitigated contamination ceiling.")
    print("'local_only' establishes the (zero cross-repo utility) floor.")
    print("Compare gate/local_first rates against 'open' to assess mitigation")
    print("value, and remember E4.8's finding: a gate can shrink the eligible")
    print("pool without shrinking the contamination rate *within* it.")


if __name__ == "__main__":
    main()
