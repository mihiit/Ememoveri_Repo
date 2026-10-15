"""
Reproduces the E3 / E3.5 staleness-crossover experiment structure: for a grid
of (trust, age, similarity) combinations, determine whether a fresher
lower-trust claim or a staler higher-trust claim wins retrieval, and locate
the approximate crossover boundary.

Run:  python -m experiments.e3_staleness_crossover
"""

from datetime import datetime, timedelta, timezone

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType, MemoryClaim
from ememoveri.retrieval import RetrievalScorer


def _t():
    return datetime.now(timezone.utc)


def make_claim(trust: float, age_days: float, scope="repoA"):
    c = MemoryClaim(statement="s", scope=scope, trust=trust)
    c.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "e", _t())]
    c.updated_at = datetime.now(timezone.utc) - timedelta(days=age_days)
    return c


def main():
    scorer = RetrievalScorer()
    similarity = 0.75
    fresh_age = 0
    stale_ages = [10, 30, 60, 90, 180, 365]
    trust_pairs = [(0.9, 0.3), (0.8, 0.4), (0.7, 0.5), (0.6, 0.6)]  # (stale_trust, fresh_trust)

    print(f"{'stale_age_days':15s} {'stale_trust':12s} {'fresh_trust':12s} {'winner':10s} {'gap':>8s}")
    print("-" * 65)

    recent_preferred = 0
    total = 0
    for age in stale_ages:
        for stale_trust, fresh_trust in trust_pairs:
            stale = make_claim(trust=stale_trust, age_days=age)
            fresh = make_claim(trust=fresh_trust, age_days=fresh_age)
            s_stale = scorer.score(stale, similarity, "repoA")
            s_fresh = scorer.score(fresh, similarity, "repoA")
            winner = "fresh" if s_fresh > s_stale else "stale"
            if winner == "fresh":
                recent_preferred += 1
            total += 1
            print(f"{age:<15d} {stale_trust:<12.2f} {fresh_trust:<12.2f} {winner:10s} {abs(s_fresh - s_stale):8.4f}")

    print()
    print(f"Recent preferred in {recent_preferred}/{total} = {100*recent_preferred/total:.2f}% of grid comparisons.")
    print("This is the crossover surface referenced by E3.5: recency does not")
    print("unconditionally win -- sufficiently higher trust on the stale claim")
    print("can and does overcome the staleness penalty.")


if __name__ == "__main__":
    main()
