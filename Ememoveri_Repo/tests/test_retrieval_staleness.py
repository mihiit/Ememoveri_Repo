from datetime import datetime, timedelta, timezone

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType, MemoryClaim
from ememoveri.retrieval import RetrievalScorer


def _t(offset=0):
    return datetime.now(timezone.utc) + timedelta(seconds=offset)


def make_claim(trust=0.8, evidence_strength=1.0, age_days=0, scope="repoA"):
    claim = MemoryClaim(statement="s", scope=scope, trust=trust)
    claim.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "e", _t(), strength=evidence_strength)]
    claim.updated_at = datetime.now(timezone.utc) - timedelta(days=age_days)
    return claim


def test_fresh_claim_scores_higher_than_stale_when_similarity_and_trust_equal():
    scorer = RetrievalScorer()
    fresh = make_claim(age_days=0)
    stale = make_claim(age_days=365)
    s_fresh = scorer.score(fresh, similarity=0.8, query_scope="repoA")
    s_stale = scorer.score(stale, similarity=0.8, query_scope="repoA")
    assert s_fresh > s_stale


def test_high_trust_stale_can_beat_low_trust_fresh_under_small_gap():
    scorer = RetrievalScorer()
    stale_high_trust = make_claim(trust=0.95, age_days=200)
    fresh_low_trust = make_claim(trust=0.3, age_days=0)
    s1 = scorer.score(stale_high_trust, similarity=0.7, query_scope="repoA")
    s2 = scorer.score(fresh_low_trust, similarity=0.7, query_scope="repoA")
    # This demonstrates a genuine crossover case (E3.5): trust can dominate
    # staleness rather than "recent always wins".
    assert s1 > s2


def test_staleness_monotonically_increases_with_age():
    scorer = RetrievalScorer()
    ages = [0, 10, 50, 200, 1000]
    staleness_values = []
    for a in ages:
        c = make_claim(age_days=a)
        staleness_values.append(scorer.staleness(c))
    assert staleness_values == sorted(staleness_values)


def test_scope_exact_match_beats_cross_repo_at_equal_everything_else():
    scorer = RetrievalScorer()
    local = make_claim(scope="repoA")
    cross = make_claim(scope="repoB")
    s_local = scorer.score(local, similarity=0.7, query_scope="repoA")
    s_cross = scorer.score(cross, similarity=0.7, query_scope="repoA")
    assert s_local > s_cross


def test_rank_orders_candidates_descending():
    scorer = RetrievalScorer()
    a = make_claim(trust=0.9, age_days=0)
    b = make_claim(trust=0.2, age_days=500)
    ranked = scorer.rank([(a, 0.6), (b, 0.6)], query_scope="repoA")
    assert ranked[0][0] is a
    assert ranked[0][2] > ranked[1][2]


def test_weights_must_sum_to_one_on_positive_terms():
    import pytest
    from ememoveri.retrieval import RetrievalWeights
    bad = RetrievalWeights(similarity=0.5, trust=0.5, evidence=0.5, scope=0.5, staleness=0.1)
    with pytest.raises(ValueError):
        bad.validate()
