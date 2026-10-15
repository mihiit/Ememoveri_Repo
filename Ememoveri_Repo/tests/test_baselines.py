from datetime import datetime, timedelta, timezone

from ememoveri.baselines import BaselineCondition, run_condition, ALL_CONDITIONS
from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType, MemoryClaim


def _t():
    return datetime.now(timezone.utc)


def make_claim(scope, trust, evidence_strength=1.0, age_days=0):
    c = MemoryClaim(statement="s", scope=scope, trust=trust)
    c.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "e", _t(), strength=evidence_strength)]
    c.updated_at = datetime.now(timezone.utc) - timedelta(days=age_days)
    return c


def test_no_memory_never_returns_a_winner():
    local = make_claim("repoA", trust=0.9)
    result = run_condition([(local, 0.9)], "repoA", BaselineCondition.NO_MEMORY)
    assert result.winner is None
    assert result.ranked == []
    assert not result.abstained


def test_plain_rag_ignores_scope_and_can_pick_cross_repo_purely_on_similarity():
    local_low_sim = make_claim("repoA", trust=0.9)
    cross_high_sim = make_claim("repoB", trust=0.1)
    result = run_condition(
        [(local_low_sim, 0.4), (cross_high_sim, 0.95)],
        "repoA",
        BaselineCondition.PLAIN_RAG,
    )
    assert result.winner is cross_high_sim  # pure similarity, no scope penalty at all


def test_naive_memory_respects_scope_but_ignores_trust():
    local_low_trust = make_claim("repoA", trust=0.1)
    cross_high_trust = make_claim("repoB", trust=0.99)
    result = run_condition(
        [(local_low_trust, 0.5), (cross_high_trust, 0.9)],
        "repoA",
        BaselineCondition.NAIVE_MEMORY,
    )
    # local_only policy means the cross-repo claim is never even eligible,
    # regardless of its similarity or trust.
    assert result.winner is local_low_trust


def test_naive_memory_ignores_trust_within_scope_unlike_ememoveri():
    low_trust_high_sim = make_claim("repoA", trust=0.1)
    high_trust_low_sim = make_claim("repoA", trust=0.95)
    naive = run_condition(
        [(low_trust_high_sim, 0.9), (high_trust_low_sim, 0.55)],
        "repoA",
        BaselineCondition.NAIVE_MEMORY,
    )
    assert naive.winner is low_trust_high_sim  # similarity alone decides

    ememoveri = run_condition(
        [(low_trust_high_sim, 0.9), (high_trust_low_sim, 0.55)],
        "repoA",
        BaselineCondition.EMEMOVERI,
        gap_threshold=0.0,  # force a decision either way for this comparison
    )
    # With trust/evidence factored in, EmemoVeri's choice need not match naive's.
    assert ememoveri.winner is not None


def test_ememoveri_can_abstain_when_others_cannot():
    a = make_claim("repoA", trust=0.55)
    b = make_claim("repoA", trust=0.50)
    naive = run_condition([(a, 0.6), (b, 0.6)], "repoA", BaselineCondition.NAIVE_MEMORY)
    ememoveri = run_condition([(a, 0.6), (b, 0.6)], "repoA", BaselineCondition.EMEMOVERI, gap_threshold=0.5)
    assert not naive.abstained and naive.winner is not None
    assert ememoveri.abstained and ememoveri.winner is None


def test_all_four_conditions_run_without_error_on_same_input():
    local = make_claim("repoA", trust=0.7, age_days=5)
    cross = make_claim("repoB", trust=0.3, age_days=100)
    for condition in ALL_CONDITIONS:
        result = run_condition([(local, 0.7), (cross, 0.6)], "repoA", condition)
        assert result.condition == condition
