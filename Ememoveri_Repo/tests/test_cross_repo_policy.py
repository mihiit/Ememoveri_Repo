from datetime import datetime, timedelta, timezone

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType, MemoryClaim
from ememoveri.retrieval import RetrievalScorer
from ememoveri.store import CrossRepoPolicy


def _t(offset=0):
    return datetime.now(timezone.utc) + timedelta(seconds=offset)


def make_claim(scope, trust=0.7, age_days=0):
    c = MemoryClaim(statement="s", scope=scope, trust=trust)
    c.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "e", _t())]
    c.updated_at = datetime.now(timezone.utc) - timedelta(days=age_days)
    return c


def scored(scorer, claim, sim, query_scope):
    return (claim, sim, scorer.score(claim, sim, query_scope))


def test_local_only_never_returns_cross_repo_claims():
    scorer = RetrievalScorer()
    local = make_claim("repoA", trust=0.3)
    cross = make_claim("repoB", trust=0.99)
    policy = CrossRepoPolicy(mode="local_only")
    ranked = policy.filter_and_rank(
        [scored(scorer, local, 0.5, "repoA")],
        [scored(scorer, cross, 0.99, "repoA")],
    )
    assert all(c.scope == "repoA" for c, _, _ in ranked)


def test_similarity_gate_blocks_low_similarity_cross_repo():
    scorer = RetrievalScorer()
    local = make_claim("repoA", trust=0.5)
    cross_low_sim = make_claim("repoB", trust=0.99)
    policy = CrossRepoPolicy(mode="similarity_gate", similarity_threshold=0.9)
    ranked = policy.filter_and_rank(
        [scored(scorer, local, 0.5, "repoA")],
        [scored(scorer, cross_low_sim, 0.3, "repoA")],
    )
    assert all(c.scope == "repoA" for c, _, _ in ranked)


def test_similarity_gate_alone_does_not_eliminate_high_similarity_contamination():
    """Mirrors the E4.8 null finding: a similarity/evidence gate reduces
    eligible coverage but does not, by itself, guarantee zero contamination
    among claims that pass the gate."""
    scorer = RetrievalScorer()
    local = make_claim("repoA", trust=0.4)
    cross_high_sim_irrelevant = make_claim("repoB", trust=0.95)
    policy = CrossRepoPolicy(mode="similarity_gate", similarity_threshold=0.5)
    ranked = policy.filter_and_rank(
        [scored(scorer, local, 0.5, "repoA")],
        [scored(scorer, cross_high_sim_irrelevant, 0.95, "repoA")],
    )
    top = ranked[0][0]
    # The irrelevant-but-high-similarity, high-trust cross-repo claim CAN win —
    # this is the contamination risk the gate alone does not solve.
    assert top.scope in ("repoA", "repoB")


def test_local_first_requires_large_margin_to_override():
    scorer = RetrievalScorer()
    local = make_claim("repoA", trust=0.6)
    cross_slightly_better = make_claim("repoB", trust=0.65)
    policy = CrossRepoPolicy(mode="local_first", override_margin=0.15)
    ranked = policy.filter_and_rank(
        [scored(scorer, local, 0.6, "repoA")],
        [scored(scorer, cross_slightly_better, 0.6, "repoA")],
    )
    assert ranked[0][0] is local


def test_local_first_allows_override_when_cross_repo_dominates():
    scorer = RetrievalScorer()
    local = make_claim("repoA", trust=0.2, age_days=1000)
    cross_strong = make_claim("repoB", trust=0.99, age_days=0)
    policy = CrossRepoPolicy(mode="local_first", override_margin=0.05)
    ranked = policy.filter_and_rank(
        [scored(scorer, local, 0.3, "repoA")],
        [scored(scorer, cross_strong, 0.95, "repoA")],
    )
    assert ranked[0][0] is cross_strong


def test_open_mode_ranks_purely_by_score():
    scorer = RetrievalScorer()
    local = make_claim("repoA", trust=0.1)
    cross = make_claim("repoB", trust=0.99)
    policy = CrossRepoPolicy(mode="open")
    ranked = policy.filter_and_rank(
        [scored(scorer, local, 0.5, "repoA")],
        [scored(scorer, cross, 0.99, "repoA")],
    )
    assert ranked[0][0] is cross
