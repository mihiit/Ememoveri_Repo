from ememoveri.conflict import ConflictResolver, ResolutionOutcome
from ememoveri.schema import MemoryClaim


def make_claim(cid, scope="repoA"):
    return MemoryClaim(statement=f"claim-{cid}", scope=scope, claim_id=cid)


def test_resolves_when_gap_exceeds_threshold():
    resolver = ConflictResolver(gap_threshold=0.2)
    a, b = make_claim("a"), make_claim("b")
    resolution = resolver.resolve([(a, 0.9), (b, 0.5)])
    assert resolution.outcome == ResolutionOutcome.RESOLVED
    assert resolution.winner is a


def test_abstains_when_gap_below_threshold():
    resolver = ConflictResolver(gap_threshold=0.2)
    a, b = make_claim("a"), make_claim("b")
    resolution = resolver.resolve([(a, 0.61), (b, 0.60)])
    assert resolution.outcome == ResolutionOutcome.ABSTAINED
    assert resolution.winner is None


def test_single_candidate_always_resolves():
    resolver = ConflictResolver(gap_threshold=0.2)
    a = make_claim("a")
    resolution = resolver.resolve([(a, 0.1)])
    assert resolution.outcome == ResolutionOutcome.RESOLVED
    assert resolution.winner is a


def test_ignores_lower_ranked_claims_beyond_top_two():
    resolver = ConflictResolver(gap_threshold=0.2)
    a, b, c = make_claim("a"), make_claim("b"), make_claim("c")
    resolution = resolver.resolve([(a, 0.9), (b, 0.1), (c, 0.05)])
    assert resolution.outcome == ResolutionOutcome.RESOLVED
    assert resolution.winner is a


def test_abstention_precision_and_coverage():
    resolver = ConflictResolver(gap_threshold=0.2)
    a, b = make_claim("a"), make_claim("b")
    resolutions = [
        resolver.resolve([(a, 0.9), (b, 0.5)]),   # resolved
        resolver.resolve([(a, 0.55), (b, 0.5)]),  # abstained
        resolver.resolve([(a, 0.52), (b, 0.5)]),  # abstained
    ]
    was_ambiguous = [False, True, False]
    precision = ConflictResolver.abstention_precision(resolutions, was_ambiguous)
    assert precision == 0.5  # 1 correct out of 2 abstentions
    coverage = ConflictResolver.coverage(resolutions)
    assert coverage == 1 / 3
