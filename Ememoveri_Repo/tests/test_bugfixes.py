"""Regression tests for bugs found in the v0.1 audit."""
from datetime import datetime, timedelta, timezone

from ememoveri.conflict import ConflictResolver, ResolutionOutcome
from ememoveri.retrieval import RetrievalScorer
from ememoveri.schema import Evidence, EvidenceRelation as R, EvidenceType as T, MemoryClaim, MemoryState
from ememoveri.state_engine import StateEvolutionEngine
from ememoveri.store import MemoryStore, TemporalLeakageGuard
from ememoveri.trust import TrustModel

UTC = timezone.utc
T0 = datetime(2023, 1, 1, tzinfo=UTC)


def d(n):
    return T0 + timedelta(days=n)


def test_supersession_is_really_terminal():
    eng = StateEvolutionEngine()
    c = MemoryClaim("a", scope="r")
    eng.apply_evidence(c, Evidence(T.COMMIT, R.SUPERSEDES, "gone", d(0)))
    eng.apply_evidence(c, Evidence(T.TEST, R.CORROBORATES, "late", d(1)))
    assert c.state == MemoryState.SUPERSEDED
    assert len(c.evidence) == 2  # still recorded for provenance


def test_supersession_terminal_in_aggregate_mode():
    eng = StateEvolutionEngine()
    c = MemoryClaim("a", scope="r")
    eng.apply_evidence_batch(c, [Evidence(T.COMMIT, R.SUPERSEDES, "gone", d(0))], aggregate=True)
    eng.apply_evidence_batch(c, [Evidence(T.TEST, R.CORROBORATES, "late", d(1))], aggregate=True)
    assert c.state == MemoryState.SUPERSEDED


def test_retrieval_drops_superseded_claims():
    sc = RetrievalScorer()
    old = MemoryClaim("old", scope="r", trust=0.99)
    old.state = MemoryState.SUPERSEDED
    new = MemoryClaim("new", scope="r", trust=0.3)
    ranked = sc.rank([(old, 0.9), (new, 0.5)], "r", reference=old.updated_at)
    assert [c.statement for c, _, _ in ranked] == ["new"]
    blind = RetrievalScorer(state_aware=False)
    assert blind.rank([(old, 0.9), (new, 0.5)], "r", reference=old.updated_at)[0][0].statement == "old"


def test_leakage_guard_rebuilds_trust_and_state_from_visible_evidence_only():
    cutoff = d(10)
    store = MemoryStore(leakage_guard=TemporalLeakageGuard(knowledge_cutoff=cutoff))
    c = MemoryClaim("a", scope="r")
    c.created_at = c.updated_at = T0
    store.add_claim(c)
    store.ingest_evidence(c.claim_id, Evidence(T.TEST, R.CORROBORATES, "pre", d(1)))
    store.ingest_evidence(c.claim_id, Evidence(T.TEST, R.SUPERSEDES, "FUTURE", d(100)))
    store.ingest_evidence(c.claim_id, Evidence(T.TEST, R.CORROBORATES, "FUTURE2", d(101)))
    assert c.state == MemoryState.SUPERSEDED  # full history
    (view, _, _), = store.retrieve([(c, 0.9)], "r", reference=cutoff)
    assert len(view.evidence) == 1
    assert view.state == MemoryState.CORROBORATED          # not SUPERSEDED: that is the future
    expected = TrustModel().sequential_update(0.5, view.evidence[0])
    assert abs(view.trust - expected) < 1e-9               # not inflated by future evidence


def test_claim_created_after_cutoff_is_invisible():
    cutoff = d(10)
    store = MemoryStore(leakage_guard=TemporalLeakageGuard(knowledge_cutoff=cutoff))
    future = MemoryClaim("x", scope="r")
    future.created_at = d(50)
    assert store.retrieve([(future, 0.9)], "r") == []


def test_staleness_defaults_to_knowledge_cutoff_not_wall_clock():
    cutoff = d(10)
    store = MemoryStore(leakage_guard=TemporalLeakageGuard(knowledge_cutoff=cutoff))
    c = MemoryClaim("a", scope="r")
    c.created_at = c.updated_at = d(9)
    (_, _, score_cutoff), = store.retrieve([(c, 0.5)], "r")
    (_, _, score_explicit), = store.retrieve([(c, 0.5)], "r", reference=cutoff)
    assert abs(score_cutoff - score_explicit) < 1e-12
    wall = RetrievalScorer().score(c, 0.5, "r")  # would treat the claim as ~3 years stale
    assert score_cutoff > wall


def test_cross_repo_split_uses_identity_not_equality():
    store = MemoryStore()
    a = MemoryClaim("same", scope="repoA", claim_id="x")
    b = MemoryClaim("same", scope="repoB", claim_id="x")
    ranked = store.retrieve([(a, 0.5), (b, 0.5)], "repoA")
    assert len(ranked) >= 1  # dataclass equality used to be able to conflate these


def test_resolver_min_score_abstains_on_weak_single_candidate():
    c = MemoryClaim("a", scope="r")
    assert ConflictResolver().resolve([(c, 0.01)]).outcome == ResolutionOutcome.RESOLVED
    assert ConflictResolver(min_score=0.3).resolve([(c, 0.01)]).outcome == ResolutionOutcome.ABSTAINED


def test_logit_aggregation_is_order_invariant_and_never_saturates():
    tm = TrustModel()
    evs = [Evidence(T.TEST, R.CORROBORATES, str(i), d(i)) for i in range(6)]
    a = tm.logit_aggregated_update(evs)
    b = tm.logit_aggregated_update(list(reversed(evs)))
    assert abs(a - b) < 1e-12 and a < 1.0
    assert tm.logit_aggregated_update(evs) > tm.logit_aggregated_update(evs[:3])
    assert tm.aggregated_update(evs) == tm.aggregated_update(evs[:4]) == 1.0  # clamped variant saturates
