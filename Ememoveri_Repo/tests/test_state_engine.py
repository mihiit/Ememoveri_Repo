import pytest
from datetime import datetime, timedelta, timezone

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType, MemoryClaim, MemoryState
from ememoveri.state_engine import StateEvolutionEngine


def _t(offset_seconds=0):
    return datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)


def make_claim(scope="pytorch/optim"):
    return MemoryClaim(statement="Adam optimizer uses decoupled weight decay", scope=scope)


def test_new_claim_is_provisional():
    claim = make_claim()
    assert claim.state == MemoryState.PROVISIONAL


def test_corroboration_moves_to_corroborated():
    engine = StateEvolutionEngine()
    claim = make_claim()
    e = Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "confirms behavior", _t())
    engine.apply_evidence(claim, e)
    assert claim.state == MemoryState.CORROBORATED


def test_contradiction_alone_moves_to_contested_when_strong():
    engine = StateEvolutionEngine()
    claim = make_claim()
    e = Evidence(EvidenceType.COMMIT, EvidenceRelation.CONTRADICTS, "this is wrong", _t(), strength=1.0)
    engine.apply_evidence(claim, e)
    assert claim.state == MemoryState.CONTESTED


def test_refinement_without_contradiction_is_refined():
    engine = StateEvolutionEngine()
    claim = make_claim()
    e = Evidence(EvidenceType.DOCUMENTATION, EvidenceRelation.REFINES, "adds nuance", _t())
    engine.apply_evidence(claim, e)
    assert claim.state == MemoryState.REFINED


def test_supersession_is_terminal_regardless_of_prior_state():
    engine = StateEvolutionEngine()
    claim = make_claim()
    engine.apply_evidence(claim, Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "ok", _t()))
    assert claim.state == MemoryState.CORROBORATED
    engine.apply_evidence(
        claim,
        Evidence(EvidenceType.COMMIT, EvidenceRelation.SUPERSEDES, "replaced by AdamW", _t(1),
                 metadata={"superseding_claim_id": "abc123"}),
    )
    assert claim.state == MemoryState.SUPERSEDED
    assert claim.superseded_by == "abc123"
    assert claim.state.is_terminal


def test_balanced_support_and_contradiction_is_contested():
    engine = StateEvolutionEngine()
    claim = make_claim()
    # Use matching evidence types (equal reliability) so the two items are
    # genuinely balanced in weight, isolating the "balance" condition itself
    # rather than confounding it with a reliability difference between types.
    engine.apply_evidence(claim, Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "support", _t(), strength=1.0))
    engine.apply_evidence(claim, Evidence(EvidenceType.COMMIT, EvidenceRelation.CONTRADICTS, "dispute", _t(1), strength=1.0))
    assert claim.state == MemoryState.CONTESTED


def test_ambiguous_weak_evidence_stays_provisional():
    engine = StateEvolutionEngine(corroborate_threshold=0.5)
    claim = make_claim()
    engine.apply_evidence(claim, Evidence(EvidenceType.AGENT_OBSERVATION, EvidenceRelation.CORROBORATES,
                                           "weak signal", _t(), strength=0.1))
    assert claim.state == MemoryState.PROVISIONAL


def test_contested_can_recover_to_corroborated_with_dominant_new_evidence():
    engine = StateEvolutionEngine(contest_margin=0.1)
    claim = make_claim()
    engine.apply_evidence(claim, Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "a", _t(), strength=1.0))
    engine.apply_evidence(claim, Evidence(EvidenceType.COMMIT, EvidenceRelation.CONTRADICTS, "b", _t(1), strength=1.0))
    assert claim.state == MemoryState.CONTESTED
    # Strong new corroboration should push net_support decisively positive
    engine.apply_evidence(claim, Evidence(EvidenceType.TEST, EvidenceRelation.CORROBORATES, "c", _t(2), strength=2.0))
    assert claim.state == MemoryState.CORROBORATED


def test_deterministic_given_same_evidence_sequence():
    engine1, engine2 = StateEvolutionEngine(), StateEvolutionEngine()
    claim1, claim2 = make_claim(), make_claim()
    events = [
        Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "x", _t(0), strength=0.8),
        Evidence(EvidenceType.TEST, EvidenceRelation.REFINES, "y", _t(1), strength=0.5),
    ]
    for e in events:
        engine1.apply_evidence(claim1, e)
    for e in events:
        engine2.apply_evidence(claim2, e)
    assert claim1.state == claim2.state
    assert claim1.net_support() == pytest.approx(claim2.net_support())
