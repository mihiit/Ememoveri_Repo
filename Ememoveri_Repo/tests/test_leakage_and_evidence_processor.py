from datetime import datetime, timedelta, timezone

import pytest

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType, MemoryClaim
from ememoveri.store import TemporalLeakageGuard, TemporalLeakageError
from ememoveri.evidence_processor import EvidenceProcessor


def _t(offset_days=0):
    return datetime.now(timezone.utc) + timedelta(days=offset_days)


def test_leakage_guard_filters_future_evidence():
    cutoff = _t(0)
    claim = MemoryClaim(statement="s", scope="repoA")
    claim.evidence = [
        Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "past", _t(-5)),
        Evidence(EvidenceType.COMMIT, EvidenceRelation.CONTRADICTS, "future", _t(5)),
    ]
    guard = TemporalLeakageGuard(knowledge_cutoff=cutoff)
    filtered = guard.filter_claim(claim)
    assert len(filtered.evidence) == 1
    assert filtered.evidence[0].content == "past"


def test_leakage_guard_assert_raises_on_violation():
    cutoff = _t(0)
    claim = MemoryClaim(statement="s", scope="repoA")
    claim.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "future", _t(1))]
    guard = TemporalLeakageGuard(knowledge_cutoff=cutoff)
    with pytest.raises(TemporalLeakageError):
        guard.assert_no_leakage(claim)


def test_leakage_guard_noop_without_cutoff():
    claim = MemoryClaim(statement="s", scope="repoA")
    claim.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "x", _t(100))]
    guard = TemporalLeakageGuard(knowledge_cutoff=None)
    assert guard.filter_claim(claim) is claim
    guard.assert_no_leakage(claim)  # should not raise


def test_evidence_processor_classifies_supersession_over_contradiction():
    proc = EvidenceProcessor()
    ev = proc.from_commit("replace the buggy implementation with a new API", _t(), "sha1")
    assert ev.relation == EvidenceRelation.SUPERSEDES


def test_evidence_processor_classifies_contradiction():
    proc = EvidenceProcessor()
    ev = proc.from_commit("fix incorrect handling of null values", _t(), "sha2")
    assert ev.relation == EvidenceRelation.CONTRADICTS


def test_evidence_processor_default_relation_classifier_override():
    proc = EvidenceProcessor(default_relation_classifier=lambda msg: EvidenceRelation.REFINES)
    ev = proc.from_commit("anything at all", _t(), "sha3")
    assert ev.relation == EvidenceRelation.REFINES


def test_evidence_processor_test_result_corroborates_when_passing_and_matching():
    proc = EvidenceProcessor()
    ev = proc.from_test_result("test_forward_pass", passed=True, timestamp=_t(), related_statement_matches=True)
    assert ev.relation == EvidenceRelation.CORROBORATES


def test_evidence_processor_test_result_contradicts_when_failing():
    proc = EvidenceProcessor()
    ev = proc.from_test_result("test_forward_pass", passed=False, timestamp=_t(), related_statement_matches=True)
    assert ev.relation == EvidenceRelation.CONTRADICTS
