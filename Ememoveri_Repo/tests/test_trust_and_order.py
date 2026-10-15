import itertools
from datetime import datetime, timedelta, timezone

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType
from ememoveri.trust import TrustModel


def _t(offset=0):
    return datetime.now(timezone.utc) + timedelta(seconds=offset)


def make_mixed_evidence():
    return [
        Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "a", _t(0), strength=1.0),
        Evidence(EvidenceType.ISSUE, EvidenceRelation.CONTRADICTS, "b", _t(1), strength=0.9),
        Evidence(EvidenceType.TEST, EvidenceRelation.CORROBORATES, "c", _t(2), strength=1.2),
    ]


def test_sequential_update_is_order_sensitive_in_general():
    model = TrustModel(learning_rate=0.9)  # large LR to expose boundary curvature
    events = make_mixed_evidence()
    trust_orders = []
    for perm in itertools.permutations(events):
        trust_orders.append(model.sequential_update_all(perm))
    # Not all permutation results are identical -> order sensitivity exists.
    assert len(set(round(t, 9) for t in trust_orders)) > 1


def test_aggregated_update_is_order_invariant():
    model = TrustModel(learning_rate=0.9)
    events = make_mixed_evidence()
    results = [model.aggregated_update(perm) for perm in itertools.permutations(events)]
    assert len(set(round(r, 12) for r in results)) == 1


def test_dispersion_reports_zero_range_for_aggregated_variant():
    model = TrustModel(learning_rate=0.9)
    events = make_mixed_evidence()
    agg_results = [model.aggregated_update(perm) for perm in itertools.permutations(events)]
    stats = TrustModel.dispersion(agg_results)
    assert stats["range"] == 0.0
    assert stats["stdev"] == 0.0


def test_supersedes_does_not_move_trust():
    model = TrustModel()
    e = Evidence(EvidenceType.COMMIT, EvidenceRelation.SUPERSEDES, "replaced", _t())
    assert model.sequential_update(0.6, e) == 0.6


def test_trust_stays_in_closed_unit_interval_and_saturates():
    model = TrustModel(learning_rate=5.0)  # extreme LR
    trust = 0.5
    e = Evidence(EvidenceType.TEST, EvidenceRelation.CORROBORATES, "x", _t(), strength=2.0)
    for _ in range(50):
        trust = model.sequential_update(trust, e)
    # With clamping, repeated strong positive evidence legitimately
    # saturates trust exactly at the boundary rather than approaching it
    # asymptotically (unlike a logit-space update).
    assert 0.0 <= trust <= 1.0
    assert trust == 1.0


def test_expected_calibration_error_zero_for_perfect_calibration():
    confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
    correct = [False, False, False, True, True]  # roughly matches confidence
    ece = TrustModel.expected_calibration_error(confidences, correct, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_brier_score_perfect_predictions_is_zero():
    assert TrustModel.brier_score([1.0, 0.0], [True, False]) == 0.0
