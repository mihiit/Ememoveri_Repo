"""
Trust model: converts accumulated evidence into a calibrated trust score.

The model supports two update modes:

  1. Sequential (order-dependent) — an additive update applied in
     probability space, clamped to [0, 1] after EVERY step. Clamping at
     each step is what introduces order sensitivity: once trust saturates,
     further "excess" evidence in the same direction is discarded, so
     applying evidence A then B need not equal B then A.

  2. Aggregated (order-invariant) — evidence is pooled (summed) first, and
     the clamp is applied exactly once at the end. Because addition is
     commutative, this is provably invariant to the order in which evidence
     items were *observed* (not to their timestamps or content).

Both are provided because the project's own findings (E2) show the
sequential update is order-sensitive and the aggregated variant removes that
sensitivity — this is a first-class research result, not an implementation
detail to be hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import Evidence, EvidenceRelation


@dataclass
class TrustModel:
    """Bounded, calibratable trust update.

    Parameters
    ----------
    learning_rate:
        Step size for the sequential update, applied in logit space so trust
        stays in (0, 1) without clipping artifacts.
    prior_trust:
        Trust assigned to a claim with no evidence at all.
    """

    learning_rate: float = 0.35
    prior_trust: float = 0.5

    def sequential_update(self, current_trust: float, evidence: Evidence) -> float:
        """Apply ONE evidence item to the current trust, directly in
        probability space, with clamping to [0, 1] after every step:

            trust_{t+1} = clip(trust_t + lr * signed_weight, 0, 1)

        This is intentionally order-sensitive: once trust saturates at a
        boundary (0 or 1), the clamp discards any "excess" update, so
        applying evidence A then B need not equal applying B then A. This
        is what actually produces the order-sensitivity characterized in
        E2 — note that an equivalent *logit-space* additive update would be
        exactly order-invariant (logit/sigmoid are mutual inverses, so the
        per-step deltas simply telescope), so probability-space clamping is
        the mechanism under test here, not an incidental choice.
        """
        if evidence.relation == EvidenceRelation.SUPERSEDES:
            # Superseding evidence is a hard state transition handled by the
            # state engine; it does not participate in trust arithmetic here.
            return current_trust
        updated = current_trust + self.learning_rate * evidence.signed_weight
        return min(1.0, max(0.0, updated))

    def sequential_update_all(self, evidence_sequence: Sequence[Evidence]) -> float:
        trust = self.prior_trust
        for e in evidence_sequence:
            trust = self.sequential_update(trust, e)
        return trust

    def aggregated_update(self, evidence_items: Iterable[Evidence]) -> float:
        """Order-invariant update: pool signed weights first, then apply
        the sum ONCE, with a single clamp at the end:

            trust = clip(prior_trust + lr * sum(signed_weight), 0, 1)

        Because addition is commutative and the clamp is applied exactly
        once (rather than once per step, as in `sequential_update`), the
        result does not depend on the order in which `evidence_items` is
        iterated — this is the order-invariant aggregation variant
        referenced in RQ3 / E2, and it removes the per-step saturation
        artifact that makes the sequential variant order-sensitive.
        """
        total = sum(
            e.signed_weight for e in evidence_items
            if e.relation != EvidenceRelation.SUPERSEDES
        )
        updated = self.prior_trust + self.learning_rate * total
        return min(1.0, max(0.0, updated))

    def logit_aggregated_update(self, evidence_items: Iterable[Evidence]) -> float:
        """Order-invariant AND non-saturating update.

            trust = sigmoid(logit(prior) + lr * sum(signed_weight))

        Pooling is commutative (order-invariant) and the sigmoid never hits
        the 0/1 boundary, so additional evidence always moves trust a little
        instead of being discarded by a clamp. Added because the clamped
        variants saturate after only two or three strong items at the
        default learning rate, which destroys the ranking signal.
        """
        total = sum(
            e.signed_weight for e in evidence_items
            if e.relation != EvidenceRelation.SUPERSEDES
        )
        p0 = min(1.0 - 1e-6, max(1e-6, self.prior_trust))
        z = math.log(p0 / (1.0 - p0)) + self.learning_rate * total
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def dispersion(trust_values: Sequence[float]) -> dict:
        """Summary statistics used by the E2 order-sensitivity experiment."""
        if not trust_values:
            return {"range": 0.0, "stdev": 0.0, "n": 0}
        n = len(trust_values)
        mean = sum(trust_values) / n
        var = sum((v - mean) ** 2 for v in trust_values) / n
        return {
            "range": max(trust_values) - min(trust_values),
            "stdev": math.sqrt(var),
            "mean": mean,
            "n": n,
        }

    # --- Calibration diagnostics (for an E5-O-style trust calibration study) ---

    @staticmethod
    def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
        """Standard ECE over (confidence, correctness) pairs.

        This is provided so trust scores can be validated empirically once
        outcome labels exist (e.g. from E5 task success) rather than merely
        asserted to be meaningful.
        """
        if len(confidences) != len(correct) or not confidences:
            raise ValueError("confidences and correct must be same non-empty length")
        bins = [[] for _ in range(n_bins)]
        for c, y in zip(confidences, correct):
            idx = min(int(c * n_bins), n_bins - 1)
            bins[idx].append((c, y))
        n = len(confidences)
        ece = 0.0
        for b in bins:
            if not b:
                continue
            conf_avg = sum(c for c, _ in b) / len(b)
            acc_avg = sum(1.0 for _, y in b if y) / len(b)
            ece += (len(b) / n) * abs(conf_avg - acc_avg)
        return ece

    @staticmethod
    def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
        if len(confidences) != len(correct) or not confidences:
            raise ValueError("confidences and correct must be same non-empty length")
        return sum((c - (1.0 if y else 0.0)) ** 2 for c, y in zip(confidences, correct)) / len(confidences)
