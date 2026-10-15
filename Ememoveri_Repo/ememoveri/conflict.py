"""
Conflict Resolver: decides between competing memory claims, abstaining when
the evidence gap is too small to trust a decision.

This directly implements the project's abstention rule:

    if |score_1 - score_2| < gap_threshold: ABSTAIN
    else: return the higher-scoring claim

Default gap_threshold = 0.20, matching the blueprint.

The resolver also reports an "Abstention Precision" / "Coverage" pair so the
reliability/coverage trade-off (E5-N) can be evaluated once ground-truth
outcome labels exist.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Sequence

from .schema import MemoryClaim


class ResolutionOutcome(str, enum.Enum):
    RESOLVED = "RESOLVED"
    ABSTAINED = "ABSTAINED"


@dataclass
class Resolution:
    outcome: ResolutionOutcome
    winner: Optional[MemoryClaim]
    scores: dict  # claim_id -> score
    gap: float


@dataclass
class ConflictResolver:
    gap_threshold: float = 0.20
    min_score: Optional[float] = None  # abstain when even the best claim scores below this

    def resolve(self, scored_claims: Sequence[tuple]) -> Resolution:
        """`scored_claims`: list of (MemoryClaim, score) pairs, length >= 1.

        With a single candidate, resolution is trivial (no conflict to
        adjudicate) unless its own score is degenerate (handled by caller).
        With >= 2 candidates, only the top two scores determine abstention:
        a resolver should not refuse to act just because some low-ranked
        claim exists.
        """
        if not scored_claims:
            raise ValueError("resolve() requires at least one scored claim")

        ordered = sorted(scored_claims, key=lambda t: t[1], reverse=True)
        scores = {c.claim_id: s for c, s in ordered}

        if self.min_score is not None and ordered[0][1] < self.min_score:
            gap = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else float("inf")
            return Resolution(ResolutionOutcome.ABSTAINED, None, scores, gap=gap)

        if len(ordered) == 1:
            return Resolution(ResolutionOutcome.RESOLVED, ordered[0][0], scores, gap=float("inf"))

        top_claim, top_score = ordered[0]
        second_claim, second_score = ordered[1]
        gap = top_score - second_score

        if gap < self.gap_threshold:
            return Resolution(ResolutionOutcome.ABSTAINED, None, scores, gap=gap)

        return Resolution(ResolutionOutcome.RESOLVED, top_claim, scores, gap=gap)

    # --- evaluation once ground-truth correctness labels are available -----

    @staticmethod
    def abstention_precision(resolutions: Sequence[Resolution], was_ambiguous: Sequence[bool]) -> float:
        """Fraction of ABSTAIN decisions that were made on genuinely
        ambiguous cases (i.e. correct abstentions), given ground-truth labels
        of whether each case was actually ambiguous.

        abstention_precision = correct_abstentions / all_abstentions
        """
        abstentions = [(r, amb) for r, amb in zip(resolutions, was_ambiguous)
                       if r.outcome == ResolutionOutcome.ABSTAINED]
        if not abstentions:
            return float("nan")
        correct = sum(1 for _, amb in abstentions if amb)
        return correct / len(abstentions)

    @staticmethod
    def coverage(resolutions: Sequence[Resolution]) -> float:
        if not resolutions:
            return float("nan")
        resolved = sum(1 for r in resolutions if r.outcome == ResolutionOutcome.RESOLVED)
        return resolved / len(resolutions)
