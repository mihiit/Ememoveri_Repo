"""
Evidence-aware retrieval scorer.

score = w_sim * similarity
      + w_trust * trust
      + w_evidence * normalized_evidence_strength
      + w_scope * scope_match
      - w_staleness * staleness

Default weights match the project blueprint (0.45 / 0.25 / 0.20 / 0.10 / 0.10)
and are explicitly documented as controlled configuration choices, not a
claimed-optimal result — the weights themselves are an ablation target
(E5-G / E5-J / E5-K vary sub-terms of this score).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

from .schema import MemoryClaim, MemoryState


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RetrievalWeights:
    similarity: float = 0.45
    trust: float = 0.25
    evidence: float = 0.20
    scope: float = 0.10
    staleness: float = 0.10  # subtracted

    def validate(self) -> None:
        total_positive = self.similarity + self.trust + self.evidence + self.scope
        if not math.isclose(total_positive, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Positive weights (similarity+trust+evidence+scope) must sum to 1.0, got {total_positive}"
            )


@dataclass
class RetrievalScorer:
    weights: RetrievalWeights = None  # type: ignore
    staleness_half_life_days: float = 90.0
    evidence_saturation: float = 5.0  # evidence_strength_total at which normalized evidence term saturates near 1
    state_aware: bool = True  # rank() drops SUPERSEDED claims (terminal, known-invalid)

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = RetrievalWeights()
        self.weights.validate()

    # --- component terms -------------------------------------------------

    def staleness(self, claim: MemoryClaim, reference: Optional[datetime] = None) -> float:
        """Exponential decay in [0, 1); 0 = fresh, approaches 1 as age grows.

        staleness = 1 - 2^(-age_days / half_life)
        """
        age_days = claim.age_seconds(reference) / 86400.0
        return 1.0 - math.pow(2.0, -age_days / self.staleness_half_life_days)

    def normalized_evidence(self, claim: MemoryClaim) -> float:
        """Squash accumulated |evidence weight| into [0, 1)."""
        total = claim.evidence_strength_total()
        return total / (total + self.evidence_saturation)

    def scope_match(self, claim: MemoryClaim, query_scope: str) -> float:
        """1.0 exact scope match, 0.5 parent/child scope, 0.0 unrelated.

        Scope strings are expected in "repo" or "repo/subsystem" form.
        """
        if claim.scope == query_scope:
            return 1.0
        claim_root = claim.scope.split("/", 1)[0]
        query_root = query_scope.split("/", 1)[0]
        if claim_root == query_root:
            return 0.5
        return 0.0

    # --- main scoring ------------------------------------------------------

    def score(
        self,
        claim: MemoryClaim,
        similarity: float,
        query_scope: str,
        reference: Optional[datetime] = None,
    ) -> float:
        if not (0.0 <= similarity <= 1.0):
            raise ValueError("similarity must be in [0, 1]")
        w = self.weights
        s_sim = similarity
        s_trust = claim.trust
        s_evidence = self.normalized_evidence(claim)
        s_scope = self.scope_match(claim, query_scope)
        s_stale = self.staleness(claim, reference)

        return (
            w.similarity * s_sim
            + w.trust * s_trust
            + w.evidence * s_evidence
            + w.scope * s_scope
            - w.staleness * s_stale
        )

    def score_breakdown(self, claim: MemoryClaim, similarity: float, query_scope: str,
                         reference: Optional[datetime] = None) -> dict:
        w = self.weights
        s_evidence = self.normalized_evidence(claim)
        s_scope = self.scope_match(claim, query_scope)
        s_stale = self.staleness(claim, reference)
        return {
            "claim_id": claim.claim_id,
            "similarity_term": w.similarity * similarity,
            "trust_term": w.trust * claim.trust,
            "evidence_term": w.evidence * s_evidence,
            "scope_term": w.scope * s_scope,
            "staleness_penalty": w.staleness * s_stale,
            "total": self.score(claim, similarity, query_scope, reference),
        }

    def rank(
        self,
        candidates: Sequence[tuple],  # list[(MemoryClaim, similarity)]
        query_scope: str,
        top_k: Optional[int] = None,
        reference: Optional[datetime] = None,
    ) -> list:
        """Return candidates sorted by score, highest first.

        `candidates` is a list of (MemoryClaim, similarity) pairs — similarity
        is assumed to already be computed upstream (e.g. embedding cosine
        similarity) since EmemoVeri's contribution is the re-ranking/scoring
        layer, not the embedding model itself.
        """
        scored = [
            (claim, sim, self.score(claim, sim, query_scope, reference))
            for claim, sim in candidates
            if not (self.state_aware and claim.state == MemoryState.SUPERSEDED)
        ]
        scored.sort(key=lambda t: t[2], reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return scored
