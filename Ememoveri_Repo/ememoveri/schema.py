"""
Core data model for EmemoVeri: Evidence and MemoryClaim.

Design notes
------------
A MemoryClaim is deliberately NOT just a piece of retrieved text. It is a
structured, evidence-backed statement about the engineering state of a
repository, carrying:

  - a natural-language `statement` (what is believed to be true)
  - a `state` in the memory-state lattice (see MemoryState)
  - a `trust` score in [0, 1]
  - an ordered list of `Evidence` items supporting or contradicting it
  - `scope` metadata (which repository / subsystem it applies to)
  - timestamps for staleness computation

This module has zero external dependencies so it can be reused inside an
agent harness, a notebook, or a test suite without pulling in the rest of
the stack.
"""

from __future__ import annotations

import enum
import itertools
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EvidenceType(str, enum.Enum):
    """Provenance category of a piece of evidence.

    Ordering below encodes a *default* prior-reliability ranking used by
    :class:`~ememoveri.trust.TrustModel` when no repository-specific
    calibration is available. This ranking is a configurable default, not a
    claimed universal truth about evidence reliability.
    """

    TEST = "TEST"                          # executable, verifiable
    COMMIT = "COMMIT"                      # authoritative code change
    DOCUMENTATION = "DOCUMENTATION"        # maintained but can drift
    ISSUE = "ISSUE"                        # discussion, may be resolved/stale
    AGENT_OBSERVATION = "AGENT_OBSERVATION"  # agent's own inference
    EXTERNAL = "EXTERNAL"                  # outside the repository entirely

    @property
    def default_prior_reliability(self) -> float:
        return {
            EvidenceType.TEST: 0.95,
            EvidenceType.COMMIT: 0.90,
            EvidenceType.DOCUMENTATION: 0.70,
            EvidenceType.ISSUE: 0.55,
            EvidenceType.AGENT_OBSERVATION: 0.40,
            EvidenceType.EXTERNAL: 0.35,
        }[self]


class EvidenceRelation(str, enum.Enum):
    """How a new piece of evidence relates to an existing claim."""

    CORROBORATES = "CORROBORATES"
    CONTRADICTS = "CONTRADICTS"
    REFINES = "REFINES"          # narrows / adds nuance without contradiction
    SUPERSEDES = "SUPERSEDES"    # explicitly replaces the claim
    UNRELATED = "UNRELATED"


class MemoryState(str, enum.Enum):
    """Node in the memory-state lattice.

    Transitions are governed by :class:`~ememoveri.state_engine.StateEvolutionEngine`
    and are intentionally NOT a simple linear progression: CONTESTED and
    SUPERSEDED can be reached from any state, and CONTESTED can recover back
    to CORROBORATED if new corroborating evidence dominates.
    """

    PROVISIONAL = "PROVISIONAL"
    CORROBORATED = "CORROBORATED"
    REFINED = "REFINED"
    CONTESTED = "CONTESTED"
    SUPERSEDED = "SUPERSEDED"

    @property
    def is_terminal(self) -> bool:
        return self is MemoryState.SUPERSEDED


@dataclass
class Evidence:
    """A single piece of evidence bearing on a memory claim."""

    evidence_type: EvidenceType
    relation: EvidenceRelation
    content: str
    timestamp: datetime = field(default_factory=_utcnow)
    source_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    strength: float = 1.0  # relative weight of this specific evidence item, in [0, 2]
    reliability_override: Optional[float] = None  # bypass EvidenceType default
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.strength <= 2.0):
            raise ValueError(f"Evidence.strength must be in [0, 2], got {self.strength}")
        if self.reliability_override is not None and not (0.0 <= self.reliability_override <= 1.0):
            raise ValueError("reliability_override must be in [0, 1]")

    @property
    def reliability(self) -> float:
        if self.reliability_override is not None:
            return self.reliability_override
        return self.evidence_type.default_prior_reliability

    @property
    def signed_weight(self) -> float:
        """Positive for corroborating/refining evidence, negative for contradicting.

        SUPERSEDES is treated separately by the state engine (it does not
        participate in the additive trust update — it is a hard transition).
        """
        sign = {
            EvidenceRelation.CORROBORATES: 1.0,
            EvidenceRelation.REFINES: 1.0,
            EvidenceRelation.CONTRADICTS: -1.0,
            EvidenceRelation.SUPERSEDES: 0.0,
            EvidenceRelation.UNRELATED: 0.0,
        }[self.relation]
        return sign * self.strength * self.reliability


@dataclass
class MemoryClaim:
    """An evidence-backed, evolving statement about repository knowledge."""

    statement: str
    scope: str  # e.g. repository name, or "repo/subsystem"
    claim_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: MemoryState = MemoryState.PROVISIONAL
    trust: float = 0.5
    evidence: list = field(default_factory=list)  # list[Evidence]
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    superseded_by: Optional[str] = None  # claim_id of the superseding claim
    tags: tuple = field(default_factory=tuple)

    def evidence_strength_total(self) -> float:
        """Sum of absolute evidence weight — used by the retrieval scorer's
        'evidence' term, independent of direction."""
        return sum(abs(e.signed_weight) for e in self.evidence)

    def net_support(self) -> float:
        return sum(e.signed_weight for e in self.evidence)

    def age_seconds(self, reference: Optional[datetime] = None) -> float:
        ref = reference or _utcnow()
        return max(0.0, (ref - self.updated_at).total_seconds())

    def evidence_by_relation(self, relation: EvidenceRelation) -> list:
        return [e for e in self.evidence if e.relation == relation]

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "scope": self.scope,
            "state": self.state.value,
            "trust": round(self.trust, 6),
            "evidence_count": len(self.evidence),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "superseded_by": self.superseded_by,
            "tags": list(self.tags),
        }


def new_claim_id() -> str:
    return uuid.uuid4().hex[:12]


def event_permutations(events: list, limit: Optional[int] = None):
    """Yield permutations of an evidence-event sequence.

    Used by the E2 order-sensitivity experiment. For sequences longer than 8
    items, `limit` should be set (random sampling is used instead of full
    enumeration) — see experiments/e2_order_sensitivity.py.
    """
    for perm in itertools.permutations(events):
        yield perm
        if limit is not None:
            limit -= 1
            if limit <= 0:
                return
