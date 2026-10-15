"""
State Evolution Engine: PROVISIONAL / CORROBORATED / REFINED / CONTESTED / SUPERSEDED.

Transition logic (deterministic, documented explicitly so E1 results are
reproducible from first principles rather than emergent from tuning):

  - Start: every new claim is PROVISIONAL.
  - SUPERSEDES evidence -> SUPERSEDED (terminal), regardless of current state.
  - Otherwise, compute net_support = sum(signed_weight) over CORROBORATES /
    CONTRADICTS / REFINES evidence (REFINES counts as positive support but is
    tracked separately to allow REFINED as a distinguishable end state).
  - If there is BOTH contradicting and corroborating/refining evidence whose
    magnitudes are within `contest_margin` of each other -> CONTESTED.
  - Else if net_support > 0 and there is at least one REFINES item and no
    CONTRADICTS items -> REFINED.
  - Else if net_support > `corroborate_threshold` -> CORROBORATED.
  - Else if net_support < -`corroborate_threshold` -> CONTESTED (evidence
    is one-sidedly negative, not enough to supersede but too much to ignore).
  - Else -> PROVISIONAL (insufficient/ambiguous evidence).

CONTESTED can recover to CORROBORATED if subsequent evidence shifts
net_support decisively positive and the contest margin condition no longer
holds — i.e. the lattice is not a strict DAG traversal, matching the
project's stated design ("CONTESTED ... contradictory ... refined").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .schema import Evidence, EvidenceRelation, MemoryClaim, MemoryState


@dataclass
class StateEvolutionEngine:
    corroborate_threshold: float = 0.15
    contest_margin: float = 0.20  # how close pos/neg evidence must be to count as "contested"

    def apply_evidence(self, claim: MemoryClaim, evidence: Evidence) -> MemoryClaim:
        """Apply a single evidence item to a claim, mutating and returning it.

        This is the SEQUENTIAL update path (order-sensitive by design; see
        trust.py). For the order-invariant path used in the RQ3 ablation,
        use `apply_evidence_batch(..., aggregate=True)`.
        """
        claim.evidence.append(evidence)
        claim.updated_at = max(claim.updated_at, evidence.timestamp)

        # SUPERSEDED is terminal: late evidence is still recorded for
        # provenance, but can never resurrect the claim.
        if claim.state == MemoryState.SUPERSEDED:
            return claim

        if evidence.relation == EvidenceRelation.SUPERSEDES:
            claim.state = MemoryState.SUPERSEDED
            claim.superseded_by = evidence.metadata.get("superseding_claim_id")
            return claim

        claim.state = self._resolve_state(claim)
        return claim

    def apply_evidence_batch(self, claim: MemoryClaim, evidence_items, aggregate: bool = False) -> MemoryClaim:
        """Apply a sequence of evidence items.

        aggregate=False (default): apply one at a time, sequentially
        (order-sensitive — matches the original mechanism under test in E2).

        aggregate=True: append all evidence first, then resolve state ONCE
        from the pooled evidence set (order-invariant — the RQ3 mitigation).
        """
        supersede = next((e for e in evidence_items if e.relation == EvidenceRelation.SUPERSEDES), None)
        if not aggregate:
            for e in evidence_items:
                self.apply_evidence(claim, e)
            return claim

        for e in evidence_items:
            claim.evidence.append(e)
        if evidence_items:
            claim.updated_at = max(claim.updated_at, max(e.timestamp for e in evidence_items))
        if claim.state == MemoryState.SUPERSEDED:
            return claim  # terminal
        if supersede is not None:
            claim.state = MemoryState.SUPERSEDED
            claim.superseded_by = supersede.metadata.get("superseding_claim_id")
        else:
            claim.state = self._resolve_state(claim)
        return claim

    def _resolve_state(self, claim: MemoryClaim) -> MemoryState:
        contradicts = claim.evidence_by_relation(EvidenceRelation.CONTRADICTS)
        corroborates = claim.evidence_by_relation(EvidenceRelation.CORROBORATES)
        refines = claim.evidence_by_relation(EvidenceRelation.REFINES)

        pos_weight = sum(e.signed_weight for e in corroborates + refines)
        neg_weight = sum(abs(e.signed_weight) for e in contradicts)
        net_support = pos_weight - neg_weight

        if contradicts and (corroborates or refines):
            if abs(pos_weight - neg_weight) <= self.contest_margin:
                return MemoryState.CONTESTED

        if contradicts and not corroborates and not refines:
            if neg_weight > self.corroborate_threshold:
                return MemoryState.CONTESTED

        if net_support > self.corroborate_threshold:
            if refines and not contradicts:
                return MemoryState.REFINED
            return MemoryState.CORROBORATED

        if net_support < -self.corroborate_threshold:
            return MemoryState.CONTESTED

        return MemoryState.PROVISIONAL

    def trust_delta_summary(self, claim: MemoryClaim) -> dict:
        """Diagnostic summary used by E1/E2 assertions and logging."""
        contradicts = claim.evidence_by_relation(EvidenceRelation.CONTRADICTS)
        corroborates = claim.evidence_by_relation(EvidenceRelation.CORROBORATES)
        refines = claim.evidence_by_relation(EvidenceRelation.REFINES)
        return {
            "state": claim.state.value,
            "n_corroborates": len(corroborates),
            "n_contradicts": len(contradicts),
            "n_refines": len(refines),
            "net_support": claim.net_support(),
            "evidence_strength_total": claim.evidence_strength_total(),
        }
