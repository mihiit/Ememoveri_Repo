"""
Evidence Processor: extracts, classifies, and timestamps raw repository
signals (commit metadata, test results, doc diffs, issue events) into
`Evidence` objects that the rest of the system can consume.

This module defines the INTERFACE and a reference implementation for
structured inputs (dicts already extracted by a Git-mining tool such as
PyDriller). It deliberately does not perform its own Git mining — E5's
repository-mining step is a separate, environment-specific pipeline (see
experiments/README.md) that must run against real cloned repositories.

Keeping this boundary explicit matters for the temporal-leakage guarantee:
`classify_commit_event` never looks at anything the caller doesn't hand it,
so leakage control is enforced by whoever builds the input dict (the E5
temporal-screening step), not silently inside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .schema import Evidence, EvidenceRelation, EvidenceType


RELATION_KEYWORDS = {
    EvidenceRelation.CONTRADICTS: (
        "revert", "fix", "incorrect", "bug", "broken", "regression", "wrong",
    ),
    EvidenceRelation.SUPERSEDES: (
        "replace", "deprecate", "remove", "migrate", "supersede", "rewrite",
    ),
    EvidenceRelation.REFINES: (
        "refactor", "improve", "clarify", "optimize", "extend", "generalize",
    ),
    EvidenceRelation.CORROBORATES: (
        "add", "implement", "support", "confirm", "validate",
    ),
}


@dataclass
class EvidenceProcessor:
    """Turns structured repository events into Evidence objects.

    `relation_classifier` may be overridden with a model-based classifier
    (e.g. an LLM prompt or a fine-tuned text classifier); the default here is
    a transparent keyword heuristic used for controlled experiments (E1-E4)
    where ground-truth relations are already known and this heuristic is not
    load-bearing for correctness.
    """

    default_relation_classifier: Optional[callable] = None

    def classify_relation(self, message: str) -> EvidenceRelation:
        if self.default_relation_classifier is not None:
            return self.default_relation_classifier(message)
        text = message.lower()
        # Priority order matters: supersession keywords dominate contradiction
        # keywords (e.g. "replace the buggy implementation").
        for relation in (
            EvidenceRelation.SUPERSEDES,
            EvidenceRelation.CONTRADICTS,
            EvidenceRelation.REFINES,
            EvidenceRelation.CORROBORATES,
        ):
            if any(kw in text for kw in RELATION_KEYWORDS[relation]):
                return relation
        return EvidenceRelation.CORROBORATES  # conservative default

    def from_commit(
        self,
        message: str,
        timestamp: datetime,
        commit_sha: str,
        relation: Optional[EvidenceRelation] = None,
        strength: float = 1.0,
    ) -> Evidence:
        return Evidence(
            evidence_type=EvidenceType.COMMIT,
            relation=relation or self.classify_relation(message),
            content=message,
            timestamp=timestamp,
            source_id=commit_sha,
            strength=strength,
            metadata={"commit_sha": commit_sha},
        )

    def from_test_result(
        self,
        test_name: str,
        passed: bool,
        timestamp: datetime,
        related_statement_matches: bool,
        strength: float = 1.0,
    ) -> Evidence:
        """A passing test that matches a claim corroborates it; a failing
        test that was expected to validate the claim contradicts it."""
        relation = (
            EvidenceRelation.CORROBORATES
            if (passed and related_statement_matches)
            else EvidenceRelation.CONTRADICTS
        )
        return Evidence(
            evidence_type=EvidenceType.TEST,
            relation=relation,
            content=f"{test_name}: {'PASSED' if passed else 'FAILED'}",
            timestamp=timestamp,
            source_id=test_name,
            strength=strength,
            metadata={"passed": passed},
        )

    def from_documentation(
        self, text: str, timestamp: datetime, doc_path: str,
        relation: Optional[EvidenceRelation] = None, strength: float = 0.8,
    ) -> Evidence:
        return Evidence(
            evidence_type=EvidenceType.DOCUMENTATION,
            relation=relation or self.classify_relation(text),
            content=text,
            timestamp=timestamp,
            source_id=doc_path,
            strength=strength,
            metadata={"doc_path": doc_path},
        )

    def from_issue(
        self, text: str, timestamp: datetime, issue_id: str,
        resolved: bool = False, relation: Optional[EvidenceRelation] = None,
        strength: float = 0.6,
    ) -> Evidence:
        return Evidence(
            evidence_type=EvidenceType.ISSUE,
            relation=relation or self.classify_relation(text),
            content=text,
            timestamp=timestamp,
            source_id=issue_id,
            strength=strength,
            metadata={"issue_id": issue_id, "resolved": resolved},
        )

    def from_agent_observation(
        self, text: str, timestamp: datetime, relation: EvidenceRelation,
        confidence: float = 0.5,
    ) -> Evidence:
        return Evidence(
            evidence_type=EvidenceType.AGENT_OBSERVATION,
            relation=relation,
            content=text,
            timestamp=timestamp,
            strength=confidence,
        )
