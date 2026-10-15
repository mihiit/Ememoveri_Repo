"""
Reproduces the E1 "state transition validity" experiment structure described
in the project blueprint: 9 scenarios covering corroboration, contradiction,
refinement, supersession, provisional/no-evidence, clear conflict, ambiguous
conflict, balanced support/contradiction, and stale-vs-recent evidence.

Run:  python -m experiments.e1_state_transitions
"""

from datetime import datetime, timedelta, timezone

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType, MemoryClaim
from ememoveri.state_engine import StateEvolutionEngine


def _t(days=0):
    return datetime.now(timezone.utc) + timedelta(days=days)


def run_scenario(name, evidence_items, corroborate_threshold=0.15, contest_margin=0.20):
    engine = StateEvolutionEngine(corroborate_threshold=corroborate_threshold, contest_margin=contest_margin)
    claim = MemoryClaim(statement=f"scenario: {name}", scope="testrepo")
    for e in evidence_items:
        engine.apply_evidence(claim, e)
    return name, claim.state.value, engine.trust_delta_summary(claim)


SCENARIOS = [
    ("corroboration", [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "confirmed", _t())]),
    ("contradiction", [Evidence(EvidenceType.COMMIT, EvidenceRelation.CONTRADICTS, "wrong", _t())]),
    ("refinement", [Evidence(EvidenceType.DOCUMENTATION, EvidenceRelation.REFINES, "nuance", _t())]),
    ("supersession", [Evidence(EvidenceType.COMMIT, EvidenceRelation.SUPERSEDES, "replaced", _t(),
                                metadata={"superseding_claim_id": "new-claim"})]),
    ("no_evidence_provisional", []),
    ("clear_conflict", [
        Evidence(EvidenceType.TEST, EvidenceRelation.CONTRADICTS, "test fails", _t(), strength=1.5),
    ]),
    ("ambiguous_conflict", [
        Evidence(EvidenceType.AGENT_OBSERVATION, EvidenceRelation.CORROBORATES, "weak yes", _t(), strength=0.1),
        Evidence(EvidenceType.AGENT_OBSERVATION, EvidenceRelation.CONTRADICTS, "weak no", _t(1), strength=0.1),
    ]),
    ("balanced_support_contradiction", [
        Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "support", _t(), strength=1.0),
        Evidence(EvidenceType.COMMIT, EvidenceRelation.CONTRADICTS, "dispute", _t(1), strength=1.0),
    ]),
    ("stale_vs_recent", [
        Evidence(EvidenceType.DOCUMENTATION, EvidenceRelation.CORROBORATES, "old doc", _t(-400), strength=0.6),
        Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "recent commit", _t(-1), strength=0.9),
    ]),
]


def main():
    print(f"{'scenario':32s} {'state':14s} summary")
    print("-" * 90)
    for name, evidence_items in SCENARIOS:
        scenario_name, state, summary = run_scenario(name, evidence_items)
        print(f"{scenario_name:32s} {state:14s} {summary}")


if __name__ == "__main__":
    main()
