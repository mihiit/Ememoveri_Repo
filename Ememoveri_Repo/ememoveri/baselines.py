"""
Baseline memory conditions for the E5-E main comparison.

The blueprint names three conditions (NO_MEMORY, NAIVE_MEMORY, EMEMOVERI).
In practice a reviewer will ask "how does this compare to plain RAG?" --
and NAIVE_MEMORY as originally scoped ("flat semantic memory") is close to
that but conflates two independent design axes:

    1. Does retrieval respect repository scope at all?
    2. Does retrieval use trust/evidence/staleness, or similarity alone?

Collapsing both axes into one "naive" condition hides which axis actually
matters. This module separates them into a four-point ladder:

    NO_MEMORY     -- no retrieval at all (agent sees only the repo snapshot)
    PLAIN_RAG     -- similarity-only, NO repository scoping (searches ALL
                     claims globally, exactly like a standard RAG baseline
                     in the literature -- this is the condition a reviewer
                     means by "plain RAG")
    NAIVE_MEMORY  -- similarity-only, BUT local-repository-scoped (isolates
                     the value of scope awareness alone, before adding
                     trust/evidence/staleness/state)
    EMEMOVERI     -- full evidence-aware scoring + conflict resolution with
                     abstention (the full system)

Only EMEMOVERI can abstain; the other three conditions always commit to a
top-1 answer (that is itself part of what is being compared -- does
principled abstention change the reliability profile?).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from .conflict import ConflictResolver, Resolution, ResolutionOutcome
from .retrieval import RetrievalScorer, RetrievalWeights
from .schema import MemoryClaim
from .store import CrossRepoPolicy


class BaselineCondition(str, enum.Enum):
    NO_MEMORY = "NO_MEMORY"
    PLAIN_RAG = "PLAIN_RAG"
    NAIVE_MEMORY = "NAIVE_MEMORY"
    EMEMOVERI = "EMEMOVERI"


_SIMILARITY_ONLY_WEIGHTS = RetrievalWeights(similarity=1.0, trust=0.0, evidence=0.0, scope=0.0, staleness=0.0)


@dataclass
class ConditionResult:
    condition: BaselineCondition
    ranked: List[Tuple[MemoryClaim, float, float]]  # (claim, similarity, score)
    winner: Optional[MemoryClaim]
    abstained: bool


def _scorer_for(condition: BaselineCondition) -> RetrievalScorer:
    if condition in (BaselineCondition.PLAIN_RAG, BaselineCondition.NAIVE_MEMORY):
        return RetrievalScorer(weights=_SIMILARITY_ONLY_WEIGHTS, state_aware=False)
    return RetrievalScorer()  # EMEMOVERI: full default weights


def _cross_repo_policy_for(condition: BaselineCondition) -> CrossRepoPolicy:
    if condition == BaselineCondition.PLAIN_RAG:
        return CrossRepoPolicy(mode="open")          # no scoping at all
    if condition == BaselineCondition.NAIVE_MEMORY:
        return CrossRepoPolicy(mode="local_only")     # scoped, but nothing else evidence-aware
    return CrossRepoPolicy(mode="local_first")         # EMEMOVERI default mitigation policy


def run_condition(
    candidates_with_similarity: Sequence[Tuple[MemoryClaim, float]],
    query_scope: str,
    condition: BaselineCondition,
    reference: Optional[datetime] = None,
    gap_threshold: float = 0.20,
) -> ConditionResult:
    """Run one of the four memory conditions against a candidate pool.

    `candidates_with_similarity` is a flat list of (claim, similarity) pairs
    spanning both local and cross-repository claims -- exactly the same
    input every condition receives, so the ONLY thing that differs between
    conditions is the memory mechanism itself (matching the blueprint's
    requirement that "only the intended memory component should change").
    """
    if condition == BaselineCondition.NO_MEMORY:
        return ConditionResult(condition, ranked=[], winner=None, abstained=False)

    scorer = _scorer_for(condition)
    policy = _cross_repo_policy_for(condition)
    if scorer.state_aware:  # terminal SUPERSEDED claims are known-invalid
        from .schema import MemoryState
        candidates_with_similarity = [(c, s) for c, s in candidates_with_similarity
                                      if c.state != MemoryState.SUPERSEDED]

    local = [(c, s) for c, s in candidates_with_similarity
             if c.scope.split("/", 1)[0] == query_scope.split("/", 1)[0]]
    local_ids = {id(c) for c, _ in local}
    cross = [(c, s) for c, s in candidates_with_similarity if id(c) not in local_ids]

    local_scored = [(c, s, scorer.score(c, s, query_scope, reference)) for c, s in local]
    cross_scored = [(c, s, scorer.score(c, s, query_scope, reference)) for c, s in cross]
    ranked = policy.filter_and_rank(local_scored, cross_scored)

    if not ranked:
        return ConditionResult(condition, ranked=[], winner=None, abstained=False)

    if condition == BaselineCondition.EMEMOVERI:
        resolver = ConflictResolver(gap_threshold=gap_threshold)
        resolution = resolver.resolve([(c, s) for c, _, s in ranked])
        if resolution.outcome == ResolutionOutcome.ABSTAINED:
            return ConditionResult(condition, ranked=ranked, winner=None, abstained=True)
        return ConditionResult(condition, ranked=ranked, winner=resolution.winner, abstained=False)

    # PLAIN_RAG and NAIVE_MEMORY always commit to the top-ranked item -- they
    # have no abstention mechanism, which is itself part of what's compared.
    return ConditionResult(condition, ranked=ranked, winner=ranked[0][0], abstained=False)


ALL_CONDITIONS = (
    BaselineCondition.NO_MEMORY,
    BaselineCondition.PLAIN_RAG,
    BaselineCondition.NAIVE_MEMORY,
    BaselineCondition.EMEMOVERI,
)
