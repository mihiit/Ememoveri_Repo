"""
EmemoVeri
=========

Evidence-Aware Engineering Memory for Repository-Scoped Coding Agents.

This package implements the *mechanism layer* of EmemoVeri:

    - Evidence representation and processing
    - Memory claims with trust, scope, and provenance
    - State evolution (PROVISIONAL -> CORROBORATED / REFINED / CONTESTED / SUPERSEDED)
    - Evidence-aware, staleness- and scope-sensitive retrieval scoring
    - Conflict resolution with principled abstention

It intentionally does NOT include the E5 real-repository benchmark runner,
LLM coding-agent harness, or statistical-analysis notebooks — those require
live repositories, model API access, and compute that must be run in your
own environment. The `experiments/` directory contains runnable scripts that
reproduce the E1-E4.8 *mechanism validation* experiments described in the
project blueprint, using this package directly.
"""

from .schema import Evidence, EvidenceType, MemoryState, MemoryClaim
from .state_engine import StateEvolutionEngine
from .retrieval import RetrievalScorer, RetrievalWeights
from .conflict import ConflictResolver, ResolutionOutcome
from .trust import TrustModel

__all__ = [
    "Evidence",
    "EvidenceType",
    "MemoryState",
    "MemoryClaim",
    "StateEvolutionEngine",
    "RetrievalScorer",
    "RetrievalWeights",
    "ConflictResolver",
    "ResolutionOutcome",
    "TrustModel",
]

__version__ = "0.6.5"
