"""
MemoryStore: the top-level object an agent harness talks to.

Combines StateEvolutionEngine + TrustModel + RetrievalScorer + ConflictResolver
into one coherent API:

    store = MemoryStore()
    store.add_claim(claim)
    store.ingest_evidence(claim_id, evidence)
    results = store.retrieve(query_embedding_sims, query_scope="pytorch/optim")
    resolution = store.resolve(results)

Also includes:
  - `TemporalLeakageGuard`: enforces the project's non-negotiable rule that
    no evidence timestamped after a task's knowledge cutoff is ever visible
    to retrieval for that task.
  - `CrossRepoPolicy`: local-first / similarity-gated / evidence-gated
    cross-repository retrieval policies used in the E4.x mitigation study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from .conflict import ConflictResolver, Resolution
from .retrieval import RetrievalScorer
from .schema import Evidence, MemoryClaim, MemoryState
from .state_engine import StateEvolutionEngine
from .trust import TrustModel


class TemporalLeakageError(RuntimeError):
    """Raised when a caller attempts to expose post-cutoff evidence."""


@dataclass
class TemporalLeakageGuard:
    """Enforces: task-visible evidence timestamp <= knowledge_cutoff.

    This is a hard runtime check, not just a benchmark-construction-time
    audit step — it is meant to be wired into the retrieval path itself so a
    bug in benchmark construction cannot silently leak future information
    into an agent evaluation run.
    """

    knowledge_cutoff: Optional[datetime] = None

    def is_visible(self, claim: MemoryClaim) -> bool:
        """A claim created after the cutoff did not exist yet."""
        return self.knowledge_cutoff is None or claim.created_at <= self.knowledge_cutoff

    def filter_claim(self, claim: MemoryClaim, trust_model=None, engine=None) -> MemoryClaim:
        """Return a view of `claim` as it was at the cutoff.

        Dropping post-cutoff evidence is not enough: `trust`, `state`,
        `superseded_by` and `updated_at` were all derived from that evidence
        and would silently carry the future into retrieval. The visible
        evidence is therefore REPLAYED chronologically through the same trust
        model and state engine, so every derived field is rebuilt from
        pre-cutoff information only.
        """
        if self.knowledge_cutoff is None:
            return claim
        if all(e.timestamp <= self.knowledge_cutoff for e in claim.evidence):
            return claim
        tm = trust_model or TrustModel()
        eng = engine or StateEvolutionEngine()
        visible = sorted((e for e in claim.evidence if e.timestamp <= self.knowledge_cutoff),
                         key=lambda e: e.timestamp)
        filtered = MemoryClaim(
            statement=claim.statement,
            scope=claim.scope,
            claim_id=claim.claim_id,
            state=MemoryState.PROVISIONAL,
            trust=tm.prior_trust,
            evidence=[],
            created_at=claim.created_at,
            updated_at=claim.created_at,
            tags=claim.tags,
        )
        for e in visible:
            eng.apply_evidence(filtered, e)
            filtered.trust = tm.sequential_update(filtered.trust, e)
        return filtered

    def assert_no_leakage(self, claim: MemoryClaim) -> None:
        if self.knowledge_cutoff is None:
            return
        for e in claim.evidence:
            if e.timestamp > self.knowledge_cutoff:
                raise TemporalLeakageError(
                    f"Evidence {e.source_id} timestamped {e.timestamp} exceeds "
                    f"cutoff {self.knowledge_cutoff} for claim {claim.claim_id}"
                )


@dataclass
class CrossRepoPolicy:
    """Controls how much weight/eligibility cross-repository claims get.

    mode:
        "local_only"      -- cross-repo claims never eligible
        "similarity_gate"  -- cross-repo claims eligible only if similarity >= threshold
        "local_first"      -- local claims always ranked ahead of cross-repo
                               claims unless the cross-repo score exceeds the
                               best local score by `override_margin`
        "open"             -- no restriction (baseline for measuring
                               contamination in E4.5/E4.6)
    """

    mode: str = "local_first"
    similarity_threshold: float = 0.80
    override_margin: float = 0.15

    def filter_and_rank(
        self,
        local_scored: List[tuple],   # (claim, similarity, score)
        cross_scored: List[tuple],
    ) -> List[tuple]:
        if self.mode == "local_only":
            return sorted(local_scored, key=lambda t: t[2], reverse=True)

        if self.mode == "similarity_gate":
            eligible_cross = [c for c in cross_scored if c[1] >= self.similarity_threshold]
            combined = local_scored + eligible_cross
            return sorted(combined, key=lambda t: t[2], reverse=True)

        if self.mode == "local_first":
            best_local = max((c[2] for c in local_scored), default=float("-inf"))
            eligible_cross = [c for c in cross_scored if c[2] >= best_local + self.override_margin]
            combined = local_scored + eligible_cross
            return sorted(combined, key=lambda t: t[2], reverse=True)

        if self.mode == "open":
            combined = local_scored + cross_scored
            return sorted(combined, key=lambda t: t[2], reverse=True)

        raise ValueError(f"Unknown CrossRepoPolicy.mode: {self.mode}")


@dataclass
class MemoryStore:
    state_engine: StateEvolutionEngine = field(default_factory=StateEvolutionEngine)
    trust_model: TrustModel = field(default_factory=TrustModel)
    retrieval_scorer: RetrievalScorer = field(default_factory=RetrievalScorer)
    conflict_resolver: ConflictResolver = field(default_factory=ConflictResolver)
    cross_repo_policy: CrossRepoPolicy = field(default_factory=CrossRepoPolicy)
    leakage_guard: TemporalLeakageGuard = field(default_factory=TemporalLeakageGuard)

    _claims: Dict[str, MemoryClaim] = field(default_factory=dict)

    def add_claim(self, claim: MemoryClaim) -> None:
        self._claims[claim.claim_id] = claim

    def get(self, claim_id: str) -> Optional[MemoryClaim]:
        return self._claims.get(claim_id)

    def all_claims(self, scope_prefix: Optional[str] = None) -> List[MemoryClaim]:
        claims = list(self._claims.values())
        if scope_prefix is not None:
            claims = [c for c in claims if c.scope == scope_prefix or c.scope.startswith(scope_prefix + "/")]
        return claims

    def ingest_evidence(self, claim_id: str, evidence: Evidence, aggregate: bool = False) -> MemoryClaim:
        claim = self._claims[claim_id]
        if aggregate:
            self.state_engine.apply_evidence_batch(claim, [evidence], aggregate=True)
        else:
            self.state_engine.apply_evidence(claim, evidence)
        # trust is updated via the sequential model by default; callers doing
        # an aggregated-ablation run should call recompute_trust_aggregated().
        claim.trust = self.trust_model.sequential_update(claim.trust, evidence)
        return claim

    def recompute_trust_aggregated(self, claim_id: str) -> float:
        claim = self._claims[claim_id]
        claim.trust = self.trust_model.aggregated_update(claim.evidence)
        return claim.trust

    def retrieve(
        self,
        candidates_with_similarity: Sequence[tuple],  # (claim, similarity)
        query_scope: str,
        reference: Optional[datetime] = None,
        top_k: Optional[int] = None,
        enforce_leakage_guard: bool = True,
    ) -> List[tuple]:
        # Staleness must be measured from the knowledge cutoff, not the wall
        # clock, otherwise every historical claim looks ~fully stale.
        if reference is None:
            reference = self.leakage_guard.knowledge_cutoff
        prepared = []
        for claim, sim in candidates_with_similarity:
            if enforce_leakage_guard and not self.leakage_guard.is_visible(claim):
                continue
            c = (self.leakage_guard.filter_claim(claim, self.trust_model, self.state_engine)
                 if enforce_leakage_guard else claim)
            if enforce_leakage_guard:
                self.leakage_guard.assert_no_leakage(c)
            prepared.append((c, sim))

        local = [(c, s) for c, s in prepared if c.scope.split("/", 1)[0] == query_scope.split("/", 1)[0]]
        local_ids = {id(c) for c, _ in local}
        cross = [(c, s) for c, s in prepared if id(c) not in local_ids]

        if self.retrieval_scorer.state_aware:
            local = [(c, s) for c, s in local if c.state != MemoryState.SUPERSEDED]
            cross = [(c, s) for c, s in cross if c.state != MemoryState.SUPERSEDED]

        local_scored = [(c, s, self.retrieval_scorer.score(c, s, query_scope, reference)) for c, s in local]
        cross_scored = [(c, s, self.retrieval_scorer.score(c, s, query_scope, reference)) for c, s in cross]

        ranked = self.cross_repo_policy.filter_and_rank(local_scored, cross_scored)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    def resolve(self, ranked: Sequence[tuple]) -> Resolution:
        """`ranked`: output of retrieve(), i.e. (claim, similarity, score) tuples."""
        scored_claims = [(c, s) for c, _, s in ranked]
        return self.conflict_resolver.resolve(scored_claims)
