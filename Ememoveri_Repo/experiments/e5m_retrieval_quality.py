"""
E5-M (mechanism-layer simulation) — Retrieval quality evaluation.

SCOPE NOTE: real E5-M evaluates retrieval quality against hand-labeled
relevant-claim sets for real coding tasks. This script evaluates the SAME
retrieval scorer against synthetically labeled relevance (a claim is
"relevant" by construction if it shares the query scope and is not
SUPERSEDED), which validates the ranking mechanism's IR properties in a
controlled setting.

Run:  python -m experiments.e5m_retrieval_quality
"""

import json
import os
import random
from datetime import datetime, timedelta, timezone

from ememoveri.schema import Evidence, EvidenceRelation, EvidenceType, MemoryClaim
from ememoveri.retrieval import RetrievalScorer
from ememoveri.metrics import precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k, mean

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "data", "e5m_retrieval_quality.json")


def make_pool(rng: random.Random, query_scope="repoA", n_relevant=4, n_irrelevant=8):
    pool = []
    for i in range(n_relevant):
        c = MemoryClaim(statement=f"relevant-{i}", scope=query_scope, trust=round(rng.uniform(0.4, 0.9), 2))
        c.evidence = [Evidence(EvidenceType.COMMIT, EvidenceRelation.CORROBORATES, "e", datetime.now(timezone.utc))]
        c.updated_at = datetime.now(timezone.utc) - timedelta(days=rng.uniform(0, 60))
        sim = round(rng.uniform(0.5, 0.95), 2)
        pool.append((c, sim, True))
    for i in range(n_irrelevant):
        other_scope = rng.choice(["repoB", "repoC", query_scope])  # some same-scope-but-irrelevant, some cross-repo
        c = MemoryClaim(statement=f"irrelevant-{i}", scope=other_scope, trust=round(rng.uniform(0.1, 0.6), 2))
        c.evidence = [Evidence(EvidenceType.ISSUE, EvidenceRelation.CORROBORATES, "e", datetime.now(timezone.utc))]
        c.updated_at = datetime.now(timezone.utc) - timedelta(days=rng.uniform(0, 400))
        sim = round(rng.uniform(0.2, 0.7), 2)
        pool.append((c, sim, False))
    rng.shuffle(pool)
    return pool


def evaluate_query(scorer, pool, query_scope, k=5):
    candidates = [(c, sim) for c, sim, _ in pool]
    relevance_by_id = {c.claim_id: rel for c, _, rel in pool}
    ranked = scorer.rank(candidates, query_scope=query_scope)
    ranked_relevance = [relevance_by_id[c.claim_id] for c, _, _ in ranked]
    ranked_gains = [1.0 if r else 0.0 for r in ranked_relevance]
    n_relevant_total = sum(1 for r in ranked_relevance if r)
    return {
        "precision_at_k": precision_at_k(ranked_relevance, k),
        "recall_at_k": recall_at_k(ranked_relevance, k, n_relevant_total),
        "mrr": reciprocal_rank(ranked_relevance),
        "ndcg_at_k": ndcg_at_k(ranked_gains, k),
    }


def main(n_queries: int = 200, k: int = 5, seed: int = 21):
    rng = random.Random(seed)
    scorer = RetrievalScorer()
    per_query = []
    for _ in range(n_queries):
        pool = make_pool(rng)
        per_query.append(evaluate_query(scorer, pool, "repoA", k=k))

    results = {
        "n_queries": n_queries,
        "k": k,
        f"precision_at_{k}": mean([q["precision_at_k"] for q in per_query]),
        f"recall_at_{k}": mean([q["recall_at_k"] for q in per_query]),
        "mrr": mean([q["mrr"] for q in per_query]),
        f"ndcg_at_{k}": mean([q["ndcg_at_k"] for q in per_query]),
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("E5-M (mechanism-layer retrieval quality) results:")
    for key, v in results.items():
        print(f"  {key}: {v}")
    print(f"\nSaved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
