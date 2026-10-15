# E5 Integration Guide

This document is a concrete plan for wiring the mechanism layer in
`ememoveri/` into a real E5 real-repository benchmark and agent evaluation,
in your own compute environment. Nothing here is run automatically by this
repository — it is a specification for the pipeline you build next.

**For which models to use, see `docs/MODELS.md` first** — it pins the exact
provider API IDs (Groq-hosted Llama 3.3 70B Versatile, Qwen3 32B, and
optionally GPT-OSS-20B) that §4 below assumes.

## 1. Repository mining -> Evidence

Use a Git-mining tool (e.g. [PyDriller](https://pydriller.readthedocs.io/))
against a local clone of each target repository. For each commit at or
before a task's `before_commit` cutoff:

```python
from pydriller import Repository
from ememoveri.evidence_processor import EvidenceProcessor

proc = EvidenceProcessor()
evidence_items = []
for commit in Repository(repo_path, to_commit=before_commit_sha).traverse_commits():
    evidence_items.append(proc.from_commit(
        message=commit.msg,
        timestamp=commit.committer_date,
        commit_sha=commit.hash,
    ))
```

Do the equivalent for test results (CI logs / test file diffs), documentation
changes (`.md`/`.rst` diffs), and issue/PR discussion (GitHub API, filtered
to `created_at <= cutoff`).

**Critical:** never construct an `Evidence` object from anything timestamped
after the task's `before_commit`. This is the temporal rule the blueprint
calls non-negotiable.

## 2. Memory construction at each cutoff

For each task, build a fresh `MemoryStore` (or a `MemoryStore` scoped to
that repository) and set:

```python
from ememoveri.store import MemoryStore, TemporalLeakageGuard

store = MemoryStore(leakage_guard=TemporalLeakageGuard(knowledge_cutoff=before_commit_timestamp))
```

Then aggregate mined evidence into `MemoryClaim`s (one claim per distinct
engineering fact — e.g. "function X requires argument Y", "API Z is
deprecated in favor of W") and ingest evidence via
`store.ingest_evidence(claim_id, evidence)`.

Claim *construction* (deciding what counts as one claim vs. two) is itself a
design decision the blueprint leaves open — a reasonable starting point is
one claim per (file-region, semantic-topic) pair extracted via an LLM
prompt over the pre-cutoff commit/doc history, with manual audit of a
sample (as the blueprint's E5-B / task-quality-filtering step already does
for tasks).

## 3. Leakage validation (E5-A)

Before running any agent evaluation, assert on every constructed claim:

```python
for claim in store.all_claims():
    store.leakage_guard.assert_no_leakage(claim)
```

This raises `TemporalLeakageError` immediately on any violation, giving you
a hard automated gate rather than a manual audit step alone.

## 4. Agent harness wiring

At task time, the coding agent needs:

1. The pre-cutoff repository snapshot (checkout `before_commit`).
2. Retrieved memory context:
   ```python
   ranked = store.retrieve(candidates_with_similarity, query_scope=f"{repo_name}/{subsystem}")
   resolution = store.resolve(ranked)
   ```
   where `candidates_with_similarity` comes from embedding the task
   description against each claim's `statement` (any embedding model; this
   package does not include one, since EmemoVeri's contribution is the
   re-ranking/scoring layer on top of similarity, not the embedding step).
3. The task's specification-driven prompt (E5-C), NOT the raw commit diff.

Feed `resolution.winner` (if `RESOLVED`) or an explicit "memory is
ambiguous, proceed with caution" note (if `ABSTAINED`) into the agent's
prompt.

4. Call the coding-agent model itself. Using the free Groq tier (see
   `docs/MODELS.md` for the full rationale and caveats):

   ```python
   from groq import Groq

   client = Groq()  # reads GROQ_API_KEY from the environment

   response = client.chat.completions.create(
       model="llama-3.3-70b-versatile",  # or "qwen/qwen3-32b", or "openai/gpt-oss-20b"
       messages=[
           {"role": "system", "content": system_prompt_with_memory_context},
           {"role": "user", "content": task_specification},
       ],
       temperature=0.0,      # pin identically across ALL models and conditions
       max_tokens=4096,      # pin identically across ALL models and conditions
   )
   patch_text = response.choices[0].message.content
   ```

   Run this once per (task, model, condition) triple — i.e. every task runs
   through all three models in `docs/MODELS.md`, and each model runs through
   all four conditions in §5 below, holding the task, prompt template,
   source context, and decoding parameters fixed across that entire grid.

## 5. Conditions to compare (E5-E)

This repo's mechanism layer implements a **four-way** condition ladder in
`ememoveri/baselines.py` (`BaselineCondition` + `run_condition()`), which
separates two design axes the blueprint's original three-condition list
conflated: whether retrieval is repository-scope-aware, and whether it uses
trust/evidence/staleness at all. Use the same four conditions for the real
agent evaluation:

| Condition | `MemoryStore` / scorer configuration | Isolates |
|---|---|---|
| `NO_MEMORY` | Skip retrieval entirely; agent sees only the repo snapshot. | Baseline floor. |
| `PLAIN_RAG` | `RetrievalWeights(similarity=1, trust=0, evidence=0, scope=0, staleness=0)` + `CrossRepoPolicy(mode="open")` (no scoping at all — searches every repository's memory). | The standard literature RAG baseline reviewers will ask for. |
| `NAIVE_MEMORY` | Same similarity-only weights, but `CrossRepoPolicy(mode="local_only")`. | The value of repository scope alone, before any evidence-awareness. |
| `EMEMOVERI` | Full default `RetrievalWeights` + `CrossRepoPolicy(mode="local_first")` + `ConflictResolver` (can abstain). | The full system, including principled abstention. |

`ememoveri.baselines.run_condition(candidates_with_similarity, query_scope, condition)`
implements exactly this for the mechanism layer — for a real agent, wire an
LLM call and patch-generation step after `run_condition()` returns its
`winner` (or a "memory is ambiguous" note when `result.abstained` is True),
using the SAME candidate pool and query scope across all four conditions per
task, per the blueprint's requirement that only the memory mechanism differ.

## 6. Ablations (E5-G, E5-J, E5-K, E5-L)

Each ablation is a targeted configuration change to the same `MemoryStore`:

- **Trust only**: set `RetrievalWeights(similarity=0.45, trust=0.55, evidence=0.0, scope=0.0, staleness=0.0)` (renormalize positive terms to sum to 1) and skip the state engine's REFINED/CONTESTED distinctions (always treat non-superseded claims as usable).
- **Evidence only**: symmetric — zero out `trust`, keep `evidence`.
- **State only**: use `MemoryState` to filter (e.g. drop SUPERSEDED, deprioritize CONTESTED) but skip the continuous trust score (treat all non-superseded claims as trust=0.5).
- **No contradiction handling / no supersession / no refinement**: subclass or configure `StateEvolutionEngine` to skip the corresponding branch in `_resolve_state`.
- **Cross-repo ablation**: swap `CrossRepoPolicy.mode` across `local_only` / `open` / `similarity_gate` / `local_first`.

## 7. Metrics (E5-M, E5-N, E5-O, E5-R)

- Retrieval quality: compute Recall@K / Precision@K / MRR / NDCG using
  `store.retrieve(..., top_k=K)` output against a hand-labeled
  relevant-claim set for each task.
- Abstention: use `ConflictResolver.abstention_precision()` and
  `.coverage()` once you have ground-truth "was this genuinely ambiguous"
  labels per task.
- Trust calibration: feed `(claim.trust, task_success)` pairs into
  `TrustModel.expected_calibration_error()` and `.brier_score()`.
- Statistical significance for paired NO_MEMORY vs EMEMOVERI task outcomes:
  McNemar's test (`statsmodels.stats.contingency_tables.mcnemar`) is
  appropriate for paired binary success/failure outcomes on the same task
  set; report the effect size and a bootstrap confidence interval alongside
  the p-value rather than the p-value alone.

## 8. What to log per task (for the error taxonomy in E5-S)

Log, per task: **model API ID** (per `docs/MODELS.md`), condition, retrieved
claim IDs + scores, resolver outcome, generated patch (or lack thereof),
whether the patch applied, whether tests passed, and a manually- or
LLM-assigned error category from the blueprint's taxonomy (No patch
generated / Malformed patch / Patch does not apply / Wrong file / Wrong API
/ Stale-memory error / Contradiction-resolution error / Retrieval error /
Correct patch but failing tests / Environment-or-build failure). This log is
what the error-taxonomy analysis and the efficiency analysis (tokens/latency
vs. success) will be computed from — logging the model ID on every row is
what makes the per-model breakdown table (blueprint's Table 8) possible
without re-running anything.

## 9. Multi-model reporting (E5-P)

Once every (task, model, condition) triple has a logged result (§8), build
Table 8 exactly as `docs/MODELS.md` specifies: one row per (model,
condition) pair, reporting both absolute performance and the paired
EmemoVeri-minus-baseline delta for that model specifically. Do not average
across models into a single number — report per-model rows and, if you want
a pooled summary statistic, add it as an explicit additional row clearly
labeled "pooled across models," never in place of the per-model breakdown.
