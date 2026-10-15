# EmemoVeri — Evidence-Aware Memory for Repository-Scoped Coding Agents

EmemoVeri maintains *claims* about a code repository (one claim per source file), attaches typed, timestamped *evidence* to them
(commits, tests, reverts, deletions, renames), evolves a claim *state* (PROVISIONAL, CORROBORATED, REFINED, CONTESTED, SUPERSEDED),
keeps a *trust* value, and ranks claims with an evidence-, staleness- and scope-aware score.

This repository contains
1. the mechanism library (`ememoveri/`),
2. a **leakage-controlled benchmark that replays real git history** and the analyses built on it (v0.2 to v0.4),
3. the original synthetic mechanism experiments E1-E5-W, kept for completeness,
4. harnesses for real issue-text queries and test-based patch generation (written and unit-tested, **not yet run on real data**),
5. an earlier small agent pilot on fastapi (`pilot/`), and the paper sources (`paper/`).

> **Read this first.** Cite the real-history results (`results/real/`, `results/real_ext/`). The synthetic experiments
> (`results/SUMMARY.md`) use scenarios whose generator shares structure with the scoring function; they are mechanism sanity checks,
> not evidence of real-world benefit. Results produced before the v0.2 audit (`docs/AUDIT.md`) must not be reused.

## What the benchmark measures
*Memory-guided change localisation.* Given only the subject line of a **future** commit and memory built from strictly earlier
history, rank the source files that commit will modify. Ground truth is read from `git`, never from a classifier. Repositories are
replayed on one chronological clock; relations come from structural git events (`A` corroborates, `M` refines, `D` / renamed-away
supersedes, revert trailer contradicts). Integrity controls: memory is updated only after a task is scored; IDF uses pre-cutoff claims;
a runtime assertion rejects evidence newer than the task; a leak canary must inflate accuracy; a shuffled-query control measures the
query-independent part of a score; weights are tuned on an earlier dev period and reported on a later test period.

## Headline findings
Thirty-six repositories in five languages, evaluated under three successive protocols written before the respective results
(`docs/PROTOCOL_V04.md`, `V05`, `V06`); unit of inference for held-out sets = repository.

* **The original design does not hold up.** Tuned EmemoVeri-1 beats snapshot RAG (similarity + dead files removed) by +0.111 Hit@5 on the
  development repositories, but is indistinguishable from a tuned *similarity + state + recency + popularity* baseline; its trust value
  saturates (AUROC 0.50), its abstention rule is no better than a similarity gap, and retrieval over individual evidence items beats it on held-out Python.
* **EmemoVeri-2 (retrieval over the evidence ledger) is confirmed on 16 fresh repositories** (tuned on dev only, frozen, five declared hypotheses):
  +0.132 Hit@5 over snapshot RAG (15/15 non-tied repositories, Holm p < 0.001), +0.041 over the original design, +0.043 over Sim+State+Rec+Pop,
  not worse than a linear all-channel scorer or a gradient-boosted ranker.
* **Not confirmed:** weighting evidence items by strength (+0.035 on dev-test) adds +0.001 [-0.005, 0.007] on the fresh set.
  Revert cancellation changes no Hit@5 outcome; benchmark reverts are too rare to exercise contradiction.
* **Component analysis (post hoc, frozen system):** the evidence-item term (+0.049 Hit@5), state filter (+0.024), claim similarity (+0.018) and recency prior (+0.010) carry the gain; strength weighting (+0.005), rename carry-over (+0.004) and the frequency prior (+0.001) are small or null. All 48 blends of the tuning grid beat snapshot RAG, the original design and Sim+State+Rec+Pop on the fresh set.

## Reproduce
```bash
pip install -e ".[all]"                    # numpy, scipy, scikit-learn, matplotlib, pytest
pip install spacy && python -m spacy download en_core_web_md      # only for the static-word-vector backend (collect_ext.py)
python -m pytest -q                        # 90 tests
bash scripts/reproduce_real.sh repos       # clones 39 repos (blobless), reruns everything (~1.5 h, 1 core, ~2.5 GB RAM)
```
Histories move on: results depend on the clone date. Pinned heads are in `results/real/repo_heads.txt` (3 development repositories)
and `results/real_ext/repo_heads_new.txt` (12 held-out); `scripts/reproduce_real.sh` explains how to check them out. Reproduction was
checked bit-for-bit on the main benchmark; see `docs/REPRODUCIBILITY.md`.

| step | script | output |
|---|---|---|
| main benchmark (3 repos) | `experiments/real_bench.py`, `analyze_real.py`, `figures_real.py` | `results/real/` |
| ladder, history-prior baselines, backends, LORO, degradation, staleness, lifecycle | `collect_ext.py --set dev`, `analyze_ext.py` | `results/real_ext/summary_ext.json` |
| held-out confirmation (12 repos) | `collect_ext.py --set heldout --slim`, `analyze_heldout.py` | `results/real_ext/summary_heldout.json` |
| multi-language set, established baselines, learned rankers, repository-level tests | `collect_ext.py --set multilang`, `analyze_v05.py`, `task_flags.py` | `results/real_ext/summary_v05.json` |
| EmemoVeri-2: tuning, fresh-set confirmation, degradation, revert cancellation | `tune_ev2.py`, `analyze_v06.py`, `analyze_v06b.py` | `results/real_ext/ev2_*.json`, `summary_v06*.json` |
| paper tables and figures | `paper/make_tables.py`, `make_ext.py`, `make_heldout.py`, `make_v06.py`, `paper/ieee/build.sh` | `paper/ieee/` and `paper/springer2/` (current, IEEE and Springer LNCS), `paper/springer/` (v0.4 text, superseded) |
| synthetic experiments | `python -m experiments.run_all` | `results/data`, `results/figures` |

## Not yet run (needs your credentials; see `docs/RUN_V04.md`)
* **Real issue/PR queries**: `experiments/issue_tasks.py` (needs `GITHUB_TOKEN`) then `collect_issue.py`, `analyze_heldout.py ... tfidf:full`.
* **Patch generation with test execution**: `experiments/agent_eval.py` validate / solve / report (needs `ANTHROPIC_API_KEY`).
  Single-shot patching conditioned on the files each memory retrieves; not a multi-turn agent. The validation step has been run on
  real flask commits; the model-calling step has not.
* A neural dense retriever (the "dense" backend here is averaged static word vectors) and a new trust model (the benchmark has almost
  no negative or independent evidence, so none could be validated).

## Layout
| path | purpose |
|---|---|
| `ememoveri/` | schema, trust, state engine, retrieval scorer, conflict resolver, store + leakage guard |
| `ememoveri/mining.py` | git-history miner; `DEFAULT_REPOS` (3 development repos) and `HELDOUT_REPOS` (12) |
| `ememoveri/realbench.py` | chronological replay, vectorised scoring, tuning, controls |
| `ememoveri/evidence_retrieval.py` | EmemoVeri-2: evidence-level scorer over the ledger prefix at task time |
| `ememoveri/realbench_ext.py` | BM25, static word vectors, query degradation, extra metrics, lifecycle events, one-pass collector, external (issue) tasks |
| `ememoveri/stats.py` | paired bootstrap, exact McNemar, Holm, AURC / AUROC / ECE |
| `experiments/` | benchmark pipelines above, issue-task and patch-generation harnesses, synthetic experiments `e1...e5w` |
| `pilot/fastapi_pilot/` | earlier 9-task agent pilot with its caveats (`README.md` there) |
| `paper/` | `ieee/` (IEEEtran, current), `springer2/` (LNCS, current, 15 pages), `springer/` (LNCS, v0.4 text, superseded), the earlier article-class draft, table / figure generators |
| `docs/` | `AUDIT.md`, `BENCHMARK.md`, `PROTOCOL_V04.md`, `PROTOCOL_V05.md`, `PROTOCOL_V06.md`, `RUN_V04.md`, `EXTENDED.md`, `REPRODUCIBILITY.md` |
| `results/` | real-history results (`real*`, `real_ext`), synthetic results (`data`, `figures`) |

## Limitations
File-level localisation from commit subjects is a proxy for agent tasks; no language model is in the loop in any reported result;
thirty-six popular open-source repositories in five languages (file-level granularity); task outcomes are not independent (hot files recur), so
block and cluster bootstraps are reported alongside task-level intervals. See `docs/BENCHMARK.md` and the paper's threats to validity.
