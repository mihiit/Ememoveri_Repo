# Changelog

## 0.6.5 (2026-10)
* IEEE version now uses the IEEEtran **conference** template (two author blocks, letter size); the journal-mode build is replaced.

## 0.6.4 (2026-10)
* Second author (Priyanka Tyagi), corresponding-author email, and declarations (no funding, no competing interests) added to both papers and to the package metadata.

## 0.6.3 (2026-10)
* Figures redrawn for legibility (larger type, vertical layout, no overlapping labels): vector PDF in the papers, 300-dpi PNG copies in `paper/figures_hd/`.
* Acknowledgment statement corrected to say accurately who did what; wording condensed so the Springer version stays at 15 pages; spelling checked (British spelling used consistently).

## 0.6.2 (2026-10)
* Plain-language rewrite of the paper text (shorter sentences, terms defined at first use, a short 'in plain terms' paragraph); Flesch reading ease 51 -> 60, average sentence length 23.6 -> 18.6 words; every reported number unchanged. Related-work and contribution wording rewritten in new words. The post-hoc table was folded into the text so the Springer version stays within 15 pages.

## 0.6.1 (2026-10)
* Springer LNCS version of the current paper (`paper/springer2/`, 15 pages including references; `build.sh`, `to_lncs.py`).
* Out-of-sample component analysis of the frozen EmemoVeri-2 and blend-sensitivity analysis (`experiments/ablate_ev2.py`,
  `results/real_ext/ablate_ev2.json`, `sens_ev2.json`): the evidence-item term, state filter, claim similarity and recency prior carry the gain;
  strength weighting, rename carry-over and the frequency prior are small or null; all 48 blends of the tuning grid beat all three baselines on the fresh set.
* `collect_ext.py --no-carry` (replay without rename carry-over); `paper/ieee/gen/` (text template + generator that fills every number from the result files).

## 0.6.0 (2026-10)
* **EmemoVeri-2**: retrieval over the evidence ledger (`ememoveri/evidence_retrieval.py`): a claim inherits its best-matching evidence item,
  blended with claim similarity and two activity priors; trust and abstention removed. Tuned on the original dev split only
  (`experiments/tune_ev2.py`, `results/real_ext/ev2_config.json`), frozen, and tested on **16 fresh repositories** under `docs/PROTOCOL_V06.md`
  (written before any fresh result): beats snapshot retrieval, the original design and tuned recency+popularity; not worse than a gradient-boosted ranker.
  Evidence-strength weighting (selected on dev) does **not** replicate on the fresh set (H3 not confirmed).
* `ExtMemory` now records an evidence ledger (per-claim commit ids and relations; per-commit clock, strength, TEST flag, files touched, revert cancellation),
  `ledger()` export, commit-body query variant, degraded-query evidence cosines; `collect(slim=True, track_events=False)` for low memory.
* `FRESH_REPOS` (starlette, uvicorn, click, jinja, werkzeug, attrs, tornado, marshmallow, tqdm, rq, koa, fastify, echo, fd, bat, mockito).
* Revert-cancellation and query-degradation analyses (`experiments/analyze_v06b.py`); `analyze_v06.py` (confirmatory analysis); `paper/make_v06.py`.
* New paper (`paper/ieee/`): title and structure rewritten around the redesign; tautological findings removed from the headline; the EmemoVeri-E label is gone.
* Tests: `tests/test_evidence_retrieval.py` (prefix/no-future-evidence, strength, revert cancellation); suite is now 90 tests.


## 0.5.0 (2026-10)
* **Multi-language held-out set** (express, axios, gin, cobra, ripgrep, serde, jackson-databind, commons-lang): `RepoConfig` now takes
  `exts`, `test_suffixes`, `exclude`; `MULTILANG_REPOS` in `ememoveri/mining.py` (Python behaviour unchanged).
* **Established baselines (adapted) and learned rankers**: similar-history retrieval, co-change recommendation, linear all-channel scorer,
  logistic and gradient-boosted rankers (`ExtMemory.extra`, `experiments/analyze_v05.py`); all tuned/trained on the original dev split and frozen.
* **Repository-level inference**: exact sign-flip test and Cohen's d_z (`ememoveri/stats.py`), Holm within contrast families.
* `experiments/task_flags.py` (routine-task robustness check), `docs/PROTOCOL_V05.md` (written before the new results),
  `paper/make_v05.py`, IEEEtran paper in `paper/ieee/` (the LNCS version in `paper/springer/` is the v0.4 text and is superseded).
* Finding: the original EmemoVeri scorer is dominated by retrieval over individual evidence items (EmemoVeri-E) on held-out Python.

## 0.4.0 (2026-10)
* Held-out confirmation on 12 repositories never used in design or tuning (`docs/PROTOCOL_V04.md`, `experiments/analyze_heldout.py`,
  `results/real_ext/summary_heldout.json`); `HELDOUT_REPOS` in `ememoveri/mining.py`.
* Issue/PR task pipeline (`experiments/issue_tasks.py`, `collect_issue.py`, `ememoveri.realbench_ext.collect_external`) and a
  retrieval-conditioned patch-generation harness with test execution (`experiments/agent_eval.py`, `export_retrievals.py`).
  Both need credentials and have **not** been run on real data; see `docs/RUN_V04.md`.
* `pilot/fastapi_pilot/`: the earlier small agent pilot, with its caveats.
* Springer LNCS version of the paper in `paper/springer/` (15 pages including references).

## 0.3.0
* `ememoveri/realbench_ext.py`: BM25 and static-word-vector similarity, query degradation, NDCG/Recall/Precision@5, rename
  carry-over switch, lifecycle events, one-pass collector.
* `experiments/analyze_ext.py`: component ladder, tuned history-prior baselines, backends, leave-one-repository-out, degradation,
  staleness sweep, task / repo-month / file-cluster bootstrap, lifecycle evaluation (`results/real_ext/summary_ext.json`).

## 0.2.0
* Real-history benchmark (`mining.py`, `realbench.py`, `stats.py`), integrity controls, audit and eight fixes (`docs/AUDIT.md`).

## 0.1.0
* Mechanism library and synthetic experiments E1-E5-W (results from this version should not be reused).
