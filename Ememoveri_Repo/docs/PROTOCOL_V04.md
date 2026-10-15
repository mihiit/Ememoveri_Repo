# v0.4 confirmatory protocol (written before any new-repository result was computed)

**Question.** Do the v0.3 conclusions hold on repositories that played no role in designing, debugging or tuning the system?

**Held-out set.** 12 repositories not used in v0.1-v0.3: flask, requests, httpx, rich, black, poetry, pydantic, scrapy, pytest, aiohttp, celery, sphinx
(pinned heads in `results/real_ext/repo_heads_new.txt`, source roots in `ememoveri/mining.py`). Same task definition, warm-up (35%), selection (<=400 evenly spaced tasks per repository) and replay as the main benchmark; all tasks are test tasks because nothing is tuned on them.

**Frozen systems (no re-tuning).** Snapshot RAG; EmemoVeri blueprint; EmemoVeri-T (w_sim .8, w_trust 0, w_ev .1, w_scope .1, w_stale .05, H 365); every ladder rung and history-prior baseline with the weights selected on the *original* dev split (listed in `summary_ext.json`: `ladder[*].cfg`, `strong[*].cfg`); recency and popularity priors; naive memory.

**Primary hypotheses (one-sided direction fixed in advance).**
H1: EmemoVeri-T > Snapshot RAG in Hit@5 (repository-level macro-average).
H2: EmemoVeri-T - Sim+State+Rec+Pop is not distinguishable from 0 (v0.3 finding).
H3: EmemoVeri-D (blueprint) is not reliably better than Snapshot RAG.
**Inference.** The unit is the repository: macro-average of per-repository Hit@5 and a 10,000-sample bootstrap that resamples repositories (n=12); per-repository task-level tables are descriptive. Sign counts (number of repositories where A > B) are reported. No multiple-comparison correction beyond the three pre-declared contrasts being reported together.
**Reporting.** All 12 repositories are reported; none is dropped after seeing results. Tasks and repositories with fewer than 40 eligible tasks are kept and flagged.
