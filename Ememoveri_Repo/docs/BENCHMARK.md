# Real-repository benchmark protocol

**Repositories.** Development: fastapi, yt-dlp, browser-use (pinned HEADs in `results/real/repo_heads.txt`). Held-out (v0.4, never used for design or tuning): flask, requests, httpx, rich, black, poetry, pydantic, scrapy, pytest, aiohttp, celery, sphinx (`results/real_ext/repo_heads_new.txt`; protocol in `PROTOCOL_V04.md`).
Bare blobless clones; only commit metadata and `--name-status` tree diffs are read.

**Clock.** Commits are replayed in topological order; the clock is the running max of
committer timestamps (real histories contain non-monotone stamps: see `summary.json`).

**Claims.** One claim per live `.py` file under the repo's source root. A rename
carries the file's evidence and text forward and supersedes the old path; a deleted
file is superseded; a path re-created later is a new claim.

**Evidence.** Type TEST if the commit also edits a test file, else COMMIT. Strength 1.0,
or 0.3 for bulk commits touching >3 source files. Relation from git status (see README).

**Tasks.** Non-merge, non-bot, non-revert commits, ≥4 words after cleaning, modifying
1–4 source files, all of which already have a live claim (cold files excluded and
counted), occurring after the first 35% of the repo's source-touching commits. Up to
400 evenly spaced tasks per repo. Query = cleaned subject line.

**Split.** The earliest 30% of tasks (global clock) are *dev*; weights are chosen on dev
only (420-point grid over similarity/trust/evidence weights, staleness weight and
half-life, trust mode). The rest is *test*. browser-use appears only in test.

**Similarity.** Hashed unigram+bigram TF-IDF; IDF is recomputed from pre-cutoff claims
at every task (no fitted vocabulary, hence no cross-time leakage).

**Conditions.** Random, Recency, Popularity (query-independent priors); Plain RAG (all
repos, includes dead files); Naive memory (local, includes dead files); Snapshot RAG
(local, dead files removed — what an agent with a checkout gets for free); EmemoVeri
(blueprint weights); EmemoVeri dev-tuned; 11 ablations.

**Statistics.** Task-level paired bootstrap (10k) 95% CIs; exact McNemar on Hit@1/Hit@5
with Holm correction over the 16 binary comparisons. Robustness runs: warm-up 0.20/0.50
and 150 tasks/repo (`results/real_*`).

## Threats to validity
* Proxy task: file localisation from a commit subject, not issue text, not patch quality.
* Subject lines are written after the change and can be more specific than real task prompts.
* Tasks are not independent (hot files recur; same-author bursts); bootstrap CIs over tasks are optimistic.
* Three Python repositories; yt-dlp's one-file-per-site layout makes path tokens very informative.
* Dev-tuned weights are tuned on the same task family; test evidence for generalisation is
  the browser-use split, which dev never sees.
* No LLM in the loop: nothing here shows an agent would write better patches.

## Extensions (v0.3 / v0.4)
* **Scoring family** for ladders and baselines: `sim + lp*pop + le*evid + lt*trust + lr*2^(-age/H)` with local scope and optional state
  filtering; each system is tuned on the dev split with the same rule (`experiments/analyze_ext.py`).
* **Backends**: hashed TF-IDF (main), BM25 over the same hashed terms, mean static word vectors (spaCy `en_core_web_md`).
* **Inference**: task bootstrap, repository-month block bootstrap, file-cluster bootstrap; for the held-out set the unit is the repository.
* **Lifecycle events**: for each deletion / rename after warm-up the query is the latest subject touching the file; top-5 membership of
  the old and new path is evaluated before and after the event is ingested, with and without rename carry-over.
* **External tasks** (`collect_external`): queries that are not commit subjects (issue text); memory sees only commits up to the task
  cutoff (issue creation time) and gold files must already exist at the cutoff.
