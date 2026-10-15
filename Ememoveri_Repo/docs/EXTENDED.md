# v0.3 extended analyses
Run on the pinned heads in `results/real/repo_heads.txt` (the main benchmark reproduces bit-for-bit: 0 Hit@5 mismatches over 1,096 tasks).
* `ememoveri/realbench_ext.py` -- `ExtMemory` (raw change counts, rename carry-over switch, resurrection counter, BM25, static-word-vector cosine), query degradation, NDCG@5/Recall@5/Precision@5, collector.
* `experiments/analyze_ext.py` -- ladder, tuned Sim/State/Recency/Popularity baselines, backends, leave-one-repository-out, degradation, staleness sweep, task / repo-month / file-cluster bootstrap, lifecycle events.
* `results/real_ext/summary_ext.json` -- all numbers used in the paper's new sections; `paper/make_ext.py` regenerates the tables and figures.
Not done (needs resources not available offline): issue/PR text queries (GitHub token), coding-agent evaluation (LLM), neural dense retriever, new trust model.
