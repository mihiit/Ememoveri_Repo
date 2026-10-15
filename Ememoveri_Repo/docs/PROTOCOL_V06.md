# v0.6 protocol (written and committed BEFORE any result on the 16 fresh repositories was computed)

**Why.** v0.5 found that retrieval over individual evidence items beat the original EmemoVeri scorer on the held-out sets. The redesign below
was therefore motivated by results on those sets, so they can no longer confirm it. A fresh set of 16 repositories, chosen and configured
(`ememoveri/mining.py::FRESH_REPOS`, heads in `results/real_ext/repo_heads_fresh.txt`) before any result on them existed, is the confirmatory test.
The 12 Python and 8 multi-language repositories of v0.4/v0.5 are re-reported as *post-hoc* (design influenced by them).

**Fresh set.** Python: starlette, uvicorn, click, jinja, werkzeug, attrs, tornado, marshmallow, tqdm, rq.
Other: koa, fastify (JavaScript), echo (Go), fd, bat (Rust), mockito (Java). Same task definition, warm-up, selection and replay as before.

**EmemoVeri-2 (frozen; tuned on the original dev split only, `results/real_ext/ev2_config.json`, `tune_ev2.py`).**
A claim's score is a blend of (i) claim similarity, (ii) the maximum over its evidence items of (cosine between the query and the item's
subject) x (item strength), where strength is the ledger's 1.0 / 0.3 (bulk-commit) value, (iii) a change-frequency prior (weight 0.05) and
(iv) a recency prior (weight 0.05, half-life 30 d); same-repository live claims only. Trust and abstention are removed.
Not selected on dev, hence off: TEST bonus, dilution weighting, evidence decay, revert cancellation, noisy-OR / sum aggregation, dense (static word-vector) mixing.

**Frozen comparators.** Snapshot RAG; EmemoVeri-D; EmemoVeri-T (the original design, tuned); Sim+State+Rec+Pop; EV2-plain (the same scorer with
every evidence-aware weight off, separately tuned on dev with the same blend grid); evidence-level similar-history retrieval (v0.5 B1);
linear all-channel scorer; logistic and gradient-boosted rankers (trained on dev, v0.5).

**Hypotheses (directions fixed in advance; unit = repository, n = 16).**
H1 EV2 > Snapshot RAG in Hit@5.
H2 EV2 > EmemoVeri-T (the redesign improves on the original design).
H3 EV2 > EV2-plain (evidence strength carries signal beyond plain similar-history retrieval).
H4 EV2 > Sim+State+Rec+Pop.
H5 EV2 is not worse than the all-channel linear scorer or the gradient-boosted ranker (reported with CIs; no directional claim).
**Inference.** Macro-average over repositories, bootstrap CI over repositories, exact (n=16) sign-flip test, Cohen's d_z, Holm across H1-H5 (six contrasts).
Subsets (10 Python, 6 other-language) are descriptive. Routine tasks (duplicate subject, release/version housekeeping) are excluded in a pre-declared robustness analysis.
**Failure reporting.** If H2 or H3 fails the paper says so; no fresh repository is dropped; EV2 is not re-tuned after seeing fresh results.

**Outcome (appended after the analysis; the text above was not changed).** H1, H2, H4 confirmed; H5 not worse than either comparator; **H3 not confirmed** (+0.001, CI includes 0).
Results: `results/real_ext/summary_v06.json`. The frozen configuration hash is in `docs/REPRODUCIBILITY.md`.
