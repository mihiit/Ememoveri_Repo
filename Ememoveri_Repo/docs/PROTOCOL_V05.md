# v0.5 protocol (written before any multi-language or new-baseline result was computed)

**Questions.**
Q1 (scope) Do the v0.3/v0.4 conclusions hold on repositories in other languages (JavaScript, Go, Rust, Java)?
Q2 (baselines) Does tuned EmemoVeri beat established history-based localisation ideas and a learned ranker, not only similarity+priors?

**Multi-language set.** express, axios (JS); gin, cobra (Go); ripgrep, serde (Rust); jackson-databind, commons-lang (Java);
pinned heads in `results/real_ext/repo_heads_multilang.txt`; source/test layout in `ememoveri/mining.py::MULTILANG_REPOS`.
Same task definition, warm-up, selection (<=400 evenly spaced tasks per repository) and replay as before; all tasks are test tasks.

**Frozen systems.** Everything from PROTOCOL_V04 (weights chosen on the original dev split) plus the new baselines below, whose
hyper-parameters are chosen on the **original dev split only** and then frozen.

**New baselines (all adaptations to a commit-subject query; none uses bug reports, which this task has not).**
B1 *Similar-history retrieval* (the history component of IR bug localisation, Zhou et al.): score a file by the cosine between the query and the
 subjects of past commits that touched it (max over commits; and sum of cosines); state filter on; tuned combination with claim similarity and priors.
B2 *Co-change recommendation* (Zimmermann et al.): seed = top-1 file by claim similarity; score the other files by the share of the
 seed's past non-bulk commits (<=10 files) in which they also changed; tuned combination with similarity.
B3 *Learned ranker*: pointwise logistic regression and a small gradient-boosted tree model over the feature set
 {claim sim, similar-history max/sum, co-change, frequency, evidence, trust, recency}; trained on dev tasks, candidates = top-50 by claim similarity.

**Hypotheses (directions fixed in advance).**
H1 EmemoVeri-T > Snapshot RAG in Hit@5 on the multi-language set (repository-level macro-average).
H2 EmemoVeri-T - Sim+State+Rec+Pop is indistinguishable from 0.
H3 EmemoVeri-D (blueprint) is not reliably better than Snapshot RAG.
H4 No new baseline (B1-B3) is better than EmemoVeri-T by a margin whose repository-level CI excludes 0 on the 12 Python held-out repositories or on the
 multi-language set. (If one is, the paper reports that EmemoVeri-T is dominated.)
**Inference.** Unit = repository; macro-average with bootstrap over repositories; exact sign-flip permutation test over repositories;
Holm correction across the contrasts reported together in one table (H1-H4 family per repository set). Effect sizes: paired mean difference and Cohen's d_z over repositories.
**Reporting.** All repositories are reported; none is dropped after seeing results.
