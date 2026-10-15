# Audit of v0.1 and fixes (v0.2)

Each item was reproduced with a probe script before being fixed; each fix has a
regression test in `tests/test_bugfixes.py`.

| # | Defect in v0.1 | Effect | Fix |
|---|---|---|---|
| 1 | SUPERSEDED was not terminal: later CORROBORATES moved the claim back to CORROBORATED | dead knowledge could be resurrected | terminal state in `apply_evidence` and `apply_evidence_batch` (late evidence kept for provenance) |
| 2 | Retrieval/conflict code never read `claim.state` | state machine was decorative; superseded claim with higher trust ranked first | `RetrievalScorer(state_aware=True)` drops SUPERSEDED; same in `MemoryStore.retrieve` and baselines |
| 3 | `TemporalLeakageGuard.filter_claim` dropped future evidence but kept trust/state/superseded_by derived from it | future information leaked into ranking (trust 1.0 vs 0.83 in probe) | visible evidence is **replayed** through the trust model and state engine; claims created after the cutoff are invisible |
| 4 | Staleness defaulted to the wall clock | every historical claim looked ~100% stale | `MemoryStore.retrieve` defaults `reference` to the knowledge cutoff |
| 5 | Keyword relation classifier ("Fix typo"→CONTRADICTS, "padding"→CORROBORATES via substring "add") | unusable labels on real commits | real-repo pipeline uses structural git relations; keyword classifier is retained only for the synthetic layer and must not be used for real data without validation |
| 6 | Clamped trust updates saturate after 2–3 strong items | trust stops discriminating | added order-invariant, non-saturating `logit_aggregated_update` (benchmark shows it does **not** rescue the trust signal) |
| 7 | Abstention: single candidate always RESOLVED; fixed 0.20 gap | degenerate coverage | optional `min_score`; real-repo analysis reports full risk–coverage curves instead of a fixed threshold |
| 8 | Cross-repo split used dataclass equality in the store but `id()` in baselines | inconsistent/conflated claims | identity-based in both |

Not fixed, by design: the synthetic E5-E/H/M/N/O/U/V/W experiments still draw the
"correct" claim from a distribution that favours the features the scorer rewards.
They are labelled mechanism-layer simulations and must not be presented as
real-world evidence.
