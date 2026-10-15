# Real-repository benchmark summary (test n=768, dev n=328)

Tuned on dev only: `{"w_sim": 0.8, "w_trust": 0.0, "w_ev": 0.1, "w_scope": 0.1, "w_stale": 0.05, "half_life": 365.0, "trust_mode": "seq", "dev_mrr": 0.6930781701360971, "dev_hit5": 0.823170731707317, "n_dev_tasks": 328, "grid_size": 420}`

| condition | Hit@1 | Hit@3 | Hit@5 | MRR | Recall@5 | Stale@5 | Contam@5 |
|---|---|---|---|---|---|---|---|
| RANDOM_LOCAL | 0.016 | 0.047 | 0.081 | 0.051 | 0.056 | 0.000 | 0.000 |
| RECENCY | 0.189 | 0.375 | 0.451 | 0.307 | 0.408 | 0.000 | 0.000 |
| POPULARITY | 0.083 | 0.371 | 0.451 | 0.244 | 0.386 | 0.000 | 0.000 |
| PLAIN_RAG | 0.333 | 0.495 | 0.551 | 0.433 | 0.507 | 0.179 | 0.267 |
| NAIVE_MEMORY | 0.357 | 0.531 | 0.612 | 0.470 | 0.557 | 0.181 | 0.000 |
| SNAPSHOT_RAG | 0.378 | 0.564 | 0.641 | 0.497 | 0.584 | 0.000 | 0.000 |
| EMEMOVERI | 0.406 | 0.607 | 0.684 | 0.532 | 0.621 | 0.000 | 0.000 |
| ABL_no_trust | 0.411 | 0.616 | 0.704 | 0.539 | 0.641 | 0.000 | 0.000 |
| ABL_no_evidence | 0.423 | 0.605 | 0.672 | 0.541 | 0.613 | 0.000 | 0.000 |
| ABL_no_staleness | 0.430 | 0.639 | 0.719 | 0.555 | 0.655 | 0.000 | 0.000 |
| ABL_state_blind | 0.404 | 0.603 | 0.682 | 0.529 | 0.620 | 0.015 | 0.000 |
| ABL_policy_open | 0.406 | 0.607 | 0.684 | 0.532 | 0.621 | 0.000 | 0.000 |
| ABL_policy_local_only | 0.406 | 0.607 | 0.684 | 0.532 | 0.621 | 0.000 | 0.000 |
| ABL_policy_sim_gate | 0.406 | 0.607 | 0.684 | 0.532 | 0.621 | 0.000 | 0.000 |
| ABL_trust_aggclamp | 0.408 | 0.609 | 0.688 | 0.534 | 0.624 | 0.000 | 0.000 |
| ABL_trust_logit | 0.397 | 0.600 | 0.677 | 0.523 | 0.612 | 0.000 | 0.000 |
| ABL_halflife_30d | 0.392 | 0.579 | 0.671 | 0.515 | 0.606 | 0.000 | 0.000 |
| ABL_halflife_365d | 0.422 | 0.628 | 0.699 | 0.545 | 0.638 | 0.000 | 0.000 |
| EMEMOVERI_TUNED | 0.454 | 0.661 | 0.751 | 0.585 | 0.690 | 0.000 | 0.000 |

## Controls
| control | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| SHUFFLED::SNAPSHOT_RAG | 0.020 | 0.130 | 0.076 |
| SHUFFLED::EMEMOVERI | 0.095 | 0.461 | 0.260 |
| LEAKED::SNAPSHOT_RAG | 0.999 | 1.000 | 0.999 |
| LEAKED::EMEMOVERI | 0.999 | 1.000 | 0.999 |

## Per repository (test)
| repo (n) | cond | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|
| browser-use (400) | POPULARITY | 0.147 | 0.463 | 0.289 |
| browser-use (400) | RECENCY | 0.255 | 0.505 | 0.372 |
| browser-use (400) | PLAIN_RAG | 0.258 | 0.502 | 0.364 |
| browser-use (400) | SNAPSHOT_RAG | 0.315 | 0.605 | 0.442 |
| browser-use (400) | EMEMOVERI | 0.393 | 0.698 | 0.523 |
| browser-use (400) | EMEMOVERI_TUNED | 0.403 | 0.698 | 0.532 |
| fastapi (172) | POPULARITY | 0.012 | 0.640 | 0.275 |
| fastapi (172) | RECENCY | 0.081 | 0.576 | 0.294 |
| fastapi (172) | PLAIN_RAG | 0.198 | 0.384 | 0.293 |
| fastapi (172) | SNAPSHOT_RAG | 0.244 | 0.541 | 0.387 |
| fastapi (172) | EMEMOVERI | 0.297 | 0.645 | 0.464 |
| fastapi (172) | EMEMOVERI_TUNED | 0.326 | 0.715 | 0.493 |
| yt-dlp (196) | POPULARITY | 0.015 | 0.260 | 0.125 |
| yt-dlp (196) | RECENCY | 0.148 | 0.230 | 0.187 |
| yt-dlp (196) | PLAIN_RAG | 0.607 | 0.796 | 0.697 |
| yt-dlp (196) | SNAPSHOT_RAG | 0.622 | 0.801 | 0.708 |
| yt-dlp (196) | EMEMOVERI | 0.531 | 0.689 | 0.611 |
| yt-dlp (196) | EMEMOVERI_TUNED | 0.673 | 0.893 | 0.773 |

## Paired comparisons (test)
| A | B | metric | diff | 95% CI | p (McNemar) | p (Holm) |
|---|---|---|---|---|---|---|
| EMEMOVERI | PLAIN_RAG | hit1 | +0.073 | [+0.036, +0.109] | 0.0001 | 0.0008 |
| EMEMOVERI | PLAIN_RAG | hit5 | +0.133 | [+0.090, +0.176] | 0.0000 | 0.0000 |
| EMEMOVERI | PLAIN_RAG | rr | +0.099 | [+0.067, +0.130] | - | - |
| EMEMOVERI | PLAIN_RAG | stale5 | -0.179 | [-0.195, -0.165] | - | - |
| EMEMOVERI | NAIVE_MEMORY | hit1 | +0.049 | [+0.013, +0.086] | 0.0094 | 0.0282 |
| EMEMOVERI | NAIVE_MEMORY | hit5 | +0.072 | [+0.029, +0.115] | 0.0014 | 0.0070 |
| EMEMOVERI | NAIVE_MEMORY | rr | +0.062 | [+0.032, +0.093] | - | - |
| EMEMOVERI | NAIVE_MEMORY | stale5 | -0.181 | [-0.198, -0.165] | - | - |
| EMEMOVERI | SNAPSHOT_RAG | hit1 | +0.029 | [-0.009, +0.065] | 0.1413 | 0.1413 |
| EMEMOVERI | SNAPSHOT_RAG | hit5 | +0.043 | [+0.000, +0.086] | 0.0561 | 0.1121 |
| EMEMOVERI | SNAPSHOT_RAG | rr | +0.035 | [+0.004, +0.065] | - | - |
| EMEMOVERI | SNAPSHOT_RAG | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI | POPULARITY | hit1 | +0.323 | [+0.285, +0.359] | 0.0000 | 0.0000 |
| EMEMOVERI | POPULARITY | hit5 | +0.233 | [+0.201, +0.266] | 0.0000 | 0.0000 |
| EMEMOVERI | POPULARITY | rr | +0.288 | [+0.258, +0.318] | - | - |
| EMEMOVERI | POPULARITY | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | hit1 | +0.077 | [+0.052, +0.103] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | hit5 | +0.111 | [+0.082, +0.142] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | rr | +0.087 | [+0.069, +0.105] | - | - |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI_TUNED | EMEMOVERI | hit1 | +0.048 | [+0.020, +0.077] | 0.0016 | 0.0070 |
| EMEMOVERI_TUNED | EMEMOVERI | hit5 | +0.068 | [+0.038, +0.100] | 0.0000 | 0.0003 |
| EMEMOVERI_TUNED | EMEMOVERI | rr | +0.053 | [+0.030, +0.076] | - | - |
| EMEMOVERI_TUNED | EMEMOVERI | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI_TUNED | POPULARITY | hit1 | +0.371 | [+0.331, +0.411] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | POPULARITY | hit5 | +0.301 | [+0.258, +0.345] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | POPULARITY | rr | +0.341 | [+0.306, +0.375] | - | - |
| EMEMOVERI_TUNED | POPULARITY | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| SNAPSHOT_RAG | PLAIN_RAG | hit1 | +0.044 | [+0.031, +0.060] | 0.0000 | 0.0000 |
| SNAPSHOT_RAG | PLAIN_RAG | hit5 | +0.090 | [+0.070, +0.111] | 0.0000 | 0.0000 |
| SNAPSHOT_RAG | PLAIN_RAG | rr | +0.064 | [+0.054, +0.074] | - | - |
| SNAPSHOT_RAG | PLAIN_RAG | stale5 | -0.179 | [-0.195, -0.165] | - | - |

## Ablations (delta vs EMEMOVERI default)
| ablation | dHit@5 [95% CI] | dMRR [95% CI] |
|---|---|---|
| ABL_no_trust | +0.021 [+0.010, +0.033] | +0.007 [+0.001, +0.013] |
| ABL_no_evidence | -0.012 [-0.047, +0.023] | +0.009 [-0.016, +0.033] |
| ABL_no_staleness | +0.035 [+0.018, +0.052] | +0.023 [+0.011, +0.037] |
| ABL_state_blind | -0.001 [-0.004, +0.000] | -0.003 [-0.005, -0.001] |
| ABL_policy_open | +0.000 [+0.000, +0.000] | -0.000 [-0.001, -0.000] |
| ABL_policy_local_only | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| ABL_policy_sim_gate | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| ABL_trust_aggclamp | +0.004 [+0.000, +0.009] | +0.002 [-0.001, +0.006] |
| ABL_trust_logit | -0.007 [-0.016, +0.003] | -0.009 [-0.016, -0.003] |
| ABL_halflife_30d | -0.013 [-0.027, +0.001] | -0.017 [-0.027, -0.008] |
| ABL_halflife_365d | +0.016 [+0.003, +0.029] | +0.013 [+0.004, +0.022] |

## Selective prediction (top-1)
| system | acc(all) | AURC gap | AURC score | AURC trust | AUROC gap | AUROC score | AUROC trust | gap>=0.20 coverage / acc |
|---|---|---|---|---|---|---|---|---|
| EMEMOVERI | 0.406 | 0.376 | 0.391 | 0.586 | 0.751 | 0.752 | 0.500 | 0.003 / 1.000 |
| EMEMOVERI_TUNED | 0.454 | 0.307 | 0.349 | 0.503 | 0.783 | 0.755 | 0.489 | 0.051 / 0.974 |
| SNAPSHOT_RAG | 0.378 | 0.362 | 0.399 | 0.548 | 0.820 | 0.797 | 0.570 | 0.062 / 0.938 |

EMEMOVERI trust calibration on top-1: ECE=0.594, Brier=0.594, mean trust=1.000

EMEMOVERI_TUNED trust calibration on top-1: ECE=0.545, Brier=0.545, mean trust=0.995
