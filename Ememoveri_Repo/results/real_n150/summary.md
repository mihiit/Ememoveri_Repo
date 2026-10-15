# Real-repository benchmark summary (test n=315, dev n=135)

Tuned on dev only: `{"w_sim": 0.7, "w_trust": 0.1, "w_ev": 0.1, "w_scope": 0.1, "w_stale": 0.0, "half_life": 90.0, "trust_mode": "seq", "dev_mrr": 0.6653538025760248, "dev_hit5": 0.837037037037037, "n_dev_tasks": 135, "grid_size": 420}`

| condition | Hit@1 | Hit@3 | Hit@5 | MRR | Recall@5 | Stale@5 | Contam@5 |
|---|---|---|---|---|---|---|---|
| RANDOM_LOCAL | 0.003 | 0.032 | 0.067 | 0.038 | 0.056 | 0.000 | 0.000 |
| RECENCY | 0.175 | 0.352 | 0.438 | 0.298 | 0.401 | 0.000 | 0.000 |
| POPULARITY | 0.060 | 0.416 | 0.492 | 0.245 | 0.424 | 0.000 | 0.000 |
| PLAIN_RAG | 0.375 | 0.505 | 0.581 | 0.463 | 0.533 | 0.178 | 0.263 |
| NAIVE_MEMORY | 0.403 | 0.546 | 0.632 | 0.504 | 0.571 | 0.162 | 0.000 |
| SNAPSHOT_RAG | 0.425 | 0.587 | 0.663 | 0.531 | 0.601 | 0.000 | 0.000 |
| EMEMOVERI | 0.425 | 0.616 | 0.689 | 0.546 | 0.630 | 0.000 | 0.000 |
| ABL_no_trust | 0.422 | 0.622 | 0.705 | 0.546 | 0.642 | 0.000 | 0.000 |
| ABL_no_evidence | 0.473 | 0.632 | 0.717 | 0.582 | 0.655 | 0.000 | 0.000 |
| ABL_no_staleness | 0.422 | 0.644 | 0.714 | 0.553 | 0.654 | 0.000 | 0.000 |
| ABL_state_blind | 0.425 | 0.613 | 0.689 | 0.545 | 0.628 | 0.013 | 0.000 |
| ABL_policy_open | 0.425 | 0.616 | 0.689 | 0.545 | 0.630 | 0.000 | 0.000 |
| ABL_policy_local_only | 0.425 | 0.616 | 0.689 | 0.546 | 0.630 | 0.000 | 0.000 |
| ABL_policy_sim_gate | 0.425 | 0.616 | 0.689 | 0.546 | 0.630 | 0.000 | 0.000 |
| ABL_trust_aggclamp | 0.425 | 0.619 | 0.695 | 0.547 | 0.634 | 0.000 | 0.000 |
| ABL_trust_logit | 0.419 | 0.613 | 0.689 | 0.539 | 0.626 | 0.000 | 0.000 |
| ABL_halflife_30d | 0.413 | 0.622 | 0.686 | 0.537 | 0.622 | 0.000 | 0.000 |
| ABL_halflife_365d | 0.429 | 0.632 | 0.695 | 0.549 | 0.637 | 0.000 | 0.000 |
| EMEMOVERI_TUNED | 0.470 | 0.683 | 0.775 | 0.600 | 0.715 | 0.000 | 0.000 |

## Controls
| control | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| SHUFFLED::SNAPSHOT_RAG | 0.010 | 0.143 | 0.080 |
| SHUFFLED::EMEMOVERI | 0.105 | 0.508 | 0.283 |
| LEAKED::SNAPSHOT_RAG | 1.000 | 1.000 | 1.000 |
| LEAKED::EMEMOVERI | 0.997 | 1.000 | 0.998 |

## Per repository (test)
| repo (n) | cond | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|
| browser-use (150) | POPULARITY | 0.120 | 0.440 | 0.266 |
| browser-use (150) | RECENCY | 0.273 | 0.493 | 0.384 |
| browser-use (150) | PLAIN_RAG | 0.353 | 0.560 | 0.437 |
| browser-use (150) | SNAPSHOT_RAG | 0.407 | 0.647 | 0.507 |
| browser-use (150) | EMEMOVERI | 0.420 | 0.680 | 0.537 |
| browser-use (150) | EMEMOVERI_TUNED | 0.453 | 0.727 | 0.576 |
| fastapi (89) | POPULARITY | 0.011 | 0.719 | 0.306 |
| fastapi (89) | RECENCY | 0.056 | 0.539 | 0.266 |
| fastapi (89) | PLAIN_RAG | 0.180 | 0.404 | 0.284 |
| fastapi (89) | SNAPSHOT_RAG | 0.258 | 0.528 | 0.390 |
| fastapi (89) | EMEMOVERI | 0.303 | 0.730 | 0.493 |
| fastapi (89) | EMEMOVERI_TUNED | 0.315 | 0.742 | 0.488 |
| yt-dlp (76) | POPULARITY | 0.000 | 0.329 | 0.133 |
| yt-dlp (76) | RECENCY | 0.118 | 0.211 | 0.165 |
| yt-dlp (76) | PLAIN_RAG | 0.645 | 0.829 | 0.723 |
| yt-dlp (76) | SNAPSHOT_RAG | 0.658 | 0.855 | 0.745 |
| yt-dlp (76) | EMEMOVERI | 0.579 | 0.658 | 0.626 |
| yt-dlp (76) | EMEMOVERI_TUNED | 0.684 | 0.908 | 0.778 |

## Paired comparisons (test)
| A | B | metric | diff | 95% CI | p (McNemar) | p (Holm) |
|---|---|---|---|---|---|---|
| EMEMOVERI | PLAIN_RAG | hit1 | +0.051 | [-0.006, +0.108] | 0.1011 | 0.5057 |
| EMEMOVERI | PLAIN_RAG | hit5 | +0.108 | [+0.038, +0.178] | 0.0034 | 0.0270 |
| EMEMOVERI | PLAIN_RAG | rr | +0.083 | [+0.032, +0.135] | - | - |
| EMEMOVERI | PLAIN_RAG | stale5 | -0.178 | [-0.201, -0.155] | - | - |
| EMEMOVERI | NAIVE_MEMORY | hit1 | +0.022 | [-0.035, +0.079] | 0.5052 | 1.0000 |
| EMEMOVERI | NAIVE_MEMORY | hit5 | +0.057 | [-0.010, +0.124] | 0.1172 | 0.5057 |
| EMEMOVERI | NAIVE_MEMORY | rr | +0.042 | [-0.008, +0.091] | - | - |
| EMEMOVERI | NAIVE_MEMORY | stale5 | -0.162 | [-0.186, -0.138] | - | - |
| EMEMOVERI | SNAPSHOT_RAG | hit1 | +0.000 | [-0.054, +0.057] | 1.0000 | 1.0000 |
| EMEMOVERI | SNAPSHOT_RAG | hit5 | +0.025 | [-0.038, +0.092] | 0.5085 | 1.0000 |
| EMEMOVERI | SNAPSHOT_RAG | rr | +0.015 | [-0.034, +0.064] | - | - |
| EMEMOVERI | SNAPSHOT_RAG | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI | POPULARITY | hit1 | +0.365 | [+0.308, +0.422] | 0.0000 | 0.0000 |
| EMEMOVERI | POPULARITY | hit5 | +0.197 | [+0.143, +0.251] | 0.0000 | 0.0000 |
| EMEMOVERI | POPULARITY | rr | +0.301 | [+0.255, +0.347] | - | - |
| EMEMOVERI | POPULARITY | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | hit1 | +0.044 | [+0.006, +0.086] | 0.0436 | 0.3049 |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | hit5 | +0.111 | [+0.067, +0.159] | 0.0000 | 0.0001 |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | rr | +0.069 | [+0.038, +0.098] | - | - |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI_TUNED | EMEMOVERI | hit1 | +0.044 | [+0.003, +0.086] | 0.0436 | 0.3049 |
| EMEMOVERI_TUNED | EMEMOVERI | hit5 | +0.086 | [+0.038, +0.133] | 0.0007 | 0.0066 |
| EMEMOVERI_TUNED | EMEMOVERI | rr | +0.054 | [+0.020, +0.088] | - | - |
| EMEMOVERI_TUNED | EMEMOVERI | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI_TUNED | POPULARITY | hit1 | +0.410 | [+0.349, +0.470] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | POPULARITY | hit5 | +0.283 | [+0.219, +0.346] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | POPULARITY | rr | +0.354 | [+0.303, +0.406] | - | - |
| EMEMOVERI_TUNED | POPULARITY | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| SNAPSHOT_RAG | PLAIN_RAG | hit1 | +0.051 | [+0.029, +0.076] | 0.0000 | 0.0003 |
| SNAPSHOT_RAG | PLAIN_RAG | hit5 | +0.083 | [+0.054, +0.114] | 0.0000 | 0.0000 |
| SNAPSHOT_RAG | PLAIN_RAG | rr | +0.068 | [+0.052, +0.085] | - | - |
| SNAPSHOT_RAG | PLAIN_RAG | stale5 | -0.178 | [-0.201, -0.155] | - | - |

## Ablations (delta vs EMEMOVERI default)
| ablation | dHit@5 [95% CI] | dMRR [95% CI] |
|---|---|---|
| ABL_no_trust | +0.016 [+0.003, +0.032] | -0.000 [-0.008, +0.008] |
| ABL_no_evidence | +0.029 [-0.022, +0.083] | +0.036 [-0.003, +0.075] |
| ABL_no_staleness | +0.025 [+0.003, +0.048] | +0.007 [-0.009, +0.024] |
| ABL_state_blind | +0.000 [+0.000, +0.000] | -0.001 [-0.002, -0.000] |
| ABL_policy_open | +0.000 [+0.000, +0.000] | -0.001 [-0.002, -0.000] |
| ABL_policy_local_only | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| ABL_policy_sim_gate | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| ABL_trust_aggclamp | +0.006 [+0.000, +0.016] | +0.002 [-0.004, +0.009] |
| ABL_trust_logit | +0.000 [-0.016, +0.016] | -0.007 [-0.018, +0.004] |
| ABL_halflife_30d | -0.003 [-0.025, +0.019] | -0.009 [-0.021, +0.002] |
| ABL_halflife_365d | +0.006 [-0.010, +0.025] | +0.003 [-0.005, +0.012] |

## Selective prediction (top-1)
| system | acc(all) | AURC gap | AURC score | AURC trust | AUROC gap | AUROC score | AUROC trust | gap>=0.20 coverage / acc |
|---|---|---|---|---|---|---|---|---|
| EMEMOVERI | 0.425 | 0.362 | 0.360 | 0.588 | 0.758 | 0.774 | 0.500 | 0.013 / 1.000 |
| EMEMOVERI_TUNED | 0.470 | 0.303 | 0.333 | 0.539 | 0.776 | 0.760 | 0.502 | 0.041 / 1.000 |
| SNAPSHOT_RAG | 0.425 | 0.344 | 0.375 | 0.528 | 0.782 | 0.758 | 0.573 | 0.070 / 0.955 |

EMEMOVERI trust calibration on top-1: ECE=0.575, Brier=0.575, mean trust=1.000

EMEMOVERI_TUNED trust calibration on top-1: ECE=0.531, Brier=0.530, mean trust=0.995
