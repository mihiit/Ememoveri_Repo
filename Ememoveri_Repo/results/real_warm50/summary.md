# Real-repository benchmark summary (test n=712, dev n=305)

Tuned on dev only: `{"w_sim": 0.8, "w_trust": 0.0, "w_ev": 0.1, "w_scope": 0.1, "w_stale": 0.0, "half_life": 90.0, "trust_mode": "seq", "dev_mrr": 0.7305235259419902, "dev_hit5": 0.8557377049180328, "n_dev_tasks": 305, "grid_size": 420}`

| condition | Hit@1 | Hit@3 | Hit@5 | MRR | Recall@5 | Stale@5 | Contam@5 |
|---|---|---|---|---|---|---|---|
| RANDOM_LOCAL | 0.006 | 0.025 | 0.039 | 0.033 | 0.030 | 0.000 | 0.000 |
| RECENCY | 0.188 | 0.365 | 0.442 | 0.301 | 0.403 | 0.000 | 0.000 |
| POPULARITY | 0.074 | 0.317 | 0.417 | 0.223 | 0.358 | 0.000 | 0.000 |
| PLAIN_RAG | 0.372 | 0.525 | 0.593 | 0.469 | 0.549 | 0.187 | 0.238 |
| NAIVE_MEMORY | 0.395 | 0.559 | 0.643 | 0.502 | 0.592 | 0.188 | 0.000 |
| SNAPSHOT_RAG | 0.417 | 0.594 | 0.673 | 0.530 | 0.621 | 0.000 | 0.000 |
| EMEMOVERI | 0.449 | 0.614 | 0.680 | 0.553 | 0.623 | 0.000 | 0.000 |
| ABL_no_trust | 0.448 | 0.619 | 0.687 | 0.555 | 0.630 | 0.000 | 0.000 |
| ABL_no_evidence | 0.483 | 0.642 | 0.704 | 0.585 | 0.654 | 0.000 | 0.000 |
| ABL_no_staleness | 0.469 | 0.645 | 0.708 | 0.576 | 0.654 | 0.000 | 0.000 |
| ABL_state_blind | 0.447 | 0.611 | 0.676 | 0.551 | 0.620 | 0.016 | 0.000 |
| ABL_policy_open | 0.449 | 0.614 | 0.680 | 0.553 | 0.623 | 0.000 | 0.000 |
| ABL_policy_local_only | 0.449 | 0.614 | 0.680 | 0.553 | 0.623 | 0.000 | 0.000 |
| ABL_policy_sim_gate | 0.449 | 0.614 | 0.680 | 0.553 | 0.623 | 0.000 | 0.000 |
| ABL_trust_aggclamp | 0.448 | 0.615 | 0.681 | 0.553 | 0.623 | 0.000 | 0.000 |
| ABL_trust_logit | 0.434 | 0.601 | 0.663 | 0.539 | 0.603 | 0.000 | 0.000 |
| ABL_halflife_30d | 0.428 | 0.605 | 0.671 | 0.538 | 0.616 | 0.000 | 0.000 |
| ABL_halflife_365d | 0.458 | 0.631 | 0.691 | 0.563 | 0.637 | 0.000 | 0.000 |
| EMEMOVERI_TUNED | 0.493 | 0.680 | 0.761 | 0.611 | 0.711 | 0.000 | 0.000 |

## Controls
| control | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| SHUFFLED::SNAPSHOT_RAG | 0.022 | 0.114 | 0.075 |
| SHUFFLED::EMEMOVERI | 0.107 | 0.458 | 0.264 |
| LEAKED::SNAPSHOT_RAG | 1.000 | 1.000 | 1.000 |
| LEAKED::EMEMOVERI | 0.999 | 1.000 | 0.999 |

## Per repository (test)
| repo (n) | cond | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|
| browser-use (400) | POPULARITY | 0.125 | 0.395 | 0.246 |
| browser-use (400) | RECENCY | 0.255 | 0.477 | 0.357 |
| browser-use (400) | PLAIN_RAG | 0.350 | 0.595 | 0.457 |
| browser-use (400) | SNAPSHOT_RAG | 0.400 | 0.680 | 0.522 |
| browser-use (400) | EMEMOVERI | 0.468 | 0.713 | 0.573 |
| browser-use (400) | EMEMOVERI_TUNED | 0.485 | 0.757 | 0.603 |
| fastapi (137) | POPULARITY | 0.000 | 0.628 | 0.264 |
| fastapi (137) | RECENCY | 0.073 | 0.584 | 0.289 |
| fastapi (137) | PLAIN_RAG | 0.190 | 0.350 | 0.276 |
| fastapi (137) | SNAPSHOT_RAG | 0.241 | 0.511 | 0.373 |
| fastapi (137) | EMEMOVERI | 0.292 | 0.628 | 0.454 |
| fastapi (137) | EMEMOVERI_TUNED | 0.321 | 0.672 | 0.479 |
| yt-dlp (175) | POPULARITY | 0.017 | 0.303 | 0.138 |
| yt-dlp (175) | RECENCY | 0.126 | 0.251 | 0.181 |
| yt-dlp (175) | PLAIN_RAG | 0.566 | 0.777 | 0.650 |
| yt-dlp (175) | SNAPSHOT_RAG | 0.594 | 0.783 | 0.669 |
| yt-dlp (175) | EMEMOVERI | 0.531 | 0.646 | 0.588 |
| yt-dlp (175) | EMEMOVERI_TUNED | 0.646 | 0.840 | 0.733 |

## Paired comparisons (test)
| A | B | metric | diff | 95% CI | p (McNemar) | p (Holm) |
|---|---|---|---|---|---|---|
| EMEMOVERI | PLAIN_RAG | hit1 | +0.077 | [+0.039, +0.117] | 0.0001 | 0.0008 |
| EMEMOVERI | PLAIN_RAG | hit5 | +0.087 | [+0.044, +0.131] | 0.0001 | 0.0008 |
| EMEMOVERI | PLAIN_RAG | rr | +0.084 | [+0.051, +0.118] | - | - |
| EMEMOVERI | PLAIN_RAG | stale5 | -0.187 | [-0.203, -0.171] | - | - |
| EMEMOVERI | NAIVE_MEMORY | hit1 | +0.055 | [+0.017, +0.094] | 0.0069 | 0.0308 |
| EMEMOVERI | NAIVE_MEMORY | hit5 | +0.037 | [-0.006, +0.080] | 0.1137 | 0.3410 |
| EMEMOVERI | NAIVE_MEMORY | rr | +0.051 | [+0.019, +0.084] | - | - |
| EMEMOVERI | NAIVE_MEMORY | stale5 | -0.188 | [-0.205, -0.171] | - | - |
| EMEMOVERI | SNAPSHOT_RAG | hit1 | +0.032 | [-0.006, +0.070] | 0.1149 | 0.3410 |
| EMEMOVERI | SNAPSHOT_RAG | hit5 | +0.007 | [-0.035, +0.051] | 0.7967 | 0.7967 |
| EMEMOVERI | SNAPSHOT_RAG | rr | +0.024 | [-0.008, +0.057] | - | - |
| EMEMOVERI | SNAPSHOT_RAG | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI | POPULARITY | hit1 | +0.375 | [+0.336, +0.413] | 0.0000 | 0.0000 |
| EMEMOVERI | POPULARITY | hit5 | +0.263 | [+0.226, +0.299] | 0.0000 | 0.0000 |
| EMEMOVERI | POPULARITY | rr | +0.331 | [+0.299, +0.363] | - | - |
| EMEMOVERI | POPULARITY | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | hit1 | +0.076 | [+0.048, +0.103] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | hit5 | +0.088 | [+0.060, +0.118] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | rr | +0.082 | [+0.062, +0.101] | - | - |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI_TUNED | EMEMOVERI | hit1 | +0.044 | [+0.013, +0.073] | 0.0062 | 0.0308 |
| EMEMOVERI_TUNED | EMEMOVERI | hit5 | +0.081 | [+0.048, +0.115] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | EMEMOVERI | rr | +0.058 | [+0.033, +0.082] | - | - |
| EMEMOVERI_TUNED | EMEMOVERI | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI_TUNED | POPULARITY | hit1 | +0.419 | [+0.376, +0.461] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | POPULARITY | hit5 | +0.344 | [+0.298, +0.390] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | POPULARITY | rr | +0.388 | [+0.352, +0.425] | - | - |
| EMEMOVERI_TUNED | POPULARITY | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| SNAPSHOT_RAG | PLAIN_RAG | hit1 | +0.045 | [+0.031, +0.060] | 0.0000 | 0.0000 |
| SNAPSHOT_RAG | PLAIN_RAG | hit5 | +0.080 | [+0.060, +0.100] | 0.0000 | 0.0000 |
| SNAPSHOT_RAG | PLAIN_RAG | rr | +0.060 | [+0.050, +0.071] | - | - |
| SNAPSHOT_RAG | PLAIN_RAG | stale5 | -0.187 | [-0.203, -0.171] | - | - |

## Ablations (delta vs EMEMOVERI default)
| ablation | dHit@5 [95% CI] | dMRR [95% CI] |
|---|---|---|
| ABL_no_trust | +0.007 [-0.001, +0.017] | +0.002 [-0.004, +0.007] |
| ABL_no_evidence | +0.024 [-0.011, +0.058] | +0.032 [+0.005, +0.058] |
| ABL_no_staleness | +0.028 [+0.008, +0.048] | +0.022 [+0.009, +0.036] |
| ABL_state_blind | -0.004 [-0.010, +0.000] | -0.002 [-0.004, -0.000] |
| ABL_policy_open | +0.000 [+0.000, +0.000] | -0.000 [-0.001, -0.000] |
| ABL_policy_local_only | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| ABL_policy_sim_gate | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| ABL_trust_aggclamp | +0.001 [+0.000, +0.004] | -0.000 [-0.003, +0.003] |
| ABL_trust_logit | -0.017 [-0.029, -0.006] | -0.014 [-0.022, -0.007] |
| ABL_halflife_30d | -0.008 [-0.022, +0.006] | -0.016 [-0.024, -0.008] |
| ABL_halflife_365d | +0.011 [-0.003, +0.027] | +0.009 [+0.001, +0.018] |

## Selective prediction (top-1)
| system | acc(all) | AURC gap | AURC score | AURC trust | AUROC gap | AUROC score | AUROC trust | gap>=0.20 coverage / acc |
|---|---|---|---|---|---|---|---|---|
| EMEMOVERI | 0.449 | 0.327 | 0.324 | 0.555 | 0.766 | 0.794 | 0.497 | 0.006 / 1.000 |
| EMEMOVERI_TUNED | 0.493 | 0.271 | 0.304 | 0.508 | 0.789 | 0.776 | 0.495 | 0.052 / 0.973 |
| SNAPSHOT_RAG | 0.417 | 0.330 | 0.364 | 0.524 | 0.806 | 0.792 | 0.585 | 0.067 / 0.958 |

EMEMOVERI trust calibration on top-1: ECE=0.551, Brier=0.551, mean trust=0.999

EMEMOVERI_TUNED trust calibration on top-1: ECE=0.497, Brier=0.500, mean trust=0.990
