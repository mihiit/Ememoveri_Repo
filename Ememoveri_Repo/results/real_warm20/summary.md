# Real-repository benchmark summary (test n=839, dev n=359)

Tuned on dev only: `{"w_sim": 0.8, "w_trust": 0.0, "w_ev": 0.1, "w_scope": 0.1, "w_stale": 0.0, "half_life": 90.0, "trust_mode": "seq", "dev_mrr": 0.6226216216466912, "dev_hit5": 0.766016713091922, "n_dev_tasks": 359, "grid_size": 420}`

| condition | Hit@1 | Hit@3 | Hit@5 | MRR | Recall@5 | Stale@5 | Contam@5 |
|---|---|---|---|---|---|---|---|
| RANDOM_LOCAL | 0.014 | 0.046 | 0.067 | 0.051 | 0.054 | 0.000 | 0.000 |
| RECENCY | 0.145 | 0.347 | 0.435 | 0.275 | 0.388 | 0.000 | 0.000 |
| POPULARITY | 0.087 | 0.409 | 0.493 | 0.262 | 0.427 | 0.000 | 0.000 |
| PLAIN_RAG | 0.350 | 0.487 | 0.541 | 0.439 | 0.500 | 0.164 | 0.269 |
| NAIVE_MEMORY | 0.373 | 0.528 | 0.608 | 0.480 | 0.557 | 0.155 | 0.000 |
| SNAPSHOT_RAG | 0.393 | 0.558 | 0.638 | 0.505 | 0.585 | 0.000 | 0.000 |
| EMEMOVERI | 0.412 | 0.613 | 0.681 | 0.535 | 0.617 | 0.000 | 0.000 |
| ABL_no_trust | 0.414 | 0.620 | 0.696 | 0.539 | 0.633 | 0.000 | 0.000 |
| ABL_no_evidence | 0.428 | 0.605 | 0.683 | 0.544 | 0.629 | 0.000 | 0.000 |
| ABL_no_staleness | 0.429 | 0.629 | 0.712 | 0.551 | 0.650 | 0.000 | 0.000 |
| ABL_state_blind | 0.410 | 0.609 | 0.678 | 0.533 | 0.614 | 0.018 | 0.000 |
| ABL_policy_open | 0.412 | 0.613 | 0.681 | 0.534 | 0.617 | 0.000 | 0.000 |
| ABL_policy_local_only | 0.412 | 0.613 | 0.681 | 0.535 | 0.617 | 0.000 | 0.000 |
| ABL_policy_sim_gate | 0.412 | 0.613 | 0.681 | 0.535 | 0.617 | 0.000 | 0.000 |
| ABL_trust_aggclamp | 0.412 | 0.614 | 0.682 | 0.535 | 0.618 | 0.000 | 0.000 |
| ABL_trust_logit | 0.403 | 0.602 | 0.667 | 0.522 | 0.602 | 0.000 | 0.000 |
| ABL_halflife_30d | 0.392 | 0.585 | 0.659 | 0.515 | 0.595 | 0.000 | 0.000 |
| ABL_halflife_365d | 0.424 | 0.621 | 0.701 | 0.545 | 0.642 | 0.000 | 0.000 |
| EMEMOVERI_TUNED | 0.460 | 0.673 | 0.750 | 0.590 | 0.693 | 0.000 | 0.000 |

## Controls
| control | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| SHUFFLED::SNAPSHOT_RAG | 0.020 | 0.130 | 0.084 |
| SHUFFLED::EMEMOVERI | 0.106 | 0.495 | 0.280 |
| LEAKED::SNAPSHOT_RAG | 0.999 | 1.000 | 0.999 |
| LEAKED::EMEMOVERI | 0.996 | 1.000 | 0.998 |

## Per repository (test)
| repo (n) | cond | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|
| browser-use (400) | POPULARITY | 0.158 | 0.510 | 0.314 |
| browser-use (400) | RECENCY | 0.207 | 0.517 | 0.340 |
| browser-use (400) | PLAIN_RAG | 0.292 | 0.475 | 0.378 |
| browser-use (400) | SNAPSHOT_RAG | 0.347 | 0.590 | 0.461 |
| browser-use (400) | EMEMOVERI | 0.410 | 0.703 | 0.538 |
| browser-use (400) | EMEMOVERI_TUNED | 0.412 | 0.715 | 0.548 |
| fastapi (209) | POPULARITY | 0.010 | 0.617 | 0.271 |
| fastapi (209) | RECENCY | 0.077 | 0.545 | 0.285 |
| fastapi (209) | PLAIN_RAG | 0.230 | 0.416 | 0.324 |
| fastapi (209) | SNAPSHOT_RAG | 0.278 | 0.579 | 0.415 |
| fastapi (209) | EMEMOVERI | 0.301 | 0.651 | 0.466 |
| fastapi (209) | EMEMOVERI_TUNED | 0.335 | 0.694 | 0.499 |
| yt-dlp (230) | POPULARITY | 0.035 | 0.352 | 0.163 |
| yt-dlp (230) | RECENCY | 0.100 | 0.191 | 0.153 |
| yt-dlp (230) | PLAIN_RAG | 0.561 | 0.770 | 0.650 |
| yt-dlp (230) | SNAPSHOT_RAG | 0.578 | 0.774 | 0.665 |
| yt-dlp (230) | EMEMOVERI | 0.517 | 0.670 | 0.591 |
| yt-dlp (230) | EMEMOVERI_TUNED | 0.657 | 0.861 | 0.745 |

## Paired comparisons (test)
| A | B | metric | diff | 95% CI | p (McNemar) | p (Holm) |
|---|---|---|---|---|---|---|
| EMEMOVERI | PLAIN_RAG | hit1 | +0.062 | [+0.025, +0.098] | 0.0012 | 0.0061 |
| EMEMOVERI | PLAIN_RAG | hit5 | +0.139 | [+0.098, +0.182] | 0.0000 | 0.0000 |
| EMEMOVERI | PLAIN_RAG | rr | +0.096 | [+0.064, +0.128] | - | - |
| EMEMOVERI | PLAIN_RAG | stale5 | -0.164 | [-0.178, -0.150] | - | - |
| EMEMOVERI | NAIVE_MEMORY | hit1 | +0.039 | [+0.004, +0.076] | 0.0407 | 0.1221 |
| EMEMOVERI | NAIVE_MEMORY | hit5 | +0.073 | [+0.032, +0.116] | 0.0007 | 0.0042 |
| EMEMOVERI | NAIVE_MEMORY | rr | +0.055 | [+0.024, +0.087] | - | - |
| EMEMOVERI | NAIVE_MEMORY | stale5 | -0.155 | [-0.170, -0.141] | - | - |
| EMEMOVERI | SNAPSHOT_RAG | hit1 | +0.019 | [-0.017, +0.056] | 0.3369 | 0.3369 |
| EMEMOVERI | SNAPSHOT_RAG | hit5 | +0.043 | [+0.001, +0.085] | 0.0474 | 0.1221 |
| EMEMOVERI | SNAPSHOT_RAG | rr | +0.029 | [-0.002, +0.061] | - | - |
| EMEMOVERI | SNAPSHOT_RAG | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI | POPULARITY | hit1 | +0.325 | [+0.288, +0.361] | 0.0000 | 0.0000 |
| EMEMOVERI | POPULARITY | hit5 | +0.187 | [+0.156, +0.218] | 0.0000 | 0.0000 |
| EMEMOVERI | POPULARITY | rr | +0.273 | [+0.245, +0.300] | - | - |
| EMEMOVERI | POPULARITY | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | hit1 | +0.067 | [+0.044, +0.091] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | hit5 | +0.112 | [+0.085, +0.141] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | rr | +0.084 | [+0.068, +0.102] | - | - |
| EMEMOVERI_TUNED | SNAPSHOT_RAG | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI_TUNED | EMEMOVERI | hit1 | +0.048 | [+0.018, +0.077] | 0.0024 | 0.0095 |
| EMEMOVERI_TUNED | EMEMOVERI | hit5 | +0.069 | [+0.037, +0.101] | 0.0000 | 0.0003 |
| EMEMOVERI_TUNED | EMEMOVERI | rr | +0.055 | [+0.031, +0.079] | - | - |
| EMEMOVERI_TUNED | EMEMOVERI | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| EMEMOVERI_TUNED | POPULARITY | hit1 | +0.373 | [+0.334, +0.411] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | POPULARITY | hit5 | +0.256 | [+0.216, +0.296] | 0.0000 | 0.0000 |
| EMEMOVERI_TUNED | POPULARITY | rr | +0.328 | [+0.295, +0.360] | - | - |
| EMEMOVERI_TUNED | POPULARITY | stale5 | +0.000 | [+0.000, +0.000] | - | - |
| SNAPSHOT_RAG | PLAIN_RAG | hit1 | +0.043 | [+0.030, +0.057] | 0.0000 | 0.0000 |
| SNAPSHOT_RAG | PLAIN_RAG | hit5 | +0.097 | [+0.076, +0.117] | 0.0000 | 0.0000 |
| SNAPSHOT_RAG | PLAIN_RAG | rr | +0.067 | [+0.057, +0.077] | - | - |
| SNAPSHOT_RAG | PLAIN_RAG | stale5 | -0.164 | [-0.178, -0.150] | - | - |

## Ablations (delta vs EMEMOVERI default)
| ablation | dHit@5 [95% CI] | dMRR [95% CI] |
|---|---|---|
| ABL_no_trust | +0.015 [+0.007, +0.025] | +0.004 [-0.002, +0.009] |
| ABL_no_evidence | +0.002 [-0.032, +0.037] | +0.010 [-0.016, +0.035] |
| ABL_no_staleness | +0.031 [+0.014, +0.048] | +0.016 [+0.004, +0.029] |
| ABL_state_blind | -0.002 [-0.007, +0.002] | -0.002 [-0.004, +0.000] |
| ABL_policy_open | +0.000 [+0.000, +0.000] | -0.001 [-0.001, -0.000] |
| ABL_policy_local_only | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| ABL_policy_sim_gate | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| ABL_trust_aggclamp | +0.001 [-0.002, +0.005] | +0.000 [-0.002, +0.003] |
| ABL_trust_logit | -0.013 [-0.023, -0.004] | -0.013 [-0.020, -0.007] |
| ABL_halflife_30d | -0.021 [-0.035, -0.008] | -0.020 [-0.029, -0.011] |
| ABL_halflife_365d | +0.020 [+0.008, +0.033] | +0.010 [+0.001, +0.019] |

## Selective prediction (top-1)
| system | acc(all) | AURC gap | AURC score | AURC trust | AUROC gap | AUROC score | AUROC trust | gap>=0.20 coverage / acc |
|---|---|---|---|---|---|---|---|---|
| EMEMOVERI | 0.412 | 0.369 | 0.369 | 0.575 | 0.751 | 0.773 | 0.496 | 0.006 / 1.000 |
| EMEMOVERI_TUNED | 0.460 | 0.308 | 0.331 | 0.493 | 0.774 | 0.763 | 0.481 | 0.048 / 0.975 |
| SNAPSHOT_RAG | 0.393 | 0.354 | 0.381 | 0.508 | 0.810 | 0.787 | 0.568 | 0.060 / 0.940 |

EMEMOVERI trust calibration on top-1: ECE=0.589, Brier=0.588, mean trust=0.999

EMEMOVERI_TUNED trust calibration on top-1: ECE=0.536, Brier=0.539, mean trust=0.992
