> **Mechanism-layer SIMULATIONS (v0.2 code).** These scenarios are generated with features the scorer rewards; they are sanity checks, not real-world evidence. Real-repository results: `results/real/summary.md`.

# EmemoVeri Mechanism-Layer Results Summary

_Generated: 2026-09-30T13:17:05.952615+00:00_

**Scope note:** entries marked `*` (E5-H/M/N/O/U/V/W) are mechanism-layer synthetic stress tests of the memory engine itself, NOT results from a real coding agent evaluated on real repositories. E1-E4 reproduce the controlled mechanism-validation experiments from the project blueprint. The real-repository E5 benchmark (task construction, agent evaluation, multi-model comparison, statistics on coding-task success) is not run by this script -- see `docs/E5_INTEGRATION_GUIDE.md`.

## E1 — State-transition validity (9 scenarios)

```
scenario                         state          summary
------------------------------------------------------------------------------------------
corroboration                    CORROBORATED   {'state': 'CORROBORATED', 'n_corroborates': 1, 'n_contradicts': 0, 'n_refines': 0, 'net_support': 0.9, 'evidence_strength_total': 0.9}
contradiction                    CONTESTED      {'state': 'CONTESTED', 'n_corroborates': 0, 'n_contradicts': 1, 'n_refines': 0, 'net_support': -0.9, 'evidence_strength_total': 0.9}
refinement                       REFINED        {'state': 'REFINED', 'n_corroborates': 0, 'n_contradicts': 0, 'n_refines': 1, 'net_support': 0.7, 'evidence_strength_total': 0.7}
supersession                     SUPERSEDED     {'state': 'SUPERSEDED', 'n_corroborates': 0, 'n_contradicts': 0, 'n_refines': 0, 'net_support': 0.0, 'evidence_strength_total': 0.0}
no_evidence_provisional          PROVISIONAL    {'state': 'PROVISIONAL', 'n_corroborates': 0, 'n_contradicts': 0, 'n_refines': 0, 'net_support': 0, 'evidence_strength_total': 0}
clear_conflict                   CONTESTED      {'state': 'CONTESTED', 'n_corroborates': 0, 'n_contradicts': 1, 'n_refines': 0, 'net_support': -1.4249999999999998, 'evidence_strength_total': 1.4249999999999998}
ambiguous_conflict               CONTESTED      {'state': 'CONTESTED', 'n_corroborates': 1, 'n_contradicts': 1, 'n_refines': 0, 'net_support': 0.0, 'evidence_strength_total': 0.08000000000000002}
balanced_support_contradiction   CONTESTED      {'state': 'CONTESTED', 'n_corroborates': 1, 'n_contradicts': 1, 'n_refines': 0, 'net_support': 0.0, 'evidence_strength_total': 1.8}
stale_vs_recent                  CORROBORATED   {'state': 'CORROBORATED', 'n_corroborates': 2, 'n_contradicts': 0, 'n_refines': 0, 'net_support': 1.23, 'evidence_strength_total': 1.23}
```

## E2 — Evidence-order sensitivity vs order-invariant aggregation

```
Scenarios: 25 | Events/scenario: 4 | Total permutations evaluated: 600
Sequential update  -- mean trust-range across scenarios: 0.353040
Aggregated update  -- mean trust-range across scenarios: 0.000000

Interpretation: a non-zero sequential range with a ~zero aggregated
range demonstrates that the sequential mechanism is order-sensitive
and that pooled aggregation removes that sensitivity (RQ3).
```

## E3 — Staleness-aware retrieval crossover

```
stale_age_days  stale_trust  fresh_trust  winner          gap
-----------------------------------------------------------------
10              0.90         0.30         stale        0.1426
10              0.80         0.40         stale        0.0926
10              0.70         0.50         stale        0.0426
10              0.60         0.60         fresh        0.0074
30              0.90         0.30         stale        0.1294
30              0.80         0.40         stale        0.0794
30              0.70         0.50         stale        0.0294
30              0.60         0.60         fresh        0.0206
60              0.90         0.30         stale        0.1130
60              0.80         0.40         stale        0.0630
60              0.70         0.50         stale        0.0130
60              0.60         0.60         fresh        0.0370
90              0.90         0.30         stale        0.1000
90              0.80         0.40         stale        0.0500
90              0.70         0.50         stale        0.0000
90              0.60         0.60         fresh        0.0500
180             0.90         0.30         stale        0.0750
180             0.80         0.40         stale        0.0250
180             0.70         0.50         fresh        0.0250
180             0.60         0.60         fresh        0.0750
365             0.90         0.30         stale        0.0560
365             0.80         0.40         stale        0.0060
365             0.70         0.50         fresh        0.0440
365             0.60         0.60         fresh        0.0940

Recent preferred in 8/24 = 33.33% of grid comparisons.
This is the crossover surface referenced by E3.5: recency does not
unconditionally win -- sufficiently higher trust on the stale claim
can and does overcome the staleness penalty.
```

## E4 — Cross-repository contamination sensitivity/mitigation

```
policy_mode          contamination_rate   n_grid_cells
------------------------------------------------------------
local_only                         0.00% 20
open                              35.00% 20
similarity_gate_0.7               30.00% 20
similarity_gate_0.9               15.00% 20
local_first_m0.10                  5.00% 20
local_first_m0.20                  0.00% 20

Note: 'open' establishes the unmitigated contamination ceiling.
'local_only' establishes the (zero cross-repo utility) floor.
Compare gate/local_first rates against 'open' to assess mitigation
value, and remember E4.8's finding: a gate can shrink the eligible
pool without shrinking the contamination rate *within* it.
```

## E5-E* — Four-way baseline comparison: NO_MEMORY/PLAIN_RAG/NAIVE_MEMORY/EMEMOVERI (mechanism-layer simulation)

```
condition      correct    correct|resolved  contaminated  abstained  coverage
NO_MEMORY      0.000      nan               0.000         0.000      0.000
PLAIN_RAG      0.240      0.240             0.596         0.000      1.000
NAIVE_MEMORY   0.614      0.614             0.000         0.000      1.000
EMEMOVERI      0.504      1.000             0.000         0.496      0.504

Saved to results/data/e5e_baseline_comparison.json
```

## E5-H* — Contradiction stress (mechanism-layer simulation)

```
E5-H (mechanism-layer contradiction stress) results:
  n_scenarios: 500
  contradiction_detection_rate: 0.964
  resolution_correct_rate_given_resolved: 1.0
  inappropriate_commitment_rate_given_resolved: 0.0
  abstention_rate: 0.514
  abstention_precision: 0.0622568093385214
  coverage: 0.486

Saved to results/data/e5h_contradiction_stress.json
```

## E5-M* — Retrieval quality: P@K/R@K/MRR/NDCG (mechanism-layer simulation)

```
E5-M (mechanism-layer retrieval quality) results:
  n_queries: 200
  k: 5
  precision_at_5: 0.799
  recall_at_5: 0.99875
  mrr: 1.0
  ndcg_at_5: 0.9978980290856048

Saved to results/data/e5m_retrieval_quality.json
```

## E5-N* — Abstention coverage/reliability sweep (mechanism-layer simulation)

```
threshold  coverage   correct|resolved   abst_precision 
0.00       1.000      0.755              nan            
0.05       0.660      0.814              0.03676470588235294
0.10       0.370      0.838              0.01984126984126984
0.15       0.185      0.919              0.015337423312883436
0.20       0.070      1.000              0.013440860215053764
0.25       0.033      1.000              0.012919896640826873
0.30       0.015      1.000              0.012690355329949238
0.40       0.000      nan                0.0125         
0.50       0.000      nan                0.0125         

Saved to results/data/e5n_abstention.json
```

## E5-O* — Trust calibration: ECE/Brier (mechanism-layer simulation)

```
E5-O (mechanism-layer trust calibration): ECE=0.0356, Brier=0.1366
Saved to results/data/e5o_trust_calibration.json
```

## E5-U* — Memory corruption robustness (mechanism-layer simulation)

```
corruption_level   ememoveri_wins   naive_wins
0.00               1.000            0.727
0.10               0.917            0.673
0.25               0.827            0.767
0.50               0.623            0.703
0.75               0.493            0.737
1.00               0.303            0.780

Saved to results/data/e5u_memory_corruption.json
```

## E5-V* — Adversarial memory poisoning (mechanism-layer simulation)

```
E5-V (mechanism-layer adversarial poisoning) — worst-case adversarial win rate by policy:
  open                : 100.0% adversarial win rate across similarity sweep
  local_only          : 0.0% adversarial win rate across similarity sweep
  similarity_gate_0.9 : 42.9% adversarial win rate across similarity sweep
  local_first_m0.15   : 57.1% adversarial win rate across similarity sweep
  local_first_m0.30   : 0.0% adversarial win rate across similarity sweep

Saved to results/data/e5v_memory_poisoning.json
```

## E5-W* — Long-horizon memory evolution (mechanism-layer simulation)

```
E5-W (mechanism-layer long-horizon evolution):
  initial         state=PROVISIONAL    trust=0.5000 n_evidence=0
  corroboration   state=CORROBORATED   trust=0.8150 n_evidence=1
  refinement      state=REFINED        trust=1.0000 n_evidence=2
  contradiction   state=CONTESTED      trust=0.3350 n_evidence=3
  supersession    state=SUPERSEDED     trust=0.3350 n_evidence=4
State-consistency check passed: True
Matches expected demo lifecycle: True

Saved to results/data/e5w_long_horizon.json
```

## Generated figures

- `figures/fig2_state_transition_diagram.png`
- `figures/fig3_e2_order_sensitivity.png`
- `figures/fig4_e3_staleness_crossover.png`
- `figures/fig5_e4_contamination_curves.png`
- `figures/fig_e5n_abstention_tradeoff.png`
- `figures/fig_e5o_trust_calibration.png`
- `figures/fig_e5u_corruption_degradation.png`
- `figures/fig_e5w_long_horizon_trajectory.png`
- `figures/fig_e5e_baseline_comparison.png`
