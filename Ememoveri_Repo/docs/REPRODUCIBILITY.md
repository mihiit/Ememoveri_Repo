# Reproducibility

**Verified (2026-10-01, Python 3.12, 1 CPU core, no GPU).** From a clean checkout, with the repositories pinned to the heads in
`results/real/repo_heads.txt` and `results/real_ext/repo_heads_new.txt`:

| regenerated | compared with | result |
|---|---|---|
| `experiments/real_bench.py` + `analyze_real.py` (main benchmark, 1,096 tasks) | `results/real/summary.json` | identical |
| `collect_ext.py --set dev` + `analyze_ext.py` | `results/real_ext/summary_ext.json` | identical (all numbers, including seeded bootstrap intervals) |
| `collect_ext.py --set heldout --slim` + `analyze_heldout.py` | `results/real_ext/summary_heldout.json` | identical (re-checked after the v0.5 channels were added) |
| `collect_ext.py --set multilang/dev/heldout` + `analyze_v05.py` (run twice) | `results/real_ext/summary_v05.json` | identical between runs (seeded bootstrap, deterministic training) |

| `collect_ext.py --set dev/fresh/multilang/heldout` + `analyze_v06.py` (run twice) | `results/real_ext/summary_v06.json` | identical between runs; the v0.5 comparator numbers on the held-out sets were reproduced by the new slim collections |

`ltr_models.pkl` (scikit-learn 1.8) is stored for convenience; `analyze_v05.py` retrains it deterministically from the dev split.

Costs: main benchmark ~1 min; dev collection ~1 min and analysis ~3 min; held-out collection ~3 min (~1.8 GB RAM) and analysis ~3 min;
mining the 12 held-out histories ~4 min the first time (cached in `.cache/`).

## Pinning
The fresh-set heads are in `results/real_ext/repo_heads_fresh.txt` (16 more mirrors). The multi-language heads are in `results/real_ext/repo_heads_multilang.txt` (8 more mirrors, same procedure).
`scripts/reproduce_real.sh` clones bare, blob-less mirrors of the current heads. To reproduce the published numbers, pin each mirror
to the recorded head before running:

    git --git-dir=repos/<name>.git cat-file -t <sha>                      # must print "commit"
    git --git-dir=repos/<name>.git update-ref refs/heads/pinned <sha>
    git --git-dir=repos/<name>.git symbolic-ref HEAD refs/heads/pinned

(The miner reads `HEAD`.) A force-pushed or deleted upstream commit would make a pinned head unavailable.

## Randomness
Task selection is deterministic (evenly spaced). Bootstrap resampling uses fixed seeds. Query degradation is seeded by commit SHA.
`ememoveri/realbench_ext.py::collect` has no stochastic component.

## What cannot be reproduced offline
Issue-text tasks (GitHub API token), patch generation (Anthropic API key), neural dense retrieval. See `docs/RUN_V04.md`.

## Memory
The fresh-set collection stores 5,832 tasks with per-task commit cosines; use `--slim --no-events` (peak ~1.5 GB). Without `--slim` the
three-repository development set needs ~1 GB; the 12-repository sets do not fit in 3 GB without it.

## Configuration freeze
`results/real_ext/ev2_config.json` (sha256 `15cdaae3bc225659f94e45259390d080d048f8ff4ff5c33dd630c51307288eb7`) was produced by
`experiments/tune_ev2.py` on the original dev split and committed with `docs/PROTOCOL_V06.md` before any fresh-set result existed.
