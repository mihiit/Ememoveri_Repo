# EmemoVeri: Evidence-Aware Repository Memory for Coding Agents

EmemoVeri is a memory for a coding agent that works on one repository over a long time. It keeps an evidence record for each file and tracks whether what it remembers is still current, so it can cope with renamed or deleted files, reverted changes and outdated knowledge.

This repository holds the implementation, the replay benchmark built from real git history, the pre-written test plans, the stored results and the paper.

> **Paper:** *Retrieving Over Evidence Items: Repository Memory for Coding Agents, Evaluated on Real Git History* (Mihit Nanda, Priyanka Tyagi, IILM University). The manuscript is in [`Ememoveri_Repo/paper/`](Ememoveri_Repo/paper/).

**All code lives in [`Ememoveri_Repo/`](Ememoveri_Repo/).** Run every command below from inside that folder.

## What it does

The benchmark replays the history of an open-source repository. At each step the memory sees only the past, receives the one-line subject of a commit that has not happened yet, and must rank the source files that commit will change. A task counts as a hit if a correct file is among the top five (Hit@5).

- **Claims and evidence.** A claim describes one source file. Evidence items record git events (add, modify, delete, rename, revert) with a type, a relation, a time and a strength. A claim moves between PROVISIONAL, CORROBORATED, REFINED and CONTESTED, and SUPERSEDED is final.
- **EmemoVeri-2.** Each past commit is compared with the query, and a file takes the score of its best-matching commit. Only live files are ranked. Renames carry history to the new path. Two small priors (change frequency, recency) are added.
- **EmemoVeri-1.** The original design, with trust and abstention terms, is kept in the code because the paper reports that those terms carry no useful signal.

## Main results

Confirmatory test on 16 repositories that were fixed before any result existed (5,832 tasks, five languages). Hit@5, macro-averaged over repositories, all systems frozen:

| System | Hit@5 |
|---|---|
| Snapshot retrieval (working tree) | 0.703 |
| EmemoVeri-1, tuned | 0.794 |
| Tuned similarity + state + recency + popularity | 0.792 |
| Gradient-boosted ranker | 0.826 |
| **EmemoVeri-2** | **0.835** |

EmemoVeri-2 beats snapshot retrieval by 0.132 [0.108, 0.155] (better in 15 of 16 repositories), the original design by 0.041 [0.027, 0.057], and the tuned recency-and-popularity baseline by 0.043 [0.027, 0.060]. Its difference from the gradient-boosted ranker (0.009 [-0.001, 0.020]) is small, and no clear difference is established.

**Negative result:** evidence-strength weighting, which looked helpful during design, adds 0.001 [-0.005, 0.007] on the fresh repositories. Hypothesis H3 is not confirmed.

Stored results: [`Ememoveri_Repo/results/real_ext/summary_v06.json`](Ememoveri_Repo/results/real_ext/summary_v06.json).

## Quick start

Tested with Python 3.11 and 3.12 on CPU only.

```bash
cd Ememoveri_Repo
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# Linux / macOS:       source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m pytest -q          # 90 tests
```

The static-word-vector backend (only used by `collect_ext.py`) also needs `pip install spacy` and `python -m spacy download en_core_web_md`.

## Reproducing the results

Step-by-step instructions, expected run times and the pinned repository heads are in [`docs/REPRODUCIBILITY.md`](Ememoveri_Repo/docs/REPRODUCIBILITY.md). In short:

1. Clone the evaluation repositories with `scripts/reproduce_real.sh`.
2. Pin each one to the head recorded under `results/real/` and `results/real_ext/`.
3. Run the collection and analysis scripts in `experiments/`.

The configuration used for the confirmatory test is frozen in `results/real_ext/ev2_config.json`. Its SHA-256 hash is recorded in `docs/REPRODUCIBILITY.md`.

## Test plans written in advance

The held-out and fresh sets were evaluated under plans written before the results existed:

- [`PROTOCOL_V04.md`](Ememoveri_Repo/docs/PROTOCOL_V04.md): held-out Python repositories
- [`PROTOCOL_V05.md`](Ememoveri_Repo/docs/PROTOCOL_V05.md): multi-language repositories
- [`PROTOCOL_V06.md`](Ememoveri_Repo/docs/PROTOCOL_V06.md): the 16 fresh repositories (confirmatory)

[`AUDIT.md`](Ememoveri_Repo/docs/AUDIT.md) describes the defects found and fixed in the first implementation.

## Repository layout

```
Ememoveri_Repo/
  ememoveri/     core package (evidence ledger, state engine, retrieval, baselines, statistics, metrics, git mining)
  experiments/   collection and analysis scripts for each benchmark set
  tests/         90 unit and regression tests
  docs/          benchmark description, protocols, audit, reproducibility notes
  results/       stored results and figures
  paper/         manuscript sources and figures
  pilot/         early pilot on a single repository
  scripts/       helper scripts for cloning and pinning repositories
```

## Limitations

- No language model is in the loop, and queries are commit subjects, not issue reports. Results say nothing about patch quality.
- Ranking is at file level. All evaluation repositories are public open-source projects.
- A neural sentence encoder was not run.
- Contradiction handling (the CONTESTED state, API replacement, documentation-versus-code conflicts) is not validated, and neither is any trust model built on it.
- Tasks using real issue text and retrieval-conditioned patch generation need a GitHub token and an API key. Scripts exist (see [`docs/RUN_V04.md`](Ememoveri_Repo/docs/RUN_V04.md)), but those experiments have **not** been run and no results from them are reported.

## Citation

See [`CITATION.cff`](Ememoveri_Repo/CITATION.cff).

## License

See [`LICENSE`](Ememoveri_Repo/LICENSE).
