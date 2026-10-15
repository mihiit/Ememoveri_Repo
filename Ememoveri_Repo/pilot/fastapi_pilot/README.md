# fastapi agent pilot (Sep 2026, pre-dates the v0.2 audit)

A small LLM-in-the-loop pilot that predates the real-history benchmark. It is kept for provenance and as a starting point for the
end-to-end experiment; **it is not evidence for or against EmemoVeri** and is not used in the paper. Read the caveats first.

## What it did
* `build_pilot_tasks.py` / `candidates.csv` / `pilot_tasks.json` -- 9 hand-picked fastapi commits tagged by category
  (DEPRECATION, SUPERSESSION, ...), with parent commit, changed files and timestamp.
* `build_pilot_memory.py` -- per task, mine only commits at or before the cutoff for the declared changed files (PyDriller), build
  claims with the EmemoVeri engine, check the leakage guard. Output: `pilot_memory/`.
* `build_pilot_agent_run.py`, `run_pilot_grid.py` -- prompt a model with NO_MEMORY / PLAIN_RAG / NAIVE_MEMORY / EMEMOVERI context
  and write a patch (Groq or Ollama models). Output: `pilot_grid_results*.csv|jsonl`, `pilot_agent_runs/`.
* `check_patch.py`, `analyze_pilot_results.py` -- patch well-formedness taxonomy and per-model / per-condition tables
  (`analysis_baseline.txt`).

## Caveats (why it is not a result)
1. **Outcome = patch applies cleanly (`git apply`)**, not "the repository's tests pass". A patch can apply and still be wrong.
2. **Tiny and noisy:** 9 tasks; 198 unique cells, of which 60 were infrastructure failures (rate limits, timeouts, model errors) and
   14 were truncated outputs that looked like successes. Per-cell n is 5-9, so every confidence interval is very wide.
3. **Memory was built with the v0.1 code**, before the audit fixed the leakage guard, terminal SUPERSEDED state and staleness clock
   (`docs/AUDIT.md`). `pilot_memory/` here is the original; rebuilding with the current library works
   (`PYTHONPATH=<repo> python build_pilot_memory.py` from inside a fastapi checkout) and will give different claim states.
4. Models are small or open-weight; no repeated runs per cell.

## How to run (from inside a fastapi clone)
```bash
pip install pydriller groq            # + GROQ_API_KEY, or run Ollama locally
cp <this dir>/*.py <this dir>/pilot_tasks.json <fastapi clone>/ && cd <fastapi clone>
python build_pilot_memory.py
python run_pilot_grid.py --models openai/gpt-oss-120b --conditions NO_MEMORY PLAIN_RAG NAIVE_MEMORY EMEMOVERI
python analyze_pilot_results.py
```
For a test-based, repository-scale version of this idea see `experiments/agent_eval.py` and `docs/RUN_V04.md`.
