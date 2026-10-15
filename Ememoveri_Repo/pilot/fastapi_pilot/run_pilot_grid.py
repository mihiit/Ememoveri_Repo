"""
run_pilot_grid.py

Runs the full pilot grid: every task in pilot_tasks.json x every model you
specify x every memory condition (NO_MEMORY / PLAIN_RAG / NAIVE_MEMORY /
EMEMOVERI), unattended. For each cell it:

  1. Checks out the task's before_commit automatically (no more manual
     `git checkout` between tasks -- this was a repeated source of false
     failures earlier in the pilot).
  2. Calls the model via build_pilot_agent_run's machinery (full_file mode
     by default -- the mechanically-diffed approach that fixed every
     format-level failure found earlier in this pilot).
  3. Validates the result the same way check_patch.py does (hunk arithmetic
     + git apply --check), so every result already has your same
     four-category error taxonomy applied automatically.
  4. Appends one row to pilot_grid_results.csv and one full record to
     pilot_grid_results.jsonl -- so a crash partway through loses nothing
     already completed, and you can resume by re-running (it skips any
     (task, model, condition) already present in the CSV).

Requires: pip install groq
Requires: GROQ_API_KEY set in your environment
Run from inside your repo checkout, next to pilot_tasks.json and pilot_memory/.

Usage:
    python run_pilot_grid.py --models openai/gpt-oss-120b qwen/qwen3.6-27b
    python run_pilot_grid.py --models openai/gpt-oss-120b --tasks fastapi_002 fastapi_006
    python run_pilot_grid.py --models openai/gpt-oss-120b --conditions NO_MEMORY EMEMOVERI
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import build_pilot_agent_run as bpar  # reuses all the machinery already built and tested

TASKS_PATH = "pilot_tasks.json"
CSV_PATH = "pilot_grid_results.csv"
JSONL_PATH = "pilot_grid_results.jsonl"

CSV_FIELDS = [
    "task_id", "category", "model", "provider", "condition", "output_mode",
    "before_commit", "finish_reason",
    "looks_like_a_diff", "hunks_well_formed", "hunk_arithmetic_ok",
    "git_apply_ok", "error_category", "notes",
]

CONDITIONS = ["NO_MEMORY", "PLAIN_RAG", "NAIVE_MEMORY", "EMEMOVERI"]


def run_git(args):
    result = subprocess.run(["git"] + args, capture_output=True, encoding="utf-8", errors="replace")
    return result.returncode, (result.stdout or "").strip(), (result.stderr or "").strip()


def checkout(sha: str) -> bool:
    code, _, err = run_git(["checkout", "-q", sha])
    if code != 0:
        print(f"    ERROR: git checkout {sha[:12]} failed: {err}", file=sys.stderr)
        return False
    return True


def already_done(task_id: str, model: str, provider: str, condition: str) -> bool:
    """A row only counts as 'done' if it reflects a genuine completed
    attempt -- an 'API error' row (rate limit, dead model ID, transient
    network failure) is NOT a real result and must be retried, or fixing
    the underlying cause (e.g. a corrected model ID) would never actually
    take effect on re-run."""
    if not os.path.exists(CSV_PATH):
        return False
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row["task_id"] == task_id and row["model"] == model
                    and row.get("provider", "groq") == provider
                    and row["condition"] == condition
                    and row.get("error_category") != "API error"):
                return True
    return False


def append_csv_row(row: dict):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl_record(record: dict):
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_diff(diff_text: str) -> dict:
    """Same checks as check_patch.py, inlined so the grid runner doesn't
    need to shell out to a second script per cell."""
    is_diff = bpar.looks_like_a_diff(diff_text)
    hunks_ok = bpar.has_well_formed_hunk_headers(diff_text) if is_diff else False

    arithmetic_ok = True
    if is_diff:
        import re
        header_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        lines = diff_text.splitlines()
        current = None
        for line in lines:
            m = header_re.match(line)
            if m:
                if current is not None:
                    old_ok = current["context"] + current["removed"] == current["declared_old"]
                    new_ok = current["context"] + current["added"] == current["declared_new"]
                    if not (old_ok and new_ok):
                        arithmetic_ok = False
                old_c, old_n, new_c, new_n = m.groups()
                current = {
                    "declared_old": int(old_n) if old_n else 1,
                    "declared_new": int(new_n) if new_n else 1,
                    "context": 0, "removed": 0, "added": 0,
                }
            elif current is not None:
                if line.startswith(" "):
                    current["context"] += 1
                elif line.startswith("-") and not line.startswith("---"):
                    current["removed"] += 1
                elif line.startswith("+") and not line.startswith("+++"):
                    current["added"] += 1
        if current is not None:
            old_ok = current["context"] + current["removed"] == current["declared_old"]
            new_ok = current["context"] + current["added"] == current["declared_new"]
            if not (old_ok and new_ok):
                arithmetic_ok = False

    git_apply_ok = False
    if is_diff:
        with open("_grid_test.patch", "w", encoding="utf-8", newline="") as f:
            f.write(diff_text)
        code, _, _ = run_git(["apply", "--check", "_grid_test.patch"])
        git_apply_ok = (code == 0)
        if os.path.exists("_grid_test.patch"):
            os.remove("_grid_test.patch")

    return {
        "looks_like_a_diff": is_diff,
        "hunks_well_formed": hunks_ok,
        "hunk_arithmetic_ok": arithmetic_ok,
        "git_apply_ok": git_apply_ok,
    }


def classify_error(v: dict, raw_output: str, finish_reason: str = "stop") -> str:
    if "NO_PATCH_POSSIBLE" in raw_output:
        return "No patch (model declined)"
    if not v["looks_like_a_diff"]:
        return "No patch generated / unparseable"
    if v["git_apply_ok"] and finish_reason == "length":
        # In full_file mode a response cut off by max_tokens is a SHORTENED
        # file, so the generated diff deletes the missing tail. It passes
        # `git apply --check` but is destructive, so it must never be
        # counted as a success.
        return "Truncated output (applies, but destructive)"
    if v["git_apply_ok"]:
        return "OK -- applies cleanly"
    if not v["hunks_well_formed"]:
        return "Malformed patch -- header format"
    if not v["hunk_arithmetic_ok"]:
        return "Malformed patch -- hunk arithmetic"
    return "Malformed patch -- other (content corruption / rejection)"


def run_one_cell(task: dict, model: str, provider: str, condition: str, output_mode: str, prompt_version: str,
                  temperature: float, max_tokens: int, adaptive_tokens: bool):
    if not checkout(task["before_commit"]):
        return None

    prompt = bpar.build_prompt(task, condition, prompt_version, output_mode)

    effective_max_tokens = max_tokens
    if adaptive_tokens and output_mode == "full_file":
        effective_max_tokens = bpar.estimate_max_tokens(task["before_commit"], task["changed_files"], model,
                                                          ceiling=max_tokens)
        if effective_max_tokens != max_tokens:
            print(f"    (adaptive max_tokens: {effective_max_tokens} instead of {max_tokens}, "
                  f"based on source size / model cap)")

    try:
        raw_output, finish_reason = bpar.call_model(provider, model, prompt["system"], prompt["user"], temperature, effective_max_tokens)
    except SystemExit:
        raise
    except Exception as e:
        print(f"    ERROR calling model: {e}", file=sys.stderr)
        return {
            "row": {
                "task_id": task["task_id"], "category": task["category"], "model": model,
                "provider": provider,
                "condition": condition, "output_mode": output_mode, "before_commit": task["before_commit"],
                "finish_reason": "API_ERROR", "looks_like_a_diff": False, "hunks_well_formed": False,
                "hunk_arithmetic_ok": False, "git_apply_ok": False,
                "error_category": "API error", "notes": str(e)[:200],
            },
            "record": {"task_id": task["task_id"], "model": model, "provider": provider, "condition": condition, "error": str(e)},
        }

    if output_mode == "full_file":
        parsed_files = bpar.parse_full_file_response(raw_output)
        diff_text = bpar.generate_mechanical_diff(task["before_commit"], parsed_files) if parsed_files else ""
    else:
        diff_text = raw_output

    v = validate_diff(diff_text)
    error_category = classify_error(v, raw_output, finish_reason)

    row = {
        "task_id": task["task_id"], "category": task["category"], "model": model,
        "provider": provider,
        "condition": condition, "output_mode": output_mode, "before_commit": task["before_commit"],
        "finish_reason": finish_reason,
        "looks_like_a_diff": v["looks_like_a_diff"], "hunks_well_formed": v["hunks_well_formed"],
        "hunk_arithmetic_ok": v["hunk_arithmetic_ok"], "git_apply_ok": v["git_apply_ok"],
        "error_category": error_category, "notes": "",
    }
    record = {
        "task_id": task["task_id"], "model": model, "provider": provider, "condition": condition, "output_mode": output_mode,
        "finish_reason": finish_reason, "raw_output": raw_output, "generated_diff": diff_text,
        "validation": v, "error_category": error_category,
    }
    return {"row": row, "record": record}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                         help="One or more model IDs for the chosen --provider (Groq API IDs, or Ollama tags)")
    parser.add_argument("--provider", default="groq", choices=["groq", "ollama"],
                         help="Applies to ALL models in this invocation. Run separate invocations to mix "
                              "Groq and Ollama models in one pilot_grid_results.csv (they'll coexist fine "
                              "since 'provider' is tracked per-row).")
    parser.add_argument("--tasks", nargs="+", default=None, help="Task IDs to run (default: all in pilot_tasks.json)")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--output-mode", default="full_file", choices=["diff", "full_file"])
    parser.add_argument("--prompt-version", default="v3", choices=list(bpar.PROMPT_VERSIONS.keys()))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096,
                         help="Ceiling on requested output tokens. With --adaptive-tokens (default on), "
                              "this is a maximum, not a fixed request -- smaller tasks request less.")
    parser.add_argument("--no-adaptive-tokens", action="store_true",
                         help="Disable adaptive max_tokens sizing; always request exactly --max-tokens "
                              "(the old behavior). Adaptive sizing is ON by default because it directly "
                              "reduces avoidable rate-limit rejections on models with small output-token "
                              "budgets (e.g. observed 950-1000 OTPM on qwen/qwen3.8-27b's free tier).")
    parser.add_argument("--sleep-seconds", type=float, default=2.0,
                         help="Pause between calls. Use 0 for --provider ollama (no rate limits, no reason to wait).")
    args = parser.parse_args()

    if not os.path.exists(TASKS_PATH):
        print(f"{TASKS_PATH} not found.", file=sys.stderr)
        sys.exit(1)
    with open(TASKS_PATH, encoding="utf-8") as f:
        all_tasks = {t["task_id"]: t for t in json.load(f)}

    task_ids = args.tasks or list(all_tasks.keys())
    for tid in task_ids:
        if tid not in all_tasks:
            print(f"Unknown task '{tid}'", file=sys.stderr)
            sys.exit(1)

    original_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, encoding="utf-8").stdout.strip()

    total_cells = len(task_ids) * len(args.models) * len(args.conditions)
    done = 0
    skipped = 0
    print(f"Grid: {len(task_ids)} tasks x {len(args.models)} models x {len(args.conditions)} conditions "
          f"= {total_cells} cells (provider={args.provider})\n")

    for task_id in task_ids:
        task = all_tasks[task_id]
        for model in args.models:
            for condition in args.conditions:
                if already_done(task_id, model, args.provider, condition):
                    print(f"[skip] {task_id} | {args.provider}:{model} | {condition} (already in {CSV_PATH})")
                    skipped += 1
                    continue

                print(f"[run]  {task_id} | {args.provider}:{model} | {condition} ...")
                result = run_one_cell(task, model, args.provider, condition, args.output_mode, args.prompt_version,
                                       args.temperature, args.max_tokens, not args.no_adaptive_tokens)
                if result is None:
                    continue
                append_csv_row(result["row"])
                append_jsonl_record(result["record"])
                done += 1
                print(f"       -> {result['row']['error_category']}")
                time.sleep(args.sleep_seconds)

    if original_head:
        checkout(original_head)  # restore original checkout state when done

    print(f"\nDone. {done} cells run, {skipped} skipped (already present), "
          f"{total_cells - done - skipped} remaining (errors above, if any).")
    print(f"Results: {CSV_PATH} (summary table), {JSONL_PATH} (full records with raw output)")


if __name__ == "__main__":
    main()