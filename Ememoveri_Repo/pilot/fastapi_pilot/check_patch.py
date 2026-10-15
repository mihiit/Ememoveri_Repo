"""
check_patch.py

Extracts the raw_output field from a saved pilot_agent_runs/*.json file and
writes it to a .patch file using explicit UTF-8 encoding (avoiding the
Windows cp1252-default crash that plain `print(...) > file` hits when a
model's output contains emoji or other non-ASCII characters), then runs
`git apply --check` against it and reports the verdict.

Usage:
    python check_patch.py pilot_agent_runs/fastapi_002__openai_gpt-oss-120b__EMEMOVERI.json

Run this from inside the repo, checked out at the task's before_commit
(same as you were already doing).
"""

import argparse
import io
import json
import re
import subprocess
import sys


def get_current_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, encoding="utf-8", errors="replace")
    return result.stdout.strip() if result.returncode == 0 else None


def verify_correct_checkout(task_id: str) -> None:
    """This exact class of mistake -- checking a patch against whatever
    commit happens to be checked out, rather than the task's actual
    before_commit -- has already produced a false 'patch does not apply'
    failure once in this pilot. Fail loudly and early instead of letting
    git's confusing error message stand in for the real cause."""
    import os
    if not os.path.exists("pilot_tasks.json"):
        return  # can't verify without the task list; proceed silently
    with open("pilot_tasks.json", encoding="utf-8") as f:
        tasks = {t["task_id"]: t for t in json.load(f)}
    if task_id not in tasks:
        return
    expected = tasks[task_id]["before_commit"]
    current = get_current_head()
    if current is None:
        return
    if not expected.startswith(current) and not current.startswith(expected):
        print(f"\n*** CHECKOUT MISMATCH ***", file=sys.stderr)
        print(f"Task '{task_id}' expects before_commit={expected[:12]}, "
              f"but the repo is currently checked out at {current[:12]}.", file=sys.stderr)
        print(f"Any git apply result below is meaningless until you fix this. Run:\n"
              f"    git checkout {expected}\nthen re-run this check.", file=sys.stderr)
        sys.exit(1)


def parse_hunks(patch_text: str):
    """Split a unified diff into hunks and return, for each, the declared
    header numbers alongside the ACTUAL counted lines -- catching the
    common LLM failure mode where a hunk header has the right shape
    (`@@ -N,M +N,M @@`) but wrong arithmetic (miscounted M)."""
    lines = patch_text.splitlines()
    header_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    hunks = []
    current = None
    for line in lines:
        m = header_re.match(line)
        if m:
            if current is not None:
                hunks.append(current)
            old_start, old_count, new_start, new_count = m.groups()
            current = {
                "header": line,
                "declared_old_count": int(old_count) if old_count is not None else 1,
                "declared_new_count": int(new_count) if new_count is not None else 1,
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
        hunks.append(current)
    return hunks


def check_hunk_arithmetic(patch_text: str):
    """Returns a list of mismatch descriptions (empty if every hunk's
    declared counts match its actual content)."""
    problems = []
    for i, h in enumerate(parse_hunks(patch_text), start=1):
        actual_old = h["context"] + h["removed"]
        actual_new = h["context"] + h["added"]
        if actual_old != h["declared_old_count"] or actual_new != h["declared_new_count"]:
            problems.append(
                f"Hunk {i} ({h['header']!r}): declared -,{h['declared_old_count']} "
                f"+,{h['declared_new_count']} but actual content has "
                f"{actual_old} old-side lines ({h['context']} context + {h['removed']} removed) "
                f"and {actual_new} new-side lines ({h['context']} context + {h['added']} added)."
            )
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_json", help="Path to a saved pilot_agent_runs/*.json file")
    parser.add_argument("--out", default="test.patch", help="Where to write the extracted patch")
    args = parser.parse_args()

    with open(args.run_json, encoding="utf-8") as f:
        data = json.load(f)

    verify_correct_checkout(data["task_id"])

    # full_file mode stores the mechanically-generated diff separately from
    # the model's raw (non-diff) response; diff mode's raw_output IS the diff.
    if data.get("output_mode") == "full_file":
        raw_output = data.get("generated_diff", "")
        if not raw_output:
            print("No generated_diff found (model likely produced no parseable ### FILE blocks). "
                  "Nothing to check.")
            sys.exit(1)
    else:
        raw_output = data["raw_output"]

    with io.open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(raw_output)

    print(f"Wrote {len(raw_output)} chars to {args.out}")
    print(f"Task: {data['task_id']} | Model: {data['model']} | Condition: {data['condition']}")
    if data.get("finish_reason") and data["finish_reason"] != "stop":
        print(f"NOTE: finish_reason='{data['finish_reason']}' -- response may be truncated; "
              f"treat any failure below as ambiguous until this is ruled out.")

    arithmetic_problems = check_hunk_arithmetic(raw_output)
    if arithmetic_problems:
        print("\n--- HUNK ARITHMETIC CHECK: MISMATCHES FOUND ---")
        for p in arithmetic_problems:
            print(f"  {p}")
    else:
        print("\n--- HUNK ARITHMETIC CHECK: all declared counts match actual content ---")

    result = subprocess.run(["git", "apply", "--check", args.out], capture_output=True, encoding="utf-8", errors="replace")

    if result.returncode == 0:
        print("\nVERDICT: patch is structurally valid and applies cleanly (git apply --check passed).")
    else:
        print("\nVERDICT: patch REJECTED by git apply --check.")
        print("--- git's error output ---")
        print(result.stderr.strip())
        print("---")
        if arithmetic_problems:
            print("Log this as 'Malformed patch -- hunk arithmetic error' (header shape correct, "
                  "line-count numbers wrong) in your error taxonomy.")
        else:
            print("Hunk arithmetic checks out, so this is a different failure (e.g. content-level "
                  "corruption around lines that themselves start with '-'/'+', or a genuine "
                  "structural issue) -- log it as 'Malformed patch -- other' with git's message above.")
        print("Do NOT hand-fix the patch and re-test; the failure itself is the data point.")


if __name__ == "__main__":
    main()