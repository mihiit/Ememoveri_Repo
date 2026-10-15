"""
build_pilot_tasks.py

Turns a hand-picked list of commit SHAs into a leakage-safe pilot task table:
for each SHA, resolves the parent commit (before_commit), the changed files,
the commit timestamp, and writes everything to pilot_tasks.json in the
format ememoveri's E5 pipeline expects.

This does NOT call any LLM and does NOT touch ememoveri's memory classes --
it is pure Git metadata extraction (Step 3 of the E5 pilot). Run it from
inside your cloned repo (e.g. the `fastapi` checkout).

Usage:
    python build_pilot_tasks.py --sha 48e983573 --category DEPRECATION --note "deprecate ORJSONResponse/UJSONResponse"
    python build_pilot_tasks.py --sha 1990ecb44 --category SUPERSESSION --note "remove slim package stub"
    ... (repeat --sha/--category/--note as separate calls, or use --from-file)

    python build_pilot_tasks.py --from-file candidates.csv

candidates.csv format (no header needed):
    sha,category,note
    48e983573,DEPRECATION,deprecate ORJSONResponse/UJSONResponse
    1990ecb44,SUPERSESSION,remove slim package stub

Output: pilot_tasks.json, appended to (not overwritten) across multiple runs
unless --reset is passed.
"""

import argparse
import csv
import json
import os
import subprocess
import sys

OUTPUT_PATH = "pilot_tasks.json"
VALID_CATEGORIES = {"VALID_KNOWLEDGE", "STALE_KNOWLEDGE", "CONTRADICTION", "REFINEMENT", "DEPRECATION", "SUPERSESSION"}


def run_git(args):
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        encoding="utf-8",
        errors="replace",   # never crash on an unexpected byte; substitute instead
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return (result.stdout or "").strip()


def resolve_full_sha(short_sha: str) -> str:
    return run_git(["rev-parse", short_sha])


def get_parent_sha(sha: str) -> str:
    parents = run_git(["log", "-1", "--format=%P", sha]).split()
    if not parents:
        raise ValueError(f"Commit {sha} has no parent (likely the repo's root commit) -- pick a different task.")
    if len(parents) > 1:
        print(f"  WARNING: {sha} is a merge commit with {len(parents)} parents; using the first: {parents[0]}", file=sys.stderr)
    return parents[0]


def get_commit_timestamp(sha: str) -> str:
    # ISO 8601, committer date (not author date) -- committer date is when
    # the change actually landed on the branch, which is the more defensible
    # "knowledge became available" timestamp for the leakage cutoff.
    return run_git(["show", "-s", "--format=%cI", sha])


def get_commit_message(sha: str) -> str:
    return run_git(["log", "-1", "--format=%B", sha])


def get_changed_files(sha: str) -> list:
    out = run_git(["show", "--stat", "--format=", sha])
    files = []
    for line in out.splitlines():
        line = line.strip()
        if "|" in line:
            files.append(line.split("|")[0].strip())
    return files


def build_task(sha: str, category: str, note: str, task_index: int, repo_name: str) -> dict:
    if category not in VALID_CATEGORIES:
        print(f"  WARNING: category '{category}' is not one of {sorted(VALID_CATEGORIES)} -- keeping it anyway, "
              f"but double check this matches your taxonomy.", file=sys.stderr)

    full_sha = resolve_full_sha(sha)
    parent_sha = get_parent_sha(full_sha)
    timestamp = get_commit_timestamp(full_sha)
    parent_timestamp = get_commit_timestamp(parent_sha)
    changed_files = get_changed_files(full_sha)
    message = get_commit_message(full_sha)

    task = {
        "task_id": f"{repo_name}_{task_index:03d}",
        "repo": repo_name,
        "before_commit": parent_sha,
        "before_commit_timestamp": parent_timestamp,
        "after_commit": full_sha,
        "after_commit_timestamp": timestamp,
        "category": category,
        "note": note,
        "changed_files": changed_files,
        "commit_message_first_line": message.splitlines()[0] if message else "",
    }
    return task


def load_existing(path: str) -> list:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sha", help="A single commit SHA (short or full)")
    parser.add_argument("--category", help="Task category (VALID_KNOWLEDGE / STALE_KNOWLEDGE / CONTRADICTION / REFINEMENT / DEPRECATION / SUPERSESSION)")
    parser.add_argument("--note", default="", help="One-line human note about what this task tests")
    parser.add_argument("--from-file", help="CSV file of sha,category,note rows (no header)")
    parser.add_argument("--repo-name", default=None, help="Repo name for task IDs (default: current directory name)")
    parser.add_argument("--reset", action="store_true", help="Overwrite pilot_tasks.json instead of appending")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    repo_name = args.repo_name or os.path.basename(os.getcwd())

    existing = [] if args.reset else load_existing(args.output)
    start_index = len(existing) + 1

    new_rows = []
    if args.from_file:
        with open(args.from_file, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                sha, category = row[0].strip(), row[1].strip()
                note = row[2].strip() if len(row) > 2 else ""
                new_rows.append((sha, category, note))
    elif args.sha:
        new_rows.append((args.sha, args.category or "UNSPECIFIED", args.note))
    else:
        parser.error("Provide either --sha (single) or --from-file (batch)")

    built = []
    for i, (sha, category, note) in enumerate(new_rows, start=start_index):
        print(f"Processing {sha} ({category}) ...")
        try:
            task = build_task(sha, category, note, i, repo_name)
            built.append(task)
            print(f"  -> task_id={task['task_id']}  before={task['before_commit'][:10]}  "
                  f"after={task['after_commit'][:10]}  files={len(task['changed_files'])}")
        except Exception as e:
            print(f"  SKIPPED ({sha}): {e}", file=sys.stderr)

    all_tasks = existing + built
    with open(args.output, "w") as f:
        json.dump(all_tasks, f, indent=2)

    print(f"\nWrote {len(built)} new task(s), {len(all_tasks)} total, to {args.output}")


if __name__ == "__main__":
    main()