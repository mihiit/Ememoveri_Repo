"""
build_pilot_memory.py

Step 4 of the E5 pilot: for every task in pilot_tasks.json, mine ONLY
commits that touched the task's declared changed_files and that landed at
or before that task's before_commit timestamp, convert them into Evidence,
group them into per-file MemoryClaims, run them through EmemoVeri's real
state engine + trust model, and verify with TemporalLeakageGuard that
nothing after the cutoff leaked in.

Requires: pip install pydriller
Requires: ememoveri installed (you already did `pip install -e ".[all]"`)

Run from inside your repo checkout (e.g. the `fastapi` folder), next to
pilot_tasks.json:

    python build_pilot_memory.py

Output: pilot_memory/<task_id>.json -- one file per task, containing every
constructed MemoryClaim (state, trust, evidence list) plus a leakage-check
verdict. Read a few of these by hand before moving to Step 5 (the actual
model calls) -- this is your "does the mechanism see sensible engineering
facts" checkpoint.
"""

import json
import os
import sys
from datetime import datetime, timedelta

try:
    from pydriller import Repository
except ImportError:
    print("PyDriller is required. Install it with:\n    pip install pydriller", file=sys.stderr)
    sys.exit(1)

try:
    from ememoveri.evidence_processor import EvidenceProcessor
    from ememoveri.schema import MemoryClaim
    from ememoveri.state_engine import StateEvolutionEngine
    from ememoveri.trust import TrustModel
    from ememoveri.store import TemporalLeakageGuard, TemporalLeakageError
except ImportError as e:
    print(f"Could not import ememoveri ({e}).\n"
          f"Make sure you ran `pip install -e \".[all]\"` inside the ememoveri project folder.",
          file=sys.stderr)
    sys.exit(1)

TASKS_PATH = "pilot_tasks.json"
OUTPUT_DIR = "pilot_memory"
REPO_PATH = "."  # run this script from inside the git repo checkout


def subsystem_scope(repo_name: str, file_path: str) -> str:
    """First path component as a coarse 'subsystem' -- e.g.
    'fastapi/routing.py' -> scope 'fastapi_repo/fastapi', so claims about
    files in the same top-level module share a scope for retrieval."""
    top = file_path.split("/")[0] if "/" in file_path else file_path
    return f"{repo_name}/{top}"


def mine_evidence_for_file(repo_path: str, file_path: str, cutoff_sha: str, cutoff_timestamp: datetime,
                            proc: EvidenceProcessor, lookback_days: int, max_commits: int):
    """Mine commits that touched `file_path`, up to and including
    `cutoff_sha`, restricted to a recent lookback window and a hard cap on
    commit count -- pooling a file's ENTIRE history (which can span a
    decade-plus and hundreds of commits for old, actively-maintained files)
    into one claim would blur ancient and recent engineering facts together.
    A claim should represent "what's currently believed," not an unweighted
    average of everything that ever happened to a file."""
    since = cutoff_timestamp - timedelta(days=lookback_days) if lookback_days else None
    evidence_items = []
    try:
        for commit in Repository(repo_path, filepath=file_path, to_commit=cutoff_sha).traverse_commits():
            if since is not None and commit.committer_date < since:
                continue
            evidence_items.append(proc.from_commit(
                message=commit.msg,
                timestamp=commit.committer_date,
                commit_sha=commit.hash,
            ))
    except Exception as e:
        print(f"    WARNING: mining failed for {file_path}: {e}", file=sys.stderr)

    if max_commits and len(evidence_items) > max_commits:
        # Keep the MOST RECENT max_commits -- older evidence within the
        # lookback window is still less relevant to "current" knowledge.
        evidence_items = sorted(evidence_items, key=lambda e: e.timestamp)[-max_commits:]

    return evidence_items


def build_claim_for_file(repo_name: str, file_path: str, evidence_items: list) -> MemoryClaim:
    claim = MemoryClaim(
        statement=f"Historical engineering evidence trail for {file_path}",
        scope=subsystem_scope(repo_name, file_path),
        tags=(file_path,),
    )
    if not evidence_items:
        return claim  # stays PROVISIONAL with no evidence -- valid, just uninformative

    engine = StateEvolutionEngine()
    engine.apply_evidence_batch(claim, evidence_items, aggregate=True)

    trust_model = TrustModel()
    claim.trust = trust_model.aggregated_update(claim.evidence)
    return claim


def process_task(task: dict, proc: EvidenceProcessor, lookback_days: int, max_commits: int) -> dict:
    repo_name = task["repo"]
    cutoff_sha = task["before_commit"]
    cutoff_timestamp = datetime.fromisoformat(task["before_commit_timestamp"])
    guard = TemporalLeakageGuard(knowledge_cutoff=cutoff_timestamp)

    claims = []
    leakage_failures = []

    for file_path in task["changed_files"]:
        print(f"    mining {file_path} up to {cutoff_sha[:10]} "
              f"(lookback={lookback_days}d, max_commits={max_commits}) ...")
        evidence_items = mine_evidence_for_file(REPO_PATH, file_path, cutoff_sha, cutoff_timestamp,
                                                 proc, lookback_days, max_commits)
        claim = build_claim_for_file(repo_name, file_path, evidence_items)
        try:
            guard.assert_no_leakage(claim)
        except TemporalLeakageError as e:
            leakage_failures.append({"file": file_path, "error": str(e)})
            continue  # do NOT include a leaking claim in the output
        claims.append(claim)

    return {
        "task_id": task["task_id"],
        "before_commit": cutoff_sha,
        "knowledge_cutoff": task["before_commit_timestamp"],
        "n_claims": len(claims),
        "n_leakage_failures": len(leakage_failures),
        "leakage_failures": leakage_failures,
        "claims": [
            {
                **claim.to_dict(),
                "evidence": [
                    {
                        "type": e.evidence_type.value,
                        "relation": e.relation.value,
                        "content": e.content,
                        "timestamp": e.timestamp.isoformat(),
                        "source_id": e.source_id,
                        "strength": e.strength,
                    }
                    for e in claim.evidence
                ],
            }
            for claim in claims
        ],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=730,
                         help="Only mine commits within this many days before each task's cutoff "
                              "(default: 730 = ~2 years). Pass 0 to disable and mine full history.")
    parser.add_argument("--max-commits-per-file", type=int, default=20,
                         help="Hard cap on commits pooled into one claim per file, keeping the "
                              "most recent ones (default: 20). Pass 0 to disable.")
    args = parser.parse_args()

    if not os.path.exists(TASKS_PATH):
        print(f"{TASKS_PATH} not found. Run build_pilot_tasks.py first.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(os.path.join(REPO_PATH, ".git")):
        print("This does not look like a git repository root (no .git found). "
              "Run this script from inside your repo checkout.", file=sys.stderr)
        sys.exit(1)

    with open(TASKS_PATH) as f:
        tasks = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    proc = EvidenceProcessor()

    total_leakage_failures = 0
    for task in tasks:
        print(f"Processing {task['task_id']} ({task['category']}) ...")
        result = process_task(task, proc, args.lookback_days, args.max_commits_per_file)
        out_path = os.path.join(OUTPUT_DIR, f"{task['task_id']}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        total_leakage_failures += result["n_leakage_failures"]
        flag = "  <-- LEAKAGE DETECTED, EXCLUDED FROM OUTPUT" if result["n_leakage_failures"] else ""
        print(f"  -> {result['n_claims']} claims built, "
              f"{result['n_leakage_failures']} leakage failures{flag}")
        print(f"  -> saved to {out_path}")

    print(f"\nDone. {len(tasks)} tasks processed.")
    if total_leakage_failures:
        print(f"WARNING: {total_leakage_failures} leakage failure(s) detected across all tasks. "
              f"Check the 'leakage_failures' field in the affected task JSON files before proceeding.")
    else:
        print("No leakage failures detected across any task.")


if __name__ == "__main__":
    main()