"""Replay with issue-text tasks (needs issue_tasks.json from issue_tasks.py).
  python experiments/collect_issue.py --git-dir-root repos --tasks issue_tasks.json --out issue_collected.pkl
Memory for each task contains only commits up to the issue's creation time; gold = files of the later fix that already existed."""
import argparse, json, os, pickle, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ememoveri.mining import mine_repo
from ememoveri.realbench_ext import collect_external, WordVectors

ap = argparse.ArgumentParser()
ap.add_argument("--git-dir-root", required=True); ap.add_argument("--tasks", required=True)
ap.add_argument("--out", default="issue_collected.pkl"); ap.add_argument("--cache", default=".cache")
ap.add_argument("--dense", action="store_true", help="also compute static word-vector similarity (needs spaCy en_core_web_md)")
a = ap.parse_args()
tasks = json.load(open(a.tasks))
repos = sorted({t["repo"] for t in tasks})
cs = {r: mine_repo(os.path.join(a.git_dir_root, f"{r}.git"), r, cache_dir=a.cache) for r in repos}
out = collect_external(cs, repos, tasks, wv=WordVectors() if a.dense else None)
pickle.dump(out, open(a.out, "wb"), protocol=4)
print(len(tasks), "issue tasks ->", len(out["tasks"]), "scored;", out["dropped_gold"], "gold files dropped (did not exist at the cutoff)")
