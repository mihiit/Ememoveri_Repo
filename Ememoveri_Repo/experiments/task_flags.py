"""Flag 'routine' benchmark tasks (robustness check for the history-based baselines).

A task is routine if its cleaned subject (a) equals the cleaned subject of an earlier commit in the same repository, or
(b) looks like release / version-bump / changelog housekeeping.  History-based retrieval can exploit such recurring subjects
legitimately, but a reviewer will ask how much of its advantage comes from them.

  python experiments/task_flags.py --git-dir-root repos --collected collected_dev.pkl collected_heldout.pkl collected_multilang.pkl --out results/real_ext/task_flags.json"""
import argparse, json, os, pickle, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ememoveri.mining import mine_repo, clean_subject

ROUTINE = re.compile(r"(?i)\b(release[sd]?|bump(ed|s)?|version|changelog|changes?log|prepare|pre-?release|v?\d+\.\d+(\.\d+)?)\b|^\s*(chore|build)\b|dependabot")
ap = argparse.ArgumentParser()
ap.add_argument("--git-dir-root", required=True); ap.add_argument("--collected", nargs="+", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--cache", default=".cache")
a = ap.parse_args()
tasks = {}
for p in a.collected:
    for t in pickle.load(open(p, "rb"))["tasks"]:
        tasks[t.sha] = t.repo
flags = {}
for repo in sorted(set(tasks.values())):
    cs = mine_repo(os.path.join(a.git_dir_root, f"{repo}.git"), repo, cache_dir=a.cache)
    seen = set()
    for c in sorted(cs, key=lambda c: c.clock):          # same ordering rule as the replay (clock, then stream order)
        q = clean_subject(c.subject)
        if c.sha in tasks:
            flags[c.sha] = {"dup": q in seen, "pattern": bool(ROUTINE.search(c.subject)), "repo": repo}
        seen.add(q)
n = len(flags)
print(n, "tasks;", sum(f["dup"] for f in flags.values()), "duplicate subject;", sum(f["pattern"] for f in flags.values()), "pattern;",
      sum(f["dup"] or f["pattern"] for f in flags.values()), "routine")
json.dump(flags, open(a.out, "w"))
