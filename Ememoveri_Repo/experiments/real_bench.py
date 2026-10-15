"""Real-repository benchmark (E5-real). Regenerates every number in the paper.

Usage:
    python experiments/real_bench.py --git-dir-root /path/to/bare/clones --out results/real
Bare blobless clones:  git clone --bare --filter=blob:none https://github.com/<org>/<repo>.git <repo>.git
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ememoveri.mining import mine_repo, timestamp_anomalies
from ememoveri.realbench import BenchConfig, run_benchmark

ap = argparse.ArgumentParser()
ap.add_argument("--git-dir-root", required=True)
ap.add_argument("--repos", default="fastapi,yt-dlp,browser-use")
ap.add_argument("--max-tasks-per-repo", type=int, default=400)
ap.add_argument("--warmup-frac", type=float, default=0.35)
ap.add_argument("--out", default="results/real")
ap.add_argument("--cache", default=".cache")
a = ap.parse_args()

repos = tuple(a.repos.split(","))
commits = {r: mine_repo(os.path.join(a.git_dir_root, f"{r}.git"), r, cache_dir=a.cache) for r in repos}
meta = {r: {"commits": len(c), "timestamp_anomalies": timestamp_anomalies(c),
            "first": c[0].dt.isoformat(), "last": c[-1].dt.isoformat()} for r, c in commits.items()}
res = run_benchmark(commits, BenchConfig(repos=repos, max_tasks_per_repo=a.max_tasks_per_repo,
                                         warmup_frac=a.warmup_frac))
res["repo_meta"] = meta
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "raw.json"), "w") as f:
    json.dump(res, f, default=float)
print("wrote", os.path.join(a.out, "raw.json"), "tasks:", len(res["rows"]), "runtime", round(res["runtime_s"]), "s")
