"""Collect per-task features for the v0.3/v0.4 analyses (one chronological replay).

  python experiments/collect_ext.py --git-dir-root repos --set dev     --out collected.pkl           # fastapi, yt-dlp, browser-use
  python experiments/collect_ext.py --git-dir-root repos --set heldout --out collected_heldout.pkl --slim   # 12 held-out repositories
  python experiments/collect_ext.py --git-dir-root repos --set multilang --out collected_multilang.pkl --slim   # 8 JS/Go/Rust/Java repositories

--slim keeps only the TF-IDF similarity and the similar-history / co-change channels per task (what the held-out analyses need).
Needs: pip install spacy && python -m spacy download en_core_web_md   (static word vectors for the 'dense' backend)."""
import argparse, os, pickle, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ememoveri.mining import mine_repo, DEFAULT_REPOS, HELDOUT_REPOS, MULTILANG_REPOS, FRESH_REPOS
from ememoveri.realbench_ext import collect, WordVectors

ap = argparse.ArgumentParser()
ap.add_argument("--git-dir-root", required=True)
ap.add_argument("--set", choices=["dev", "heldout", "multilang", "fresh"], default="dev")
ap.add_argument("--out", default="collected.pkl")
ap.add_argument("--cache", default=".cache")
ap.add_argument("--slim", action="store_true")
ap.add_argument("--no-carry", action="store_true", help="ablation: do not carry history across renames")
ap.add_argument("--no-events", action="store_true", help="skip lifecycle events (lower memory)")
a = ap.parse_args()
repos = {"heldout": list(HELDOUT_REPOS), "multilang": list(MULTILANG_REPOS), "fresh": list(FRESH_REPOS)}.get(a.set, ["fastapi", "yt-dlp", "browser-use"])
cs = {r: mine_repo(os.path.join(a.git_dir_root, f"{r}.git"), r, cache_dir=a.cache) for r in repos}
t = time.time()
out = collect(cs, repos, None if a.slim else WordVectors(), slim=a.slim, track_events=not a.no_events, carry=not a.no_carry)
pickle.dump(out, open(a.out, "wb"), protocol=4)
print("tasks", len(out["tasks"]), "events", out["counts"], f"{time.time() - t:.0f}s")
