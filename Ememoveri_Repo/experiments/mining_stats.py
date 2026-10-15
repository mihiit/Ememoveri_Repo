"""Counts of structural events in the mined histories (how much each state-machine path is exercised)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ememoveri.mining import DEFAULT_REPOS, mine_repo
root, out = sys.argv[1], sys.argv[2]
res = {}
for r, cfg in DEFAULT_REPOS.items():
    cs = mine_repo(os.path.join(root, f"{r}.git"), r, cache_dir=".cache")
    ch = [fc for c in cs for fc in c.changes]
    res[r] = {"commits": len(cs),
              "source_adds": sum(1 for f in ch if f.status == "A" and cfg.is_source(f.path)),
              "source_mods": sum(1 for f in ch if f.status == "M" and cfg.is_source(f.path)),
              "source_deletes": sum(1 for f in ch if f.status == "D" and cfg.is_source(f.path)),
              "source_renames": sum(1 for f in ch if f.status == "R" and cfg.is_source(f.path)),
              "revert_commits": sum(1 for c in cs if c.reverts),
              "revert_commits_touching_source": sum(1 for c in cs if c.reverts and any(cfg.is_source(f.path) for f in c.changes))}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
