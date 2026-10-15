"""Retrieval-conditioned patch generation with test execution (v0.4).  Needs ANTHROPIC_API_KEY for `solve`.

  validate : check that each candidate task is a real fail-to-pass task (tests fail at the parent commit with the
             test patch applied, pass once the gold source patch is applied).
  solve    : for each validated task and each memory condition, give the model the issue text, the source file tree
             and the top-5 files retrieved by that condition (contents read at the base commit), ask for SEARCH/REPLACE
             edits, apply them, run the task's tests.
  report   : resolved rate per condition, paired exact McNemar, bootstrap CI, tokens, dead-path references.

This isolates what the memory contributes (which files the model sees); it is not a multi-turn agent."""
import argparse, json, os, re, subprocess, sys, tempfile, time, urllib.request, shutil
from collections import defaultdict

BLOCK = re.compile(r"<<<<<<< SEARCH ([^\n]+)\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE", re.S)
SYSTEM = ("You are fixing a bug or implementing a change in a Python repository. Reply only with edits in this format, one "
          "block per edit, the SEARCH text copied exactly from the file shown:\n<<<<<<< SEARCH path/to/file.py\n<old lines>\n"
          "=======\n<new lines>\n>>>>>>> REPLACE")


def parse_blocks(text):
    return [(m.group(1).strip(), m.group(2), m.group(3)) for m in BLOCK.finditer(text)]


def apply_blocks(root, blocks):
    """Returns (n_applied, errors).  Each SEARCH must occur in the file; the first occurrence is replaced."""
    applied, errors = 0, []
    for path, old, new in blocks:
        f = os.path.join(root, path)
        if not os.path.isfile(f):
            errors.append(f"missing file {path}"); continue
        src = open(f, encoding="utf-8", errors="replace").read()
        if old not in src:
            errors.append(f"search text not found in {path}"); continue
        open(f, "w", encoding="utf-8").write(src.replace(old, new, 1)); applied += 1
    return applied, errors


def sh(cmd, cwd=None, env=None, timeout=900):
    return subprocess.run(cmd, cwd=cwd, env=env, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)


def git(gd, *a):
    return subprocess.run(["git", f"--git-dir={gd}", *a], capture_output=True, text=True)


def make_worktree(gd, sha, dest):
    r = git(gd, "worktree", "add", "--detach", dest, sha)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return dest


def drop_worktree(gd, dest):
    git(gd, "worktree", "remove", "--force", dest)


def run_tests(wd, venv_python, test_files, pythonpath):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(pythonpath))
    r = sh([venv_python, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *test_files], cwd=wd, env=env, timeout=1200)
    return r.returncode == 0, (r.stdout + r.stderr)[-1500:]


def candidate(task, gd, cfg):
    """Split the fix commit into gold source patch and test patch (parent -> fix)."""
    parent = git(gd, "rev-parse", task["sha"] + "^1").stdout.strip()
    files = git(gd, "diff", "--name-only", parent, task["sha"]).stdout.split()
    tests = [f for f in files if cfg.is_test(f)]
    src = [f for f in files if cfg.is_source(f)]
    if not tests or not src:
        return None
    return {"base": parent, "test_files": tests, "src_files": src,
            "gold_patch": git(gd, "diff", parent, task["sha"], "--", *src).stdout,
            "test_patch": git(gd, "diff", parent, task["sha"], "--", *tests).stdout}


def validate_task(task, c, gd, venv_python, pythonpath_rel):
    with tempfile.TemporaryDirectory() as td:
        wd = make_worktree(gd, c["base"], os.path.join(td, "wd"))
        try:
            pp = [os.path.join(wd, p) for p in pythonpath_rel]
            def apply(patch):
                p = os.path.join(td, "x.patch"); open(p, "w").write(patch)
                return sh(["git", "apply", "--whitespace=nowarn", p], cwd=wd).returncode == 0
            if not apply(c["test_patch"]):
                return False, "test patch does not apply"
            before, _ = run_tests(wd, venv_python, c["test_files"], pp)
            if before:
                return False, "tests already pass at parent"
            if not apply(c["gold_patch"]):
                return False, "gold patch does not apply"
            after, log = run_tests(wd, venv_python, c["test_files"], pp)
            return (after, "ok" if after else "tests fail after gold patch: " + log[-300:])
        finally:
            drop_worktree(gd, wd)


def llm(prompt, model, key, max_tokens=4000):
    body = json.dumps({"model": model, "max_tokens": max_tokens, "system": SYSTEM, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body, {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    t = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    return "".join(b.get("text", "") for b in d["content"]), d["usage"], time.time() - t


def build_prompt(task, tree, files):
    parts = [f"## Issue\n{task['queries']['full']}\n", "## Source files in the repository\n" + "\n".join(tree[:400])]
    for path, text in files:
        parts.append(f"## File: {path}\n```python\n{text[:24000]}\n```")
    return "\n\n".join(parts)


def solve(args):
    key = os.environ["ANTHROPIC_API_KEY"]
    tasks = json.load(open(args.validated)); retr = json.load(open(args.retrievals))
    out = []
    for t in tasks:
        gd = os.path.join(args.git_dir_root, f"{t['repo']}.git")
        cfg_py = args.python[t["repo"]] if isinstance(args.python, dict) else args.python
        for cond in args.conditions:
            paths = [] if cond == "no_memory" else (t["src_files"] if cond == "oracle" else retr[t["id"]][cond][:5])
            with tempfile.TemporaryDirectory() as td:
                wd = make_worktree(gd, t["base"], os.path.join(td, "wd"))
                try:
                    tree = sorted(git(gd, "ls-tree", "-r", "--name-only", t["base"]).stdout.split())
                    tree = [p for p in tree if p.endswith(".py") and not p.startswith(("tests/", "test/", "testing/", "docs/"))]
                    files = [(p, open(os.path.join(wd, p), errors="replace").read()) for p in paths if os.path.isfile(os.path.join(wd, p))]
                    dead = [p for p in paths if not os.path.isfile(os.path.join(wd, p))]
                    reply, usage, secs = llm(build_prompt(t, tree, files), args.model, key)
                    blocks = parse_blocks(reply)
                    pt = os.path.join(td, "t.patch"); open(pt, "w").write(t["test_patch"]); sh(["git", "apply", "--whitespace=nowarn", pt], cwd=wd)
                    n, errs = apply_blocks(wd, blocks)
                    ok = False
                    if n and not errs:
                        ok, _ = run_tests(wd, cfg_py, t["test_files"], [os.path.join(wd, p) for p in args.pythonpath[t["repo"]]])
                    out.append({"id": t["id"], "cond": cond, "resolved": bool(ok), "applied": n, "errors": errs, "n_blocks": len(blocks),
                                "in_tok": usage["input_tokens"], "out_tok": usage["output_tokens"], "secs": secs, "dead_refs": len(dead),
                                "hit5": bool(set(paths) & set(t["src_files"])), "edited": sorted({b[0] for b in blocks})})
                    print(t["id"], cond, "resolved" if ok else "no", flush=True)
                finally:
                    drop_worktree(gd, wd)
            json.dump(out, open(args.out, "w"), indent=1)


def validate_cmd(args):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ememoveri.mining import DEFAULT_REPOS
    vc = json.load(open(args.venv_config)); tasks = json.load(open(args.tasks)); out = []
    for t in tasks:
        if t["repo"] not in vc:
            continue
        gd = os.path.join(args.git_dir_root, f"{t['repo']}.git")
        c = candidate(t, gd, DEFAULT_REPOS[t["repo"]])
        if not c:
            continue
        ok, msg = validate_task(t, c, gd, vc[t["repo"]]["python"], vc[t["repo"]]["pythonpath"])
        print(t["id"], "valid" if ok else "invalid: " + msg[:100], flush=True)
        if ok:
            out.append({**t, **c})
            json.dump(out, open(args.out, "w"), indent=1)
    print(len(out), "validated tasks of", len(tasks))


def report(args):
    import numpy as np
    from scipy.stats import binomtest
    R = json.load(open(args.results)); by = defaultdict(dict)
    for r in R:
        by[r["cond"]][r["id"]] = r
    ids = sorted(set.intersection(*[set(v) for v in by.values()]))
    rng = np.random.default_rng(0)
    print(f"{len(ids)} tasks with all conditions")
    for c, v in by.items():
        a = np.array([v[i]["resolved"] for i in ids], float)
        print(f"{c:14s} resolved {a.mean():.3f}  applied {np.mean([v[i]['applied']>0 and not v[i]['errors'] for i in ids]):.3f}  "
              f"tokens in/out {np.mean([v[i]['in_tok'] for i in ids]):.0f}/{np.mean([v[i]['out_tok'] for i in ids]):.0f}  dead refs {np.mean([v[i]['dead_refs'] for i in ids]):.2f}  hit@5 {np.mean([v[i]['hit5'] for i in ids]):.3f}")
    base = args.baseline
    for c, v in by.items():
        if c == base: continue
        d = np.array([v[i]["resolved"] for i in ids], float) - np.array([by[base][i]["resolved"] for i in ids], float)
        b, cc = int((d > 0).sum()), int((d < 0).sum())
        p = binomtest(b, b + cc, 0.5).pvalue if b + cc else 1.0
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        print(f"{c} - {base}: {d.mean():+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]  only-A/only-B {b}/{cc}  McNemar p={p:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve"); s.add_argument("--validated", required=True); s.add_argument("--retrievals", required=True)
    s.add_argument("--git-dir-root", required=True); s.add_argument("--model", default="claude-sonnet-5-5"); s.add_argument("--out", default="agent_results.json")
    s.add_argument("--conditions", nargs="+", default=["no_memory", "snapshot", "sim_state_rec_pop", "ememoveri_tuned", "oracle"])
    s.add_argument("--venv-config", required=True, help="JSON {repo: {python: path-to-venv-python, pythonpath: [rel dirs]}}")
    v = sub.add_parser("validate"); v.add_argument("--tasks", required=True); v.add_argument("--git-dir-root", required=True)
    v.add_argument("--venv-config", required=True); v.add_argument("--out", default="validated_tasks.json")
    r = sub.add_parser("report"); r.add_argument("--results", required=True); r.add_argument("--baseline", default="no_memory")
    a = ap.parse_args()
    if a.cmd == "validate":
        validate_cmd(a)
    elif a.cmd == "solve":
        vc = json.load(open(a.venv_config)); a.python = {k: v["python"] for k, v in vc.items()}; a.pythonpath = {k: v["pythonpath"] for k, v in vc.items()}
        solve(a)
    else:
        report(a)
