"""
Git-history mining for the real-repository benchmark.

Design goals
------------
* **No keyword guessing for relations.** The relation between a commit and a
  file-level memory claim is read from git's own structural signal:
  ``A`` (added) -> CORROBORATES, ``M`` (modified) -> REFINES, ``D`` (deleted)
  and ``R`` (renamed away) -> SUPERSEDES, and commits that carry git's
  ``This reverts commit <sha>`` trailer -> CONTRADICTS. Nothing here depends
  on the English wording of a commit message (see the audit note on the v0.1
  keyword classifier).
* **Leakage-safe clock.** Commit timestamps in real repositories are not
  monotone (rebases, cherry-picks, clock skew). We therefore replay commits in
  topological order and use the running maximum of committer timestamps as
  the canonical clock (``Commit.clock``); by construction nothing replayed
  later can have an earlier clock value.
* Works on a *bare, blobless* clone (``git clone --bare --filter=blob:none``),
  because only commit metadata and tree diffs (``--name-status``) are needed.
  No working tree, no file contents.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

_SEP_REC = "\x1e"
_SEP_FLD = "\x1f"

_REVERT_RE = re.compile(r"This reverts commit ([0-9a-f]{7,40})", re.I)
_NOISE_SUBJECT = re.compile(
    r"^\W*(?:bump\b|update (?:release notes|contributors|sponsors|translations?)|pre-commit autoupdate|"
    r"\[pre-commit|merge (?:branch|pull request|remote)|(?:prepare )?release\b|version\b|\d+\.\d+)",
    re.I,
)
_BOT_AUTHORS = ("dependabot", "pre-commit-ci", "github-actions", "allcontributors", "renovate", "weblate")
_EMOJI_PREFIX = re.compile(r"^[^\w\(\[]+")
_PR_SUFFIX = re.compile(r"\s*\((?:#\d+|[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#\d+)\)\s*$")


@dataclass
class RepoConfig:
    """Per-repository layout knowledge (explicit, not inferred).

    Defaults describe a Python repository (``.py`` files under ``source_roots``, test files excluded).  Other languages set
    ``exts`` and, where tests live next to sources, ``test_suffixes`` / ``exclude`` (substring match on the path)."""

    name: str
    source_roots: Tuple[str, ...]          # directories whose files with a source extension are 'source'
    test_roots: Tuple[str, ...] = ("tests/", "test/")
    exts: Tuple[str, ...] = (".py",)
    test_suffixes: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ()

    def is_source(self, path: str) -> bool:
        return path.endswith(self.exts) and any(path.startswith(r) for r in self.source_roots) \
            and "/tests/" not in path and not os.path.basename(path).startswith("test_") \
            and not (self.test_suffixes and path.endswith(self.test_suffixes)) \
            and not any(x in path for x in self.exclude)

    def is_test(self, path: str) -> bool:
        if self.test_suffixes and path.endswith(self.test_suffixes):
            return True
        return any(path.startswith(r) for r in self.test_roots) and path.endswith(self.exts)


DEFAULT_REPOS: Dict[str, RepoConfig] = {
    "fastapi": RepoConfig("fastapi", ("fastapi/",), ("tests/",)),
    "yt-dlp": RepoConfig("yt-dlp", ("yt_dlp/",), ("test/",)),
    "browser-use": RepoConfig("browser-use", ("browser_use/",), ("tests/",)),
}


# Held-out repositories added in v0.4 (never used for design, debugging or tuning).
HELDOUT_REPOS: Dict[str, RepoConfig] = {
    "flask": RepoConfig("flask", ("src/flask/", "flask/"), ("tests/",)),
    "requests": RepoConfig("requests", ("src/requests/", "requests/"), ("tests/",)),
    "httpx": RepoConfig("httpx", ("httpx/",), ("tests/",)),
    "rich": RepoConfig("rich", ("rich/",), ("tests/",)),
    "black": RepoConfig("black", ("src/black/", "src/blib2to3/", "black/"), ("tests/",)),
    "poetry": RepoConfig("poetry", ("src/poetry/", "poetry/"), ("tests/",)),
    "pydantic": RepoConfig("pydantic", ("pydantic/",), ("tests/",)),
    "scrapy": RepoConfig("scrapy", ("scrapy/",), ("tests/",)),
    "pytest": RepoConfig("pytest", ("src/_pytest/", "src/pytest/", "_pytest/"), ("testing/",)),
    "aiohttp": RepoConfig("aiohttp", ("aiohttp/",), ("tests/",)),
    "celery": RepoConfig("celery", ("celery/",), ("t/",)),
    "sphinx": RepoConfig("sphinx", ("sphinx/",), ("tests/",)),
}
DEFAULT_REPOS.update(HELDOUT_REPOS)


# Multi-language held-out repositories (v0.5): JavaScript, Go, Rust, Java.  Never used for design, debugging or tuning.
MULTILANG_REPOS: Dict[str, RepoConfig] = {
    "express": RepoConfig("express", ("lib/",), ("test/",), exts=(".js",)),
    "axios": RepoConfig("axios", ("lib/",), ("test/", "tests/"), exts=(".js",)),
    "gin": RepoConfig("gin", ("",), (), exts=(".go",), test_suffixes=("_test.go",), exclude=("testdata/", "examples/", "docs/")),
    "cobra": RepoConfig("cobra", ("",), (), exts=(".go",), test_suffixes=("_test.go",), exclude=("testdata/", "examples/", "docs/")),
    "ripgrep": RepoConfig("ripgrep", ("crates/", "src/"), ("tests/",), exts=(".rs",), exclude=("/tests/", "/benches/", "/examples/")),
    "serde": RepoConfig("serde", ("serde/src/", "serde_core/src/", "serde_derive/src/", "serde_derive_internals/src/", "src/"), ("test_suite/",),
                        exts=(".rs",), exclude=("/tests/", "/benches/")),
    "jackson-databind": RepoConfig("jackson-databind", ("src/main/java/",), ("src/test/",), exts=(".java",)),
    "commons-lang": RepoConfig("commons-lang", ("src/main/java/",), ("src/test/",), exts=(".java",)),
}
DEFAULT_REPOS.update(MULTILANG_REPOS)


# Fresh confirmation set (v0.6): 16 repositories chosen and configured before any result on them was computed.
FRESH_REPOS: Dict[str, RepoConfig] = {
    "starlette": RepoConfig("starlette", ("starlette/",)),
    "uvicorn": RepoConfig("uvicorn", ("uvicorn/",)),
    "click": RepoConfig("click", ("src/click/", "click/")),
    "jinja": RepoConfig("jinja", ("src/jinja2/", "jinja2/")),
    "werkzeug": RepoConfig("werkzeug", ("src/werkzeug/", "werkzeug/")),
    "attrs": RepoConfig("attrs", ("src/attr/", "src/attrs/", "attr/")),
    "tornado": RepoConfig("tornado", ("tornado/",)),
    "marshmallow": RepoConfig("marshmallow", ("src/marshmallow/", "marshmallow/")),
    "tqdm": RepoConfig("tqdm", ("tqdm/",)),
    "rq": RepoConfig("rq", ("rq/",)),
    "koa": RepoConfig("koa", ("lib/",), ("test/", "__tests__/"), exts=(".js",)),
    "fastify": RepoConfig("fastify", ("lib/", "fastify.js"), ("test/",), exts=(".js",)),
    "echo": RepoConfig("echo", ("",), (), exts=(".go",), test_suffixes=("_test.go",), exclude=("_fixture/", "testdata/", "examples/", "docs/")),
    "fd": RepoConfig("fd", ("src/",), ("tests/",), exts=(".rs",), exclude=("/tests/",)),
    "bat": RepoConfig("bat", ("src/",), ("tests/",), exts=(".rs",), exclude=("/tests/",)),
    "mockito": RepoConfig("mockito", ("mockito-core/src/main/java/", "src/main/java/"), ("mockito-core/src/test/", "src/test/"), exts=(".java",)),
}
DEFAULT_REPOS.update(FRESH_REPOS)


@dataclass
class FileChange:
    status: str                  # A | M | D | R | C | T
    path: str                    # new path (for R: destination)
    old_path: Optional[str] = None


@dataclass
class Commit:
    sha: str
    clock: float                 # monotone (running-max) committer timestamp, seconds
    raw_ts: float                # committer timestamp as recorded
    author: str
    subject: str
    body: str
    changes: List[FileChange] = field(default_factory=list)
    reverts: Optional[str] = None  # sha (possibly abbreviated) of reverted commit
    repo: str = ""

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.clock, tz=timezone.utc)


def clean_subject(subject: str) -> str:
    """Strip emoji prefixes and trailing PR numbers; keeps the human wording."""
    s = _PR_SUFFIX.sub("", subject).strip()
    s = _EMOJI_PREFIX.sub("", s).strip()
    return s


def is_noise(c: Commit) -> bool:
    a = c.author.lower()
    if any(b in a for b in _BOT_AUTHORS):
        return True
    return bool(_NOISE_SUBJECT.match(c.subject))


def parse_log(raw: str, repo: str = "") -> List[Commit]:
    """Parse the output of ``git log --name-status`` produced by `mine_repo`."""
    commits: List[Commit] = []
    running = 0.0
    for rec in raw.split(_SEP_REC):
        if not rec.strip():
            continue
        parts = rec.split(_SEP_FLD)
        if len(parts) < 6:
            continue
        sha, ts, author, subject, body, tail = parts[0], parts[1], parts[2], parts[3], parts[4], _SEP_FLD.join(parts[5:])
        raw_ts = float(ts)
        running = max(running, raw_ts)
        changes: List[FileChange] = []
        for line in tail.splitlines():
            line = line.strip("\n")
            if not line.strip():
                continue
            cols = line.split("\t")
            st = cols[0]
            if st.startswith("R") and len(cols) >= 3:
                changes.append(FileChange("R", cols[2], cols[1]))
            elif st.startswith("C") and len(cols) >= 3:
                changes.append(FileChange("A", cols[2], None))  # copy == new file
            elif len(cols) >= 2:
                changes.append(FileChange(st[0], cols[1]))
        m = _REVERT_RE.search(body)
        commits.append(Commit(sha=sha, clock=running, raw_ts=raw_ts, author=author, subject=subject,
                              body=body, changes=changes, reverts=m.group(1) if m else None, repo=repo))
    return commits


def mine_repo(git_dir: str, repo: str, cache_dir: Optional[str] = None, refresh: bool = False) -> List[Commit]:
    """Mine all non-merge commits of a (bare) repository, oldest first."""
    cache = os.path.join(cache_dir, f"{repo}.commits.json") if cache_dir else None
    if cache and os.path.exists(cache) and not refresh:
        with open(cache) as f:
            blob = json.load(f)
        return [Commit(
            sha=c["sha"], clock=c["clock"], raw_ts=c["raw_ts"], author=c["author"], subject=c["subject"],
            body=c["body"], reverts=c["reverts"], repo=repo,
            changes=[FileChange(*fc) for fc in c["changes"]]) for c in blob["commits"]]

    fmt = f"{_SEP_REC}%H{_SEP_FLD}%ct{_SEP_FLD}%an{_SEP_FLD}%s{_SEP_FLD}%b{_SEP_FLD}"
    cmd = ["git", "--git-dir", git_dir, "log", "--reverse", "--topo-order", "--no-merges",
           "-M", "--name-status", f"--format={fmt}"]
    raw = subprocess.run(cmd, capture_output=True, text=True, check=True, errors="replace").stdout
    commits = parse_log(raw, repo)
    head = subprocess.run(["git", "--git-dir", git_dir, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache, "w") as f:
            json.dump({"repo": repo, "head": head, "n": len(commits), "commits": [
                {"sha": c.sha, "clock": c.clock, "raw_ts": c.raw_ts, "author": c.author, "subject": c.subject,
                 "body": c.body, "reverts": c.reverts,
                 "changes": [[fc.status, fc.path, fc.old_path] for fc in c.changes]} for c in commits]}, f)
    return commits


def timestamp_anomalies(commits: Sequence[Commit]) -> int:
    """Number of commits whose recorded timestamp precedes the running clock."""
    return sum(1 for c in commits if c.raw_ts < c.clock)
