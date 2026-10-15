"""Build issue/PR-based tasks (v0.4).  Needs a GitHub token with read access to public repositories:

    export GITHUB_TOKEN=...            # fine-grained or classic token, no scopes needed
    python experiments/issue_tasks.py --git-dir-root repos --repos flask requests --per-repo 40 --out issue_tasks.json

For every squash-merged PR (commit subject ending in "(#N)") that closes an issue ("Fixes #X"), the task is
    query  = issue title + body            (what a developer would hand an agent)
    cutoff = issue creation time           (memory sees only commits up to then)
    gold   = source files modified by the fix commit that already existed at the cutoff
API responses are cached under --cache so the script is resumable across rate-limit windows."""
import argparse, hashlib, json, os, re, sys, time, urllib.request, urllib.error
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ememoveri.mining import mine_repo, DEFAULT_REPOS

SLUG = {"fastapi": "fastapi/fastapi", "yt-dlp": "yt-dlp/yt-dlp", "browser-use": "browser-use/browser-use",
        "flask": "pallets/flask", "requests": "psf/requests", "httpx": "encode/httpx", "rich": "Textualize/rich",
        "black": "psf/black", "poetry": "python-poetry/poetry", "pydantic": "pydantic/pydantic", "scrapy": "scrapy/scrapy",
        "pytest": "pytest-dev/pytest", "aiohttp": "aio-libs/aiohttp", "celery": "celery/celery", "sphinx": "sphinx-doc/sphinx"}
PR_SUBJ = re.compile(r"\(#(\d+)\)\s*$")
CLOSES = re.compile(r"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+(?:https?://github\.com/[\w.-]+/[\w.-]+/issues/|[\w.-]+/[\w.-]+#|#)(\d+)")


def clean_text(t, limit=3000):
    t = re.sub(r"<!--.*?-->", " ", t or "", flags=re.S)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", t)).strip()
    return t[:limit]


class GitHub:
    def __init__(self, token, cache):
        self.token, self.cache = token, cache
        os.makedirs(cache, exist_ok=True)

    def get(self, path):
        f = os.path.join(self.cache, hashlib.sha1(path.encode()).hexdigest() + ".json")
        if os.path.exists(f):
            return json.load(open(f))
        while True:
            req = urllib.request.Request("https://api.github.com" + path, headers={
                "Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json", "User-Agent": "ememoveri-issue-tasks"})
            try:
                data = json.load(urllib.request.urlopen(req, timeout=30))
                json.dump(data, open(f, "w"))
                return data
            except urllib.error.HTTPError as e:
                if e.code == 404 or e.code == 410:
                    json.dump(None, open(f, "w")); return None
                if e.code in (403, 429):
                    reset = int(e.headers.get("x-ratelimit-reset", time.time() + 60))
                    wait = max(5, reset - int(time.time()) + 2)
                    print(f"rate limited; sleeping {wait}s", flush=True); time.sleep(wait); continue
                raise


def ts(iso):
    import datetime as dt
    return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def build(repo, commits, gh, per_repo, min_chars=60):
    cfg, slug, tasks = DEFAULT_REPOS[repo], SLUG[repo], []
    first_clock = commits[int(len(commits) * 0.35)].clock if commits else 0
    cand = [c for c in commits if PR_SUBJ.search(c.subject) and c.clock >= first_clock]
    step = max(1, len(cand) // (per_repo * 4))          # oversample evenly, then keep the first per_repo that qualify
    for c in cand[::step]:
        if len(tasks) >= per_repo:
            break
        gold = sorted({fc.path for fc in c.changes if cfg.is_source(fc.path) and fc.status in ("M", "R")})
        if not 1 <= len(gold) <= 4 or any("bot" in c.author.lower() for _ in [0]):
            continue
        n = int(PR_SUBJ.search(c.subject).group(1))
        pr = gh.get(f"/repos/{slug}/pulls/{n}")
        if not pr or not pr.get("merged_at"):
            continue
        for x in dict.fromkeys(int(m) for m in CLOSES.findall(pr.get("body") or "")):
            iss = gh.get(f"/repos/{slug}/issues/{x}")
            if not iss or "pull_request" in iss:
                continue
            created = ts(iss["created_at"])
            text = clean_text((iss["title"] or "") + "\n\n" + (iss.get("body") or ""))
            if created >= c.clock or len(text) < min_chars:
                continue
            tasks.append({"id": f"{repo}#{x}", "repo": repo, "sha": c.sha, "pr": n, "issue": x, "cutoff": created,
                          "gold": gold, "queries": {"title": clean_text(iss["title"], 300), "full": text}})
            break
    return tasks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-dir-root", required=True); ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--per-repo", type=int, default=40); ap.add_argument("--out", default="issue_tasks.json")
    ap.add_argument("--cache", default=".gh_cache"); ap.add_argument("--mine-cache", default=".cache")
    a = ap.parse_args()
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok:
        sys.exit("GITHUB_TOKEN is not set (unauthenticated access allows only 60 requests/hour)")
    gh, allt = GitHub(tok, a.cache), []
    for r in a.repos:
        cs = mine_repo(os.path.join(a.git_dir_root, f"{r}.git"), r, cache_dir=a.mine_cache)
        t = build(r, cs, gh, a.per_repo); allt += t
        print(r, len(t), "tasks", flush=True)
        json.dump(allt, open(a.out, "w"), indent=1)
    print("wrote", a.out, len(allt))
