"""Shared toy repository history for the extension tests (kept separate so test modules do not import each other)."""
from ememoveri import mining
from ememoveri.mining import Commit, FileChange, RepoConfig

CFG = RepoConfig("toy", ("pkg/",), ("tests/",))
mining.DEFAULT_REPOS["toy"] = CFG
DAY = 86400.0


def C(i, changes, subject="Fix handling of widgets in parser", reverts=None):
    chg = [FileChange(st, p, *(o[:1])) for st, p, *o in changes]
    clock = 1_600_000_000.0 + i * DAY
    return Commit(sha=f"{i:040x}", clock=clock, raw_ts=clock, author="dev", subject=subject, body="", changes=chg,
                  reverts=reverts, repo="toy")


def history():
    cs = [C(0, [("A", "pkg/parser.py"), ("A", "pkg/render.py")], "Add parser and renderer")]
    for i in range(1, 30):
        f = "pkg/parser.py" if i % 3 else "pkg/render.py"
        cs.append(C(i, [("M", f)], f"Improve {'parser' if f.endswith('parser.py') else 'renderer'} handling {i}"))
    cs.append(C(30, [("R", "pkg/render2.py", "pkg/render.py")], "Rename renderer module"))
    cs.append(C(31, [("D", "pkg/parser.py")], "Remove parser"))
    cs.append(C(32, [("M", "pkg/render2.py")], "Improve renderer output"))
    return cs
