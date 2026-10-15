import os, random, subprocess, tempfile
from datetime import datetime, timezone

import numpy as np
import pytest

from ememoveri.mining import (Commit, FileChange, RepoConfig, clean_subject, is_noise, mine_repo, parse_log,
                              timestamp_anomalies)
from ememoveri.realbench import (BenchConfig, Cond, Features, Memory, eligible_task_indices, rank_claims,
                                 run_benchmark, score_vec)
from ememoveri.retrieval import RetrievalScorer, RetrievalWeights
from ememoveri.schema import MemoryClaim, MemoryState
from ememoveri.stats import aurc, auroc, ece, holm, mcnemar_exact, paired_bootstrap

CFG = RepoConfig("toy", ("pkg/",), ("tests/",))


def C(i, status_paths, subject="Fix handling of widgets in parser", clock=None, reverts=None, author="dev"):
    chg = []
    for st, p, *old in status_paths:
        chg.append(FileChange(st, p, old[0] if old else None))
    return Commit(sha=f"{i:040x}", clock=float(clock or 1_600_000_000 + i * 86400), raw_ts=float(clock or 1_600_000_000 + i * 86400),
                  author=author, subject=subject, body="", changes=chg, reverts=reverts, repo="toy")


def test_mining_real_git_repo_structural_relations():
    with tempfile.TemporaryDirectory() as d:
        run = lambda *a: subprocess.run(a, cwd=d, check=True, capture_output=True, text=True,
                                        env={**os.environ, "GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": "a@a", "GIT_COMMITTER_NAME": "a", "GIT_COMMITTER_EMAIL": "a@a"})
        run("git", "init", "-q")
        os.makedirs(os.path.join(d, "pkg"))
        open(os.path.join(d, "pkg/a.py"), "w").write("x = 1\n" * 30)
        run("git", "add", "."); run("git", "commit", "-qm", "Add module a with parsing support")
        open(os.path.join(d, "pkg/a.py"), "a").write("y = 2\n")
        run("git", "commit", "-qam", "Improve module a parsing")
        run("git", "mv", "pkg/a.py", "pkg/b.py"); run("git", "commit", "-qm", "Rename a to b")
        run("git", "rm", "-q", "pkg/b.py"); run("git", "commit", "-qm", "Remove b")
        commits = mine_repo(os.path.join(d, ".git"), "toy")
    st = [[(fc.status, fc.path, fc.old_path) for fc in c.changes] for c in commits]
    assert st[0] == [("A", "pkg/a.py", None)] and st[1] == [("M", "pkg/a.py", None)]
    assert st[2] == [("R", "pkg/b.py", "pkg/a.py")] and st[3] == [("D", "pkg/b.py", None)]
    assert [c.clock for c in commits] == sorted(c.clock for c in commits)


def test_parse_revert_trailer_and_clock_monotone():
    raw = ("\x1e" + "a" * 40 + "\x1f100\x1fn\x1fRevert thing\x1fThis reverts commit " + "b" * 40 + ".\x1f\nM\tpkg/x.py\n"
           "\x1e" + "c" * 40 + "\x1f50\x1fn\x1fLater but earlier stamp\x1f\x1f\nM\tpkg/x.py\n")
    cs = parse_log(raw, "toy")
    assert cs[0].reverts == "b" * 40
    assert cs[1].clock == 100.0 and cs[1].raw_ts == 50.0 and timestamp_anomalies(cs) == 1


def test_subject_cleaning_and_noise():
    assert clean_subject("\U0001F41B Fix bug in router (#1234)") == "Fix bug in router"
    assert is_noise(C(1, [("M", "pkg/a.py")], author="dependabot[bot]"))
    assert not is_noise(C(1, [("M", "pkg/a.py")]))


def test_vectorised_scorer_matches_reference_scorer():
    rng = random.Random(3)
    sc = RetrievalScorer(weights=RetrievalWeights(0.4, 0.3, 0.2, 0.1, 0.05), staleness_half_life_days=60.0)
    mem = Memory({"toy": CFG})
    t0 = 1_600_000_000
    mem.ingest(C(0, [("A", "pkg/a.py")], clock=t0)); mem.ingest(C(1, [("M", "pkg/a.py")], clock=t0 + 86400 * 20))
    mem.ingest(C(2, [("A", "pkg/b.py")], clock=t0 + 86400 * 5))
    now = t0 + 86400 * 90
    F = mem.features(now)
    cond = Cond("x", 0.4, 0.3, 0.2, 0.1, 0.05, half_life=60.0)
    sim = np.array([0.7, 0.2])
    vec = score_vec(cond, sim, F, np.array([True, True]))
    for i, c in enumerate(mem.claims):
        ref = sc.score(c, float(sim[i]), "toy", reference=datetime.fromtimestamp(now, tz=timezone.utc))
        assert abs(ref - vec[i]) < 1e-9


def test_rename_inherits_history_and_supersedes_old_claim():
    mem = Memory({"toy": CFG})
    mem.ingest(C(0, [("A", "pkg/a.py")])); mem.ingest(C(1, [("M", "pkg/a.py")]))
    mem.ingest(C(2, [("R", "pkg/b.py", "pkg/a.py")]))
    old, new = mem.claims[0], mem.claims[1]
    assert old.state == MemoryState.SUPERSEDED and old.superseded_by == new.claim_id
    assert len(new.evidence) == len(old.evidence) and (0, "pkg/b.py") in mem.live and (0, "pkg/a.py") not in mem.live
    mem.ingest(C(3, [("M", "pkg/a.py")]))   # path reused later => new independent claim
    assert mem.claims[-1].path_of if False else mem.path_of[-1] == "pkg/a.py" and mem.claims[-1].state != MemoryState.SUPERSEDED


def test_revert_contradicts_and_delete_supersedes():
    mem = Memory({"toy": CFG})
    mem.ingest(C(0, [("A", "pkg/a.py")])); mem.ingest(C(1, [("A", "pkg/z.py")]))
    mem.ingest(C(2, [("M", "pkg/a.py")], reverts="b" * 40))
    assert mem.claims[0].state == MemoryState.CONTESTED or any(e.relation.value == "CONTRADICTS" for e in mem.claims[0].evidence)
    mem.ingest(C(3, [("D", "pkg/z.py")]))
    assert mem.superseded == [False, True]


def test_tasks_exclude_cold_files_reverts_and_structural_commits():
    cs = [C(i, [("A", f"pkg/f{i}.py")], subject=f"Add feature number {i} to package") for i in range(6)]
    cs += [C(10, [("M", "pkg/f0.py")], subject="Tune parser behaviour for widgets"),
           C(11, [("M", "pkg/never_seen.py")], subject="Tune parser behaviour for gadgets"),
           C(12, [("M", "pkg/f1.py")], subject="Revert tuning of parser", reverts="a" * 40),
           C(13, [("M", "pkg/f1.py"), ("D", "pkg/f2.py")], subject="Tune parser behaviour and remove file")]
    idx, st = eligible_task_indices(cs, CFG, warmup_frac=0.0)
    assert idx == [6] and st["cold_file_excluded"] == 1 and st["structural_excluded"] == 7 and st["noise_or_short"] == 1  # 6 adds + 1 delete; 1 revert


def test_replay_never_sees_future_and_leak_canary_fires():
    rng = random.Random(0)
    names = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]
    cs = []
    for i in range(60):
        n = names[i % len(names)]
        st = "A" if i < len(names) else "M"
        cs.append(C(i, [(st, f"pkg/{n}.py")], subject=f"Improve {n} handling of request {rng.choice(['cache','queue','retry'])} logic", clock=1_600_000_000 + i * 86400))
    res = run_benchmark({"toy": cs}, BenchConfig(repos=("toy",), warmup_frac=0.3, max_tasks_per_repo=30), log=lambda *_: None,
                        repo_configs={"toy": CFG})
    assert res["rows"], "no tasks scored"
    leaked = np.mean([r["m"]["LEAKED::SNAPSHOT_RAG"]["hit1"] for r in res["rows"]])
    honest = np.mean([r["m"]["SNAPSHOT_RAG"]["hit1"] for r in res["rows"]])
    assert leaked == 1.0 and honest <= leaked


def test_stats_helpers():
    a, b = [1, 1, 1, 0, 1, 1, 1, 1], [0, 0, 1, 0, 0, 1, 0, 1]
    d, lo, hi = paired_bootstrap(a, b, n=2000)
    assert d > 0 and lo <= d <= hi
    n01, n10, p = mcnemar_exact(a, b)
    assert (n01, n10) == (0, 4) and abs(p - 0.125) < 1e-12
    assert holm({"a": 0.01, "b": 0.04})["a"] == pytest.approx(0.02) and holm({"a": 0.01, "b": 0.04})["b"] == pytest.approx(0.04)
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert aurc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) < aurc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])
    assert ece([1.0, 1.0], [0, 0]) == 1.0
