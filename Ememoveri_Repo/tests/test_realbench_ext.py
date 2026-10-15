"""Tests for the v0.3/v0.4 extensions: BM25, query degradation, metrics, rename carry-over, collector, external tasks."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from ememoveri import mining
from ememoveri.mining import Commit, FileChange, RepoConfig
from ememoveri.realbench_ext import (ExtMemory, METRICS, TaskRec, collect, collect_external, degrade, metrics_vec,
                                     rank_generic)

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from helpers_toy import CFG, C, DAY, history  # noqa: E402


def test_degrade_is_deterministic_and_never_empty():
    q = "fix handling of widgets in the parser module"
    assert degrade(q, 0.5, "abc") == degrade(q, 0.5, "abc")
    assert degrade(q, 0.0, "abc") == q
    for k in range(50):
        out = degrade(q, 0.99, f"k{k}")
        assert out and set(out.split()) <= set(q.split())
    assert len(degrade(q, 0.5, "x").split()) < len(q.split())


def test_metrics_vec_ndcg_recall_and_dedupe():
    mem = SimpleNamespace(repo_of=[0, 0, 0, 0], path_of=["a.py", "b.py", "c.py", "a.py"])
    F = SimpleNamespace(sup=np.array([False, False, True, False]))
    idx = np.array([2, 0, 3, 1])                # c.py (dead), a.py, a.py (duplicate path), b.py
    m = dict(zip(METRICS, metrics_vec(mem, F, idx, {(0, "a.py"), (0, "b.py")}, 0)))
    assert m["hit1"] == 0.0 and m["hit3"] == 1.0 and m["hit5"] == 1.0
    assert m["rr"] == pytest.approx(1 / 2)
    assert m["rec5"] == 1.0 and m["prec5"] == pytest.approx(2 / 5)
    assert m["ndcg5"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert m["dead5"] == pytest.approx(1 / 3)   # c.py is one of three distinct top-5 files
    assert m["contam5"] == 0.0


def test_rank_generic_orders_by_score_then_recency():
    s = np.array([0.5, 0.9, 0.9, 0.1]); upd = np.array([1.0, 1.0, 5.0, 9.0]); mask = np.array([True, True, True, False])
    assert list(rank_generic(s, mask, upd)) == [2, 1, 0]


def test_bm25_prefers_matching_claim_and_is_normalised():
    mem = ExtMemory({"toy": CFG})
    for c in history()[:10]:
        mem.ingest(c)
    s = mem.bm25("improve parser handling")
    paths = mem.path_of
    assert s.max() == pytest.approx(1.0) and s.min() >= 0
    assert paths[int(np.argmax(s))] == "pkg/parser.py"
    assert mem.bm25("zzzz qqqq").max() == 0.0


def test_rename_carry_over_switch():
    cs = history()
    for carry in (True, False):
        mem = ExtMemory({"toy": CFG}, carry=carry)
        for c in cs[:31]:
            mem.ingest(c)
        new = mem.live[(mem.repo_ids["toy"], "pkg/render2.py")]
        old = mem.claims  # noqa
        n_touch = mem.n_touch[new]
        if carry:
            assert n_touch >= 5 and any("renderer" in s for s in mem.subjects[new])
        else:
            assert n_touch <= 1 and not any("renderer handling" in s for s in mem.subjects[new])


def test_superseded_claim_never_revived():
    mem = ExtMemory({"toy": CFG})
    for c in history():
        mem.ingest(c)
    assert mem.resurrected == 0
    assert (mem.repo_ids["toy"], "pkg/parser.py") not in mem.live


def test_collect_scores_tasks_before_ingest_and_records_lifecycle():
    cs = {"toy": history()}
    out = collect(cs, ["toy"], None, warmup_frac=0.2, max_tasks_per_repo=10, dev_frac=0.3, max_events_per_type=10)
    assert out["tasks"], "no tasks selected"
    for t in out["tasks"]:
        assert isinstance(t, TaskRec) and {"tfidf", "bm25"} <= set(t.sims)
        assert t.F.sup.shape == t.sims["tfidf"].shape == t.n_touch.shape
        assert t.split in ("dev", "test")
    assert out["counts"]["delete"] == 1 and out["counts"]["rename"] == 1
    ev = {e["type"]: e for e in out["events"]}
    assert ev["delete"]["post"]["carry:naive"]["old_in_top5"] is True       # naive memory keeps returning the dead file
    assert ev["delete"]["post"]["carry:snapshot"]["old_in_top5"] is False   # state filter removes it
    assert ev["rename"]["post"]["carry:snapshot"]["new_in_top5"] is True    # history followed the rename
    assert ev["rename"]["post"]["nocarry:snapshot"]["new_in_top5"] <= ev["rename"]["post"]["carry:snapshot"]["new_in_top5"]


def test_collect_external_respects_cutoff_and_drops_unknown_gold():
    cs = {"toy": history()}
    cutoff = cs["toy"][12].clock + 1                      # after commit 12, before commit 13
    tasks = [{"id": "toy#1", "repo": "toy", "sha": "x", "cutoff": cutoff, "gold": ["pkg/parser.py", "pkg/render2.py"],
              "queries": {"full": "parser handling is broken", "title": "parser broken"}},
             {"id": "toy#2", "repo": "toy", "sha": "y", "cutoff": cutoff, "gold": ["pkg/render2.py"],   # did not exist yet
              "queries": {"full": "renderer", "title": "renderer"}}]
    out = collect_external(cs, ["toy"], tasks)
    assert [t.sha for t in out["tasks"]] == ["toy#1"]     # task 2 has no gold file alive at the cutoff
    assert out["dropped_gold"] == 2                        # render2.py dropped from both tasks
    t = out["tasks"][0]
    assert {"tfidf:full", "tfidf:title", "bm25:full"} <= set(t.sims)
    assert {p for _, p in t.gold} == {"pkg/parser.py"}
    assert t.F.sup.shape[0] == len(set(p for p in out["path_of"][:t.F.sup.shape[0]]))   # no claims from the future


def test_similar_history_and_cochange_channels():
    mem = ExtMemory({"toy": CFG})
    cs = history()
    for c in cs[:20]:
        mem.ingest(c)
    n = len(mem.claims)
    sim = mem.similarities("improve parser handling")
    mask = np.array([not s for s in mem.superseded], dtype=bool)
    ex = mem.extra("improve parser handling", sim, mask)
    assert set(ex) == {"knn_max", "knn_sum", "cochange"} and all(v.shape == (n,) for v in ex.values())
    parser = mem.live[(mem.repo_ids["toy"], "pkg/parser.py")]
    render = mem.live[(mem.repo_ids["toy"], "pkg/render.py")]
    assert ex["knn_max"][parser] > ex["knn_max"][render] >= 0.0           # parser subjects match the query better
    assert ex["knn_sum"][parser] >= ex["knn_max"][parser]
    assert ex["cochange"].max() <= 1.0 and ex["cochange"][int(np.argmax(sim))] == 0.0   # the seed is never its own neighbour
    # history is built only from commits already ingested: nothing from the future
    assert max(max(h) for h in mem.hist if h) < len(mem.c_idx)
    # a rename keeps the history of the renamed file
    for c in cs[20:31]:
        mem.ingest(c)
    new = mem.live[(mem.repo_ids["toy"], "pkg/render2.py")]
    assert len(mem.hist[new]) >= 5
