"""EmemoVeri-2: evidence-level scoring over the prefix of the ledger that existed at task time."""
import numpy as np
import pytest

from ememoveri import mining
from ememoveri.evidence_retrieval import EvidenceConfig, item_scores, rank
from ememoveri.mining import RepoConfig
from ememoveri.realbench_ext import collect
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from helpers_toy import C, history, CFG  # noqa: E402

mining.DEFAULT_REPOS["toy"] = CFG


def collected():
    cs = {"toy": history()}
    return collect(cs, ["toy"], None, warmup_frac=0.2, max_tasks_per_repo=12, dev_frac=0.3, max_events_per_type=5)


def test_max_aggregation_reproduces_the_online_channel():
    out = collected(); L = out["ledger"]
    cfg = EvidenceConfig(agg="max")
    for t in out["tasks"]:
        assert np.allclose(item_scores(L, t, cfg), t.sims["knn_max"], atol=1e-6)
        assert np.allclose(item_scores(L, t, EvidenceConfig(agg="sum")), t.sims["knn_sum"], atol=1e-5)


def test_no_future_evidence_is_used():
    out = collected(); L = out["ledger"]
    t = out["tasks"][0]
    n = len(t.F.sup)
    assert t.nc <= len(L["c_clock"])
    # every ledger entry used at this task belongs to a commit registered before the task and a claim that already existed
    end = L["ptr"][n]
    used = L["commit"][:end][L["commit"][:end] < t.nc]
    assert (L["c_clock"][used] <= t.clock).all()


def test_strength_weighting_downweights_bulk_commits():
    out = collected(); L = out["ledger"]
    L2 = {k: v.copy() for k, v in L.items()}
    L2["c_strength"][:] = 1.0
    L3 = {k: v.copy() for k, v in L.items()}
    L3["c_strength"][:] = 0.3
    t = out["tasks"][-1]
    cfg = EvidenceConfig(agg="max", strength_exp=1.0)
    a, b = item_scores(L2, t, cfg), item_scores(L3, t, cfg)
    assert np.allclose(b, 0.3 * a, atol=1e-6) and a.max() > 0


def test_revert_cancellation_zeroes_cancelled_items():
    out = collected(); L = out["ledger"]
    t = out["tasks"][-1]
    L2 = {k: v.copy() for k, v in L.items()}
    L2["c_cancel"][:] = 0.0                      # pretend every commit was reverted before the task
    cfg = EvidenceConfig(agg="max", revert_cancel=True)
    assert item_scores(L2, t, cfg).max() == 0.0
    assert item_scores(L, t, cfg).max() > 0.0    # nothing is cancelled in the toy history


def test_rank_returns_live_local_claims_only():
    out = collected(); L = out["ledger"]
    cfg = EvidenceConfig(w_sim=1.0, w_ev=1.0, l_pop=0.05, l_rec=0.05, H=30.0)
    for t in out["tasks"]:
        idx = rank(L, t, cfg)
        assert len(idx) and not t.F.sup[idx].any() and t.local[idx].all()
