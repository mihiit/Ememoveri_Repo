from ememoveri.metrics import precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k, mean


def test_precision_at_k_basic():
    rel = [True, False, True, False]
    assert precision_at_k(rel, 2) == 0.5
    assert precision_at_k(rel, 4) == 0.5


def test_recall_at_k_basic():
    rel = [True, False, True, False]
    assert recall_at_k(rel, 4, n_relevant_total=2) == 1.0
    assert recall_at_k(rel, 1, n_relevant_total=2) == 0.5


def test_reciprocal_rank_first_hit():
    assert reciprocal_rank([False, True, True]) == 0.5
    assert reciprocal_rank([True]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg_perfect_ranking_is_one():
    gains = [3, 2, 1, 0]
    assert ndcg_at_k(gains, 4) == 1.0


def test_ndcg_reversed_ranking_is_lower():
    perfect = ndcg_at_k([3, 2, 1, 0], 4)
    reversed_ = ndcg_at_k([0, 1, 2, 3], 4)
    assert reversed_ < perfect


def test_mean_skips_nan():
    assert mean([1.0, 2.0, float("nan"), 3.0]) == 2.0
