"""Metric validation. Run: pytest -q   (all must pass before any modelling)."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.metrics import auc, impression_metrics


# --- official MIND reference implementations (for cross-checking) -------------
def mind_mrr(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    return np.sum(y_true / (np.arange(len(y_true)) + 1)) / np.sum(y_true)


def mind_dcg(y_true, y_score, k):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.sum((2 ** y_true - 1) / np.log2(np.arange(len(y_true)) + 2))


def mind_ndcg(y_true, y_score, k):
    return mind_dcg(y_true, y_score, k) / mind_dcg(y_true, y_true, k)


# --- 1. hand-checked toy cases (same as the handbook) ---------------------------
@pytest.mark.parametrize("labels, scores, exp_auc, exp_mrr, exp_ndcg5", [
    ([1, 0, 0, 0], [0.9, 0.5, 0.3, 0.1], 1.00, 1.00, 1.00),
    ([1, 0, 0, 0], [0.1, 0.5, 0.3, 0.9], 0.00, 0.25, 1 / np.log2(5)),
    ([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 0.75, (1 + 1 / 3) / 2, 1.5 / (1 + 1 / np.log2(3))),
])
def test_toy_cases(labels, scores, exp_auc, exp_mrr, exp_ndcg5):
    m = impression_metrics(labels, scores)
    assert m["auc"] == pytest.approx(exp_auc)
    assert m["mrr"] == pytest.approx(exp_mrr)
    assert m["ndcg@5"] == pytest.approx(exp_ndcg5)


def test_precision_recall_hitrate():
    labels = [0, 1, 0, 0, 0, 0, 1]
    scores = [7, 6, 5, 4, 3, 2, 1]          # item at index 6 is ranked 7th
    m = impression_metrics(labels, scores)
    assert m["p@5"] == pytest.approx(1 / 5)
    assert m["r@5"] == pytest.approx(1 / 2)
    assert m["hr@5"] == 1.0
    assert m["p@10"] == pytest.approx(2 / 10)
    assert m["r@10"] == pytest.approx(1.0)


# --- 2. agreement with sklearn and the official MIND code -----------------------
def random_impressions(n=300, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        size = rng.integers(2, 40)
        y = np.zeros(size)
        y[rng.choice(size, rng.integers(1, size), replace=False)] = 1
        if y.sum() == size:
            y[0] = 0
        s = rng.random(size)
        if ties:
            s = np.round(s, 1)
        yield y, s


def test_auc_matches_sklearn_including_ties():
    for y, s in random_impressions(ties=True):
        assert auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_mrr_and_ndcg_match_official_mind_code():
    for y, s in random_impressions(ties=False):
        m = impression_metrics(y, s)
        assert m["mrr"] == pytest.approx(mind_mrr(y, s))
        assert m["ndcg@5"] == pytest.approx(mind_ndcg(y, s, 5))
        assert m["ndcg@10"] == pytest.approx(mind_ndcg(y, s, 10))


# --- 3. behaviour checks -----------------------------------------------------
def test_constant_scores_give_auc_half():
    assert impression_metrics([1, 0, 0, 1, 0], [0, 0, 0, 0, 0])["auc"] == 0.5


def test_monotonic_transform_changes_nothing():
    y, s = [0, 1, 0, 1, 0, 0], np.array([0.2, 0.9, 0.4, 0.3, 0.8, 0.1])
    a, b = impression_metrics(y, s), impression_metrics(y, 3 * np.exp(s) + 7)
    for key in a:
        assert a[key] == pytest.approx(b[key])


def test_list_shorter_than_k():
    m = impression_metrics([1, 0, 0], [0.9, 0.2, 0.1])
    assert m["ndcg@10"] == pytest.approx(1.0)
    assert m["hr@10"] == 1.0


def test_ties_are_not_resolved_by_log_order():
    # clicked item is logged first; a constant scorer must NOT always rank it first
    rng = np.random.default_rng(1)
    mrrs = [impression_metrics([1] + [0] * 9, [0.0] * 10, rng=rng)["mrr"] for _ in range(2000)]
    assert np.mean(mrrs) == pytest.approx(np.mean(1 / np.arange(1, 11)), abs=0.02)
