import numpy as np
import pandas as pd
import pytest

from src.congruence import cosine_rows, neg_kl_rows, user_profiles

VEC = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])


def test_user_profiles_recency_weights_and_cold_start():
    fallback = np.array([0.2, 0.8])
    prof, used = user_profiles([[0, 1], [], [-1, 2], [-1]], VEC, fallback, decay=0.5, max_items=50)
    # [0, 1]: newest (1) weight 1, older (0) weight 0.5 -> (0.5, 1) / 1.5
    assert np.allclose(prof[0], [1 / 3, 2 / 3]) and used[0] == 2
    assert np.allclose(prof[1], fallback) and used[1] == 0           # empty history -> corpus mean
    assert np.allclose(prof[2], [0.5, 0.5]) and used[2] == 1         # unknown ids skipped
    assert np.allclose(prof[3], fallback) and used[3] == 0


def test_user_profiles_keeps_only_last_items():
    prof, used = user_profiles([[0, 0, 0, 1]], VEC, np.zeros(2), decay=0.9, max_items=1)
    assert np.allclose(prof[0], [0.0, 1.0]) and used[0] == 1


def test_cosine_rows():
    a = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    b = np.array([[2.0, 0.0], [1.0, -1.0], [1.0, 1.0]])
    assert np.allclose(cosine_rows(a, b), [1.0, 0.0, 0.0])           # zero vector -> 0, no division error


def test_neg_kl_rows():
    p = np.array([[0.5, 0.5], [0.9, 0.1]])
    q = np.array([[0.5, 0.5], [0.5, 0.5]])
    out = neg_kl_rows(p, q, eps=0.0)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(-(0.9 * np.log(0.9 / 0.5) + 0.1 * np.log(0.1 / 0.5)))
    assert np.isfinite(neg_kl_rows(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), eps=1e-6)).all()
    assert neg_kl_rows(np.array([[2.0, 2.0]]), np.array([[1.0, 1.0]]), eps=0.0)[0] == pytest.approx(0.0)  # renormalised


def test_plot_dz_panels(tmp_path):
    from src.click_signal import plot_dz_panels
    t = pd.DataFrame({"row": ["a", "a", "b"], "series": ["x", "y", "x"], "dz": [0.1, -0.1, 0.05],
                      "dz_ci_low": [0.0, -0.2, 0.0], "dz_ci_high": [0.2, 0.0, 0.1],
                      "significant_both": [True, False, True]})
    path = tmp_path / "p.png"
    plot_dz_panels([("one", t), ("two", t)], path, "title", "subtitle")
    assert path.stat().st_size > 0
