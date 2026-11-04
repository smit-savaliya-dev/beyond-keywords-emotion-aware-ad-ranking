"""Tests for src/fusion.py (min-max, fusion, D54 checks), with small fakes: no internet, GPU or real lexicons."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate
from src.fusion import emotion_scores, fit_profile_pool, fuse, grid_configs, minmax_within, same_metrics, shuffled_control


def test_minmax_is_within_group_and_constant_goes_to_zero():
    out = minmax_within([0, 0, 0, 1, 1, 2], [1, 2, 3, 10, 30, 7])
    assert out.tolist() == [0, 0.5, 1, 0, 1, 0]
    assert minmax_within([0, 0], [5, 5]).tolist() == [0, 0]


def test_fuse_endpoints():
    b, e = np.array([0, 1.0]), np.array([1.0, 0])
    assert fuse(b, e, 1.0).tolist() == [0, 1] and fuse(b, e, 0.0).tolist() == [1, 0]
    assert fuse(b, e, 0.3).tolist() == pytest.approx([0.7, 0.3])


def test_grid_order_favours_primary_and_m2_side():
    g = grid_configs([0.0, 0.5, 1.0], ["a", "b"])
    assert g[0] == {"variant": "a", "alpha": 1.0} and g[2]["alpha"] == 0.0 and g[3]["variant"] == "b" and len(g) == 6


def test_emotion_scores_relations():
    vectors = np.array([[1.0, 0.2], [0.1, 1.0], [0.5, 0.5]])
    out = emotion_scores(np.array([0, 0, 1]), np.array([0, 1, 2]), [np.array([0]), np.empty(0, dtype=np.int64)], vectors, other=np.zeros(3))
    assert out["cosine"][0] == pytest.approx(1.0)
    assert out["mood_repair"] == pytest.approx(1 - out["cosine"])
    assert out["user_specific"] == pytest.approx(out["cosine"])
    assert out["personal"][2] == pytest.approx(0, abs=1e-6)     # cold start uses the corpus mean, so personal is 0
    assert "user_specific" not in emotion_scores(np.array([0]), np.array([0]), [np.array([0])], vectors)


def test_fit_profile_pool_reads_fit_only_and_drops_cold_profiles(tmp_path):
    beh = pd.DataFrame({"user_id": ["u1", "u2", "u3"], "part": ["fit", "fit", "dev"],
                        "history": [np.array(["N0"], dtype=object), np.array([], dtype=object), np.array(["N1"], dtype=object)]})
    beh.to_parquet(tmp_path / "behaviors.parquet")
    pool, users = fit_profile_pool(pd.Index(["N0", "N1"]), np.array([[1.0, 0.0], [0.0, 1.0]]), tmp_path)
    assert users.tolist() == ["u1"] and pool.tolist() == [[1.0, 0.0]]


def per(keys, auc):
    return pd.DataFrame({"imp_key": keys, "auc": auc})


def test_same_metrics():
    a = per(["a", "b"], [0.5, 0.7])
    assert same_metrics(a, a.copy(), ["auc"]) == (True, 0.0)
    assert not same_metrics(a, per(["a", "b"], [0.5, 0.71]), ["auc"])[0]
    assert not same_metrics(a, per(["a"], [0.5]), ["auc"])[0]


def test_alpha_one_reproduces_the_harness_result():
    truth = pd.DataFrame({"imp_key": ["a"] * 3 + ["b"] * 3, "news_id": list("xyzxyz"), "label": [1, 0, 0, 0, 0, 1], "position": [0, 1, 2] * 2})
    s = np.array([0.2, 0.2, 0.1, -0.3, 0.1, 0.0])
    imps = pd.factorize(truth["imp_key"])[0]
    p1, _ = evaluate(truth, truth[["imp_key", "news_id"]].assign(score=s))
    fused = fuse(minmax_within(imps, s), np.array([1.0, 0, 0, 0, 1, 0]), 1.0)
    p2, _ = evaluate(truth, truth[["imp_key", "news_id"]].assign(score=fused))
    assert same_metrics(p1, p2)[0]


def test_shuffled_control_flags_a_real_gain_only():
    rng = np.random.default_rng(0)
    keys = [str(i) for i in range(400)]
    base = rng.uniform(0.4, 0.8, 400)
    t, ok = shuffled_control(per(keys, base), {0.5: per(keys, base + rng.normal(0, 0.01, 400))})
    assert ok and not t["beats_m2"].any()
    t, ok = shuffled_control(per(keys, base), {0.5: per(keys, base - 0.01), 0.9: per(keys, base + 0.05)})
    assert not ok and t.set_index("alpha")["beats_m2"].to_dict() == {0.5: False, 0.9: True}
