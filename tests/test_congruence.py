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


def test_neg_abs_distance():
    from src.congruence import neg_abs_distance
    assert np.allclose(neg_abs_distance(np.array([[0.5], [-0.2]]), np.array([[0.1], [-0.2]])), [-0.4, 0.0])


def test_history_index_lists_maps_unknown_to_minus_one():
    from src.congruence import history_index_lists
    out = history_index_lists([np.array(["N2", "N9", "N1"]), np.array([], dtype=object)], pd.Index(["N1", "N2"]))
    assert out[0].tolist() == [1, -1, 0] and out[1].tolist() == []
    assert history_index_lists([np.array([], dtype=object)], pd.Index(["N1"]))[0].tolist() == []


def test_congruence_features_raw_generic_personal_and_chunking():
    from src.congruence import congruence_features
    vec = {"v": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), "val": np.array([0.5, -0.5, 0.0])}
    histories = [np.array([0]), np.array([], dtype=np.int64)]       # profile 0 = item 0; profile 1 cold start
    groups = np.array([0, 0, 1, 1])
    item_idx = np.array([0, 1, 0, 2])
    measures = [("v", "cos", "main"), ("val", "absdist", "variant")]
    out, used = congruence_features(groups, item_idx, histories, vec, measures, chunk_size=3)
    assert used.tolist() == [1, 0]
    assert np.allclose(out["cong_v_cos"], [1.0, 0.0, cosine_rows([[1, 0]], [[2 / 3, 2 / 3]])[0], 1.0])
    mean_cos = cosine_rows(vec["v"][item_idx], np.tile(vec["v"].mean(0), (4, 1)))
    assert np.allclose(out["cong_v_cos_generic"], mean_cos)
    assert np.allclose(out["cong_v_cos_personal"], out["cong_v_cos"] - out["cong_v_cos_generic"])
    assert np.allclose(out["cong_val_absdist"], [0.0, -1.0, -0.5, 0.0], atol=1e-6)   # cold start -> mean 0.0
    assert np.allclose(out["cong_val_absdist_personal"][2:], 0.0)                      # cold start: personal = 0
    assert all(a.dtype == np.float32 for a in out.values())
    whole, _ = congruence_features(groups, item_idx, histories, vec, measures, chunk_size=100)
    assert all(np.array_equal(whole[k], out[k]) for k in out)


def test_congruence_features_other_user_control():
    from src.congruence import congruence_features
    vec = {"v": np.array([[1.0, 0.0], [0.0, 1.0]])}
    histories = [np.array([0]), np.array([1])]                      # user 0 likes item 0, user 1 likes item 1
    groups, item_idx = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    out, _ = congruence_features(groups, item_idx, histories, vec, [("v", "cos", "main")], shuffle_seed=0)
    perm = np.random.default_rng(0).permutation(2)
    other = vec["v"][histories[0]][0], vec["v"][histories[1]][0]
    expected = [cosine_rows([vec["v"][i]], [other[perm[g]]])[0] for g, i in zip(groups, item_idx)]
    assert np.allclose(out["cong_v_cos_otheruser"], expected)
    assert np.allclose(out["cong_v_cos_userspecific"], out["cong_v_cos"] - out["cong_v_cos_otheruser"])
    plain, _ = congruence_features(groups, item_idx, histories, vec, [("v", "cos", "main")])
    assert "cong_v_cos_otheruser" not in plain


def test_masked_values_is_lazy_view():
    from src.click_signal import MaskedValues
    m = MaskedValues({"a": np.arange(4), "b": np.arange(4) * 2}, np.array([True, False, True, False]))
    assert list(m) == ["a", "b"] and len(m) == 2 and m["b"].tolist() == [0, 4]


def test_window_histories_skip_size_and_unknown_items():
    from src.congruence import window_histories
    h = [np.array([1, -1, 2, 3, 4, 5, 6]), np.array([7]), np.array([], dtype=np.int64)]
    assert [x.tolist() for x in window_histories(h, size=2)] == [[5, 6], [7], []]
    assert [x.tolist() for x in window_histories(h, skip=2, size=2)] == [[3, 4], [], []]
    assert [x.tolist() for x in window_histories(h)] == [[1, 2, 3, 4, 5, 6], [7], []]


def test_congruence_features_window_no_decay_is_plain_mean():
    from src.congruence import congruence_features, window_histories
    vec = {"v": np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])}
    hist = window_histories([np.array([0, 1])], size=2)
    plain, _ = congruence_features(np.array([0]), np.array([2]), hist, vec, [("v", "cos", "main")], decay=1.0, max_items=2)
    assert plain["cong_v_cos"][0] == pytest.approx(1.0)                  # mean of items 0 and 1 points along item 2
    decayed, _ = congruence_features(np.array([0]), np.array([2]), hist, vec, [("v", "cos", "main")], decay=0.1, max_items=2)
    assert decayed["cong_v_cos"][0] < 0.9                                # newest item dominates


def test_window_congruence_returns_three_roles():
    from src.congruence import window_congruence
    vec = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = window_congruence(np.array([0, 0]), np.array([0, 1]), [np.array([0, 1])], vec, "w", {"skip": 1, "size": 1, "decay": 1.0}, 0)
    assert sorted(out) == ["w", "w_personal", "w_userspecific"] and out["w"].tolist() == pytest.approx([1.0, 0.0])
