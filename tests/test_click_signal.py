import numpy as np
import pandas as pd
import pytest

from src.click_signal import ColumnValues, holm, signal_table, signal_test, within_group_diffs


def test_within_group_diffs_skips_one_class_groups_and_nan():
    groups = [0, 0, 0, 1, 1, 2, 2, 2]
    labels = [1, 0, 0, 1, 1, 1, 0, 0]
    values = [5.0, 1.0, 3.0, 9.0, 9.0, 4.0, np.nan, 2.0]
    d = within_group_diffs(groups, labels, values)
    assert np.allclose(d, [5 - 2, 4 - 2])                 # group 1 has no unclicked; NaN ignored in group 2
    all_nan = within_group_diffs([0, 0], [1, 0], [np.nan, 1.0])
    assert len(all_nan) == 0


def test_signal_test_direction_and_zeros():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 1.0, 400)
    r = signal_test(d)
    assert r["n_groups"] == 400 and r["dz"] == pytest.approx(d.mean() / d.std(ddof=1))
    assert r["p"] < 1e-6 and r["dz_ci_low"] < r["dz"] < r["dz_ci_high"]
    z = signal_test(np.zeros(10))
    assert z["p"] == 1.0 and z["dz"] == 0.0 and z["p_wilcoxon"] == 1.0
    from scipy.stats import ttest_1samp
    assert r["p"] == pytest.approx(ttest_1samp(d, 0).pvalue)
    assert r["t_stat"] == pytest.approx(r["dz"] * np.sqrt(400))           # t = dz * sqrt(n)


def test_within_group_diffs_returns_group_ids():
    d, gid = within_group_diffs([0, 0, 1, 1, 2, 2], [1, 0, 1, 1, 0, 1], [3.0, 1.0, 5.0, 5.0, 2.0, 6.0],
                                return_groups=True)
    assert gid.tolist() == [0, 2] and np.allclose(d, [2.0, 4.0])


def test_user_level_test_averages_per_user():
    from src.click_signal import user_level_test
    diffs = np.array([1.0, 3.0, 2.0, 4.0, 6.0])
    gid = np.array([0, 1, 2, 3, 4])
    users = np.array(["a", "a", "b", "c", "c"])            # per-user means 2, 2, 5
    r = user_level_test(diffs, gid, users)
    m = np.array([2.0, 2.0, 5.0])
    assert r["n_users"] == 3 and r["user_dz"] == pytest.approx(m.mean() / m.std(ddof=1))


def test_holm_matches_hand_computation():
    p = [0.01, 0.04, 0.03, 0.5]
    # sorted 0.01, 0.03, 0.04, 0.5 -> 0.04, 0.09, 0.09 (monotone), 0.5
    assert np.allclose(holm(p), [0.04, 0.09, 0.09, 0.5])
    assert np.allclose(holm([0.9, 0.8]), [1.0, 1.0])


def test_signal_table():
    rng = np.random.default_rng(1)
    n_imp = 300
    groups = np.repeat(np.arange(n_imp), 4)
    labels = np.tile([1, 0, 0, 0], n_imp)
    signal = labels + rng.normal(0, 1, len(labels))
    noise = rng.normal(0, 1, len(labels))
    users = np.arange(n_imp) % 50
    t = signal_table(groups, labels, {"signal": signal, "noise": noise}, group_users=users)
    assert t.loc[0, "n_users"] == 50 and t.loc[0, "significant_both"]
    assert {"p_wilcoxon_holm", "user_p_holm", "significant_user"} <= set(t.columns)
    assert t["feature"].tolist() == ["signal", "noise"]
    assert t.loc[0, "significant_holm"] and t.loc[0, "dz"] > 0.5
    assert t.loc[0, "p_holm"] == pytest.approx(min(1, 2 * t.loc[0, "p"]))      # smallest of 2 p-values x 2


def test_holm_family_recomputes_all_flags_across_rows():
    from src.click_signal import holm_family
    t = pd.DataFrame({"p": [0.02, 0.04], "p_wilcoxon": [0.1, 0.2], "user_p": [0.01, 0.9], "dz": [0.1, 0.1],
                      "user_dz": [0.1, 0.1]})
    out = holm_family(t)
    assert out["p_holm"].tolist() == pytest.approx([0.04, 0.04]) and out["p_wilcoxon_holm"].tolist() == pytest.approx([0.2, 0.2])
    assert out["significant_both"].tolist() == [True, False]


def test_rank_biserial_sign():
    assert signal_test(np.array([1.0, 2.0, 3.0]))["rank_biserial"] == pytest.approx(1.0)
    assert signal_test(np.array([-3.0, 1.0, 2.0]))["rank_biserial"] == pytest.approx(0.0)


def test_permute_within_groups_keeps_counts_per_group():
    from src.click_signal import permute_within_groups
    groups = np.array([2, 0, 0, 1, 2, 0, 1, 2, 2])
    labels = np.array([1, 0, 1, 0, 0, 0, 1, 0, 1])
    perm = permute_within_groups(groups, labels, seed=0)
    for g in np.unique(groups):
        assert perm[groups == g].sum() == labels[groups == g].sum()
    assert np.array_equal(perm, permute_within_groups(groups, labels, seed=0))
    many = [permute_within_groups(groups, labels, seed=s) for s in range(20)]
    assert any(not np.array_equal(m, labels) for m in many)


def test_column_values_builds_arrays_on_demand():
    feats = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    v = ColumnValues(feats, ["b"], np.array([2, 0]))
    assert list(v) == ["b"] and len(v) == 1 and v["b"].tolist() == [6.0, 4.0]
