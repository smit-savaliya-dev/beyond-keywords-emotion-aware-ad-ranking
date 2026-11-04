import numpy as np
import pandas as pd
import pytest

import src.lightgbm_ranker as ltr
from src.config import M4_FEATURES, M4_NAME, M5_ALL_NAME, M5_NAME, M5_ROBUSTNESS_ITEM_FEATURES
from src.evaluate import evaluate
from src.lightgbm_ranker import (category_codes, group_sizes, impression_sizes, mean_impression_auc, model_features, past_popularity, sample_negatives,
                     shuffle_within_impressions, spec_features, train_ranker, user_category_affinity)


def day(n):
    return np.datetime64("2019-11-09") + np.timedelta64(n, "D")


def impressions():
    """Three days of rows: impression, news id, label, day."""
    rows = [("a", "n1", 1, 0), ("a", "n2", 0, 0), ("b", "n1", 0, 0), ("b", "n3", 1, 0),
            ("c", "n1", 1, 1), ("c", "n2", 0, 1), ("d", "n3", 1, 1), ("d", "n2", 0, 1),
            ("e", "n1", 1, 2), ("e", "n3", 0, 2), ("e", "n9", 0, 2)]
    t = pd.DataFrame(rows, columns=["imp", "news_id", "label", "d"])
    return t.assign(day=[day(d) for d in t["d"]])


def test_day_one_gets_the_fixed_prior_and_later_days_only_earlier_days():
    t = impressions()
    out = past_popularity(t, t, a=10, prior=0.04)
    assert (out[t["d"] == 0] == 0.04).all()
    # day 1 sees day 0 only: n1 had 1 click in 2 shows, global CTR 2/4
    assert out[(t["imp"] == "c") & (t["news_id"] == "n1")][0] == pytest.approx((1 + 10 * 0.5) / (2 + 10))
    # day 2 sees days 0 and 1: n1 had 2 clicks in 3 shows, global CTR 4/8; n9 was never shown, so it gets the global CTR
    assert out[(t["imp"] == "e") & (t["news_id"] == "n1")][0] == pytest.approx((2 + 10 * 0.5) / (3 + 10))
    assert out[t["news_id"] == "n9"][0] == pytest.approx(0.5)


@pytest.mark.parametrize("imp", ["a", "c", "e"])
def test_a_rows_own_impression_never_contributes_to_its_features(imp):
    t = impressions()
    base = past_popularity(t, t, a=10, prior=0.04)
    flipped = t.copy()
    rows = t["imp"] == imp
    flipped.loc[rows, "label"] = 1 - flipped.loc[rows, "label"]
    out = past_popularity(flipped, flipped, a=10, prior=0.04)
    d = t.loc[rows, "d"].iloc[0]
    upto = (t["d"] <= d).to_numpy()
    assert np.array_equal(base[upto], out[upto])                  # its own day and earlier days cannot see it
    if d < 2:
        assert not np.array_equal(base[~upto], out[~upto])        # later days do, so the check has power


def test_a_row_cannot_see_its_own_day_or_future_days():
    t = impressions()
    base = past_popularity(t, t, a=10, prior=0.04)
    extra = pd.DataFrame({"imp": "z", "news_id": "n1", "label": [1] * 50, "d": 1, "day": day(1)})   # 50 clicks on day 1
    stats = pd.concat([t, extra], ignore_index=True)
    out = past_popularity(t, stats, a=10, prior=0.04)
    assert np.array_equal(base[t["d"] <= 1], out[t["d"] <= 1]) and not np.array_equal(base[t["d"] == 2], out[t["d"] == 2])


def test_valid_uses_all_fit_days_like_m0():
    from src.baselines import item_ctr, popularity_scores
    fit = impressions()
    valid = pd.DataFrame({"news_id": ["n1", "n9"], "day": [day(5), day(5)]})
    ctr, p = item_ctr(fit)
    assert past_popularity(valid, fit, a=10) == pytest.approx(popularity_scores(valid, ctr, p, 10))


def test_feature_lists_never_contain_position_or_label():
    for model in (M4_NAME, M5_NAME, M5_ALL_NAME):
        f = model_features(model)
        assert not {"position", "label", "imp_key", "news_id", "day"} & set(f) and len(f) == len(set(f))


def test_m5_is_m4_plus_the_spec_and_m5_all_the_candidates():
    items, cong = spec_features()
    assert model_features(M5_NAME) == [*M4_FEATURES, *items, *cong] and len(cong) == 4
    assert model_features(M5_ALL_NAME) == [*M4_FEATURES, *M5_ROBUSTNESS_ITEM_FEATURES, *cong] and len(M5_ROBUSTNESS_ITEM_FEATURES) == 19
    assert set(items) <= set(M5_ROBUSTNESS_ITEM_FEATURES)


def test_affinity_counts_the_last_50_history_items_in_the_candidates_category():
    codes = np.array([0, 0, 1, 2])
    h = [np.array([0, 1, 2, 3]), np.empty(0, dtype=np.int64), np.array([-1, 2])]
    out = user_category_affinity([0, 0, 1, 2], [0, 3, 0, 2], h, codes, 3)
    assert out.tolist() == [0.5, 0.25, 0.0, 1.0]


def test_impression_size_and_category_codes():
    assert impression_sizes(["a", "a", "b"]).tolist() == [2, 2, 1]
    assert category_codes(np.array(["z", "a", "z"])).tolist() == [1, 0, 1]


def test_sampling_keeps_all_clicks_caps_negatives_and_is_seeded():
    keys = np.repeat(["a", "b", "c"], [30, 6, 12])
    labels = np.zeros(48, dtype=int)
    labels[[0, 1, 30, 36]] = 1
    keep = sample_negatives(keys, labels, 4, seed=1)
    assert keep[labels == 1].all()
    kept = pd.Series(labels[keep]).groupby(keys[keep]).agg(["sum", "size"])
    assert kept.loc["a", "size"] == 2 + 8 and kept.loc["b", "size"] == 1 + 4 and kept.loc["c", "size"] == 1 + 4
    assert np.array_equal(keep, sample_negatives(keys, labels, 4, seed=1)) and not np.array_equal(keep, sample_negatives(keys, labels, 4, seed=2))


def test_group_sizes_refuses_split_impressions():
    assert group_sizes(np.array(["a", "a", "b"])).tolist() == [2, 1]
    with pytest.raises(ValueError, match="consecutive"):
        group_sizes(np.array(["a", "b", "a"]))


def test_fast_auc_matches_the_harness_with_ties():
    rng = np.random.default_rng(0)
    keys = np.repeat([f"i{k}" for k in range(40)], 12)
    labels = (rng.random(len(keys)) < 0.3).astype(int)
    scores = rng.integers(0, 5, len(keys)).astype(float)
    truth = pd.DataFrame({"imp_key": keys, "news_id": [str(i) for i in range(len(keys))], "label": labels, "position": np.tile(np.arange(12), 40)})
    _, summary = evaluate(truth, truth[["imp_key", "news_id"]].assign(score=scores))
    assert mean_impression_auc(pd.factorize(keys)[0], labels, scores) == pytest.approx(summary["auc"])


def test_shuffle_stays_within_impressions_and_keeps_other_columns():
    t = pd.DataFrame({"imp_key": ["a"] * 50 + ["b"] * 50, "x": np.arange(100.0), "y": np.arange(100.0)})
    out = shuffle_within_impressions(t, ["x"], 1)
    assert sorted(out["x"][:50]) == list(range(50)) and not np.array_equal(out["x"], t["x"]) and np.array_equal(out["y"], t["y"])


def fake_frame(n_imps, seed):
    rng = np.random.default_rng(seed)
    keys = np.repeat([f"i{seed}_{k:03d}" for k in range(n_imps)], 10)
    signal = rng.random(len(keys))
    labels = (signal + rng.normal(0, 0.2, len(keys)) > 0.95).astype(np.int8)
    labels[::10] = 1
    return pd.DataFrame({"imp_key": keys, "label": labels, "good": signal.astype(np.float32), "noise": rng.random(len(keys)).astype(np.float32),
                         "category": rng.integers(0, 3, len(keys)).astype(np.int16)})


def test_train_ranker_learns_stops_early_and_repeats_exactly():
    pytest.importorskip("lightgbm")
    tr, va = fake_frame(200, 1), fake_frame(60, 2)
    params = {"num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 5}
    booster, best, rows, auc = train_ranker(tr, va, ["good", "noise", "category"], params, seed=42, max_rounds=60, patience=10)
    assert auc > 0.65 and 1 <= best <= 60 and rows < len(tr)
    scores = booster.predict(va[["good", "noise", "category"]], num_iteration=best)
    assert mean_impression_auc(pd.factorize(va["imp_key"])[0], va["label"], scores) == pytest.approx(auc)
    again = train_ranker(tr, va, ["good", "noise", "category"], params, seed=42, max_rounds=60, patience=10)
    assert again[1] == best and np.array_equal(again[0].predict(va[["good", "noise", "category"]], num_iteration=best), scores)


def test_run_id_changes_with_every_input():
    a = ltr.run_id(["x"], {"num_leaves": 31}, 42, "", "sha")
    assert a != ltr.run_id(["y"], {"num_leaves": 31}, 42, "", "sha") and a != ltr.run_id(["x"], {"num_leaves": 63}, 42, "", "sha")
    assert a != ltr.run_id(["x"], {"num_leaves": 31}, 43, "", "sha") and a != ltr.run_id(["x"], {"num_leaves": 31}, 42, "shuffled", "sha")


def test_fit_or_load_resumes_and_scores_the_saved_booster(tmp_path):
    pytest.importorskip("lightgbm")
    tr, va = fake_frame(120, 3), fake_frame(40, 4)
    f = ["good", "noise", "category"]
    p = {"num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 5}
    booster, info = ltr.fit_or_load(tr, va, f, p, 42, "", "sha", False, run_dir=tmp_path)
    assert info["best_iteration"] >= 1 and booster.num_trees() == info["best_iteration"]
    calls = []
    real = ltr.train_ranker
    ltr.train_ranker = lambda *a, **k: calls.append(1) or real(*a, **k)
    try:
        again, _ = ltr.fit_or_load(tr, va, f, p, 42, "", "sha", True, run_dir=tmp_path)
        assert not calls and np.array_equal(again.predict(va[f]), booster.predict(va[f]))
        ltr.fit_or_load(tr, va, f, p, 42, "", "sha", False, run_dir=tmp_path)
        assert calls == [1]                                      # without --resume a run always trains again
    finally:
        ltr.train_ranker = real


def test_load_model_refuses_a_changed_file(tmp_path):
    pytest.importorskip("lightgbm")
    (tmp_path / "M4_lightgbm.txt").write_text("not the frozen model")
    with pytest.raises(SystemExit, match="not the frozen model"):
        ltr.load_model("M4_lightgbm", "0" * 64, model_dir=tmp_path)


def test_grid_has_eight_configs():
    g = ltr.grid_configs()
    assert len(g) == 8 and g[0] == {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50}


def test_check_saved_scores_stops_on_a_gap(tmp_path):
    frame = pd.DataFrame({"imp_key": ["a", "a"], "news_id": ["x", "y"], "pop_ctr": [0.1, 0.2], "m1_score": [0.3, 0.4], "m2_score": [0.5, 0.6]})
    (tmp_path / "scores").mkdir()
    for model, col in (("M0_popularity", "pop_ctr"), ("M1_tfidf", "m1_score"), ("M2_sbert", "m2_score")):
        frame[["imp_key", "news_id", col]].rename(columns={col: "score"}).to_parquet(tmp_path / "scores" / f"{model}__valid.parquet")
    assert ltr.check_saved_scores(frame, "valid", results=tmp_path)["pop_ctr"] == 0
    with pytest.raises(SystemExit, match="D67b"):
        ltr.check_saved_scores(frame.assign(m2_score=[0.5, 0.7]), "valid", results=tmp_path)


def test_feature_families():
    assert [ltr.feature_family(f) for f in ("pop_ctr", "m2_score", "imp_size", "go_pct_fear", "cong_pct_cos")] == ["popularity", "content", "context", "emotion", "emotion"]


def test_importance_table_ranks_the_real_signal_first():
    pytest.importorskip("lightgbm")
    tr, va = fake_frame(200, 5), fake_frame(60, 6)
    f = ["good", "noise", "category"]
    booster, best, _, _ = train_ranker(tr, va, f, {"num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 5}, seed=42, max_rounds=30, patience=10)
    t = ltr.importance_table(booster, va, f, 20, 1)
    assert t.loc[0, "feature"] == "good" and t["contrib_rank"].tolist() == [1, 2, 3]
    assert t["gain_share"].sum() == pytest.approx(1) and t["contrib_share"].sum() == pytest.approx(1)
    s = ltr.family_shares(t.assign(family=["content", "emotion", "emotion"]))
    assert s["emotion"]["n_features"] == 2 and s["content"]["contrib_share"] + s["emotion"]["contrib_share"] == pytest.approx(1)


def fake_emotion_tables(path, ids):
    from src.config import GO_FEATURES
    from src.emotion import GOEMOTIONS_LABELS
    items, _ = spec_features()
    columns = list(dict.fromkeys([*items, *M5_ROBUSTNESS_ITEM_FEATURES, *[f"go_pct_{e}" for e in GO_FEATURES], "go_vad_valence"]))
    rng = np.random.default_rng(0)
    f = pd.DataFrame({c: rng.random(len(ids)) for c in columns}).assign(news_id=ids)
    f.to_parquet(path / "news_emotion_features_v2.parquet")
    pd.DataFrame({c: rng.random(len(ids)) for c in GOEMOTIONS_LABELS}).assign(news_id=ids).to_parquet(path / "news_emotion_raw28_v2.parquet")
    return f


def test_item_swap_keeps_identity_and_values_but_moves_them_to_other_items(tmp_path):
    ids = [f"n{i}" for i in range(6)]
    f = fake_emotion_tables(tmp_path, ids)
    groups, item_idx = np.array([0, 0, 0, 1, 1, 1]), np.array([0, 1, 2, 0, 3, 4])
    hist = [np.array([5]), np.array([1, 2])]
    plain = ltr.emotion_block(groups, item_idx, hist, ids, tmp_path)
    assert plain["go_vad_dominance"] == pytest.approx(f["go_vad_dominance"].to_numpy()[item_idx])
    perm = np.array([1, 2, 3, 4, 5, 0])
    swapped = ltr.emotion_block(groups, item_idx, hist, ids, tmp_path, perm)
    for c in model_features(M5_NAME)[len(M4_FEATURES):len(M4_FEATURES) + 13]:
        assert swapped[c] == pytest.approx(f[c].to_numpy()[perm][item_idx])          # item 0 carries item 1's vector, in every row
    assert swapped["go_vad_dominance"][0] == swapped["go_vad_dominance"][3]          # same item, same fingerprint: identity survives
    assert not np.allclose(plain["cong_pct_cos"], swapped["cong_pct_cos"])           # congruence uses the swapped vectors too


def test_impression_aucs_nan_for_one_class():
    out = ltr.impression_aucs(np.array([0, 0, 1, 1]), np.array([1, 0, 1, 1]), np.array([0.9, 0.1, 0.5, 0.2]))
    assert out[0] == 1.0 and np.isnan(out[1])


def test_fresh_split_scores_only_inside_each_subset():
    from src.results_tables import fresh_split
    truth = pd.DataFrame({"imp_key": ["a"] * 4 + ["b"] * 4, "news_id": ["old1", "old2", "new1", "new2"] * 2, "label": [1, 0, 1, 0] * 2})
    a = truth[["imp_key", "news_id"]].assign(score=[0.9, 0.1, 0.1, 0.9] * 2)       # right on old items, wrong on new ones
    b = truth[["imp_key", "news_id"]].assign(score=[0.9, 0.1, 0.9, 0.1] * 2)       # right on both
    t = fresh_split(truth, a, b, {"old1", "old2"}, "valid").set_index("subset")
    assert t.loc["unseen in fit", "mean_diff"] == 1.0 and t.loc["seen in fit", "mean_diff"] == 0.0
    assert t.loc["unseen in fit", "pct_candidates"] == 50 and t.loc["seen in fit", "impressions"] == 2

