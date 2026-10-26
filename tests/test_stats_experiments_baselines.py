import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.baselines import (cosine_scores, fit_tfidf, item_ctr, popularity_scores, profile_weights, unique_profiles,
                           unseen_share)
from src.experiments import (append_run, check_final_dev, config_hash, dev_already_run, frozen_marker, load_frozen,
                             save_frozen)
from src.stats import compare_models, holm, paired_test


# ---------- stats ----------

def test_holm_moved_and_reexported():
    from src.click_signal import holm as holm2
    assert holm2 is holm and np.allclose(holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])


def test_paired_test_matches_scipy_and_ci():
    from scipy import stats as st
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 300)
    b = a + rng.normal(0.02, 0.05, 300)
    r = paired_test(a, b)
    d = b - a
    assert r["mean_diff"] == pytest.approx(d.mean()) and r["p"] == pytest.approx(st.ttest_rel(b, a).pvalue)
    half = st.t.ppf(0.975, 299) * d.std(ddof=1) / np.sqrt(300)
    assert r["ci_low"] == pytest.approx(d.mean() - half) and r["dz"] == pytest.approx(d.mean() / d.std(ddof=1))
    nan = paired_test([1.0, np.nan, 3.0], [2.0, 5.0, np.nan])
    assert nan["n"] == 1 and nan["p"] == 1.0


def test_compare_models_alignment_and_holm():
    per_a = pd.DataFrame({"imp_key": ["x", "y", "z"], "auc": [0.5, 0.6, 0.7], "mrr": [0.1, 0.2, 0.3]})
    per_b = pd.DataFrame({"imp_key": ["z", "x", "y"], "auc": [0.9, 0.6, 0.7], "mrr": [0.3, 0.1, 0.2]})
    out = compare_models(per_a, per_b, ["auc", "mrr"], "M0", "M1")
    assert out.loc[0, "mean_diff"] == pytest.approx(0.1333333, rel=1e-5)     # aligned by imp_key, not row order
    assert out.loc[1, "mean_diff"] == 0 and {"p_holm", "significant_holm"} <= set(out.columns)
    with pytest.raises(ValueError, match="different impressions"):
        compare_models(per_a, per_b.iloc[:2], ["auc"])


# ---------- experiments ----------

def test_run_log_appends_rows(tmp_path):
    log = tmp_path / "runs.csv"
    append_run("M0", "valid", "tune", {"a": 10}, {"auc": 0.6}, path=log)
    append_run("M0", "valid", "tune", {"a": 100}, {"auc": 0.61}, path=log)
    t = pd.read_csv(log)
    assert len(t) == 2 and t["config_hash"].nunique() == 2 and {"git_commit", "timestamp_utc"} <= set(t.columns)
    assert json.loads(t.loc[1, "config"]) == {"a": 100}


def test_final_dev_guard(tmp_path):
    frozen, decisions, results, log = tmp_path / "frozen", tmp_path / "decisions.md", tmp_path / "res", tmp_path / "l.csv"
    kw = dict(frozen_dir=frozen, decisions=decisions, results=results, runs_log=log)
    with pytest.raises(SystemExit, match="--tune first"):
        check_final_dev("M0", **kw)
    doc = save_frozen("M0", {"a": 10}, {"valid_auc": 0.6}, frozen)
    assert load_frozen("M0", frozen)["config_hash"] == config_hash({"a": 10})
    decisions.write_text("| D1 | ... |\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="record the frozen settings"):
        check_final_dev("M0", **kw)
    decisions.write_text(f"| D1 | settings {frozen_marker('M0', doc['config_hash'])} |\n", encoding="utf-8")
    assert check_final_dev("M0", **kw)["config"] == {"a": 10}                 # first dev run allowed
    append_run("M0", "dev", "final-dev", {"a": 10}, {"auc": 0.5}, path=log)
    assert dev_already_run("M0", results, log)
    with pytest.raises(SystemExit, match="already has a dev result"):
        check_final_dev("M0", **kw)
    check_final_dev("M0", rerun_reason="harness bug fixed", **kw)
    assert "harness bug fixed" in decisions.read_text(encoding="utf-8")


# ---------- M0 popularity ----------

def test_popularity_smoothing_and_unseen():
    stats = pd.DataFrame({"news_id": ["A", "A", "A", "A", "B"], "label": [1, 1, 0, 0, 0]})
    ctr, p = item_ctr(stats)
    assert p == pytest.approx(0.4) and ctr.loc["A", "clicks"] == 2 and ctr.loc["A", "shown"] == 4
    cand = pd.DataFrame({"news_id": ["A", "B", "C"]})
    s = popularity_scores(cand, ctr, p, a=10)
    assert s[0] == pytest.approx((2 + 4) / 14) and s[1] == pytest.approx(4 / 11) and s[2] == pytest.approx(0.4)
    assert unseen_share(cand, ctr) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        popularity_scores(cand, ctr, p, a=0)


# ---------- M1 TF-IDF ----------

def test_fit_tfidf_uses_fit_texts_only():
    vec, X = fit_tfidf(["stock market falls", "market rally stock"], ["stock market", "zebra unicorn", ""], False,
                       min_df=1, stop_words=None)
    assert "zebra" not in vec.vocabulary_ and X.shape[0] == 3
    assert X[1].nnz == 0 and X[2].nnz == 0 and np.isclose(sparse.linalg.norm(X[0]), 1.0)


def test_profile_weights_mean_decay_last_items():
    W = profile_weights([np.array([0, 1, 2]), np.array([], dtype=np.int64), np.array([-1, 2])], 3, "decay",
                        decay=0.5, max_items=2).toarray()
    assert np.allclose(W[0], [0, 0.5 / 1.5, 1 / 1.5]) and np.allclose(W[1], 0) and np.allclose(W[2], [0, 0, 1])
    M = profile_weights([np.array([0, 1, 2])], 3, "mean", max_items=50).toarray()
    assert np.allclose(M[0], 1 / 3)
    with pytest.raises(ValueError):
        profile_weights([np.array([0])], 1, "max")


def test_cosine_scores_chunked_and_cold_start():
    items = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    profiles = sparse.csr_matrix(np.array([[2.0, 0.0], [0.0, 0.0]]))
    s = cosine_scores(profiles, items, np.array([0, 0, 0, 1]), np.array([0, 1, 2, 0]), chunk_size=2)
    assert np.allclose(s, [1.0, 0.0, 1 / np.sqrt(2), 0.0])


def test_unique_profiles_dedupes_same_user_history():
    beh = pd.DataFrame({"imp_key": ["a", "b", "c"], "user_id": ["U1", "U1", "U2"],
                        "history": [np.array(["N1", "N2"]), np.array(["N1", "N2"]), np.array(["N1", "N2"])]})
    codes, hist = unique_profiles(beh)
    assert codes.tolist() == [0, 0, 1] and len(hist) == 2


def test_pick_best_ties_go_to_first():
    from src.experiments import pick_best
    assert pick_best([{"auc": 0.6}, {"auc": 0.7}, {"auc": 0.7}], "auc") == 1
