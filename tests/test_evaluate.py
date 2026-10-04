import numpy as np
import pandas as pd
import pytest

from src.evaluate import attach_scores, evaluate, evaluate_model, load_truth


def _truth(fake_processed):
    return load_truth("fit", fake_processed)


def test_truth_is_sorted_and_stringly_typed(fake_processed):
    t = _truth(fake_processed)
    assert list(t["news_id"]) == ["N3", "N2", "N1"]
    assert t["imp_key"].map(type).eq(str).all() and t["news_id"].map(type).eq(str).all()


def test_perfect_scores(fake_processed):
    t = _truth(fake_processed)
    s = t[["imp_key", "news_id"]].assign(score=[0.9, 0.1, 0.2])
    per_imp, summary = evaluate(t, s)
    assert summary["auc"] == 1.0 and summary["hr@5"] == 1.0 and len(per_imp) == 1


def test_contract_violations(fake_processed):
    t = _truth(fake_processed)
    good = t[["imp_key", "news_id"]].assign(score=[0.9, 0.1, 0.2])
    with pytest.raises(ValueError, match="every candidate"):
        attach_scores(t, good.iloc[:2])
    with pytest.raises(ValueError, match="NaN"):
        attach_scores(t, good.assign(score=[np.nan, 0.1, 0.2]))
    with pytest.raises(ValueError, match="duplicate"):
        attach_scores(t, pd.concat([good.iloc[:2], good.iloc[:1]]))
    with pytest.raises(ValueError, match="missing columns"):
        attach_scores(t, good.drop(columns="score"))
    wrong_key = good.assign(news_id=["X1", "X2", "X3"])
    with pytest.raises(ValueError, match="no score"):
        attach_scores(t, wrong_key)


def test_row_order_of_score_file_does_not_matter(fake_processed):
    t = _truth(fake_processed)
    s = t[["imp_key", "news_id"]].assign(score=[0.3, 0.9, 0.1])
    _, a = evaluate(t, s)
    _, b = evaluate(t, s.sample(frac=1, random_state=0))
    assert a.equals(b)


def test_deterministic_ties(fake_processed):
    t = _truth(fake_processed)
    s = t[["imp_key", "news_id"]].assign(score=0.0)
    _, a = evaluate(t, s)
    _, b = evaluate(t, s)
    assert a.equals(b)


def test_evaluate_model_saves_and_replaces_row(fake_processed, tmp_path):
    t = _truth(fake_processed)
    s = t[["imp_key", "news_id"]].assign(score=[0.9, 0.1, 0.2])
    evaluate_model("demo", "fit", s, processed=fake_processed, results=tmp_path)
    evaluate_model("demo", "fit", s, processed=fake_processed, results=tmp_path)
    table = pd.read_csv(tmp_path / "tables" / "summary.csv")
    assert len(table) == 1 and table.loc[0, "model"] == "demo"
    assert (tmp_path / "scores" / "demo__fit.parquet").exists()
    assert (tmp_path / "per_impression" / "demo__fit.parquet").exists()
