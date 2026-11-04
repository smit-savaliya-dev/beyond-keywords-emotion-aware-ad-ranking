"""Tests for src/where_it_helps.py (RQ5 breakdowns), with small fakes: no internet, GPU or real lexicons."""
import numpy as np
import pandas as pd

from src.where_it_helps import (breakdown, clicked_category, dominant_emotion, example_impressions, history_bin, top_categories)


def test_history_bins():
    assert history_bin(np.array([0, 1, 5, 6, 20, 21, 50])).tolist() == ["0", "1-5", "1-5", "6-20", "6-20", "21+", "21+"]


def test_dominant_emotion_decay_neutral_and_cold():
    labels = np.array(["joy", "anger", "neutral"])
    h = [np.array([0, 1]), np.array([2]), np.empty(0, dtype=np.int64), np.array([1, 0, 0])]
    out, used = dominant_emotion(h, labels)
    assert out.tolist() == ["anger", "neutral", "cold start", "joy"]   # newest weighs most in the first row
    assert used.tolist() == [2, 1, 0, 3]


def test_top_categories_and_clicked_category():
    category_of = pd.Series({"a": "news", "b": "sports", "c": "food"})
    assert top_categories(np.array(["a", "a", "b", "c", "b", "a"]), category_of, 2) == ["news", "sports"]
    truth = pd.DataFrame({"imp_key": ["i", "i", "j", "j"], "news_id": ["b", "a", "c", "a"], "label": [1, 1, 0, 1], "position": [0, 1, 0, 1]})
    truth2 = truth.assign(news_id=["a", "b", "a", "c"], label=[0, 1, 0, 1])
    assert clicked_category(truth, category_of, ["sports"]).to_dict() == {"i": "sports", "j": "other"}
    assert clicked_category(truth2, category_of, ["food"]).to_dict() == {"i": "other", "j": "food"}


def test_breakdown_keeps_order_and_reports_ci():
    t = breakdown([0.1, 0.3, -0.1, -0.1], ["x", "x", "y", "y"], "b", ["y", "x", "z"])
    assert t["group"].tolist() == ["y", "x"]
    assert t.loc[1, "mean_diff"] == 0.2 and t.loc[1, "ci_low"] < 0.2 < t.loc[1, "ci_high"] and t.loc[0, "pct_m3_better"] == 0


def test_examples_skip_short_lists_and_break_ties_by_key():
    diff = pd.Series({"a": 0.5, "b": 0.5, "c": 0.9, "d": -0.2, "e": 0.0})
    sizes = pd.Series({"a": 10, "b": 10, "c": 2, "d": 20, "e": 12})
    helped, hurt = example_impressions(diff, sizes, n=2, min_candidates=10)
    assert helped == ["b", "a"] and hurt == ["d", "e"]
