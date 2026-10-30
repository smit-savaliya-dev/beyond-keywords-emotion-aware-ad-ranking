import numpy as np
import pandas as pd
import pytest

from src.emotion import GOEMOTIONS_LABELS
from src.emotion_v2 import (class_shares, combine_share, example_rows, intensity_table, load_part_context,
                            select_setting, threshold_labels)

IDX = {l: i for i, l in enumerate(GOEMOTIONS_LABELS)}


def raw(ids, rows):
    m = np.full((len(ids), 28), 0.01)
    for r, spec in enumerate(rows):
        for k, v in spec.items():
            m[r, IDX[k]] = v
    df = pd.DataFrame(m, columns=GOEMOTIONS_LABELS)
    df.insert(0, "news_id", ids)
    return df


def test_combine_share_title_only_half_and_no_abstract():
    t = raw(["A", "B"], [{"joy": 0.8}, {"joy": 0.4}])
    a = raw(["A"], [{"joy": 0.2}])
    half = combine_share(t, a, 0.5)
    assert half.loc[0, "joy"] == pytest.approx(0.5) and half.loc[1, "joy"] == pytest.approx(0.4)
    only = combine_share(t, a, 1.0)
    assert only.loc[0, "joy"] == pytest.approx(0.8) and only["has_abstract"].tolist() == [True, False]
    third = combine_share(t, a, 1 / 3)
    assert third.loc[0, "joy"] == pytest.approx(0.8 / 3 + 0.2 * 2 / 3)


def test_intensity_ignores_neutral_and_threshold_is_inclusive():
    r = raw(["A", "B", "C"], [{"neutral": 0.95, "grief": 0.05}, {"approval": 0.2}, {"anger": 0.049}])
    t = intensity_table(r)
    assert t["top_label"].tolist() == ["grief", "approval", "anger"]
    assert t["group"].tolist() == ["sadness", "joy", "anger"]
    assert threshold_labels(t, 0.05).tolist() == ["sadness", "joy", "neutral"]      # >= t, neutral score ignored
    s = class_shares(threshold_labels(t, 0.05))
    assert s["pct_neutral"] == pytest.approx(100 / 3) and sum(s.values()) == pytest.approx(100)


def grid(rows):
    return pd.DataFrame(rows, columns=["share", "t", "pct_neutral", "fit_dz", "valid_dz"])


def test_select_setting_prefers_low_neutral_tier_and_higher_t_on_ties():
    g = grid([(0.5, 0.03, 12, 0.25, 0.24), (0.5, 0.05, 25, 0.25, 0.24), (0.5, 0.1, 45, 0.25, 0.24),
              (0.7, 0.05, 29, 0.24, 0.23), (1.0, 0.1, 35, 0.30, 0.30)])
    row, tier, passing = select_setting(g, 0.237)
    assert (row["share"], row["t"], tier) == (0.5, 0.05, 30.0) and len(passing) == 3   # 1.0 is only in the 40% tier


def test_select_setting_valid_gate_and_fallback_tier():
    g = grid([(0.5, 0.1, 35, 0.3, 0.20), (0.6, 0.1, 38, 0.25, 0.22), (0.7, 0.1, 55, 0.3, 0.3)])
    row, tier, _ = select_setting(g, 0.237)
    assert (row["share"], tier) == (0.6, 40.0)                                         # 0.20 fails the 10% gate
    row, tier, passing = select_setting(grid([(0.5, 0.1, 55, 0.3, 0.3)]), 0.237)
    assert row is None and tier is None and passing.empty


def test_example_rows_exclude_hand_label_items():
    labels = pd.DataFrame({"news_id": [f"N{i}" for i in range(12)], "label": ["joy"] * 6 + ["neutral"] * 6,
                           "intensity": 0.2, "top_label": "joy", "neutral_score": 0.5})
    news = pd.DataFrame({"news_id": labels["news_id"], "title": "t", "abstract": "a"})
    ex = example_rows(labels, news, ["N0", "N1"], n=5, seed=0)
    assert not ex["news_id"].isin(["N0", "N1"]).any()
    assert ex["label"].value_counts().to_dict() == {"joy": 4, "neutral": 5}
    assert ex.equals(example_rows(labels, news, ["N0", "N1"], n=5, seed=0))


def test_load_part_context_uses_history_and_refuses_dev(fake_processed):
    index = pd.Index(pd.read_parquet(fake_processed / "news.parquet")["news_id"])
    ctx = load_part_context(fake_processed, "fit", index)
    assert ctx.n_impressions == 1 and len(ctx.item_idx) == 3 and ctx.labels.sum() == 1
    assert [index[i] for i in ctx.histories[0]] == ["N1", "N2"]
    with pytest.raises(ValueError, match="only load fit or valid"):
        load_part_context(fake_processed, "dev", index)
