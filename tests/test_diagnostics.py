import numpy as np
import pandas as pd
import pytest

from src.diagnostics import correlation_block, feature_distributions, pair_correlations, plot_correlation_block


def _df():
    return pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [0.0, 2.0, 4.0, 7.0], "c": [3.0, 2.0, np.nan, 0.0],
                         "hit": [1, 1, 1, 0]})


def test_feature_distributions():
    d = feature_distributions(_df(), ["a", "c"])
    assert d.loc["a", "pct_zero"] == 25.0 and d.loc["c", "pct_nan"] == 25.0 and d.loc["a", "pct_gt_0.5"] == 75.0
    assert d.loc["a", "mean"] == pytest.approx(1.5)


def test_pair_correlations_mask_and_nan():
    out = pair_correlations(_df(), [("a", "b", None), ("a", "b", "hit"), ("a", "c", None)])
    assert out["n"].tolist() == [4, 3, 3] and out.loc[1, "rows"] == "hit"
    assert out.loc[0, "spearman"] == pytest.approx(1.0) and out.loc[1, "pearson"] == pytest.approx(1.0)
    assert out.loc[2, "spearman"] == pytest.approx(-1.0)


def test_correlation_block_and_plot(tmp_path):
    block = correlation_block(_df(), ["a"], ["b", "c"])
    assert block.shape == (1, 2) and block.loc["a", "b"] == pytest.approx(1.0)
    path = tmp_path / "f.png"
    plot_correlation_block(block, path, "t", "s", highlight=[("a", "b"), ("zz", "b")])
    assert path.stat().st_size > 0
