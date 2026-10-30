import pytest

from src.agreement import agreement, compare_sources, kappa_band


def test_agreement_known_values():
    res = agreement(["joy", "joy", "sadness", "sadness"], ["joy", "sadness", "sadness", "sadness"], n_boot=200)
    assert res["kappa"] == pytest.approx(0.5) and res["accuracy"] == pytest.approx(0.75)
    perfect = agreement(["joy", "fear", "neutral"] * 20, ["joy", "fear", "neutral"] * 20, n_boot=200)
    assert perfect["kappa"] == pytest.approx(1.0)
    lo, hi = perfect["kappa_ci95"]
    assert lo <= 1.0 <= hi + 1e-9


def test_kappa_band():
    assert kappa_band(-0.1) == "poor" and kappa_band(0.5) == "moderate" and kappa_band(0.9) == "almost perfect"


def test_compare_sources_summary():
    y = ["joy", "neutral", "anger", "neutral", "fear", "neutral"]
    summary, results = compare_sources(y, {"perfect": y, "all_neutral": ["neutral"] * 6}, n_boot=50)
    assert summary["source"].tolist() == ["perfect", "all_neutral"]
    assert summary.loc[0, "kappa"] == 1.0 and summary.loc[0, "accuracy"] == 1.0
    assert summary.loc[1, "kappa"] == 0.0 and summary.loc[1, "pct_pred_neutral"] == 100.0
    assert summary.loc[1, "accuracy"] == 0.5 and "confusion" in results["perfect"]
