import numpy as np
import pandas as pd
import pytest

from src.annotation import (agreement, build_sheet, kappa_band, make_sample, read_sheet, sheet_problems,
                            vader_correlation)


def _news(n=300):
    return pd.DataFrame({"news_id": [f"N{i}" for i in range(n)], "title": [f"T{i}" for i in range(n)],
                         "abstract": [""] * n})


def test_sample_is_deterministic_and_has_no_predictions():
    a, b = make_sample(_news(), 50), make_sample(_news(), 50)
    assert a.equals(b) and a["news_id"].is_unique and len(a) == 50
    sheet = build_sheet(a)
    assert list(sheet.columns) == ["sample_id", "news_id", "title", "abstract", "your_label"]
    assert (sheet["your_label"] == "").all()


def test_sheet_roundtrip_with_excel_encodings(tmp_path):
    sheet = build_sheet(make_sample(_news(), 5))
    sheet.loc[0, "title"] = "Caf\u00e9 \u2019quoted\u2019"
    sheet["your_label"] = [" Joy ", "sadness", "ANGER", "fear", "neutral"]
    p = tmp_path / "s.csv"
    sheet.to_csv(p, index=False, encoding="utf-8-sig")
    got = read_sheet(p)
    assert got.loc[0, "title"] == "Caf\u00e9 \u2019quoted\u2019" and list(got["your_label"])[:3] == ["joy", "sadness", "anger"]
    p2 = tmp_path / "ansi.csv"                                # Excel "CSV (Comma delimited)" on Windows
    sheet.assign(title="Caf\u00e9").to_csv(p2, index=False, encoding="cp1252")
    assert read_sheet(p2).loc[0, "title"] == "Caf\u00e9"


def test_sheet_problems():
    sheet = build_sheet(make_sample(_news(), 4))
    assert "empty" in sheet_problems(sheet)[0]
    sheet["your_label"] = ["joy", "happy", "fear", "neutral"]
    assert "happy" in sheet_problems(sheet)[0]
    sheet["your_label"] = ["joy", "sadness", "fear", "neutral"]
    assert sheet_problems(sheet) == []


def test_agreement_known_values():
    res = agreement(["joy", "joy", "sadness", "sadness"], ["joy", "sadness", "sadness", "sadness"], n_boot=200)
    assert res["kappa"] == pytest.approx(0.5) and res["accuracy"] == pytest.approx(0.75)
    perfect = agreement(["joy", "fear", "neutral"] * 20, ["joy", "fear", "neutral"] * 20, n_boot=200)
    assert perfect["kappa"] == pytest.approx(1.0)
    lo, hi = perfect["kappa_ci95"]
    assert lo <= 1.0 <= hi + 1e-9


def test_kappa_band():
    assert kappa_band(-0.1) == "poor" and kappa_band(0.5) == "moderate" and kappa_band(0.9) == "almost perfect"


def test_vader_correlation_handles_flat_input():
    assert vader_correlation(["abc"] * 5, [0.1] * 5) is None
