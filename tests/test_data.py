import pytest

from src.data import build_all, check_official_counts, integrity_problems


def test_parsing(fake_raw):
    news, beh, imps, summary = build_all(fake_raw)
    n = news.set_index("news_id")
    assert len(news) == 4                                    # N3 deduplicated
    assert n.loc["N1", "title"] == 'Man says "hello" to crowd'
    assert "caf\u00e9" in n.loc["N1", "abstract"]            # unicode survives
    assert n.loc["N4", "title"] == "NA"                      # not turned into NaN
    assert n.loc["N2", "text"] == "Team wins final"          # empty abstract
    assert beh["imp_key"].is_unique
    assert beh.set_index("imp_key").loc["train-2", "history"] == []
    assert dict(zip(beh["imp_key"], beh["part"])) == {"train-1": "fit", "train-2": "valid", "dev-1": "dev"}
    first = imps[imps["imp_key"] == "train-1"].sort_values("position")
    assert list(first["news_id"].astype(str)) == ["N3", "N2", "N1"]
    assert list(first["label"]) == [1, 0, 0]
    assert len(imps) == 7


def test_summary_and_checks(fake_raw):
    _, _, _, summary = build_all(fake_raw)
    assert summary["missing_candidate_ids"] == 0 and summary["missing_history_ids"] == 0
    assert summary["parts"]["valid"]["cold_start_pct"] == 100.0
    assert len(summary["raw_sha256"]) == 4
    assert integrity_problems(summary) == []
    assert len(check_official_counts(summary)) == 3          # fake data is not official MIND-small


def test_integrity_problems_detected():
    summary = {"missing_candidate_ids": 2, "missing_history_ids": 0,
               "parts": {"dev": {"impressions_without_click": 1, "impressions_without_nonclick": 0}}}
    assert len(integrity_problems(summary)) == 2


def test_missing_split_folder_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="01_download_mind"):
        build_all(tmp_path)
