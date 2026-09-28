"""Parsing checks on a tiny fake MIND folder (same format as the real files)."""
import numpy as np
import pandas as pd
import pytest

from src.data import build_all
from src.evaluate import attach_scores, evaluate

NEWS_TRAIN = [
    "N1\tnews\tnewsus\tMan says \"hello\" to crowd\tAn abstract.\thttp://a\t[]\t[]",
    "N2\tsports\tfootball\tTeam wins final\t\thttp://b\t[]\t[]",
    "N3\tfinance\tmarkets\tStocks fall\tMarkets drop sharply.\thttp://c\t[]\t[]",
]
NEWS_DEV = [
    "N3\tfinance\tmarkets\tStocks fall\tMarkets drop sharply.\thttp://c\t[]\t[]",
    "N4\tlifestyle\tfood\tNA\tA title that says NA.\thttp://d\t[]\t[]",
]
BEH_TRAIN = [
    "1\tU1\t11/13/2019 9:00:00 AM\tN1 N2\tN3-1 N2-0 N1-0",
    "2\tU2\t11/14/2019 3:30:00 PM\t\tN1-0 N3-1",
]
BEH_DEV = [
    "1\tU1\t11/15/2019 8:00:00 AM\tN1\tN4-1 N3-0",
]


@pytest.fixture
def fake_raw(tmp_path):
    for split, news, beh in [("train", NEWS_TRAIN, BEH_TRAIN), ("dev", NEWS_DEV, BEH_DEV)]:
        d = tmp_path / split
        d.mkdir()
        (d / "news.tsv").write_text("\n".join(news) + "\n")
        (d / "behaviors.tsv").write_text("\n".join(beh) + "\n")
    return tmp_path


def test_parsing(fake_raw):
    news, beh, imps, summary = build_all(fake_raw)
    assert len(news) == 4                                   # N3 deduplicated
    assert news.set_index("news_id").loc["N1", "title"] == 'Man says "hello" to crowd'
    assert news.set_index("news_id").loc["N4", "title"] == "NA"   # not turned into NaN
    assert news.set_index("news_id").loc["N2", "text"] == "Team wins final"
    assert beh["imp_key"].is_unique                          # train-1 and dev-1 differ
    assert beh.set_index("imp_key").loc["train-2", "history"] == []
    assert dict(zip(beh["imp_key"], beh["part"])) == {
        "train-1": "fit", "train-2": "valid", "dev-1": "dev"}
    assert len(imps) == 7
    first = imps[imps["imp_key"] == "train-1"].sort_values("position")
    assert list(first["news_id"].astype(str)) == ["N3", "N2", "N1"]
    assert list(first["label"]) == [1, 0, 0]
    assert summary["missing_candidate_ids"] == 0
    assert summary["missing_history_ids"] == 0
    assert summary["parts"]["valid"]["cold_start_pct"] == 100.0


def test_harness_contract(fake_raw):
    _, _, imps, _ = build_all(fake_raw)
    truth = imps[imps["part"] == "fit"][["imp_key", "news_id", "label", "position"]].copy()
    for c in ("imp_key", "news_id"):
        truth[c] = truth[c].astype(str)
    good = truth[["imp_key", "news_id"]].assign(score=[0.9, 0.1, 0.2])
    per_imp, summary = evaluate(truth, good)
    assert summary["auc"] == 1.0
    with pytest.raises(ValueError):                         # a row is missing
        attach_scores(truth, good.iloc[:2])
    with pytest.raises(ValueError):                         # NaN score
        attach_scores(truth, good.assign(score=[np.nan, 0.1, 0.2]))
