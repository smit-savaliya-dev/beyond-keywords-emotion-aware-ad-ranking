import pandas as pd
import pytest

NEWS_TRAIN = [
    "N1\tnews\tnewsus\tMan says \"hello\" to crowd\tAn abstract with caf\u00e9.\thttp://a\t[]\t[]",
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
BEH_DEV = ["1\tU1\t11/15/2019 8:00:00 AM\tN1\tN4-1 N3-0"]


@pytest.fixture
def fake_raw(tmp_path):
    for split, news, beh in [("train", NEWS_TRAIN, BEH_TRAIN), ("dev", NEWS_DEV, BEH_DEV)]:
        d = tmp_path / split
        d.mkdir()
        (d / "news.tsv").write_text("\n".join(news) + "\n", encoding="utf-8")
        (d / "behaviors.tsv").write_text("\n".join(beh) + "\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def fake_processed(tmp_path_factory, fake_raw):
    from src.data import build_all, save_all
    out = tmp_path_factory.mktemp("processed")
    news, beh, imps, summary = build_all(fake_raw)
    save_all(news, beh, imps, summary, out)
    return out
