import json

import numpy as np
import pandas as pd
import pytest

from src.emotion import GOEMOTIONS_LABELS, build_emotion_table, label_corpus, texts_fingerprint
from src.emotion_features import (GO_COLUMNS, GO_EMOTIONS, combine_fields, field_texts, go_features,
                                  neutral_summary, title_share, truncation_report)
from tests.test_emotion import CountingClf, fake_clf

QUIET = lambda *_: None


def _news():
    return pd.DataFrame({"news_id": ["N1", "N2", "N3"], "title": ["Big win", " Storm hits ", "Quiet day"],
                         "abstract": ["Fans cheer.", "", "  "]})


def _raw(ids, rows):
    df = pd.DataFrame(np.asarray(rows, dtype=float), columns=GOEMOTIONS_LABELS)
    df.insert(0, "news_id", ids)
    return df


def test_field_texts_drops_empty_abstracts_and_strips():
    news = _news()
    assert field_texts(news, "title")["title"].tolist() == ["Big win", "Storm hits", "Quiet day"]
    ab = field_texts(news, "abstract")
    assert ab["news_id"].tolist() == ["N1"] and ab["abstract"].tolist() == ["Fans cheer."]
    with pytest.raises(ValueError, match="empty title"):
        field_texts(news.assign(title=["a", "", "c"]), "title")


def test_label_corpus_text_col_and_default_unchanged(tmp_path):
    news = _news().assign(text=lambda d: d["title"])
    texts = field_texts(news, "title")
    raw = label_corpus(texts, CountingClf(), tmp_path / "t", {"model": "fake", "max_length": 128, "field": "title"},
                       2, log=QUIET, text_col="title")
    assert np.allclose(raw[GOEMOTIONS_LABELS].to_numpy(),
                       [[d["score"] for d in sorted(fake_clf([t])[0], key=lambda d: d["label"])]
                        for t in texts["title"]])
    assert texts_fingerprint(news) == texts_fingerprint(news, "text") != texts_fingerprint(news, "abstract")
    label_corpus(news, CountingClf(), tmp_path / "d", {"model": "fake", "max_length": 128}, 2, log=QUIET)
    saved = json.loads((tmp_path / "d" / "settings.json").read_text(encoding="utf-8"))
    assert set(saved) == {"model", "max_length", "chunk_size", "n_items", "texts_sha", "labels"}


def test_title_share():
    assert title_share(0.3, 0.6) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        title_share(0, 0.6)


def test_combine_fields_weights_and_title_fallback():
    t = _raw(["N1", "N2"], [np.full(28, 0.3), np.full(28, 0.9)])
    a = _raw(["N1"], [np.full(28, 0.6)])
    w = combine_fields(t, a, 0.3, 0.6)
    assert w["news_id"].tolist() == ["N1", "N2"] and w["has_abstract"].tolist() == [True, False]
    assert np.allclose(w.loc[0, GOEMOTIONS_LABELS].astype(float), 0.3 / 3 + 0.6 * 2 / 3)
    assert np.allclose(w.loc[1, GOEMOTIONS_LABELS].astype(float), 0.9)
    with pytest.raises(ValueError, match="no title scores"):
        combine_fields(t, _raw(["N9"], [np.zeros(28)]))
    with pytest.raises(ValueError, match="duplicate"):
        combine_fields(pd.concat([t, t]), a)


def test_go_features_noisy_or_mass_and_dist():
    row = np.zeros(28)
    idx = {l: i for i, l in enumerate(GOEMOTIONS_LABELS)}
    row[idx["joy"]], row[idx["amusement"]], row[idx["fear"]], row[idx["neutral"]] = 0.5, 0.5, 0.25, 0.4
    zero = np.zeros(28)
    tiny = zero.copy()
    tiny[idx["curiosity"]] = 1e-9
    go = go_features(_raw(["N1", "N2", "N3"], [row, zero, tiny]))
    assert list(go.columns) == ["news_id", *GO_COLUMNS]
    assert go.loc[0, "go_joy"] == pytest.approx(0.75)                  # 1 - 0.5 * 0.5
    assert go.loc[0, "go_emotion_mass"] == pytest.approx(1.0)
    assert go.loc[0, "go_dist_joy"] == pytest.approx(0.75) and go.loc[0, "go_dist_fear"] == pytest.approx(0.25)
    assert go.loc[0, "go_emotionality"] == pytest.approx(0.6)
    dist = go[[f"go_dist_{e}" for e in GO_EMOTIONS]]
    assert np.allclose(dist.sum(axis=1), 1.0)
    assert np.allclose(dist.loc[1], 1 / 8)                              # exact zero -> uniform
    assert go.loc[2, "go_dist_curiosity"] == pytest.approx(1.0)         # tiny but non-zero -> not uniform


def test_truncation_report():
    tok = lambda texts, truncation=False: {"input_ids": [[0] * len(t.split()) for t in texts]}
    rep = truncation_report(tok, ["a " * 100, "a " * 200, "a " * 300, "a"], limits=(128, 256))
    assert rep["n"] == 4 and rep["max_tokens"] == 300
    assert rep["pct_truncated_at_128"] == 50.0 and rep["pct_truncated_at_256"] == 25.0


def test_neutral_summary():
    rows = [np.eye(28)[GOEMOTIONS_LABELS.index(l)] for l in ("neutral", "neutral", "joy", "grief")]
    raw = _raw(["A", "B", "C", "D"], rows)
    s = neutral_summary(build_emotion_table(raw), raw)
    assert s["dominant_pct"]["neutral"] == 50.0 and s["dominant_pct"]["anger"] == 0.0
    assert s["neutral_share_6class"]["pct_gt_0.9"] == 50.0 and s["neutral_raw_sigmoid"]["pct_lt_0.5"] == 50.0
