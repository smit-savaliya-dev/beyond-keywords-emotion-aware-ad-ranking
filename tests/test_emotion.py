import numpy as np
import pandas as pd
import pytest

from src.emotion import (EMOTION_GROUPS, GOEMOTIONS_LABELS, MAPPING, SIX_CLASSES, build_emotion_table,
                         collapse, label_corpus, score_texts, single_label_gold, to_matrix)

CANONICAL_28 = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]


def fake_clf(texts, batch_size=64, truncation=True, max_length=128):
    """Deterministic fake: each text's scores depend only on that text."""
    out = []
    for t in texts:
        base = (sum(map(ord, t)) % 97) / 100
        out.append([{"label": lab, "score": min(1.0, base + i / 100)} for i, lab in enumerate(CANONICAL_28)])
    return out


class CountingClf:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts, **kw):
        self.calls += 1
        return fake_clf(texts)


def test_mapping_covers_all_28_labels_once():
    assert GOEMOTIONS_LABELS == sorted(CANONICAL_28)
    flat = [l for ls in EMOTION_GROUPS.values() for l in ls]
    assert len(flat) == len(set(flat)) == 28
    assert SIX_CLASSES == ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
    assert (MAPPING.sum(axis=1) == 1).all()                  # every label feeds exactly one class


def test_collapse_one_hot_and_uniform():
    raw = np.zeros((2, 28))
    raw[0, GOEMOTIONS_LABELS.index("grief")] = 1.0
    raw[1] = 1.0
    v = collapse(raw)
    assert v[0, SIX_CLASSES.index("sadness")] == pytest.approx(1.0)
    for g, labels in EMOTION_GROUPS.items():
        assert v[1, SIX_CLASSES.index(g)] == pytest.approx(len(labels) / 28)
    assert np.allclose(v.sum(axis=1), 1.0)


def test_collapse_all_zero_is_flat():
    assert np.allclose(collapse(np.zeros((1, 28))), 1 / 6)


def test_to_matrix_errors_on_missing_label():
    items = [{lab: 0.1 for lab in CANONICAL_28 if lab != "joy"}]
    with pytest.raises(ValueError, match="lacks labels"):
        to_matrix(items)


def test_score_texts_keeps_input_order_despite_length_sorting():
    texts = ["a much longer headline than the rest", "short", "medium length one", "x", ""]
    got = score_texts(fake_clf, texts)
    expected = to_matrix([{d["label"]: d["score"] for d in fake_clf([t if t.strip() else "."])[0]} for t in texts])
    assert np.allclose(got, expected)


def test_score_texts_rejects_top1_output_and_bad_range():
    with pytest.raises(ValueError, match="top_k"):
        score_texts(lambda texts, **kw: [{"label": "joy", "score": 0.9} for _ in texts], ["a"])
    bad = lambda texts, **kw: [[{"label": l, "score": 5.0} for l in CANONICAL_28] for _ in texts]
    with pytest.raises(ValueError, match="outside"):
        score_texts(bad, ["a"])


def _news(n=25):
    return pd.DataFrame({"news_id": [f"N{i}" for i in range(n)], "text": [f"headline {i} " * (i % 5 + 1) for i in range(n)]})


def test_label_corpus_resume_and_alignment(tmp_path):
    news = _news()
    clf = CountingClf()
    settings = {"model": "fake", "max_length": 128}
    raw1 = label_corpus(news, clf, tmp_path, settings, chunk_size=10, log=lambda *_: None)
    assert clf.calls == 3 and len(raw1) == 25
    raw2 = label_corpus(news, clf, tmp_path, settings, chunk_size=10, log=lambda *_: None)
    assert clf.calls == 3                                     # everything resumed from disk
    assert raw1.equals(raw2)
    direct = score_texts(fake_clf, news["text"].tolist())
    assert np.allclose(raw1[GOEMOTIONS_LABELS].to_numpy(), direct)
    assert not list(tmp_path.glob("*.tmp"))


def test_label_corpus_refuses_changed_settings_or_texts(tmp_path):
    news = _news()
    label_corpus(news, CountingClf(), tmp_path, {"model": "fake", "max_length": 128}, 10, log=lambda *_: None)
    with pytest.raises(ValueError, match="settings differ"):
        label_corpus(news, CountingClf(), tmp_path, {"model": "fake", "max_length": 64}, 10, log=lambda *_: None)
    changed = news.assign(text=news["text"] + " edited")
    with pytest.raises(ValueError, match="settings differ"):
        label_corpus(changed, CountingClf(), tmp_path, {"model": "fake", "max_length": 128}, 10, log=lambda *_: None)
    with pytest.raises(ValueError, match="settings differ"):
        label_corpus(news, CountingClf(), tmp_path, {"model": "fake", "max_length": 128}, 7, log=lambda *_: None)


def test_emotion_table(tmp_path):
    raw = label_corpus(_news(), CountingClf(), tmp_path, {"model": "fake", "max_length": 128}, 10, log=lambda *_: None)
    table = build_emotion_table(raw)
    assert list(table.columns[:7]) == ["news_id", *SIX_CLASSES]
    assert np.allclose(table[SIX_CLASSES].sum(axis=1), 1.0)
    assert (table["dominant_emotion"] == table[SIX_CLASSES].idxmax(axis=1)).all()


def test_single_label_gold():
    names = ["joy", "grief", "neutral"]
    idx, gold = single_label_gold([[0], [1, 2], [2], []], names)
    assert idx == [0, 2] and gold == ["joy", "neutral"]
    with pytest.raises(ValueError, match="unexpected"):
        single_label_gold([[0]], ["not_a_label"])
