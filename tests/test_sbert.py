import json

import numpy as np
import pandas as pd
import pytest

from src.sbert import embedding_paths, encode_batches, encode_corpus, load_embeddings, mean_pool, profile_cosine


def fake_encode(calls=None):
    def encode(texts):
        if calls is not None:
            calls.append(list(texts))
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)
    return encode


def news(n=5):
    return pd.DataFrame({"news_id": [f"N{i}" for i in range(n)], "title": ["a" * (n - i) for i in range(n)]})


def test_mean_pool_ignores_padding_and_normalises():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[3.0, 4.0], [100.0, 100.0]], [[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[1, 0], [1, 1]])
    out = mean_pool(hidden, mask).numpy()
    assert out[0] == pytest.approx([0.6, 0.8])
    assert out[1] == pytest.approx([2 ** -0.5, 2 ** -0.5])


def test_encode_batches_keeps_input_order():
    texts = ["ccc", "a", "dddd", "bb"]
    calls = []
    out = encode_batches(texts, fake_encode(calls), 2)
    assert out[:, 0].tolist() == [3, 1, 4, 2]
    assert calls == [["a", "bb"], ["ccc", "dddd"]]


def test_encode_corpus_resumes_and_guards_settings(tmp_path):
    d, calls = tmp_path / "chunks", []
    first = encode_corpus(news(), fake_encode(calls), d, {"model": "m"}, "title", chunk_size=2, batch_size=2, log=lambda *_: None)
    assert first[:, 0].tolist() == [5, 4, 3, 2, 1]
    assert json.loads((d / "settings.json").read_text())["n_items"] == 5
    (d / "chunk_0001.npy").unlink()
    calls.clear()
    again = encode_corpus(news(), fake_encode(calls), d, {"model": "m"}, "title", chunk_size=2, batch_size=2, log=lambda *_: None)
    assert np.array_equal(first, again) and sum(map(len, calls)) == 2  # only the missing chunk
    with pytest.raises(ValueError, match="settings differ"):
        encode_corpus(news(), fake_encode(), d, {"model": "other"}, "title", chunk_size=2, log=lambda *_: None)
    assert not list(d.glob("*.tmp"))


def test_encode_corpus_refuses_empty_text(tmp_path):
    t = news().assign(title=["a", "", "b", "c", "d"])
    with pytest.raises(ValueError, match="empty"):
        encode_corpus(t, fake_encode(), tmp_path / "c", {}, "title", log=lambda *_: None)


def test_load_embeddings_refuses_other_items(tmp_path):
    mat, ids = embedding_paths("title", tmp_path)
    np.save(mat, np.eye(2, dtype=np.float32))
    pd.DataFrame({"news_id": ["N0", "N1"]}).to_parquet(ids)
    assert load_embeddings("title", ["N0", "N1"], tmp_path).shape == (2, 2)
    with pytest.raises(ValueError, match="does not match"):
        load_embeddings("title", ["N1", "N0"], tmp_path)


def test_profile_cosine_chunked_and_cold_start():
    profiles = np.array([[1.0, 0.0], [0.0, 0.0]])
    items = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = profile_cosine(profiles, items, np.array([0, 0, 1]), np.array([0, 1, 0]), chunk_size=2)
    assert out.tolist() == [1.0, 0.0, 0.0]
