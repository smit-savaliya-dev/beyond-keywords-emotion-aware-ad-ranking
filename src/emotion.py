"""GoEmotions (28 labels) -> 6 emotion classes, plus resumable corpus labelling."""
import hashlib
import json

import numpy as np
import pandas as pd

from .config import EMOTION_MAX_LENGTH, EMOTION_MODEL

EMOTION_GROUPS = {
    "joy": ["joy", "amusement", "excitement", "gratitude", "optimism", "love",
            "pride", "relief", "admiration", "approval", "caring", "desire"],
    "sadness": ["sadness", "grief", "disappointment", "remorse"],
    "anger": ["anger", "annoyance", "disapproval", "disgust"],
    "fear": ["fear", "nervousness", "embarrassment"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}
SIX_CLASSES = list(EMOTION_GROUPS)
LABEL_TO_GROUP = {label: g for g, labels in EMOTION_GROUPS.items() for label in labels}
GOEMOTIONS_LABELS = sorted(LABEL_TO_GROUP)  # 28, alphabetical: column order of every raw-score table
_IDX = {label: i for i, label in enumerate(GOEMOTIONS_LABELS)}

MAPPING = np.zeros((len(GOEMOTIONS_LABELS), len(SIX_CLASSES)))
for _j, _labels in enumerate(EMOTION_GROUPS.values()):
    for _label in _labels:
        MAPPING[_IDX[_label], _j] = 1.0


def to_matrix(items):
    """List of {label: score} dicts -> (n, 28) float32 in GOEMOTIONS_LABELS order."""
    out = np.empty((len(items), len(GOEMOTIONS_LABELS)), dtype=np.float32)
    for i, scores in enumerate(items):
        missing = LABEL_TO_GROUP.keys() - scores.keys()
        if missing:
            raise ValueError(f"item {i}: classifier output lacks labels {sorted(missing)}")
        out[i] = [scores[label] for label in GOEMOTIONS_LABELS]
    return out


def collapse(raw):
    """(n, 28) raw scores -> (n, 6) vectors that sum to 1."""
    sums = np.asarray(raw, dtype=np.float64) @ MAPPING
    total = sums.sum(axis=1, keepdims=True)
    flat = np.full_like(sums, 1.0 / len(SIX_CLASSES))
    return np.where(total > 0, sums / np.where(total > 0, total, 1.0), flat)


def corpus_mean(vectors):
    return np.asarray(vectors).mean(axis=0)


def pick_device(arg="auto"):
    if str(arg) != "auto":
        return int(arg)
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


def load_classifier(model_name=EMOTION_MODEL, device=-1):
    from transformers import pipeline
    return pipeline("text-classification", model=model_name, top_k=None, device=device)


def score_texts(clf, texts, batch_size=64, max_length=EMOTION_MAX_LENGTH):
    """Raw (n, 28) scores in input order. Texts are length-sorted internally for speed."""
    texts = [t if t.strip() else "." for t in texts]
    order = np.argsort([len(t) for t in texts], kind="stable")
    raw = clf([texts[i] for i in order], batch_size=batch_size, truncation=True, max_length=max_length)
    if raw and isinstance(raw[0], dict):
        raise ValueError("classifier returned only the top label; load it with top_k=None")
    mat = to_matrix([{d["label"]: d["score"] for d in item} for item in raw])
    if mat.min() < 0 or mat.max() > 1 + 1e-6:
        raise ValueError(f"scores outside [0, 1]: {mat.min():.3f}..{mat.max():.3f}")
    out = np.empty_like(mat)
    out[order] = mat
    return out


def texts_fingerprint(news):
    h = hashlib.sha256()
    for nid, text in zip(news["news_id"], news["text"]):
        h.update(nid.encode("utf-8"))
        h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def label_corpus(news, clf, chunk_dir, settings, chunk_size=2000, batch_size=64, log=print):
    """Raw 28-score table for every item. Resumable; refuses to mix chunks made with different settings."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    settings = {**settings, "chunk_size": chunk_size, "n_items": len(news),
                "texts_sha": texts_fingerprint(news), "labels": GOEMOTIONS_LABELS}
    meta = chunk_dir / "settings.json"
    if meta.exists():
        if json.loads(meta.read_text(encoding="utf-8")) != settings:
            raise ValueError(f"settings differ from the saved chunks in {chunk_dir}; "
                             "delete that folder to relabel from scratch")
    else:
        meta.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    n_chunks = -(-len(news) // chunk_size)
    frames = []
    for i in range(n_chunks):
        path = chunk_dir / f"chunk_{i:04d}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
            log(f"[skip] chunk {i + 1}/{n_chunks}")
            continue
        part = news.iloc[i * chunk_size:(i + 1) * chunk_size]
        mat = score_texts(clf, part["text"].tolist(), batch_size, settings["max_length"])
        df = pd.DataFrame(mat, columns=GOEMOTIONS_LABELS)
        df.insert(0, "news_id", part["news_id"].to_numpy())
        tmp = path.with_name(path.name + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)  # atomic: an interrupted run never leaves a half-written chunk
        frames.append(df)
        log(f"[ok]   chunk {i + 1}/{n_chunks}")
    raw = pd.concat(frames, ignore_index=True)
    if raw["news_id"].tolist() != news["news_id"].tolist():
        raise ValueError("labelled news ids do not match news.parquet")
    return raw


def build_emotion_table(raw):
    out = pd.DataFrame(collapse(raw[GOEMOTIONS_LABELS].to_numpy()), columns=SIX_CLASSES)
    out.insert(0, "news_id", raw["news_id"].to_numpy())
    out["dominant_emotion"] = out[SIX_CLASSES].idxmax(axis=1)
    out["dominant_score"] = out[SIX_CLASSES].max(axis=1)
    return out


def single_label_gold(label_lists, names):
    """GoEmotions rows with exactly one gold label -> (row indices, 6-class gold)."""
    unknown = set(names) - LABEL_TO_GROUP.keys()
    if unknown:
        raise ValueError(f"unexpected GoEmotions label names: {sorted(unknown)}")
    idx = [i for i, labels in enumerate(label_lists) if len(labels) == 1]
    return idx, [LABEL_TO_GROUP[names[label_lists[i][0]]] for i in idx]
