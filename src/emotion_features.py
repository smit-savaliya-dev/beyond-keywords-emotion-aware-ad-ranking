"""Weighted title/abstract GoEmotions layer: field texts, weighted raw scores, go_* and go_dist_* features."""
import numpy as np
import pandas as pd

from .config import ABSTRACT_WEIGHT, GO_FEATURES, TITLE_WEIGHT
from .emotion import GOEMOTIONS_LABELS, SIX_CLASSES

GO_EMOTIONS = list(GO_FEATURES)
GO_COLUMNS = ([f"go_{e}" for e in GO_EMOTIONS] + ["go_neutral", "go_emotionality", "go_emotion_mass"]
              + [f"go_dist_{e}" for e in GO_EMOTIONS])


def field_texts(news, field):
    """news_id + stripped, non-empty `field` texts. Empty titles are an error; empty abstracts are dropped."""
    text = news[field].fillna("").astype(str).str.strip()
    keep = (text != "").to_numpy()
    if field == "title" and not keep.all():
        raise ValueError(f"{int((~keep).sum())} items have an empty title; nothing to fall back to")
    out = pd.DataFrame({"news_id": news["news_id"].to_numpy(), field: text.to_numpy()})
    return out[keep].reset_index(drop=True)


def title_share(w_title=TITLE_WEIGHT, w_abstract=ABSTRACT_WEIGHT):
    if w_title <= 0 or w_abstract <= 0:
        raise ValueError("field weights must be positive")
    return w_title / (w_title + w_abstract)


def combine_fields(title_raw, abstract_raw, w_title=TITLE_WEIGHT, w_abstract=ABSTRACT_WEIGHT):
    """Weighted mean of raw 28 scores; items without an abstract keep their title scores. Title order is kept."""
    for name, df in (("title", title_raw), ("abstract", abstract_raw)):
        if df["news_id"].duplicated().any():
            raise ValueError(f"duplicate news ids in {name} scores")
        if df[GOEMOTIONS_LABELS].isna().any().any():
            raise ValueError(f"missing values in {name} scores")
    extra = set(abstract_raw["news_id"]) - set(title_raw["news_id"])
    if extra:
        raise ValueError(f"{len(extra)} abstract ids have no title scores")
    wt = title_share(w_title, w_abstract)
    t = title_raw.set_index("news_id")[GOEMOTIONS_LABELS].astype(np.float64)
    a = abstract_raw.set_index("news_id")[GOEMOTIONS_LABELS].astype(np.float64).reindex(t.index)
    has = a.notna().all(axis=1).to_numpy()
    out = t.copy()
    out.loc[has] = wt * t.loc[has] + (1 - wt) * a.loc[has]
    out = out.reset_index()
    out["has_abstract"] = has
    return out


def go_features(raw):
    """go_* (noisy-OR for multi-label features), go_neutral, go_emotionality, go_emotion_mass, go_dist_*."""
    p = raw[GOEMOTIONS_LABELS].astype(np.float64)
    out = pd.DataFrame({"news_id": raw["news_id"].to_numpy()})
    for name, labels in GO_FEATURES.items():
        vals = p[labels].to_numpy()
        out[f"go_{name}"] = vals[:, 0] if len(labels) == 1 else 1 - np.prod(1 - vals, axis=1)
    emo = out[[f"go_{e}" for e in GO_EMOTIONS]].to_numpy()
    mass = emo.sum(axis=1)
    out["go_neutral"] = p["neutral"].to_numpy()
    out["go_emotionality"] = 1 - out["go_neutral"]
    out["go_emotion_mass"] = mass
    dist = np.full_like(emo, 1.0 / len(GO_EMOTIONS))
    nz = mass > 0  # uniform only at an exact zero sum
    dist[nz] = emo[nz] / mass[nz, None]
    for j, name in enumerate(GO_EMOTIONS):
        out[f"go_dist_{name}"] = dist[:, j]
    return out


def truncation_report(tokenizer, texts, limits=(128, 256)):
    """Token counts (with special tokens) and % of texts longer than each limit."""
    lengths = np.array([len(ids) for ids in tokenizer(list(texts), truncation=False)["input_ids"]])
    rep = {"n": int(len(lengths)), "median_tokens": float(np.median(lengths)),
           "p95_tokens": float(np.percentile(lengths, 95)), "p99_tokens": float(np.percentile(lengths, 99)),
           "max_tokens": int(lengths.max())}
    for limit in limits:
        rep[f"pct_truncated_at_{limit}"] = round(100 * float((lengths > limit).mean()), 3)
    return rep


def neutral_summary(table, raw):
    """Dominant-class shares, the 6-class neutral share and the raw neutral sigmoid, summarised."""
    def stats(s):
        d = s.describe()
        return {**{k: round(float(v), 4) for k, v in d.items()},
                "pct_gt_0.9": round(100 * float((s > 0.9).mean()), 2),
                "pct_lt_0.5": round(100 * float((s < 0.5).mean()), 2)}
    dom = table["dominant_emotion"].value_counts(normalize=True).reindex(SIX_CLASSES, fill_value=0)
    return {"dominant_pct": {k: round(100 * float(v), 2) for k, v in dom.items()},
            "neutral_share_6class": stats(table["neutral"]), "neutral_raw_sigmoid": stats(raw["neutral"])}
