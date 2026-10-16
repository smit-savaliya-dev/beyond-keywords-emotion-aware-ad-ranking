"""Weighted title/abstract GoEmotions layer: field texts, weighted raw scores, go_* and go_dist_* features."""
import numpy as np
import pandas as pd

from .config import ABSTRACT_WEIGHT, GO_FEATURES, GO_VAD_OVERRIDES, TITLE_WEIGHT, VAD_DIMS
from .emotion import GOEMOTIONS_LABELS, SIX_CLASSES
from .lexicon import resolve_terms

GO_EMOTIONS = list(GO_FEATURES)
GO_VAD_LABELS = [label for label in GOEMOTIONS_LABELS if label != "neutral"]
GO_VAD_COLUMNS = [f"go_vad_{d}" for d in VAD_DIMS]
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


def go_vad(raw, vad, overrides=GO_VAD_OVERRIDES):
    """Probability-weighted mean VAD of the 27 non-neutral label words (0 if all 27 scores are 0).
    -> (DataFrame news_id + go_vad_*, labels that used an override)."""
    unigrams = vad.index[~vad.index.str.contains(" ")]
    mapping, used = resolve_terms(GO_VAD_LABELS, unigrams, overrides)
    label_vad = vad.loc[[mapping[l] for l in GO_VAD_LABELS], VAD_DIMS].to_numpy()
    p = raw[GO_VAD_LABELS].to_numpy(dtype=np.float64)
    total = p.sum(axis=1, keepdims=True)
    vals = np.where(total > 0, (p @ label_vad) / np.where(total > 0, total, 1.0), 0.0)
    out = pd.DataFrame(vals, columns=GO_VAD_COLUMNS)
    out.insert(0, "news_id", raw["news_id"].to_numpy())
    return out, used


def prefixed(go, prefix):
    """go_* columns -> go_<prefix>_* (e.g. go_joy -> go_title_joy)."""
    return go.rename(columns={c: f"go_{prefix}_{c[3:]}" for c in go.columns if c.startswith("go_")})


def go_percentiles(go, prefix="go"):
    """Percentile rank (0, 1] of each <prefix>_<emotion> across items; NaN stays NaN. Item-level only."""
    out = pd.DataFrame({"news_id": go["news_id"].to_numpy()})
    for e in GO_EMOTIONS:
        out[f"{prefix}_pct_{e}"] = go[f"{prefix}_{e}"].rank(pct=True, method="average").to_numpy()
    return out


def variant_features(raw, prefix, news_ids):
    """go_<prefix>_* and go_<prefix>_pct_* from one field's raw scores, aligned to news_ids (NaN where unscored)."""
    go = prefixed(go_features(raw), prefix)
    go = go.merge(go_percentiles(go, f"go_{prefix}"), on="news_id", validate="one_to_one")
    return go.set_index("news_id").reindex(pd.Index(news_ids, name="news_id")).reset_index()


def merge_features(news_ids, frames):
    """Left-join feature frames on news_id in news order; refuses duplicate ids or clashing column names."""
    out = pd.DataFrame({"news_id": list(news_ids)})
    for f in frames:
        if f["news_id"].duplicated().any():
            raise ValueError("duplicate news ids in a feature frame")
        clash = (set(f.columns) & set(out.columns)) - {"news_id"}
        if clash:
            raise ValueError(f"columns already present: {sorted(clash)}")
        out = out.merge(f, on="news_id", how="left", validate="one_to_one")
    return out
