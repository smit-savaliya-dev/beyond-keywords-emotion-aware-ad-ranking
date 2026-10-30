"""Threshold views of the emotion labels on the full corpus (item-level, no labels, no clicks)."""
import numpy as np
import pandas as pd

from .config import GO_FEATURES, THRESHOLDS_PCT, THRESHOLDS_RAW
from .emotion import GOEMOTIONS_LABELS, LABEL_TO_GROUP, SIX_CLASSES
from .emotion_features import GO_EMOTIONS

NON_NEUTRAL = [label for label in GOEMOTIONS_LABELS if label != "neutral"]
PCT_GROUP = {e: LABEL_TO_GROUP[labels[0]] for e, labels in GO_FEATURES.items()}  # e.g. curiosity -> surprise


def threshold_labels_raw(raw28, t):
    """Class of the strongest non-neutral GoEmotions label if its sigmoid > t, else neutral."""
    scores = raw28[NON_NEUTRAL].to_numpy(dtype=np.float64)
    best = scores.argmax(axis=1)
    groups = np.array([LABEL_TO_GROUP[NON_NEUTRAL[j]] for j in best], dtype=object)
    return np.where(scores.max(axis=1) > t, groups, "neutral")


def threshold_labels_pct(go_pct, t):
    """Class of the highest go_pct_* emotion if its percentile > t, else neutral."""
    cols = [f"go_pct_{e}" for e in GO_EMOTIONS]
    scores = go_pct[cols].to_numpy(dtype=np.float64)
    best = scores.argmax(axis=1)
    groups = np.array([PCT_GROUP[GO_EMOTIONS[j]] for j in best], dtype=object)
    return np.where(scores.max(axis=1) > t, groups, "neutral")


def threshold_corpus_shares(raw28, go_pct):
    """Item-level (no labels): class shares on the full corpus for every pre-declared threshold."""
    rows = []
    for kind, ts, fn, data in (("raw_max_sigmoid", THRESHOLDS_RAW, threshold_labels_raw, raw28),
                               ("pct_max_percentile", THRESHOLDS_PCT, threshold_labels_pct, go_pct)):
        for t in ts:
            lab = pd.Series(fn(data, t))
            shares = lab.value_counts(normalize=True).reindex(SIX_CLASSES, fill_value=0)
            rows.append({"score": kind, "threshold": t, **{f"share_{c}": round(100 * float(v), 2)
                                                           for c, v in shares.items()}})
    return pd.DataFrame(rows)
