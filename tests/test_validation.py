
import numpy as np
import pandas as pd
import pytest

from src.emotion import GOEMOTIONS_LABELS, SIX_CLASSES
from src.emotion_features import GO_EMOTIONS
from src.validation import PCT_GROUP, threshold_corpus_shares, threshold_labels_pct, threshold_labels_raw

IDX = {l: i for i, l in enumerate(GOEMOTIONS_LABELS)}


def raw_rows(spec):
    """spec: list of {label: score}; other labels 0.01."""
    rows = np.full((len(spec), 28), 0.01)
    for r, d in enumerate(spec):
        for k, v in d.items():
            rows[r, IDX[k]] = v
    return rows


def test_threshold_labels_raw_uses_non_neutral_max_and_group():
    raw = pd.DataFrame(raw_rows([{"grief": 0.6, "neutral": 0.9}, {"curiosity": 0.15}, {"anger": 0.04}]),
                       columns=GOEMOTIONS_LABELS)
    assert threshold_labels_raw(raw, 0.1).tolist() == ["sadness", "surprise", "neutral"]
    assert threshold_labels_raw(raw, 0.5).tolist() == ["sadness", "neutral", "neutral"]   # neutral score ignored


def test_threshold_labels_pct_and_groups():
    assert PCT_GROUP["curiosity"] == "surprise" and PCT_GROUP["nervousness"] == "fear"
    assert "sadness" not in PCT_GROUP.values()                                             # documented limitation
    pct = pd.DataFrame([[0.5] * 8, [0.5] * 7 + [0.97]], columns=[f"go_pct_{e}" for e in GO_EMOTIONS])
    assert threshold_labels_pct(pct, 0.9).tolist() == ["neutral", PCT_GROUP[GO_EMOTIONS[-1]]]


def test_threshold_corpus_shares_rows_and_sums():
    raw = pd.DataFrame(raw_rows([{"joy": 0.3}, {"fear": 0.08}]), columns=GOEMOTIONS_LABELS)
    pct = pd.DataFrame([[0.99] + [0.1] * 7, [0.1] * 8], columns=[f"go_pct_{e}" for e in GO_EMOTIONS])
    t = threshold_corpus_shares(raw, pct)
    assert len(t) == 11 and set(t["score"]) == {"raw_max_sigmoid", "pct_max_percentile"}
    share_cols = [f"share_{c}" for c in SIX_CLASSES]
    assert np.allclose(t[share_cols].sum(axis=1), 100)
    assert t.loc[(t["score"] == "raw_max_sigmoid") & (t["threshold"] == 0.05), "share_neutral"].item() == 0
