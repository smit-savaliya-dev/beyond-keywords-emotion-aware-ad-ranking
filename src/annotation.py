"""Manual-label sheet and agreement statistics for validating the emotion labels."""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score

from .config import SEED
from .emotion import SIX_CLASSES

BANDS = [(-1.0, "poor"), (0.0, "slight"), (0.2, "fair"), (0.4, "moderate"), (0.6, "substantial"), (0.8, "almost perfect")]


def kappa_band(k):
    name = BANDS[0][1]
    for threshold, label in BANDS:
        if k >= threshold:
            name = label
    return name


def make_sample(news, n=180, seed=SEED):
    return news.sample(n=min(n, len(news)), random_state=seed).reset_index(drop=True)


def build_sheet(sample):
    sheet = sample[["news_id", "title", "abstract"]].copy()
    sheet.insert(0, "sample_id", range(1, len(sheet) + 1))
    sheet["your_label"] = ""
    return sheet


def read_sheet(path):
    for enc in ("utf-8-sig", "cp1252"):
        try:
            sheet = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"cannot decode {path}; save it as CSV UTF-8")
    sheet["your_label"] = sheet["your_label"].str.strip().str.lower()
    return sheet


def sheet_problems(sheet):
    problems = []
    empty = int((sheet["your_label"] == "").sum())
    if empty:
        problems.append(f"{empty}/{len(sheet)} rows have an empty your_label")
    bad = sorted(set(sheet["your_label"]) - set(SIX_CLASSES) - {""})
    if bad:
        problems.append(f"labels outside {SIX_CLASSES}: {bad}")
    return problems


def agreement(y_true, y_pred, n_boot=1000, seed=SEED):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=SIX_CLASSES)
    rng = np.random.default_rng(seed)
    boots = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            i = rng.integers(0, len(y_true), len(y_true))
            boots.append(cohen_kappa_score(y_true[i], y_pred[i], labels=SIX_CLASSES))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=SIX_CLASSES, average="macro", zero_division=0)),
        "kappa": float(kappa), "kappa_ci95": [float(lo), float(hi)], "kappa_band": kappa_band(kappa),
        "confusion": pd.DataFrame(confusion_matrix(y_true, y_pred, labels=SIX_CLASSES),
                                  index=[f"true_{c}" for c in SIX_CLASSES],
                                  columns=[f"pred_{c}" for c in SIX_CLASSES]),
        "report": classification_report(y_true, y_pred, labels=SIX_CLASSES, zero_division=0),
    }


def vader_correlation(texts, valence):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return None
    sia = SentimentIntensityAnalyzer()
    compound = np.array([sia.polarity_scores(t)["compound"] for t in texts])
    valence = np.asarray(valence, dtype=float)
    if compound.std() == 0 or valence.std() == 0:
        return None
    return float(np.corrcoef(compound, valence)[0, 1])
