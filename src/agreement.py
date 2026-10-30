"""Agreement statistics between emotion labels: kappa with a bootstrap CI, accuracy, macro-F1."""
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


def compare_sources(y_true, predictions, n_boot=1000, seed=SEED):
    """predictions: {source: labels aligned with y_true}. -> (summary table, {source: agreement() result})."""
    results, rows = {}, []
    for name, y_pred in predictions.items():
        res = agreement(y_true, y_pred, n_boot=n_boot, seed=seed)
        results[name] = res
        lo, hi = res["kappa_ci95"]
        pred = pd.Series(np.asarray(y_pred))
        rows.append({"source": name, "n": res["n"], "kappa": res["kappa"], "kappa_ci_low": lo, "kappa_ci_high": hi,
                     "kappa_band": res["kappa_band"], "accuracy": res["accuracy"], "macro_f1": res["macro_f1"],
                     "pct_pred_neutral": 100 * float((pred == "neutral").mean())})
    return pd.DataFrame(rows), results
