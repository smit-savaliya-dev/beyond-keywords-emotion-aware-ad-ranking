"""The one evaluation path every model goes through.

Score file contract: columns imp_key, news_id, score for EVERY candidate of the part.
"""
import warnings

import numpy as np
import pandas as pd

from .config import K_VALUES, PROCESSED, RESULTS, SEED
from .metrics import impression_metrics

KEY = ["imp_key", "news_id"]


def load_truth(part, processed=PROCESSED):
    imps = pd.read_parquet(processed / "impressions.parquet",
                           columns=["imp_key", "part", "news_id", "label", "position"])
    truth = imps[imps["part"].astype(str) == part].drop(columns="part")
    if truth.empty:
        raise ValueError(f"No impressions for part '{part}'")
    for col in KEY:
        truth[col] = truth[col].astype(str)
    return truth.sort_values(["imp_key", "position"], kind="mergesort").reset_index(drop=True)


def attach_scores(truth, scores):
    missing = {"imp_key", "news_id", "score"} - set(scores.columns)
    if missing:
        raise ValueError(f"score file is missing columns: {missing}")
    s = scores[["imp_key", "news_id", "score"]].copy()
    for col in KEY:
        s[col] = s[col].astype(str)
    s["score"] = s["score"].astype(float)
    if s.duplicated(KEY).any():
        raise ValueError("duplicate (imp_key, news_id) rows in score file")
    if not np.isfinite(s["score"].to_numpy()).all():
        raise ValueError("score file contains NaN or inf")
    if len(s) != len(truth):
        raise ValueError(f"score file has {len(s)} rows, ground truth has {len(truth)}: "
                         "every model must score every candidate of every impression")
    df = truth.merge(s, on=KEY, how="left", validate="one_to_one")
    if df["score"].isna().any():
        raise ValueError(f"{int(df['score'].isna().sum())} candidates have no score (wrong part or keys)")
    return df


def evaluate(truth, scores, ks=K_VALUES, seed=SEED):
    """Returns (per-impression metrics, mean of each metric). Same seed => same tie-breaks for every model."""
    df = attach_scores(truth, scores).sort_values(["imp_key", "position"], kind="mergesort")
    keys = df["imp_key"].to_numpy()
    y = df["label"].to_numpy(dtype=float)
    s = df["score"].to_numpy(dtype=float)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    rng = np.random.default_rng(seed)
    per_imp = pd.DataFrame([impression_metrics(y[a:b], s[a:b], ks, rng) for a, b in zip(starts, ends)])
    per_imp.insert(0, "imp_key", keys[starts])
    n_nan = int(per_imp["auc"].isna().sum())
    if n_nan:
        warnings.warn(f"{n_nan} impressions have undefined AUC (only one label class)")
    return per_imp, per_imp.drop(columns="imp_key").mean()


def save_results(model, part, scores, per_imp, summary, results=RESULTS):
    for sub in ("scores", "per_impression", "tables"):
        (results / sub).mkdir(parents=True, exist_ok=True)
    scores[["imp_key", "news_id", "score"]].to_parquet(results / "scores" / f"{model}__{part}.parquet", index=False)
    per_imp.to_parquet(results / "per_impression" / f"{model}__{part}.parquet", index=False)

    table = results / "tables" / "summary.csv"
    row = pd.DataFrame([{"model": model, "part": part, "impressions": len(per_imp), **summary.round(4).to_dict()}])
    if table.exists():
        old = pd.read_csv(table)
        old = old[~((old["model"] == model) & (old["part"] == part))]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(table, index=False)


def evaluate_model(model, part, scores, save=True, processed=PROCESSED, results=RESULTS):
    """What every model script calls: score a model on one part, optionally save."""
    truth = load_truth(part, processed)
    per_imp, summary = evaluate(truth, scores)
    if save:
        save_results(model, part, scores, per_imp, summary, results)
    return per_imp, summary
