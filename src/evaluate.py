"""The evaluation harness. EVERY model is scored through this file, the same way.

Score-file contract (one parquet per model per part):
    imp_key, news_id, score        (higher score = rank higher)
Labels are taken from the ground truth, never from the model's file.
"""
import numpy as np
import pandas as pd

from .config import K_VALUES, PROCESSED, RESULTS, SEED
from .metrics import impression_metrics

KEY = ["imp_key", "news_id"]


def load_truth(part, processed=PROCESSED):
    """Ground-truth candidates + labels for one part: 'fit', 'valid' or 'dev'."""
    imps = pd.read_parquet(processed / "impressions.parquet",
                           columns=["imp_key", "part", "news_id", "label", "position"])
    truth = imps[imps["part"].astype(str) == part].drop(columns="part")
    for col in KEY:
        truth[col] = truth[col].astype(str)
    if truth.empty:
        raise ValueError(f"No impressions for part '{part}'")
    return truth.reset_index(drop=True)


def attach_scores(truth, scores):
    """Join a model's scores to the ground truth and enforce the contract."""
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
        raise ValueError(f"score file has {len(s)} rows but ground truth has {len(truth)}. "
                         "Every model must score every candidate of every impression.")
    df = truth.merge(s, on=KEY, how="left", validate="one_to_one")
    if df["score"].isna().any():
        raise ValueError(f"{int(df['score'].isna().sum())} candidates have no score "
                         "(wrong part, or keys do not match)")
    return df


def evaluate(truth, scores, ks=K_VALUES, seed=SEED):
    """Returns (per_impression DataFrame, mean of each metric).

    Impressions are processed in a fixed order with a fixed seed, so every model
    gets the SAME random tie-breaks. That keeps paired comparisons fair."""
    df = attach_scores(truth, scores).sort_values(["imp_key", "position"], kind="mergesort")
    keys = df["imp_key"].to_numpy()
    y = df["label"].to_numpy(dtype=float)
    s = df["score"].to_numpy(dtype=float)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    rng = np.random.default_rng(seed)
    rows = [impression_metrics(y[a:b], s[a:b], ks, rng) for a, b in zip(starts, ends)]
    per_imp = pd.DataFrame(rows)
    per_imp.insert(0, "imp_key", keys[starts])
    return per_imp, per_imp.drop(columns="imp_key").mean()


def save_results(model, part, scores, per_imp, summary, results=RESULTS):
    """Saves scores + per-impression metrics, and updates results/tables/summary.csv."""
    for sub in ("scores", "per_impression", "tables"):
        (results / sub).mkdir(parents=True, exist_ok=True)
    scores[["imp_key", "news_id", "score"]].to_parquet(
        results / "scores" / f"{model}__{part}.parquet", index=False)
    per_imp.to_parquet(results / "per_impression" / f"{model}__{part}.parquet", index=False)

    table = results / "tables" / "summary.csv"
    row = pd.DataFrame([{"model": model, "part": part, "impressions": len(per_imp),
                         **summary.round(4).to_dict()}])
    if table.exists():
        old = pd.read_csv(table)
        old = old[~((old["model"] == model) & (old["part"] == part))]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(table, index=False)
