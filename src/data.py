"""Load, clean and reshape MIND-small.

Creates in data/processed/:
  news.parquet         one row per news item (train + dev combined, no duplicates)
  behaviors.parquet    one row per impression, click history parsed into a list
  impressions.parquet  one row per (impression, candidate) with its 0/1 click label
  data_summary.json    counts and checks to quote in the thesis

Parts (column "part"):
  fit   = train impressions except the last day  -> train models here
  valid = last day of train impressions          -> tune alpha / hyperparameters here
  dev   = MIND-small dev set                     -> final test, report once
"""
import csv
import json

import numpy as np
import pandas as pd

from .config import BEHAVIOR_COLS, NEWS_COLS, PROCESSED, RAW, TIME_FORMAT

SPLITS = ("train", "dev")


def find_split_dir(split, raw=RAW):
    """Folder holding behaviors.tsv + news.tsv (works whether or not the zip had a subfolder)."""
    hits = sorted((raw / split).rglob("behaviors.tsv"))
    if not hits:
        raise FileNotFoundError(
            f"No behaviors.tsv under {raw / split}. Run scripts/01_download_mind.py first.")
    return hits[0].parent


def read_tsv(path, cols):
    # QUOTE_NONE: some titles contain " characters; normal quoting silently merges rows.
    # keep_default_na=False: empty abstract stays "" (not NaN) and a title like "NA" stays text.
    return pd.read_csv(path, sep="\t", header=None, names=cols, dtype=str,
                       quoting=csv.QUOTE_NONE, keep_default_na=False)


def load_news(raw=RAW):
    frames = [read_tsv(find_split_dir(s, raw) / "news.tsv", NEWS_COLS) for s in SPLITS]
    news = pd.concat(frames, ignore_index=True).drop_duplicates("news_id", keep="first")
    news = news.drop(columns=["url", "title_entities", "abstract_entities"])
    news["title"] = news["title"].str.strip()
    news["abstract"] = news["abstract"].str.strip()
    # text used later by TF-IDF, SBERT and the emotion classifier
    news["text"] = np.where(news["abstract"] == "", news["title"],
                            news["title"] + ". " + news["abstract"])
    return news.reset_index(drop=True)


def assign_parts(beh):
    """fit = train minus its last day, valid = last day of train, dev = dev."""
    part = pd.Series("dev", index=beh.index, dtype=object)
    is_train = (beh["split"] == "train").to_numpy()
    day = beh["time"].dt.normalize()
    last_train_day = day[is_train].max()
    part[is_train] = np.where(day[is_train] == last_train_day, "valid", "fit")
    return part


def load_behaviors(raw=RAW):
    frames = []
    for split in SPLITS:
        b = read_tsv(find_split_dir(split, raw) / "behaviors.tsv", BEHAVIOR_COLS)
        b["split"] = split
        frames.append(b)
    beh = pd.concat(frames, ignore_index=True)
    beh["time"] = pd.to_datetime(beh["time"], format=TIME_FORMAT)
    # impression IDs are only unique inside one split, so build a key unique across splits
    beh["imp_key"] = beh["split"].astype(str) + "-" + beh["impression_id"].astype(str)
    # history is in time order: oldest first, newest last. "" becomes []
    beh["history"] = beh["history"].astype(str).apply(str.split)
    beh["history_len"] = beh["history"].apply(len).astype("int32")
    beh["part"] = assign_parts(beh)
    return beh


def explode_impressions(beh):
    """One row per candidate. `position` = order shown in the log; NEVER use it as a model feature."""
    ex = beh[["imp_key", "part", "user_id", "time", "impressions"]].copy()
    ex["impressions"] = ex["impressions"].astype(str).apply(str.split)
    ex = ex.explode("impressions", ignore_index=True).dropna(subset=["impressions"])
    ex["position"] = ex.groupby("imp_key", sort=False).cumcount().astype("int16")
    pair = ex["impressions"].astype(str).str.rsplit("-", n=1, expand=True)
    ex["news_id"] = pair[0]
    ex["label"] = pair[1].astype("int8")
    ex = ex.drop(columns="impressions").reset_index(drop=True)
    for col in ("imp_key", "part", "user_id", "news_id"):
        ex[col] = ex[col].astype("category")
    return ex


def summarise(news, beh, imps):
    """Numbers to check now and to quote in the Data section of the thesis."""
    known = news["news_id"]
    history_ids = pd.Series(sorted({i for h in beh["history"] for i in h}), dtype=object)
    summary = {
        "news_items": int(len(news)),
        "empty_abstract_pct": round(100 * float((news["abstract"] == "").mean()), 2),
        "categories": int(news["category"].nunique()),
        "missing_candidate_ids": int((~imps["news_id"].astype(str).isin(known)).sum()),
        "missing_history_ids": int((~history_ids.isin(known)).sum()),
        "parts": {},
    }
    per_imp = imps.groupby("imp_key", observed=True)["label"].agg(n="size", clicks="sum")
    per_imp.index = per_imp.index.astype(str)
    part_of = beh.set_index("imp_key")["part"]
    per_imp["part"] = part_of.reindex(per_imp.index).to_numpy()

    for part in ("fit", "valid", "dev"):
        b = beh[beh["part"] == part]
        g = per_imp[per_imp["part"] == part]
        if b.empty:
            continue
        summary["parts"][part] = {
            "impressions": int(len(b)),
            "users": int(b["user_id"].nunique()),
            "candidate_rows": int(g["n"].sum()),
            "candidates_per_impression": {
                "mean": round(float(g["n"].mean()), 2), "median": float(g["n"].median()),
                "min": int(g["n"].min()), "max": int(g["n"].max())},
            "clicks_per_impression_mean": round(float(g["clicks"].mean()), 3),
            "impressions_without_click": int((g["clicks"] == 0).sum()),
            "impressions_without_nonclick": int((g["clicks"] == g["n"]).sum()),
            "cold_start_pct": round(100 * float((b["history_len"] == 0).mean()), 2),
            "history_len_median": float(b["history_len"].median()),
            "first_time": str(b["time"].min()),
            "last_time": str(b["time"].max()),
        }
    return summary


def build_all(raw=RAW):
    news = load_news(raw)
    beh = load_behaviors(raw)
    imps = explode_impressions(beh)
    return news, beh, imps, summarise(news, beh, imps)


def save_all(news, beh, imps, summary, out=PROCESSED):
    out.mkdir(parents=True, exist_ok=True)
    news.to_parquet(out / "news.parquet", index=False)
    beh.drop(columns="impressions").to_parquet(out / "behaviors.parquet", index=False)
    imps.to_parquet(out / "impressions.parquet", index=False)
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2))


def load_processed(name, out=PROCESSED):
    """Later steps use this, e.g. load_processed('news')."""
    return pd.read_parquet(out / f"{name}.parquet")
