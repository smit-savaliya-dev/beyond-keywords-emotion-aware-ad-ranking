import csv
import json

import numpy as np
import pandas as pd

from .config import BEHAVIOR_COLS, NEWS_COLS, OFFICIAL_COUNTS, PROCESSED, RAW, TIME_FORMAT
from .provenance import run_info, sha256_file

SPLITS = ("train", "dev")
PARTS = ("fit", "valid", "dev")


def find_split_dir(split, raw=RAW):
    hits = sorted((raw / split).rglob("behaviors.tsv"))
    if not hits:
        raise FileNotFoundError(f"No behaviors.tsv under {raw / split}. Run scripts/01_download_mind.py first.")
    return hits[0].parent


def read_tsv(path, cols):
    # QUOTE_NONE: titles contain stray quotes. keep_default_na=False: "" and "NA" stay text.
    return pd.read_csv(path, sep="\t", header=None, names=cols, dtype=str,
                       quoting=csv.QUOTE_NONE, keep_default_na=False, encoding="utf-8")


def load_news(raw=RAW):
    frames = [read_tsv(find_split_dir(s, raw) / "news.tsv", NEWS_COLS) for s in SPLITS]
    news = pd.concat(frames, ignore_index=True).drop_duplicates("news_id", keep="first")
    news = news.drop(columns=["url", "title_entities", "abstract_entities"])
    news["title"] = news["title"].str.strip()
    news["abstract"] = news["abstract"].str.strip()
    news["text"] = np.where(news["abstract"] == "", news["title"], news["title"] + ". " + news["abstract"])
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
    # impression ids repeat across splits, so build a globally unique key
    beh["imp_key"] = beh["split"].astype(str) + "-" + beh["impression_id"].astype(str)
    if not beh["imp_key"].is_unique:
        raise ValueError("duplicate impression ids inside a split")
    beh["history"] = beh["history"].astype(str).apply(str.split)  # oldest -> newest
    beh["history_len"] = beh["history"].apply(len).astype("int32")
    beh["part"] = assign_parts(beh)
    return beh


def explode_impressions(beh):
    """One row per candidate. `position` is log order: never use it as a model feature."""
    ex = beh[["imp_key", "part", "user_id", "time", "impressions"]].copy()
    ex["impressions"] = ex["impressions"].astype(str).apply(str.split)
    ex = ex.explode("impressions", ignore_index=True).dropna(subset=["impressions"])
    ex["position"] = ex.groupby("imp_key", sort=False).cumcount().astype("int16")
    pair = ex["impressions"].astype(str).str.rsplit("-", n=1, expand=True)
    ex["news_id"] = pair[0]
    ex["label"] = pair[1].astype("int8")
    if not ex["label"].isin([0, 1]).all():
        raise ValueError("labels other than 0/1 found in impressions")
    ex = ex.drop(columns="impressions").reset_index(drop=True)
    for col in ("imp_key", "part", "user_id", "news_id"):
        ex[col] = ex[col].astype("category")
    return ex


def summarise(news, beh, imps):
    known = news["news_id"]
    history_ids = pd.Series(sorted({i for h in beh["history"] for i in h}), dtype=object)
    summary = {
        "news_items": int(len(news)),
        "empty_abstract_pct": round(100 * float((news["abstract"] == "").mean()), 2),
        "empty_title": int((news["title"] == "").sum()),
        "categories": int(news["category"].nunique()),
        "missing_candidate_ids": int((~imps["news_id"].astype(str).isin(known)).sum()),
        "missing_history_ids": int((~history_ids.isin(known)).sum()),
        "parts": {},
    }
    per_imp = imps.groupby("imp_key", observed=True)["label"].agg(n="size", clicks="sum")
    per_imp.index = per_imp.index.astype(str)
    per_imp["part"] = beh.set_index("imp_key")["part"].reindex(per_imp.index).to_numpy()

    for part in PARTS:
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


def integrity_problems(summary):
    problems = []
    if summary["missing_candidate_ids"]:
        problems.append("some candidate ids are not in the news table")
    if summary["missing_history_ids"]:
        problems.append("some history ids are not in the news table")
    for part, p in summary["parts"].items():
        if p["impressions_without_click"] or p["impressions_without_nonclick"]:
            problems.append(f"{part}: impressions with only one label class")
    return problems


def check_official_counts(summary, official=OFFICIAL_COUNTS):
    parts = summary["parts"]
    got = {
        "news_items": summary["news_items"],
        "train_impressions": parts["fit"]["impressions"] + parts["valid"]["impressions"],
        "dev_impressions": parts["dev"]["impressions"],
    }
    return [f"{k}: expected {official[k]:,}, got {got[k]:,}" for k in official if got[k] != official[k]]


def build_all(raw=RAW):
    news = load_news(raw)
    beh = load_behaviors(raw)
    imps = explode_impressions(beh)
    summary = summarise(news, beh, imps)
    summary["raw_sha256"] = {f"{s}/{n}": sha256_file(find_split_dir(s, raw) / n)
                             for s in SPLITS for n in ("news.tsv", "behaviors.tsv")}
    summary["run_info"] = run_info()
    return news, beh, imps, summary


def save_all(news, beh, imps, summary, out=PROCESSED):
    out.mkdir(parents=True, exist_ok=True)
    news.to_parquet(out / "news.parquet", index=False)
    beh.drop(columns="impressions").to_parquet(out / "behaviors.parquet", index=False)
    imps.to_parquet(out / "impressions.parquet", index=False)
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_processed(name, out=PROCESSED):
    return pd.read_parquet(out / f"{name}.parquet")
