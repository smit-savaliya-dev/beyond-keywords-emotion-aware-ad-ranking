"""RQ5, descriptive (D56, D68): per-impression AUC difference on dev, by category, profile emotion and history length, plus examples.
Runs once by hand after the dev runs. It reads dev, so it is not a pipeline step. It selects and changes nothing.

  python scripts/04_08_where_it_helps.py                                     # M3 minus M2, every breakdown and the examples
  python scripts/04_08_where_it_helps.py --base M4_lightgbm --model M5_lightgbm_emotion --out m5_where_it_helps --compact
"""
import argparse
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.baselines import load_part
from src.config import CONGRUENCE_MAX_HISTORY, M2_NAME, M3_NAME, PROCESSED, RESULTS, RQ5_HISTORY_SHOWN, TABLES
from src.congruence import history_index_lists
from src.run_registry import load_rows
from src.provenance import run_info
from src.where_it_helps import (HISTORY_BINS, breakdown, clicked_category, dominant_emotion, example_impressions, history_bin, top_categories)


def example_rows(keys, kind, per, truth, histories_of, titles):
    rows = []
    for k in keys:
        cand = truth[truth["imp_key"] == k]
        base = {"kind": kind, "imp_key": k, "m2_auc": per.loc[k, "auc_m2"], "m3_auc": per.loc[k, "auc_m3"]}
        rows += [{**base, "role": "history", "clicked": "", "title": titles[n]} for n in histories_of[k] if n in titles]
        rows += [{**base, "role": "candidate", "clicked": int(r["label"]), "title": titles[r["news_id"]]} for _, r in cand.iterrows()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=M2_NAME)
    ap.add_argument("--model", default=M3_NAME)
    ap.add_argument("--out", default="m3_where_it_helps")
    ap.add_argument("--compact", action="store_true", help="category and history length only, no examples")
    args = ap.parse_args()
    p = PROCESSED
    per = pd.read_parquet(RESULTS / "per_impression" / f"{args.base}__dev.parquet").merge(
        pd.read_parquet(RESULTS / "per_impression" / f"{args.model}__dev.parquet"), on="imp_key", suffixes=("_m2", "_m3"), validate="one_to_one")
    per = per.assign(imp_key=per["imp_key"].astype(str)).set_index("imp_key")
    diff = per["auc_m3"] - per["auc_m2"]

    news = pd.read_parquet(p / "news.parquet", columns=["news_id", "category", "title"])
    category_of = news.set_index("news_id")["category"].astype(str)
    titles = news.set_index("news_id")["title"].to_dict()
    labels = pd.read_parquet(p / "news_emotion_labels_variants.parquet", columns=["news_id", "label_v2b"])
    index = pd.Index(labels["news_id"])

    truth, groups, histories, _ = load_part("dev", p)
    profile_of = pd.Series(groups, index=truth["imp_key"]).groupby(level=0).first()
    profile_emotion, used = dominant_emotion(history_index_lists(histories, index), labels["label_v2b"].to_numpy())
    rows = profile_of.reindex(per.index).to_numpy()

    fit = load_rows(p, ["fit"], ["news_id", "label"])
    top = top_categories(fit.loc[fit["label"] == 1, "news_id"].astype(str).to_numpy(), category_of)
    cat = clicked_category(truth, category_of, top).reindex(per.index).to_numpy()

    parts = [breakdown(diff, cat, "clicked category", [*top, "other"]),
             breakdown(diff, history_bin(used[rows]), "known history items", [b[2] for b in HISTORY_BINS])]
    if not args.compact:
        parts.insert(1, breakdown(diff, profile_emotion[rows], "dominant profile emotion"))
    table = pd.concat(parts, ignore_index=True)
    if args.compact:
        table = table.rename(columns={"pct_m3_better": "pct_model_better"})
    table.to_csv(TABLES / f"{args.out}_dev.csv", index=False)

    if not args.compact:
        sizes = truth.groupby("imp_key").size().reindex(per.index)
        helped, hurt = example_impressions(diff, sizes)
        hist_ids = {}
        for k in helped + hurt:
            h = np.asarray(histories[profile_of[k]], dtype=object)
            hist_ids[k] = [str(n) for n in h[-RQ5_HISTORY_SHOWN:]]
        examples = example_rows(helped, "helped", per, truth, hist_ids, titles) + example_rows(hurt, "hurt", per, truth, hist_ids, titles)
        pd.DataFrame(examples).to_csv(TABLES / f"{args.out}_examples_dev.csv", index=False)
    meta = {"rule": "D56" if not args.compact else "D68", "pair": [args.base, args.model], "top_categories_from_fit": top, "max_history_used": CONGRUENCE_MAX_HISTORY, "impressions": len(per),
            "impressions_without_defined_auc": int(diff.isna().sum()), "run_info": run_info()}
    (TABLES / f"{args.out}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.set_option("display.width", 170)
    print(table.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
