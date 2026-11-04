"""Paired comparisons for M4 and M5 (D68): per-impression metrics with Holm, 95% CI and dz; user-level AUC as a robustness check.
Reads saved per-impression results only; it never scores a model.

  python scripts/05_02_compare_m4m5.py --part valid
  python scripts/05_02_compare_m4m5.py --part dev       # by hand, after both --final-dev runs
"""
import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import (M2_NAME, M3_NAME, M4_NAME, M4_SAME_NAME, M5_ALL_NAME, M5_CONG_NAME, M5_FAIR_NAME, M5_ITEM_NAME, M5_NAME, M5_PERM_NAME, PROCESSED,
                        RESULTS, TABLES)
from src.evaluate import load_truth
from src.run_registry import load_rows
from src.results_tables import compare_pairs, fresh_split, merge_part

PAIRS = {"valid": [(M4_NAME, M5_NAME), (M3_NAME, M5_NAME), (M2_NAME, M4_NAME), (M5_NAME, M5_ALL_NAME), (M4_NAME, M5_FAIR_NAME), (M5_PERM_NAME, M5_NAME),
                   (M4_SAME_NAME, M5_NAME), (M4_SAME_NAME, M5_PERM_NAME), (M4_SAME_NAME, M5_CONG_NAME), (M4_SAME_NAME, M5_ITEM_NAME)],
         "dev": [(M4_NAME, M5_NAME), (M3_NAME, M5_NAME), (M2_NAME, M4_NAME), (M5_PERM_NAME, M5_NAME), (M4_NAME, M5_PERM_NAME), (M4_NAME, M5_CONG_NAME)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["valid", "dev"], default="valid")
    args = ap.parse_args()
    comp, user = compare_pairs(RESULTS / "per_impression", PAIRS[args.part], args.part)
    merge_part(TABLES / "m4m5_comparison.csv", comp, args.part)
    merge_part(TABLES / "m4m5_user_level.csv", user, args.part)
    fit_news = set(load_rows(PROCESSED, ["fit"], ["news_id"])["news_id"].astype(str))
    scores = [pd.read_parquet(RESULTS / "scores" / f"{m}__{args.part}.parquet") for m in (M4_NAME, M5_NAME)]
    fresh = fresh_split(load_truth(args.part, PROCESSED), *scores, fit_news, args.part)
    merge_part(TABLES / "m5_fresh_items.csv", fresh, args.part)
    pd.set_option("display.width", 170)
    print(fresh.round(5).to_string(index=False))
    show = ["model_a", "model_b", "metric", "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high", "dz", "p_holm", "significant_holm"]
    print(f"[{args.part}] B minus A")
    print(comp[comp["metric"].isin(["auc", "mrr", "ndcg@10"])][show].round(5).to_string(index=False))
    print(comp.groupby(["model_a", "model_b"])["significant_holm"].sum().to_string())
    print("\nUser-level AUC:")
    print(user[["model_a", "model_b", "users", "mean_diff", "ci_low", "ci_high", "dz", "p"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
