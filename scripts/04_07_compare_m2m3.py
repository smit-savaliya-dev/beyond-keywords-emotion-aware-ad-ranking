"""Paired comparisons for M2 and M3 (D51, D55): per-impression metrics with Holm, 95% CI and dz; user-level AUC as a robustness check.
Reads saved per-impression results only; it never scores a model.

  python scripts/04_07_compare_m2m3.py --part valid
  python scripts/04_07_compare_m2m3.py --part dev       # by hand, after both --final-dev runs
"""
import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import M0_NAME, M1_NAME, M2_NAME, M3_NAME, RESULTS, TABLES
from src.results_tables import compare_pairs, merge_part

PAIRS = {"valid": [(M1_NAME, M2_NAME), (M2_NAME, M3_NAME)],
         "dev": [(M2_NAME, M3_NAME), (M0_NAME, M3_NAME), (M1_NAME, M3_NAME), (M0_NAME, M2_NAME), (M1_NAME, M2_NAME)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["valid", "dev"], default="valid")
    args = ap.parse_args()
    comp, user = compare_pairs(RESULTS / "per_impression", PAIRS[args.part], args.part)
    merge_part(TABLES / "m2m3_comparison.csv", comp, args.part)
    merge_part(TABLES / "m2m3_user_level.csv", user, args.part)
    pd.set_option("display.width", 170)
    show = ["model_a", "model_b", "metric", "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high", "dz", "p_holm", "significant_holm"]
    print(f"[{args.part}] B minus A")
    print(comp[comp["metric"].isin(["auc", "mrr", "ndcg@10"])][show].round(5).to_string(index=False))
    print("\nUser-level AUC:")
    print(user[["model_a", "model_b", "users", "mean_diff", "ci_low", "ci_high", "dz", "p"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
