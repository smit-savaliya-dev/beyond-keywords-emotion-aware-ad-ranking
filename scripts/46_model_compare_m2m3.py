"""Paired comparisons for M2 and M3 (D51, D55): per-impression metrics with Holm, 95% CI and dz; user-level AUC as a robustness check.
Reads saved per-impression results only; it never scores a model.

  python scripts/46_model_compare_m2m3.py --part valid
  python scripts/46_model_compare_m2m3.py --part dev       # by hand, after both --final-dev runs
"""
import argparse
import json

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import M0_NAME, M1_NAME, M2_NAME, M3_NAME, METRIC_COLUMNS, PROCESSED, RESULTS, TABLES
from src.stats import compare_models, paired_test

PAIRS = {"valid": [(M1_NAME, M2_NAME), (M2_NAME, M3_NAME)],
         "dev": [(M2_NAME, M3_NAME), (M0_NAME, M3_NAME), (M1_NAME, M3_NAME), (M0_NAME, M2_NAME), (M1_NAME, M2_NAME)]}


def merge_part(path, new, part):
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old["part"] != part]
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["valid", "dev"], default="valid")
    args = ap.parse_args()

    expected = json.loads((PROCESSED / "data_summary.json").read_text(encoding="utf-8"))["parts"][args.part]["impressions"]
    per = {}
    for model in {m for pair in PAIRS[args.part] for m in pair}:
        path = RESULTS / "per_impression" / f"{model}__{args.part}.parquet"
        if not path.exists():
            raise SystemExit(f"{path} missing: run the {model} script first")
        per[model] = pd.read_parquet(path)
        if len(per[model]) != expected:
            raise SystemExit(f"{model} has {len(per[model])} impressions on {args.part}, expected {expected}")
    comp = pd.concat([compare_models(per[a], per[b], METRIC_COLUMNS, a, b) for a, b in PAIRS[args.part]], ignore_index=True)
    comp = comp.assign(part=args.part)

    beh = pd.read_parquet(PROCESSED / "behaviors.parquet", columns=["imp_key", "part", "user_id"], filters=[("part", "==", args.part)])
    users = beh.set_index(beh["imp_key"].astype(str))["user_id"].astype(str)
    rows = []
    for a, b in PAIRS[args.part]:
        m = per[a].merge(per[b], on="imp_key", suffixes=("_a", "_b"), validate="one_to_one")
        u = m.groupby(m["imp_key"].astype(str).map(users))[["auc_a", "auc_b"]].mean()
        rows.append({"part": args.part, "model_a": a, "model_b": b, "users": len(u), **paired_test(u["auc_a"], u["auc_b"])})
    user = pd.DataFrame(rows)

    merge_part(TABLES / "m2m3_comparison.csv", comp, args.part)
    merge_part(TABLES / "m2m3_user_level.csv", user, args.part)
    pd.set_option("display.width", 170)
    show = ["model_a", "model_b", "metric", "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high", "dz", "p_holm", "significant_holm"]
    print(f"[{args.part}] {expected:,} impressions; B minus A")
    print(comp[comp["metric"].isin(["auc", "mrr", "ndcg@10"])][show].round(5).to_string(index=False))
    print("\nUser-level AUC:")
    print(user[["model_a", "model_b", "users", "mean_diff", "ci_low", "ci_high", "dz", "p"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
