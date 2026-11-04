"""Paired comparison M1 vs M0 on per-impression metrics (paired t-test, 95% CI, dz, Holm across metrics; Wilcoxon
robustness only), plus sanity checks: same impression count for every model, each model above random (AUC 0.5).
Reads saved per-impression results only; it never scores a model.
"""
import argparse
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import M0_NAME, M1_NAME, METRIC_COLUMNS, PROCESSED, RESULTS, TABLES
from src.stats import compare_models, paired_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["valid", "dev"], default="valid")
    args = ap.parse_args()

    expected = json.loads((PROCESSED / "data_summary.json").read_text(encoding="utf-8"))["parts"][args.part]
    per = {}
    for model in (M0_NAME, M1_NAME):
        path = RESULTS / "per_impression" / f"{model}__{args.part}.parquet"
        if not path.exists():
            raise SystemExit(f"{path} missing: run the {model} script first")
        per[model] = pd.read_parquet(path)
        if len(per[model]) != expected["impressions"]:
            raise SystemExit(f"{model} has {len(per[model])} impressions on {args.part}, expected "
                             f"{expected['impressions']}")
    comp = compare_models(per[M0_NAME], per[M1_NAME], METRIC_COLUMNS, M0_NAME, M1_NAME).assign(part=args.part)

    random_rows = []
    for model, t in per.items():
        r = paired_test(np.full(len(t), 0.5), t["auc"].to_numpy())
        random_rows.append({"part": args.part, "model": model, "auc": r["mean_b"], "auc_minus_0.5": r["mean_diff"],
                            "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p"],
                            "above_random": bool(r["ci_low"] > 0)})
    sanity = pd.DataFrame(random_rows)

    out = TABLES / "baselines_comparison.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old["part"] != args.part]
    pd.concat([old, comp], ignore_index=True).to_csv(out, index=False)
    sane = TABLES / "baselines_sanity.csv"
    old = pd.read_csv(sane) if sane.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old["part"] != args.part]
    pd.concat([old, sanity], ignore_index=True).to_csv(sane, index=False)

    pd.set_option("display.width", 170)
    print(f"[{args.part}] {expected['impressions']:,} impressions for both models")
    print(sanity.round(4).to_string(index=False))
    show = ["metric", "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high", "dz", "p_holm", "significant_holm",
            "pct_b_better", "p_wilcoxon_holm"]
    print(f"\n{M1_NAME} minus {M0_NAME}:")
    print(comp[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
