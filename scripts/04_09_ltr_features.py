"""Feature tables for M4 and M5 on fit and valid (D63-D65), with check D67b: valid features must equal the saved M0, M1 and M2 scores.
Dev features are built only inside --final-dev of script 49.

  python scripts/04_09_ltr_features.py
"""
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import LTR_FEATURE_FILE, PROCESSED, TABLES
from src.lightgbm_ranker import build_features, check_saved_scores
from src.provenance import run_info


def main():
    p = PROCESSED
    frames = {}
    for part in ("fit", "valid"):
        frames[part] = build_features(part, p)
        frames[part].to_parquet(p / LTR_FEATURE_FILE.format(part=part), index=False)
        print(f"{part}: {frames[part].shape}, NaN cells {int(frames[part].isna().sum().sum())}")

    checks = check_saved_scores(frames["valid"], "valid")
    print("largest gap to the saved scores:", checks)

    fit = frames["fit"]
    days = fit.groupby("day").agg(rows=("label", "size"), clicks=("label", "sum"), pop_distinct=("pop_ctr", "nunique"))
    days.index = days.index.astype(str)
    meta = {"rules": "D63, D64, D65", "rows": {k: len(v) for k, v in frames.items()}, "check_valid_max_gap": checks,
            "fit_days": days.to_dict("index"), "columns": list(fit.columns), "run_info": run_info()}
    Path(TABLES / "ltr_features_meta.json").write_text(json.dumps(meta, indent=2, default=int), encoding="utf-8")
    print(days.to_string())


if __name__ == "__main__":
    main()
