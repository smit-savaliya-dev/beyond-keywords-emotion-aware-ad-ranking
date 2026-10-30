"""Q03 / D40: do the results change if go_pct_* percentiles are fitted on train-split items only (data/raw/train/news.tsv)?
Reruns the 8 go_pct_* click signals and cong_pct_cos on fit and valid (dev is never read) and compares dz with the current
features. Current features stay in use; this is a robustness check.

  python scripts/37_percentile_robustness.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.baselines import train_news_ids
from src.click_signal import signal_table
from src.config import PERCENTILE_MAX_DZ_CHANGE, PROCESSED, RESULTS, SEED, SIGNAL_ALPHA, V2_MAIN_MEASURE
from src.congruence import congruence_features
from src.emotion_features import GO_EMOTIONS
from src.emotion_v2 import ecdf_percentiles, load_part_context
from src.provenance import run_info

TABLES = RESULTS / "tables"
ROLES = ("", "_personal", "_otheruser", "_userspecific")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    p = args.processed

    raw_cols = [f"go_{e}" for e in GO_EMOTIONS]
    pct_cols = [f"go_pct_{e}" for e in GO_EMOTIONS]
    feats = pd.read_parquet(p / "news_emotion_features_v2.parquet", columns=["news_id", *raw_cols, *pct_cols])
    index = pd.Index(feats["news_id"])
    in_train = feats["news_id"].isin(train_news_ids()).to_numpy()
    current = feats[pct_cols].to_numpy(np.float64)
    refit = ecdf_percentiles(feats[raw_cols].to_numpy(np.float64), in_train)
    signal = pd.read_csv(TABLES / "emotion_signal_v2.csv").set_index(["part", "feature"])

    rows = []
    for part in ("fit", "valid"):
        ctx = load_part_context(p, part, index)
        values = {c: refit[ctx.item_idx, j].astype(np.float32) for j, c in enumerate(pct_cols)}
        cong, _ = congruence_features(ctx.groups, ctx.item_idx, ctx.histories, {"pct": refit}, V2_MAIN_MEASURE, shuffle_seed=SEED)
        values.update({f"cong_pct_cos{r}": cong[f"cong_pct_cos{r}"] for r in ROLES})
        t = signal_table(ctx.groups, ctx.labels, values, SIGNAL_ALPHA, ctx.group_users).set_index("feature")
        for f in values:
            rows.append({"part": part, "feature": f, "dz_current": float(signal.loc[(part, f), "dz"]), "dz_train_ecdf": float(t.loc[f, "dz"]),
                         "significant_both_current": bool(signal.loc[(part, f), "significant_both"]),
                         "significant_both_train_ecdf": bool(t.loc[f, "significant_both"])})
    table = pd.DataFrame(rows)
    table["change"] = table["dz_train_ecdf"] - table["dz_current"]
    largest = float(table["change"].abs().max())
    keep = largest < PERCENTILE_MAX_DZ_CHANGE

    TABLES.mkdir(parents=True, exist_ok=True)
    table.round(6).to_csv(TABLES / "percentile_robustness.csv", index=False)
    meta = {"rule": "D40", "n_items": len(feats), "n_train_items": int(in_train.sum()), "max_abs_dz_change": round(largest, 5),
            "limit": PERCENTILE_MAX_DZ_CHANGE, "keep_current_features": bool(keep),
            "max_abs_percentile_change": round(float(np.abs(refit - current).max()), 5),
            "mean_abs_percentile_change": round(float(np.abs(refit - current).mean()), 5),
            "significance_changes": int((table["significant_both_current"] != table["significant_both_train_ecdf"]).sum()),
            "run_info": run_info()}
    (TABLES / "percentile_robustness_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.set_option("display.width", 150)
    print(table.round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in meta.items() if k != "run_info"}, indent=1))


if __name__ == "__main__":
    main()
