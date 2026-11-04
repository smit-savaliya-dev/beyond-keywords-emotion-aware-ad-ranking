"""Where do emotion features rank inside M5? LightGBM gain importance and mean |pred_contrib| on seeded valid impressions (D68). Dev is never read.

  python scripts/05_03_feature_importance.py
"""
import json

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import DIAGNOSTICS, LTR_FEATURE_FILE, LTR_IMPORTANCE_IMPRESSIONS, M5_NAME, PROCESSED, SEED, TABLES
from src.run_registry import load_frozen
from src.lightgbm_ranker import family_shares, importance_table, load_model, model_features
from src.provenance import run_info
from src.results_charts import importance_chart


def main():
    booster = load_model(M5_NAME, load_frozen(M5_NAME)["config"]["model_sha256"])
    frame = pd.read_parquet(PROCESSED / LTR_FEATURE_FILE.format(part="valid"))
    t = importance_table(booster, frame, model_features(M5_NAME), LTR_IMPORTANCE_IMPRESSIONS, SEED)
    t.to_csv(TABLES / "m5_feature_importance.csv", index=False)
    shares = family_shares(t)
    best = t[t["family"] == "emotion"].iloc[0]
    meta = {"rule": "D68", "part": "valid", "impressions": LTR_IMPORTANCE_IMPRESSIONS, "family_shares": shares,
            "top_emotion_feature": best["feature"], "top_emotion_rank": int(best["contrib_rank"]), "n_features": len(t), "run_info": run_info()}
    (TABLES / "m5_feature_importance_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    caption = (f"Mean absolute contribution to the M5 score on {LTR_IMPORTANCE_IMPRESSIONS:,} valid impressions: emotion features hold "
               f"{100 * shares['emotion']['contrib_share']:.0f}% of the total and the strongest, {best['feature']}, ranks {int(best['contrib_rank'])} of {len(t)}.")
    importance_chart(t, DIAGNOSTICS / "m5_feature_importance.png", "What M5 uses: contribution of each feature", caption)
    pd.set_option("display.width", 150)
    print(t.round(4).to_string(index=False))
    print(shares)


if __name__ == "__main__":
    main()
