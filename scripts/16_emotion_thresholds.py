"""Item-level threshold sensitivity (no labels, no clicks): class shares on the full corpus for every pre-declared
threshold (THRESHOLDS_RAW / THRESHOLDS_PCT in config). Kappa against the hand labels is added by script 06 later.

  python scripts/16_emotion_thresholds.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import FIGURES, PROCESSED, RESULTS, THRESHOLDS_PCT, THRESHOLDS_RAW
from src.provenance import run_info
from src.validation import PCT_GROUP, plot_threshold_panels, threshold_corpus_shares

TABLES = RESULTS / "tables"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    raw = pd.read_parquet(args.processed / "news_emotion_raw28_weighted.parquet")
    pct = pd.read_parquet(args.processed / "news_emotion_features.parquet",
                          columns=["news_id", *[f"go_pct_{e}" for e in PCT_GROUP]])
    if raw["news_id"].tolist() != pct["news_id"].tolist():
        raise ValueError("raw and percentile tables are not aligned")
    base = pd.read_parquet(args.processed / "news_emotion_weighted.parquet")["dominant_emotion"]
    shares = threshold_corpus_shares(raw, pct)
    TABLES.mkdir(parents=True, exist_ok=True)
    shares.to_csv(TABLES / "emotion_threshold_neutral_share.csv", index=False)
    meta = {"n_items": len(raw), "thresholds_raw": THRESHOLDS_RAW, "thresholds_pct": THRESHOLDS_PCT,
            "rule": "non-neutral class (6-class group of the top label/emotion) if its score > t, else neutral",
            "primary": "raw_max_sigmoid: max raw GoEmotions sigmoid over the 27 non-neutral labels (weighted)",
            "secondary": "pct_max_percentile: max go_pct_* over the 8 emotions. Percentiles are ranks, so the "
                         "neutral share is largely fixed by construction (about P(max of 8 correlated ranks <= t)); "
                         "this view also cannot produce sadness (none of the 8 emotions maps to it)",
            "pct_group": PCT_GROUP,
            "baseline_argmax_neutral_pct": round(100 * float((base == "neutral").mean()), 2),
            "declared": "2026-10-26, before the first run", "run_info": run_info()}
    (TABLES / "emotion_threshold_neutral_share_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_threshold_panels(shares, "share_neutral", "% of all items labelled neutral",
                          FIGURES / "emotion_threshold_neutral_share.png",
                          "Neutral share on the full corpus depends strongly on the threshold",
                          f"{len(raw):,} MIND items (title/abstract weighted). Argmax 6-class baseline: "
                          f"{meta['baseline_argmax_neutral_pct']:.1f}% neutral. Item-level only, no labels or clicks.")
    print(shares.to_string(index=False))
    print(f"\nargmax 6-class baseline (weighted): {meta['baseline_argmax_neutral_pct']}% neutral")


if __name__ == "__main__":
    main()
