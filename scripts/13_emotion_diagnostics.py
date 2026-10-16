"""Item-level diagnostics of news_emotion_features.parquet: distributions, cross-source and title-vs-abstract
correlations, one heatmap. No click data.

  python scripts/13_emotion_diagnostics.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import CROSS_SOURCE_PAIRS, FIGURES, LOW_MASS_THRESHOLD, NRC_EMOTIONS, PROCESSED, RESULTS, VAD_DIMS
from src.diagnostics import correlation_block, feature_distributions, pair_correlations, plot_correlation_block
from src.emotion_features import GO_EMOTIONS

TABLES = RESULTS / "tables"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    df = pd.read_parquet(args.processed / "news_emotion_features.parquet")
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c not in ("news_id", "has_abstract")]
    dist = feature_distributions(df, feature_cols)
    dist.round(4).to_csv(TABLES / "emotion_feature_distributions.csv")

    cross = pair_correlations(df, CROSS_SOURCE_PAIRS)
    cross.round(4).to_csv(TABLES / "emotion_cross_source_correlations.csv", index=False)

    ta_pairs = [(f"go_title_{s}", f"go_abs_{s}", "has_abstract") for s in [*GO_EMOTIONS, "neutral", "emotion_mass"]]
    ta = pair_correlations(df, ta_pairs)
    ta.round(4).to_csv(TABLES / "emotion_title_vs_abstract.csv", index=False)

    low = {v: round(100 * float((df[c].dropna() < LOW_MASS_THRESHOLD).mean()), 2)
           for v, c in [("weighted", "go_emotion_mass"), ("title", "go_title_emotion_mass"),
                        ("abstract", "go_abs_emotion_mass")]}
    (TABLES / "emotion_low_mass.json").write_text(json.dumps({"threshold": LOW_MASS_THRESHOLD, "pct_items": low},
                                                             indent=2), encoding="utf-8")

    rows = [f"go_{e}" for e in GO_EMOTIONS] + ["go_neutral", "go_emotionality"] + [f"go_vad_{d}" for d in VAD_DIMS]
    cols = ([f"nrc_{e}" for e in NRC_EMOTIONS] + ["nrc_positive", "nrc_negative", "nrc_coverage"]
            + [f"vad_{d}" for d in VAD_DIMS])
    hits = df[df["nrc_has_hits"] == 1]
    block = correlation_block(hits, rows, cols)
    block.round(4).to_csv(TABLES / "emotion_cross_source_spearman.csv")
    plot_correlation_block(
        block, FIGURES / "emotion_cross_source_spearman.png",
        "GoEmotions and NRC lexicons agree moderately on fear and valence, weakly elsewhere",
        f"Spearman correlation, {len(hits):,} MIND news items with at least one EmoLex emotion word "
        f"(title + abstract weighted 1/3 : 2/3). Outlined: expected matches.",
        highlight=[(a, b) for a, b, _ in CROSS_SOURCE_PAIRS])

    pd.set_option("display.width", 160)
    print("Distributions (selected):")
    print(dist.loc[rows + cols + ["go_emotion_mass"], ["mean", "std", "50%", "max", "pct_zero"]].round(3).to_string())
    print("\nCross-source pairs:\n" + cross.round(3).to_string(index=False))
    print("\nTitle vs abstract (items with abstract):\n" + ta.round(3).to_string(index=False))
    print(f"\n% items with go_emotion_mass < {LOW_MASS_THRESHOLD}: {low}")
    print(f"\nSaved tables to {TABLES} and figure to {FIGURES / 'emotion_cross_source_spearman.png'}")


if __name__ == "__main__":
    main()
