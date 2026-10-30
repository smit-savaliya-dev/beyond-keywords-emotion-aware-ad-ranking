"""Merge all item-level emotion features into data/processed/news_emotion_features.parquet (+ metadata JSON).
Uses only saved outputs of the scoring and lexicon steps; no model re-run.

  python scripts/23_emotion_merge_features.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import LOW_MASS_THRESHOLD, PROCESSED
from src.emotion_features import GO_COLUMNS, go_percentiles, merge_features, variant_features
from src.provenance import run_info, sha256_file

INPUTS = ["news.parquet", "news_emotion_go_features.parquet", "news_emotion_raw28_title.parquet",
          "news_emotion_raw28_abstract.parquet", "news_emotion_raw28_weighted.parquet",
          "news_emotion_go_vad.parquet", "news_emotion_nrc_features.parquet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    p = args.processed

    news = pd.read_parquet(p / "news.parquet", columns=["news_id", "abstract"])
    ids = news["news_id"].tolist()
    go = pd.read_parquet(p / "news_emotion_go_features.parquet")
    weighted = pd.read_parquet(p / "news_emotion_raw28_weighted.parquet", columns=["news_id", "has_abstract"])
    frames = [
        weighted,
        go,
        go_percentiles(go),
        pd.read_parquet(p / "news_emotion_go_vad.parquet"),
        pd.read_parquet(p / "news_emotion_nrc_features.parquet"),
        variant_features(pd.read_parquet(p / "news_emotion_raw28_title.parquet"), "title", ids),
        variant_features(pd.read_parquet(p / "news_emotion_raw28_abstract.parquet"), "abs", ids),
    ]
    for f in frames[:5]:
        if f["news_id"].tolist() != ids:
            raise ValueError("a weighted/lexicon feature frame does not match news.parquet order")
    table = merge_features(ids, frames)

    if (table["has_abstract"].to_numpy() != (news["abstract"].str.strip() != "").to_numpy()).any():
        raise ValueError("has_abstract disagrees with news.parquet")
    abs_cols = [c for c in table.columns if c.startswith("go_abs_")]
    if table.drop(columns=abs_cols).isna().any().any():
        raise ValueError("unexpected NaN outside go_abs_*")
    if table.loc[table["has_abstract"], abs_cols].isna().any().any() or \
            table.loc[~table["has_abstract"], abs_cols].notna().any().any():
        raise ValueError("go_abs_* must be NaN exactly where there is no abstract")

    mass = table["go_emotion_mass"]
    groups = {
        "go (weighted)": GO_COLUMNS,
        "go_pct (weighted)": [c for c in table.columns if c.startswith("go_pct_")],
        "go_vad": [c for c in table.columns if c.startswith("go_vad_")],
        "nrc": [c for c in table.columns if c.startswith("nrc_")],
        "vad": [c for c in table.columns if c.startswith("vad_")],
        "go_title": [c for c in table.columns if c.startswith("go_title_")],
        "go_abs": abs_cols,
    }
    meta = {
        "file": "news_emotion_features.parquet", "n_items": len(table), "n_columns": table.shape[1],
        "one_row_per": "news_id (order of news.parquet)",
        "column_groups": groups,
        "notes": {
            "go_pct_*": "percentile rank (0, 1] of each go_* emotion across all items (item-level, no clicks). "
                        "Corrects for different label floors on neutral text.",
            "go_dist_*": f"unreliable when go_emotion_mass is small: {100 * float((mass < LOW_MASS_THRESHOLD).mean()):.1f}% "
                         f"of items have mass < {LOW_MASS_THRESHOLD}, where it mostly reflects label floors. "
                         "Weight or filter congruence by go_emotion_mass, or use go_pct_*.",
            "go_emotion_mass": f"sum of 8 independent sigmoids, can exceed 1 by design "
                               f"(max {mass.max():.3f}, {int((mass > 1).sum())} items > 1)",
            "go_title_* / go_abs_*": "same features from title-only / abstract-only raw scores; go_abs_* is NaN "
                                     "without abstract; go_*_pct_* ranked within that variant's scored items",
            "has_abstract": "False -> weighted features equal title-only",
        },
        "inputs_sha256": {f: sha256_file(p / f) for f in INPUTS},
        "sources": {"goemotions": json.loads((p / "news_emotion_weighted_meta.json").read_text(encoding="utf-8")),
                    "lexicons": json.loads((p / "news_emotion_lexicon_meta.json").read_text(encoding="utf-8"))},
        "run_info": run_info(),
    }
    table.to_parquet(p / "news_emotion_features.parquet", index=False)
    (p / "news_emotion_features_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {len(table)} items x {table.shape[1]} columns -> news_emotion_features.parquet")
    for name, cols in groups.items():
        print(f"  {name:18s} {len(cols):3d}")
    print("\n" + "\n".join(f"{k}: {v}" for k, v in meta["notes"].items()))


if __name__ == "__main__":
    main()
