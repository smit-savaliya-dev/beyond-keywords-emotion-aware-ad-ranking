"""Stance-excluded labelling variants (D38): v2b ignores approval and disapproval, v2c also realization.

Reads:  news_emotion_raw28_v2.parquet and emotion_v2_choice.json (no GPU).
Writes: news_emotion_labels_variants.parquet, results/tables/emotion_label_variants_grid.csv and its meta file.
Reporting labels only: soft model features do not change. The variant is chosen on the SemEval trial split (D47).
Run:    python scripts/02_09_label_variants.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import LABEL_VARIANTS, PROCESSED, TABLES, V2_T_GRID
from src.emotion_v2 import class_shares, intensity_table, threshold_labels
from src.provenance import run_info


def main():
    """Label the corpus under each variant and tabulate the shares."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    p = args.processed
    raw = pd.read_parquet(p / "news_emotion_raw28_v2.parquet")
    t_main = float(json.loads((TABLES / "emotion_v2_choice.json").read_text(encoding="utf-8"))["chosen_t"])

    labels = pd.DataFrame({"news_id": raw["news_id"].to_numpy()})
    rows, tops = [], {}
    for name, exclude in LABEL_VARIANTS.items():
        table = intensity_table(raw, exclude)
        for t in V2_T_GRID:
            rows.append({"variant": name, "t": t, **class_shares(threshold_labels(table, t))})
        main_labels = threshold_labels(table, t_main)
        emotional = table[main_labels != "neutral"]
        mix = emotional.groupby("group")["top_label"].value_counts(normalize=True).mul(100).round(1)
        tops[name] = {c: mix[c].head(3).to_dict() for c in mix.index.levels[0]}
        labels[f"intensity_{name}"] = table["intensity"].to_numpy()
        labels[f"group_{name}"] = table["group"].to_numpy()
        labels[f"label_{name}"] = main_labels
    grid = pd.DataFrame(rows)

    labels.to_parquet(p / "news_emotion_labels_variants.parquet", index=False)
    TABLES.mkdir(parents=True, exist_ok=True)
    grid.round(4).to_csv(TABLES / "emotion_label_variants_grid.csv", index=False)
    meta = {"rule": "D38", "variants_excluded_labels": LABEL_VARIANTS, "weight": "title 0.5, abstract 0.5 (same as v2)",
            "t_for_comparison": t_main, "use": "reporting and hand-label validation only, never a model feature",
            "top_raw_labels_per_class_pct": tops, "run_info": run_info()}
    (TABLES / "emotion_label_variants_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.set_option("display.width", 150)
    print(grid.pivot(index="variant", columns="t", values="pct_neutral").round(1).to_string())
    print("\n" + grid[grid["t"] == t_main].round(1).to_string(index=False))
    print("\n" + json.dumps(tops, indent=1))


if __name__ == "__main__":
    main()
