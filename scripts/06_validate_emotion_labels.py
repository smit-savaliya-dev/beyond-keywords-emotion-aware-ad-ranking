"""Compare your 6-class hand labels with each emotion source and the pre-declared threshold views.
Sources: GoEmotions combined text, weighted title/abstract, title-only, NRC EmoLex mapped to 6 classes, all-neutral
reference. Per source: kappa (95% bootstrap CI), accuracy, macro-F1, per-class recall, confusion matrix.
Threshold sensitivity: kappa per pre-declared threshold (a sensitivity analysis on 180 items, NOT a way to pick t).
Intra-rater kappa when annotation/relabel_sample.csv is filled. Runs only once every row of the sheet is labelled.

  python scripts/06_validate_emotion_labels.py
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src.annotation import read_sheet, sheet_problems, vader_correlation
from src.config import FIGURES, LABEL_SHEET, MIN_CLASS_COUNT, NRC_SIX_MAP, PROCESSED, RELABEL_SHEET, RESULTS, SEED
from src.provenance import run_info
from src.validation import plot_threshold_panels, run_validation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path, default=LABEL_SHEET)
    ap.add_argument("--relabel-sheet", type=Path, default=RELABEL_SHEET)
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "tables")
    ap.add_argument("--fig-dir", type=Path, default=FIGURES)
    args = ap.parse_args()

    if not args.sheet.exists():
        raise SystemExit(f"{args.sheet} not found. Run scripts/05_make_annotation_sample.py first.")
    problems = sheet_problems(read_sheet(args.sheet))
    if problems:
        print("Fix the sheet first (python scripts/label_tool.py):")
        for p in problems:
            print(" -", p)
        sys.exit(1)

    v = run_validation(args.sheet, args.processed, args.relabel_sheet)
    summary, per_class, sens = v["summary"], v["per_class"], v["sensitivity"]
    counts = v["label_counts"]
    rare = [c for c, n in counts.items() if n < MIN_CLASS_COUNT]

    pd.set_option("display.width", 170)
    print(f"Your labels (n = {int(counts.sum())}): {counts.to_dict()}")
    if rare:
        print(f"WARNING: fewer than {MIN_CLASS_COUNT} labels for {rare}: their per-class results are unreliable")
    print("\n" + summary.round(3).to_string(index=False))
    print("\nPer-class recall (rows with reliable = False have < "
          f"{MIN_CLASS_COUNT} hand labels):")
    print(per_class.pivot(index="class", columns="source", values="recall").round(3).to_string())
    print("\nThreshold sensitivity (180 items: read the curve, do not pick t by maximum kappa):")
    print(sens.round(3).to_string(index=False))
    intra = v["intra"]
    print("\nIntra-rater:", "not available (re-label sheet missing or incomplete)" if intra is None else
          f"kappa {intra['kappa']:.3f} (95% CI {intra['kappa_ci95'][0]:.3f} to {intra['kappa_ci95'][1]:.3f}), "
          f"n = {intra['n']}")

    sheet = v["sheet"]
    texts = (sheet["title"] + ". " + sheet["abstract"]).str.strip(". ")
    vader = {}
    for name in ("combined", "weighted"):
        t = v["extra"][name]
        vader[name] = vader_correlation(texts, (t["joy"] - t[["sadness", "anger", "fear"]].sum(axis=1)).to_numpy())

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    combined = v["results"]["goemotions_combined"]  # original output, unchanged format
    legacy = {k: val for k, val in combined.items() if k not in ("confusion", "report")}
    legacy["confusion"] = combined["confusion"].to_dict()
    legacy["vader_correlation"] = vader["combined"]
    (out / "emotion_validation.json").write_text(json.dumps(legacy, indent=2), encoding="utf-8")
    summary.to_csv(out / "emotion_validation_sources.csv", index=False)
    per_class.to_csv(out / "emotion_validation_per_class.csv", index=False)
    sens.to_csv(out / "emotion_threshold_sensitivity.csv", index=False)
    detail = {name: {**{k: val for k, val in r.items() if k not in ("confusion", "report")},
                     "confusion": r["confusion"].to_dict()} for name, r in v["results"].items()}
    meta = {"n_labelled": int(counts.sum()), "label_counts": {k: int(n) for k, n in counts.items()},
            "classes_below_min_count": rare, "min_class_count": MIN_CLASS_COUNT, "sources": detail,
            "nrc_mapping": {**v["nrc_info"], "mapping": NRC_SIX_MAP, "tie_break": f"seeded random choice ({SEED})"},
            "intra_rater": None if intra is None else {k: val for k, val in intra.items()
                                                       if k not in ("confusion", "report")},
            "vader_correlation": vader,
            "caution": "180 items: wide kappa CIs and rare classes. Threshold rows are a sensitivity analysis; "
                       "t is not chosen by maximum kappa on these items.",
            "run_info": run_info()}
    (out / "emotion_validation_sources.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    plot_threshold_panels(sens, "kappa", "Cohen's kappa vs your labels", args.fig_dir / "emotion_threshold_kappa.png",
                          "Agreement with hand labels across pre-declared thresholds",
                          f"n = {int(counts.sum())} hand-labelled items; band = 95% bootstrap CI. Small sample: "
                          "read the shape, not the maximum.", ci=("kappa_ci_low", "kappa_ci_high"))
    print(f"\nSaved to {out} and {args.fig_dir / 'emotion_threshold_kappa.png'}")


if __name__ == "__main__":
    main()
