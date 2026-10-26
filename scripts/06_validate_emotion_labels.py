"""Compare your 6-class hand labels with each emotion source: kappa (+95% CI), accuracy, macro-F1, confusion matrices.
Sources: GoEmotions combined text, GoEmotions weighted title/abstract, NRC EmoLex mapped to 6 classes, plus an
all-neutral reference. Runs only once every row of annotation/manual_sample.csv has a label.

  python scripts/06_validate_emotion_labels.py
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src.annotation import agreement, compare_sources, read_sheet, sheet_problems, vader_correlation
from src.config import ANNOTATION, NRC_SIX_MAP, PROCESSED, RESULTS, SEED
from src.lexicon import nrc_to_six
from src.provenance import run_info


def load_predictions(processed, news_ids):
    ids = pd.Index(news_ids)
    preds = {}
    for name, f in [("goemotions_combined", "news_emotion.parquet"),
                    ("goemotions_weighted", "news_emotion_weighted.parquet")]:
        t = pd.read_parquet(processed / f).set_index("news_id").reindex(ids)
        if t["dominant_emotion"].isna().any():
            raise SystemExit(f"some sampled news ids are missing from {f}")
        preds[name] = t
    nrc = pd.read_parquet(processed / "news_emotion_nrc_features.parquet").set_index("news_id").reindex(ids)
    if nrc["nrc_has_hits"].isna().any():
        raise SystemExit("some sampled news ids are missing from news_emotion_nrc_features.parquet")
    labels, tied = nrc_to_six(nrc, SEED)
    return preds, nrc, labels, tied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path, default=ANNOTATION / "manual_sample.csv")
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "tables")
    args = ap.parse_args()

    if not args.sheet.exists():
        raise SystemExit(f"{args.sheet} not found. Run scripts/05_make_annotation_sample.py first.")
    sheet = read_sheet(args.sheet)
    problems = sheet_problems(sheet)
    if problems:
        print("Fix the sheet first:")
        for p in problems:
            print(" -", p)
        sys.exit(1)

    preds, nrc, nrc_labels, nrc_tied = load_predictions(args.processed, sheet["news_id"])
    y = sheet["your_label"].to_numpy()
    predictions = {"goemotions_combined": preds["goemotions_combined"]["dominant_emotion"].to_numpy(),
                   "goemotions_weighted": preds["goemotions_weighted"]["dominant_emotion"].to_numpy(),
                   "nrc_emolex_6class": nrc_labels.to_numpy(),
                   "reference_all_neutral": ["neutral"] * len(y)}
    summary, results = compare_sources(y, predictions)

    no_hits = nrc["nrc_has_hits"].to_numpy() == 0
    nrc_info = {"tie_rate_pct": round(100 * float(nrc_tied.mean()), 2), "n_ties": int(nrc_tied.sum()),
                "pct_no_hits_to_neutral": round(100 * float(no_hits.mean()), 2),
                "pct_trust_anticipation_only_to_neutral": round(
                    100 * float(((nrc_labels == "neutral").to_numpy() & ~no_hits).mean()), 2),
                "mapping": NRC_SIX_MAP, "tie_break": f"seeded random choice (seed {SEED})"}

    pd.set_option("display.width", 160)
    print(f"Your labels: {pd.Series(y).value_counts().to_dict()}\n")
    print(summary.round(3).to_string(index=False))
    print(f"\nNRC mapping: ties {nrc_info['n_ties']} ({nrc_info['tie_rate_pct']}%), no hits -> neutral "
          f"{nrc_info['pct_no_hits_to_neutral']}%, trust/anticipation only -> neutral "
          f"{nrc_info['pct_trust_anticipation_only_to_neutral']}%")
    for name in predictions:
        if name.startswith("reference"):
            continue
        print(f"\n== {name}: confusion matrix (rows = your label, columns = source)")
        print(results[name]["confusion"].to_string())
        print(results[name]["report"])

    texts = (sheet["title"] + ". " + sheet["abstract"]).str.strip(". ")
    vader = {}
    for name in ("goemotions_combined", "goemotions_weighted"):
        t = preds[name]
        vader[name] = vader_correlation(texts, t["joy"] - t[["sadness", "anger", "fear"]].sum(axis=1))
    print("VADER vs (joy - sadness - anger - fear):", {k: "n/a" if v is None else round(v, 3) for k, v in vader.items()})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined = results["goemotions_combined"]  # original output, unchanged format
    legacy = {k: v for k, v in combined.items() if k not in ("confusion", "report")}
    legacy["confusion"] = combined["confusion"].to_dict()
    legacy["vader_correlation"] = vader["goemotions_combined"]
    (args.out_dir / "emotion_validation.json").write_text(json.dumps(legacy, indent=2), encoding="utf-8")
    summary.to_csv(args.out_dir / "emotion_validation_sources.csv", index=False)
    detail = {name: {**{k: v for k, v in r.items() if k not in ("confusion", "report")},
                     "confusion": r["confusion"].to_dict()} for name, r in results.items()}
    out = {"n_labelled": int(len(y)), "label_counts": pd.Series(y).value_counts().to_dict(), "sources": detail,
           "nrc_mapping": nrc_info, "vader_correlation": vader, "run_info": run_info()}
    (args.out_dir / "emotion_validation_sources.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.out_dir}: emotion_validation_sources.csv/.json (+ emotion_validation.json, combined)")


if __name__ == "__main__":
    main()
