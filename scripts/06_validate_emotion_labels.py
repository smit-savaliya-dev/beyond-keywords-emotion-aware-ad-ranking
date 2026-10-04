"""Compare your hand labels with the model: kappa (+95% CI), accuracy, macro-F1, confusion matrix, VADER check."""
import json
import sys

import pandas as pd

import _bootstrap  # noqa: F401
from src.annotation import agreement, read_sheet, sheet_problems, vader_correlation
from src.config import ANNOTATION, PROCESSED, RESULTS
from src.emotion import SIX_CLASSES


def main():
    path = ANNOTATION / "manual_sample.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/05_make_annotation_sample.py first.")
    sheet = read_sheet(path)
    problems = sheet_problems(sheet)
    if problems:
        print("Fix the sheet first:")
        for p in problems:
            print(" -", p)
        sys.exit(1)

    emo = pd.read_parquet(PROCESSED / "news_emotion.parquet")
    merged = sheet.merge(emo, on="news_id", how="left", validate="one_to_one")
    if merged["dominant_emotion"].isna().any():
        raise SystemExit("some sampled news ids are missing from news_emotion.parquet")

    res = agreement(merged["your_label"], merged["dominant_emotion"])
    lo, hi = res["kappa_ci95"]
    print(f"n = {res['n']}   accuracy = {res['accuracy']:.3f}   macro-F1 = {res['macro_f1']:.3f}")
    print(f"Cohen's kappa = {res['kappa']:.3f}  (95% CI {lo:.3f} to {hi:.3f}, {res['kappa_band']})")
    print("\nConfusion matrix (rows = your label, columns = model):")
    print(res["confusion"].to_string())
    print("\n" + res["report"])

    texts = (merged["title"] + ". " + merged["abstract"]).str.strip(". ")
    valence = merged["joy"] - merged[["sadness", "anger", "fear"]].sum(axis=1)
    corr = vader_correlation(texts, valence)
    print("VADER vs (joy - sadness - anger - fear) correlation:", "n/a" if corr is None else f"{corr:.3f}")

    out = {k: v for k, v in res.items() if k not in ("confusion", "report")}
    out["confusion"] = res["confusion"].to_dict()
    out["vader_correlation"] = corr
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS / "tables" / "emotion_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
