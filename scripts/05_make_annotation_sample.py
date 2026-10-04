"""Write the sheet you label by hand. The model's predictions are deliberately NOT in it."""
import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src.annotation import build_sheet, make_sample
from src.config import ANNOTATION, PROCESSED
from src.emotion import SIX_CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--out", type=Path, default=ANNOTATION / "manual_sample.csv")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists. Delete it only if you want a new sample "
                         "(labels already entered would be lost).")
    news = pd.read_parquet(PROCESSED / "news.parquet", columns=["news_id", "title", "abstract"])
    sheet = build_sheet(make_sample(news, args.n))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[ok] {len(sheet)} headlines -> {args.out}")
    print("Fill your_label with one of:", " / ".join(SIX_CLASSES))
    print("Label what the text conveys, before looking at any model output. Save as CSV UTF-8.")


if __name__ == "__main__":
    main()
