"""Step 2: parse MIND-small into clean parquet files + a data summary.

Run from the repo root:  python scripts/02_prepare_mind.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED, RAW  # noqa: E402
from src.data import build_all, save_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=PROCESSED)
    args = ap.parse_args()

    news, beh, imps, summary = build_all(args.raw)
    save_all(news, beh, imps, summary, args.out)
    print(json.dumps(summary, indent=2))

    problems = []
    if summary["missing_candidate_ids"]:
        problems.append("some candidate IDs are not in news.parquet")
    if summary["missing_history_ids"]:
        problems.append("some history IDs are not in news.parquet")
    for part, p in summary["parts"].items():
        if p["impressions_without_click"] or p["impressions_without_nonclick"]:
            problems.append(f"{part}: impressions with only one label class")
    print("\nCHECKS:", "all passed" if not problems else "; ".join(problems))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
