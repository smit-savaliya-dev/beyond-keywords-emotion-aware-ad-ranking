"""Parse MIND-small into data/processed/*.parquet and verify it."""
import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import PROCESSED, RAW
from src.data import build_all, check_official_counts, integrity_problems, save_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=PROCESSED)
    ap.add_argument("--skip-official-check", action="store_true")
    args = ap.parse_args()

    news, beh, imps, summary = build_all(args.raw)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("raw_sha256", "run_info")}, indent=2))

    problems = integrity_problems(summary)
    if not args.skip_official_check:
        problems += check_official_counts(summary)
    if problems:
        print("\nCHECKS FAILED (nothing saved):")
        for p in problems:
            print(" -", p)
        sys.exit(1)

    save_all(news, beh, imps, summary, args.out)
    note = "" if args.skip_official_check else " (counts match official MIND-small)"
    print(f"\nCHECKS: all passed{note}\nSaved to {args.out}")


if __name__ == "__main__":
    main()
