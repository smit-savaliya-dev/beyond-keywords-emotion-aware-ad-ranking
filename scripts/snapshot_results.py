"""Prove that a change did not alter results: --take stores a manifest and copies, --check compares (floats within 1e-9).

  python scripts/snapshot_results.py --take
  python scripts/snapshot_results.py --check
"""
import argparse
import json
import sys

import _bootstrap  # noqa: F401
from src.config import FROZEN_DIR, PROCESSED, ROOT, SNAPSHOT_COPY, SNAPSHOT_MANIFEST, TABLES
from src.snapshot import compare, take

ROOTS = [("tables", TABLES), ("frozen", FROZEN_DIR), ("processed", PROCESSED)]


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--take", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.take:
        manifest = take(ROOTS, SNAPSHOT_COPY)
        SNAPSHOT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"[ok] {len(manifest)} files -> {SNAPSHOT_MANIFEST.relative_to(ROOT)} (copies in {SNAPSHOT_COPY.relative_to(ROOT)})")
        return
    manifest = json.loads(SNAPSHOT_MANIFEST.read_text(encoding="utf-8"))
    problems = compare(manifest, SNAPSHOT_COPY, ROOTS)
    for p in problems:
        print(" -", p)
    print(f"{len(manifest) - len(problems)} of {len(manifest)} snapshot files identical")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
