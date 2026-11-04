"""Prove that a change did not alter results: --take stores a manifest and copies, --check compares (floats within 1e-9).

  python scripts/snapshot_results.py --take [--tag NAME]
  python scripts/snapshot_results.py --check [--tag NAME]

The default tag is pre_cleanup. Tables, frozen settings and processed data are compared with a tolerance;
scores, per-impression files, boosters and figures are compared byte for byte (hash manifest NAME_hashes.json).
"""
import argparse
import json
import sys

import _bootstrap  # noqa: F401
from src.config import FIGURES, FROZEN_DIR, PROCESSED, RESULTS, ROOT, SNAPSHOT_COPY, SNAPSHOT_MANIFEST, TABLES
from src.snapshot import compare, compare_hashes, hash_files, take

ROOTS = [("tables", TABLES), ("frozen", FROZEN_DIR), ("processed", PROCESSED)]
HASH_ROOTS = [("scores", RESULTS / "scores"), ("per_impression", RESULTS / "per_impression"),
              ("ltr_models", PROCESSED / "ltr_models"), ("figures", FIGURES)]


def paths_for(tag):
    """Copy folder, manifest and hash manifest of a snapshot; the default tag keeps the original locations."""
    if tag == "pre_cleanup":
        copy, manifest = SNAPSHOT_COPY, SNAPSHOT_MANIFEST
    else:
        copy, manifest = SNAPSHOT_COPY.with_name(tag), SNAPSHOT_MANIFEST.with_name(f"{tag}_snapshot.json")
    return copy, manifest, manifest.with_name(f"{tag}_hashes.json")


def main():
    """Take a snapshot (--take) or check the results against one (--check)."""
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--take", action="store_true")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--tag", default="pre_cleanup")
    args = ap.parse_args()
    copy_dir, manifest_path, hashes_path = paths_for(args.tag)
    if args.take:
        manifest = take(ROOTS, copy_dir)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"[ok] {len(manifest)} files -> {manifest_path.relative_to(ROOT)} (copies in {copy_dir.relative_to(ROOT)})")
        if args.tag != "pre_cleanup":
            hashes = hash_files(HASH_ROOTS)
            hashes_path.write_text(json.dumps(hashes, indent=1), encoding="utf-8")
            print(f"[ok] {len(hashes)} byte-compared files -> {hashes_path.relative_to(ROOT)}")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = compare(manifest, copy_dir, ROOTS)
    for p in problems:
        print(" -", p)
    print(f"{len(manifest) - len(problems)} of {len(manifest)} snapshot files identical")
    if hashes_path.exists():
        hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        bad = compare_hashes(hashes, HASH_ROOTS)
        for p in bad:
            print(" -", p)
        print(f"{len(hashes) - len(bad)} of {len(hashes)} byte-compared files identical")
        problems += bad
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
