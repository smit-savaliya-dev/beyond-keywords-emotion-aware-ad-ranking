"""Hand-label the annotation sheet in the terminal. Shows title + abstract only (never model output).
Keys: 1 joy, 2 sadness, 3 anger, 4 fear, 5 surprise, 6 neutral, s skip, b back, q quit. Saves after every answer,
backs the sheet up before the first change of each session, resumes where you stopped.
Guidelines: docs/labelling_guidelines.md

  python scripts/label_tool.py                    # label annotation/manual_sample.csv
  python scripts/label_tool.py --make-relabel     # after finishing: create the 30-item blind re-label sheet
  python scripts/label_tool.py --relabel          # at least a day later: label annotation/relabel_sample.csv
"""
import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import LABEL_BACKUPS, LABEL_SHEET, RELABEL_MIN_HOURS, RELABEL_N, RELABEL_SHEET
from src.labelling import hours_since_created, make_relabel_sample, run_session


def read_key():
    try:
        import msvcrt
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):  # arrow / function keys send two codes
            msvcrt.getwch()
            return ""
        return key
    except ImportError:
        return input("> ")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--make-relabel", action="store_true")
    mode.add_argument("--relabel", action="store_true")
    ap.add_argument("--sheet", type=Path, default=LABEL_SHEET)
    ap.add_argument("--relabel-sheet", type=Path, default=RELABEL_SHEET)
    args = ap.parse_args()

    if args.make_relabel:
        sub = make_relabel_sample(args.sheet, args.relabel_sheet, RELABEL_N)
        print(f"[ok] {len(sub)} items -> {args.relabel_sheet}. Label them with --relabel in {RELABEL_MIN_HOURS} hours "
              "or later, without looking at your first labels.")
        return
    path = args.relabel_sheet if args.relabel else args.sheet
    if not path.exists():
        sys.exit(f"{path} not found" + (" (create it with --make-relabel)" if args.relabel else ""))
    if args.relabel:
        hours = hours_since_created(path)
        if hours < RELABEL_MIN_HOURS:
            answer = input(f"The re-label sheet was created {hours:.1f} h ago (recommended: {RELABEL_MIN_HOURS} h). "
                           "Continue anyway? [y/N] ")
            if answer.strip().lower() != "y":
                return
    res = run_session(path, read_key, backup_dir=LABEL_BACKUPS)
    print(f"\n{res['labelled_total']}/{res['of']} labelled ({res['labelled_now']} this session)."
          + (f" Backup: {res['backup']}" if res["backup"] else ""))


if __name__ == "__main__":
    main()
