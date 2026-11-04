"""Print a short summary of the results from saved tables. Reads no impression data, so dev is never loaded.

Reads:  results/tables/*.csv and data/processed/data_summary.json.
Writes: nothing (prints to the terminal).
Run:    python scripts/show_results.py
"""
import sys

import _bootstrap  # noqa: F401
from src.config import PROCESSED, TABLES
from src.terminal_summary import build_summary

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(build_summary(TABLES, PROCESSED))
