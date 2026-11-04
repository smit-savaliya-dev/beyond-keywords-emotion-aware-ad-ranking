"""EDA figures for MIND-small.

Reads:  news, behaviors and impressions (fit part only) and data_summary.json in data/processed/.
Writes: six PNGs, eda_overview.png and eda_stats.json in results/figures/diagnostics/.
Run:    python scripts/01_04_eda.py
"""
import _bootstrap  # noqa: F401
from src.config import DIAGNOSTICS, PROCESSED
from src.eda import run

if __name__ == "__main__":
    run(PROCESSED, DIAGNOSTICS)
