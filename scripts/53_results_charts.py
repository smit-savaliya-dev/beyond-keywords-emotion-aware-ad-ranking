"""The results charts for the supervisor (results/figures/results/), from saved tables only. Reads no impression data.

  python scripts/53_results_charts.py
"""
import _bootstrap  # noqa: F401
from src.config import RESULTS_FIGS, TABLES
from src.results_charts import build_charts

if __name__ == "__main__":
    for name, caption in build_charts(TABLES, RESULTS_FIGS).items():
        print(f"{name}: {caption}")
