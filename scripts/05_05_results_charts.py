"""The results charts for the supervisor, from saved tables only. Reads no impression data.

Reads:  results/tables/model_ladder.csv, m2m3_comparison.csv, m4m5_comparison.csv, m3_fusion_grid_valid.csv, m3_fusion_meta.json,
        m3_where_it_helps_dev.csv, semeval2007_validation.csv.
Writes: results/figures/results/*.png (six charts) and results/tables/results_chart_captions.json.
Run:    python scripts/05_05_results_charts.py
"""
import _bootstrap  # noqa: F401
from src.config import RESULTS_FIGS, TABLES
from src.results_charts import build_charts

if __name__ == "__main__":
    for name, caption in build_charts(TABLES, RESULTS_FIGS).items():
        print(f"{name}: {caption}")
