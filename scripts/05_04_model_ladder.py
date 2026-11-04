"""Mean dev AUC with a 95% CI for every model that has run, from the saved per-impression results.

Reads:  results/per_impression/*__dev.parquet. Scores nothing and reads no impressions.
Writes: results/tables/model_ladder.csv.
Run by hand after a --final-dev run; it is not a pipeline step, because it reads dev result files.
Run:    python scripts/05_04_model_ladder.py
"""
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import RESULTS, TABLES
from src.results_tables import ladder_table


def main():
    """Build the ladder table from the saved dev results."""
    t = ladder_table(RESULTS / "per_impression")
    t.to_csv(TABLES / "model_ladder.csv", index=False)
    pd.set_option("display.width", 150)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
