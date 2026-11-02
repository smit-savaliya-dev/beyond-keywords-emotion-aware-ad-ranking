"""Mean dev AUC with a 95% CI per model, from the saved per-impression results. Reads no impressions and scores nothing.
Run by hand after a --final-dev run; it is not a pipeline step, because it reads dev result files.

  python scripts/53_results_model_ladder.py
"""
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import RESULTS, TABLES
from src.results_tables import ladder_table


def main():
    t = ladder_table(RESULTS / "per_impression")
    t.to_csv(TABLES / "model_ladder.csv", index=False)
    pd.set_option("display.width", 150)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
