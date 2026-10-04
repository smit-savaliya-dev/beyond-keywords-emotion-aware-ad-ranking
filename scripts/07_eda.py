"""EDA figures: six individual PNGs + overview in results/figures/."""
import _bootstrap  # noqa: F401
from src.config import FIGURES, PROCESSED
from src.eda import run

if __name__ == "__main__":
    run(PROCESSED, FIGURES)
