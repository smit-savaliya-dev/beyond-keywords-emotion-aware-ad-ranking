"""EDA figures: six individual PNGs + overview in results/figures/diagnostics/."""
import _bootstrap  # noqa: F401
from src.config import DIAGNOSTICS, PROCESSED
from src.eda import run

if __name__ == "__main__":
    run(PROCESSED, DIAGNOSTICS)
