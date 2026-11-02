"""Summary tables built from saved per-impression results."""
import numpy as np
import pandas as pd

from .config import MODEL_LADDER
from .stats import paired_test


def ladder_table(per_dir, part="dev"):
    """One row per model with a saved result, in ladder order: mean AUC per impression and its t-based 95% CI."""
    rows = []
    for model, label, emotion in MODEL_LADDER:
        path = per_dir / f"{model}__{part}.parquet"
        if path.exists():
            auc = pd.read_parquet(path, columns=["auc"])["auc"].to_numpy()
            r = paired_test(np.zeros(len(auc)), auc)
            rows.append({"model": model, "label": label, "emotion": emotion, "part": part, "n": r["n"], "auc": r["mean_b"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    if not rows:
        raise SystemExit(f"no saved {part} results in {per_dir}")
    return pd.DataFrame(rows)
