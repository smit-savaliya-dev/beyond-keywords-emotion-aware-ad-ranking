"""Summary tables built from saved per-impression results."""
import json

import numpy as np
import pandas as pd

from .config import METRIC_COLUMNS, MODEL_LADDER, PROCESSED
from .stats import compare_models, paired_test


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


def merge_part(path, new, part):
    """Replace the rows of one part in a results table, keeping the other parts."""
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old["part"] != part]
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


def compare_pairs(per_dir, pairs, part, processed=PROCESSED):
    """pairs: [(model a, model b)]. -> (per-metric table with Holm, user-level AUC table), each B minus A."""
    expected = json.loads((processed / "data_summary.json").read_text(encoding="utf-8"))["parts"][part]["impressions"]
    per = {}
    for model in {m for pair in pairs for m in pair}:
        path = per_dir / f"{model}__{part}.parquet"
        if not path.exists():
            raise SystemExit(f"{path} missing: run the {model} script first")
        per[model] = pd.read_parquet(path)
        if len(per[model]) != expected:
            raise SystemExit(f"{model} has {len(per[model])} impressions on {part}, expected {expected}")
    comp = pd.concat([compare_models(per[a], per[b], METRIC_COLUMNS, a, b) for a, b in pairs], ignore_index=True).assign(part=part)
    beh = pd.read_parquet(processed / "behaviors.parquet", columns=["imp_key", "part", "user_id"], filters=[("part", "==", part)])
    users = beh.set_index(beh["imp_key"].astype(str))["user_id"].astype(str)
    rows = []
    for a, b in pairs:
        m = per[a].merge(per[b], on="imp_key", suffixes=("_a", "_b"), validate="one_to_one")
        u = m.groupby(m["imp_key"].astype(str).map(users))[["auc_a", "auc_b"]].mean()
        rows.append({"part": part, "model_a": a, "model_b": b, "users": len(u), **paired_test(u["auc_a"], u["auc_b"])})
    return comp, pd.DataFrame(rows)

