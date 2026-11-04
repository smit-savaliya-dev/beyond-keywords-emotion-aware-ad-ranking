"""M3 late fusion: emotion score variants, per-impression min-max and the checks of D54.

Inputs: M2 scores, user histories and emotion vectors. Outputs: fused scores for a grid of variants and alphas.
"""
import numpy as np
import pandas as pd

from .baselines import unique_profiles
from .config import M3_VARIANTS, METRIC_COLUMNS, PROCESSED, SIGNAL_ALPHA, V2_MAIN_MEASURE
from .congruence import congruence_features, history_index_lists, user_profiles
from .stats import holm, paired_test


def minmax_within(groups, x):
    """Min-max within each group; a constant group becomes 0 (D53)."""
    s = pd.Series(np.asarray(x, dtype=np.float64))
    g = s.groupby(np.asarray(groups))
    low, high = g.transform("min").to_numpy(), g.transform("max").to_numpy()
    span = high - low
    return np.where(span > 0, (s.to_numpy() - low) / np.where(span > 0, span, 1.0), 0.0)


def fuse(base_norm, emotion_norm, alpha):
    """alpha * base + (1 - alpha) * emotion, both already min-max scaled within the impression."""
    return alpha * base_norm + (1.0 - alpha) * emotion_norm


def fit_profile_pool(item_index, vectors, processed=PROCESSED):
    """FIT profiles with at least one known history item, and their users: the source of the K other users (D52)."""
    beh = pd.read_parquet(processed / "behaviors.parquet", columns=["user_id", "part", "history"], filters=[("part", "==", "fit")])
    if set(beh["part"].astype(str)) != {"fit"}:
        raise ValueError("behaviors of other parts loaded")
    codes, histories = unique_profiles(beh)
    users = beh["user_id"].astype(str).groupby(codes).first().to_numpy()
    profiles, used = user_profiles(history_index_lists(histories, item_index), vectors, vectors.mean(axis=0))
    return profiles[used > 0], users[used > 0]


def emotion_scores(groups, item_idx, histories, vectors, other=None):
    """D52 variants per candidate. other: mean cosine with K other users; without it, user_specific is left out."""
    cong, _ = congruence_features(groups, item_idx, histories, {"pct": vectors}, V2_MAIN_MEASURE)
    raw = cong["cong_pct_cos"].astype(np.float64)
    out = {"cosine": raw, "personal": cong["cong_pct_cos_personal"].astype(np.float64), "mood_repair": 1.0 - raw}
    if other is not None:
        out["user_specific"] = raw - other
    return out


def grid_configs(alphas, variants=M3_VARIANTS):
    """Variant order, then alpha descending: pick_best ties go to the primary variant and the alpha closest to M2 (D53)."""
    return [{"variant": v, "alpha": a} for v in variants for a in sorted(alphas, reverse=True)]


def same_metrics(per_a, per_b, metrics=METRIC_COLUMNS):
    """True if both per-impression tables are identical on every metric (D54a). -> (identical, largest absolute difference)."""
    m = per_a.merge(per_b, on="imp_key", suffixes=("_a", "_b"), validate="one_to_one")
    if len(m) != len(per_a) or len(m) != len(per_b):
        return False, np.inf
    a, b = m[[f"{k}_a" for k in metrics]].to_numpy(), m[[f"{k}_b" for k in metrics]].to_numpy()
    return bool(np.array_equal(a, b, equal_nan=True)), float(np.nanmax(np.abs(a - b)))


def shuffled_control(per_m2, per_shuffled, alpha_level=SIGNAL_ALPHA):
    """D54c. per_shuffled: {alpha: per-impression table with the emotion score permuted}. Fails if any alpha beats M2 (Holm)."""
    rows = []
    for alpha, per in per_shuffled.items():
        m = per_m2.merge(per, on="imp_key", suffixes=("_m2", "_s"), validate="one_to_one")
        rows.append({"alpha": alpha, **paired_test(m["auc_m2"], m["auc_s"])})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"].to_numpy())
    t["beats_m2"] = (t["mean_diff"] > 0) & (t["p_holm"] < alpha_level)
    return t, not bool(t["beats_m2"].any())

