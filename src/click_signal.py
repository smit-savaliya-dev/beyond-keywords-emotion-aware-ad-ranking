"""Descriptive click-signal tests: within-impression clicked-minus-unclicked feature differences."""
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_1samp, wilcoxon

from .congruence import history_index_lists
from .stats import holm


class MaskedValues(Mapping):
    """{name: array} restricted to rows[mask], built on demand (avoids copying every feature at once)."""

    def __init__(self, values, mask):
        self.values, self.mask = values, mask

    def __getitem__(self, name):
        return self.values[name][self.mask]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


class ColumnValues(Mapping):
    """Per-candidate feature arrays from a feature table, built on demand (101 features at once would need ~4 GB)."""

    def __init__(self, feats, cols, rows):
        self.feats, self.cols, self.rows = feats, list(cols), rows

    def __getitem__(self, col):
        return self.feats[col].to_numpy(dtype=np.float64)[self.rows]

    def __iter__(self):
        return iter(self.cols)

    def __len__(self):
        return len(self.cols)


def load_part_impressions(processed, part):
    """Candidate rows of one analysis part (fit or valid; dev is refused). Other parts are never read."""
    if part not in ("fit", "valid"):
        raise ValueError(f"analysis scripts may only load fit or valid, not {part!r}")
    imps = pd.read_parquet(processed / "impressions.parquet", columns=["imp_key", "part", "user_id", "news_id", "label"],
                           filters=[("part", "==", part)])
    imps = imps[imps["part"].astype(str) == part]
    if set(imps["part"].astype(str)) != {part}:
        raise ValueError(f"rows of other parts loaded for {part}")
    return imps.reset_index(drop=True)


@dataclass
class PartContext:
    """Candidates of one analysis part mapped to item rows and impression (profile) rows."""
    groups: np.ndarray        # impression row per candidate
    group_users: np.ndarray   # user code per impression row
    labels: np.ndarray        # click label per candidate
    item_idx: np.ndarray      # item row per candidate
    histories: list           # item rows per impression row, oldest -> newest
    n_impressions: int
    imp_keys: pd.Index = None # impression key per impression row


def load_part_context(processed, part, item_index):
    """fit or valid only (dev is refused by the loader). History comes from the behaviors table, never from clicks."""
    imps = load_part_impressions(processed, part)
    groups, imp_keys = pd.factorize(imps["imp_key"].astype(str))
    users = pd.Series(pd.factorize(imps["user_id"].astype(str))[0]).groupby(groups).first().to_numpy()
    idx = item_index.get_indexer(imps["news_id"].astype(str))
    if (idx < 0).any():
        raise ValueError(f"{part} candidates without emotion vectors")
    beh = pd.read_parquet(processed / "behaviors.parquet", columns=["imp_key", "part", "history"],
                          filters=[("part", "==", part)]).set_index("imp_key").reindex(imp_keys)
    if beh["history"].isna().any():
        raise ValueError(f"{part} impressions without a behaviors row")
    return PartContext(groups, users, imps["label"].to_numpy(), idx,
                       history_index_lists(beh["history"].to_numpy(), item_index), len(imp_keys), imp_keys)


def within_group_diffs(groups, labels, values, return_groups=False):
    """Per group: mean(values | label 1) - mean(values | label 0), ignoring NaN values.
    Only groups with at least one non-NaN value in both classes are returned (optionally with their ids)."""
    groups = np.asarray(groups)
    labels = np.asarray(labels).astype(np.int64)
    values = np.asarray(values, dtype=np.float64)
    ok = ~np.isnan(values)
    n = int(groups.max()) + 1 if len(groups) else 0
    key = groups[ok] * 2 + labels[ok]
    sums = np.bincount(key, weights=values[ok], minlength=2 * n)
    counts = np.bincount(key, minlength=2 * n)
    c0, c1 = counts[0::2], counts[1::2]
    both = (c0 > 0) & (c1 > 0)
    diffs = sums[1::2][both] / c1[both] - sums[0::2][both] / c0[both]
    return (diffs, np.flatnonzero(both)) if return_groups else diffs


def signal_test(diffs):
    """Primary: one-sample t-test of the differences against 0, effect size dz = mean / sd.
    Robustness only: Wilcoxon signed-rank (zeros dropped) and matched-pairs rank-biserial; Wilcoxon is biased here
    because the null distribution of clicked-minus-unclicked means is asymmetric for skewed features."""
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else np.nan
    dz = float(d.mean() / sd) if n > 1 and sd > 0 else 0.0
    if n > 1 and sd > 0:
        t = ttest_1samp(d, 0.0)
        t_stat, p = float(t.statistic), float(t.pvalue)
    else:
        t_stat, p = np.nan, 1.0
    nz = d[d != 0]
    p_w = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    ranks = rankdata(np.abs(nz))
    r_rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()) if len(nz) else 0.0
    half = 1.96 * np.sqrt(1 / n + dz ** 2 / (2 * n)) if n > 0 else np.nan
    return {"n_groups": n, "mean_diff": float(d.mean()) if n else np.nan,
            "median_diff": float(np.median(d)) if n else np.nan, "sd_diff": sd, "dz": dz,
            "dz_ci_low": dz - half, "dz_ci_high": dz + half, "t_stat": t_stat, "p": p,
            "pct_positive": 100 * float((nz > 0).mean()) if len(nz) else np.nan,
            "rank_biserial": r_rb, "p_wilcoxon": p_w}


def user_level_test(diffs, group_ids, group_users):
    """Average each user's per-impression differences, then one-sample t-test across users."""
    users = np.asarray(group_users)[group_ids]
    _, inv = np.unique(users, return_inverse=True)
    means = np.bincount(inv, weights=diffs) / np.bincount(inv)
    r = signal_test(means)
    return {"n_users": r["n_groups"], "user_dz": r["dz"], "user_p": r["p"]}


def permute_within_groups(groups, labels, seed):
    """Labels shuffled within each group (same click count per group): a null with no feature-click link."""
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    shuffled = np.lexsort((rng.random(len(groups)), groups))
    by_group = np.argsort(groups, kind="stable")
    out = np.empty_like(labels)
    out[by_group] = labels[shuffled]
    return out


def holm_family(t, alpha=0.05):
    """Recompute Holm and the significance flags across all rows of `t` (one family of tests)."""
    for col in ("p", "p_wilcoxon", "user_p"):
        t[f"{col}_holm"] = holm(t[col].to_numpy())
    t["significant_holm"] = t["p_holm"] < alpha
    t["significant_user"] = (t["user_p_holm"] < alpha) & (np.sign(t["user_dz"]) == np.sign(t["dz"]))
    t["significant_both"] = t["significant_holm"] & t["significant_user"]
    return t


def signal_table(groups, labels, feature_values, alpha=0.05, group_users=None):
    """feature_values: {name: per-row values}. -> one row per feature; Holm across all of them for the t-test
    (primary), Wilcoxon (robustness) and, with group_users (user of each group id), the user-level t-test."""
    rows = []
    for name, v in feature_values.items():
        d, gid = within_group_diffs(groups, labels, v, return_groups=True)
        row = {"feature": name, **signal_test(d)}
        if group_users is not None:
            row.update(user_level_test(d, gid, group_users))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p"].to_numpy())
    out["significant_holm"] = out["p_holm"] < alpha
    out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"].to_numpy())
    if group_users is not None:
        out["user_p_holm"] = holm(out["user_p"].to_numpy())
        out["significant_user"] = (out["user_p_holm"] < alpha) & (np.sign(out["user_dz"]) == np.sign(out["dz"]))
        out["significant_both"] = out["significant_holm"] & out["significant_user"]
    return out
