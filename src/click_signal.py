"""Descriptive click-signal tests: within-impression clicked-minus-unclicked feature differences."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_1samp, wilcoxon


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


def holm(pvalues):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


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

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]  # validated categorical slots 1-3 + shapes


def _dots(ax, y, t, color, marker, label):
    sig = t["significant_holm"].to_numpy()
    ax.hlines(y, t["dz_ci_low"], t["dz_ci_high"], color=color, linewidth=1.2, zorder=2)
    ax.scatter(t["dz"][sig], y[sig], s=34, marker=marker, color=color, edgecolor=SURFACE, linewidth=1.2,
               zorder=3, label=label)
    ax.scatter(t["dz"][~sig], y[~sig], s=34, marker=marker, facecolor=SURFACE, edgecolor=color,
               linewidth=1.4, zorder=3)


def _style(ax):
    ax.axvline(0, color=INK_2, linewidth=0.9, zorder=1)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0, colors=INK_2, labelsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("effect size dz (mean / sd of clicked − unclicked, per impression)", fontsize=8.5, color=INK_2)


def plot_signal(core, variants, path, title, subtitle, sources, variant_labels,
                variant_title="Text variants (candidates with an abstract)"):
    """Left: dz with 95% CI per feature, coloured by source. Right: dz per emotion for each text variant.
    sources: {label: [features]}; variants: rows with columns emotion, variant, dz, CI, significant_holm."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 8.2), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    order = [f for feats in sources.values() for f in feats][::-1]
    t = core.set_index("feature").loc[order].reset_index()
    y = np.arange(len(t))
    for (label, feats), (color, marker) in zip(sources.items(), SERIES):
        m = t["feature"].isin(feats).to_numpy()
        _dots(a1, y[m], t[m], color, marker, label)
    a1.set_yticks(y, t["feature"])
    a1.set_title("Weighted features (all fit impressions)", loc="left", fontsize=10, color=INK, pad=26)
    _style(a1)
    a1.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)

    emotions = list(dict.fromkeys(variants["emotion"]))[::-1]
    base = np.arange(len(emotions))
    for k, (vname, (color, marker)) in enumerate(zip(variant_labels, SERIES)):
        v = variants[variants["variant"] == vname].set_index("emotion").loc[emotions].reset_index()
        _dots(a2, base + (1 - k) * 0.22, v, color, marker, variant_labels[vname])
    a2.set_yticks(base, emotions)
    a2.set_title(variant_title, loc="left", fontsize=10, color=INK, pad=26)
    _style(a2)
    a2.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)

    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.935, subtitle, fontsize=8.5, color=INK_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
