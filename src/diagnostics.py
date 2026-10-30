"""Item-level diagnostics of the emotion feature table: distributions and cross-source correlations."""
import numpy as np
import pandas as pd

from .story_charts import INK, INK_2, SURFACE

DIVERGING = ["#184f95", "#2a78d6", "#86b6ef", "#f0efec", "#f0a3a2", "#e34948", "#a8302f"]


def feature_distributions(df, cols):
    """describe() per feature plus % zero, % NaN and the share above 0.5."""
    rows = {}
    for c in cols:
        s = df[c].astype(float)
        d = s.describe()
        rows[c] = {**d.to_dict(), "pct_nan": 100 * float(s.isna().mean()),
                   "pct_zero": 100 * float((s == 0).mean()), "pct_gt_0.5": 100 * float((s > 0.5).mean())}
    return pd.DataFrame(rows).T


def pair_correlations(df, pairs):
    """pairs: (a, b, mask column or None). Pearson + Spearman on rows where both are present (and mask == 1)."""
    out = []
    for a, b, mask in pairs:
        sub = df if mask is None else df[df[mask] == 1]
        sub = sub[[a, b]].dropna()
        out.append({"feature_a": a, "feature_b": b, "rows": mask or "all", "n": len(sub),
                    "pearson": sub[a].corr(sub[b]), "spearman": sub[a].corr(sub[b], method="spearman")})
    return pd.DataFrame(out)


def correlation_block(df, rows, cols, method="spearman"):
    """rows x cols correlation matrix (pairwise complete)."""
    corr = df[rows + cols].corr(method=method)
    return corr.loc[rows, cols]


def plot_correlation_block(block, path, title, subtitle, highlight=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * block.shape[1], 1.6 + 0.48 * block.shape[0]))
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(block.to_numpy(dtype=float), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            v = block.iat[i, j]
            ax.text(j, i, "" if np.isnan(v) else f"{v:.2f}".replace("0.", ".", 1).replace("-.", "−."),
                    ha="center", va="center", fontsize=7.5, color="white" if abs(v) > 0.55 else INK)
    for a, b in highlight:
        if a in block.index and b in block.columns:
            i, j = block.index.get_loc(a), block.columns.get_loc(b)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.8, zorder=5))
    ax.set_xticks(range(block.shape[1]), block.columns, rotation=45, ha="right", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(block.shape[0]), block.index, fontsize=8.5, color=INK_2)
    ax.set_xticks(np.arange(-0.5, block.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, block.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
