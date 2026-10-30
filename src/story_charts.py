"""The eight story charts of base-v2, in one academic style. Static PNGs: every chart has a title, labelled axes with
units and a one-sentence caption. Palette validated with the dataviz checker; hollow marker = not significant at both levels.
"""
import textwrap

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
EMOTION_COLORS = {"joy": "#2a78d6", "sadness": "#eb6834", "anger": "#1baf7a", "fear": "#eda100", "surprise": "#e87ba4",
                  "neutral": "#c9c8c0"}
LINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
LINE_MARKERS = ["o", "s", "^", "D", "v", "P"]
PART_STYLE = {"fit": ("#2a78d6", "o", "fit (9-13 Nov)"), "valid": ("#eb6834", "s", "valid (14 Nov, next day)")}
RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DZ_LABEL = "Effect size dz (SD units; 0 = no difference)"


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _frame(ax, grid="both"):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)


def _finish(fig, path, title, caption, height):
    wrapped = "\n".join(textwrap.wrap(caption, 118))
    fig.text(0.01, 1 - 0.14 / height, title, fontsize=12, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.01, 0.12 / height, wrapped, fontsize=8.5, color=INK_2, ha="left", va="bottom")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _new(width, height, top=0.62, bottom=0.95, left=0.1, right=0.97, **kw):
    _style()
    fig, ax = plt.subplots(figsize=(width, height), **kw)
    fig.subplots_adjust(top=1 - top / height, bottom=bottom / height, left=left, right=right)
    return fig, ax


def emotion_mix(mix, path, title, caption):
    """mix: index = version names (top bar first), columns = classes, values = % of items. Stacked bars."""
    height = 2.5 + 0.55 * len(mix)
    fig, ax = _new(8, height, left=0.24, top=0.72)
    y = np.arange(len(mix))[::-1]
    left = np.zeros(len(mix))
    for c in mix.columns:
        vals = mix[c].to_numpy()
        ax.barh(y, vals, left=left, height=0.56, color=EMOTION_COLORS[c], edgecolor=SURFACE, linewidth=1.5, label=c)
        for yi, v, lo in zip(y, vals, left):
            if v >= 4:
                ink = "white" if c in ("joy",) else INK
                ax.text(lo + v / 2, yi, f"{v:.1f}", ha="center", va="center", fontsize=8.5, color=ink)
        left += vals
    ax.set_yticks(y, mix.index, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of all news items (%)")
    _frame(ax, "x")
    ax.legend(frameon=False, ncol=6, loc="lower left", bbox_to_anchor=(-0.01, 1.0), fontsize=8.5, labelcolor=INK,
              handlelength=1.1, columnspacing=1.2)
    _finish(fig, path, title, caption, height)


def neutral_curve(grid, chosen_share, chosen_t, path, title, caption, extra=None):
    """grid: share, t, pct_neutral. One line per title share, the chosen point ringed, neutral limits marked.
    extra: optional (name, frame with t, pct_neutral) drawn as one more line."""
    fig, ax = _new(8, 4.6, left=0.09, top=0.95)
    shares = sorted(grid["share"].unique(), reverse=True)
    for i, s in enumerate(shares):
        g = grid[grid["share"] == s].sort_values("t")
        name = "title only" if s == 1 else f"title {s:.2f} / abstract {1 - s:.2f}"
        ax.plot(g["t"], g["pct_neutral"], color=LINE_COLORS[i], marker=LINE_MARKERS[i], markersize=5.5, linewidth=1.8,
                label=name)
    if extra is not None:
        name, g = extra
        g = g.sort_values("t")
        ax.plot(g["t"], g["pct_neutral"], color=LINE_COLORS[5], marker=LINE_MARKERS[5], markersize=6, linewidth=1.8,
                linestyle="--", label=name)
    pick = grid[(grid["share"] == chosen_share) & (grid["t"] == chosen_t)].iloc[0]
    ax.scatter([chosen_t], [pick["pct_neutral"]], s=170, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=5)
    ax.annotate(f"chosen: title {chosen_share:.1f}, t = {chosen_t}\n{pick['pct_neutral']:.1f}% neutral",
                (chosen_t, pick["pct_neutral"]), textcoords="offset points", xytext=(22, -30), fontsize=8.5, color=INK,
                arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})
    for level, name in ((30, "preferred limit 30%"), (40, "hard limit 40%")):
        ax.axhline(level, color=INK_2, linewidth=0.9)
        ax.text(0.3, level + 0.8, name, ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.set_xscale("log")
    ts = sorted(grid["t"].unique())
    ax.set_xticks(ts, [str(t) for t in ts])
    ax.minorticks_off()
    ax.set_ylim(0, 100)
    ax.set_xlabel("Threshold t on the strongest non-neutral emotion score (log scale)")
    ax.set_ylabel("Items labelled neutral (%)")
    _frame(ax)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(-0.01, 1.0), fontsize=8.5, labelcolor=INK)
    _finish(fig, path, title, caption, 4.6)


def intensity_compare(title_scores, abstract_scores, t, path, title, caption):
    """Share of items whose field-level intensity is at least x, for titles and abstracts of the same items."""
    fig, ax = _new(8, 4.4, left=0.09, top=0.95)
    xs = np.logspace(np.log10(0.005), 0, 120)
    for scores, color, marker, name in ((title_scores, "#2a78d6", "o", "titles"),
                                        (abstract_scores, "#eb6834", "s", "abstracts")):
        s = np.sort(np.asarray(scores))
        y = 100 * (1 - np.searchsorted(s, xs, side="left") / len(s))
        ax.plot(xs, y, color=color, linewidth=2, label=f"{name} (median {np.median(s):.3f})", marker=marker,
                markevery=14, markersize=5)
    ax.axvline(t, color=INK_2, linewidth=0.9)
    ax.text(t * 1.06, 3, f"t = {t}", fontsize=8.5, color=INK_2)
    ax.set_xscale("log")
    ticks = [0.01, 0.03, 0.1, 0.3, 1.0]
    ax.set_xticks(ticks, [str(v) for v in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 100)
    ax.set_xlabel("Emotion intensity x: strongest non-neutral score (0-1, log scale)")
    ax.set_ylabel("Items with intensity of at least x (%)")
    _frame(ax)
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.01, 1.0), fontsize=8.5, labelcolor=INK)
    _finish(fig, path, title, caption, 4.4)


def category_heatmap(table, path, title, caption):
    """table: index = categories, columns = classes, values = % of the category's items."""
    h = 1.75 + 0.36 * len(table)
    fig, ax = _new(8, h, left=0.17, right=0.98, top=0.62, bottom=1.0)
    vals = table.to_numpy(dtype=float)
    ax.imshow(vals, cmap=RAMP, vmin=0, vmax=max(60.0, np.nanmax(vals)), aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                    color="white" if vals[i, j] >= 30 else INK)
    ax.set_xticks(range(vals.shape[1]), table.columns, color=INK)
    ax.set_yticks(range(vals.shape[0]), table.index, color=INK)
    ax.set_xlabel("Emotion label (% of the category's items)")
    ax.set_ylabel("News category")
    ax.set_xticks(np.arange(-0.5, vals.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, vals.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    _finish(fig, path, title, caption, h)


def valence_arousal_map(table, offsets, path, title, caption):
    """table: category, valence, arousal, v_low, v_high, a_low, a_high (centroid and interquartile spread)."""
    fig, ax = _new(8, 5.2, left=0.09, top=0.62)
    color = "#2a78d6"
    ax.errorbar(table["valence"], table["arousal"],
                xerr=[table["valence"] - table["v_low"], table["v_high"] - table["valence"]],
                yerr=[table["arousal"] - table["a_low"], table["a_high"] - table["arousal"]],
                fmt="none", ecolor="#b7d3f6", elinewidth=1.3, zorder=2)
    ax.scatter(table["valence"], table["arousal"], s=46, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    for _, r in table.iterrows():
        dx, dy = offsets.get(r["category"], (6, 5))
        ax.annotate(r["category"], (r["valence"], r["arousal"]), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8.5, color=INK, zorder=4)
    ax.axvline(0, color=INK_2, linewidth=0.9)
    ax.set_xlabel("Valence (-1 negative to +1 positive)")
    ax.set_ylabel("Arousal (-1 calm to +1 excited)")
    _frame(ax)
    _finish(fig, path, title, caption, 5.2)


def _two_part_dots(ax, t, rows, value_labels=False):
    """t: columns row, part, dz, dz_ci_low, dz_ci_high, significant_both. rows: top to bottom."""
    y0 = np.arange(len(rows))[::-1]
    pos = dict(zip(rows, y0))
    for offset, part in ((0.17, "fit"), (-0.17, "valid")):
        color, marker, name = PART_STYLE[part]
        p = t[t["part"] == part]
        y = np.array([pos[r] for r in p["row"]]) + offset
        ax.hlines(y, p["dz_ci_low"], p["dz_ci_high"], color=color, linewidth=1.3, zorder=2)
        sig = p["significant_both"].to_numpy(dtype=bool)
        ax.scatter(p["dz"][sig], y[sig], s=34, marker=marker, color=color, edgecolor=SURFACE, linewidth=1, zorder=3,
                   label=name)
        ax.scatter(p["dz"][~sig], y[~sig], s=34, marker=marker, facecolor=SURFACE, edgecolor=color, linewidth=1.3, zorder=3)
        if value_labels:
            for yi, hi, v in zip(y, p["dz_ci_high"], p["dz"]):
                ax.text(hi + 0.004, yi, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=0.9, zorder=1)
    ax.set_yticks(y0, rows, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(DZ_LABEL)
    _frame(ax, "x")


def clicked_effects(t, flips, path, title, caption):
    """t: row, part, dz, CI, significant_both. flips: rows whose sign differs between fit and valid."""
    rows = t[t["part"] == "fit"].sort_values("dz", ascending=False)["row"].tolist()
    h = 1.9 + 0.3 * len(rows)
    fig, ax = _new(8, h, left=0.27, right=0.9, top=0.72, bottom=1.0)
    _two_part_dots(ax, t, rows)
    y = dict(zip(rows, np.arange(len(rows))[::-1]))
    right = ax.get_xlim()[1]
    for r in flips:
        ax.text(right, y[r], "sign flips", ha="left", va="center", fontsize=8, color=INK_2)
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.01, 1.0), fontsize=8.5, labelcolor=INK)
    _finish(fig, path, title, caption, h)


def congruence_effects(t, path, title, caption):
    """t: row (raw, personal, user-specific), part, dz, CI, significant_both."""
    rows = list(dict.fromkeys(t["row"]))
    fig, ax = _new(8, 3.7, left=0.3, right=0.95, top=0.72)
    _two_part_dots(ax, t, rows, value_labels=True)
    ax.set_xlim(0, max(0.3, float(t["dz_ci_high"].max()) + 0.04))
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.01, 1.0), fontsize=8.5, labelcolor=INK)
    _finish(fig, path, title, caption, 3.7)


def context_effects(main, control, path, title, caption):
    """main and control: row (window), part, dz, CI, significant_both. Two panels on one dz scale."""
    _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(top=1 - 0.95 / 4, bottom=1.0 / 4, left=0.16, right=0.96, wspace=0.42)
    _two_part_dots(a1, main, list(dict.fromkeys(main["row"])), value_labels=True)
    _two_part_dots(a2, control, list(dict.fromkeys(control["row"])), value_labels=True)
    a1.set_xlim(0, 0.2)
    a1.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
    a2.set_xticks([0, 0.1, 0.2])
    a1.set_title("All impressions", loc="left", fontsize=9.5, color=INK)
    a2.set_title("Same size, different age (history >= 10)", loc="left", fontsize=9.5, color=INK)
    a2.set_xlabel("Effect size dz (SD units)")
    a1.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.01, 1.07), fontsize=8.5, labelcolor=INK)
    _finish(fig, path, title, caption, 4)
