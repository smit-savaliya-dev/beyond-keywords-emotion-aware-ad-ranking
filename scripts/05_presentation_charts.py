"""Six simple presentation charts for slides, from saved tables only (no training, no impression data, no dev scoring).

Reads:  results/tables/model_ladder.csv, m2m3_comparison.csv, m4m5_comparison.csv, emotion_signal_v2.csv,
        semeval2007_validation.csv, m3_where_it_helps_dev.csv, story_emotion_mix.csv.
Writes: results/figures/presentation/{1_model_scores, 2_emotion_gain, 3_personal_match, 4_emotion_validation, 5_where_it_helps,
        6_neutral_reduction}.png (12 x 6.75 in, 300 dpi). Not a pipeline step: it only presents saved results.
Run:    python scripts/05_presentation_charts.py
"""
import time

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from src.config import M2_NAME, M3_NAME, M4_NAME, M5_NAME, ROOT, TABLES
from src.terminal_summary import ROLE_NAMES

OUT = ROOT / "results" / "figures" / "presentation"
BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#999999"  # blue = without emotion, orange = with emotion, grey = reference
INK = "#1a1a1a"
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "axes.titlesize": 22, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.edgecolor": GREY, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def new(title, subtitle=None):
    """A 12 x 6.75 in figure with a white background, light gridlines on the value axis and no top or right frame."""
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
    ax.set_title(title, pad=34 if subtitle else 18)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=15, color=GREY, va="bottom")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.14)
    return fig, ax


def save(fig, name):
    """Write the PNG; retried because an open image preview can lock the file for a moment on Windows."""
    for attempt in range(5):
        try:
            fig.savefig(OUT / name, dpi=300, facecolor="white")
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2)
    plt.close(fig)
    print("wrote", name)


def dev_auc_row(comparison, a, b):
    t = comparison[(comparison["model_a"] == a) & (comparison["model_b"] == b) & (comparison["part"] == "dev") & (comparison["metric"] == "auc")]
    return t.iloc[0]


def chart_model_scores():
    t = pd.read_csv(TABLES / "model_ladder.csv")
    t = t[t["part"] == "dev"]
    fig, ax = new("Meaning-based models rank clicks best")
    ax.grid(axis="y", color="#e6e6e6")
    x = range(len(t))
    ax.bar(x, t["auc"] - 0.5, bottom=0.5, color=[ORANGE if e else BLUE for e in t["emotion"]], width=0.65)
    for i, v in zip(x, t["auc"]):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=17, fontweight="bold")
    ax.axhline(0.5, color=GREY, linestyle="--", linewidth=2)
    ax.text(-0.45, 0.4965, "random guessing", ha="left", va="top", fontsize=15, color=INK)
    ax.set_xticks(list(x), [lab.replace(" = ", "\n= ") for lab in t["label"]], fontsize=14)
    ax.set_ylim(0.47, 0.70)
    ax.set_ylabel("Ranking score on the final test (AUC)\nbars start at 0.5 = random guessing")
    ax.legend(handles=[Patch(color=BLUE, label="Without emotion"), Patch(color=ORANGE, label="With emotion")], loc="upper left", frameon=False, fontsize=15)
    save(fig, "1_model_scores.png")


def chart_emotion_gain():
    rows = [("Sentence-BERT\n(M3 minus M2)", dev_auc_row(pd.read_csv(TABLES / "m2m3_comparison.csv"), M2_NAME, M3_NAME)),
            ("LightGBM\n(M5 minus M4)", dev_auc_row(pd.read_csv(TABLES / "m4m5_comparison.csv"), M4_NAME, M5_NAME))]
    fig, ax = new("Adding emotion gives a small gain, but not on LightGBM")
    ax.grid(axis="y", color="#e6e6e6")
    for i, (_, r) in enumerate(rows):
        ax.bar(i, r["mean_diff"], color=ORANGE, width=0.5, yerr=[[r["mean_diff"] - r["ci_low"]], [r["ci_high"] - r["mean_diff"]]],
               error_kw={"elinewidth": 2.5, "capsize": 10, "capthick": 2.5, "ecolor": INK})
        verdict = "significant" if r["significant_holm"] else "not significant"
        up = r["mean_diff"] > 0
        ax.text(i, (r["ci_high"] + 0.0003) if up else (r["ci_low"] - 0.0003), f"{r['mean_diff']:+.4f}\n{verdict}", ha="center",
                va="bottom" if up else "top", fontsize=17, fontweight="bold")
    ax.axhline(0, color=INK, linewidth=1.8)
    ax.set_xticks(range(len(rows)), [n for n, _ in rows], fontsize=16)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(-0.0085, 0.0085)
    ax.set_ylabel("Change in ranking score (AUC)\nfinal test, error bars = 95% range")
    save(fig, "2_emotion_gain.png")


def chart_personal_match():
    t = pd.read_csv(TABLES / "emotion_signal_v2.csv")
    t = t[t["feature"].isin(ROLE_NAMES)]
    fig, ax = new("People click news that matches their own emotional history")
    ax.grid(axis="y", color="#e6e6e6")
    width = 0.36
    for j, (part, color, label) in enumerate((("fit", BLUE, "Training days"), ("valid", ORANGE, "Next day"))):
        s = t[t["part"] == part].set_index("feature").loc[list(ROLE_NAMES)]
        xs = [i + (j - 0.5) * width for i in range(len(s))]
        ax.bar(xs, s["dz"], width=width, color=color, label=label, yerr=[s["dz"] - s["dz_ci_low"], s["dz_ci_high"] - s["dz"]],
               error_kw={"elinewidth": 2, "capsize": 6, "capthick": 2, "ecolor": INK})
        for x, v, hi in zip(xs, s["dz"], s["dz_ci_high"]):
            ax.text(x, hi + 0.008, f"{v:.2f}", ha="center", fontsize=16, fontweight="bold")
    ax.set_xticks(range(len(ROLE_NAMES)), ["Plain match", "Personal match\n(minus the average user)", "Strict personal match\n(minus a random user)"], fontsize=14)
    ax.set_ylim(0, 0.34)
    ax.set_ylabel("How much closer clicked news is than unclicked\n(effect size, 0 = no difference)")
    ax.legend(loc="upper right", frameon=False, fontsize=15)
    save(fig, "3_personal_match.png")


def chart_emotion_validation():
    t = pd.read_csv(TABLES / "semeval2007_validation.csv")
    t = t[t["kind"] == "correlation"]
    pick = [(g, "goemotions_group") for g in ("anger", "fear", "joy", "sadness", "surprise")] + [("valence", "go_vad_valence")]
    v = sorted(((g, float(t[(t["gold"] == g) & (t["source"] == s)]["spearman"].iloc[0])) for g, s in pick), key=lambda kv: -kv[1])
    fig, ax = new("Our emotion scores agree with human ratings")
    ax.grid(axis="y", color="#e6e6e6")
    ax.bar([g for g, _ in v], [x for _, x in v], color=BLUE, width=0.62)
    for i, (_, x) in enumerate(v):
        ax.text(i, x + 0.015, f"{x:.2f}", ha="center", fontsize=18, fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.set_xlabel(f"Emotion (valence = positive versus negative feeling); {int(t['n'].iloc[0]):,} news headlines rated by people")
    ax.set_ylabel("Agreement with human ratings\n(correlation, 1 = perfect)")
    ax.tick_params(axis="x", labelsize=16)
    save(fig, "4_emotion_validation.png")


def chart_where_it_helps():
    t = pd.read_csv(TABLES / "m3_where_it_helps_dev.csv")
    t = t[t["breakdown"] == "clicked category"].sort_values("mean_diff")
    fig, ax = new("Emotion helps most for news and music", "Descriptive: gain by category of the clicked news, final test, M3 minus M2")
    ax.grid(axis="x", color="#e6e6e6")
    ax.barh(t["group"], t["mean_diff"], color=[ORANGE if d > 0 else GREY for d in t["mean_diff"]], height=0.62)
    for y, d in enumerate(t["mean_diff"]):
        ax.text(d + (0.0005 if d > 0 else -0.0005), y, f"{d:+.4f}", va="center", ha="left" if d > 0 else "right", fontsize=16, fontweight="bold")
    ax.axvline(0, color=INK, linewidth=1.8)
    ax.set_xlim(-0.0135, 0.0265)
    ax.set_xlabel("Change in ranking score (AUC) when emotion is added to Sentence-BERT")
    ax.tick_params(axis="y", labelsize=16)
    ax.spines["left"].set_visible(False)
    save(fig, "5_where_it_helps.png")


def chart_neutral_reduction():
    t = pd.read_csv(TABLES / "story_emotion_mix.csv", index_col=0)
    rows = [("Before\n(first rule)", t.loc["v1 (1/3 title, argmax)"]), ("After\n(final rule)", t.loc["v2b (v2 without stance labels)"])]
    order = [("neutral", GREY), ("joy", BLUE), ("sadness", ORANGE), ("surprise", "#56B4E9"), ("anger", "#CC79A7"), ("fear", "#F0E442")]
    fig, ax = new(f"Neutral news reduced from {rows[0][1]['neutral']:.0f}% to {rows[1][1]['neutral']:.0f}%")
    ax.grid(axis="y", color="#e6e6e6")
    bottom = [0.0, 0.0]
    for name, color in order:
        vals = [float(r[name]) for _, r in rows]
        ax.bar(range(2), vals, bottom=bottom, color=color, width=0.5, edgecolor="white", linewidth=1.5, label=name)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 3:
                ax.text(i, b + v / 2, f"{name} {v:.0f}%" if name != "neutral" else f"neutral {v:.0f}%", ha="center", va="center", fontsize=15,
                        fontweight="bold", color=INK)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(2), [n for n, _ in rows], fontsize=17)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of all news items (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=14, reverse=True)
    fig.subplots_adjust(right=0.85)
    save(fig, "6_neutral_reduction.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for chart in (chart_model_scores, chart_emotion_gain, chart_personal_match, chart_emotion_validation, chart_where_it_helps, chart_neutral_reduction):
        chart()
