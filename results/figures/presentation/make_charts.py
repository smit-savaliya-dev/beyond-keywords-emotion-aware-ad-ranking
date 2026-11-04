"""Three plain-language presentation charts, built from saved tables only (no training, no impression data, no dev scoring).

Reads:  results/tables/model_ladder.csv, m2m3_comparison.csv, m4m5_comparison.csv, semeval2007_validation.csv.
Writes: the three PNGs next to this file. Prints a short M4/M5 summary (also reads m5_feature_importance.csv).
Run:    python results/figures/presentation/make_charts.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[3]
TABLES, OUT = ROOT / "results" / "tables", Path(__file__).resolve().parent
BLUE, ORANGE, GREEN, RED, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#d6452a", "#1a1a1a", "#6b6b6b"
plt.rcParams.update({"font.size": 16, "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def finish(fig, name, caption):
    """Save the chart with a one-line caption; no title, the slide carries it."""
    fig.text(0.02, 0.025, caption, fontsize=14, color=GREY, ha="left", va="bottom")
    fig.savefig(OUT / name, dpi=200, facecolor="white")
    plt.close(fig)


def frame(ax, axis):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=axis, color="#e3e3e3", linewidth=1)
    ax.set_axisbelow(True)


def auc_row(comparison, a, b):
    t = comparison[(comparison["model_a"] == a) & (comparison["model_b"] == b) & (comparison["part"] == "dev") & (comparison["metric"] == "auc")]
    return t.iloc[0]


ladder = pd.read_csv(TABLES / "model_ladder.csv")
ladder = ladder[ladder["part"] == "dev"].sort_values("auc")

# 1. dev AUC per model
fig, ax = plt.subplots(figsize=(13, 7.2))
fig.subplots_adjust(left=0.22, right=0.96, top=0.95, bottom=0.2)
frame(ax, "x")
colors = [ORANGE if e else BLUE for e in ladder["emotion"]]
ax.barh(ladder["label"], ladder["auc"], color=colors, height=0.62)
for y, v in enumerate(ladder["auc"]):
    ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=18, fontweight="bold", color=INK)
ax.axvline(0.5, color=INK, linestyle="--", linewidth=1.8)
ax.text(0.5025, len(ladder) - 0.45, "random guessing", fontsize=16, color=INK, va="bottom")
ax.set_xlim(0.45, 0.72)
ax.set_xlabel("AUC on the final test (higher is better)", fontsize=17)
ax.tick_params(axis="y", labelsize=18)
ax.legend(handles=[Patch(color=BLUE, label="Without emotion"), Patch(color=ORANGE, label="With emotion")], loc="lower right", frameon=False, fontsize=16)
best = ladder.iloc[-1]
finish(fig, "1_how_well_each_model_ranks.png",
       f"Final test, {int(ladder['n'].iloc[0]):,} impressions. {best['label']} is best at {best['auc']:.3f}; the emotion versions are the orange bars.")

# 2. does emotion help
m23 = auc_row(pd.read_csv(TABLES / "m2m3_comparison.csv"), "M2_sbert", "M3_sbert_emotion")
m45 = auc_row(pd.read_csv(TABLES / "m4m5_comparison.csv"), "M4_lightgbm", "M5_lightgbm_emotion")
rows = [("Sentence-BERT\n(M3 minus M2)", m23), ("LightGBM\n(M5 minus M4)", m45)]
fig, ax = plt.subplots(figsize=(13, 7.2))
fig.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.2)
frame(ax, "y")
for i, (name, r) in enumerate(rows):
    color = GREEN if r["mean_diff"] > 0 else RED
    ax.bar(i, r["mean_diff"], width=0.5, color=color, yerr=[[r["mean_diff"] - r["ci_low"]], [r["ci_high"] - r["mean_diff"]]],
           error_kw={"elinewidth": 3, "capsize": 12, "capthick": 3, "ecolor": INK})
    verdict = "significant" if r["significant_holm"] else "not significant"
    if r["significant_holm"]:
        verdict += " (better)" if r["mean_diff"] > 0 else " (worse)"
    top = r["ci_high"] if r["mean_diff"] > 0 else r["ci_low"]
    ax.text(i, top + (0.00025 if r["mean_diff"] > 0 else -0.00025), f"{r['mean_diff']:+.4f}\n{verdict}", ha="center",
            va="bottom" if r["mean_diff"] > 0 else "top", fontsize=19, fontweight="bold", color=INK)
ax.axhline(0, color=INK, linewidth=2)
ax.set_xticks(range(len(rows)), [n for n, _ in rows], fontsize=19)
ax.set_xlim(-0.6, len(rows) - 0.4)
ax.set_ylim(-0.0085, 0.0085)
ax.set_ylabel("Change in AUC when emotion is added", fontsize=17)
finish(fig, "2_does_adding_emotion_help.png",
       f"Final test, with 95% confidence bars. Emotion adds {m23['mean_diff']:+.4f} to Sentence-BERT and {m45['mean_diff']:+.4f} to LightGBM "
       "(Holm-corrected test).")

# 3. emotion scores against humans
sem = pd.read_csv(TABLES / "semeval2007_validation.csv")
sem = sem[sem["kind"] == "correlation"]
pick = [("anger", "goemotions_group"), ("fear", "goemotions_group"), ("joy", "goemotions_group"), ("sadness", "goemotions_group"),
        ("surprise", "goemotions_group"), ("valence", "go_vad_valence")]
vals = [float(sem[(sem["gold"] == g) & (sem["source"] == s)]["spearman"].iloc[0]) for g, s in pick]
names = [g for g, _ in pick]
fig, ax = plt.subplots(figsize=(13, 7.2))
fig.subplots_adjust(left=0.1, right=0.96, top=0.95, bottom=0.2)
frame(ax, "y")
ax.bar(names, vals, color=[BLUE] * 5 + [ORANGE], width=0.62)
for i, v in enumerate(vals):
    ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=20, fontweight="bold", color=INK, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
ax.axhline(0.2, color=INK, linestyle="--", linewidth=1.6)
ax.text(-0.45, 0.80, "dashed line = the minimum we required (0.2)", ha="left", fontsize=15, color=INK)
ax.set_ylim(0, 0.85)
ax.set_ylabel("Agreement with human ratings\n(correlation, 1 = perfect)", fontsize=17)
ax.tick_params(axis="x", labelsize=19)
finish(fig, "3_do_emotion_scores_agree_with_humans.png",
       f"{int(sem['n'].iloc[0]):,} news headlines rated by people (SemEval-2007 test set). Orange = positive-versus-negative feeling (valence).")

# M4 / M5 summary, in plain words
m4, m5 = ladder[ladder["model"] == "M4_lightgbm"].iloc[0], ladder[ladder["model"] == "M5_lightgbm_emotion"].iloc[0]
imp = pd.read_csv(TABLES / "m5_feature_importance.csv")
top_gain = imp.sort_values("gain_rank").head(3)["feature"].tolist()
top_contrib = imp.sort_values("contrib_rank").head(3)["feature"].tolist()
print(f"1. M4 (LightGBM, no emotion) scores AUC {m4['auc']:.4f} on the final test.")
print(f"2. M5 (the same model plus emotion) scores AUC {m5['auc']:.4f}.")
print(f"3. The paired gain is {m45['mean_diff']:+.4f} AUC (95% CI {m45['ci_low']:+.4f} to {m45['ci_high']:+.4f}): emotion made it slightly worse, not better.")
seeds = pd.read_csv(TABLES / "m4m5_seed_variability_valid.csv")
spread = seeds[seeds["model"] == "M4_lightgbm"]["auc"].agg(lambda x: x.max() - x.min())
print(f"4. Significant after Holm correction: {'yes' if m45['significant_holm'] else 'no'}. It is small: M4's own seed-to-seed spread on validation is {spread:.4f}.")
print(f"5. Top 3 features inside M5 by gain: {', '.join(top_gain)}.")
print(f"6. Top 3 by average contribution to the score: {', '.join(top_contrib)}.")
