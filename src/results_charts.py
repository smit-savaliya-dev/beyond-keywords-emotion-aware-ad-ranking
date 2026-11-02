"""The results charts for the supervisor: built from saved tables only, captions computed from the same numbers. Max 6 charts."""
import json

import numpy as np
import pandas as pd

from .config import M2_NAME, M3_NAME, METRIC_COLUMNS, RESULT_CHARTS
from .story_charts import INK, INK_2, LINE_COLORS, PART_STYLE, SURFACE, _finish, _frame, _new, alpha_curve, semeval_correlations

METRIC_LABELS = {"auc": "AUC", "mrr": "MRR", "ndcg@5": "nDCG@5", "ndcg@10": "nDCG@10", "p@5": "P@5", "p@10": "P@10", "r@5": "R@5",
                 "r@10": "R@10", "hr@5": "HR@5", "hr@10": "HR@10"}
EMOTION_NAMES = {"anger": "anger", "fear": "fear", "joy": "joy", "sadness": "sadness"}


def _errors(t):
    return [t["mean_diff"] - t["ci_low"], t["ci_high"] - t["mean_diff"]]


def model_ladder(t, path, title, caption):
    """t: label, emotion, auc, ci_low, ci_high, in ladder order (top first). Colour and marker show with or without emotion."""
    height = 2.5 + 0.48 * len(t)
    fig, ax = _new(8, height, left=0.25, top=0.62, bottom=1.25)
    _frame(ax, "x")
    y = np.arange(len(t))[::-1]
    for emotion, color, marker, label in ((False, LINE_COLORS[0], "o", "No emotion"), (True, LINE_COLORS[1], "s", "With emotion")):
        keep = t["emotion"].to_numpy() == emotion
        s = t[keep]
        if len(s):
            ax.errorbar(s["auc"], y[keep], xerr=[s["auc"] - s["ci_low"], s["ci_high"] - s["auc"]], fmt=marker, color=color, markersize=7,
                        elinewidth=1.4, markeredgecolor=SURFACE, markeredgewidth=0.7, label=label)
    span = float(t["ci_high"].max() - t["ci_low"].min())
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.text(r["ci_high"] + 0.04 * span, yi, f"{r['auc']:.4f}", va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(y, t["label"], color=INK)
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.set_xlim(t["ci_low"].min() - 0.08 * span, t["ci_high"].max() + 0.2 * span)
    ax.set_xlabel("Mean AUC per impression (0.5 = random)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1 * 6 / height), ncol=2, frameon=False, fontsize=8, labelcolor=INK)
    _finish(fig, path, title, caption, height)


def gain_forest(c, path, title, caption):
    """c: part, metric, mean_diff, ci_low, ci_high, significant_holm. Per metric, valid and dev; hollow = not significant after Holm."""
    metrics = METRIC_COLUMNS[::-1]
    fig, ax = _new(8, 6.6, left=0.13, top=0.62, bottom=1.3)
    _frame(ax, "x")
    base = {m: i for i, m in enumerate(metrics)}
    for j, part in enumerate(("valid", "dev")):
        color, marker, label = PART_STYLE["valid"] if part == "valid" else (LINE_COLORS[2], "^", "dev (15 Nov, final test)")
        s = c[c["part"] == part]
        ys = np.array([base[m] for m in s["metric"]]) + (0.17 if part == "dev" else -0.17)
        ax.errorbar(s["mean_diff"], ys, xerr=_errors(s), fmt="none", ecolor=color, elinewidth=1.4)
        solid = s["significant_holm"].to_numpy()
        ax.scatter(s["mean_diff"][solid], ys[solid], marker=marker, s=44, color=color, edgecolor=SURFACE, linewidth=0.7, zorder=3, label=label)
        ax.scatter(s["mean_diff"][~solid], ys[~solid], marker=marker, s=44, facecolor=SURFACE, edgecolor=color, linewidth=1.3, zorder=3)
    ax.scatter([], [], marker="o", s=44, facecolor=SURFACE, edgecolor=INK_2, linewidth=1.3, label="Hollow: not significant after Holm")
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(list(base.values()), [METRIC_LABELS[m] for m in base], color=INK)
    ax.set_ylim(-0.7, len(metrics) - 0.3)
    ax.set_xlabel("M3 minus M2, difference in the metric (positive = emotion helps)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    _finish(fig, path, title, caption, 6.6)


def category_forest(t, path, title, caption, xlabel):
    """t: group, n, mean_diff, ci_low, ci_high, in plot order (top first)."""
    height = 2.2 + 0.46 * len(t)
    fig, ax = _new(8, height, left=0.2, top=0.62, bottom=0.95)
    _frame(ax, "x")
    y = np.arange(len(t))[::-1]
    ax.errorbar(t["mean_diff"], y, xerr=_errors(t), fmt="o", color=LINE_COLORS[0], markersize=7, elinewidth=1.4, markeredgecolor=SURFACE,
                markeredgewidth=0.7)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(y, [f"{g} (n = {n:,})" for g, n in zip(t["group"], t["n"])], color=INK)
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.set_xlabel(xlabel)
    _finish(fig, path, title, caption, height)


def history_gain(t, path, title, caption, ylabel):
    """t: group, n, mean_diff, ci_low, ci_high, in bin order."""
    fig, ax = _new(8, 4.9, left=0.1, top=0.62, bottom=1.25)
    _frame(ax, "y")
    x = np.arange(len(t))
    ax.errorbar(x, t["mean_diff"], yerr=_errors(t), fmt="o", color=LINE_COLORS[0], markersize=7, elinewidth=1.4, markeredgecolor=SURFACE,
                markeredgewidth=0.7)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(x, [f"{g}\n(n = {n:,})" for g, n in zip(t["group"], t["n"])], color=INK)
    ax.set_xlim(-0.5, len(t) - 0.5)
    ax.set_xlabel("Known items in the user's history")
    ax.set_ylabel(ylabel)
    _finish(fig, path, title, caption, 4.9)


def _pair(comparison, a, b):
    return comparison[(comparison["model_a"] == a) & (comparison["model_b"] == b)]


def build_charts(tables, out_dir):
    """All result charts from saved tables. -> {chart: caption}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    caps = {}
    paths = {n: out_dir / f"{n}.png" for n in RESULT_CHARTS}

    ladder = pd.read_csv(tables / "model_ladder.csv")
    best = ladder.loc[ladder["auc"].idxmax()]
    named = ", ".join(f"{r['label'].split()[0]} {r['auc']:.3f}" for _, r in ladder.iterrows())
    caps["1_model_ladder"] = (f"Dev AUC with 95% confidence intervals over {int(ladder['n'].iloc[0]):,} impressions: {named}; "
                              f"{best['label']} is the highest so far.")
    model_ladder(ladder, paths["1_model_ladder"], "Final test AUC by model", caps["1_model_ladder"])

    comp = _pair(pd.read_csv(tables / "m2m3_comparison.csv"), M2_NAME, M3_NAME)
    dev, valid = comp[comp["part"] == "dev"], comp[comp["part"] == "valid"]
    auc = dev[dev["metric"] == "auc"].iloc[0]
    caps["2_m3_gain_by_metric"] = (f"M3 minus M2 with 95% confidence intervals: dev AUC gain {auc['mean_diff']:+.4f} ({auc['ci_low']:.4f} to "
                                   f"{auc['ci_high']:.4f}), and {int(dev['significant_holm'].sum())} of {len(dev)} metrics stay significant "
                                   f"after Holm on dev, {int(valid['significant_holm'].sum())} of {len(valid)} on valid.")
    gain_forest(comp, paths["2_m3_gain_by_metric"], "Gain from adding emotion to Sentence-BERT, per metric", caps["2_m3_gain_by_metric"])

    grid = pd.read_csv(tables / "m3_fusion_grid_valid.csv")
    meta = json.loads((tables / "m3_fusion_meta.json").read_text(encoding="utf-8"))
    chosen = meta["chosen"]
    mood = grid[(grid["variant"] == "mood_repair") & (grid["alpha"] < 1)]["auc"].max()
    caps["3_alpha_ablation"] = (f"The best blend is {chosen['variant']} emotion at alpha {chosen['alpha']} (valid AUC {meta['valid_auc']:.4f}, against "
                                f"{meta['m2_valid_auc']:.4f} for M2 alone at the dashed line); mood repair peaks at {mood:.4f}, below M2 at every "
                                "alpha, and title-only was reported but not eligible.")
    alpha_curve(grid, meta["m2_valid_auc"], paths["3_alpha_ablation"], "Validation AUC by fusion weight and emotion score", caps["3_alpha_ablation"])

    rq5 = pd.read_csv(tables / "m3_where_it_helps_dev.csv")
    cat = rq5[rq5["breakdown"] == "clicked category"].sort_values("mean_diff", ascending=False).reset_index(drop=True)
    top, low = cat.iloc[0], cat.iloc[-1]
    caps["4_m3_gain_by_category"] = (f"Descriptive, not causal: on dev the M3 minus M2 AUC gain runs from {top['mean_diff']:+.4f} ({top['group']}) to "
                                     f"{low['mean_diff']:+.4f} ({low['group']}), around {auc['mean_diff']:+.4f} overall; intervals are 95%.")
    category_forest(cat, paths["4_m3_gain_by_category"], "Where emotion helps M3, by clicked category (descriptive)", caps["4_m3_gain_by_category"],
                    "M3 minus M2 AUC per impression")

    hist = rq5[rq5["breakdown"] == "known history items"].reset_index(drop=True)
    caps["5_m3_gain_by_history"] = (f"Descriptive: on dev the gain is {hist['mean_diff'].iloc[-1]:+.4f} for users with {hist['group'].iloc[-1]} known history "
                                    f"items, {hist['mean_diff'].iloc[1]:+.4f} for {hist['group'].iloc[1]}, and exactly 0 without history, where the "
                                    "personal score is constant; intervals are 95%.")
    history_gain(hist, paths["5_m3_gain_by_history"], "Gain from emotion by length of the user's history (descriptive)", caps["5_m3_gain_by_history"],
                 "M3 minus M2 AUC per impression")

    sem = pd.read_csv(tables / "semeval2007_validation.csv")
    sem = sem[sem["kind"] == "correlation"]
    rho = lambda gold, source: float(sem[(sem["gold"] == gold) & (sem["source"] == source)]["spearman"].iloc[0])
    groups = [rho(e, "goemotions_group") for e in EMOTION_NAMES]
    caps["6_emotion_validation"] = (f"Spearman correlation with human ratings on {int(sem['n'].iloc[0]):,} SemEval-2007 test headlines, 95% bootstrap "
                                    f"intervals: valence {rho('valence', 'go_vad_valence'):.2f} for GoEmotions and {rho('valence', 'vad_valence'):.2f} for "
                                    f"NRC, main emotions {min(groups):.2f} to {max(groups):.2f} for GoEmotions; dashed line = pre-declared minimum 0.2.")
    semeval_correlations(sem, paths["6_emotion_validation"], "Emotion scores against human ratings of news headlines", caps["6_emotion_validation"])
    (tables / "results_chart_captions.json").write_text(json.dumps(caps, indent=2), encoding="utf-8")
    return caps

