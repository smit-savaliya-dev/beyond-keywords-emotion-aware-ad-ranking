"""Step 3 of base-v2: the eight story charts in results/figures/story/ (and the data behind them in results/tables/).
Reads saved tables only: item-level emotion files and fit/valid results. No clicks are loaded and dev is never read.

  python scripts/37_analysis_story_charts.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import story_charts as sc
from src.config import PROCESSED, TABLES, STORY_DIR, STORY_TOP_CATEGORIES
from src.emotion import SIX_CLASSES
from src.emotion_v2 import intensity_table

NAMES = {"go_pct_curiosity": "curiosity (pct)", "go_pct_desire": "desire (pct)", "go_pct_excitement": "excitement (pct)",
         "go_pct_joy": "joy (pct)", "go_pct_optimism": "optimism (pct)", "go_pct_nervousness": "nervousness (pct)",
         "go_pct_annoyance": "annoyance (pct)", "go_pct_fear": "fear (pct)", "go_emotionality": "emotionality",
         "go_vad_valence": "valence (classifier)", "go_vad_dominance": "dominance (classifier)",
         "go_vad_arousal": "arousal (classifier)", "vad_valence": "valence (lexicon)", "vad_dominance": "dominance (lexicon)",
         "nrc_trust": "trust (lexicon)", "nrc_anticipation": "anticipation (lexicon)", "go_dist_optimism": "optimism (dist)",
         "go_dist_nervousness": "nervousness (dist)", "go_dist_fear": "fear (dist)", "go_intensity": "intensity"}
VA_OFFSETS = {"health": (6, -13), "autos": (-36, 3), "sports": (6, -13), "travel": (6, 5), "weather": (-44, 6),
              "finance": (8, -13), "lifestyle": (-14, 10), "foodanddrink": (6, 5), "video": (6, 5), "news": (6, 5)}


def part_rows(t, names):
    """Keep `names` {feature: row label}, in that order, for both parts."""
    t = t[t["feature"].isin(names)].assign(row=lambda d: d["feature"].map(names))
    return t.assign(order=t["feature"].map({f: i for i, f in enumerate(names)})).sort_values("order")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    p, out = args.processed, STORY_DIR
    out.mkdir(parents=True, exist_ok=True)
    choice = json.loads((TABLES / "emotion_v2_choice.json").read_text(encoding="utf-8"))
    share, t = choice["chosen_title_share"], choice["chosen_t"]
    labels = pd.read_parquet(p / "news_emotion_labels_v2.parquet")
    variants = pd.read_parquet(p / "news_emotion_labels_variants.parquet", columns=["news_id", "label_v2b"])

    # 1 emotion mix
    def mix_of(s):
        return s.value_counts(normalize=True).mul(100).reindex(SIX_CLASSES, fill_value=0)
    mix = pd.DataFrame({"Original (combined text)": mix_of(pd.read_parquet(p / "news_emotion.parquet")["dominant_emotion"]),
                        "v1 (1/3 title, argmax)": mix_of(pd.read_parquet(p / "news_emotion_weighted.parquet")["dominant_emotion"]),
                        f"v2 (title {share}, t = {t})": mix_of(labels["label"]),
                        "v2b (v2 without stance labels)": mix_of(variants["label_v2b"])}).T
    mix.round(2).to_csv(TABLES / "story_emotion_mix.csv")
    v1 = pd.read_parquet(p / "news_emotion_weighted.parquet").set_index("news_id")["dominant_emotion"]
    lab = labels.set_index("news_id")
    moved = lab[(v1.reindex(lab.index) == "neutral") & (lab["label"] != "neutral")]
    moved_pct, still = 100 * len(moved) / len(lab), 100 * (moved["neutral_score"] > 0.5).mean()
    sc.emotion_mix(mix, out / "1_emotion_mix.png",
                   f"Neutral falls from {mix.iloc[1]['neutral']:.0f}% to {mix.iloc[2]['neutral']:.0f}% (v2), or to "
                   f"{mix.iloc[3]['neutral']:.0f}% without stance labels (v2b)",
                   f"Emotion labels of all 65,238 news items under each rule. The classifier is unchanged: {moved_pct:.1f}% of "
                   f"items left neutral in v2 because it counts weaker scores as emotion, and for {still:.1f}% of those the "
                   "classifier's own neutral score is still above 0.5. v2b ignores approval and disapproval.")

    # 2 neutral vs threshold
    grid = pd.read_csv(TABLES / "emotion_v2_grid.csv")
    v2b = pd.read_csv(TABLES / "emotion_label_variants_grid.csv").query("variant == 'v2b'")
    sc.neutral_curve(grid, share, t, out / "2_neutral_vs_threshold.png", "The threshold drives the neutral share far more than the title weight",
                     "Share of items labelled neutral for each title weight and threshold. The ringed point is the setting "
                     "chosen by the rule written before the analysis (D30). The dashed line ignores approval and disapproval.",
                     extra=("v2b: title 0.50, no stance labels", v2b))

    # 3 title vs abstract intensity
    title_i = intensity_table(pd.read_parquet(p / "news_emotion_raw28_title.parquet"))
    abstract_i = intensity_table(pd.read_parquet(p / "news_emotion_raw28_abstract.parquet"))
    same = title_i[title_i["news_id"].isin(abstract_i["news_id"])]
    pd.DataFrame({"field": ["title", "abstract"], "n": [len(same), len(abstract_i)],
                  "median": [same["intensity"].median(), abstract_i["intensity"].median()],
                  f"pct_at_least_{t}": [100 * (same["intensity"] >= t).mean(), 100 * (abstract_i["intensity"] >= t).mean()]
                  }).round(4).to_csv(TABLES / "story_intensity.csv", index=False)
    sc.intensity_compare(same["intensity"], abstract_i["intensity"], t, out / "3_title_vs_abstract_intensity.png",
                         STORY_TITLES[3], f"Emotion intensity of titles and abstracts, scored separately, on the same "
                         f"{len(same):,} items that have both. A higher curve means more items reach that intensity.")

    # 4 category heatmap and 5 valence-arousal map
    news = pd.read_parquet(p / "news.parquet", columns=["news_id", "category"])
    feats = pd.read_parquet(p / "news_emotion_features_v2.parquet", columns=["news_id", "go_vad_valence", "go_vad_arousal"])
    items = news.merge(labels[["news_id", "label"]], on="news_id", validate="one_to_one").merge(feats, on="news_id",
                                                                                                 validate="one_to_one")
    top = items["category"].value_counts().head(STORY_TOP_CATEGORIES).index
    sub = items[items["category"].isin(top)]
    heat = pd.crosstab(sub["category"], sub["label"], normalize="index").mul(100).reindex(columns=SIX_CLASSES, fill_value=0)
    heat = heat.loc[heat["neutral"].sort_values().index]
    heat.round(1).to_csv(TABLES / "story_category_emotion.csv")
    sc.category_heatmap(heat, out / "4_emotion_by_category.png", STORY_TITLES[4],
                        f"Emotion labels within the {STORY_TOP_CATEGORIES} largest categories, sorted by neutral share. Each "
                        "row sums to 100%. Item-level only, no clicks.")
    g = sub.groupby("category")
    va = pd.DataFrame({"valence": g["go_vad_valence"].mean(), "arousal": g["go_vad_arousal"].mean(),
                       "v_low": g["go_vad_valence"].quantile(0.25), "v_high": g["go_vad_valence"].quantile(0.75),
                       "a_low": g["go_vad_arousal"].quantile(0.25), "a_high": g["go_vad_arousal"].quantile(0.75)}).reset_index()
    va["v_low"], va["v_high"] = np.minimum(va["v_low"], va["valence"]), np.maximum(va["v_high"], va["valence"])
    va["a_low"], va["a_high"] = np.minimum(va["a_low"], va["arousal"]), np.maximum(va["a_high"], va["arousal"])
    va.round(3).to_csv(TABLES / "story_category_valence_arousal.csv", index=False)
    sc.valence_arousal_map(va, VA_OFFSETS, out / "5_valence_arousal_map.png", STORY_TITLES[5],
                           "Mean valence and arousal per category (dots), with the middle 50% of items as bars. Scores are the "
                           "VAD values of GoEmotions labels, weighted by probability. The spread within a category is large.")

    # 6 clicked vs not, 7 congruence
    sig = pd.read_csv(TABLES / "emotion_signal_v2.csv")
    item_t = part_rows(sig, NAMES)
    wide = item_t.pivot(index="row", columns="part", values="dz")
    flips = wide.index[np.sign(wide["fit"]) != np.sign(wide["valid"])].tolist()
    sc.clicked_effects(item_t, flips, out / "6_clicked_vs_not_effects.png", STORY_TITLES[6],
                       "How much clicked items differ from unclicked items in the same impression, per emotion feature. Lines are "
                       "95% CIs; hollow = not significant at both impression and user level (Holm).")
    cong = part_rows(sig, {"cong_pct_cos": "raw match", "cong_pct_cos_personal": "personal (minus corpus mean)",
                           "cong_pct_cos_userspecific": "user-specific (minus other user)"})
    sc.congruence_effects(cong, out / "7_congruence_effects.png", STORY_TITLES[7],
                          "Clicked items match the emotional context inferred from the user's recent reading more than unclicked "
                          "items do. User-specific removes what a random other user's history also predicts.")

    # 8 context windows
    win = pd.read_csv(TABLES / "context_windows_signal.csv")
    win = win[win["role"] == "user-specific"]
    names = {"long_50": "last 50 (decay 0.9)", "short_5": "last 5", "short_3": "last 3", "short_1": "last 1"}
    main_t = win[win["sample"] == "all"].assign(row=lambda d: d["window"].map(names))
    main_t = main_t.assign(order=main_t["window"].map({k: i for i, k in enumerate(names)})).sort_values("order")
    ctrl = win[win["sample"].str.startswith("history") & win["window"].isin(["short_5", "older_5"])]
    ctrl = ctrl.assign(row=ctrl["window"].map({"short_5": "last 5", "older_5": "previous 5"}))
    ctrl = ctrl.assign(order=ctrl["window"].map({"short_5": 0, "older_5": 1})).sort_values("order")
    sc.context_effects(main_t, ctrl, out / "8_short_vs_long_context.png", STORY_TITLES[8],
                       "User-specific congruence for each history window. The right panel holds size equal: the newest 5 items do "
                       "not beat the 5 before them, so the gain from longer windows does not come from recency.")
    print(f"wrote {len(list(out.glob('*.png')))} charts to {out}")


STORY_TITLES = {3: "Abstracts reach higher emotion intensity than titles", 4: "Emotion mix differs by news category",
                5: "Categories sit in different parts of the valence-arousal space",
                6: "Many clicked-vs-not effects change between fit and valid",
                7: "Emotional congruence holds on both days, and part of it is user-specific",
                8: "A long reading history carries more emotional context than the last few items"}

if __name__ == "__main__":
    main()
