"""Congruence signal on part == "fit" ONLY: is a candidate's emotional similarity to the user's HISTORY profile
higher (congruence-seeking, dz > 0) or lower (mood repair, dz < 0) for clicked items? Descriptive, no model.

Profile = recency-weighted mean (decay 0.9, newest weight 1, last 50) of history item vectors; cold start = corpus mean.
Measures (CONGRUENCE_MEASURES in config): cosine on go_pct_* (main) and on the weighted 6-class vector,
-KL on the 6-class vector, valence distance -|profile - item| on go_vad_valence; -KL on go_pct_* is exploratory.
Each also against a generic profile (corpus mean, same for every user); personal = raw - generic isolates the
user-specific part from the item's own emotion effect. Shared code: src.congruence.congruence_features.

  python scripts/31_analysis_congruence.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.click_signal import MaskedValues, holm_family, load_part_context, permute_within_groups, signal_table
from src.config import (CONGRUENCE_DECAY, CONGRUENCE_EPS, CONGRUENCE_MAX_HISTORY, CONGRUENCE_MEASURES,
                        PROCESSED, TABLES, SEED, SIGNAL_ALPHA, SIGNAL_TOP_CATEGORIES)
from src.congruence import congruence_features
from src.emotion import SIX_CLASSES
from src.emotion_features import GO_EMOTIONS
from src.provenance import run_info

VERSIONS = {"pct": "go_pct_* (8 emotions, percentile)", "six": "weighted 6-class distribution",
            "valence": "go_vad_valence (classifier-derived valence)"}


def direction(t):
    return np.select([t["significant_both"] & (t["dz"] > 0), t["significant_both"] & (t["dz"] < 0)],
                     ["congruence-seeking (clicked more similar)", "mood repair (clicked less similar)"],
                     "no clear signal")


def describe(t):
    base = t["feature"].str.replace(r"_(generic|personal|otheruser|userspecific)$", "", regex=True)
    status = {f"cong_{v}_{m}": s for v, m, s in CONGRUENCE_MEASURES}
    t.insert(1, "measure", base)
    ends = [t["feature"].str.endswith(f"_{r}") for r in ("generic", "personal", "otheruser", "userspecific")]
    t.insert(2, "role", np.select(ends, ["generic (corpus-mean profile)", "personal (raw - generic)",
                                         "control: random other user's profile",
                                         "user-specific (raw - other user), robustness"], "raw (user profile)"))
    t.insert(3, "status", base.map(status))
    t["direction"] = direction(t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    p = args.processed

    pct_cols = [f"go_pct_{e}" for e in GO_EMOTIONS]
    feats = pd.read_parquet(p / "news_emotion_features.parquet", columns=["news_id", *pct_cols, "go_vad_valence"])
    six = pd.read_parquet(p / "news_emotion_weighted.parquet").set_index("news_id").reindex(feats["news_id"])
    if six[SIX_CLASSES].isna().any().any():
        raise ValueError("weighted 6-class table does not cover every item")
    news = pd.read_parquet(p / "news.parquet", columns=["news_id", "category"]).set_index("news_id")
    ctx = load_part_context(p, "fit", pd.Index(feats["news_id"]))
    groups, group_users, labels, idx, histories = ctx.groups, ctx.group_users, ctx.labels, ctx.item_idx, ctx.histories

    vectors = {"pct": feats[pct_cols].to_numpy(np.float64), "six": six[SIX_CLASSES].to_numpy(np.float64),
               "valence": feats["go_vad_valence"].to_numpy(np.float64)}
    values, used = congruence_features(groups, idx, histories, vectors, CONGRUENCE_MEASURES, shuffle_seed=SEED)
    cold_pct = round(100 * float((used == 0).mean()), 2)
    warm = used[groups] > 0
    print(f"fit: {ctx.n_impressions} impressions, {len(set(group_users))} users, {len(groups)} candidates; "
          f"cold start {cold_pct}% of impressions")

    main_t = holm_family(signal_table(groups, labels, values, SIGNAL_ALPHA, group_users).assign(sample="all"), SIGNAL_ALPHA)
    null_t = signal_table(groups, permute_within_groups(groups, labels, SEED), values, SIGNAL_ALPHA, group_users)
    for c in ("dz", "p_holm", "user_p_holm", "significant_both"):
        main_t[f"null_{c}"] = null_t[c].to_numpy()
    warm_t = holm_family(signal_table(groups[warm], labels[warm], MaskedValues(values, warm),
                                      SIGNAL_ALPHA, group_users).assign(sample="warm users (history > 0)"), SIGNAL_ALPHA)
    table = describe(pd.concat([main_t, warm_t], ignore_index=True))

    category = news["category"].reindex(feats["news_id"]).to_numpy()[idx]
    clicks = pd.Series(category[labels == 1]).value_counts()
    top = clicks.index[:SIGNAL_TOP_CATEGORIES].tolist()
    parts = []
    for c in top:
        mc = category == c
        t = signal_table(groups[mc], labels[mc], MaskedValues(values, mc), SIGNAL_ALPHA, group_users)
        t.insert(0, "category", c)
        parts.append(t)
    by_cat = pd.concat(parts, ignore_index=True)
    by_cat = describe(holm_family(by_cat, SIGNAL_ALPHA))

    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "congruence_signal_fit.csv", index=False)
    by_cat.to_csv(TABLES / "congruence_signal_by_category_fit.csv", index=False)
    false_pos = {"t_test": int(null_t["significant_holm"].sum()), "user_t_test": int(null_t["significant_user"].sum()),
                 "both_levels": int(null_t["significant_both"].sum()), "of": len(null_t),
                 "max_abs_null_dz": round(float(null_t["dz"].abs().max()), 4)}
    personal = main_t[main_t["feature"].str.endswith("_personal")]
    userspec = main_t[main_t["feature"].str.endswith("_userspecific")]
    meta = {
        "part": "fit only", "profile_source": "behaviors.history (clicks before the log period), never impression clicks",
        "profile": {"decay": CONGRUENCE_DECAY, "max_history": CONGRUENCE_MAX_HISTORY,
                    "weights": "decay**k, k = 0 (weight 1) for the newest history item", "cold_start": "corpus mean",
                    "pct_impressions_cold_start": cold_pct, "median_history_items_used": float(np.median(used))},
        "versions": VERSIONS,
        "measures": {"cos": "cosine(profile, item)",
                     "negkl": f"-KL(item || profile), eps {CONGRUENCE_EPS}, vectors renormalised to sum 1",
                     "absdist": "-|profile - item| (valence)"},
        "measure_status": {f"cong_{v}_{m}": s for v, m, s in CONGRUENCE_MEASURES},
        "roles": {"raw": "measure against the user's history profile",
                  "_generic": "same measure against the corpus-mean profile (identical for every user)",
                  "_personal": "raw - generic = personal_congruence (pre-declared)",
                  "_otheruser": f"measure against a random other impression's profile (seeded permutation, {SEED})",
                  "_userspecific": "raw - _otheruser. Robustness control ADDED AFTER seeing results: corpus-mean "
                                   "'generic' differs from the typical clicked-history profile, so _personal can be "
                                   "positive without personalisation; _userspecific removes that shared shift."},
        "interpretation_rule": "if raw is significant but _personal is not, the match is item emotion, not "
                               "personalisation",
        "personal_significant_both_levels": dict(zip(personal["feature"], personal["significant_both"].astype(bool))),
        "userspecific_significant_both_levels": dict(zip(userspec["feature"],
                                                         userspec["significant_both"].astype(bool))),
        "method": "as Phase 5: per-impression clicked-minus-unclicked mean difference, dz, one-sample t-test with Holm "
                  "(one family per sample, exploratory measure included), user-level t-test with Holm, "
                  "shuffled-label null; Wilcoxon robustness only.",
        "direction_rule": "dz > 0 and significant at both levels -> congruence-seeking; dz < 0 -> mood repair",
        "shuffled_label_false_positives": false_pos,
        "top_categories_by_fit_clicks": {c: int(clicks[c]) for c in top},
        "limitations": ["descriptive association only", "no popularity control (LightGBM M4 vs M5 later)",
                        "MIND history is one fixed list per user, so profiles do not evolve within the log period",
                        "emotion tracks topic, so personal congruence may partly be topical similarity (M2 vs M3)",
                        "all-positive percentile vectors give high, compressed cosine values; dz is scale-free"],
        "alpha": SIGNAL_ALPHA, "run_info": run_info(),
    }
    (TABLES / "congruence_signal_fit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pd.set_option("display.width", 190)
    show = ["feature", "status", "dz", "dz_ci_low", "dz_ci_high", "user_dz", "p_holm", "user_p_holm",
            "significant_both", "direction"]
    for s_name, t in table.groupby("sample", sort=False):
        print(f"\n[{s_name}]")
        print(t[show].round(4).to_string(index=False))
    print(f"\nnull (shuffled labels): {false_pos}")
    print("\nwithin-category dz (* = significant at both levels):")
    cell = by_cat["dz"].round(3).astype(str) + np.where(by_cat["significant_both"], "*", " ")
    print(by_cat.assign(cell=cell).pivot(index="feature", columns="category", values="cell")
          .loc[list(values), top].to_string())


if __name__ == "__main__":
    main()
