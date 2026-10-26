"""Congruence signal on part == "fit" ONLY: is a candidate's emotional similarity to the user's HISTORY profile
higher (congruence-seeking, dz > 0) or lower (mood repair, dz < 0) for clicked items? Descriptive, no model.

Profile = recency-weighted mean (decay 0.9, last 50) of history item vectors; cold start = corpus mean.
Main vector: go_pct_* (8 emotions); variant: weighted 6-class. Congruence: cosine and -KL(item || profile).
Control: the same measures against a generic profile (corpus mean, identical for every user) and the excess
(personal - generic); only the excess isolates user-specific congruence from item-level emotion effects.

  python scripts/15_congruence_signal.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.click_signal import holm, load_fit_impressions, permute_within_groups, plot_dz_panels, signal_table
from src.config import (CONGRUENCE_DECAY, CONGRUENCE_EPS, CONGRUENCE_MAX_HISTORY, FIGURES, PROCESSED, RESULTS, SEED,
                        SIGNAL_ALPHA, SIGNAL_TOP_CATEGORIES)
from src.congruence import cosine_rows, neg_kl_rows, user_profiles
from src.emotion import SIX_CLASSES
from src.emotion_features import GO_EMOTIONS
from src.provenance import run_info

TABLES = RESULTS / "tables"
VERSIONS = {"pct": "go_pct_* (8 emotions, percentile)", "six": "weighted 6-class distribution"}
MEASURES = {"cos": cosine_rows, "negkl": neg_kl_rows}


def direction(t):
    return np.select([t["significant_both"] & (t["dz"] > 0), t["significant_both"] & (t["dz"] < 0)],
                     ["congruence-seeking (clicked more similar)", "mood repair (clicked less similar)"],
                     "no clear signal")


def holm_family(t):
    for p_col in ("p", "p_wilcoxon", "user_p"):
        t[f"{p_col}_holm"] = holm(t[p_col].to_numpy())
    t["significant_holm"] = t["p_holm"] < SIGNAL_ALPHA
    t["significant_user"] = (t["user_p_holm"] < SIGNAL_ALPHA) & (np.sign(t["user_dz"]) == np.sign(t["dz"]))
    t["significant_both"] = t["significant_holm"] & t["significant_user"]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    p = args.processed

    fit = load_fit_impressions(p)
    groups, imp_keys = pd.factorize(fit["imp_key"].astype(str))
    group_users = pd.Series(pd.factorize(fit["user_id"].astype(str))[0]).groupby(groups).first().to_numpy()
    labels = fit["label"].to_numpy()

    beh = pd.read_parquet(p / "behaviors.parquet", columns=["imp_key", "part", "history"])
    beh = beh[beh["part"].astype(str) == "fit"].set_index("imp_key").reindex(imp_keys)
    if beh["history"].isna().any():
        raise ValueError("some fit impressions have no behaviors row")

    pct_cols = [f"go_pct_{e}" for e in GO_EMOTIONS]
    feats = pd.read_parquet(p / "news_emotion_features.parquet", columns=["news_id", *pct_cols])
    six = pd.read_parquet(p / "news_emotion_weighted.parquet").set_index("news_id").reindex(feats["news_id"])
    if six[SIX_CLASSES].isna().any().any():
        raise ValueError("weighted 6-class table does not cover every item")
    news = pd.read_parquet(p / "news.parquet", columns=["news_id", "category"]).set_index("news_id")
    pos = pd.Index(feats["news_id"])
    idx = pos.get_indexer(fit["news_id"].astype(str))
    if (idx < 0).any():
        raise ValueError("fit candidates without emotion vectors")
    lengths = beh["history"].apply(len).to_numpy()
    flat = pos.get_indexer(np.concatenate([np.asarray(h, dtype=object) for h in beh["history"]]).astype(str))
    histories = np.split(flat, np.cumsum(lengths)[:-1])

    matrices = {"pct": feats[pct_cols].to_numpy(np.float64),
                "six": six[SIX_CLASSES].to_numpy(np.float64)}
    values, profile_stats = {}, {}
    for v, mat in matrices.items():
        mean = mat.mean(axis=0)  # corpus mean over all items: item-level, no clicks
        prof, used = user_profiles(histories, mat, mean)
        item = mat[idx]
        personal = prof[groups]
        generic = np.broadcast_to(mean, item.shape)
        for mname, fn in MEASURES.items():
            a, b = fn(item, personal), fn(item, generic)
            values[f"cong_{v}_{mname}"] = a
            values[f"cong_{v}_{mname}_generic"] = b
            values[f"cong_{v}_{mname}_excess"] = a - b
        profile_stats[v] = {"corpus_mean": dict(zip(GO_EMOTIONS if v == "pct" else SIX_CLASSES, mean.round(4))),
                            "median_history_items_used": float(np.median(used)),
                            "pct_impressions_cold_start": round(100 * float((used == 0).mean()), 2)}
    warm = (used[groups] > 0)
    print(f"fit: {len(imp_keys)} impressions, {len(set(group_users))} users, {len(fit)} candidates; "
          f"cold start {profile_stats['pct']['pct_impressions_cold_start']}% of impressions")

    main_t = holm_family(signal_table(groups, labels, values, SIGNAL_ALPHA, group_users).assign(sample="all"))
    null_t = signal_table(groups, permute_within_groups(groups, labels, SEED), values, SIGNAL_ALPHA, group_users)
    for c in ("dz", "p_holm", "user_p_holm", "significant_both"):
        main_t[f"null_{c}"] = null_t[c].to_numpy()
    warm_t = holm_family(signal_table(groups[warm], labels[warm], {k: x[warm] for k, x in values.items()},
                                      SIGNAL_ALPHA, group_users).assign(sample="warm users (history > 0)"))
    table = pd.concat([main_t, warm_t], ignore_index=True)
    table["direction"] = direction(table)

    category = news["category"].reindex(feats["news_id"]).to_numpy()[idx]
    clicks = pd.Series(category[labels == 1]).value_counts()
    top = clicks.index[:SIGNAL_TOP_CATEGORIES].tolist()
    parts = []
    for c in top:
        mc = category == c
        t = signal_table(groups[mc], labels[mc], {k: x[mc] for k, x in values.items()}, SIGNAL_ALPHA, group_users)
        t.insert(0, "category", c)
        parts.append(t)
    by_cat = holm_family(pd.concat(parts, ignore_index=True))
    by_cat["direction"] = direction(by_cat)

    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "congruence_signal_fit.csv", index=False)
    by_cat.to_csv(TABLES / "congruence_signal_by_category_fit.csv", index=False)
    false_pos = {"t_test": int(null_t["significant_holm"].sum()), "user_t_test": int(null_t["significant_user"].sum()),
                 "both_levels": int(null_t["significant_both"].sum()), "of": len(null_t),
                 "max_abs_null_dz": round(float(null_t["dz"].abs().max()), 4)}
    meta = {
        "part": "fit only", "profile_source": "behaviors.history (clicks before the log period), never impression clicks",
        "profile": {"decay": CONGRUENCE_DECAY, "max_history": CONGRUENCE_MAX_HISTORY,
                    "weights": "decay**k, k = 0 for the newest history item", "cold_start": "corpus mean"},
        "versions": VERSIONS,
        "measures": {"cos": "cosine(profile, item)",
                     "negkl": f"-KL(item || profile), eps {CONGRUENCE_EPS}, vectors renormalised to sum 1"},
        "controls": {"_generic": "same measure against the corpus-mean profile (identical for every user)",
                     "_excess": "personal minus generic: user-specific part of congruence"},
        "method": "as Phase 5: per-impression clicked-minus-unclicked mean difference, dz, one-sample t-test with Holm "
                  "(per sample family), user-level t-test with Holm, shuffled-label null; Wilcoxon robustness only.",
        "direction_rule": "dz > 0 and significant at both levels -> congruence-seeking; dz < 0 -> mood repair",
        "profile_stats": profile_stats, "shuffled_label_false_positives": false_pos,
        "top_categories_by_fit_clicks": {c: int(clicks[c]) for c in top},
        "limitations": ["descriptive association only", "no popularity control (LightGBM M4 vs M5 later)",
                        "MIND history is one fixed list per user, so profiles do not evolve within the log period",
                        "all-positive percentile vectors give high, compressed cosine values; dz is scale-free"],
        "alpha": SIGNAL_ALPHA, "run_info": run_info(),
    }
    (TABLES / "congruence_signal_fit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    panel1 = table.assign(row=table["feature"], series=np.where(table["sample"] == "all", "all fit impressions",
                                                                  "warm users only"))
    main_cos = ["cong_pct_cos", "cong_pct_cos_generic", "cong_pct_cos_excess"]
    names = {"cong_pct_cos": "personal profile", "cong_pct_cos_generic": "generic profile",
             "cong_pct_cos_excess": "excess (personal - generic)"}
    panel2 = by_cat[by_cat["feature"].isin(main_cos)].assign(row=lambda d: d["category"],
                                                              series=lambda d: d["feature"].map(names))
    panel2 = panel2.set_index("series").loc[list(names.values())].reset_index()
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_dz_panels([("All congruence features", panel1), ("cong_pct_cos within category", panel2)],
                   FIGURES / "congruence_signal_fit.png",
                   "Emotional congruence between user history and clicked vs unclicked news (fit only)",
                   f"{len(imp_keys):,} impressions. Dot = dz, line = 95% CI; filled = significant at impression and user "
                   f"level (t-test, Holm {SIGNAL_ALPHA}). dz > 0: clicked items closer to the profile.")

    pd.set_option("display.width", 170)
    show = ["feature", "dz", "dz_ci_low", "dz_ci_high", "p_holm", "user_dz", "user_p_holm", "significant_both",
            "direction"]
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
