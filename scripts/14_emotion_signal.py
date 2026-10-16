"""Does emotion relate to clicks? Descriptive signal analysis on part == "fit" ONLY (valid and dev never loaded).
Per impression: mean(feature | clicked) - mean(feature | not clicked). Primary test: one-sample t-test
across impressions, dz = mean / sd, Holm within each family; user-level check on per-user mean differences.
Wilcoxon is kept only as a robustness column (biased here). No model training.

  python scripts/14_emotion_signal.py
"""
import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.click_signal import holm, permute_within_groups, plot_signal, signal_table
from src.config import FIGURES, PROCESSED, RESULTS, SEED, SIGNAL_ALPHA, SIGNAL_CORE_FEATURES, SIGNAL_TOP_CATEGORIES
from src.emotion_features import GO_EMOTIONS
from src.provenance import run_info

TABLES = RESULTS / "tables"
VARIANT_STEMS = [*GO_EMOTIONS, "neutral", "emotionality", "emotion_mass"]
VARIANTS = {"weighted": "go_{}", "title": "go_title_{}", "abstract": "go_abs_{}"}
PCT_VARIANTS = {"weighted": "go_pct_{}", "title": "go_title_pct_{}", "abstract": "go_abs_pct_{}"}


class RowValues(Mapping):
    """Per-candidate feature arrays built on demand (all 101 at once would need ~4 GB)."""

    def __init__(self, feats, cols, rows):
        self.feats, self.cols, self.rows = feats, list(cols), rows

    def __getitem__(self, col):
        return self.feats[col].to_numpy(dtype=np.float64)[self.rows]

    def __iter__(self):
        return iter(self.cols)

    def __len__(self):
        return len(self.cols)


def load_fit(processed):
    imps = pd.read_parquet(processed / "impressions.parquet", columns=["imp_key", "part", "user_id", "news_id", "label"],
                           filters=[("part", "==", "fit")])
    imps = imps[imps["part"].astype(str) == "fit"]
    if set(imps["part"].astype(str)) != {"fit"}:
        raise ValueError("non-fit rows loaded")
    return imps.reset_index(drop=True)


def variant_rows(table, patterns, stems, sample):
    rows = []
    for stem in stems:
        for variant, pat in patterns.items():
            r = table.loc[table["feature"] == pat.format(stem)].iloc[0].to_dict()
            rows.append({"emotion": stem, "variant": variant, "sample": sample, **r})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    p = args.processed

    fit = load_fit(p)
    feats = pd.read_parquet(p / "news_emotion_features.parquet")
    news = pd.read_parquet(p / "news.parquet", columns=["news_id", "category"])
    idx = pd.Index(feats["news_id"]).get_indexer(fit["news_id"].astype(str))
    if (idx < 0).any():
        raise ValueError(f"{int((idx < 0).sum())} fit candidates have no emotion features")
    groups = pd.factorize(fit["imp_key"].astype(str))[0]
    group_users = pd.Series(pd.factorize(fit["user_id"].astype(str))[0]).groupby(groups).first().to_numpy()
    labels = fit["label"].to_numpy()
    category = news.set_index("news_id")["category"].reindex(feats["news_id"]).to_numpy()[idx]
    has_abstract = feats["has_abstract"].to_numpy()[idx]
    n_users = len(set(group_users))
    print(f"fit: {groups.max() + 1} impressions, {n_users} users, {len(fit)} candidate rows, "
          f"{int(labels.sum())} clicks")

    cols = [c for c in feats.columns if c not in ("news_id", "has_abstract")]
    values = RowValues(feats, cols, idx)

    # 1. every feature, all fit impressions (one Holm family per test)
    main_t = signal_table(groups, labels, values, SIGNAL_ALPHA, group_users)

    # 1b. null check: labels permuted within each impression, so any "significant" feature is a false positive
    null_t = signal_table(groups, permute_within_groups(groups, labels, SEED), values, SIGNAL_ALPHA, group_users)
    for c in ("dz", "p_holm", "p_wilcoxon_holm", "user_p_holm"):
        main_t[f"null_{c}"] = null_t[c].to_numpy()
    false_pos = {"t_test": int((null_t["p_holm"] < SIGNAL_ALPHA).sum()),
                 "wilcoxon": int((null_t["p_wilcoxon_holm"] < SIGNAL_ALPHA).sum()),
                 "user_t_test": int((null_t["user_p_holm"] < SIGNAL_ALPHA).sum()),
                 "both_levels": int(null_t["significant_both"].sum())}

    # 2. text variants on a common sample: candidates with an abstract (one Holm family)
    var_cols = [pat.format(s) for pats in (VARIANTS, PCT_VARIANTS) for s in
                (VARIANT_STEMS if pats is VARIANTS else GO_EMOTIONS) for pat in pats.values()]
    m = has_abstract
    common = signal_table(groups[m], labels[m], RowValues(feats, var_cols, idx[m]), SIGNAL_ALPHA, group_users)
    variants = pd.concat([
        variant_rows(main_t, VARIANTS, VARIANT_STEMS, "all candidates"),
        variant_rows(main_t, PCT_VARIANTS, GO_EMOTIONS, "all candidates"),
        variant_rows(common, VARIANTS, VARIANT_STEMS, "candidates with abstract"),
        variant_rows(common, PCT_VARIANTS, GO_EMOTIONS, "candidates with abstract"),
    ], ignore_index=True)
    variants["kind"] = np.where(variants["feature"].str.contains("_pct_"), "percentile", "score")

    # 3. within-category: only candidates of one category inside each impression (one Holm family)
    clicks = pd.Series(category[labels == 1]).value_counts()
    top = clicks.index[:SIGNAL_TOP_CATEGORIES].tolist()
    parts = []
    for c in top:
        mc = category == c
        t = signal_table(groups[mc], labels[mc], RowValues(feats, SIGNAL_CORE_FEATURES, idx[mc]), SIGNAL_ALPHA,
                         group_users)
        t.insert(0, "category", c)
        parts.append(t)
    by_cat = pd.concat(parts, ignore_index=True)
    for p_col in ("p", "p_wilcoxon", "user_p"):
        by_cat[f"{p_col}_holm"] = holm(by_cat[p_col].to_numpy())
    by_cat["significant_holm"] = by_cat["p_holm"] < SIGNAL_ALPHA
    by_cat["significant_user"] = ((by_cat["user_p_holm"] < SIGNAL_ALPHA)
                                  & (np.sign(by_cat["user_dz"]) == np.sign(by_cat["dz"])))
    by_cat["significant_both"] = by_cat["significant_holm"] & by_cat["significant_user"]

    TABLES.mkdir(parents=True, exist_ok=True)
    main_t.to_csv(TABLES / "emotion_signal_fit.csv", index=False)
    variants.to_csv(TABLES / "emotion_signal_variants_fit.csv", index=False)
    by_cat.to_csv(TABLES / "emotion_signal_by_category_fit.csv", index=False)
    n_both = int(main_t["significant_both"].sum())
    meta = {
        "part": "fit only", "n_impressions": int(groups.max() + 1), "n_users": n_users,
        "n_candidates": int(len(fit)), "n_clicks": int(labels.sum()),
        "method": "per impression: mean(feature | clicked) - mean(feature | not clicked), NaN values ignored; "
                  "impressions without both classes skipped. Effect size dz = mean/sd of the differences "
                  "(approx. 95% CI). Primary test: one-sample t-test of the differences against 0, Holm within "
                  "each family. User-level check: each user's mean difference, one-sample t-test across users, "
                  "Holm; significant_user also requires the same sign as dz. significant_both = both levels.",
        "why_not_wilcoxon": "Wilcoxon signed-rank assumes differences symmetric around 0 under the null. With 1-2 "
                            "clicks against ~35 non-clicks and right-skewed features, the null distribution is "
                            "asymmetric: with labels shuffled within impressions, Wilcoxon flagged "
                            f"{false_pos['wilcoxon']}/{len(null_t)} features (t-test: {false_pos['t_test']}). "
                            "Wilcoxon p, its Holm p and the rank-biserial are kept as robustness columns only.",
        "robustness_columns": {"p_wilcoxon, p_wilcoxon_holm, rank_biserial": "biased here, see why_not_wilcoxon",
                               "null_*": f"same analysis with labels shuffled within impressions (seed {SEED})"},
        "shuffled_label_false_positives": {**false_pos, "of": len(null_t),
                                           "max_abs_null_dz": round(float(null_t["dz"].abs().max()), 4)},
        "significant": {"impression_level": int(main_t["significant_holm"].sum()),
                        "user_level": int(main_t["significant_user"].sum()), "both": n_both, "of": len(main_t)},
        "families": {"emotion_signal_fit.csv": f"{len(main_t)} features",
                     "emotion_signal_variants_fit.csv": "rows 'all candidates' are taken from the main family; "
                                                        f"'candidates with abstract' is its own family ({len(common)})",
                     "emotion_signal_by_category_fit.csv": f"{len(by_cat)} tests ({len(top)} categories x "
                                                           f"{len(SIGNAL_CORE_FEATURES)} core features)"},
        "top_categories_by_fit_clicks": {c: int(clicks[c]) for c in top},
        "limitations": ["descriptive association only, not causal",
                        "no popularity control here: emotional tone may correlate with general item popularity; "
                        "this is controlled later by the LightGBM M4 vs M5 pair",
                        "features are correlated, so Holm is conservative",
                        "with ~10^5 impressions tiny effects reach significance: read dz, not p"],
        "alpha": SIGNAL_ALPHA, "run_info": run_info(),
    }
    (TABLES / "emotion_signal_fit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    go_feats = [f for f in SIGNAL_CORE_FEATURES if f.startswith("go_")]
    plot_variants = variants[(variants["sample"] == "candidates with abstract") & (variants["kind"] == "percentile")]
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_signal(
        main_t[main_t["feature"].isin(SIGNAL_CORE_FEATURES)], plot_variants,
        FIGURES / "emotion_signal_fit.png",
        "Clicked vs unclicked news within the same impression (fit only)",
        f"{meta['n_impressions']:,} fit impressions, {n_users:,} users. Dot = dz, line = 95% CI; filled = significant "
        f"at impression and user level (t-test, Holm {SIGNAL_ALPHA}). dz > 0: clicked items score higher.",
        {"GoEmotions": go_feats, "NRC lexicons": [f for f in SIGNAL_CORE_FEATURES if f not in go_feats]},
        {"weighted": "weighted (1/3 : 2/3)", "title": "title only", "abstract": "abstract only"},
        "Text variants, percentile features go_*pct_* (candidates with an abstract)")

    pd.set_option("display.width", 170)
    show = ["feature", "dz", "dz_ci_low", "dz_ci_high", "p_holm", "user_dz", "user_p_holm", "significant_both",
            "null_dz", "null_p_holm"]
    top_t = main_t.reindex(main_t["dz"].abs().sort_values(ascending=False).index)
    print("\nTop 25 features by |dz| (all fit impressions):")
    print(top_t[show].head(25).round(4).to_string(index=False))
    print(f"\nSignificant (Holm {SIGNAL_ALPHA}): impression level {meta['significant']['impression_level']}, "
          f"user level {meta['significant']['user_level']}, both {n_both} of {len(main_t)} features")
    print(f"Shuffled-label false positives (of {len(null_t)}): {false_pos}")
    star = lambda d: d["dz"].round(3).astype(str) + np.where(d["significant_both"], "*", " ")
    print("\nVariants, candidates with abstract (dz; * = significant at both levels):")
    pv = variants[variants["sample"] == "candidates with abstract"].assign(cell=star)
    print(pv.pivot_table(index=["kind", "emotion"], columns="variant", values="cell", aggfunc="first",
                         sort=False)[list(VARIANTS)].to_string())
    print("\nWithin-category dz (core features; * = significant at both levels):")
    bc = by_cat.assign(cell=star)
    print(bc.pivot(index="feature", columns="category", values="cell").loc[SIGNAL_CORE_FEATURES, top].to_string())
    print(f"\nwithin-category significant at both levels: {int(by_cat['significant_both'].sum())}/{len(by_cat)}")


if __name__ == "__main__":
    main()
