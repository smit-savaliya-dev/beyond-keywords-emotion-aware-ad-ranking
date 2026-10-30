"""True recency test (D39), analysis only, fit and valid (dev is never read). Short-term context = the user's clicks in earlier
impressions of the log (strictly earlier time, last 1, 3, 5, no decay); long-term = the history profile (last 50, decay 0.9).
Sample: impressions with >= 1 earlier log click; candidates the user already clicked earlier are removed.
Does recent context add anything beyond long-term? Residual test: short-term congruence minus what long-term explains inside the
impression (slope estimated on fit, applied to valid). Control: last k vs previous k log clicks, same size.

  python scripts/36_true_recency.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.click_signal import MaskedValues, holm_family, permute_within_groups, signal_table, within_group_diffs
from src.config import (CONTEXT_WINDOWS, PROCESSED, RECENCY_WINDOWS, RESULTS, SEED, SIGNAL_ALPHA, SIGNAL_TOP_CATEGORIES,
                        V2_MAIN_MEASURE)
from src.congruence import congruence_features, window_histories
from src.emotion_features import GO_EMOTIONS
from src.emotion_v2 import load_part_context
from src.provenance import run_info
from src.recency import earlier_click_lists, load_log, repeated_candidates, time_matched_partner, within_slope
from src.stats import dz_difference, holm

TABLES = RESULTS / "tables"
ROLES = {"": "raw", "_personal": "personal", "_userspecific": "user-specific"}
USER = "_userspecific"
LONG = CONTEXT_WINDOWS["long_50"]


def sample(ctx, keep_group, drop_row):
    """Re-code candidates of the kept impressions, so the other-user control draws only from this sample."""
    rows = keep_group[ctx.groups] & ~drop_row
    remap = np.full(ctx.n_impressions, -1, dtype=np.int64)
    sel = np.flatnonzero(keep_group)
    remap[sel] = np.arange(len(sel))
    return sel, remap[ctx.groups[rows]], ctx.item_idx[rows], ctx.labels[rows], ctx.group_users[sel], rows


def window_features(groups, items, history_lists, vectors, name, skip, size, decay):
    hist = window_histories(history_lists, skip, size)
    out, _ = congruence_features(groups, items, hist, {"pct": vectors}, V2_MAIN_MEASURE, shuffle_seed=SEED, decay=decay,
                                 max_items=size)
    return {f"{name}{suffix}": out[f"cong_pct_cos{suffix}"] for suffix in ROLES}


def tidy(table, part, sample_name):
    t = table.copy()
    t["window"] = t["feature"].str.replace(r"(_personal|_userspecific)$", "", regex=True)
    t["role"] = np.select([t["feature"].str.endswith("_personal"), t["feature"].str.endswith(USER),
                           t["feature"].str.startswith("resid_")], ["personal", "user-specific", "residual"], "raw")
    t.insert(0, "part", part)
    t.insert(1, "sample", sample_name)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    p = args.processed

    pct_cols = [f"go_pct_{e}" for e in GO_EMOTIONS]
    feats = pd.read_parquet(p / "news_emotion_features_v2.parquet", columns=["news_id", *pct_cols])
    vectors = feats[pct_cols].to_numpy(np.float64)
    index = pd.Index(feats["news_id"])
    clicks, imp_table = load_log(p)
    click_items = index.get_indexer(clicks["news_id"])
    news = pd.read_parquet(p / "news.parquet", columns=["news_id", "category"]).set_index("news_id")
    category = news["category"].reindex(feats["news_id"]).to_numpy()

    slopes, tables, comparisons, resid_cat, coverage, nulls = {}, [], [], [], {}, {}
    for part in ("fit", "valid"):
        ctx = load_part_context(p, part, index)
        info = imp_table.loc[ctx.imp_keys]
        lists = earlier_click_lists(info["user_id"].to_numpy(), info["time"].to_numpy(), clicks["user_id"].to_numpy(),
                                    clicks["time"].to_numpy(), click_items)
        n_earlier = np.fromiter((len(h) for h in lists), dtype=np.int64, count=len(lists))
        has = n_earlier >= 1
        repeat = repeated_candidates(ctx.groups, ctx.item_idx, lists)
        sel, groups, items, labels, users, rows = sample(ctx, has, repeat)
        hist_short = [lists[i] for i in sel]
        hist_long = [ctx.histories[i] for i in sel]
        coverage[part] = {
            "n_impressions": ctx.n_impressions, "impressions_with_earlier_click": int(has.sum()),
            "pct_impressions_covered": round(100 * float(has.mean()), 2), "users_covered": int(len(set(users))),
            "median_earlier_clicks": float(np.median(n_earlier[has])),
            "pct_candidates_removed_as_repeats": round(100 * float(repeat[has[ctx.groups]].mean()), 3),
            "clicks_removed_as_repeats": int(ctx.labels[repeat & has[ctx.groups]].sum())}
        print(f"{part}: {coverage[part]}")

        values = window_features(groups, items, hist_long, vectors, "long_50", 0, LONG["size"], LONG["decay"])
        for k in RECENCY_WINDOWS:
            values.update(window_features(groups, items, hist_short, vectors, f"log_{k}", 0, k, 1.0))
        x = values["long_50"]
        for k in RECENCY_WINDOWS:
            y = values[f"log_{k}"]
            if part == "fit":
                slopes[k] = {"short_on_long": within_slope(groups, x, y), "long_on_short": within_slope(groups, y, x)}
            values[f"resid_log_{k}"] = (y - slopes[k]["short_on_long"] * x).astype(np.float32)
            values[f"resid_long_given_log_{k}"] = (x - slopes[k]["long_on_short"] * y).astype(np.float32)
        tables.append(tidy(signal_table(groups, labels, values, SIGNAL_ALPHA, users), part, "earlier click >= 1"))
        null = signal_table(groups, permute_within_groups(groups, labels, SEED), values, SIGNAL_ALPHA, users)
        nulls[part] = {"both_levels": int(null["significant_both"].sum()), "of": len(null),
                       "max_abs_null_dz": round(float(null["dz"].abs().max()), 4)}

        # D41 (post-hoc): same residual built from a different user's recent clicks at nearly the same time
        partner = time_matched_partner(info["time"].to_numpy()[sel], users)
        hist_partner = [hist_short[j] for j in partner]
        keep = ~repeated_candidates(groups, items, hist_partner)
        placebo = {}
        for k in RECENCY_WINDOWS:
            y = window_features(groups, items, hist_partner, vectors, f"partner_{k}", 0, k, 1.0)[f"partner_{k}"]
            if part == "fit":
                slopes[k]["partner_on_long"] = within_slope(groups, x, y)
            placebo[f"resid_partner_{k}"] = (y - slopes[k]["partner_on_long"] * x).astype(np.float32)
            placebo[f"resid_log_{k}"] = values[f"resid_log_{k}"]
        tables.append(tidy(signal_table(groups[keep], labels[keep], MaskedValues(placebo, keep), SIGNAL_ALPHA, users), part,
                           "time-matched placebo"))
        for k in RECENCY_WINDOWS:
            comparisons.append({"part": part, "sample": "time-matched placebo", "window_a": f"resid_log_{k}",
                                "window_b": f"resid_partner_{k}",
                                **dz_difference(within_group_diffs(groups[keep], labels[keep], placebo[f"resid_log_{k}"][keep]),
                                                within_group_diffs(groups[keep], labels[keep],
                                                                   placebo[f"resid_partner_{k}"][keep]))})

        for k in RECENCY_WINDOWS:
            da = within_group_diffs(groups, labels, values[f"log_{k}{USER}"])
            db = within_group_diffs(groups, labels, values[f"long_50{USER}"])
            comparisons.append({"part": part, "sample": "earlier click >= 1", "window_a": f"log_{k}", "window_b": "long_50",
                                **dz_difference(da, db)})

        cat_rows = category[items]
        clicked = cat_rows[labels == 1]
        if part == "fit":
            top = pd.Series(clicked).value_counts().index[:SIGNAL_TOP_CATEGORIES].tolist()
        resid = {f"resid_log_{k}": values[f"resid_log_{k}"] for k in RECENCY_WINDOWS}
        parts = []
        for c in top:
            mc = cat_rows == c
            t = signal_table(groups[mc], labels[mc], MaskedValues(resid, mc), SIGNAL_ALPHA, users)
            parts.append(t.assign(category=c))
        resid_cat.append(holm_family(pd.concat(parts, ignore_index=True).assign(part=part), SIGNAL_ALPHA))
        del values, resid

        for k in RECENCY_WINDOWS:  # same-size control: last k vs the k clicks before them
            keep = n_earlier >= 2 * k
            sel2, g2, it2, lab2, us2, _ = sample(ctx, keep, repeat)
            h2 = [lists[i] for i in sel2]
            ctrl = window_features(g2, it2, h2, vectors, f"log_{k}", 0, k, 1.0)
            ctrl.update(window_features(g2, it2, h2, vectors, f"prev_{k}", k, k, 1.0))
            tables.append(tidy(signal_table(g2, lab2, ctrl, SIGNAL_ALPHA, us2), part, f"control k={k}"))
            comparisons.append({"part": part, "sample": f"control k={k}", "window_a": f"log_{k}", "window_b": f"prev_{k}",
                                **dz_difference(within_group_diffs(g2, lab2, ctrl[f"log_{k}{USER}"]),
                                                within_group_diffs(g2, lab2, ctrl[f"prev_{k}{USER}"]))})

    table = pd.concat(tables, ignore_index=True)
    comp = pd.DataFrame(comparisons)
    comp["p_holm"] = comp.groupby("part")["p"].transform(lambda s: holm(s.to_numpy()))
    cats = pd.concat(resid_cat, ignore_index=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "true_recency_signal.csv", index=False)
    comp.to_csv(TABLES / "true_recency_comparison.csv", index=False)
    cats.to_csv(TABLES / "true_recency_residual_by_category.csv", index=False)

    main_rows = table[table["sample"] == "earlier click >= 1"].set_index(["part", "feature"])
    adds = {}
    for k in RECENCY_WINDOWS:
        f = f"resid_log_{k}"
        ok = [bool(main_rows.loc[(part, f), "significant_both"] and main_rows.loc[(part, f), "dz"] > 0)
              for part in ("fit", "valid")]
        adds[f"log_{k}"] = {"adds_beyond_long_term": all(ok), "fit_dz": round(float(main_rows.loc[("fit", f), "dz"]), 4),
                            "valid_dz": round(float(main_rows.loc[("valid", f), "dz"]), 4)}
    meta = {"rule": "D39 in docs/decisions.md; time-matched placebo is post-hoc (D41)", "parts": "fit and valid (dev never read)", "coverage": coverage, "slopes": slopes,
            "adds_beyond_long_term": adds, "shuffled_label_false_positives": nulls,
            "limitations": ["analysis only: standard MIND protocol uses the history column alone (Q09)",
                            "active users only: the sample needs at least one earlier log click",
                            "emotion tracks topic: recent clicks may share a topic with candidates",
                            "a positive residual is an association, not a ranking gain"],
            "run_info": run_info()}
    (TABLES / "true_recency_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pd.set_option("display.width", 170)
    s = table[table["sample"] == "earlier click >= 1"]
    print(s.pivot_table(index=["window", "role"], columns="part", values=["dz", "user_dz"], sort=False).round(4).to_string())
    print(s.groupby("part")["significant_both"].agg(["sum", "size"]).to_string())
    print("\n" + comp.round(4).to_string(index=False))
    plc = table[table["sample"] == "time-matched placebo"]
    print("\nplacebo:\n" + plc.pivot_table(index="window", columns="part", values=["dz", "user_dz"]).round(4).to_string())
    print(plc[["part", "feature", "significant_both"]].to_string(index=False))
    ctrl = table[table["sample"].str.startswith("control") & (table["role"] == "user-specific")]
    print("\n" + ctrl.pivot_table(index=["sample", "window"], columns="part", values="dz", sort=False).round(4).to_string())
    print("\nresidual by category (dz):\n" + cats.pivot_table(index=["feature", "category"], columns="part", values="dz",
                                                             sort=False).round(3).to_string())
    print(f"\nadds beyond long-term: {json.dumps(adds, indent=1)}\nshuffled labels: {nulls}\nslopes: {slopes}")


if __name__ == "__main__":
    main()
