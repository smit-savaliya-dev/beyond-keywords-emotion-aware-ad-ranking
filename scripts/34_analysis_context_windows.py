"""Step 2 of base-v2 (D31): which window of the user's reading history carries the emotional context?
Windows: long-term (last 50, decay 0.9) vs short-term (last 1, 3, 5, no decay), on the v2 go_pct_* vector; cosine raw,
personal, user-specific; fit then valid (dev is never read). Control: last 5 vs the 5 items before them, same size,
on users with >= 10 history items, because small windows are noisier.
Limitation: MIND history is one list per user from before the log period, so "last 1" is not the click just before the
impression; it is the emotional context inferred from the user's most recent reading in that list.

  python scripts/34_analysis_context_windows.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.click_signal import MaskedValues, load_part_context, permute_within_groups, signal_table, within_group_diffs
from src.config import (CONGRUENCE_ROLES as ROLES, CONTEXT_CONTROL, CONTEXT_CONTROL_MIN_HISTORY, CONTEXT_WINDOWS, PROCESSED,
                        TABLES, SEED, SIGNAL_ALPHA)
from src.congruence import window_congruence
from src.emotion_features import GO_EMOTIONS
from src.provenance import run_info
from src.stats import dz_difference, holm

USER = "_userspecific"


def window_values(ctx, vectors, windows):
    """{window + role suffix: candidate values} for each window; profiles use only the history column."""
    values = {}
    for name, w in windows.items():
        values.update(window_congruence(ctx.groups, ctx.item_idx, ctx.histories, vectors, name, w, SEED))
    return values


def tidy(table, part, sample):
    t = table.copy()
    split = t["feature"].str.extract(r"^(?P<window>[a-z]+_\d+)(?P<suffix>.*)$")
    t.insert(0, "window", split["window"])
    t.insert(1, "role", split["suffix"].map(ROLES))
    t.insert(0, "part", part)
    t.insert(1, "sample", sample)
    return t


def compare(ctx, values, pairs, mask, part, sample):
    rows = []
    groups, labels = ctx.groups[mask], ctx.labels[mask]
    for a, b in pairs:
        da = within_group_diffs(groups, labels, values[f"{a}{USER}"][mask])
        db = within_group_diffs(groups, labels, values[f"{b}{USER}"][mask])
        rows.append({"part": part, "sample": sample, "window_a": a, "window_b": b, **dz_difference(da, db)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    p = args.processed

    pct_cols = [f"go_pct_{e}" for e in GO_EMOTIONS]
    feats = pd.read_parquet(p / "news_emotion_features_v2.parquet", columns=["news_id", *pct_cols])
    vectors = feats[pct_cols].to_numpy(np.float64)
    index = pd.Index(feats["news_id"])

    tables, comparisons, nulls, known = [], [], {}, {}
    for part in ("fit", "valid"):
        ctx = load_part_context(p, part, index)
        n_known = np.fromiter(((np.asarray(h) >= 0).sum() for h in ctx.histories), dtype=np.int64)
        long_enough = (n_known >= CONTEXT_CONTROL_MIN_HISTORY)[ctx.groups]
        known[part] = {"n_impressions": ctx.n_impressions,
                       "pct_impressions_cold_start": round(100 * float((n_known == 0).mean()), 2),
                       f"pct_impressions_history_ge_{CONTEXT_CONTROL_MIN_HISTORY}":
                           round(100 * float((n_known >= CONTEXT_CONTROL_MIN_HISTORY).mean()), 2)}

        main_v = window_values(ctx, vectors, CONTEXT_WINDOWS)
        tables.append(tidy(signal_table(ctx.groups, ctx.labels, main_v, SIGNAL_ALPHA, ctx.group_users), part, "all"))
        null = signal_table(ctx.groups, permute_within_groups(ctx.groups, ctx.labels, SEED), main_v, SIGNAL_ALPHA,
                            ctx.group_users)
        nulls[part] = {"both_levels": int(null["significant_both"].sum()), "of": len(null),
                       "max_abs_null_dz": round(float(null["dz"].abs().max()), 4)}
        pairs = [("short_1", "long_50"), ("short_3", "long_50"), ("short_5", "long_50")]
        comparisons += compare(ctx, main_v, pairs, np.ones(len(ctx.groups), dtype=bool), part, "all")

        control = {**main_v, **window_values(ctx, vectors, CONTEXT_CONTROL)}
        keep = [k for k in control if k.startswith(("short_5", "older_5", "long_50"))]
        sub = {k: control[k] for k in keep}
        tables.append(tidy(signal_table(ctx.groups[long_enough], ctx.labels[long_enough], MaskedValues(sub, long_enough),
                                        SIGNAL_ALPHA, ctx.group_users), part, "history >= 10 (control)"))
        comparisons += compare(ctx, control, [("short_5", "older_5")], long_enough, part, "history >= 10 (control)")
        print(f"{part}: done; shuffled labels {nulls[part]}; {known[part]}")

    table = pd.concat(tables, ignore_index=True)
    comp = pd.DataFrame(comparisons)
    comp["p_holm"] = comp.groupby("part")["p"].transform(lambda x: holm(x.to_numpy()))
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "context_windows_signal.csv", index=False)
    comp.to_csv(TABLES / "context_windows_comparison.csv", index=False)

    user = table[(table["sample"] == "all") & (table["role"] == "user-specific")].pivot(index="window", columns="part",
                                                                                        values="dz")
    best = {part: user[part].idxmax() for part in ("fit", "valid")}
    meta = {"rule": "D31 in docs/decisions.md", "vector": "v2 go_pct_* (title share 0.5)", "windows": CONTEXT_WINDOWS,
            "control": CONTEXT_CONTROL, "best_window_by_user_specific_dz": best,
            "same_winner_on_fit_and_valid": best["fit"] == best["valid"], "parts": known,
            "shuffled_label_false_positives": nulls,
            "limitation": "history is one list per user from before the log period; a short window is the newest part of "
                          "that list, not the reading just before the impression",
            "run_info": run_info()}
    (TABLES / "context_windows_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pd.set_option("display.width", 170)
    show = table[table["role"] != ""].pivot_table(index=["sample", "window", "role"], columns="part",
                                                  values=["dz", "user_dz"], sort=False)
    print(show.round(4).to_string())
    print("\nboth levels:\n" + table.groupby(["part", "sample"])["significant_both"].agg(["sum", "size"]).to_string())
    print("\n" + comp.round(4).to_string(index=False))
    print(f"\nbest window: {best}")


if __name__ == "__main__":
    main()
