"""D32: re-apply the D27 replication rule on the v2 features (fit, then valid; dev is never read).
Writes the clicked-vs-not signal table used by the story charts and the candidate selection behind spec v2.

  python scripts/33_analysis_spec_v2_replication.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.click_signal import load_part_context, permute_within_groups, signal_table
from src.config import (PROCESSED, TABLES, SEED, SIGNAL_ALPHA, SPEC_CONGRUENCE_CANDIDATES, SPEC_ITEM_CANDIDATES,
                        V2_ITEM_EXTRA)
from src.congruence import congruence_features
from src.emotion import GOEMOTIONS_LABELS, collapse
from src.emotion_features import GO_EMOTIONS
from src.emotion_spec import replication_selection
from src.provenance import run_info

MEASURES = [("pct", "cos", "main"), ("six", "negkl", "variant"), ("valence", "absdist", "variant")]
CONTROLS = ["cong_pct_cos_otheruser", "cong_pct_cos_userspecific"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    p = args.processed

    items = SPEC_ITEM_CANDIDATES + V2_ITEM_EXTRA
    pct_cols = [f"go_pct_{e}" for e in GO_EMOTIONS]
    feats = pd.read_parquet(p / "news_emotion_features_v2.parquet",
                            columns=["news_id", *dict.fromkeys(items + pct_cols + ["go_vad_valence"])])
    raw = pd.read_parquet(p / "news_emotion_raw28_v2.parquet", columns=["news_id", *GOEMOTIONS_LABELS])
    if raw["news_id"].tolist() != feats["news_id"].tolist():
        raise ValueError("raw and feature tables are not aligned")
    index = pd.Index(feats["news_id"])
    vectors = {"pct": feats[pct_cols].to_numpy(np.float64), "six": collapse(raw[GOEMOTIONS_LABELS].to_numpy()),
               "valence": feats["go_vad_valence"].to_numpy(np.float64)}
    item_mat = feats[items].to_numpy(np.float32)
    names = items + SPEC_CONGRUENCE_CANDIDATES + CONTROLS

    tables, nulls = {}, {}
    for part in ("fit", "valid"):
        ctx = load_part_context(p, part, index)
        cong, _ = congruence_features(ctx.groups, ctx.item_idx, ctx.histories, vectors, MEASURES, shuffle_seed=SEED)
        values = {c: item_mat[ctx.item_idx, j] for j, c in enumerate(items)}
        values.update({c: cong[c] for c in SPEC_CONGRUENCE_CANDIDATES + CONTROLS})
        tables[part] = signal_table(ctx.groups, ctx.labels, values, SIGNAL_ALPHA, ctx.group_users).set_index("feature")
        null = signal_table(ctx.groups, permute_within_groups(ctx.groups, ctx.labels, SEED), values, SIGNAL_ALPHA,
                            ctx.group_users)
        nulls[part] = {"both_levels": int(null["significant_both"].sum()), "of": len(null),
                       "max_abs_null_dz": round(float(null["dz"].abs().max()), 4)}
        print(f"{part}: {ctx.n_impressions:,} impressions; shuffled labels {nulls[part]}")

    cands = items + SPEC_CONGRUENCE_CANDIDATES
    sel = replication_selection(tables["fit"].loc[cands], tables["valid"].loc[cands])
    sel.insert(0, "group", ["item"] * len(items) + ["congruence"] * len(SPEC_CONGRUENCE_CANDIDATES))
    sel.index.name = "feature"

    TABLES.mkdir(parents=True, exist_ok=True)
    long = pd.concat([t.loc[names].reset_index().assign(part=part) for part, t in tables.items()], ignore_index=True)
    long.to_csv(TABLES / "emotion_signal_v2.csv", index=False)
    sel.reset_index().to_csv(TABLES / "emotion_spec_v2_selection.csv", index=False)
    meta = {"part": "fit + valid (dev never read)", "rule": "D32 (D27 rule on v2 features)",
            "kept": sel.index[sel["keep"]].tolist(),
            "dropped": dict(zip(sel.index[~sel["keep"]], sel.loc[~sel["keep"], "reason"])),
            "shuffled_label_false_positives": nulls, "run_info": run_info()}
    (TABLES / "emotion_spec_v2_selection_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.set_option("display.width", 170)
    print(sel[["group", "fit_dz", "fit_user_dz", "valid_dz", "valid_ci_low", "valid_ci_high", "keep", "reason"]]
          .round(4).to_string())
    print(f"\nkept {int(sel['keep'].sum())}/{len(sel)}")


if __name__ == "__main__":
    main()
