"""Valid-set replication of the emotion spec candidates (D27). Same statistics as Phase 5 (per-impression
clicked-minus-unclicked dz, t-test, Holm, user level, shuffled-label null), on part == "valid" only; dev is never read.
Rule (pre-declared): keep a candidate if it is significant at both levels on fit AND on valid has the same sign with an
impression-level 95% CI that excludes 0.

  python scripts/32_analysis_spec_v1_replication.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.click_signal import load_part_context, permute_within_groups, signal_table
from src.config import (CONGRUENCE_MEASURES, PROCESSED, TABLES, SEED, SIGNAL_ALPHA, SPEC_CONGRUENCE_CANDIDATES,
                        SPEC_ITEM_CANDIDATES)
from src.congruence import congruence_features
from src.emotion import SIX_CLASSES
from src.emotion_spec import replication_selection
from src.emotion_features import GO_EMOTIONS
from src.provenance import run_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    p = args.processed

    pct_cols = [f"go_pct_{e}" for e in GO_EMOTIONS]
    feats = pd.read_parquet(p / "news_emotion_features.parquet",
                            columns=["news_id", *dict.fromkeys(SPEC_ITEM_CANDIDATES + pct_cols + ["go_vad_valence"])])
    ctx = load_part_context(p, "valid", pd.Index(feats["news_id"]))
    groups, group_users, labels, idx, imp_keys = ctx.groups, ctx.group_users, ctx.labels, ctx.item_idx, ctx.imp_keys
    values = {c: feats[c].to_numpy(np.float64)[idx] for c in SPEC_ITEM_CANDIDATES}

    six = pd.read_parquet(p / "news_emotion_weighted.parquet").set_index("news_id").reindex(feats["news_id"])
    vectors = {"pct": feats[pct_cols].to_numpy(np.float64), "six": six[SIX_CLASSES].to_numpy(np.float64),
               "valence": feats["go_vad_valence"].to_numpy(np.float64)}
    measures = [m for m in CONGRUENCE_MEASURES if f"cong_{m[0]}_{m[1]}" in
                {c.replace("_personal", "") for c in SPEC_CONGRUENCE_CANDIDATES}]
    cong, used = congruence_features(groups, idx, ctx.histories, vectors, measures)
    values.update({c: cong[c] for c in SPEC_CONGRUENCE_CANDIDATES})
    print(f"valid: {len(imp_keys):,} impressions, {len(set(group_users)):,} users, {len(groups):,} candidates; "
          f"cold start {100 * float((used == 0).mean()):.2f}%")

    valid_t = signal_table(groups, labels, values, SIGNAL_ALPHA, group_users).set_index("feature")
    null_t = signal_table(groups, permute_within_groups(groups, labels, SEED), values, SIGNAL_ALPHA, group_users)
    fit_item = pd.read_csv(TABLES / "emotion_signal_fit.csv").set_index("feature")
    fit_cong = pd.read_csv(TABLES / "congruence_signal_fit.csv")
    fit_cong = fit_cong[fit_cong["sample"] == "all"].set_index("feature")
    fit = pd.concat([fit_item.loc[SPEC_ITEM_CANDIDATES], fit_cong.loc[SPEC_CONGRUENCE_CANDIDATES]])
    sel = replication_selection(fit, valid_t.loc[list(values)])
    sel.insert(0, "group", ["item"] * len(SPEC_ITEM_CANDIDATES) + ["congruence"] * len(SPEC_CONGRUENCE_CANDIDATES))
    sel.index.name = "feature"

    TABLES.mkdir(parents=True, exist_ok=True)
    sel.reset_index().to_csv(TABLES / "emotion_spec_selection.csv", index=False)
    valid_t.reset_index().to_csv(TABLES / "emotion_signal_valid.csv", index=False)
    false_pos = {"both_levels": int(null_t["significant_both"].sum()), "of": len(null_t),
                 "max_abs_null_dz": round(float(null_t["dz"].abs().max()), 4)}
    meta = {"part": "valid only (dev never read)", "rule": "D27 in docs/decisions.md",
            "n_impressions": len(imp_keys), "kept": sel.index[sel["keep"]].tolist(),
            "dropped": dict(zip(sel.index[~sel["keep"]], sel.loc[~sel["keep"], "reason"])),
            "shuffled_label_false_positives": false_pos, "run_info": run_info()}
    (TABLES / "emotion_spec_selection_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.set_option("display.width", 170)
    print(sel.round(4).to_string())
    print(f"\nshuffled-label null on valid: {false_pos}")
    print(f"kept {int(sel['keep'].sum())}/{len(sel)}; dropped: {meta['dropped'] or 'none'}")


if __name__ == "__main__":
    main()
