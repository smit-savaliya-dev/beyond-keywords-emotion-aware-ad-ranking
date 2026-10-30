"""Step 1 of base-v2: title-weight / threshold grid on the SAVED raw 28 scores (no GPU), selection rule D30.
Neutral and class shares per (weight, t); personal congruence per weight on fit and valid (dev is never read).

  python scripts/30_emotion_v2_grid.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.click_signal import permute_within_groups, signal_table
from src.config import (PROCESSED, RESULTS, SEED, SIGNAL_ALPHA, V2_CURRENT_VALID_DZ, V2_T_GRID, V2_TITLE_SHARES,
                        V2_MAIN_MEASURE)
from src.congruence import congruence_features
from src.emotion_features import GO_EMOTIONS, go_features, go_percentiles
from src.emotion_v2 import class_shares, combine_share, intensity_table, load_part_context, select_setting, threshold_labels
from src.provenance import run_info

TABLES = RESULTS / "tables"
NAME = "cong_pct_cos"


def pct_vectors(raw):
    return go_percentiles(go_features(raw))[[f"go_pct_{e}" for e in GO_EMOTIONS]].to_numpy(np.float64)


def congruence_table(ctx, vectors, shuffle=False):
    values, _ = congruence_features(ctx.groups, ctx.item_idx, ctx.histories, {"pct": vectors}, V2_MAIN_MEASURE,
                                    shuffle_seed=SEED)
    labels = permute_within_groups(ctx.groups, ctx.labels, SEED) if shuffle else ctx.labels
    return signal_table(ctx.groups, labels, values, SIGNAL_ALPHA, ctx.group_users).set_index("feature")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    p = args.processed

    title = pd.read_parquet(p / "news_emotion_raw28_title.parquet")
    abstract = pd.read_parquet(p / "news_emotion_raw28_abstract.parquet")
    item_index = pd.Index(title["news_id"])
    ctx = {part: load_part_context(p, part, item_index) for part in ("fit", "valid")}

    grid, weights = [], []
    vectors = {}
    for share in V2_TITLE_SHARES:
        raw = combine_share(title, abstract, share)
        table = intensity_table(raw)
        vectors[share] = pct_vectors(raw)
        res = {part: congruence_table(c, vectors[share]) for part, c in ctx.items()}
        row = {"share": round(share, 4)}
        for part, t in res.items():
            for role in ("", "_generic", "_personal", "_otheruser", "_userspecific"):
                row[f"{part}{role or '_raw'}_dz"] = float(t.loc[f"{NAME}{role}", "dz"])
            row[f"{part}_personal_user_dz"] = float(t.loc[f"{NAME}_personal", "user_dz"])
            row[f"{part}_both"] = bool(t.loc[f"{NAME}_personal", "significant_both"])
        weights.append(row)
        print(f"share {share:.3f}: personal dz fit {row['fit_personal_dz']:.4f}, valid {row['valid_personal_dz']:.4f}; "
              f"user-specific fit {row['fit_userspecific_dz']:.4f}, valid {row['valid_userspecific_dz']:.4f}")
        for t_value in V2_T_GRID:
            grid.append({"share": round(share, 4), "t": t_value, **class_shares(threshold_labels(table, t_value)),
                         "fit_dz": row["fit_personal_dz"], "valid_dz": row["valid_personal_dz"]})
    grid, weights = pd.DataFrame(grid), pd.DataFrame(weights)

    current = weights.loc[weights["share"].sub(round(1 / 3, 4)).abs() < 1e-9].iloc[0]
    if abs(current["valid_personal_dz"] - V2_CURRENT_VALID_DZ) > 1e-3:
        raise ValueError(f"current weight gives valid dz {current['valid_personal_dz']:.5f}, "
                         f"expected {V2_CURRENT_VALID_DZ}: the grid does not reproduce v1")
    chosen, tier, passing = select_setting(grid, V2_CURRENT_VALID_DZ)
    grid["passes"] = grid.set_index(["share", "t"]).index.isin(passing.set_index(["share", "t"]).index)
    if chosen is None:
        raise SystemExit("no setting passes D30: v1 stays, no v2 labelling is adopted")

    key = min(V2_TITLE_SHARES, key=lambda s: abs(round(s, 4) - chosen["share"]))
    null = {part: congruence_table(c, vectors[key], shuffle=True) for part, c in ctx.items()}
    TABLES.mkdir(parents=True, exist_ok=True)
    grid.round(4).to_csv(TABLES / "emotion_v2_grid.csv", index=False)
    weights.round(4).to_csv(TABLES / "emotion_v2_weight_congruence.csv", index=False)
    meta = {"rule": "D30 in docs/decisions.md", "chosen_title_share": key, "chosen_t": float(chosen["t"]),
            "neutral_tier_used": tier, "pct_neutral": round(float(chosen["pct_neutral"]), 2),
            "fit_personal_dz": round(float(chosen["fit_dz"]), 4), "valid_personal_dz": round(float(chosen["valid_dz"]), 4),
            "valid_gate": round(0.9 * V2_CURRENT_VALID_DZ, 5), "n_passing_settings": int(len(passing)),
            "shuffled_label_false_positives": {part: int(t["significant_both"].sum()) for part, t in null.items()},
            "part_used": "fit + valid (dev never read)", "run_info": run_info()}
    (TABLES / "emotion_v2_choice.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pd.set_option("display.width", 170)
    print("\n" + grid.pivot(index="share", columns="t", values="pct_neutral").round(1).to_string())
    print("\n" + weights[["share", "fit_raw_dz", "fit_personal_dz", "fit_userspecific_dz", "valid_raw_dz",
                          "valid_personal_dz", "valid_userspecific_dz"]].round(4).to_string(index=False))
    print(f"\nchosen: {json.dumps({k: v for k, v in meta.items() if k != 'run_info'}, indent=1)}")


if __name__ == "__main__":
    main()
