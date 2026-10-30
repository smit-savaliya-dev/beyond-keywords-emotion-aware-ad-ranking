"""M3 = M2 + emotion, late fusion (spec v3): final = alpha * norm(M2) + (1 - alpha) * norm(emotion), min-max per impression. D52-D54.

  python scripts/45_model_m3_fusion.py --tune         # valid only: 4 variants x 11 alphas, checks, title-only reported
  python scripts/45_model_m3_fusion.py --final-dev    # once, frozen settings
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.baselines import load_part
from src.click_signal import permute_within_groups
from src.config import (DIAGNOSTICS, GO_PCT_COLUMNS, M2_NAME, M3_K_OTHER, M3_NAME, M3_REPORT_ONLY, M3_VARIANTS, PROCESSED, RESULTS,
                        SEED, SELECTION_METRIC, TABLES)
from src.congruence import history_index_lists, other_user_cosine
from src.emotion_features import GO_EMOTIONS
from src.emotion_spec import load_spec
from src.evaluate import attach_scores, evaluate, save_results
from src.experiments import append_run, check_final_dev, frozen_marker, load_frozen, pick_best, save_frozen
from src.fusion import emotion_scores, fit_profile_pool, fuse, grid_configs, minmax_within, same_metrics, shuffled_control
from src.provenance import run_info
from src.story_charts import alpha_curve


def load_inputs(part, processed, need_other):
    """Harness truth, normalised M2, and the emotion score variants per candidate, all in truth order."""
    feats = pd.read_parquet(processed / "news_emotion_features_v2.parquet", columns=["news_id", *GO_PCT_COLUMNS])
    index = pd.Index(feats["news_id"])
    vectors = feats[GO_PCT_COLUMNS].to_numpy(np.float64)
    truth, groups, histories, users = load_part(part, processed)
    item_idx = index.get_indexer(truth["news_id"])
    if (item_idx < 0).any():
        raise ValueError("candidates without emotion vectors")
    hist_rows = history_index_lists(histories, index)
    other = None
    if need_other:
        pool, pool_users = fit_profile_pool(index, vectors, processed)
        other = other_user_cosine(groups, item_idx, vectors, pool, pool_users, users, M3_K_OTHER, SEED)
        print(f"K = {M3_K_OTHER} other users from {len(pool):,} fit profiles")
    scores = emotion_scores(groups, item_idx, hist_rows, vectors, other)
    m2 = pd.read_parquet(RESULTS / "scores" / f"{M2_NAME}__{part}.parquet")
    base = attach_scores(truth, m2)["score"].to_numpy()
    imps = pd.factorize(truth["imp_key"])[0]
    return truth, imps, minmax_within(imps, base), scores, (index, groups, item_idx, hist_rows)


def title_only_score(processed, ctx):
    """The spec's pre-declared title-only congruence (D52): reported, never selected."""
    index, groups, item_idx, hist_rows = ctx
    cols = [f"go_title_pct_{e}" for e in GO_EMOTIONS]
    t = pd.read_parquet(processed / "news_emotion_features.parquet", columns=["news_id", *cols]).set_index("news_id").reindex(index)
    return emotion_scores(groups, item_idx, hist_rows, t[cols].to_numpy(np.float64))["cosine"]


def tune(processed):
    alphas = load_spec()["fusion"]["alpha_grid"]
    truth, imps, base, scores, ctx = load_inputs("valid", processed, need_other=True)
    per_m2 = pd.read_parquet(RESULTS / "per_impression" / f"{M2_NAME}__valid.parquet")
    keys = set(per_m2["imp_key"])
    norm = {v: minmax_within(imps, s) for v, s in scores.items()}
    norm[M3_REPORT_ONLY] = minmax_within(imps, title_only_score(processed, ctx))

    def run(emotion, alpha):
        out = truth[["imp_key", "news_id"]].assign(score=fuse(base, emotion, alpha))
        per, summary = evaluate(truth, out)
        return out, per, summary

    configs = grid_configs(alphas)
    runs = []
    alpha_one, same_imps = {}, True
    for cfg in configs:
        out, per, summary = run(norm[cfg["variant"]], cfg["alpha"])
        same_imps &= set(per["imp_key"]) == keys
        if cfg["alpha"] == 1.0:
            alpha_one[cfg["variant"]] = same_metrics(per_m2, per)
        runs.append((cfg, summary, out, per))
        print(f"  {cfg}: {SELECTION_METRIC} {summary[SELECTION_METRIC]:.5f}")
    grid = [{**cfg, "eligible": True, **s.round(6).to_dict()} for cfg, s, _, _ in runs]
    for alpha in sorted(alphas, reverse=True):
        _, _, summary = run(norm[M3_REPORT_ONLY], alpha)
        grid.append({"variant": M3_REPORT_ONLY, "alpha": alpha, "eligible": False, **summary.round(6).to_dict()})
    grid = pd.DataFrame(grid)
    TABLES.mkdir(parents=True, exist_ok=True)
    grid.to_csv(TABLES / "m3_fusion_grid_valid.csv", index=False)

    best_cfg, best, best_scores, best_per = runs[pick_best([r[1] for r in runs], SELECTION_METRIC)]
    shuffled = permute_within_groups(imps, norm[best_cfg["variant"]], SEED)
    per_shuffled = {a: run(shuffled, a)[1] for a in alphas if a < 1.0}
    control, control_ok = shuffled_control(per_m2, per_shuffled)
    control.to_csv(TABLES / "m3_shuffled_control_valid.csv", index=False)

    checks = {"alpha_1_reproduces_m2": {v: {"identical": r[0], "max_abs_diff": r[1]} for v, r in alpha_one.items()},
              "identical_impressions": bool(same_imps), "shuffled_control_passes": control_ok,
              "shuffled_control_best_mean_diff": float(control["mean_diff"].max())}
    alpha_curve(grid, float(per_m2[SELECTION_METRIC].mean()), DIAGNOSTICS / "m3_alpha_curve.png", "Valid AUC as emotion gets more weight",
                "Each line blends M2 with one emotion score. At alpha = 1 every line is M2 alone; at alpha = 0 it is emotion alone. "
                "Title-only is reported, not eligible for selection. Chosen on valid; dev not used.")
    print("\n" + control.round(5)[["alpha", "mean_diff", "ci_low", "ci_high", "p_holm", "beats_m2"]].to_string(index=False))
    print(json.dumps(checks, indent=2))
    if not (all(r[0] for r in alpha_one.values()) and same_imps and control_ok):
        raise SystemExit("a mandatory check failed (D54): nothing frozen")

    for cfg, summary, _, _ in runs:
        append_run(M3_NAME, "valid", "tune-grid", cfg, summary)
    save_results(M3_NAME, "valid", best_scores, best_per, best)
    append_run(M3_NAME, "valid", "tune-chosen", best_cfg, best)
    config = {**best_cfg, "m2_config_hash": load_frozen(M2_NAME)["config_hash"]}
    doc = save_frozen(M3_NAME, config, {"selection": f"max valid {SELECTION_METRIC} over {len(configs)} configs (D53)", "checks": checks})
    meta = {"chosen": best_cfg, "valid_auc": float(best[SELECTION_METRIC]), "m2_valid_auc": float(per_m2[SELECTION_METRIC].mean()),
            "checks": checks, "n_configs": len(configs), "variants": M3_VARIANTS, "report_only": M3_REPORT_ONLY,
            "k_other": M3_K_OTHER, "run_info": run_info()}
    (TABLES / "m3_fusion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nchosen {best_cfg} (valid {SELECTION_METRIC} {best[SELECTION_METRIC]:.5f}); "
          f"record '{frozen_marker(M3_NAME, doc['config_hash'])}' in docs/decisions.md before --final-dev")


def final_dev(processed, reason):
    frozen = check_final_dev(M3_NAME, reason)
    cfg = frozen["config"]
    if cfg["m2_config_hash"] != load_frozen(M2_NAME)["config_hash"]:
        raise SystemExit("M2 changed after M3 was frozen")
    truth, imps, base, scores, _ = load_inputs("dev", processed, need_other=cfg["variant"] == "user_specific")
    out = truth[["imp_key", "news_id"]].assign(score=fuse(base, minmax_within(imps, scores[cfg["variant"]]), cfg["alpha"]))
    per, summary = evaluate(truth, out)
    save_results(M3_NAME, "dev", out, per, summary)
    append_run(M3_NAME, "dev", "final-dev", cfg, summary, note=reason or "")
    (TABLES / "m3_fusion_dev_meta.json").write_text(json.dumps({"config": cfg, "metrics": summary.round(4).to_dict()}, indent=2),
                                                    encoding="utf-8")
    print(summary.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tune", action="store_true")
    mode.add_argument("--final-dev", action="store_true")
    ap.add_argument("--rerun-dev", metavar="REASON")
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    if args.tune:
        tune(args.processed)
    else:
        final_dev(args.processed, args.rerun_dev)


if __name__ == "__main__":
    main()
