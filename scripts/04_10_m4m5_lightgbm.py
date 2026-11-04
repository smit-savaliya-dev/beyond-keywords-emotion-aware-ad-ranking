"""M4 LightGBM (no emotion), M5 = M4 + emotion (spec v4), and M5-all (robustness, valid only). D63-D67.

Reads:  impressions and the feature tables from 04_09.
Writes: --tune: frozen settings of M4, M5 and M5-all, results/tables/m4m5_tuning_valid.csv, m4m5_seed_variability_valid.csv,
        m4m5_meta.json, boosters in data/processed/ltr_models/, valid scores and per-impression files. Stops if a D67 check fails.
        --final-dev: dev files for M4 or M5, a summary.csv row and a runs_log.csv row; the frozen booster is loaded, never retrained.
Run:    python scripts/04_10_m4m5_lightgbm.py --tune             # valid only: 8 configs per model, seeds, shuffled control, checks
        python scripts/04_10_m4m5_lightgbm.py --tune --resume    # reuse finished runs after an interruption
        python scripts/04_10_m4m5_lightgbm.py --final-dev --model M4_lightgbm    # once, by hand
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import (LEAKAGE_AUC_WARN, LTR_FEATURE_FILE, LTR_SEEDS, M4_NAME, M5_ALL_NAME, M5_FAIR_NAME, M5_NAME, PROCESSED, RESULTS,
                        SEED, SELECTION_METRIC, SIGNAL_ALPHA, TABLES)
from src.evaluate import load_truth, save_results
from src.run_registry import append_run, check_final_dev, frozen_marker, pick_best, save_frozen
from src.lightgbm_ranker import (build_features, check_saved_scores, emotion_columns, file_sha, fit_or_load, grid_configs, load_model, model_features, model_path,
                     save_model_copy, score_frame as score, shuffle_within_impressions)
from src.provenance import run_info
from src.stats import paired_test

MODELS = [M4_NAME, M5_NAME, M5_ALL_NAME]


def tune(processed, resume):
    """Tune the three models on valid (grid, seeds, shuffled control), run the D67 checks and freeze the settings."""
    fit_path, valid_path = (processed / LTR_FEATURE_FILE.format(part=p) for p in ("fit", "valid"))
    fit, valid = pd.read_parquet(fit_path), pd.read_parquet(valid_path)
    truth = load_truth("valid", processed)
    if truth["imp_key"].tolist() != valid["imp_key"].tolist():
        raise SystemExit("feature rows are not in harness order")
    sha = file_sha(fit_path, valid_path)
    start = time.time()

    def run(frame_fit, frame_valid, features, cfg, seed, tag=""):
        booster, info = fit_or_load(frame_fit, frame_valid, features, cfg, seed, tag, sha, resume)
        print(f"  [{(time.time() - start) / 60:.1f} min] {tag or 'run'} seed {seed} {cfg}: {len(features)} features, "
              f"{info['best_iteration']} rounds, valid AUC {info['valid_auc']:.5f}, {info['seconds']:.0f}s", flush=True)
        return booster, info

    grid, chosen, same_impressions = [], {}, True
    keys = set(truth["imp_key"])
    for model in MODELS:
        features = model_features(model)
        print(model, flush=True)
        runs = []
        for cfg in grid_configs():
            booster, info = run(fit, valid, features, cfg, SEED)
            out, per, summary = score(booster, valid, features, truth)
            same_impressions &= set(per["imp_key"]) == keys
            append_run(model, "valid", "tune-grid", cfg, summary)
            runs.append((cfg, summary, info))
            grid.append({"model": model, **cfg, "best_iteration": info["best_iteration"], "train_rows": info["train_rows"],
                         "seconds": round(info["seconds"]), **summary.round(6).to_dict()})
        best = pick_best([r[1] for r in runs], SELECTION_METRIC)
        chosen[model] = runs[best]
    grid = pd.DataFrame(grid)
    TABLES.mkdir(parents=True, exist_ok=True)
    grid.to_csv(TABLES / "m4m5_tuning_valid.csv", index=False)

    m4_cfg = chosen[M4_NAME][0]
    results = {}
    for model in MODELS:
        cfg, summary, info = chosen[model]
        features = model_features(model)
        booster, _ = fit_or_load(fit, valid, features, cfg, SEED, "", sha, True)
        out, per, summary = score(booster, valid, features, truth)
        save_results(model, "valid", out, per, summary)
        append_run(model, "valid", "tune-chosen", cfg, summary)
        results[model] = per
        info["model_sha256"] = save_model_copy(info, model)

    if chosen[M4_NAME][1][SELECTION_METRIC] > LEAKAGE_AUC_WARN:
        raise SystemExit(f"M4 valid AUC above {LEAKAGE_AUC_WARN}: check for leakage first (D64); nothing frozen")

    # fairness: M5 at M4's hyperparameters, already in the grid
    features = model_features(M5_NAME)
    booster, _ = fit_or_load(fit, valid, features, m4_cfg, SEED, "", sha, True)
    _, per_fair, summary_fair = score(booster, valid, features, truth)
    per_fair.to_parquet(RESULTS / "per_impression" / f"{M5_FAIR_NAME}__valid.parquet", index=False)

    # seed variability
    seeds = []
    for model in MODELS:
        cfg = chosen[model][0]
        features = model_features(model)
        for seed in LTR_SEEDS:
            if seed == SEED:
                summary = chosen[model][1]
                rounds = chosen[model][2]["best_iteration"]
            else:
                booster, info = run(fit, valid, features, cfg, seed)
                _, _, summary = score(booster, valid, features, truth)
                rounds = info["best_iteration"]
            seeds.append({"model": model, "seed": seed, "best_iteration": rounds, **summary.round(6).to_dict()})
    seeds = pd.DataFrame(seeds)
    seeds.to_csv(TABLES / "m4m5_seed_variability_valid.csv", index=False)

    # shuffled-emotion control (D67d): M5 at M4's hyperparameters with the emotion columns permuted within impressions
    columns = emotion_columns(M5_NAME)
    features = model_features(M5_NAME)
    fit_s, valid_s = shuffle_within_impressions(fit, columns, SEED), shuffle_within_impressions(valid, columns, SEED)
    booster, _ = run(fit_s, valid_s, features, m4_cfg, SEED, "shuffled")
    _, per_shuffled, summary_shuffled = score(booster, valid_s, features, truth)
    m = results[M4_NAME].merge(per_shuffled, on="imp_key", suffixes=("_m4", "_s"), validate="one_to_one")
    control = paired_test(m["auc_m4"], m["auc_s"])
    control_ok = not (control["mean_diff"] > 0 and control["p"] < SIGNAL_ALPHA)

    checks = {"identical_impressions": bool(same_impressions), "m4_below_leakage_limit": bool(chosen[M4_NAME][1][SELECTION_METRIC] <= LEAKAGE_AUC_WARN),
              "shuffled_control_passes": control_ok, "shuffled_auc": float(summary_shuffled["auc"]), "shuffled_vs_m4": control}
    print(json.dumps({k: v for k, v in checks.items() if k != "shuffled_vs_m4"}, indent=2))
    print("shuffled minus M4:", round(control["mean_diff"], 5), "CI", round(control["ci_low"], 5), round(control["ci_high"], 5), "p", round(control["p"], 4))
    if not (same_impressions and control_ok):
        raise SystemExit("a mandatory check failed (D67): nothing frozen")

    meta = {"checks": checks, "fair_m5_auc": float(summary_fair["auc"]), "fair_config": m4_cfg, "chosen": {}, "run_info": run_info()}
    for model in MODELS:
        cfg, summary, info = chosen[model]
        config = {**cfg, "best_iteration": info["best_iteration"], "seed": SEED, "n_features": len(model_features(model)), "model_sha256": info["model_sha256"]}
        doc = save_frozen(model, config, {"selection": f"max valid {SELECTION_METRIC} over {len(grid_configs())} configs (D66)", "features": model_features(model),
                                          "train_rows": info["train_rows"], "checks": checks})
        meta["chosen"][model] = {**cfg, "best_iteration": info["best_iteration"], "valid_auc": float(summary["auc"]), "config_hash": doc["config_hash"]}
        print(f"{model}: {cfg}, {info['best_iteration']} rounds, valid AUC {summary['auc']:.5f}; marker {frozen_marker(model, doc['config_hash'])}")
    (TABLES / "m4m5_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"total {(time.time() - start) / 60:.1f} min")


def final_dev(model, processed, reason):
    """Score dev once with the frozen booster of M4 or M5 (refuses to run twice)."""
    if model not in (M4_NAME, M5_NAME):
        raise SystemExit("only M4 and M5 run on dev; M5-all is valid only (D65)")
    frozen = check_final_dev(model, reason)
    booster = load_model(model, frozen["config"]["model_sha256"])
    frame = build_features("dev", processed)
    print("largest gap to the saved scores:", check_saved_scores(frame, "dev"))
    out, per, summary = score(booster, frame, model_features(model), load_truth("dev", processed))
    save_results(model, "dev", out, per, summary)
    append_run(model, "dev", "final-dev", frozen["config"], summary, note=reason or "")
    (TABLES / f"{model}_dev_meta.json").write_text(json.dumps({"config": frozen["config"], "model_file": model_path(model).name,
                                                               "metrics": summary.round(4).to_dict()}, indent=2), encoding="utf-8")
    print(summary.round(4).to_string())


def main():
    """Run --tune (valid) or --final-dev (dev, once)."""
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tune", action="store_true")
    mode.add_argument("--final-dev", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse finished runs; a pipeline rebuild never does")
    ap.add_argument("--model", choices=[M4_NAME, M5_NAME])
    ap.add_argument("--rerun-dev", metavar="REASON")
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    if args.tune:
        tune(args.processed, args.resume)
    else:
        if not args.model:
            raise SystemExit("--final-dev needs --model")
        final_dev(args.model, args.processed, args.rerun_dev)


if __name__ == "__main__":
    main()
