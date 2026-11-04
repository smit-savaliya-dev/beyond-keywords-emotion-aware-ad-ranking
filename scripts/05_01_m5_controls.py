"""Item-identity controls for M5 (D69): does the M5 gain come from what emotion means, or from emotion scores acting as an item ID?
All controls use M5's hyperparameters and are compared with M4 at the same hyperparameters and seed.

Reads:  impressions, the feature tables from 04_09, the frozen M5 settings, news_emotion_features_v2.parquet.
Writes: --tune: frozen settings of M5_permuted_items and M5_congruence_only, results/tables/m5_controls_valid.csv, m5_controls_meta.json,
        valid per-impression files. --final-dev: dev files for one control, a summary.csv row and a runs_log.csv row.
Run:    python scripts/05_01_m5_controls.py --tune             # valid only: 3 controls x 3 seeds
        python scripts/05_01_m5_controls.py --tune --resume    # reuse finished runs after an interruption
        python scripts/05_01_m5_controls.py --final-dev --model M5_permuted_items     # once; also M5_congruence_only
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import (LTR_FEATURE_FILE, LTR_GRID, LTR_SEEDS, M4_FEATURES, M4_SAME_NAME, M5_CONG_NAME, M5_ITEM_NAME, M5_NAME, M5_PERM_NAME,
                        PROCESSED, RESULTS, SEED, TABLES)
from src.evaluate import load_truth, save_results
from src.run_registry import append_run, check_final_dev, frozen_marker, load_frozen, save_frozen
from src.lightgbm_ranker import (build_features, emotion_columns, file_sha, fit_or_load, load_model, model_features, permuted_emotion, save_model_copy, score_frame,
                     spec_features)
from src.provenance import run_info

DEV_CONTROLS = (M5_PERM_NAME, M5_CONG_NAME)


def control_features():
    """Feature list of each control: swapped item emotion, congruence only, or item emotion only."""
    items, cong = spec_features()
    return {M5_PERM_NAME: model_features(M5_NAME), M5_CONG_NAME: [*M4_FEATURES, *cong], M5_ITEM_NAME: [*M4_FEATURES, *items]}


def with_permuted(frame, part, processed):
    """Copy of a feature frame whose emotion columns are swapped between items."""
    block = permuted_emotion(part, processed)
    out = frame.copy()
    for c in emotion_columns(M5_NAME):
        out[c] = block[c].to_numpy()
    return out


def tune(processed, resume):
    """Train the controls on fit for three seeds, score valid and freeze the two dev controls."""
    fit_path, valid_path = (processed / LTR_FEATURE_FILE.format(part=p) for p in ("fit", "valid"))
    fit, valid = pd.read_parquet(fit_path), pd.read_parquet(valid_path)
    truth = load_truth("valid", processed)
    sha = file_sha(fit_path, valid_path)
    frozen = load_frozen(M5_NAME)["config"]
    params = {k: frozen[k] for k in LTR_GRID}
    fit_p, valid_p = with_permuted(fit, "fit", processed), with_permuted(valid, "valid", processed)
    features = control_features()
    rows, chosen = [], {}
    for seed in LTR_SEEDS:
        booster, info = fit_or_load(fit, valid, list(M4_FEATURES), params, seed, "", sha, resume)
        _, per_ref, ref = score_frame(booster, valid, list(M4_FEATURES), truth)
        if seed == SEED:
            per_ref.to_parquet(RESULTS / "per_impression" / f"{M4_SAME_NAME}__valid.parquet", index=False)
        booster, info = fit_or_load(fit, valid, model_features(M5_NAME), params, seed, "", sha, resume)
        _, _, m5 = score_frame(booster, valid, model_features(M5_NAME), truth)
        rows.append({"model": M4_SAME_NAME, "seed": seed, **ref.round(6).to_dict(), "gain_vs_m4_same": 0.0})
        rows.append({"model": M5_NAME, "seed": seed, "best_iteration": info["best_iteration"], **m5.round(6).to_dict(), "gain_vs_m4_same": float(m5["auc"] - ref["auc"])})
        for name, cols in features.items():
            perm = name == M5_PERM_NAME
            frame_fit, frame_valid = (fit_p, valid_p) if perm else (fit, valid)
            booster, info = fit_or_load(frame_fit, frame_valid, cols, params, seed, "permuted" if perm else "", sha, resume)
            out, per, summary = score_frame(booster, frame_valid, cols, truth)
            print(f"seed {seed} {name}: {info['best_iteration']} rounds, valid AUC {summary['auc']:.5f}, gain vs M4 {summary['auc'] - ref['auc']:+.5f}", flush=True)
            rows.append({"model": name, "seed": seed, "best_iteration": info["best_iteration"], **summary.round(6).to_dict(),
                         "gain_vs_m4_same": float(summary["auc"] - ref["auc"])})
            if seed == SEED:
                save_results(name, "valid", out, per, summary)
                append_run(name, "valid", "control", {**params, "seed": seed}, summary)
                chosen[name] = (summary, info)
        print(f"seed {seed} M4 {ref['auc']:.5f}, M5 {m5['auc']:.5f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "m5_controls_valid.csv", index=False)
    means = table.groupby("model")[["auc", "gain_vs_m4_same"]].agg(["mean", "min", "max"]).round(5)
    print(means.to_string())

    meta = {"rule": "D69", "params": params, "seed_means": {m: float(v) for m, v in table.groupby("model")["gain_vs_m4_same"].mean().items()}, "frozen": {}, "run_info": run_info()}
    for name in DEV_CONTROLS:
        summary, info = chosen[name]
        config = {**params, "best_iteration": info["best_iteration"], "seed": SEED, "n_features": len(features[name]), "model_sha256": save_model_copy(info, name)}
        doc = save_frozen(name, config, {"selection": "none: M5's hyperparameters, a pre-declared control (D69)", "features": features[name]})
        meta["frozen"][name] = doc["config_hash"]
        print(f"{name}: marker {frozen_marker(name, doc['config_hash'])}")
    (TABLES / "m5_controls_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def final_dev(model, processed, reason):
    """Score dev once with the frozen booster of one control (refuses to run twice)."""
    if model not in DEV_CONTROLS:
        raise SystemExit("only the two pre-declared controls run on dev (D69)")
    frozen = check_final_dev(model, reason)
    booster = load_model(model, frozen["config"]["model_sha256"])
    frame = build_features("dev", processed)
    if model == M5_PERM_NAME:
        frame = with_permuted(frame, "dev", processed)
    out, per, summary = score_frame(booster, frame, control_features()[model], load_truth("dev", processed))
    save_results(model, "dev", out, per, summary)
    append_run(model, "dev", "final-dev", frozen["config"], summary, note=reason or "")
    print(summary.round(4).to_string())


def main():
    """Run --tune (valid) or --final-dev (dev, once)."""
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tune", action="store_true")
    mode.add_argument("--final-dev", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse finished runs; a pipeline rebuild never does")
    ap.add_argument("--model", choices=DEV_CONTROLS)
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
