"""M2 Sentence-BERT, no emotion: user vector = mean or decay-weighted mean of the last 50 history embeddings;
score = cosine(user, candidate); cold start scores 0 (ties are broken by the harness). D51.

Reads:  impressions, behaviors, news.parquet and the embeddings from 04_04.
Writes: --tune: results/frozen/M2_sbert.json, results/tables/m2_sbert_tuning_valid.csv, valid scores and per-impression files.
        --final-dev: dev files, a summary.csv row and a runs_log.csv row.
Run:    python scripts/04_05_m2_sbert.py --tune         # valid only: 4 configs (M2_GRID), pick by valid AUC
        python scripts/04_05_m2_sbert.py --final-dev    # once, by hand, frozen settings
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.baselines import load_part, profile_weights
from src.config import LEAKAGE_AUC_WARN, M2_GRID, M2_NAME, PROCESSED, SBERT_MODEL, SBERT_REVISION, SELECTION_METRIC, TABLES
from src.congruence import history_index_lists
from src.evaluate import evaluate, save_results
from src.run_registry import append_run, check_final_dev, frozen_marker, pick_best, save_frozen
from src.sbert import load_embeddings, profile_cosine


def run_configs(part, processed, configs):
    """Score one part under each config of the grid and return the scores per config."""
    ids = pd.read_parquet(processed / "news.parquet", columns=["news_id"])["news_id"]
    truth, groups, histories, _ = load_part(part, processed)
    index = pd.Index(ids)
    item_idx = index.get_indexer(truth["news_id"])
    if (item_idx < 0).any():
        raise ValueError("candidates missing from news.parquet")
    hist_rows = history_index_lists(histories, index)
    cold = float(np.mean([not (np.asarray(h) >= 0).any() for h in hist_rows]))
    print(f"{part}: {len(truth):,} candidates, {len(histories):,} unique (user, history) profiles, {100 * cold:.2f}% cold start")
    weights, out = {}, []
    for text, group in itertools.groupby(configs, key=lambda c: c["text"]):
        X = load_embeddings(text, ids, processed)
        for cfg in group:
            start = time.time()
            agg = cfg["aggregation"]
            if agg not in weights:
                weights[agg] = profile_weights(hist_rows, len(ids), agg)
            U = weights[agg] @ X
            scores = truth[["imp_key", "news_id"]].assign(score=profile_cosine(U, X, groups, item_idx))
            per_imp, summary = evaluate(truth, scores)
            print(f"  {cfg}: {part} {SELECTION_METRIC} {summary[SELECTION_METRIC]:.4f} ({time.time() - start:.0f}s)")
            out.append((cfg, summary, scores, per_imp))
    info = {"encoder": SBERT_MODEL, "revision": SBERT_REVISION, "pct_cold_start_profiles": round(100 * cold, 2),
            "n_unique_profiles": len(histories)}
    return out, info


def grid_configs():
    """Every combination of the M2 grid as a list of config dicts."""
    keys = list(M2_GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(M2_GRID[k] for k in keys))]


def main():
    """Run --tune (valid) or --final-dev (dev, once)."""
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tune", action="store_true")
    mode.add_argument("--final-dev", action="store_true")
    ap.add_argument("--rerun-dev", metavar="REASON")
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()

    if args.tune:
        runs, info = run_configs("valid", args.processed, grid_configs())
        for cfg, summary, _, _ in runs:
            append_run(M2_NAME, "valid", "tune-grid", cfg, summary)
        best_cfg, best, scores, per_imp = runs[pick_best([r[1] for r in runs], SELECTION_METRIC)]
        save_results(M2_NAME, "valid", scores, per_imp, best)
        append_run(M2_NAME, "valid", "tune-chosen", best_cfg, best)
        grid = pd.DataFrame([{**cfg, **s.round(4).to_dict()} for cfg, s, _, _ in runs])
        TABLES.mkdir(parents=True, exist_ok=True)
        grid.to_csv(TABLES / "m2_sbert_tuning_valid.csv", index=False)
        warn = best[SELECTION_METRIC] > LEAKAGE_AUC_WARN
        doc = save_frozen(M2_NAME, best_cfg, {"selection": f"max valid {SELECTION_METRIC}", "grid": grid.to_dict("records"), **info,
                                              "leakage_warning": bool(warn)})
        print("\n" + grid.to_string(index=False))
        if warn:
            print(f"WARNING: valid {SELECTION_METRIC} > {LEAKAGE_AUC_WARN}: check for leakage before trusting this")
        print(f"\nchosen {best_cfg} (valid {SELECTION_METRIC} {best[SELECTION_METRIC]:.4f}); "
              f"record '{frozen_marker(M2_NAME, doc['config_hash'])}' in docs/decisions.md before --final-dev")
        return

    frozen = check_final_dev(M2_NAME, args.rerun_dev)
    runs, info = run_configs("dev", args.processed, [frozen["config"]])
    cfg, summary, scores, per_imp = runs[0]
    save_results(M2_NAME, "dev", scores, per_imp, summary)
    append_run(M2_NAME, "dev", "final-dev", cfg, summary, note=args.rerun_dev or "")
    (TABLES / "m2_sbert_dev_meta.json").write_text(json.dumps({**info, "config": cfg, "metrics": summary.round(4).to_dict()}, indent=2),
                                                   encoding="utf-8")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
