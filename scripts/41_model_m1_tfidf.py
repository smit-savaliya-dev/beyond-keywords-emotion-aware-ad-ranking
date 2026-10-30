"""M1 TF-IDF: vectorizer fitted on TRAIN-split news only (label-free); user vector = mean or decay-weighted mean of the
last 50 HISTORY items; score = cosine(user, candidate); cold start -> 0 (ties broken by the harness).

  python scripts/41_model_m1_tfidf.py --tune         # valid only: 8 configs (M1_GRID), pick by valid AUC
  python scripts/41_model_m1_tfidf.py --final-dev    # once, frozen settings
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.baselines import cosine_scores, fit_tfidf, profile_weights, train_news_ids, unique_profiles
from src.config import (LEAKAGE_AUC_WARN, M1_GRID, M1_NAME, PROCESSED, TABLES, SELECTION_METRIC, TFIDF_MIN_DF,
                        TFIDF_STOP_WORDS)
from src.congruence import history_index_lists
from src.evaluate import evaluate, load_truth, save_results
from src.experiments import append_run, check_final_dev, frozen_marker, pick_best, save_frozen


def load_part(part, processed):
    """Candidates (harness truth) and, per candidate, the row of its unique (user, history) profile."""
    beh = pd.read_parquet(processed / "behaviors.parquet", columns=["imp_key", "part", "user_id", "history"],
                          filters=[("part", "==", part)])
    if set(beh["part"].astype(str)) != {part}:
        raise ValueError("behaviors of other parts loaded")
    codes, histories = unique_profiles(beh)
    truth = load_truth(part, processed)
    row = pd.Index(beh["imp_key"].astype(str)).get_indexer(truth["imp_key"])
    if (row < 0).any():
        raise ValueError("candidates without a behaviors row")
    return truth, codes[row], histories


def run_configs(part, processed, configs):
    news = pd.read_parquet(processed / "news.parquet", columns=["news_id", "title", "text"])
    train_ids = train_news_ids()
    is_train = news["news_id"].isin(train_ids).to_numpy()
    truth, groups, histories = load_part(part, processed)
    item_idx = pd.Index(news["news_id"]).get_indexer(truth["news_id"])
    if (item_idx < 0).any():
        raise ValueError("candidates missing from news.parquet")
    hist_rows = history_index_lists(histories, pd.Index(news["news_id"]))
    cold = float(np.mean([not (np.asarray(h) >= 0).any() for h in hist_rows]))
    print(f"{part}: {len(truth):,} candidates, {len(histories):,} unique (user, history) profiles, "
          f"{100 * cold:.2f}% cold start; vectorizer corpus = {int(is_train.sum()):,} train-split news")
    weights = {}
    out = []
    for (text, sub), group in itertools.groupby(configs, key=lambda c: (c["text"], c["sublinear_tf"])):
        texts = news[text].fillna("")
        vec, X = fit_tfidf(texts[is_train], texts, sub, TFIDF_MIN_DF, TFIDF_STOP_WORDS)
        for cfg in group:
            start = time.time()
            agg = cfg["aggregation"]
            if agg not in weights:
                weights[agg] = profile_weights(hist_rows, len(news), agg)
            U = weights[agg] @ X
            scores = truth[["imp_key", "news_id"]].assign(score=cosine_scores(U, X, groups, item_idx))
            per_imp, summary = evaluate(truth, scores)
            print(f"  {cfg}: vocab {len(vec.vocabulary_):,}, {part} {SELECTION_METRIC} "
                  f"{summary[SELECTION_METRIC]:.4f} ({time.time() - start:.0f}s)")
            out.append((cfg, summary, scores, per_imp))
    info = {"vectorizer_fit_on": "train-split news (data/raw/train/news.tsv ids), label-free",
            "n_fit_texts": int(is_train.sum()), "min_df": TFIDF_MIN_DF, "stop_words": TFIDF_STOP_WORDS,
            "pct_cold_start_profiles": round(100 * cold, 2), "n_unique_profiles": len(histories)}
    return out, info


def grid_configs():
    keys = list(M1_GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(M1_GRID[k] for k in keys))]


def main():
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
            append_run(M1_NAME, "valid", "tune-grid", cfg, summary)
        best_cfg, best, scores, per_imp = runs[pick_best([r[1] for r in runs], SELECTION_METRIC)]
        save_results(M1_NAME, "valid", scores, per_imp, best)
        append_run(M1_NAME, "valid", "tune-chosen", best_cfg, best)
        grid = pd.DataFrame([{**cfg, **s.round(4).to_dict()} for cfg, s, _, _ in runs])
        TABLES.mkdir(parents=True, exist_ok=True)
        grid.to_csv(TABLES / "m1_tfidf_tuning_valid.csv", index=False)
        warn = best[SELECTION_METRIC] > LEAKAGE_AUC_WARN
        evidence = {"selection": f"max valid {SELECTION_METRIC}", "grid": grid.to_dict("records"), **info,
                    "leakage_warning": bool(warn)}
        doc = save_frozen(M1_NAME, best_cfg, evidence)
        print("\n" + grid.to_string(index=False))
        if warn:
            print(f"WARNING: valid {SELECTION_METRIC} > {LEAKAGE_AUC_WARN}: check for leakage before trusting this")
        print(f"\nchosen {best_cfg} (valid {SELECTION_METRIC} {best[SELECTION_METRIC]:.4f}); "
              f"record '{frozen_marker(M1_NAME, doc['config_hash'])}' in docs/decisions.md before --final-dev")
        return

    frozen = check_final_dev(M1_NAME, args.rerun_dev)
    runs, info = run_configs("dev", args.processed, [frozen["config"]])
    cfg, summary, scores, per_imp = runs[0]
    save_results(M1_NAME, "dev", scores, per_imp, summary)
    append_run(M1_NAME, "dev", "final-dev", cfg, summary, note=args.rerun_dev or "")
    (TABLES / "m1_tfidf_dev_meta.json").write_text(json.dumps({**info, "config": cfg,
                                                               "metrics": summary.round(4).to_dict()}, indent=2),
                                                   encoding="utf-8")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
