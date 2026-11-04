"""M0 popularity: Beta-smoothed item CTR (clicks + a*p) / (shown + a) towards the global CTR p; unseen items get p.

Reads:  impressions (click statistics from fit; fit + valid for the dev run).
Writes: --tune: results/frozen/M0_popularity.json, results/tables/m0_popularity_tuning_valid.csv, valid scores and per-impression files.
        --final-dev: dev scores, per-impression file, a summary.csv row and a runs_log.csv row.
Run:    python scripts/04_01_m0_popularity.py --tune         # valid only: picks a by valid AUC (part of the pipeline)
        python scripts/04_01_m0_popularity.py --final-dev    # once, by hand, with the frozen a
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src.baselines import item_ctr, popularity_scores, unseen_share
from src.config import M0_NAME, M0_SMOOTHING_GRID, PROCESSED, TABLES, SELECTION_METRIC
from src.evaluate import evaluate, load_truth, save_results
from src.run_registry import append_run, check_final_dev, frozen_marker, load_rows, pick_best, save_frozen

STATS_PARTS = {"valid": ["fit"], "dev": ["fit", "valid"]}  # click statistics only from earlier parts


def score(part, processed, a_values):
    """Popularity scores of one part for each value of the smoothing constant a."""
    stats = load_rows(processed, STATS_PARTS[part], ["news_id", "label"])
    ctr, p = item_ctr(stats)
    truth = load_truth(part, processed)
    cand = truth[["imp_key", "news_id"]]
    share = unseen_share(cand, ctr)
    print(f"{part}: stats from {STATS_PARTS[part]} ({len(stats):,} rows, global CTR {p:.4f}); "
          f"{len(truth):,} candidates, {100 * share:.1f}% unseen in the statistics parts")
    out = []
    for a in a_values:
        scores = cand.assign(score=popularity_scores(cand, ctr, p, a))
        per_imp, summary = evaluate(truth, scores)
        out.append(({"a": a}, summary, scores, per_imp))
    return out, {"global_ctr": p, "unseen_share": share, "stats_parts": STATS_PARTS[part], "n_candidates": len(truth)}


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
        runs, info = score("valid", args.processed, M0_SMOOTHING_GRID)
        for cfg, summary, _, _ in runs:
            append_run(M0_NAME, "valid", "tune-grid", cfg, summary)
        best_cfg, best, scores, per_imp = runs[pick_best([r[1] for r in runs], SELECTION_METRIC)]
        save_results(M0_NAME, "valid", scores, per_imp, best)
        append_run(M0_NAME, "valid", "tune-chosen", best_cfg, best)
        grid = pd.DataFrame([{**cfg, **s.round(4).to_dict()} for cfg, s, _, _ in runs])
        TABLES.mkdir(parents=True, exist_ok=True)
        grid.to_csv(TABLES / "m0_popularity_tuning_valid.csv", index=False)
        evidence = {"selection": f"max valid {SELECTION_METRIC}", "grid": grid.to_dict("records"), **info}
        doc = save_frozen(M0_NAME, best_cfg, evidence)
        print(grid.to_string(index=False))
        print(f"\nchosen {best_cfg} (valid {SELECTION_METRIC} {best[SELECTION_METRIC]:.4f}); "
              f"record '{frozen_marker(M0_NAME, doc['config_hash'])}' in docs/decisions.md before --final-dev")
        return

    frozen = check_final_dev(M0_NAME, args.rerun_dev)
    runs, info = score("dev", args.processed, [frozen["config"]["a"]])
    cfg, summary, scores, per_imp = runs[0]
    save_results(M0_NAME, "dev", scores, per_imp, summary)
    append_run(M0_NAME, "dev", "final-dev", cfg, summary, note=args.rerun_dev or "")
    (TABLES / "m0_popularity_dev_meta.json").write_text(json.dumps({**info, "config": cfg,
                                                                     "metrics": summary.round(4).to_dict()}, indent=2),
                                                        encoding="utf-8")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
