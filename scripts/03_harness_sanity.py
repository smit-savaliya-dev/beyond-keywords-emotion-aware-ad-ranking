"""Step 3: prove the evaluation harness works on the REAL data before any model exists.

Scores four fake "models" whose results we know in advance:
  random        -> AUC about 0.50
  constant      -> AUC exactly 0.50 (all ties)
  oracle        -> AUC, nDCG, HitRate exactly 1.0; MRR = best possible (below 1 with multi-click impressions)
  anti_oracle   -> AUC exactly 0.0
Run from the repo root:  python scripts/03_harness_sanity.py --part valid
                         python scripts/03_harness_sanity.py --part dev
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED, RESULTS, SEED  # noqa: E402
from src.evaluate import evaluate, load_truth, save_results  # noqa: E402


def fake_scores(truth, kind, seed=SEED):
    s = truth[["imp_key", "news_id"]].copy()
    if kind == "random":
        s["score"] = np.random.default_rng(seed).random(len(s))
    elif kind == "constant":
        s["score"] = 0.0
    elif kind == "oracle":
        s["score"] = truth["label"].to_numpy(dtype=float)
    elif kind == "anti_oracle":
        s["score"] = -truth["label"].to_numpy(dtype=float)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="valid", choices=["fit", "valid", "dev"])
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()

    truth = load_truth(args.part, args.processed)
    print(f"part={args.part}: {truth['imp_key'].nunique()} impressions, {len(truth)} candidate rows\n")

    # With c clicks, even a perfect ranking puts them at ranks 1..c, so the best
    # possible MIND MRR is (1 + 1/2 + ... + 1/c) / c, which is below 1 when c > 1.
    clicks = truth.groupby("imp_key")["label"].sum().to_numpy()
    best_mrr = np.mean([np.sum(1 / np.arange(1, c + 1)) / c for c in clicks])
    print(f"best possible MRR for this part = {best_mrr:.4f}\n")

    checks = {
        "random":      lambda m: abs(m["auc"] - 0.5) < 0.01,
        "constant":    lambda m: abs(m["auc"] - 0.5) < 1e-9,
        "oracle":      lambda m: (all(abs(m[c] - 1) < 1e-9 for c in ("auc", "ndcg@5", "ndcg@10", "hr@5"))
                                  and abs(m["mrr"] - best_mrr) < 1e-9),
        "anti_oracle": lambda m: abs(m["auc"]) < 1e-9,
    }
    all_ok = True
    for kind, check in checks.items():
        scores = fake_scores(truth, kind)
        per_imp, summary = evaluate(truth, scores)
        save_results(f"sanity_{kind}", args.part, scores, per_imp, summary, args.results)
        ok = check(summary)
        all_ok &= ok
        print(f"{kind:12s} {'PASS' if ok else 'FAIL'}  " +
              "  ".join(f"{k}={v:.4f}" for k, v in summary.items()))
    print("\nHARNESS:", "ready" if all_ok else "BROKEN - fix before building any model")


if __name__ == "__main__":
    main()