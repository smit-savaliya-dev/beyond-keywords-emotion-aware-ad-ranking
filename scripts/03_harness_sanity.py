"""Prove the evaluation harness on real data with fake models whose results are known in advance."""
import argparse
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from src.config import PROCESSED, SEED
from src.evaluate import evaluate, load_truth


def fake_scores(truth, kind):
    s = truth[["imp_key", "news_id"]].copy()
    label = truth["label"].to_numpy(dtype=float)
    s["score"] = {"random": lambda: np.random.default_rng(SEED).random(len(s)),
                  "constant": lambda: np.zeros(len(s)),
                  "oracle": lambda: label,
                  "anti_oracle": lambda: -label}[kind]()
    return s


def run_part(part, processed):
    truth = load_truth(part, processed)
    clicks = truth.groupby("imp_key")["label"].sum().to_numpy()
    best_mrr = np.mean([np.sum(1 / np.arange(1, c + 1)) / c for c in clicks])  # ceiling when clicks > 1
    print(f"\npart={part}: {truth['imp_key'].nunique()} impressions, {len(truth)} rows, best possible MRR = {best_mrr:.4f}")
    checks = {
        "random": lambda m: abs(m["auc"] - 0.5) < 0.01,
        "constant": lambda m: abs(m["auc"] - 0.5) < 1e-9,
        "oracle": lambda m: all(abs(m[c] - 1) < 1e-9 for c in ("auc", "ndcg@5", "ndcg@10", "hr@5"))
                            and abs(m["mrr"] - best_mrr) < 1e-9,
        "anti_oracle": lambda m: abs(m["auc"]) < 1e-9,
    }
    ok = True
    for kind, check in checks.items():
        _, summary = evaluate(truth, fake_scores(truth, kind))
        passed = bool(check(summary))
        ok &= passed
        print(f"{kind:12s} {'PASS' if passed else 'FAIL'}  " + "  ".join(f"{k}={v:.4f}" for k, v in summary.items()))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["valid", "dev"], choices=["fit", "valid", "dev"])
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    ok = all([run_part(p, args.processed) for p in args.parts])
    print("\nHARNESS:", "ready" if ok else "BROKEN - fix before building any model")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
