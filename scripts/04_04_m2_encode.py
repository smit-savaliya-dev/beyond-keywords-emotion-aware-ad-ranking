"""Encode every news item with all-mpnet-base-v2, one cache per text variant (D50). GPU, resumable, no labels involved.

  python scripts/04_04_m2_encode.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import M2_GRID, PROCESSED, SBERT_BATCH_SIZE, SBERT_MAX_LENGTH, SBERT_MODEL, SBERT_REVISION
from src.emotion import pick_device
from src.provenance import run_info
from src.sbert import embedding_paths, encode_corpus, load_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    ap.add_argument("--device", default="auto", help="auto, -1 (CPU) or 0 (first GPU)")
    args = ap.parse_args()
    p = args.processed

    news = pd.read_parquet(p / "news.parquet", columns=["news_id", *M2_GRID["text"]])
    device = pick_device(args.device)
    print(f"device: {'cuda:' + str(device) if device >= 0 else 'cpu'}; {len(news):,} items")
    encode = load_encoder(device)
    settings = {"model": SBERT_MODEL, "revision": SBERT_REVISION, "max_length": SBERT_MAX_LENGTH, "pooling": "mean, L2 normalised"}
    for text in M2_GRID["text"]:
        mat = encode_corpus(news, encode, p / f"sbert_chunks_{text}", settings, text, batch_size=SBERT_BATCH_SIZE)
        mat_path, ids_path = embedding_paths(text, p)
        np.save(mat_path, mat)
        news[["news_id"]].to_parquet(ids_path, index=False)
        norms = np.linalg.norm(mat, axis=1)
        print(f"[done] {text}: {mat.shape}, norm {norms.min():.4f} to {norms.max():.4f}")
    meta = {**settings, "n_items": len(news), "variants": M2_GRID["text"], "run_info": run_info()}
    (p / "sbert_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
