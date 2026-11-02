"""Correctness check of D50: saved transformers-based embeddings vs the sentence-transformers package, on random items, both text variants.
Optional tool, not a pipeline step (needs the model and, for speed, the GPU). No labels involved.

  python scripts/check_sbert_encoder.py
"""
import json

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import M2_GRID, PROCESSED, SBERT_BATCH_SIZE, SBERT_CHECK_MIN_COSINE, SBERT_CHECK_N, SBERT_MODEL, SBERT_REVISION, SEED, TABLES
from src.emotion import pick_device
from src.provenance import run_info
from src.sbert import compare_encoders, load_embeddings


def main():
    from sentence_transformers import SentenceTransformer
    device = pick_device("auto")
    model = SentenceTransformer(SBERT_MODEL, revision=SBERT_REVISION, device=f"cuda:{device}" if device >= 0 else "cpu")
    news = pd.read_parquet(PROCESSED / "news.parquet", columns=["news_id", *M2_GRID["text"]])
    sample = news.sample(SBERT_CHECK_N, random_state=SEED).index.to_numpy()
    rows = {}
    for text in M2_GRID["text"]:
        cached = load_embeddings(text, news["news_id"], PROCESSED)[sample]
        other = model.encode(news[text].iloc[sample].tolist(), batch_size=SBERT_BATCH_SIZE, show_progress_bar=False)
        rows[text] = compare_encoders(cached, other)
        print(text, rows[text])
    passed = all(r["min_cosine"] >= SBERT_CHECK_MIN_COSINE for r in rows.values())
    meta = {"model": SBERT_MODEL, "revision": SBERT_REVISION, "n_items": SBERT_CHECK_N, "seed": SEED, "threshold": SBERT_CHECK_MIN_COSINE,
            "max_seq_length": model.max_seq_length, "results": rows, "passed": passed, "run_info": run_info()}
    (TABLES / "sbert_encoder_check.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("PASS" if passed else f"FAIL: min cosine below {SBERT_CHECK_MIN_COSINE}")


if __name__ == "__main__":
    main()
