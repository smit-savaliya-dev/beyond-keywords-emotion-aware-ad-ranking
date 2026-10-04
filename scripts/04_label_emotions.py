"""Label every news item with emotion vectors. Resumable: re-run to continue after an interruption.

  python scripts/04_label_emotions.py --smoke    # 12 headlines, writes nothing: check the model works
  python scripts/04_label_emotions.py            # full run (GPU used automatically if available)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import EMOTION_MAX_LENGTH, EMOTION_MODEL, PROCESSED
from src.emotion import (SIX_CLASSES, build_emotion_table, collapse, label_corpus, load_classifier,
                         pick_device, score_texts)
from src.provenance import run_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    ap.add_argument("--model", default=EMOTION_MODEL)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--max-length", type=int, default=EMOTION_MAX_LENGTH)
    ap.add_argument("--device", default="auto", help="auto, -1 (CPU) or 0 (first GPU)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    news = pd.read_parquet(args.processed / "news.parquet", columns=["news_id", "text"])
    device = pick_device(args.device)
    print(f"device: {'cuda:' + str(device) if device >= 0 else 'cpu'}")
    clf = load_classifier(args.model, device)

    if args.smoke:
        sample = news.sample(12, random_state=0)
        raw = score_texts(clf, sample["text"].tolist(), args.batch_size, args.max_length)
        for text, v in zip(sample["text"], collapse(raw)):
            print(f"{SIX_CLASSES[int(v.argmax())]:9s} {np.round(v, 2)}  {text[:70]}")
        print(f"\nraw scores {raw.min():.3f}..{raw.max():.3f}; mean row sum {raw.sum(axis=1).mean():.2f} "
              "(not 1: independent sigmoids, expected)")
        return

    settings = {"model": args.model, "max_length": args.max_length}
    raw = label_corpus(news, clf, args.processed / "emotion_chunks", settings, args.chunk_size, args.batch_size)
    table = build_emotion_table(raw)
    if (abs(table[SIX_CLASSES].sum(axis=1) - 1) > 1e-6).any():
        raise ValueError("emotion vectors do not sum to 1")

    raw.to_parquet(args.processed / "news_emotion_raw28.parquet", index=False)
    table.to_parquet(args.processed / "news_emotion.parquet", index=False)
    meta = {**settings, "n_items": len(table), "run_info": run_info()}
    (args.processed / "news_emotion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[done] {len(table)} items labelled\n" + table["dominant_emotion"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
