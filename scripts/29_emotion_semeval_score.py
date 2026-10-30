"""Score the SemEval-2007 headlines with GoEmotions (same model, max_length 128). Small GPU job on new data only; MIND is not touched.
Headlines only, so this validates the title part of the MIND scoring. Resumable, with the usual settings guard.

  python scripts/29_emotion_semeval_score.py
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import EMOTION_MODEL, PROCESSED, SEMEVAL_CITATION, SEMEVAL_DIR, TITLE_BATCH_SIZE, TITLE_MAX_LENGTH
from src.emotion import label_corpus, load_classifier, pick_device
from src.provenance import run_info
from src.semeval import load_semeval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    ap.add_argument("--device", default="auto", help="auto, -1 (CPU) or 0 (first GPU)")
    args = ap.parse_args()
    p = args.processed

    data = load_semeval(SEMEVAL_DIR)[["news_id", "text"]]  # gold scores are never passed to the model
    device = pick_device(args.device)
    print(f"device: {'cuda:' + str(device) if device >= 0 else 'cpu'}; {len(data)} headlines")
    clf = load_classifier(EMOTION_MODEL, device)
    settings = {"model": EMOTION_MODEL, "max_length": TITLE_MAX_LENGTH, "field": "semeval2007_headline"}
    raw = label_corpus(data, clf, p / "emotion_chunks_semeval", settings, chunk_size=500, batch_size=TITLE_BATCH_SIZE)
    raw.to_parquet(p / "semeval_raw28.parquet", index=False)
    meta = {**settings, "n_items": len(raw), "citation": SEMEVAL_CITATION, "run_info": run_info()}
    (p / "semeval_raw28_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {len(raw)} headlines scored")


if __name__ == "__main__":
    main()
