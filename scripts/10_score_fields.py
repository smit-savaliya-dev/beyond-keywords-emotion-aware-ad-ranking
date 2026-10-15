"""Score titles and abstracts separately with GoEmotions, combine with the field weights, build go_* features.
Resumable: re-run to continue after an interruption. The combined-text baseline files are never touched.

  python scripts/10_score_fields.py --measure    # token lengths of titles/abstracts (CPU, writes a small table)
  python scripts/10_score_fields.py --smoke      # 12 items + a GPU memory check, writes nothing
  python scripts/10_score_fields.py              # full run
  python scripts/10_score_fields.py --report     # combined vs weighted summary from saved files
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import (ABSTRACT_BATCH_SIZE, ABSTRACT_MAX_LENGTH, ABSTRACT_WEIGHT, EMOTION_MODEL, GO_FEATURES,
                        PROCESSED, RESULTS, TITLE_BATCH_SIZE, TITLE_MAX_LENGTH, TITLE_WEIGHT)
from src.emotion import (GOEMOTIONS_LABELS, SIX_CLASSES, build_emotion_table, label_corpus, load_classifier,
                         pick_device, score_texts)
from src.emotion_features import (combine_fields, field_texts, go_features, neutral_summary, title_share,
                                  truncation_report)
from src.provenance import run_info

TABLES = RESULTS / "tables"


def measure(news, model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    rep = {f: truncation_report(tok, field_texts(news, f)[f]) for f in ("title", "abstract")}
    print(json.dumps(rep, indent=2))
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "token_lengths.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")


def raw_frame(news_ids, mat):
    df = pd.DataFrame(mat, columns=GOEMOTIONS_LABELS)
    df.insert(0, "news_id", np.asarray(news_ids))
    return df


def smoke(news, clf, args):
    sample = news.sample(12, random_state=0)
    titles, abstracts = field_texts(sample, "title"), field_texts(sample, "abstract")
    t = raw_frame(titles["news_id"], score_texts(clf, titles["title"].tolist(), args.title_batch_size,
                                                 args.title_max_length))
    a = raw_frame(abstracts["news_id"], score_texts(clf, abstracts["abstract"].tolist(), args.abstract_batch_size,
                                                    args.abstract_max_length))
    w = combine_fields(t, a)
    six, go = build_emotion_table(w), go_features(w)
    for (_, row), dom, mass, title in zip(w.iterrows(), six["dominant_emotion"], go["go_emotion_mass"],
                                          sample["title"]):
        print(f"{dom:9s} abstract={'y' if row['has_abstract'] else 'n'} mass={mass:.2f}  {title[:65]}")
    print(go.drop(columns="news_id").describe().loc[["mean", "min", "max"]].round(3).T.to_string())

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        longest = field_texts(news, "abstract")["abstract"]
        longest = longest.iloc[np.argsort(-longest.str.len().to_numpy())[:args.abstract_batch_size * 2]].tolist()
        score_texts(clf, longest, args.abstract_batch_size, args.abstract_max_length)
        peak = torch.cuda.max_memory_allocated() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"\nGPU memory check: {len(longest)} longest abstracts, batch {args.abstract_batch_size}, "
              f"max_length {args.abstract_max_length}: peak {peak:.2f} of {total:.1f} GB")


def report(processed):
    out = {}
    for name, table_file, raw_file in [("combined", "news_emotion.parquet", "news_emotion_raw28.parquet"),
                                       ("weighted", "news_emotion_weighted.parquet",
                                        "news_emotion_raw28_weighted.parquet")]:
        out[name] = neutral_summary(pd.read_parquet(processed / table_file), pd.read_parquet(processed / raw_file))
    print(pd.DataFrame({k: v["dominant_pct"] for k, v in out.items()}).to_string(), "\n")
    for key in ("neutral_share_6class", "neutral_raw_sigmoid"):
        print(key)
        print(pd.DataFrame({k: v[key] for k, v in out.items()}).to_string(), "\n")
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "emotion_weighted_vs_combined.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    ap.add_argument("--model", default=EMOTION_MODEL)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--title-batch-size", type=int, default=TITLE_BATCH_SIZE)
    ap.add_argument("--abstract-batch-size", type=int, default=ABSTRACT_BATCH_SIZE)
    ap.add_argument("--title-max-length", type=int, default=TITLE_MAX_LENGTH)
    ap.add_argument("--abstract-max-length", type=int, default=ABSTRACT_MAX_LENGTH)
    ap.add_argument("--device", default="auto", help="auto, -1 (CPU) or 0 (first GPU)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--measure", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        return report(args.processed)
    news = pd.read_parquet(args.processed / "news.parquet", columns=["news_id", "title", "abstract"])
    if args.measure:
        return measure(news, args.model)
    if args.abstract_max_length is None:
        raise SystemExit("ABSTRACT_MAX_LENGTH is not set in src/config.py (run --measure first)")

    device = pick_device(args.device)
    print(f"device: {'cuda:' + str(device) if device >= 0 else 'cpu'}")
    clf = load_classifier(args.model, device)
    if args.smoke:
        return smoke(news, clf, args)

    raws = {}
    for field, max_len, bs in [("title", args.title_max_length, args.title_batch_size),
                               ("abstract", args.abstract_max_length, args.abstract_batch_size)]:
        texts = field_texts(news, field)
        print(f"\n== {field}: {len(texts)} non-empty texts, max_length {max_len}, batch {bs}")
        settings = {"model": args.model, "max_length": max_len, "field": field}
        raws[field] = label_corpus(texts, clf, args.processed / f"emotion_chunks_{field}", settings,
                                   args.chunk_size, bs, text_col=field)

    weighted = combine_fields(raws["title"], raws["abstract"])
    if weighted["news_id"].tolist() != news["news_id"].tolist():
        raise ValueError("weighted news ids do not match news.parquet")
    table = build_emotion_table(weighted)
    go = go_features(weighted)
    if (abs(table[SIX_CLASSES].sum(axis=1) - 1) > 1e-6).any():
        raise ValueError("emotion vectors do not sum to 1")
    dist_cols = [c for c in go.columns if c.startswith("go_dist_")]
    if (abs(go[dist_cols].sum(axis=1) - 1) > 1e-6).any():
        raise ValueError("go_dist_* does not sum to 1")

    p = args.processed
    raws["title"].to_parquet(p / "news_emotion_raw28_title.parquet", index=False)
    raws["abstract"].to_parquet(p / "news_emotion_raw28_abstract.parquet", index=False)
    weighted.to_parquet(p / "news_emotion_raw28_weighted.parquet", index=False)
    table.to_parquet(p / "news_emotion_weighted.parquet", index=False)
    go.to_parquet(p / "news_emotion_go_features.parquet", index=False)
    wt = title_share()
    meta = {
        "model": args.model,
        "fields": {"title": {"max_length": args.title_max_length, "batch_size": args.title_batch_size,
                             "n_scored": len(raws["title"]), "chunks": "emotion_chunks_title"},
                   "abstract": {"max_length": args.abstract_max_length, "batch_size": args.abstract_batch_size,
                                "n_scored": len(raws["abstract"]), "chunks": "emotion_chunks_abstract"}},
        "weights": {"title": TITLE_WEIGHT, "abstract": ABSTRACT_WEIGHT,
                    "effective_title": wt, "effective_abstract": 1 - wt, "empty_abstract": "title only (weight 1.0)"},
        "combination": "weighted mean of the raw 28 sigmoid scores, then all features derived from it",
        "go_features": GO_FEATURES,
        "notes": {"go_joy": "noisy-OR: 1 - (1 - p_joy)(1 - p_amusement)",
                  "go_nervousness": "anxiety", "go_annoyance": "frustration",
                  "go_emotionality": "1 - go_neutral",
                  "go_emotion_mass": "sum of the 8 go_* emotion scores before normalising",
                  "go_dist_*": "8 go_* scores / go_emotion_mass (uniform only at exact zero). Noisy when "
                               "go_emotion_mass is small: weight or filter congruence by go_emotion_mass."},
        "n_items": len(weighted), "n_with_abstract": int(weighted["has_abstract"].sum()),
        "run_info": run_info(),
    }
    (p / "news_emotion_weighted_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[done] {len(weighted)} items ({meta['n_with_abstract']} with abstract)\n")
    report(p)


if __name__ == "__main__":
    main()
