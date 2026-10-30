"""Build the v2 emotion tables for the setting chosen by D30 (emotion_v2_choice.json). New files only; v1 is untouched.
No GPU: uses the saved title/abstract raw scores. Labels are for reporting and validation, never model features.

  python scripts/31_emotion_v2_features.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import LABEL_SHEET, PROCESSED, RESULTS, SEED, V2_EXAMPLES_PER_CLASS
from src.emotion_features import go_features, go_percentiles, go_vad, merge_features
from src.emotion_v2 import class_shares, combine_share, example_rows, intensity_table, threshold_labels
from src.lexicon import combine_lexicon_fields, field_lexicon_features, lexicon_counts, load_emolex, load_vad
from src.provenance import run_info

TABLES = RESULTS / "tables"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    p = args.processed
    choice = json.loads((TABLES / "emotion_v2_choice.json").read_text(encoding="utf-8"))
    share, t = choice["chosen_title_share"], choice["chosen_t"]

    news = pd.read_parquet(p / "news.parquet", columns=["news_id", "title", "abstract"])
    ids = news["news_id"].tolist()
    raw = combine_share(pd.read_parquet(p / "news_emotion_raw28_title.parquet"),
                        pd.read_parquet(p / "news_emotion_raw28_abstract.parquet"), share)
    if raw["news_id"].tolist() != ids:
        raise ValueError("raw scores are not aligned with news.parquet")
    table = intensity_table(raw)
    labels = table.assign(label=threshold_labels(table, t), neutral_score=raw["neutral"].to_numpy())

    emolex, _ = load_emolex()
    vad, _ = load_vad()
    has_abstract = raw["has_abstract"].to_numpy()
    fields = {f: field_lexicon_features(lexicon_counts(news[f].tolist(), emolex, vad)) for f in ("title", "abstract")}
    lex = combine_lexicon_fields(fields["title"], fields["abstract"], has_abstract, share)
    lex.insert(0, "news_id", news["news_id"].to_numpy())

    go = go_features(raw)
    gv, _ = go_vad(raw, vad)
    frames = [raw[["news_id", "has_abstract"]], go, go_percentiles(go), gv, lex,
              table[["news_id", "intensity"]].rename(columns={"intensity": "go_intensity"})]
    feats = merge_features(ids, frames)
    if feats.isna().any().any():
        raise ValueError("NaN in the v2 feature table")

    raw.to_parquet(p / "news_emotion_raw28_v2.parquet", index=False)
    labels.to_parquet(p / "news_emotion_labels_v2.parquet", index=False)
    feats.to_parquet(p / "news_emotion_features_v2.parquet", index=False)

    sheet_ids = pd.read_csv(LABEL_SHEET, usecols=["news_id"], dtype=str, encoding="utf-8-sig")["news_id"]
    examples = example_rows(labels, news, sheet_ids, V2_EXAMPLES_PER_CLASS, SEED)
    examples.round({"intensity": 3, "neutral_score": 3}).to_csv(TABLES / "emotion_examples_v2.csv", index=False,
                                                                encoding="utf-8-sig")
    emotional = labels[labels["label"] != "neutral"]
    mix = emotional.groupby("label")["top_label"].value_counts(normalize=True).mul(100).round(1)
    meta = {"title_share": share, "t": t, "rule": "D30", "n_items": len(feats), "class_pct": class_shares(labels["label"]),
            "emotional_items": {"median_intensity": round(float(emotional["intensity"].median()), 3),
                                "pct_intensity_below_0.1": round(100 * float((emotional["intensity"] < 0.1).mean()), 1),
                                "pct_classifier_neutral_score_above_0.5":
                                    round(100 * float((emotional["neutral_score"] > 0.5).mean()), 1)},
            "top_raw_labels_per_class_pct": {c: mix[c].head(4).to_dict() for c in mix.index.levels[0]},
            "caution": "a low t reduces neutral by definition, not by detecting more emotion; approval counts as joy and "
                       "disapproval as anger in the 28 -> 6 mapping. Hand labels are the real test.",
            "examples_exclude_hand_label_sheet": True, "run_info": run_info()}
    (TABLES / "emotion_v2_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "run_info"}, indent=1))
    print(f"{feats.shape} features; {len(examples)} examples")


if __name__ == "__main__":
    main()
