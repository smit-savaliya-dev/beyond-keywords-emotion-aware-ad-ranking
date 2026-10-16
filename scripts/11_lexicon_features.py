"""NRC EmoLex + VAD features (title/abstract weighted) and classifier-derived go_vad_* features. CPU, about a minute.

  python scripts/11_lexicon_features.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import ABSTRACT_WEIGHT, NRC_EMOTIONS, PROCESSED, RESULTS, TITLE_WEIGHT, TOKEN_PATTERN
from src.emotion_features import go_vad, title_share
from src.lexicon import (LEXICON_COLUMNS, combine_lexicon_fields, field_lexicon_features, lexicon_counts,
                         load_emolex, load_vad, top_matched_words)
from src.provenance import run_info

TABLES = RESULTS / "tables"


def field_coverage(feats, mask):
    f = feats[mask]
    return {"n_texts": int(mask.sum()),
            "pct_zero_emolex_emotion_hits": round(100 * float((f["nrc_has_hits"] == 0).mean()), 2),
            "pct_zero_vad_hits": round(100 * float((f["vad_has_hits"] == 0).mean()), 2),
            "mean_nrc_coverage": round(float(f["nrc_coverage"].mean()), 4),
            "mean_vad_coverage": round(float(f["vad_coverage"].mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    args = ap.parse_args()
    p = args.processed

    news = pd.read_parquet(p / "news.parquet", columns=["news_id", "title", "abstract"])
    emolex, emolex_info = load_emolex()
    vad, vad_info = load_vad()
    has_abstract = (news["abstract"].str.strip() != "").to_numpy()

    feats = {f: field_lexicon_features(lexicon_counts(news[f].tolist(), emolex, vad)) for f in ("title", "abstract")}
    lex = combine_lexicon_fields(feats["title"], feats["abstract"], has_abstract, title_share())
    lex.insert(0, "news_id", news["news_id"].to_numpy())

    weighted = pd.read_parquet(p / "news_emotion_raw28_weighted.parquet")
    if weighted["news_id"].tolist() != news["news_id"].tolist():
        raise ValueError("weighted raw scores do not match news.parquet")
    gv, overridden = go_vad(weighted, vad)
    print("go_vad overrides used:", overridden or "none")

    if lex[LEXICON_COLUMNS].isna().any().any() or gv.isna().any().any():
        raise ValueError("NaN in lexicon features")
    shares = lex[[f"nrc_{e}" for e in NRC_EMOTIONS]].sum(axis=1)
    if ((abs(shares - 1) > 1e-6) & (lex["nrc_has_hits"] == 1)).any() or (shares[lex["nrc_has_hits"] == 0] != 0).any():
        raise ValueError("nrc shares must sum to 1 with hits and 0 without")

    all_items = np.ones(len(news), dtype=bool)
    coverage = {"title": field_coverage(feats["title"], all_items),
                "abstract": field_coverage(feats["abstract"], has_abstract),
                "weighted": field_coverage(lex, all_items)}
    print(pd.DataFrame(coverage).to_string())

    top = top_matched_words(pd.concat([news["title"], news["abstract"]]),
                            emolex.index[emolex[NRC_EMOTIONS].sum(axis=1) > 0])
    top["emotions"] = [", ".join(e for e in NRC_EMOTIONS if emolex.loc[w, e]) for w in top["word"]]
    print("\nmost frequent EmoLex emotion words (title + abstract):\n" + top.head(20).to_string(index=False))

    lex.to_parquet(p / "news_emotion_nrc_features.parquet", index=False)
    gv.to_parquet(p / "news_emotion_go_vad.parquet", index=False)
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "lexicon_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    top.to_csv(TABLES / "lexicon_top_words.csv", index=False)
    meta = {
        "emolex": emolex_info, "vad": vad_info, "token_pattern": TOKEN_PATTERN,
        "weights": {"title": TITLE_WEIGHT, "abstract": ABSTRACT_WEIGHT, "effective_title": title_share()},
        "rules": {"nrc_shares": "8-emotion association counts / total, per field; weights renormalised over fields "
                                "with hits; 0 and nrc_has_hits = 0 when no field has hits",
                  "nrc_positive_negative": "share of tokens with the positive/negative flag; not part of the shares",
                  "nrc_coverage": "share of tokens with any of the 8 emotions; plain field weights",
                  "vad": "mean over matched unigram tokens in [-1, 1]; same renormalisation; vad_has_hits flag",
                  "go_vad": "probability-weighted mean VAD of the 27 non-neutral GoEmotions label words, "
                            "from news_emotion_raw28_weighted.parquet",
                  "limitations": "no negation, no lemmatisation, unigrams only, apostrophes/hyphens split words; "
                                 "VAD v2.1 includes some function words scored ~0, which pulls means towards 0"},
        "go_vad_overrides_used": overridden,
        "n_items": len(lex), "run_info": run_info(),
    }
    (p / "news_emotion_lexicon_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[done] {len(lex)} items -> news_emotion_nrc_features.parquet, news_emotion_go_vad.parquet")


if __name__ == "__main__":
    main()
