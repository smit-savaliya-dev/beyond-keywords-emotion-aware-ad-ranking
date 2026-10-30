"""External validation on SemEval-2007 Task 14 Affective Text (Strapparava & Mihalcea, 2007). Pre-declared in D46-D48.
Trial (250) selects the labelling rule, test (1,000) reports. Threshold-free correlations are primary; label agreement follows.
Headlines only, so this validates the title part of the scoring. Reads no MIND impressions.

  python scripts/38_analysis_semeval_validation.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import DIAGNOSTICS, EMOLEX_PATH, PROCESSED, SEMEVAL_CITATION, SEMEVAL_CUTOFF, SEMEVAL_DIR, TABLES, VAD_PATH
from src.emotion_features import go_vad
from src.lexicon import load_emolex, load_vad
from src.provenance import run_info
from src.semeval import headline_lexicon, load_semeval, run_validation
from src.story_charts import semeval_correlations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    data = load_semeval(SEMEVAL_DIR)
    raw = pd.read_parquet(PROCESSED / "semeval_raw28.parquet")
    if raw["news_id"].tolist() != data["news_id"].tolist():
        raise SystemExit("semeval_raw28.parquet does not match the headlines; rerun step 29")
    emolex, emolex_info = load_emolex()
    vad, vad_info = load_vad()
    lex = headline_lexicon(data["text"], emolex, vad)
    gv, overridden = go_vad(raw, vad)
    grid = pd.read_csv(TABLES / "emotion_label_variants_grid.csv")
    v = run_validation(data, raw, lex, gv, grid, args.n_boot)

    corr, agree = v["correlations"], v["agreement"]
    table = pd.concat([corr.assign(kind="correlation"), agree.assign(kind="agreement")], ignore_index=True)
    table.to_csv(TABLES / "semeval2007_validation.csv", index=False)
    v["selection"].to_csv(TABLES / "semeval2007_selection_trial.csv", index=False)
    v["per_class"].to_csv(TABLES / "semeval2007_per_class.csv", index=False)
    meta = {k: v[k] for k in ("chosen", "verdict_goemotions", "verdict_nrc", "neutral", "nrc_tie_rate", "nrc_hit_rate",
                              "vad_hit_rate", "confusion")}
    meta |= {"citation": SEMEVAL_CITATION, "n": data["split"].value_counts().to_dict(), "official_cutoff": SEMEVAL_CUTOFF,
             "rules": "D46, D47, D48 in docs/decisions.md", "go_vad_overrides_used": overridden,
             "n_boot": args.n_boot, "emolex": emolex_info, "vad": vad_info, "run_info": run_info()}
    (TABLES / "semeval2007_validation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    semeval_correlations(corr, DIAGNOSTICS / "semeval2007_correlations.png", "Emotion scores against human ratings of news headlines",
                         "Spearman correlation on the 1,000 SemEval-2007 test headlines, with 95% bootstrap intervals. "
                         "The dashed line is the pre-declared minimum of 0.2 for the spec check.")

    pd.set_option("display.width", 170)
    print(corr.round(3).to_string(index=False))
    print("\n" + agree.drop(columns=["kappa_band"]).round(3).to_string(index=False))
    print("\nchosen on trial:", v["chosen"])
    print("GoEmotions:", v["verdict_goemotions"])
    print("NRC:", v["verdict_nrc"])
    print("neutral %:", v["neutral"])


if __name__ == "__main__":
    main()
