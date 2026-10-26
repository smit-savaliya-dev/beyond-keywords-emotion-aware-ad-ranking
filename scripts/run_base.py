"""Whole pipeline in order, rerunning only what is out of date. Stops at the first failure.

A step reruns when its data inputs, its code (the script and the src/ modules it imports), its arguments or its
outputs changed since its last successful run (content hashes; state in data/processed/.pipeline/).
Download and the annotation sheet are create-only; label validation waits until the sheet is filled.

  python scripts/run_base.py --dry-run          # show what would run and why, run nothing
  python scripts/run_base.py                    # run everything that is out of date
  python scripts/run_base.py --adopt            # once: record existing outputs as up to date, run only the rest
  python scripts/run_base.py --force-step 14    # rerun one step (and whatever then changes downstream)
  python scripts/run_base.py --from 11          # rerun step 11 and every later step
  python scripts/run_base.py --only 13 14       # consider only these steps
"""
import argparse
import sys
import time

import _bootstrap  # noqa: F401
from src.annotation import read_sheet, sheet_problems
from src.config import (ANNOTATION, EMOLEX_PATH, FIGURES, FROZEN_DIR, M0_NAME, M1_NAME, PROCESSED, RAW, RESULTS,
                        ROOT, VAD_PATH)
from src.pipeline import Pipeline, Step, StepFailed

S = ROOT / "scripts"
P, T = PROCESSED, RESULTS / "tables"
SHEET = ANNOTATION / "manual_sample.csv"
RAW_FILES = [RAW / s / f for s in ("train", "dev") for f in ("news.tsv", "behaviors.tsv")]
NEWS, BEH, IMPS = P / "news.parquet", P / "behaviors.parquet", P / "impressions.parquet"
EDA_FIGS = ["01_part_sizes", "02_candidates_per_impression", "03_clicks_per_impression", "04_categories",
            "05_history_length", "06_integrity", "eda_overview"]


def labels_ready():
    if not SHEET.exists():
        return f"{SHEET.name} does not exist yet"
    problems = sheet_problems(read_sheet(SHEET))
    return f"waiting for hand labels ({problems[0]})" if problems else None


def build_steps(skip_official_check):
    extra = ["--skip-official-check"] if skip_official_check else []
    code_and_tests = sorted([*(ROOT / "src").glob("*.py"), *(ROOT / "tests").glob("*.py"), ROOT / "pytest.ini"])
    return [
        Step("00_tests", ["-m", "pytest", "-q"], inputs=code_and_tests, adoptable=False),
        Step("01_download", [S / "01_download_mind.py"], outputs=RAW_FILES, create_only=True),
        Step("02_prepare", [S / "02_prepare_mind.py", *extra], inputs=RAW_FILES, script=S / "02_prepare_mind.py",
             outputs=[NEWS, BEH, IMPS, P / "data_summary.json"]),
        Step("03_harness", [S / "03_harness_sanity.py"], inputs=[IMPS], script=S / "03_harness_sanity.py"),
        Step("07_eda", [S / "07_eda.py"], inputs=[NEWS, BEH, IMPS, P / "data_summary.json"], script=S / "07_eda.py",
             outputs=[FIGURES / f"{f}.png" for f in EDA_FIGS] + [FIGURES / "eda_stats.json"]),
        Step("04_label_combined", [S / "04_label_emotions.py"], inputs=[NEWS], script=S / "04_label_emotions.py",
             outputs=[P / "news_emotion_raw28.parquet", P / "news_emotion.parquet", P / "news_emotion_meta.json"],
             note="GPU; resumes from emotion_chunks/"),
        Step("05_annotation_sheet", [S / "05_make_annotation_sample.py"], inputs=[NEWS], outputs=[SHEET],
             create_only=True),
        Step("10_score_fields", [S / "10_score_fields.py"],
             inputs=[NEWS, P / "news_emotion.parquet", P / "news_emotion_raw28.parquet"],
             script=S / "10_score_fields.py",
             outputs=[P / f"news_emotion_raw28_{f}.parquet" for f in ("title", "abstract", "weighted")]
             + [P / "news_emotion_weighted.parquet", P / "news_emotion_go_features.parquet",
                P / "news_emotion_weighted_meta.json", T / "emotion_weighted_vs_combined.json"],
             note="GPU; resumes from emotion_chunks_title/ and emotion_chunks_abstract/"),
        Step("11_lexicon_features", [S / "11_lexicon_features.py"],
             inputs=[NEWS, P / "news_emotion_raw28_weighted.parquet", EMOLEX_PATH, VAD_PATH],
             script=S / "11_lexicon_features.py",
             outputs=[P / "news_emotion_nrc_features.parquet", P / "news_emotion_go_vad.parquet",
                      P / "news_emotion_lexicon_meta.json", T / "lexicon_coverage.json", T / "lexicon_top_words.csv"]),
        Step("12_merge_features", [S / "12_merge_emotion_features.py"],
             inputs=[NEWS, P / "news_emotion_go_features.parquet", P / "news_emotion_go_vad.parquet",
                     P / "news_emotion_nrc_features.parquet", P / "news_emotion_weighted_meta.json",
                     P / "news_emotion_lexicon_meta.json"]
             + [P / f"news_emotion_raw28_{f}.parquet" for f in ("title", "abstract", "weighted")],
             script=S / "12_merge_emotion_features.py",
             outputs=[P / "news_emotion_features.parquet", P / "news_emotion_features_meta.json"]),
        Step("13_diagnostics", [S / "13_emotion_diagnostics.py"], inputs=[P / "news_emotion_features.parquet"],
             script=S / "13_emotion_diagnostics.py",
             outputs=[T / f for f in ("emotion_feature_distributions.csv", "emotion_cross_source_correlations.csv",
                                      "emotion_cross_source_spearman.csv", "emotion_title_vs_abstract.csv",
                                      "emotion_low_mass.json")] + [FIGURES / "emotion_cross_source_spearman.png"]),
        Step("14_emotion_signal", [S / "14_emotion_signal.py"], inputs=[IMPS, NEWS, P / "news_emotion_features.parquet"],
             script=S / "14_emotion_signal.py",
             outputs=[T / f for f in ("emotion_signal_fit.csv", "emotion_signal_variants_fit.csv",
                                      "emotion_signal_by_category_fit.csv", "emotion_signal_fit_meta.json")]
             + [FIGURES / "emotion_signal_fit.png"], note="fit only"),
        Step("15_congruence_signal", [S / "15_congruence_signal.py"],
             inputs=[IMPS, BEH, NEWS, P / "news_emotion_features.parquet", P / "news_emotion_weighted.parquet"],
             script=S / "15_congruence_signal.py",
             outputs=[T / f for f in ("congruence_signal_fit.csv", "congruence_signal_by_category_fit.csv",
                                      "congruence_signal_fit_meta.json")] + [FIGURES / "congruence_signal_fit.png"],
             note="fit only"),
        Step("20_m0_tune", [S / "20_m0_popularity.py", "--tune"], inputs=[IMPS], script=S / "20_m0_popularity.py",
             outputs=[FROZEN_DIR / f"{M0_NAME}.json", T / "m0_popularity_tuning_valid.csv"]
             + [RESULTS / sub / f"{M0_NAME}__valid.parquet" for sub in ("scores", "per_impression")],
             note="valid only; --final-dev is never run by the pipeline"),
        Step("21_m1_tune", [S / "21_m1_tfidf.py", "--tune"], inputs=[IMPS, BEH, NEWS, RAW / "train" / "news.tsv"],
             script=S / "21_m1_tfidf.py",
             outputs=[FROZEN_DIR / f"{M1_NAME}.json", T / "m1_tfidf_tuning_valid.csv"]
             + [RESULTS / sub / f"{M1_NAME}__valid.parquet" for sub in ("scores", "per_impression")],
             note="valid only; --final-dev is never run by the pipeline"),
        Step("22_compare_valid", [S / "22_compare_baselines.py", "--part", "valid"],
             inputs=[P / "data_summary.json"]
             + [RESULTS / "per_impression" / f"{m}__valid.parquet" for m in (M0_NAME, M1_NAME)],
             script=S / "22_compare_baselines.py"),
        Step("06_validate_labels", [S / "06_validate_emotion_labels.py"],
             inputs=[SHEET, P / "news_emotion.parquet", P / "news_emotion_weighted.parquet",
                     P / "news_emotion_nrc_features.parquet"],
             script=S / "06_validate_emotion_labels.py",
             outputs=[T / "emotion_validation.json", T / "emotion_validation_sources.csv",
                      T / "emotion_validation_sources.json"], condition=labels_ready),
    ]


def resolve(names, steps):
    """'14' or '14_emotion_signal' -> full step names."""
    out = []
    for name in names:
        hits = [s.name for s in steps if s.name == name or s.name.split("_")[0] == name]
        if not hits:
            raise SystemExit(f"unknown step {name!r}; steps: {', '.join(s.name for s in steps)}")
        out += hits
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="show the plan, run nothing")
    ap.add_argument("--force", action="store_true", help="rerun every step (create-only steps still keep their files)")
    ap.add_argument("--force-step", nargs="+", default=[], metavar="STEP")
    ap.add_argument("--from", dest="from_step", metavar="STEP", help="force this step and all later steps")
    ap.add_argument("--only", nargs="+", metavar="STEP", help="consider only these steps")
    ap.add_argument("--adopt", action="store_true",
                    help="record existing outputs of out-of-date steps as current instead of rerunning them; "
                         "use once, only when you know the outputs match the current code")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-official-check", action="store_true")
    args = ap.parse_args()

    steps = build_steps(args.skip_official_check)
    if args.skip_tests:
        steps = [s for s in steps if s.name != "00_tests"]
    if args.only:
        keep = set(resolve(args.only, steps))
        steps = [s for s in steps if s.name in keep]
    names = [s.name for s in steps]
    force = set(names) if args.force else set(resolve(args.force_step, steps))
    if args.from_step:
        first = names.index(resolve([args.from_step], steps)[0])
        force |= set(names[first:])

    start = time.time()
    pipe = Pipeline(steps, ROOT, P / ".pipeline")
    try:
        report = pipe.run(force=force, dry_run=args.dry_run, adopt=args.adopt)
        failed = None
    except StepFailed as e:
        report, failed = e.report, str(e)

    print("\n" + "\n".join(f"  {name:22s} {action:10s} {secs:7.1f}s  {reason}" for name, action, reason, secs in report))
    if failed:
        raise SystemExit(f"\nSTOPPED: {failed}")
    ran = sum(a == "ran" for _, a, _, _ in report)
    label = "DRY RUN" if args.dry_run else f"PIPELINE OK: {ran} ran, {len(report) - ran} skipped or waiting"
    print(f"\n{label} ({time.time() - start:.0f}s). Tables: results/tables/  Figures: results/figures/")


if __name__ == "__main__":
    sys.exit(main())
