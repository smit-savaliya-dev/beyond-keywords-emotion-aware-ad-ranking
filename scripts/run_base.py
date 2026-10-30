"""Whole pipeline in order, rerunning only what is out of date. Stops at the first failure.

A step reruns when its data inputs, its code (the script and the src/ modules it imports), its arguments or its
outputs changed since its last successful run (content hashes; state in data/processed/.pipeline/).
Downloads are create-only.

  python scripts/run_base.py --dry-run          # show what would run and why, run nothing
  python scripts/run_base.py                    # run everything that is out of date
  python scripts/run_base.py --adopt            # once: record existing outputs as up to date, run only the rest
  python scripts/run_base.py --rebuild          # rerun everything except GPU scoring (dev runs are never in the pipeline)
  python scripts/run_base.py --force-step 30    # rerun one step (and whatever then changes downstream)
  python scripts/run_base.py --from 30          # rerun step 30 and every later step
  python scripts/run_base.py --only 30 31       # consider only these steps
"""
import argparse
import sys
import time

import _bootstrap  # noqa: F401
from src.config import (ARCHIVED_SHEET, DIAGNOSTICS, EMOLEX_PATH, FROZEN_DIR, M0_NAME, M1_NAME, PROCESSED, RAW, RESULTS, ROOT,
                        SEMEVAL_DIR, SEMEVAL_SPLITS, STORY_CHARTS, STORY_DIR, TABLES, VAD_PATH)
from src.semeval import split_files
from src.pipeline import Pipeline, Step, StepFailed, held_gpu_steps, rebuild_steps

S = ROOT / "scripts"
P, T = PROCESSED, TABLES
RAW_FILES = [RAW / s / f for s in ("train", "dev") for f in ("news.tsv", "behaviors.tsv")]
NEWS, BEH, IMPS = P / "news.parquet", P / "behaviors.parquet", P / "impressions.parquet"
SEMEVAL_FILES = [p for split in SEMEVAL_SPLITS for p in split_files(SEMEVAL_DIR, split)]
EDA_FIGS = ["01_part_sizes", "02_candidates_per_impression", "03_clicks_per_impression", "04_categories",
            "05_history_length", "06_integrity", "eda_overview"]


def build_steps(skip_official_check):
    extra = ["--skip-official-check"] if skip_official_check else []
    code_and_tests = sorted([*(ROOT / "src").glob("*.py"), *(ROOT / "tests").glob("*.py"), ROOT / "pytest.ini"])
    return [
        Step("00_tests", ["-m", "pytest", "-q"], inputs=code_and_tests, adoptable=False),
        Step("10_data_download", [S / "10_data_download.py"], outputs=RAW_FILES, create_only=True),
        Step("11_data_prepare", [S / "11_data_prepare.py", *extra], inputs=RAW_FILES, script=S / "11_data_prepare.py",
             outputs=[NEWS, BEH, IMPS, P / "data_summary.json"]),
        Step("12_data_harness_check", [S / "12_data_harness_check.py"], inputs=[IMPS], script=S / "12_data_harness_check.py"),
        Step("13_data_eda", [S / "13_data_eda.py"], inputs=[NEWS, BEH, IMPS, P / "data_summary.json"], script=S / "13_data_eda.py",
             outputs=[DIAGNOSTICS / f"{f}.png" for f in EDA_FIGS] + [DIAGNOSTICS / "eda_stats.json"]),
        Step("14_data_semeval_download", [S / "14_data_semeval_download.py"], outputs=SEMEVAL_FILES, create_only=True),
        Step("20_emotion_label_combined", [S / "20_emotion_label_combined.py"], inputs=[NEWS], script=S / "20_emotion_label_combined.py",
             outputs=[P / "news_emotion_raw28.parquet", P / "news_emotion.parquet", P / "news_emotion_meta.json"],
             gpu=True, note="GPU; resumes from emotion_chunks/"),
        Step("21_emotion_score_fields", [S / "21_emotion_score_fields.py"],
             inputs=[NEWS, P / "news_emotion.parquet", P / "news_emotion_raw28.parquet"],
             script=S / "21_emotion_score_fields.py",
             outputs=[P / f"news_emotion_raw28_{f}.parquet" for f in ("title", "abstract", "weighted")]
             + [P / "news_emotion_weighted.parquet", P / "news_emotion_go_features.parquet",
                P / "news_emotion_weighted_meta.json", T / "emotion_weighted_vs_combined.json"],
             gpu=True, note="GPU; resumes from emotion_chunks_title/ and emotion_chunks_abstract/"),
        Step("22_emotion_lexicon_features", [S / "22_emotion_lexicon_features.py"],
             inputs=[NEWS, P / "news_emotion_raw28_weighted.parquet", EMOLEX_PATH, VAD_PATH],
             script=S / "22_emotion_lexicon_features.py",
             outputs=[P / "news_emotion_nrc_features.parquet", P / "news_emotion_go_vad.parquet",
                      P / "news_emotion_lexicon_meta.json", T / "lexicon_coverage.json", T / "lexicon_top_words.csv"]),
        Step("23_emotion_merge_features", [S / "23_emotion_merge_features.py"],
             inputs=[NEWS, P / "news_emotion_go_features.parquet", P / "news_emotion_go_vad.parquet",
                     P / "news_emotion_nrc_features.parquet", P / "news_emotion_weighted_meta.json",
                     P / "news_emotion_lexicon_meta.json"]
             + [P / f"news_emotion_raw28_{f}.parquet" for f in ("title", "abstract", "weighted")],
             script=S / "23_emotion_merge_features.py",
             outputs=[P / "news_emotion_features.parquet", P / "news_emotion_features_meta.json"]),
        Step("24_emotion_diagnostics", [S / "24_emotion_diagnostics.py"], inputs=[P / "news_emotion_features.parquet"],
             script=S / "24_emotion_diagnostics.py",
             outputs=[T / f for f in ("emotion_feature_distributions.csv", "emotion_cross_source_correlations.csv",
                                      "emotion_cross_source_spearman.csv", "emotion_title_vs_abstract.csv",
                                      "emotion_low_mass.json")] + [DIAGNOSTICS / "emotion_cross_source_spearman.png"]),
        Step("25_emotion_thresholds", [S / "25_emotion_thresholds.py"],
             inputs=[P / "news_emotion_raw28_weighted.parquet", P / "news_emotion_features.parquet",
                     P / "news_emotion_weighted.parquet"], script=S / "25_emotion_thresholds.py",
             outputs=[T / "emotion_threshold_neutral_share.csv", T / "emotion_threshold_neutral_share_meta.json"], note="item-level, no labels"),
        Step("26_emotion_v2_grid", [S / "26_emotion_v2_grid.py"],
             inputs=[IMPS, BEH, P / "news_emotion_raw28_title.parquet", P / "news_emotion_raw28_abstract.parquet"],
             script=S / "26_emotion_v2_grid.py",
             outputs=[T / "emotion_v2_grid.csv", T / "emotion_v2_weight_congruence.csv", T / "emotion_v2_choice.json"],
             note="fit + valid, no GPU"),
        Step("27_emotion_v2_features", [S / "27_emotion_v2_features.py"],
             inputs=[NEWS, P / "news_emotion_raw28_title.parquet", P / "news_emotion_raw28_abstract.parquet",
                     T / "emotion_v2_choice.json", EMOLEX_PATH, VAD_PATH, ARCHIVED_SHEET],
             script=S / "27_emotion_v2_features.py",
             outputs=[P / "news_emotion_raw28_v2.parquet", P / "news_emotion_labels_v2.parquet",
                      P / "news_emotion_features_v2.parquet", T / "emotion_examples_v2.csv", T / "emotion_v2_meta.json"]),
        Step("28_emotion_label_variants", [S / "28_emotion_label_variants.py"],
             inputs=[P / "news_emotion_raw28_v2.parquet", T / "emotion_v2_choice.json"],
             script=S / "28_emotion_label_variants.py",
             outputs=[P / "news_emotion_labels_variants.parquet", T / "emotion_label_variants_grid.csv",
                      T / "emotion_label_variants_meta.json"]),
        Step("29_emotion_semeval_score", [S / "29_emotion_semeval_score.py"], inputs=SEMEVAL_FILES,
             script=S / "29_emotion_semeval_score.py", outputs=[P / "semeval_raw28.parquet", P / "semeval_raw28_meta.json"],
             gpu=True, note="GPU; small job on SemEval headlines only; resumes from emotion_chunks_semeval/"),
        Step("30_analysis_click_signal", [S / "30_analysis_click_signal.py"], inputs=[IMPS, NEWS, P / "news_emotion_features.parquet"],
             script=S / "30_analysis_click_signal.py",
             outputs=[T / f for f in ("emotion_signal_fit.csv", "emotion_signal_variants_fit.csv",
                                      "emotion_signal_by_category_fit.csv", "emotion_signal_fit_meta.json")], note="fit only"),
        Step("31_analysis_congruence", [S / "31_analysis_congruence.py"],
             inputs=[IMPS, BEH, NEWS, P / "news_emotion_features.parquet", P / "news_emotion_weighted.parquet"],
             script=S / "31_analysis_congruence.py",
             outputs=[T / f for f in ("congruence_signal_fit.csv", "congruence_signal_by_category_fit.csv",
                                      "congruence_signal_fit_meta.json")],
             note="fit only"),
        Step("32_analysis_spec_v1_replication", [S / "32_analysis_spec_v1_replication.py"],
             inputs=[IMPS, BEH, P / "news_emotion_features.parquet", P / "news_emotion_weighted.parquet",
                     T / "emotion_signal_fit.csv", T / "congruence_signal_fit.csv"], script=S / "32_analysis_spec_v1_replication.py",
             outputs=[T / "emotion_spec_selection.csv", T / "emotion_signal_valid.csv",
                      T / "emotion_spec_selection_meta.json"], note="valid only; config/emotion_spec_v1.json is frozen by hand"),
        Step("33_analysis_spec_v2_replication", [S / "33_analysis_spec_v2_replication.py"],
             inputs=[IMPS, BEH, P / "news_emotion_features_v2.parquet", P / "news_emotion_raw28_v2.parquet"],
             script=S / "33_analysis_spec_v2_replication.py",
             outputs=[T / "emotion_signal_v2.csv", T / "emotion_spec_v2_selection.csv",
                      T / "emotion_spec_v2_selection_meta.json"],
             note="fit + valid; config/emotion_spec_v2.json is frozen by hand"),
        Step("34_analysis_context_windows", [S / "34_analysis_context_windows.py"],
             inputs=[IMPS, BEH, P / "news_emotion_features_v2.parquet"], script=S / "34_analysis_context_windows.py",
             outputs=[T / "context_windows_signal.csv", T / "context_windows_comparison.csv",
                      T / "context_windows_meta.json"], note="fit + valid"),
        Step("35_analysis_true_recency", [S / "35_analysis_true_recency.py"],
             inputs=[IMPS, BEH, NEWS, P / "news_emotion_features_v2.parquet"], script=S / "35_analysis_true_recency.py",
             outputs=[T / f for f in ("true_recency_signal.csv", "true_recency_comparison.csv",
                                      "true_recency_residual_by_category.csv", "true_recency_meta.json")],
             note="fit + valid, analysis only"),
        Step("36_analysis_percentile_robustness", [S / "36_analysis_percentile_robustness.py"],
             inputs=[IMPS, BEH, P / "news_emotion_features_v2.parquet", T / "emotion_signal_v2.csv", RAW / "train" / "news.tsv"],
             script=S / "36_analysis_percentile_robustness.py",
             outputs=[T / "percentile_robustness.csv", T / "percentile_robustness_meta.json"], note="fit + valid"),
        Step("37_analysis_story_charts", [S / "37_analysis_story_charts.py"],
             inputs=[NEWS, P / "news_emotion.parquet", P / "news_emotion_weighted.parquet",
                     P / "news_emotion_labels_v2.parquet", P / "news_emotion_labels_variants.parquet",
                     P / "news_emotion_features_v2.parquet",
                     P / "news_emotion_raw28_title.parquet", P / "news_emotion_raw28_abstract.parquet",
                     T / "emotion_v2_grid.csv", T / "emotion_label_variants_grid.csv", T / "emotion_v2_choice.json", T / "emotion_signal_v2.csv",
                     T / "context_windows_signal.csv"], script=S / "37_analysis_story_charts.py",
             outputs=[STORY_DIR / f"{n}.png" for n in STORY_CHARTS]
             + [T / f for f in ("story_emotion_mix.csv", "story_intensity.csv", "story_category_emotion.csv",
                                "story_category_valence_arousal.csv")]),
        Step("38_analysis_semeval_validation", [S / "38_analysis_semeval_validation.py"],
             inputs=[*SEMEVAL_FILES, P / "semeval_raw28.parquet", T / "emotion_label_variants_grid.csv", EMOLEX_PATH, VAD_PATH],
             script=S / "38_analysis_semeval_validation.py",
             outputs=[T / f for f in ("semeval2007_validation.csv", "semeval2007_validation_meta.json",
                                      "semeval2007_selection_trial.csv", "semeval2007_per_class.csv")]
             + [DIAGNOSTICS / "semeval2007_correlations.png"], note="trial selects, test reports"),
        Step("40_model_m0_popularity", [S / "40_model_m0_popularity.py", "--tune"], inputs=[IMPS], script=S / "40_model_m0_popularity.py",
             outputs=[FROZEN_DIR / f"{M0_NAME}.json", T / "m0_popularity_tuning_valid.csv"]
             + [RESULTS / sub / f"{M0_NAME}__valid.parquet" for sub in ("scores", "per_impression")],
             note="valid only; --final-dev is never run by the pipeline"),
        Step("41_model_m1_tfidf", [S / "41_model_m1_tfidf.py", "--tune"], inputs=[IMPS, BEH, NEWS, RAW / "train" / "news.tsv"],
             script=S / "41_model_m1_tfidf.py",
             outputs=[FROZEN_DIR / f"{M1_NAME}.json", T / "m1_tfidf_tuning_valid.csv"]
             + [RESULTS / sub / f"{M1_NAME}__valid.parquet" for sub in ("scores", "per_impression")],
             note="valid only; --final-dev is never run by the pipeline"),
        Step("42_model_compare_baselines", [S / "42_model_compare_baselines.py", "--part", "valid"],
             inputs=[P / "data_summary.json"]
             + [RESULTS / "per_impression" / f"{m}__valid.parquet" for m in (M0_NAME, M1_NAME)],
             script=S / "42_model_compare_baselines.py"),
        Step("90_show_results", [S / "show_results.py"], always=True, adoptable=False),
    ]


def resolve(names, steps):
    """'14' or '30_analysis_click_signal' -> full step names."""
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
    ap.add_argument("--rebuild", action="store_true", help="rerun every step except GPU scoring; dev runs are never included")
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
    explicit = set(names) if args.force else set(resolve(args.force_step, steps))
    force = set(explicit)
    if args.rebuild:
        force |= rebuild_steps(steps)
    if args.from_step:
        first = names.index(resolve([args.from_step], steps)[0])
        force |= set(names[first:])

    start = time.time()
    pipe = Pipeline(steps, ROOT, P / ".pipeline")
    try:
        hold = held_gpu_steps(steps, explicit)
        report = pipe.run(force=force, dry_run=args.dry_run, adopt=args.adopt, hold=hold)
        failed = None
    except StepFailed as e:
        report, failed = e.report, str(e)

    print("\n" + "\n".join(f"  {name:34s} {action:10s} {secs:7.1f}s  {reason}" for name, action, reason, secs in report))
    if failed:
        raise SystemExit(f"\nSTOPPED: {failed}")
    ran = sum(a == "ran" for _, a, _, _ in report)
    label = "DRY RUN" if args.dry_run else f"PIPELINE OK: {ran} ran, {len(report) - ran} skipped or waiting"
    print(f"\n{label} ({time.time() - start:.0f}s). Tables: results/tables/  Figures: results/figures/")


if __name__ == "__main__":
    sys.exit(main())
