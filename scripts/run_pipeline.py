"""Whole pipeline in order, rerunning only what is out of date. Stops at the first failure.

A step reruns when its data inputs, its code (the script and the src/ modules it imports), its arguments or its
outputs changed since its last successful run (content hashes; state in data/processed/.pipeline/).
Downloads are create-only.

  python scripts/run_pipeline.py --dry-run          # show what would run and why, run nothing
  python scripts/run_pipeline.py                    # run everything that is out of date
  python scripts/run_pipeline.py --adopt            # once: record existing outputs as up to date, run only the rest
  python scripts/run_pipeline.py --rebuild          # rerun everything except GPU scoring (dev runs are never in the pipeline)
  python scripts/run_pipeline.py --force-step 03_01  # rerun one step (and whatever then changes downstream)
  python scripts/run_pipeline.py --from 03_01        # rerun this step and every later step
  python scripts/run_pipeline.py --only 03_01 03_02 # consider only these steps (a stage such as 03 selects all its steps)
"""
import argparse
import sys
import time

import _bootstrap  # noqa: F401
from src.config import (ARCHIVED_SHEET, DIAGNOSTICS, EMOLEX_PATH, FROZEN_DIR, M0_NAME, M1_NAME, LTR_FEATURE_FILE, M2_GRID, M2_NAME, M3_NAME, M4_NAME, M4_SAME_NAME, M5_ALL_NAME, M5_CONG_NAME, M5_FAIR_NAME, M5_ITEM_NAME, M5_NAME, M5_PERM_NAME, PROCESSED, RAW, RESULT_CHARTS, RESULTS, RESULTS_FIGS, ROOT,
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
    """The ordered list of pipeline steps with their inputs, outputs and flags."""
    extra = ["--skip-official-check"] if skip_official_check else []
    code_and_tests = sorted([*(ROOT / "src").glob("*.py"), *(ROOT / "tests").glob("*.py"), ROOT / "pytest.ini"])
    return [
        Step("00_tests", ["-m", "pytest", "-q"], inputs=code_and_tests, adoptable=False),
        Step("01_01_download_mind", [S / "01_01_download_mind.py"], outputs=RAW_FILES, create_only=True),
        Step("01_02_prepare_mind", [S / "01_02_prepare_mind.py", *extra], inputs=RAW_FILES, script=S / "01_02_prepare_mind.py",
             outputs=[NEWS, BEH, IMPS, P / "data_summary.json"]),
        Step("01_03_harness_check", [S / "01_03_harness_check.py"], inputs=[IMPS], script=S / "01_03_harness_check.py"),
        Step("01_04_eda", [S / "01_04_eda.py"], inputs=[NEWS, BEH, IMPS, P / "data_summary.json"], script=S / "01_04_eda.py",
             outputs=[DIAGNOSTICS / f"{f}.png" for f in EDA_FIGS] + [DIAGNOSTICS / "eda_stats.json"]),
        Step("01_05_download_semeval", [S / "01_05_download_semeval.py"], outputs=SEMEVAL_FILES, create_only=True),
        Step("02_01_label_combined", [S / "02_01_label_combined.py"], inputs=[NEWS], script=S / "02_01_label_combined.py",
             outputs=[P / "news_emotion_raw28.parquet", P / "news_emotion.parquet", P / "news_emotion_meta.json"],
             gpu=True, note="GPU; resumes from emotion_chunks/"),
        Step("02_02_score_fields", [S / "02_02_score_fields.py"],
             inputs=[NEWS, P / "news_emotion.parquet", P / "news_emotion_raw28.parquet"],
             script=S / "02_02_score_fields.py",
             outputs=[P / f"news_emotion_raw28_{f}.parquet" for f in ("title", "abstract", "weighted")]
             + [P / "news_emotion_weighted.parquet", P / "news_emotion_go_features.parquet",
                P / "news_emotion_weighted_meta.json", T / "emotion_weighted_vs_combined.json"],
             gpu=True, note="GPU; resumes from emotion_chunks_title/ and emotion_chunks_abstract/"),
        Step("02_03_lexicon_features", [S / "02_03_lexicon_features.py"],
             inputs=[NEWS, P / "news_emotion_raw28_weighted.parquet", EMOLEX_PATH, VAD_PATH],
             script=S / "02_03_lexicon_features.py",
             outputs=[P / "news_emotion_nrc_features.parquet", P / "news_emotion_go_vad.parquet",
                      P / "news_emotion_lexicon_meta.json", T / "lexicon_coverage.json", T / "lexicon_top_words.csv"]),
        Step("02_04_merge_features", [S / "02_04_merge_features.py"],
             inputs=[NEWS, P / "news_emotion_go_features.parquet", P / "news_emotion_go_vad.parquet",
                     P / "news_emotion_nrc_features.parquet", P / "news_emotion_weighted_meta.json",
                     P / "news_emotion_lexicon_meta.json"]
             + [P / f"news_emotion_raw28_{f}.parquet" for f in ("title", "abstract", "weighted")],
             script=S / "02_04_merge_features.py",
             outputs=[P / "news_emotion_features.parquet", P / "news_emotion_features_meta.json"]),
        Step("02_05_diagnostics", [S / "02_05_diagnostics.py"], inputs=[P / "news_emotion_features.parquet"],
             script=S / "02_05_diagnostics.py",
             outputs=[T / f for f in ("emotion_feature_distributions.csv", "emotion_cross_source_correlations.csv",
                                      "emotion_cross_source_spearman.csv", "emotion_title_vs_abstract.csv",
                                      "emotion_low_mass.json")] + [DIAGNOSTICS / "emotion_cross_source_spearman.png"]),
        Step("02_06_thresholds", [S / "02_06_thresholds.py"],
             inputs=[P / "news_emotion_raw28_weighted.parquet", P / "news_emotion_features.parquet",
                     P / "news_emotion_weighted.parquet"], script=S / "02_06_thresholds.py",
             outputs=[T / "emotion_threshold_neutral_share.csv", T / "emotion_threshold_neutral_share_meta.json"], note="item-level, no labels"),
        Step("02_07_v2_grid", [S / "02_07_v2_grid.py"],
             inputs=[IMPS, BEH, P / "news_emotion_raw28_title.parquet", P / "news_emotion_raw28_abstract.parquet"],
             script=S / "02_07_v2_grid.py",
             outputs=[T / "emotion_v2_grid.csv", T / "emotion_v2_weight_congruence.csv", T / "emotion_v2_choice.json"],
             note="fit + valid, no GPU"),
        Step("02_08_v2_features", [S / "02_08_v2_features.py"],
             inputs=[NEWS, P / "news_emotion_raw28_title.parquet", P / "news_emotion_raw28_abstract.parquet",
                     T / "emotion_v2_choice.json", EMOLEX_PATH, VAD_PATH, ARCHIVED_SHEET],
             script=S / "02_08_v2_features.py",
             outputs=[P / "news_emotion_raw28_v2.parquet", P / "news_emotion_labels_v2.parquet",
                      P / "news_emotion_features_v2.parquet", T / "emotion_examples_v2.csv", T / "emotion_v2_meta.json"]),
        Step("02_09_label_variants", [S / "02_09_label_variants.py"],
             inputs=[P / "news_emotion_raw28_v2.parquet", T / "emotion_v2_choice.json"],
             script=S / "02_09_label_variants.py",
             outputs=[P / "news_emotion_labels_variants.parquet", T / "emotion_label_variants_grid.csv",
                      T / "emotion_label_variants_meta.json"]),
        Step("02_10_semeval_score", [S / "02_10_semeval_score.py"], inputs=SEMEVAL_FILES,
             script=S / "02_10_semeval_score.py", outputs=[P / "semeval_raw28.parquet", P / "semeval_raw28_meta.json"],
             gpu=True, note="GPU; small job on SemEval headlines only; resumes from emotion_chunks_semeval/"),
        Step("03_01_click_signal", [S / "03_01_click_signal.py"], inputs=[IMPS, NEWS, P / "news_emotion_features.parquet"],
             script=S / "03_01_click_signal.py",
             outputs=[T / f for f in ("emotion_signal_fit.csv", "emotion_signal_variants_fit.csv",
                                      "emotion_signal_by_category_fit.csv", "emotion_signal_fit_meta.json")], note="fit only"),
        Step("03_02_congruence", [S / "03_02_congruence.py"],
             inputs=[IMPS, BEH, NEWS, P / "news_emotion_features.parquet", P / "news_emotion_weighted.parquet"],
             script=S / "03_02_congruence.py",
             outputs=[T / f for f in ("congruence_signal_fit.csv", "congruence_signal_by_category_fit.csv",
                                      "congruence_signal_fit_meta.json")],
             note="fit only"),
        Step("03_03_spec_v1_replication", [S / "03_03_spec_v1_replication.py"],
             inputs=[IMPS, BEH, P / "news_emotion_features.parquet", P / "news_emotion_weighted.parquet",
                     T / "emotion_signal_fit.csv", T / "congruence_signal_fit.csv"], script=S / "03_03_spec_v1_replication.py",
             outputs=[T / "emotion_spec_selection.csv", T / "emotion_signal_valid.csv",
                      T / "emotion_spec_selection_meta.json"], note="valid only; config/emotion_spec_v1.json is frozen by hand"),
        Step("03_04_spec_v2_replication", [S / "03_04_spec_v2_replication.py"],
             inputs=[IMPS, BEH, P / "news_emotion_features_v2.parquet", P / "news_emotion_raw28_v2.parquet"],
             script=S / "03_04_spec_v2_replication.py",
             outputs=[T / "emotion_signal_v2.csv", T / "emotion_spec_v2_selection.csv",
                      T / "emotion_spec_v2_selection_meta.json"],
             note="fit + valid; config/emotion_spec_v2.json is frozen by hand"),
        Step("03_05_context_windows", [S / "03_05_context_windows.py"],
             inputs=[IMPS, BEH, P / "news_emotion_features_v2.parquet"], script=S / "03_05_context_windows.py",
             outputs=[T / "context_windows_signal.csv", T / "context_windows_comparison.csv",
                      T / "context_windows_meta.json"], note="fit + valid"),
        Step("03_06_true_recency", [S / "03_06_true_recency.py"],
             inputs=[IMPS, BEH, NEWS, P / "news_emotion_features_v2.parquet"], script=S / "03_06_true_recency.py",
             outputs=[T / f for f in ("true_recency_signal.csv", "true_recency_comparison.csv",
                                      "true_recency_residual_by_category.csv", "true_recency_meta.json")],
             note="fit + valid, analysis only"),
        Step("03_07_percentile_robustness", [S / "03_07_percentile_robustness.py"],
             inputs=[IMPS, BEH, P / "news_emotion_features_v2.parquet", T / "emotion_signal_v2.csv", RAW / "train" / "news.tsv"],
             script=S / "03_07_percentile_robustness.py",
             outputs=[T / "percentile_robustness.csv", T / "percentile_robustness_meta.json"], note="fit + valid"),
        Step("03_08_story_charts", [S / "03_08_story_charts.py"],
             inputs=[NEWS, P / "news_emotion.parquet", P / "news_emotion_weighted.parquet",
                     P / "news_emotion_labels_v2.parquet", P / "news_emotion_labels_variants.parquet",
                     P / "news_emotion_features_v2.parquet",
                     P / "news_emotion_raw28_title.parquet", P / "news_emotion_raw28_abstract.parquet",
                     T / "emotion_v2_grid.csv", T / "emotion_label_variants_grid.csv", T / "emotion_v2_choice.json", T / "emotion_signal_v2.csv",
                     T / "context_windows_signal.csv"], script=S / "03_08_story_charts.py",
             outputs=[STORY_DIR / f"{n}.png" for n in STORY_CHARTS]
             + [T / f for f in ("story_emotion_mix.csv", "story_intensity.csv", "story_category_emotion.csv",
                                "story_category_valence_arousal.csv")]),
        Step("03_09_semeval_validation", [S / "03_09_semeval_validation.py"],
             inputs=[*SEMEVAL_FILES, P / "semeval_raw28.parquet", P / "news_emotion_raw28_title.parquet",
                     T / "emotion_label_variants_grid.csv", EMOLEX_PATH, VAD_PATH],
             script=S / "03_09_semeval_validation.py",
             outputs=[T / f for f in ("semeval2007_validation.csv", "semeval2007_validation_meta.json",
                                      "semeval2007_selection_trial.csv", "semeval2007_per_class.csv")]
             , note="trial selects, test reports"),
        Step("04_01_m0_popularity", [S / "04_01_m0_popularity.py", "--tune"], inputs=[IMPS], script=S / "04_01_m0_popularity.py",
             outputs=[FROZEN_DIR / f"{M0_NAME}.json", T / "m0_popularity_tuning_valid.csv"]
             + [RESULTS / sub / f"{M0_NAME}__valid.parquet" for sub in ("scores", "per_impression")],
             note="valid only; --final-dev is never run by the pipeline"),
        Step("04_02_m1_tfidf", [S / "04_02_m1_tfidf.py", "--tune"], inputs=[IMPS, BEH, NEWS, RAW / "train" / "news.tsv"],
             script=S / "04_02_m1_tfidf.py",
             outputs=[FROZEN_DIR / f"{M1_NAME}.json", T / "m1_tfidf_tuning_valid.csv"]
             + [RESULTS / sub / f"{M1_NAME}__valid.parquet" for sub in ("scores", "per_impression")],
             note="valid only; --final-dev is never run by the pipeline"),
        Step("04_03_compare_baselines", [S / "04_03_compare_baselines.py", "--part", "valid"],
             inputs=[P / "data_summary.json"]
             + [RESULTS / "per_impression" / f"{m}__valid.parquet" for m in (M0_NAME, M1_NAME)],
             script=S / "04_03_compare_baselines.py"),
        Step("04_04_m2_encode", [S / "04_04_m2_encode.py"], inputs=[NEWS], script=S / "04_04_m2_encode.py",
             outputs=[P / "sbert_meta.json"] + [P / f"sbert_{t}{e}" for t in M2_GRID["text"] for e in (".npy", "_ids.parquet")],
             gpu=True, note="GPU; resumes from sbert_chunks_title/ and sbert_chunks_text/"),
        Step("04_05_m2_sbert", [S / "04_05_m2_sbert.py", "--tune"], script=S / "04_05_m2_sbert.py",
             inputs=[IMPS, BEH, NEWS] + [P / f"sbert_{t}.npy" for t in M2_GRID["text"]],
             outputs=[FROZEN_DIR / f"{M2_NAME}.json", T / "m2_sbert_tuning_valid.csv"]
             + [RESULTS / sub / f"{M2_NAME}__valid.parquet" for sub in ("scores", "per_impression")],
             note="valid only; --final-dev is never run by the pipeline"),
        Step("04_06_m3_fusion", [S / "04_06_m3_fusion.py", "--tune"], script=S / "04_06_m3_fusion.py",
             extra_code=[ROOT / "config" / "emotion_spec_v4.json"],
             inputs=[IMPS, BEH, P / "news_emotion_features_v2.parquet", P / "news_emotion_features.parquet", FROZEN_DIR / f"{M2_NAME}.json"]
             + [RESULTS / sub / f"{M2_NAME}__valid.parquet" for sub in ("scores", "per_impression")],
             outputs=[FROZEN_DIR / f"{M3_NAME}.json"]
             + [T / f for f in ("m3_fusion_grid_valid.csv", "m3_shuffled_control_valid.csv", "m3_fusion_meta.json")]
             + [RESULTS / sub / f"{M3_NAME}__valid.parquet" for sub in ("scores", "per_impression")],
             note="valid only; stops if a D54 check fails; --final-dev is never run by the pipeline"),
        Step("04_07_compare_m2m3", [S / "04_07_compare_m2m3.py", "--part", "valid"], script=S / "04_07_compare_m2m3.py",
             inputs=[BEH, P / "data_summary.json"] + [RESULTS / "per_impression" / f"{m}__valid.parquet" for m in (M1_NAME, M2_NAME, M3_NAME)],
             outputs=[T / "m2m3_comparison.csv", T / "m2m3_user_level.csv"]),
        Step("04_09_ltr_features", [S / "04_09_ltr_features.py"], script=S / "04_09_ltr_features.py",
             inputs=[IMPS, BEH, NEWS, RAW / "train" / "news.tsv", P / "news_emotion_features_v2.parquet", P / "news_emotion_raw28_v2.parquet",
                     *[P / f"sbert_{t}.npy" for t in M2_GRID["text"]], *[FROZEN_DIR / f"{m}.json" for m in (M0_NAME, M1_NAME, M2_NAME)],
                     *[RESULTS / "scores" / f"{m}__valid.parquet" for m in (M0_NAME, M1_NAME, M2_NAME)]],
             extra_code=[ROOT / "config" / "emotion_spec_v4.json"],
             outputs=[P / LTR_FEATURE_FILE.format(part=p) for p in ("fit", "valid")] + [T / "ltr_features_meta.json"],
             note="fit + valid; click statistics strictly earlier in time (D64)"),
        Step("04_10_m4m5_lightgbm", [S / "04_10_m4m5_lightgbm.py", "--tune"], script=S / "04_10_m4m5_lightgbm.py",
             inputs=[IMPS, *[P / LTR_FEATURE_FILE.format(part=p) for p in ("fit", "valid")]],
             extra_code=[ROOT / "config" / "emotion_spec_v4.json"],
             outputs=[FROZEN_DIR / f"{m}.json" for m in (M4_NAME, M5_NAME, M5_ALL_NAME)]
             + [T / f for f in ("m4m5_tuning_valid.csv", "m4m5_seed_variability_valid.csv", "m4m5_meta.json")]
             + [RESULTS / sub / f"{m}__valid.parquet" for sub in ("scores", "per_impression") for m in (M4_NAME, M5_NAME, M5_ALL_NAME)]
             + [RESULTS / "per_impression" / f"{M5_FAIR_NAME}__valid.parquet"],
             note="valid only; stops if a D67 check fails; --final-dev is never run by the pipeline"),
        Step("05_01_m5_controls", [S / "05_01_m5_controls.py", "--tune"], script=S / "05_01_m5_controls.py",
             inputs=[BEH, IMPS, FROZEN_DIR / f"{M5_NAME}.json", *[P / LTR_FEATURE_FILE.format(part=p) for p in ("fit", "valid")],
                     P / "news_emotion_features_v2.parquet", P / "news_emotion_raw28_v2.parquet"],
             outputs=[FROZEN_DIR / f"{m}.json" for m in (M5_PERM_NAME, M5_CONG_NAME)] + [T / "m5_controls_valid.csv", T / "m5_controls_meta.json"]
             + [RESULTS / "per_impression" / f"{m}__valid.parquet" for m in (M5_PERM_NAME, M5_CONG_NAME, M5_ITEM_NAME, M4_SAME_NAME)],
             note="valid only; item-identity controls (D69)"),
        Step("05_02_compare_m4m5", [S / "05_02_compare_m4m5.py", "--part", "valid"], script=S / "05_02_compare_m4m5.py",
             inputs=[BEH, IMPS, P / "data_summary.json"] + [RESULTS / "per_impression" / f"{m}__valid.parquet"
                                                            for m in (M2_NAME, M3_NAME, M4_NAME, M5_NAME, M5_ALL_NAME, M5_FAIR_NAME, M5_PERM_NAME, M5_CONG_NAME,
                                                                      M5_ITEM_NAME, M4_SAME_NAME)],
             outputs=[T / "m4m5_comparison.csv", T / "m4m5_user_level.csv", T / "m5_fresh_items.csv"]),
        Step("05_03_feature_importance", [S / "05_03_feature_importance.py"], script=S / "05_03_feature_importance.py",
             inputs=[P / LTR_FEATURE_FILE.format(part="valid"), FROZEN_DIR / f"{M5_NAME}.json"],
             outputs=[T / "m5_feature_importance.csv", T / "m5_feature_importance_meta.json", DIAGNOSTICS / "m5_feature_importance.png"],
             note="valid only"),
        Step("05_05_results_charts", [S / "05_05_results_charts.py"], script=S / "05_05_results_charts.py",
             inputs=[T / f for f in ("model_ladder.csv", "m2m3_comparison.csv", "m4m5_comparison.csv", "m3_fusion_grid_valid.csv", "m3_fusion_meta.json",
                                     "m3_where_it_helps_dev.csv", "semeval2007_validation.csv")],
             outputs=[RESULTS_FIGS / f"{n}.png" for n in RESULT_CHARTS] + [T / "results_chart_captions.json"], note="saved tables only"),
        Step("90_show_results", [S / "show_results.py"], always=True, adoptable=False),
    ]


def resolve(names, steps):
    """A full step name, a stage ('04') or a step ('04_06') -> full step names."""
    out = []
    for name in names:
        hits = [s.name for s in steps if s.name == name or s.name.startswith(name + "_")]
        if not hits:
            raise SystemExit(f"unknown step {name!r}; steps: {', '.join(s.name for s in steps)}")
        out += hits
    return out


def main():
    """Parse the options, choose which steps to run and report what ran."""
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
