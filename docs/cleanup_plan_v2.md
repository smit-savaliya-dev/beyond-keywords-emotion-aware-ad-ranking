# Cleanup plan v2 (tidy-for-review)

Rule: no logic and no result changes. Proof: `scripts/check_logic_unchanged.py` (normalised syntax trees identical, old names mapped to new), `scripts/snapshot_results.py --check` (tables within 1e-9; scores, per-impression files, boosters and figures byte for byte), and the full test suite.

## What the audit found
- Dead code: none. Every function in `src/` is called from a script or another module; `spec_hash`, `validate_spec` and `evaluate_model` are used by tests only and are kept on purpose (hash lock for the specs, contract for future models; D44).
- Unused imports (4: `Path`, `EMOLEX_PATH`, `VAD_PATH` in the SemEval validation script, `pytest` in `test_label_thresholds.py`): kept, because removing them would change the syntax tree and the proof is strict.
- No TODO, FIXME or commented-out code was found. Debug or temporary scripts: none.
- Unreferenced result files (`m3_where_it_helps_*`, `m5_where_it_helps_*`, `*_dev_meta.json`) are written by name patterns in the scripts and are dev results: kept.
- Superseded results were already archived in earlier cleanups (`results/archive/`); nothing new needed archiving. `data/processed/ltr_runs/` and the chunk folders are resumable caches: kept.

## Deleted (untracked or ignored, nothing depends on them)
| Item | Action | Reason |
|---|---|---|
| `__pycache__/` in `scripts/`, `src/`, `tests/` | delete | Python byte-code caches; recreated on demand, git-ignored |
| `.pytest_cache/` | delete | Test cache; recreated on demand, git-ignored |
| `mlruns/` (only an empty `.gitkeep`) | delete | MLflow is not used anywhere in the code or requirements; git-ignored |
| `data/processed/.pipeline/` stamps of the old step names | delete after the final rebuild | Pipeline state keyed by step name; the old names no longer exist |

## Rename table (old -> new)
Names are `<stage>_<number>_<what>`: 01 data, 02 emotion, 03 analysis, 04 models, 05 results. The number is two digits so that 10 sorts after 9. The specs in `config/` and `docs/decisions.md` keep the old names (hash-locked or historical); use this table to translate them. Step ids of the pipeline are the file names without `.py`; `--only 04_06` selects one step and `--only 04` a whole stage.
| Old | New |
|---|---|
| `scripts/10_data_download.py` | `scripts/01_01_download_mind.py` |
| `scripts/11_data_prepare.py` | `scripts/01_02_prepare_mind.py` |
| `scripts/12_data_harness_check.py` | `scripts/01_03_harness_check.py` |
| `scripts/13_data_eda.py` | `scripts/01_04_eda.py` |
| `scripts/14_data_semeval_download.py` | `scripts/01_05_download_semeval.py` |
| `scripts/20_emotion_label_combined.py` | `scripts/02_01_label_combined.py` |
| `scripts/21_emotion_score_fields.py` | `scripts/02_02_score_fields.py` |
| `scripts/22_emotion_lexicon_features.py` | `scripts/02_03_lexicon_features.py` |
| `scripts/23_emotion_merge_features.py` | `scripts/02_04_merge_features.py` |
| `scripts/24_emotion_diagnostics.py` | `scripts/02_05_diagnostics.py` |
| `scripts/25_emotion_thresholds.py` | `scripts/02_06_thresholds.py` |
| `scripts/26_emotion_v2_grid.py` | `scripts/02_07_v2_grid.py` |
| `scripts/27_emotion_v2_features.py` | `scripts/02_08_v2_features.py` |
| `scripts/28_emotion_label_variants.py` | `scripts/02_09_label_variants.py` |
| `scripts/29_emotion_semeval_score.py` | `scripts/02_10_semeval_score.py` |
| `scripts/30_analysis_click_signal.py` | `scripts/03_01_click_signal.py` |
| `scripts/31_analysis_congruence.py` | `scripts/03_02_congruence.py` |
| `scripts/32_analysis_spec_v1_replication.py` | `scripts/03_03_spec_v1_replication.py` |
| `scripts/33_analysis_spec_v2_replication.py` | `scripts/03_04_spec_v2_replication.py` |
| `scripts/34_analysis_context_windows.py` | `scripts/03_05_context_windows.py` |
| `scripts/35_analysis_true_recency.py` | `scripts/03_06_true_recency.py` |
| `scripts/36_analysis_percentile_robustness.py` | `scripts/03_07_percentile_robustness.py` |
| `scripts/37_analysis_story_charts.py` | `scripts/03_08_story_charts.py` |
| `scripts/38_analysis_semeval_validation.py` | `scripts/03_09_semeval_validation.py` |
| `scripts/40_model_m0_popularity.py` | `scripts/04_01_m0_popularity.py` |
| `scripts/41_model_m1_tfidf.py` | `scripts/04_02_m1_tfidf.py` |
| `scripts/42_model_compare_baselines.py` | `scripts/04_03_compare_baselines.py` |
| `scripts/43_model_m2_encode.py` | `scripts/04_04_m2_encode.py` |
| `scripts/44_model_m2_sbert.py` | `scripts/04_05_m2_sbert.py` |
| `scripts/45_model_m3_fusion.py` | `scripts/04_06_m3_fusion.py` |
| `scripts/46_model_compare_m2m3.py` | `scripts/04_07_compare_m2m3.py` |
| `scripts/47_model_m3_where_it_helps.py` | `scripts/04_08_where_it_helps.py` |
| `scripts/48_model_ltr_features.py` | `scripts/04_09_ltr_features.py` |
| `scripts/49_model_m4m5_lightgbm.py` | `scripts/04_10_m4m5_lightgbm.py` |
| `scripts/50_results_m5_controls.py` | `scripts/05_01_m5_controls.py` |
| `scripts/51_results_compare_m4m5.py` | `scripts/05_02_compare_m4m5.py` |
| `scripts/52_results_feature_importance.py` | `scripts/05_03_feature_importance.py` |
| `scripts/53_results_model_ladder.py` | `scripts/05_04_model_ladder.py` |
| `scripts/54_results_charts.py` | `scripts/05_05_results_charts.py` |
| `scripts/run_base.py` | `scripts/run_pipeline.py` |
| `src/experiments.py` | `src/run_registry.py` |
| `src/validation.py` | `src/label_thresholds.py` |
| `src/summary.py` | `src/terminal_summary.py` |
| `src/ltr.py` | `src/lightgbm_ranker.py` |
| `tests/test_ltr.py` | `tests/test_lightgbm_ranker.py` |
| `tests/test_summary.py` | `tests/test_terminal_summary.py` |
| `tests/test_validation.py` | `tests/test_label_thresholds.py` |
| `tests/test_stats_experiments_baselines.py` | `tests/test_stats_run_registry_baselines.py` |
| `tests/test_run_base.py` | `tests/test_run_pipeline.py` |

Totals: 49 renamed, 0 files deleted, 3 files added (the proof tooling), 220 kept as they are.

## Every file
| File | Action | New name | Reason |
|---|---|---|---|
| `.gitignore` | keep |  | Project file |
| `CLAUDE.md` | keep |  | Project file |
| `README.md` | keep |  | Project file |
| `config/emotion_spec_v1.json` | keep |  | Frozen emotion specification; hash-locked by a test, never edited |
| `config/emotion_spec_v2.json` | keep |  | Frozen emotion specification; hash-locked by a test, never edited |
| `config/emotion_spec_v3.json` | keep |  | Frozen emotion specification; hash-locked by a test, never edited |
| `config/emotion_spec_v4.json` | keep |  | Frozen emotion specification; hash-locked by a test, never edited |
| `docs/START_HERE.md` | keep |  | Documentation |
| `docs/cleanup_plan.md` | keep |  | Documentation |
| `docs/decisions.md` | keep |  | Documentation |
| `docs/figures_guide.md` | keep |  | Documentation |
| `docs/methodology_notes.md` | keep |  | Documentation |
| `docs/open_questions.md` | keep |  | Documentation |
| `docs/progress.md` | keep |  | Documentation |
| `docs/reports/base_v2_report.md` | keep |  | Report for a finished stage (history) |
| `docs/reports/base_v3_report.md` | keep |  | Report for a finished stage (history) |
| `docs/reports/base_v4_report.md` | keep |  | Report for a finished stage (history) |
| `docs/reports/m2m3_report.md` | keep |  | Report for a finished stage (history) |
| `docs/reports/m4m5_report.md` | keep |  | Report for a finished stage (history) |
| `docs/reports/week5_report_draft.md` | keep |  | Report for a finished stage (history) |
| `docs/reports/week5_run_report.md` | keep |  | Report for a finished stage (history) |
| `pytest.ini` | keep |  | Project file |
| `requirements.txt` | keep |  | Project file |
| `results/archive/README.md` | keep |  | Archived or proof record |
| `results/archive/annotation/README.md` | keep |  | Cancelled hand-label material (D45), archived on purpose |
| `results/archive/annotation/annotation_sheet_functions.py` | keep |  | Cancelled hand-label material (D45), archived on purpose |
| `results/archive/annotation/label_tool.py` | keep |  | Cancelled hand-label material (D45), archived on purpose |
| `results/archive/annotation/labelling.py` | keep |  | Cancelled hand-label material (D45), archived on purpose |
| `results/archive/annotation/labelling_guidelines.md` | keep |  | Cancelled hand-label material (D45), archived on purpose |
| `results/archive/annotation/make_annotation_sheet.py` | keep |  | Cancelled hand-label material (D45), archived on purpose |
| `results/archive/annotation/manual_sample.csv` | keep |  | Cancelled hand-label material (D45), archived on purpose |
| `results/archive/annotation/test_labelling.py` | keep |  | Cancelled hand-label material (D45), archived on purpose |
| `results/archive/figures/congruence_signal_fit.png` | keep |  | Archived or proof record |
| `results/archive/figures/emotion_signal_fit.png` | keep |  | Archived or proof record |
| `results/archive/figures/emotion_threshold_neutral_share.png` | keep |  | Archived or proof record |
| `results/archive/pre_cleanup_snapshot.json` | keep |  | Archived or proof record |
| `results/archive/tidy_ast_baseline.json` | keep |  | Archived or proof record |
| `results/archive/tidy_ast_baseline_step0.json` | keep |  | Archived or proof record |
| `results/archive/tidy_hashes.json` | keep |  | Archived or proof record |
| `results/archive/tidy_snapshot.json` | keep |  | Archived or proof record |
| `results/figures/.gitkeep` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/diagnostics/01_part_sizes.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/diagnostics/02_candidates_per_impression.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/diagnostics/03_clicks_per_impression.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/diagnostics/04_categories.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/diagnostics/05_history_length.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/diagnostics/06_integrity.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/diagnostics/eda_overview.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/diagnostics/eda_stats.json` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/diagnostics/emotion_cross_source_spearman.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/diagnostics/m5_feature_importance.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/results/1_model_ladder.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/results/2_m3_gain_by_metric.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/results/3_alpha_ablation.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/results/4_m3_gain_by_category.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/results/5_m3_gain_by_history.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/results/6_emotion_validation.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/story/1_emotion_mix.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/story/2_neutral_vs_threshold.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/story/3_title_vs_abstract_intensity.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/story/4_emotion_by_category.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/story/5_valence_arousal_map.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/story/6_clicked_vs_not_effects.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/story/7_congruence_effects.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/figures/story/8_short_vs_long_context.png` | keep |  | Figure listed in docs/figures_guide.md |
| `results/frozen/M0_popularity.json` | keep |  | Frozen model settings; boosters are checked against them |
| `results/frozen/M1_tfidf.json` | keep |  | Frozen model settings; boosters are checked against them |
| `results/frozen/M2_sbert.json` | keep |  | Frozen model settings; boosters are checked against them |
| `results/frozen/M3_sbert_emotion.json` | keep |  | Frozen model settings; boosters are checked against them |
| `results/frozen/M4_lightgbm.json` | keep |  | Frozen model settings; boosters are checked against them |
| `results/frozen/M5_all_lightgbm.json` | keep |  | Frozen model settings; boosters are checked against them |
| `results/frozen/M5_congruence_only.json` | keep |  | Frozen model settings; boosters are checked against them |
| `results/frozen/M5_lightgbm_emotion.json` | keep |  | Frozen model settings; boosters are checked against them |
| `results/frozen/M5_permuted_items.json` | keep |  | Frozen model settings; boosters are checked against them |
| `results/reports/week6_report_draft.md` | keep |  | Report draft (history) |
| `results/reports/week7_report_draft.md` | keep |  | Report draft (history) |
| `results/tables/.gitkeep` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/M4_lightgbm_dev_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/M5_lightgbm_emotion_dev_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/baselines_comparison.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/baselines_sanity.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/congruence_signal_by_category_fit.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/congruence_signal_fit.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/congruence_signal_fit_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/context_windows_comparison.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/context_windows_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/context_windows_signal.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_cross_source_correlations.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_cross_source_spearman.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_examples_v2.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_feature_distributions.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_label_variants_grid.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_label_variants_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_low_mass.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_signal_by_category_fit.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_signal_fit.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_signal_fit_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_signal_v2.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_signal_valid.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_signal_variants_fit.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_spec_selection.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_spec_selection_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_spec_v2_selection.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_spec_v2_selection_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_threshold_neutral_share.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_threshold_neutral_share_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_title_vs_abstract.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_v2_choice.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_v2_grid.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_v2_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_v2_weight_congruence.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/emotion_weighted_vs_combined.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/lexicon_coverage.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/lexicon_top_words.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/ltr_features_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m0_popularity_dev_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m0_popularity_tuning_valid.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m1_tfidf_dev_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m1_tfidf_tuning_valid.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m2_sbert_dev_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m2_sbert_tuning_valid.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m2m3_comparison.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m2m3_user_level.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m3_fusion_dev_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m3_fusion_grid_valid.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m3_fusion_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m3_shuffled_control_valid.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m3_where_it_helps_dev.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m3_where_it_helps_examples_dev.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m3_where_it_helps_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m4m5_comparison.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m4m5_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m4m5_seed_variability_valid.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m4m5_tuning_valid.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m4m5_user_level.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m5_controls_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m5_controls_valid.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m5_feature_importance.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m5_feature_importance_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m5_fresh_items.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m5_where_it_helps_dev.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/m5_where_it_helps_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/model_ladder.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/percentile_robustness.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/percentile_robustness_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/results_chart_captions.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/runs_log.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/sbert_encoder_check.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/semeval2007_per_class.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/semeval2007_selection_trial.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/semeval2007_validation.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/semeval2007_validation_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/story_category_emotion.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/story_category_valence_arousal.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/story_emotion_mix.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/story_intensity.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/summary.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/token_lengths.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/true_recency_comparison.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/true_recency_meta.json` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/true_recency_residual_by_category.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `results/tables/true_recency_signal.csv` | keep |  | Saved result table (read by charts, summary or reports) |
| `scripts/10_data_download.py` | rename | `scripts/01_01_download_mind.py` | Pipeline step or tool; Download and unzip MIND-small into data/raw/ (skips what is already there). |
| `scripts/11_data_prepare.py` | rename | `scripts/01_02_prepare_mind.py` | Pipeline step or tool; Parse MIND-small into data/processed/*.parquet and verify it. |
| `scripts/12_data_harness_check.py` | rename | `scripts/01_03_harness_check.py` | Pipeline step or tool; Prove the evaluation harness on real data with fake models whose results are known in advance. |
| `scripts/13_data_eda.py` | rename | `scripts/01_04_eda.py` | Pipeline step or tool; EDA figures: six individual PNGs + overview in results/figures/diagnostics/. |
| `scripts/14_data_semeval_download.py` | rename | `scripts/01_05_download_semeval.py` | Pipeline step or tool; Download SemEval-2007 Task 14 Affective Text (Strapparava & Mihalcea, 2007) into data/external/semeval2007/ an |
| `scripts/20_emotion_label_combined.py` | rename | `scripts/02_01_label_combined.py` | Pipeline step or tool; Label every news item with emotion vectors. Resumable: re-run to continue after an interruption. |
| `scripts/21_emotion_score_fields.py` | rename | `scripts/02_02_score_fields.py` | Pipeline step or tool; Score titles and abstracts separately with GoEmotions, combine with the field weights, build go_* features. |
| `scripts/22_emotion_lexicon_features.py` | rename | `scripts/02_03_lexicon_features.py` | Pipeline step or tool; NRC EmoLex + VAD features (title/abstract weighted) and classifier-derived go_vad_* features. CPU, about a min |
| `scripts/23_emotion_merge_features.py` | rename | `scripts/02_04_merge_features.py` | Pipeline step or tool; Merge all item-level emotion features into data/processed/news_emotion_features.parquet (+ metadata JSON). |
| `scripts/24_emotion_diagnostics.py` | rename | `scripts/02_05_diagnostics.py` | Pipeline step or tool; Item-level diagnostics of news_emotion_features.parquet: distributions, cross-source and title-vs-abstract |
| `scripts/25_emotion_thresholds.py` | rename | `scripts/02_06_thresholds.py` | Pipeline step or tool; Item-level threshold sensitivity (no labels, no clicks): class shares on the full corpus for every pre-declare |
| `scripts/26_emotion_v2_grid.py` | rename | `scripts/02_07_v2_grid.py` | Pipeline step or tool; Step 1 of base-v2: title-weight / threshold grid on the SAVED raw 28 scores (no GPU), selection rule D30. |
| `scripts/27_emotion_v2_features.py` | rename | `scripts/02_08_v2_features.py` | Pipeline step or tool; Build the v2 emotion tables for the setting chosen by D30 (emotion_v2_choice.json). New files only; v1 is unto |
| `scripts/28_emotion_label_variants.py` | rename | `scripts/02_09_label_variants.py` | Pipeline step or tool; Stance-excluded labelling variants (D38): v2b ignores approval and disapproval, v2c also ignores realization,  |
| `scripts/29_emotion_semeval_score.py` | rename | `scripts/02_10_semeval_score.py` | Pipeline step or tool; Score the SemEval-2007 headlines with GoEmotions (same model, max_length 128). Small GPU job on new data only; |
| `scripts/30_analysis_click_signal.py` | rename | `scripts/03_01_click_signal.py` | Pipeline step or tool; Does emotion relate to clicks? Descriptive signal analysis on part == "fit" ONLY (valid and dev never loaded). |
| `scripts/31_analysis_congruence.py` | rename | `scripts/03_02_congruence.py` | Pipeline step or tool; Congruence signal on part == "fit" ONLY: is a candidate's emotional similarity to the user's HISTORY profile |
| `scripts/32_analysis_spec_v1_replication.py` | rename | `scripts/03_03_spec_v1_replication.py` | Pipeline step or tool; Valid-set replication of the emotion spec candidates (D27). Same statistics as Phase 5 (per-impression |
| `scripts/33_analysis_spec_v2_replication.py` | rename | `scripts/03_04_spec_v2_replication.py` | Pipeline step or tool; D32: re-apply the D27 replication rule on the v2 features (fit, then valid; dev is never read). |
| `scripts/34_analysis_context_windows.py` | rename | `scripts/03_05_context_windows.py` | Pipeline step or tool; Step 2 of base-v2 (D31): which window of the user's reading history carries the emotional context? |
| `scripts/35_analysis_true_recency.py` | rename | `scripts/03_06_true_recency.py` | Pipeline step or tool; True recency test (D39), analysis only, fit and valid (dev is never read). Short-term context = the user's cli |
| `scripts/36_analysis_percentile_robustness.py` | rename | `scripts/03_07_percentile_robustness.py` | Pipeline step or tool; Q03 / D40: do the results change if go_pct_* percentiles are fitted on train-split items only (data/raw/train/ |
| `scripts/37_analysis_story_charts.py` | rename | `scripts/03_08_story_charts.py` | Pipeline step or tool; Step 3 of base-v2: the eight story charts in results/figures/story/ (and the data behind them in results/table |
| `scripts/38_analysis_semeval_validation.py` | rename | `scripts/03_09_semeval_validation.py` | Pipeline step or tool; External validation on SemEval-2007 Task 14 Affective Text (Strapparava & Mihalcea, 2007). Pre-declared in D46 |
| `scripts/40_model_m0_popularity.py` | rename | `scripts/04_01_m0_popularity.py` | Pipeline step or tool; M0 popularity: Beta-smoothed item CTR (clicks + a*p) / (shown + a) toward the global CTR p; unseen items get p |
| `scripts/41_model_m1_tfidf.py` | rename | `scripts/04_02_m1_tfidf.py` | Pipeline step or tool; M1 TF-IDF: vectorizer fitted on TRAIN-split news only (label-free); user vector = mean or decay-weighted mean  |
| `scripts/42_model_compare_baselines.py` | rename | `scripts/04_03_compare_baselines.py` | Pipeline step or tool; Paired comparison M1 vs M0 on per-impression metrics (paired t-test, 95% CI, dz, Holm across metrics; Wilcoxon |
| `scripts/43_model_m2_encode.py` | rename | `scripts/04_04_m2_encode.py` | Pipeline step or tool; Encode every news item with all-mpnet-base-v2, one cache per text variant (D50). GPU, resumable, no labels inv |
| `scripts/44_model_m2_sbert.py` | rename | `scripts/04_05_m2_sbert.py` | Pipeline step or tool; M2 Sentence-BERT, no emotion: user vector = mean or decay-weighted mean of the last 50 HISTORY embeddings; |
| `scripts/45_model_m3_fusion.py` | rename | `scripts/04_06_m3_fusion.py` | Pipeline step or tool; M3 = M2 + emotion, late fusion (spec v3): final = alpha * norm(M2) + (1 - alpha) * norm(emotion), min-max per  |
| `scripts/46_model_compare_m2m3.py` | rename | `scripts/04_07_compare_m2m3.py` | Pipeline step or tool; Paired comparisons for M2 and M3 (D51, D55): per-impression metrics with Holm, 95% CI and dz; user-level AUC a |
| `scripts/47_model_m3_where_it_helps.py` | rename | `scripts/04_08_where_it_helps.py` | Pipeline step or tool; RQ5, descriptive (D56, D68): per-impression AUC difference on dev, by category, profile emotion and history le |
| `scripts/48_model_ltr_features.py` | rename | `scripts/04_09_ltr_features.py` | Pipeline step or tool; Feature tables for M4 and M5 on fit and valid (D63-D65), with check D67b: valid features must equal the saved  |
| `scripts/49_model_m4m5_lightgbm.py` | rename | `scripts/04_10_m4m5_lightgbm.py` | Pipeline step or tool; M4 LightGBM (no emotion), M5 = M4 + emotion (spec v4), M5-all (robustness, valid only). D63-D67. |
| `scripts/50_results_m5_controls.py` | rename | `scripts/05_01_m5_controls.py` | Pipeline step or tool; Item-identity controls for M5 (D69). Does the M5 gain come from what emotion means, or from emotion scores act |
| `scripts/51_results_compare_m4m5.py` | rename | `scripts/05_02_compare_m4m5.py` | Pipeline step or tool; Paired comparisons for M4 and M5 (D68): per-impression metrics with Holm, 95% CI and dz; user-level AUC as a r |
| `scripts/52_results_feature_importance.py` | rename | `scripts/05_03_feature_importance.py` | Pipeline step or tool; Where do emotion features rank inside M5? LightGBM gain importance and mean |pred_contrib| on seeded valid imp |
| `scripts/53_results_model_ladder.py` | rename | `scripts/05_04_model_ladder.py` | Pipeline step or tool; Mean dev AUC with a 95% CI per model, from the saved per-impression results. Reads no impressions and scores n |
| `scripts/54_results_charts.py` | rename | `scripts/05_05_results_charts.py` | Pipeline step or tool; The results charts for the supervisor (results/figures/results/), from saved tables only. Reads no impression  |
| `scripts/_bootstrap.py` | keep |  | Pipeline step or tool; thin script |
| `scripts/check_goemotions.py` | keep |  | Pipeline step or tool; Optional in-domain check: the classifier + 28->6 mapping on the GoEmotions test split. |
| `scripts/check_logic_unchanged.py` | keep |  | Pipeline step or tool; Prove the tidy-up changed no logic: every .py file has the same normalised syntax tree as at the base commit. |
| `scripts/check_sbert_encoder.py` | keep |  | Pipeline step or tool; Correctness check of D50: saved transformers-based embeddings vs the sentence-transformers package, on random  |
| `scripts/run_base.py` | rename | `scripts/run_pipeline.py` | Pipeline step or tool; Whole pipeline in order, rerunning only what is out of date. Stops at the first failure. |
| `scripts/show_results.py` | keep |  | Pipeline step or tool; Print a short summary of the results from saved tables. Reads no impression data, so dev is never loaded. |
| `scripts/snapshot_results.py` | keep |  | Pipeline step or tool; Prove that a change did not alter results: --take stores a manifest and copies, --check compares (floats withi |
| `src/__init__.py` | keep |  | Used; library module |
| `src/agreement.py` | keep |  | Used; Agreement statistics between emotion labels: kappa with a bootstrap CI, accuracy, macro-F1. |
| `src/baselines.py` | keep |  | Used; M0 popularity (smoothed CTR) and M1 TF-IDF (history profile cosine). No emotion. |
| `src/click_signal.py` | keep |  | Used; Descriptive click-signal tests: within-impression clicked-minus-unclicked feature differences. |
| `src/config.py` | keep |  | Used; library module |
| `src/congruence.py` | keep |  | Used; User emotion profiles from click HISTORY only, and item-profile congruence (cosine, negative KL). |
| `src/data.py` | keep |  | Used; library module |
| `src/diagnostics.py` | keep |  | Used; Item-level diagnostics of the emotion feature table: distributions and cross-source correlations. |
| `src/eda.py` | keep |  | Used; EDA figures for MIND-small: six individual captioned PNGs plus one combined overview. |
| `src/emotion.py` | keep |  | Used; GoEmotions (28 labels) -> 6 emotion classes, plus resumable corpus labelling. |
| `src/emotion_features.py` | keep |  | Used; Weighted title/abstract GoEmotions layer: field texts, weighted raw scores, go_* and go_dist_* features. |
| `src/emotion_spec.py` | keep |  | Used; Versioned emotion specification for M3/M5/M7 (config/emotion_spec_v<N>.json) and the replication rule. |
| `src/emotion_v2.py` | keep |  | Used; Emotion layer v2: title-weight / threshold grid on saved raw scores, selection rule (D30), labels and intensit |
| `src/evaluate.py` | keep |  | Used; The one evaluation path every model goes through. |
| `src/experiments.py` | rename | `src/run_registry.py` | Used; Run log, frozen model settings and the once-only dev guard shared by every model script. |
| `src/fusion.py` | keep |  | Used; M3 late fusion: emotion score variants, per-impression min-max and the checks of D54. |
| `src/lexicon.py` | keep |  | Used; NRC EmoLex and NRC VAD loaders plus the tokeniser used for lexicon matching. |
| `src/logic_check.py` | keep |  | Used; Proof that a presentation-only edit changed no logic: compare Python files as normalised syntax trees. |
| `src/ltr.py` | rename | `src/lightgbm_ranker.py` | Used; M4 and M5: LightGBM ranker features, the time-safe popularity feature, negative sampling and training (D63-D67 |
| `src/metrics.py` | keep |  | Used; Per-impression ranking metrics (official MIND definitions). |
| `src/pipeline.py` | keep |  | Used; Make-style step runner: a step reruns only when its inputs, code, arguments or outputs changed. |
| `src/provenance.py` | keep |  | Used; library module |
| `src/recency.py` | keep |  | Used; True recency (D39): emotional context from the user's clicks in EARLIER impressions of the log, strictly befor |
| `src/results_charts.py` | keep |  | Used; The results charts for the supervisor: built from saved tables only, captions computed from the same numbers.  |
| `src/results_tables.py` | keep |  | Used; Summary tables built from saved per-impression results. |
| `src/sbert.py` | keep |  | Used; M2 Sentence-BERT: resumable corpus encoding and history-profile cosine scores. Encoder pretrained, never fitte |
| `src/semeval.py` | keep |  | Used; External validation on SemEval-2007 Task 14 Affective Text (Strapparava & Mihalcea, 2007): loading, gold class |
| `src/snapshot.py` | keep |  | Used; Results snapshot: copy tables and emotion outputs, then prove a later rebuild reproduces them (floats within 1 |
| `src/stats.py` | keep |  | Used; Paired model comparison on per-impression metrics, and Holm correction. |
| `src/story_charts.py` | keep |  | Used; The eight story charts of base-v2, in one academic style. Static PNGs: every chart has a title, labelled axes  |
| `src/summary.py` | rename | `src/terminal_summary.py` | Used; Terminal summary of the project, built only from saved result tables. Impression data, including dev, is never |
| `src/validation.py` | rename | `src/label_thresholds.py` | Used; Threshold views of the emotion labels on the full corpus (item-level, no labels, no clicks). |
| `src/where_it_helps.py` | keep |  | Used; RQ5, descriptive: where M3 differs from M2 on dev (D56). Nothing here selects or changes a model. |
| `tests/__init__.py` | keep |  | Tests the module of the same name |
| `tests/conftest.py` | keep |  | Tests the module of the same name |
| `tests/test_agreement.py` | keep |  | Tests the module of the same name |
| `tests/test_click_signal.py` | keep |  | Tests the module of the same name |
| `tests/test_congruence.py` | keep |  | Tests the module of the same name |
| `tests/test_data.py` | keep |  | Tests the module of the same name |
| `tests/test_diagnostics.py` | keep |  | Tests the module of the same name |
| `tests/test_emotion.py` | keep |  | Tests the module of the same name |
| `tests/test_emotion_features.py` | keep |  | Tests the module of the same name |
| `tests/test_emotion_spec.py` | keep |  | Tests the module of the same name |
| `tests/test_emotion_v2.py` | keep |  | Tests the module of the same name |
| `tests/test_evaluate.py` | keep |  | Tests the module of the same name |
| `tests/test_fusion.py` | keep |  | Tests the module of the same name |
| `tests/test_lexicon.py` | keep |  | Tests the module of the same name |
| `tests/test_logic_check.py` | keep |  | Tests the module of the same name |
| `tests/test_ltr.py` | rename | `tests/test_lightgbm_ranker.py` | Tests the module of the same name |
| `tests/test_metrics.py` | keep |  | Tests the module of the same name |
| `tests/test_pipeline.py` | keep |  | Tests the module of the same name |
| `tests/test_recency.py` | keep |  | Tests the module of the same name |
| `tests/test_run_base.py` | rename | `tests/test_run_pipeline.py` | Tests the module of the same name |
| `tests/test_sbert.py` | keep |  | Tests the module of the same name |
| `tests/test_semeval.py` | keep |  | Tests the module of the same name |
| `tests/test_snapshot.py` | keep |  | Tests the module of the same name |
| `tests/test_stats_experiments_baselines.py` | rename | `tests/test_stats_run_registry_baselines.py` | Tests the module of the same name |
| `tests/test_story_charts.py` | keep |  | Tests the module of the same name |
| `tests/test_summary.py` | rename | `tests/test_terminal_summary.py` | Tests the module of the same name |
| `tests/test_validation.py` | rename | `tests/test_label_thresholds.py` | Tests the module of the same name |
| `tests/test_where_it_helps.py` | keep |  | Tests the module of the same name |
| `scripts/check_logic_unchanged.py` | new | | Proof that tidy-up changed no logic |
| `src/logic_check.py` | new | | Logic of that proof |
| `tests/test_logic_check.py` | new | | Tests for it |
