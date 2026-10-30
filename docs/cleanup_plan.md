# Cleanup plan (base-v3, Step 4)

Rule: no result may change. A snapshot of 76 files was taken first (`results/archive/pre_cleanup_snapshot.json`), and after the
cleanup every pipeline step except GPU scoring and dev runs is re-run and compared, floats within 1e-9.

Summary of the audit: the code was already compact. Two dead functions and one redundant script are removed; the main work is
merging duplicated helpers, grouping the scripts, and moving superseded figures to the archive.

## Merged duplicates
- `TABLES = RESULTS / "tables"` was defined in 19 scripts; now once in config.
- `RowValues` (script) and `MaskedValues` (src): both lazy views, now both in `src/click_signal.py`.
- Holm-family logic was repeated in three places; now `holm_family`.
- Part loading was repeated in three scripts; now `load_part_context`.
- The window congruence helper was in two scripts; now `window_congruence`.
- Chart colours were in three modules; now only `src/story_charts.py`.

## Not merged, on purpose
- Spec replications v1 and v2 use different feature tables and samples, and v1's output is pinned by a hash.
- `tidy` differs between the two context scripts.
- v1 tables stay in `results/tables/`: spec v1 records their paths.

## Script renames (old -> new)
| Old | New |
|---|---|
| `01_download_mind.py` | `10_data_download.py` |
| `02_prepare_mind.py` | `11_data_prepare.py` |
| `03_harness_sanity.py` | `12_data_harness_check.py` |
| `07_eda.py` | `13_data_eda.py` |
| `04_label_emotions.py` | `20_emotion_label_combined.py` |
| `10_score_fields.py` | `21_emotion_score_fields.py` |
| `11_lexicon_features.py` | `22_emotion_lexicon_features.py` |
| `12_merge_emotion_features.py` | `23_emotion_merge_features.py` |
| `13_emotion_diagnostics.py` | `24_emotion_diagnostics.py` |
| `16_emotion_thresholds.py` | `25_emotion_thresholds.py` |
| `30_emotion_v2_grid.py` | `26_emotion_v2_grid.py` |
| `31_emotion_v2_features.py` | `27_emotion_v2_features.py` |
| `35_label_variants.py` | `28_emotion_label_variants.py` |
| `14_emotion_signal.py` | `30_analysis_click_signal.py` |
| `15_congruence_signal.py` | `31_analysis_congruence.py` |
| `17_emotion_spec_valid.py` | `32_analysis_spec_v1_replication.py` |
| `32_emotion_spec_v2.py` | `33_analysis_spec_v2_replication.py` |
| `33_context_windows.py` | `34_analysis_context_windows.py` |
| `36_true_recency.py` | `35_analysis_true_recency.py` |
| `37_percentile_robustness.py` | `36_analysis_percentile_robustness.py` |
| `34_story_charts.py` | `37_analysis_story_charts.py` |
| `06_validate_emotion_labels.py` | `38_analysis_label_validation.py` |
| `20_m0_popularity.py` | `40_model_m0_popularity.py` |
| `21_m1_tfidf.py` | `41_model_m1_tfidf.py` |
| `22_compare_baselines.py` | `42_model_compare_baselines.py` |
| `05_make_annotation_sample.py` | `14_data_annotation_sheet.py` |
| `08_goemotions_check.py` | `check_goemotions.py` |
| `09_check_lexicons.py` | deleted |

## Every file
| File | Action | Reason |
|---|---|---|
| `src/__init__.py` | keep | In use, no duplicates found |
| `src/annotation.py` | keep | In use, no duplicates found |
| `src/baselines.py` | keep | In use, no duplicates found |
| `src/click_signal.py` | merge | Gains ColumnValues, PartContext and load_part_context, so all analyses share one loader. Old figure code removed |
| `src/config.py` | keep | TABLES, DIAGNOSTICS and CONGRUENCE_ROLES now defined once |
| `src/congruence.py` | merge | Gains window_congruence, shared by the two context analyses |
| `src/data.py` | keep | Dead function load_processed removed; nothing else changes |
| `src/diagnostics.py` | merge | Chart colours now come from story_charts |
| `src/eda.py` | keep | Reads fit only (it loaded dev before); part sizes come from the saved summary |
| `src/emotion.py` | keep | Dead function corpus_mean removed |
| `src/emotion_features.py` | keep | In use, no duplicates found |
| `src/emotion_spec.py` | keep | In use, no duplicates found |
| `src/emotion_v2.py` | keep | Part loader moved to click_signal |
| `src/evaluate.py` | keep | evaluate_model stays: it is the contract for future models (scripts 40-41 call its two parts directly) |
| `src/experiments.py` | keep | In use, no duplicates found |
| `src/labelling.py` | keep | In use, no duplicates found |
| `src/lexicon.py` | keep | In use, no duplicates found |
| `src/metrics.py` | keep | In use, no duplicates found |
| `src/pipeline.py` | keep | Gains gpu and always step flags for --rebuild and the summary |
| `src/provenance.py` | keep | In use, no duplicates found |
| `src/recency.py` | keep | In use, no duplicates found |
| `src/snapshot.py` | keep | In use, no duplicates found |
| `src/stats.py` | keep | In use, no duplicates found |
| `src/story_charts.py` | keep | In use, no duplicates found |
| `src/validation.py` | merge | Chart colours now come from story_charts |
| `scripts/01_download_mind.py` | keep (rename) | Becomes scripts/10_data_download.py |
| `scripts/02_prepare_mind.py` | keep (rename) | Becomes scripts/11_data_prepare.py |
| `scripts/03_harness_sanity.py` | keep (rename) | Becomes scripts/12_data_harness_check.py |
| `scripts/04_label_emotions.py` | keep (rename) | Becomes scripts/20_emotion_label_combined.py |
| `scripts/05_make_annotation_sample.py` | keep (rename) | Becomes scripts/14_data_annotation_sheet.py |
| `scripts/06_validate_emotion_labels.py` | keep (rename) | Becomes scripts/38_analysis_label_validation.py |
| `scripts/07_eda.py` | keep (rename) | Becomes scripts/13_data_eda.py |
| `scripts/08_goemotions_check.py` | keep (rename) | Becomes scripts/check_goemotions.py |
| `scripts/09_check_lexicons.py` | delete | Not in the pipeline; the lexicon step already prints format and scale and fails on missing label words. Not referenced in decisions.md; recoverable from git |
| `scripts/10_score_fields.py` | keep (rename) | Becomes scripts/21_emotion_score_fields.py |
| `scripts/11_lexicon_features.py` | keep (rename) | Becomes scripts/22_emotion_lexicon_features.py |
| `scripts/12_merge_emotion_features.py` | keep (rename) | Becomes scripts/23_emotion_merge_features.py |
| `scripts/13_emotion_diagnostics.py` | keep (rename) | Becomes scripts/24_emotion_diagnostics.py |
| `scripts/14_emotion_signal.py` | keep (rename) | Becomes scripts/30_analysis_click_signal.py |
| `scripts/15_congruence_signal.py` | keep (rename) | Becomes scripts/31_analysis_congruence.py |
| `scripts/16_emotion_thresholds.py` | keep (rename) | Becomes scripts/25_emotion_thresholds.py |
| `scripts/17_emotion_spec_valid.py` | keep (rename) | Becomes scripts/32_analysis_spec_v1_replication.py |
| `scripts/20_m0_popularity.py` | keep (rename) | Becomes scripts/40_model_m0_popularity.py |
| `scripts/21_m1_tfidf.py` | keep (rename) | Becomes scripts/41_model_m1_tfidf.py |
| `scripts/22_compare_baselines.py` | keep (rename) | Becomes scripts/42_model_compare_baselines.py |
| `scripts/30_emotion_v2_grid.py` | keep (rename) | Becomes scripts/26_emotion_v2_grid.py |
| `scripts/31_emotion_v2_features.py` | keep (rename) | Becomes scripts/27_emotion_v2_features.py |
| `scripts/32_emotion_spec_v2.py` | keep (rename) | Becomes scripts/33_analysis_spec_v2_replication.py |
| `scripts/33_context_windows.py` | keep (rename) | Becomes scripts/34_analysis_context_windows.py |
| `scripts/34_story_charts.py` | keep (rename) | Becomes scripts/37_analysis_story_charts.py |
| `scripts/35_label_variants.py` | keep (rename) | Becomes scripts/28_emotion_label_variants.py |
| `scripts/36_true_recency.py` | keep (rename) | Becomes scripts/35_analysis_true_recency.py |
| `scripts/37_percentile_robustness.py` | keep (rename) | Becomes scripts/36_analysis_percentile_robustness.py |
| `scripts/_bootstrap.py` | keep | Tool |
| `scripts/label_tool.py` | keep | Tool |
| `scripts/run_base.py` | keep | Tool (run_base gains --rebuild) |
| `scripts/snapshot_results.py` | keep | Tool |
| `scripts/show_results.py` | new | Step 5 |
| `tests/__init__.py` | keep | Unchanged apart from renamed paths |
| `tests/conftest.py` | keep | Unchanged apart from renamed paths |
| `tests/test_annotation.py` | keep | Unchanged apart from renamed paths |
| `tests/test_click_signal.py` | keep | Tests of removed plot functions dropped |
| `tests/test_congruence.py` | keep | Tests of removed plot functions dropped |
| `tests/test_data.py` | keep | Unchanged apart from renamed paths |
| `tests/test_diagnostics.py` | keep | Unchanged apart from renamed paths |
| `tests/test_emotion.py` | keep | Unchanged apart from renamed paths |
| `tests/test_emotion_features.py` | keep | Unchanged apart from renamed paths |
| `tests/test_emotion_spec.py` | keep | Unchanged apart from renamed paths |
| `tests/test_emotion_v2.py` | keep | Import path of load_part_context updated |
| `tests/test_evaluate.py` | keep | Unchanged apart from renamed paths |
| `tests/test_labelling.py` | keep | Unchanged apart from renamed paths |
| `tests/test_lexicon.py` | keep | Unchanged apart from renamed paths |
| `tests/test_metrics.py` | keep | Unchanged apart from renamed paths |
| `tests/test_pipeline.py` | keep | Unchanged apart from renamed paths |
| `tests/test_recency.py` | keep | Unchanged apart from renamed paths |
| `tests/test_snapshot.py` | keep | Unchanged apart from renamed paths |
| `tests/test_stats_experiments_baselines.py` | keep | Unchanged apart from renamed paths |
| `tests/test_story_charts.py` | keep | Unchanged apart from renamed paths |
| `tests/test_validation.py` | keep | Script path updated |
| `config/emotion_spec_v1.json` | keep | Frozen spec, hash-locked. Its text keeps the script names of its freeze date; see the rename table |
| `config/emotion_spec_v2.json` | keep | Frozen spec, hash-locked. Its text keeps the script names of its freeze date; see the rename table |
| `results/tables/.gitkeep` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/baselines_comparison.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/baselines_sanity.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/congruence_signal_by_category_fit.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/congruence_signal_fit.csv` | keep | Kept in place: frozen spec v1 records this path and its hash cannot change |
| `results/tables/congruence_signal_fit_meta.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/context_windows_comparison.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/context_windows_meta.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/context_windows_signal.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_cross_source_correlations.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_cross_source_spearman.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_examples_v2.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_feature_distributions.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_label_variants_grid.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_label_variants_meta.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_low_mass.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_signal_by_category_fit.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_signal_fit.csv` | keep | Kept in place: frozen spec v1 records this path and its hash cannot change |
| `results/tables/emotion_signal_fit_meta.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_signal_v2.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_signal_valid.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_signal_variants_fit.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_spec_selection.csv` | keep | Kept in place: frozen spec v1 records this path and its hash cannot change |
| `results/tables/emotion_spec_selection_meta.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_spec_v2_selection.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_spec_v2_selection_meta.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_threshold_neutral_share.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_threshold_neutral_share_meta.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_title_vs_abstract.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_v2_choice.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_v2_grid.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_v2_meta.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_v2_weight_congruence.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/emotion_weighted_vs_combined.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/lexicon_coverage.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/lexicon_top_words.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/m0_popularity_dev_meta.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/m0_popularity_tuning_valid.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/m1_tfidf_dev_meta.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/m1_tfidf_tuning_valid.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/percentile_robustness.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/percentile_robustness_meta.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/runs_log.csv` | keep | Append-only run log |
| `results/tables/story_category_emotion.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/story_category_valence_arousal.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/story_emotion_mix.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/story_intensity.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/summary.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/token_lengths.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/true_recency_comparison.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/true_recency_meta.json` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/true_recency_residual_by_category.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/tables/true_recency_signal.csv` | keep | Generated by the pipeline; snapshot checked |
| `results/frozen/M0_popularity.json` | keep | Frozen model settings |
| `results/frozen/M1_tfidf.json` | keep | Frozen model settings |
| `results/scores/*.parquet` | keep | Model outputs, git-ignored; dev results are never regenerated |
| `results/per_impression/*.parquet` | keep | Model outputs, git-ignored; dev results are never regenerated |
| `results/figures/01_part_sizes.png` | keep (move) | Moves to results/figures/diagnostics/ |
| `results/figures/02_candidates_per_impression.png` | keep (move) | Moves to results/figures/diagnostics/ |
| `results/figures/03_clicks_per_impression.png` | keep (move) | Moves to results/figures/diagnostics/ |
| `results/figures/04_categories.png` | keep (move) | Moves to results/figures/diagnostics/ |
| `results/figures/05_history_length.png` | keep (move) | Moves to results/figures/diagnostics/ |
| `results/figures/06_integrity.png` | keep (move) | Moves to results/figures/diagnostics/ |
| `results/figures/congruence_signal_fit.png` | archive | Superseded by story chart 7; its plotting code is removed. Moves to results/archive/figures/ |
| `results/figures/eda_overview.png` | keep (move) | Moves to results/figures/diagnostics/ |
| `results/figures/eda_stats.json` | keep (move) | Moves to results/figures/diagnostics/ |
| `results/figures/emotion_cross_source_spearman.png` | keep (move) | Moves to results/figures/diagnostics/ |
| `results/figures/emotion_signal_fit.png` | archive | Superseded by story chart 6; its plotting code is removed. Moves to results/archive/figures/ |
| `results/figures/emotion_threshold_neutral_share.png` | archive | Superseded by story chart 2; its plotting code is removed. Moves to results/archive/figures/ |
| `results/figures/story/1_emotion_mix.png` | keep | One of the 8 key charts |
| `results/figures/story/2_neutral_vs_threshold.png` | keep | One of the 8 key charts |
| `results/figures/story/3_title_vs_abstract_intensity.png` | keep | One of the 8 key charts |
| `results/figures/story/4_emotion_by_category.png` | keep | One of the 8 key charts |
| `results/figures/story/5_valence_arousal_map.png` | keep | One of the 8 key charts |
| `results/figures/story/6_clicked_vs_not_effects.png` | keep | One of the 8 key charts |
| `results/figures/story/7_congruence_effects.png` | keep | One of the 8 key charts |
| `results/figures/story/8_short_vs_long_context.png` | keep | One of the 8 key charts |
| `results/reports/week5_report_draft.md` | keep (move) | Moves to docs/reports/ |
| `results/archive/pre_cleanup_snapshot.json` | new | Snapshot manifest; the copies are in data/snapshots/ (git-ignored) |
| `docs/base_v2_report.md` | keep (move) | Moves to docs/reports/ |
| `docs/decisions.md` | keep | Current document; script names updated where they are instructions. decisions.md and progress.md keep history |
| `docs/figures_guide.md` | keep | Current document; script names updated where they are instructions. decisions.md and progress.md keep history |
| `docs/labelling_guidelines.md` | keep | Current document; script names updated where they are instructions. decisions.md and progress.md keep history |
| `docs/methodology_notes.md` | keep | Current document; script names updated where they are instructions. decisions.md and progress.md keep history |
| `docs/open_questions.md` | keep | Current document; script names updated where they are instructions. decisions.md and progress.md keep history |
| `docs/progress.md` | keep | Current document; script names updated where they are instructions. decisions.md and progress.md keep history |
| `docs/week5_run_report.md` | keep (move) | Moves to docs/reports/ |
| `docs/START_HERE.md` | new | Entry point |
| `docs/cleanup_plan.md` | new | This file |
| `notebooks/.gitkeep` | delete | Empty folder, never used |
| `data/raw, data/processed, emotion_chunks*, annotation/` | keep | Never touched |
