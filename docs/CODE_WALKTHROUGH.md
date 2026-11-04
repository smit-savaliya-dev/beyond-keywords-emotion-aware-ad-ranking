# Code walkthrough for the reviewer

## The project in five lines
1. Question: does the emotional context inferred from a user's recent reading help rank news (standing in for ads) beyond matching content?
2. Data: MIND-small. `fit` (train without its last day) trains, `valid` (the last train day) chooses settings, `dev` is the final test, used once per model.
3. Emotion layer: GoEmotions scores on title and abstract (soft probabilities), NRC lexicon features as a second source, checked against human ratings on SemEval-2007 headlines.
4. Models M0 to M5 (popularity, TF-IDF, Sentence-BERT, + emotion, LightGBM, + emotion) are built, tuned on valid and tested once on dev. M6 and M7 (NRMS) are not built.
5. Result: emotion gives a small gain on top of Sentence-BERT (M3 - M2 = +0.0029 AUC) and no gain on top of LightGBM (M5 - M4 = -0.0030); the valid gain for M5 was mostly item identity (D69 to D71).

## Folder map
| Folder | Contents |
|---|---|
| `src/` | All logic (32 modules). `config.py` holds every constant; `pipeline.py` is the step runner. |
| `scripts/` | Thin scripts, named `<stage>_<number>_<what>.py` (see the run order below), plus `run_pipeline.py` and `show_results.py`. |
| `tests/` | One test file per module, with small fakes (no internet, GPU or real data). |
| `config/` | Frozen emotion specifications `emotion_spec_v1.json` to `v4.json` (hash-locked by a test). |
| `docs/` | `decisions.md` (every choice and its evidence), `open_questions.md`, `progress.md`, `methodology_notes.md`, `figures_guide.md`, `reports/`. |
| `results/tables/` | Every result table. `summary.csv`: model metrics per part. `runs_log.csv`: every run. |
| `results/figures/` | `results/` (the 6 charts below), `story/` (8 data and emotion charts), `diagnostics/` (EDA and feature importance). |
| `results/frozen/` | Settings of each model, frozen on valid before its dev run. |
| `results/archive/` | Superseded or cancelled material and the proof records of the clean-ups. |
| `data/` | Raw data, processed tables and caches. Git-ignored. |

## Run order (one line per script)
Step ids are the script names without `.py`. Scripts marked "by hand" are not pipeline steps because they read dev results.

**01 data**
- `01_01_download_mind` downloads MIND-small. `01_02_prepare_mind` builds the news, behaviour and impression tables and checks them.
- `01_03_harness_check` proves the evaluation harness with fake models. `01_04_eda` draws the EDA figures. `01_05_download_semeval` fetches SemEval-2007.

**02 emotion**
- `02_01_label_combined` GoEmotions on title + abstract as one text (baseline, GPU). `02_02_score_fields` title and abstract scored separately and weighted (GPU).
- `02_03_lexicon_features` NRC EmoLex and VAD features. `02_04_merge_features` merges all item features. `02_05_diagnostics` distributions and cross-source correlations.
- `02_06_thresholds` neutral share per threshold. `02_07_v2_grid` title-weight and threshold grid. `02_08_v2_features` the v2 features and labels.
- `02_09_label_variants` stance-excluded labels. `02_10_semeval_score` scores the SemEval headlines (GPU).

**03 analysis (fit and valid only)**
- `03_01_click_signal` item emotion against clicks. `03_02_congruence` match with the history profile, with controls.
- `03_03_spec_v1_replication`, `03_04_spec_v2_replication` the replication rule behind the emotion specs. `03_05_context_windows` short against long history.
- `03_06_true_recency` earlier log clicks (analysis only). `03_07_percentile_robustness` train-only percentiles. `03_08_story_charts` the 8 story charts.
- `03_09_semeval_validation` selects the labelling rule on SemEval trial and reports on test.

**04 models**
- `04_01_m0_popularity`, `04_02_m1_tfidf` baselines. `04_03_compare_baselines` paired M1 against M0.
- `04_04_m2_encode` embeddings (GPU). `04_05_m2_sbert` M2. `04_06_m3_fusion` M3. `04_07_compare_m2m3` paired comparisons.
- `04_08_where_it_helps` RQ5 breakdowns on dev (by hand). `04_09_ltr_features` M4 and M5 feature tables. `04_10_m4m5_lightgbm` M4, M5 and M5-all.
- Each model script has `--tune` (valid, in the pipeline) and `--final-dev` (dev, once, by hand).

**05 results**
- `05_01_m5_controls` item-identity controls. `05_02_compare_m4m5` paired comparisons. `05_03_feature_importance` what M5 uses.
- `05_04_model_ladder` dev AUC per model (by hand). `05_05_results_charts` the 6 results charts.
- `show_results.py` prints the summary from saved tables.

## Read these first
1. `docs/reports/m4m5_report.md`: the latest findings in plain language.
2. `src/config.py`: every number the project depends on, with short comments.
3. `src/evaluate.py` and `src/metrics.py`: the one evaluation path and the metric definitions every model goes through.
4. `src/congruence.py`: how the emotional context of a user is built from the history column only.
5. `src/lightgbm_ranker.py`: M4 and M5, and the leakage rule for click statistics (D64).
6. `scripts/04_06_m3_fusion.py`: the pattern of a model script: tune on valid, freeze, dev once.
7. `src/run_registry.py` and `src/pipeline.py`: the frozen settings, the dev-once guard and the incremental runner.
8. `docs/decisions.md`: why each choice was made, with the data part it came from.

## Where the key results and charts are
| What | Where |
|---|---|
| Model ladder, dev AUC with 95% CI (M0 to M5) | `results/figures/results/1_model_ladder.png`, `results/tables/model_ladder.csv` |
| Gain from emotion per metric: M3 - M2 and M5 - M4, valid and dev, Holm | `results/figures/results/2_m3_gain_by_metric.png`, `m2m3_comparison.csv`, `m4m5_comparison.csv` |
| Alpha curve of the fusion (valid) | `results/figures/results/3_alpha_ablation.png`, `m3_fusion_grid_valid.csv` |
| Where emotion helps M3: by category and by history length (descriptive) | `4_m3_gain_by_category.png`, `5_m3_gain_by_history.png`, `m3_where_it_helps_dev.csv` |
| Emotion scores against human ratings (SemEval-2007) | `6_emotion_validation.png`, `semeval2007_validation.csv` |
| Item-identity controls and the fresh-item split behind the M5 result | `m5_controls_valid.csv`, `m5_fresh_items.csv`, `docs/decisions.md` D69 to D71 |
| Click signal and congruence analyses | `emotion_signal_v2.csv`, `congruence_signal_fit.csv`, `context_windows_signal.csv`, `results/figures/story/` |
| All model metrics, every part | `results/tables/summary.csv` (`runs_log.csv` for every run) |
| Every chart explained | `docs/figures_guide.md` |

## How to run
```bash
python scripts/run_pipeline.py     # runs what is out of date (or --dry-run to see the plan), then prints the summary
python scripts/show_results.py     # the summary alone, from saved tables
python -m pytest -q                # all tests
```
`--rebuild` reruns every step except GPU scoring; dev runs are never part of the pipeline. Setup, data locations and the options are in the README.

## Data-discipline rules
- **fit / valid / dev.** fit trains and is used for analyses, valid chooses settings and replicates findings, dev is the final test. Each model runs on dev once
  (`--final-dev`, refused a second time without a written reason), after its settings are frozen in `results/frozen/` and recorded in `docs/decisions.md`.
- **Selection rules are pre-declared.** Rules and thresholds are written in `docs/decisions.md` and committed before the numbers are computed.
- **Leakage rule.** Features are item-level text only, or built from data strictly earlier than what is scored: the user profile uses the history column only, and click
  statistics use days strictly before the row's day (D64). `position` is log order and is never a feature.
- **Dev is never used to choose anything.** Analysis scripts never load dev. The few scripts that read saved dev results (`04_08`, `05_04`) are run by hand after the dev runs and select nothing.
- **Reproducibility.** Seeds are fixed, boosters are checked by hash, and `scripts/snapshot_results.py` proves a rebuild reproduces every table.
- **Wording.** "Emotional context inferred from the user's recent reading", never "the user's emotional state".
