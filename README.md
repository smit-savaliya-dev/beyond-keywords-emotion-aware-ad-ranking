# Beyond Keywords: Emotion-Aware Ranking

Does emotional context improve click ranking beyond content matching? An 8-model ablation (M0 to M7, each with and without emotion) on
MIND-small. M0 to M5 are done. MSc Data Analytics Research Practicum, NCI.
Demo platform: [AURA](https://github.com/smit-savaliya-dev/aura-ai-powered-emotion-aware-ad-recommender-platform).

**Reviewing this project? Read [docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md) first** (folder map, run order, what to read, where results live).
Day-to-day notes are in [docs/START_HERE.md](docs/START_HERE.md).

## The commands to know
```bash
python scripts/run_pipeline.py --dry-run     # what would run and why; runs nothing
python scripts/run_pipeline.py               # run what is out of date, then print the summary
python scripts/show_results.py               # only the summary, from saved tables
python -m pytest -q                          # the test suite (no internet, GPU or real lexicons needed)
```

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Put MIND-small in `data/raw/train/` and `data/raw/dev/` (`news.tsv`, `behaviors.tsv`), or let step `01_01_download_mind` download it.
NRC lexicons (research-only licence, git-ignored, never commit). Unzip so these files exist:
- `data/external/nrc/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` ([EmoLex](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm))
- `data/external/nrc/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt` ([NRC VAD](https://saifmohammad.com/WebPages/nrc-vad.html); v1 also works, set `VAD_PATH`)

External validation data: SemEval-2007 Task 14 Affective Text (Strapparava & Mihalcea, 2007), fetched by step `01_05_download_semeval`
into `data/external/semeval2007/` (git-ignored).

## How the pipeline works
Steps run in order and stop at the first failure. A step reruns only if its data inputs, its code (the script and the `src/` modules it
imports), the config constants it uses, its arguments or its outputs changed (content hashes; state in `data/processed/.pipeline/`).
A step id is the script name without `.py`; a stage (`04`) or a step number (`04_06`) also works where a step is expected.
- `--rebuild` reruns every step except GPU scoring. Dev runs are never in the pipeline.
- GPU steps never start on their own while their outputs exist; run them with `--force-step 02_01`, `02_02`, `02_10` or `04_04`.
- `--force-step 03_01` reruns one step; `--from 03_01` reruns it and everything after; `--only 03_01 03_02` limits the run; `--force` reruns all.
- `--adopt` (once, e.g. after restoring results) records existing outputs as current without rerunning them.
- `--skip-tests`, `--skip-official-check`.

## Run order
| Step | Script | Part | Output |
|---|---|---|---|
| 01_01 | `01_01_download_mind.py` | - | `data/raw` (create-only) |
| 01_02 | `01_02_prepare_mind.py` | all | news, behaviors, impressions parquet |
| 01_03 | `01_03_harness_check.py` | valid | fake-model checks of the harness |
| 01_04 | `01_04_eda.py` | fit | EDA figures |
| 01_05 | `01_05_download_semeval.py` | - | SemEval-2007 Task 14 headlines and gold (create-only, needs `curl`) |
| 02_01 | `02_01_label_combined.py` | items | combined-text GoEmotions baseline (GPU, resumable) |
| 02_02 | `02_02_score_fields.py` | items | title, abstract, weighted GoEmotions, go_* (GPU, resumable) |
| 02_03 | `02_03_lexicon_features.py` | items | nrc_*, vad_*, go_vad_* |
| 02_04 | `02_04_merge_features.py` | items | `news_emotion_features.parquet` |
| 02_05 | `02_05_diagnostics.py` | items | distributions, cross-source correlations |
| 02_06 | `02_06_thresholds.py` | items | neutral share per pre-declared v1 threshold |
| 02_07 | `02_07_v2_grid.py` | fit + valid | title weight and threshold grid, selection rule D30 |
| 02_08 | `02_08_v2_features.py` | items | v2 raw scores, labels, features, 60 examples |
| 02_09 | `02_09_label_variants.py` | items | stance-excluded labels v2b, v2c |
| 02_10 | `02_10_semeval_score.py` | SemEval | GoEmotions on 1,250 headlines (small GPU job, resumable) |
| 03_01 | `03_01_click_signal.py` | fit | item emotion vs clicks |
| 03_02 | `03_02_congruence.py` | fit | history-profile congruence with controls |
| 03_03 | `03_03_spec_v1_replication.py` | valid | evidence behind `config/emotion_spec_v1.json` |
| 03_04 | `03_04_spec_v2_replication.py` | fit + valid | evidence behind `config/emotion_spec_v2.json` |
| 03_05 | `03_05_context_windows.py` | fit + valid | short-term vs long-term context, history column |
| 03_06 | `03_06_true_recency.py` | fit + valid | true recency from earlier log clicks, analysis only |
| 03_07 | `03_07_percentile_robustness.py` | fit + valid | train-only percentile check |
| 03_08 | `03_08_story_charts.py` | saved tables | the 8 charts in `results/figures/story/` |
| 03_09 | `03_09_semeval_validation.py` | SemEval | trial selects the rule, test reports correlations, kappa, spec verdict |
| 04_01 | `04_01_m0_popularity.py --tune` | valid | M0 grid, frozen settings |
| 04_02 | `04_02_m1_tfidf.py --tune` | valid | M1 grid, frozen settings |
| 04_03 | `04_03_compare_baselines.py --part valid` | valid | paired M1 vs M0 |
| 04_04 | `04_04_m2_encode.py` | items | all-mpnet-base-v2 embeddings, title and title+abstract (GPU, resumable) |
| 04_05 | `04_05_m2_sbert.py --tune` | valid | M2 grid, frozen settings |
| 04_06 | `04_06_m3_fusion.py --tune` | valid | M3 grid (variant, alpha), checks, frozen settings |
| 04_07 | `04_07_compare_m2m3.py --part valid` | valid | paired M2 vs M1 and M3 vs M2, user-level check |
| 04_09 | `04_09_ltr_features.py` | fit + valid | M4 and M5 feature tables, time-safe popularity, checks against M0-M2 |
| 04_10 | `04_10_m4m5_lightgbm.py --tune` | valid | M4, M5, M5-all grids, seeds, shuffled control, frozen settings |
| 05_01 | `05_01_m5_controls.py --tune` | valid | item-identity controls for M5 (D69) |
| 05_02 | `05_02_compare_m4m5.py --part valid` | valid | paired M5 vs M4, M5 vs M3, M4 vs M2, M5-all vs M5, fairness check, controls, fresh-item split |
| 05_03 | `05_03_feature_importance.py` | valid | gain and mean absolute contribution for M5 |
| 05_05 | `05_05_results_charts.py` | saved tables | the 6 charts in `results/figures/results/` |
| 90 | `show_results.py` | saved tables | terminal summary (last step) |

`04_08_where_it_helps.py` and `05_04_model_ladder.py` are not pipeline steps: they read dev results and are run once by hand (see below).

**Final test (never run by the pipeline, once per model, after its settings are frozen and recorded in `docs/decisions.md`):**
`python scripts/04_01_m0_popularity.py --final-dev`, `python scripts/04_02_m1_tfidf.py --final-dev`, then
`python scripts/04_03_compare_baselines.py --part dev`. For M2 and M3: `04_05_m2_sbert.py --final-dev`, `04_06_m3_fusion.py --final-dev`,
`04_07_compare_m2m3.py --part dev`, then `04_08_where_it_helps.py` (descriptive, reads dev, so never a pipeline step).
For M4 and M5: `04_10_m4m5_lightgbm.py --final-dev --model M4_lightgbm`, the same with `--model M5_lightgbm_emotion`,
`05_01_m5_controls.py --final-dev --model M5_permuted_items`, the same with `--model M5_congruence_only`, `05_02_compare_m4m5.py --part dev`,
`04_08_where_it_helps.py --base M4_lightgbm --model M5_lightgbm_emotion --out m5_where_it_helps --compact`, then `05_04_model_ladder.py`, which reads
the saved dev results for the ladder chart. M5-all never runs on dev (D65).
A second dev run needs `--rerun-dev "<reason>"`.

**Optional:** `check_goemotions.py` (in-domain GoEmotions test, needs internet), `check_sbert_encoder.py` (our M2 embeddings vs the
sentence-transformers package), `04_10_m4m5_lightgbm.py --tune --resume` (reuse finished runs after an interruption),
`02_02_score_fields.py --measure / --smoke / --report`, `02_01_label_combined.py --smoke`.

**Proof tools:** `snapshot_results.py --take / --check [--tag NAME]` proves a change did not alter results (tables within 1e-9, scores,
boosters and figures byte for byte); `check_logic_unchanged.py --check` proves a presentation-only change left every file's syntax tree
unchanged. Both were used for the tidy-up of `docs/cleanup_plan_v2.md` (old script names are translated there).

## Rules that protect the results
- `fit` trains, `valid` (last day of train) tunes, `dev` is reported once. Never tune on dev.
- Every model returns `imp_key, news_id, score` for all candidates and is scored through `src.evaluate`.
- `position` in the impression table is log order. Never use it as a feature.
- Click statistics only use earlier parts: valid is scored with fit statistics, dev with fit + valid.
- Raw 28-label emotion scores are saved, so changing the 28 to 6 mapping never needs a re-run of the model.
- Labelled chunks are tied to model, max length, chunk size and text hash; changed settings are refused.
