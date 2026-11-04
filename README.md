# Beyond Keywords: Emotion-Aware Ranking

Does emotional context improve click ranking beyond content matching? An 8-model ablation (M0 to M7) on MIND-small.
MSc Data Analytics Research Practicum, NCI. Demo platform: [AURA](https://github.com/smit-savaliya-dev/aura-ai-powered-emotion-aware-ad-recommender-platform).

**New here? Read [docs/START_HERE.md](docs/START_HERE.md).**

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Put MIND-small in `data/raw/train/` and `data/raw/dev/` (`news.tsv`, `behaviors.tsv`), or let step 10 download it.
NRC lexicons (research-only licence, git-ignored, never commit). Unzip so these files exist:
- `data/external/nrc/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` ([EmoLex](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm))
- `data/external/nrc/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt` ([NRC VAD](https://saifmohammad.com/WebPages/nrc-vad.html); v1 also works, set `VAD_PATH`)

External validation data: SemEval-2007 Task 14 Affective Text (Strapparava & Mihalcea, 2007), fetched by step 14 into `data/external/semeval2007/` (git-ignored).

## Run everything
```bash
python scripts/run_pipeline.py --dry-run     # what would run and why; runs nothing
python scripts/run_pipeline.py               # run what is out of date, then print the summary
python scripts/show_results.py           # only the summary, from saved tables
```
Steps run in number order and stop at the first failure. A step reruns only if its data inputs, its code (the script and the `src/`
modules it imports), the config constants it uses, its arguments or its outputs changed (content hashes; state in
`data/processed/.pipeline/`).
- `--rebuild` reruns every step except GPU scoring. Dev runs are never in the pipeline.
- GPU steps never start on their own while their outputs exist; run them with `--force-step 20`, `21`, `29` or `43`.
- `--force-step 30` reruns one step; `--from 30` reruns 30 and everything after; `--only 30 31` limits the run; `--force` reruns all.
- `--adopt` (once, e.g. after restoring results) records existing outputs as current without rerunning them.
- `--skip-tests`, `--skip-official-check`.

## Run order
| Step | Script | Part | Output |
|---|---|---|---|
| 10 | `01_01_download_mind.py` | - | `data/raw` (create-only) |
| 11 | `01_02_prepare_mind.py` | all | news, behaviors, impressions parquet |
| 12 | `01_03_harness_check.py` | valid | fake-model checks of the harness (dev only with `--parts dev`) |
| 13 | `01_04_eda.py` | fit | EDA figures |
| 14 | `01_05_download_semeval.py` | - | SemEval-2007 Task 14 headlines and gold (create-only, needs `curl`) |
| 20 | `02_01_label_combined.py` | items | combined-text GoEmotions baseline (GPU, resumable) |
| 21 | `02_02_score_fields.py` | items | title, abstract, weighted GoEmotions, go_* (GPU, resumable) |
| 22 | `02_03_lexicon_features.py` | items | nrc_*, vad_*, go_vad_* |
| 23 | `02_04_merge_features.py` | items | `news_emotion_features.parquet` |
| 24 | `02_05_diagnostics.py` | items | distributions, cross-source correlations |
| 25 | `02_06_thresholds.py` | items | neutral share per pre-declared v1 threshold |
| 26 | `02_07_v2_grid.py` | fit + valid | title weight and threshold grid, selection rule D30 |
| 27 | `02_08_v2_features.py` | items | v2 raw scores, labels, features, 60 examples |
| 28 | `02_09_label_variants.py` | items | stance-excluded labels v2b, v2c |
| 29 | `02_10_semeval_score.py` | SemEval | GoEmotions on 1,250 headlines (small GPU job, resumable) |
| 30 | `03_01_click_signal.py` | fit | item emotion vs clicks |
| 31 | `03_02_congruence.py` | fit | history-profile congruence with controls |
| 32 | `03_03_spec_v1_replication.py` | valid | evidence behind `config/emotion_spec_v1.json` |
| 33 | `03_04_spec_v2_replication.py` | fit + valid | evidence behind `config/emotion_spec_v2.json` |
| 34 | `03_05_context_windows.py` | fit + valid | short-term vs long-term context, history column |
| 35 | `03_06_true_recency.py` | fit + valid | true recency from earlier log clicks, analysis only |
| 36 | `03_07_percentile_robustness.py` | fit + valid | train-only percentile check |
| 37 | `03_08_story_charts.py` | saved tables | the 8 charts in `results/figures/story/` |
| 38 | `03_09_semeval_validation.py` | SemEval | trial selects the rule, test reports correlations, kappa, spec verdict |
| 40 | `04_01_m0_popularity.py --tune` | valid | M0 grid, frozen settings |
| 41 | `04_02_m1_tfidf.py --tune` | valid | M1 grid, frozen settings |
| 42 | `04_03_compare_baselines.py --part valid` | valid | paired M1 vs M0 |
| 43 | `04_04_m2_encode.py` | items | all-mpnet-base-v2 embeddings, title and title+abstract (GPU, resumable) |
| 44 | `04_05_m2_sbert.py --tune` | valid | M2 grid, frozen settings |
| 45 | `04_06_m3_fusion.py --tune` | valid | M3 grid (variant, alpha), checks, frozen settings |
| 46 | `04_07_compare_m2m3.py --part valid` | valid | paired M2 vs M1 and M3 vs M2, user-level check |
| 48 | `04_09_ltr_features.py` | fit + valid | M4 and M5 feature tables, time-safe popularity, checks against M0-M2 |
| 49 | `04_10_m4m5_lightgbm.py --tune` | valid | M4, M5, M5-all grids, seeds, shuffled control, frozen settings |
| 50 | `05_01_m5_controls.py --tune` | valid | item-identity controls for M5 (D69) |
| 51 | `05_02_compare_m4m5.py --part valid` | valid | paired M5 vs M4, M5 vs M3, M4 vs M2, M5-all vs M5, fairness check, controls, fresh-item split |
| 52 | `05_03_feature_importance.py` | valid | gain and mean absolute contribution for M5 |
| 54 | `05_05_results_charts.py` | saved tables | the 6 charts in `results/figures/results/` |
| - | `show_results.py` | saved tables | terminal summary (last step) |

**Final test (never run by the pipeline, once per model, after its settings are frozen and recorded in `docs/decisions.md`):**
`python scripts/04_01_m0_popularity.py --final-dev`, `python scripts/04_02_m1_tfidf.py --final-dev`, then
`python scripts/04_03_compare_baselines.py --part dev`. For M2 and M3: `04_05_m2_sbert.py --final-dev`, `04_06_m3_fusion.py --final-dev`,
`04_07_compare_m2m3.py --part dev`, then `04_08_where_it_helps.py` (descriptive, reads dev, so never a pipeline step).
For M4 and M5: `04_10_m4m5_lightgbm.py --final-dev --model M4_lightgbm`, the same with `--model M5_lightgbm_emotion`, `05_01_m5_controls.py --final-dev --model M5_permuted_items`, the same with `--model M5_congruence_only`, `05_02_compare_m4m5.py --part dev`,
`04_08_where_it_helps.py --base M4_lightgbm --model M5_lightgbm_emotion --out m5_where_it_helps --compact`, then `05_04_model_ladder.py`, which reads the saved dev
results for the ladder chart. M5-all never runs on dev (D65).
A second dev run needs `--rerun-dev "<reason>"`.

**Optional:** `check_goemotions.py` (in-domain GoEmotions test, needs internet), `check_sbert_encoder.py` (our M2 embeddings vs the sentence-transformers package), `04_10_m4m5_lightgbm.py --tune --resume` (reuse finished runs after an interruption), `02_02_score_fields.py --measure / --smoke / --report`,
`02_01_label_combined.py --smoke`, `snapshot_results.py --take / --check` (prove a change did not alter results).

## Rules that protect the results
- `fit` trains, `valid` (last day of train) tunes, `dev` is reported once. Never tune on dev.
- Every model returns `imp_key, news_id, score` for all candidates and is scored through `src.evaluate`.
- `position` in the impression table is log order. Never use it as a feature.
- Click statistics only use earlier parts: valid is scored with fit statistics, dev with fit + valid.
- Raw 28-label emotion scores are saved, so changing the 28 to 6 mapping never needs a re-run of the model.
- Labelled chunks are tied to model, max length, chunk size and text hash; changed settings are refused.
