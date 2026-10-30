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
python scripts/run_base.py --dry-run     # what would run and why; runs nothing
python scripts/run_base.py               # run what is out of date, then print the summary
python scripts/show_results.py           # only the summary, from saved tables
```
Steps run in number order and stop at the first failure. A step reruns only if its data inputs, its code (the script and the `src/`
modules it imports), the config constants it uses, its arguments or its outputs changed (content hashes; state in
`data/processed/.pipeline/`).
- `--rebuild` reruns every step except GPU scoring. Dev runs are never in the pipeline.
- GPU steps never start on their own while their outputs exist; run them with `--force-step 20`, `21` or `29`.
- `--force-step 30` reruns one step; `--from 30` reruns 30 and everything after; `--only 30 31` limits the run; `--force` reruns all.
- `--adopt` (once, e.g. after restoring results) records existing outputs as current without rerunning them.
- `--skip-tests`, `--skip-official-check`.

## Run order
| Step | Script | Part | Output |
|---|---|---|---|
| 10 | `10_data_download.py` | - | `data/raw` (create-only) |
| 11 | `11_data_prepare.py` | all | news, behaviors, impressions parquet |
| 12 | `12_data_harness_check.py` | valid | fake-model checks of the harness (dev only with `--parts dev`) |
| 13 | `13_data_eda.py` | fit | EDA figures |
| 14 | `14_data_semeval_download.py` | - | SemEval-2007 Task 14 headlines and gold (create-only, needs `curl`) |
| 20 | `20_emotion_label_combined.py` | items | combined-text GoEmotions baseline (GPU, resumable) |
| 21 | `21_emotion_score_fields.py` | items | title, abstract, weighted GoEmotions, go_* (GPU, resumable) |
| 22 | `22_emotion_lexicon_features.py` | items | nrc_*, vad_*, go_vad_* |
| 23 | `23_emotion_merge_features.py` | items | `news_emotion_features.parquet` |
| 24 | `24_emotion_diagnostics.py` | items | distributions, cross-source correlations |
| 25 | `25_emotion_thresholds.py` | items | neutral share per pre-declared v1 threshold |
| 26 | `26_emotion_v2_grid.py` | fit + valid | title weight and threshold grid, selection rule D30 |
| 27 | `27_emotion_v2_features.py` | items | v2 raw scores, labels, features, 60 examples |
| 28 | `28_emotion_label_variants.py` | items | stance-excluded labels v2b, v2c |
| 29 | `29_emotion_semeval_score.py` | SemEval | GoEmotions on 1,250 headlines (small GPU job, resumable) |
| 30 | `30_analysis_click_signal.py` | fit | item emotion vs clicks |
| 31 | `31_analysis_congruence.py` | fit | history-profile congruence with controls |
| 32 | `32_analysis_spec_v1_replication.py` | valid | evidence behind `config/emotion_spec_v1.json` |
| 33 | `33_analysis_spec_v2_replication.py` | fit + valid | evidence behind `config/emotion_spec_v2.json` |
| 34 | `34_analysis_context_windows.py` | fit + valid | short-term vs long-term context, history column |
| 35 | `35_analysis_true_recency.py` | fit + valid | true recency from earlier log clicks, analysis only |
| 36 | `36_analysis_percentile_robustness.py` | fit + valid | train-only percentile check |
| 37 | `37_analysis_story_charts.py` | saved tables | the 8 charts in `results/figures/story/` |
| 38 | `38_analysis_semeval_validation.py` | SemEval | trial selects the rule, test reports correlations, kappa, spec verdict |
| 40 | `40_model_m0_popularity.py --tune` | valid | M0 grid, frozen settings |
| 41 | `41_model_m1_tfidf.py --tune` | valid | M1 grid, frozen settings |
| 42 | `42_model_compare_baselines.py --part valid` | valid | paired M1 vs M0 |
| - | `show_results.py` | saved tables | terminal summary (last step) |

**Final test (never run by the pipeline, once per model, after its settings are frozen and recorded in `docs/decisions.md`):**
`python scripts/40_model_m0_popularity.py --final-dev`, `python scripts/41_model_m1_tfidf.py --final-dev`, then
`python scripts/42_model_compare_baselines.py --part dev`. A second dev run needs `--rerun-dev "<reason>"`.

**Optional:** `check_goemotions.py` (in-domain GoEmotions test, needs internet), `21_emotion_score_fields.py --measure / --smoke / --report`,
`20_emotion_label_combined.py --smoke`, `snapshot_results.py --take / --check` (prove a change did not alter results).

## Rules that protect the results
- `fit` trains, `valid` (last day of train) tunes, `dev` is reported once. Never tune on dev.
- Every model returns `imp_key, news_id, score` for all candidates and is scored through `src.evaluate`.
- `position` in the impression table is log order. Never use it as a feature.
- Click statistics only use earlier parts: valid is scored with fit statistics, dev with fit + valid.
- Raw 28-label emotion scores are saved, so changing the 28 to 6 mapping never needs a re-run of the model.
- Labelled chunks are tied to model, max length, chunk size and text hash; changed settings are refused.
