# Beyond Keywords: Emotion-Aware Ranking

8-model ablation (M0 to M7) testing whether emotional congruence improves click ranking on MIND-small.
MSc Data Analytics Research Practicum, NCI. Demo platform: [AURA](https://github.com/smit-savaliya-dev/aura-ai-powered-emotion-aware-ad-recommender-platform).

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Put MIND-small in `data/raw/train/` and `data/raw/dev/` (`news.tsv`, `behaviors.tsv`), or let step 01 download it.

## 1. Whole pipeline (one command)
```bash
python scripts/run_base.py --dry-run     # what would run and why; runs nothing
python scripts/run_base.py               # tests -> data -> harness -> EDA -> emotion layer -> signal analyses -> validation
```
Runs every step in order and stops at the first failure. A step reruns only if its data inputs, its code (the script and the
`src/` modules it imports), its arguments or its outputs changed since its last successful run (content hashes; state in
`data/processed/.pipeline/`). Download and `annotation/manual_sample.csv` are create-only; label validation waits until the
sheet is filled. GPU steps (04, 10) also resume from their chunk folders.
- `--force-step 14` reruns one step; `--from 11` reruns 11 and everything after; `--only 13 14` limits the run; `--force` reruns all.
- `--adopt` (once, e.g. after restoring results) records existing outputs as current without rerunning them.
- `--skip-tests`, `--skip-official-check` as before. Optional, not in the pipeline: `08_goemotions_check.py`, `09_check_lexicons.py`,
  `10_score_fields.py --measure/--smoke`.

## 2. Run order (what the pipeline runs, step by step)
NRC lexicons (research-only licence, git-ignored, never commit). Unzip so these files exist:
- `data/external/nrc/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` ([EmoLex](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm))
- `data/external/nrc/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt` ([NRC VAD](https://saifmohammad.com/WebPages/nrc-vad.html); v1 also works, set `VAD_PATH`)

| Step | Command | Part | Output |
|---|---|---|---|
| 01 | `python scripts/01_download_mind.py` | - | data/raw (create-only) |
| 02 | `python scripts/02_prepare_mind.py` | all | news / behaviors / impressions parquet |
| 03 | `python scripts/03_harness_sanity.py` | valid | fake-model checks of the harness (dev only with `--parts dev`) |
| 07 | `python scripts/07_eda.py` | fit | EDA figures |
| 04 | `python scripts/04_label_emotions.py` | items | combined-text GoEmotions baseline (GPU, resumable) |
| 05 | `python scripts/05_make_annotation_sample.py` | items | 180-item hand-label sheet (create-only) |
| 10 | `python scripts/10_score_fields.py` | items | title / abstract / weighted GoEmotions, go_* (GPU, resumable) |
| 11 | `python scripts/11_lexicon_features.py` | items | nrc_*, vad_*, go_vad_* |
| 12 | `python scripts/12_merge_emotion_features.py` | items | `news_emotion_features.parquet` |
| 13 | `python scripts/13_emotion_diagnostics.py` | items | distributions, cross-source correlations |
| 14 | `python scripts/14_emotion_signal.py` | fit | item emotion vs clicks (t-test, user level, Holm) |
| 15 | `python scripts/15_congruence_signal.py` | fit | history-profile congruence, personal and other-user controls |
| 20 | `python scripts/20_m0_popularity.py --tune` | valid | M0 grid, frozen settings |
| 21 | `python scripts/21_m1_tfidf.py --tune` | valid | M1 grid, frozen settings |
| 22 | `python scripts/22_compare_baselines.py --part valid` | valid | paired M1 vs M0 |
| 16 | `python scripts/16_emotion_thresholds.py` | items | neutral share per pre-declared threshold |
| 17 | `python scripts/17_emotion_spec_valid.py` | valid | replication behind `config/emotion_spec_v1.json` |
| 06 | `python scripts/06_validate_emotion_labels.py` | labels | per-source kappa, thresholds, intra-rater, spec verdict (waits for labels) |

**Final test (never run by the pipeline, once per model, after its settings are frozen and recorded in
`docs/decisions.md`):** `python scripts/20_m0_popularity.py --final-dev`, `python scripts/21_m1_tfidf.py --final-dev`,
then `python scripts/22_compare_baselines.py --part dev`. A second dev run needs `--rerun-dev "<reason>"`.

**Hand labels (you, not the pipeline):** `python scripts/label_tool.py` (guidelines: `docs/labelling_guidelines.md`),
then `python scripts/label_tool.py --make-relabel` and, a day later, `python scripts/label_tool.py --relabel`.

**Optional checks:** `08_goemotions_check.py` (in-domain GoEmotions test), `09_check_lexicons.py`,
`10_score_fields.py --measure / --smoke / --report`, `04_label_emotions.py --smoke`.
Colab: upload `src/`, `scripts/` and `data/processed/news.parquet`, run step 04, download `data/processed/emotion_chunks/`
and `news_emotion*.parquet` back.

## 3. Where things are documented
- `docs/decisions.md` every non-trivial choice with evidence; `docs/methodology_notes.md` the main justifications;
  `docs/progress.md` phase status; `docs/open_questions.md` items for the student; `results/tables/runs_log.csv` every run.

## Rules that protect the results
- `fit` trains, `valid` (last day of train) tunes, `dev` is reported once. Never tune on dev.
- Every model returns `imp_key, news_id, score` for all candidates and is scored with `src.evaluate.evaluate_model`.
- `position` in the impression table is log order. Never use it as a feature.
- Click statistics only use earlier parts: valid is scored with fit statistics, dev with fit + valid.
- Raw 28-label emotion scores are saved, so changing the 28 to 6 mapping never needs a re-run of the model.
- Labelled chunks are tied to model, max length, chunk size and text hash; changed settings are refused.
