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

## 2. Emotion labels
```bash
python scripts/04_label_emotions.py --smoke      # 12 headlines, writes nothing: confirms the model works
python scripts/04_label_emotions.py              # full run, resumable (use a GPU / Colab if possible)
python scripts/05_make_annotation_sample.py      # then fill your_label in annotation/manual_sample.csv BEFORE any model output
python scripts/06_validate_emotion_labels.py     # per source (combined, weighted, NRC 6-class): kappa + CI, acc, F1
python scripts/08_goemotions_check.py            # optional: in-domain check on GoEmotions test split
```
Colab: upload `src/`, `scripts/` and `data/processed/news.parquet`, run step 04, download `data/processed/emotion_chunks/` and `news_emotion*.parquet` back.

## 3. Weighted emotion features (title + abstract, GoEmotions + NRC lexicons)
NRC lexicons (research-only licence, git-ignored, never commit). Unzip so these files exist:
- `data/external/nrc/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` ([EmoLex](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm))
- `data/external/nrc/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt` ([NRC VAD](https://saifmohammad.com/WebPages/nrc-vad.html); v1 also works, set `VAD_PATH`)
```bash
python scripts/09_check_lexicons.py              # detected format/scale, GoEmotions label words in VAD
python scripts/10_score_fields.py --measure      # token lengths -> results/tables/token_lengths.json
python scripts/10_score_fields.py --smoke        # 12 items + GPU memory check, writes nothing
python scripts/10_score_fields.py                # title/abstract scoring (resumable), weighted raw28, go_* features
python scripts/10_score_fields.py --report       # combined vs weighted: dominant class and neutral distribution
python scripts/11_lexicon_features.py            # nrc_*, vad_* (weighted) and go_vad_*; coverage + top words tables
python scripts/12_merge_emotion_features.py      # one row per news_id, all variants -> news_emotion_features.parquet
python scripts/13_emotion_diagnostics.py         # distributions, cross-source + title-vs-abstract correlations, heatmap
python scripts/14_emotion_signal.py              # fit only: clicked-minus-unclicked dz, t-test + user level, Holm
python scripts/15_congruence_signal.py          # fit only: history-profile congruence (+ generic control)
```

## Rules that protect the results
- `fit` trains, `valid` (last day of train) tunes, `dev` is reported once. Never tune on dev.
- Every model returns `imp_key, news_id, score` for all candidates and is scored with `src.evaluate.evaluate_model`.
- `position` in the impression table is log order. Never use it as a feature.
- Popularity and any other train-derived statistic must be computed from `fit` only.
- Raw 28-label emotion scores are saved, so changing the 28 to 6 mapping never needs a re-run of the model.
- Labelled chunks are tied to model, max length, chunk size and text hash; changed settings are refused.
