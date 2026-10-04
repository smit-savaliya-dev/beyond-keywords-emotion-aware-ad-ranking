# Beyond Keywords: Emotion-Aware Ranking

8-model ablation (M0 to M7) testing whether emotional congruence improves click ranking on MIND-small.
MSc Data Analytics Research Practicum, NCI. Demo platform: [AURA](https://github.com/smit-savaliya-dev/aura-ai-powered-emotion-aware-ad-recommender-platform).

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Put MIND-small in `data/raw/train/` and `data/raw/dev/` (`news.tsv`, `behaviors.tsv`), or let step 01 download it.

## 1. Base (data + evaluation harness + EDA)
```bash
python scripts/run_base.py
```
Runs tests, parsing with integrity and official-count checks, the harness sanity check, and the EDA figures. Stops at the first failure.

## 2. Emotion labels
```bash
python scripts/04_label_emotions.py --smoke      # 12 headlines, writes nothing: confirms the model works
python scripts/04_label_emotions.py              # full run, resumable (use a GPU / Colab if possible)
python scripts/05_make_annotation_sample.py      # then fill your_label in annotation/manual_sample.csv BEFORE any model output
python scripts/06_validate_emotion_labels.py     # kappa + 95% CI, accuracy, macro-F1, confusion matrix
python scripts/08_goemotions_check.py            # optional: in-domain check on GoEmotions test split
```
Colab: upload `src/`, `scripts/` and `data/processed/news.parquet`, run step 04, download `data/processed/emotion_chunks/` and `news_emotion*.parquet` back.

## Rules that protect the results
- `fit` trains, `valid` (last day of train) tunes, `dev` is reported once. Never tune on dev.
- Every model returns `imp_key, news_id, score` for all candidates and is scored with `src.evaluate.evaluate_model`.
- `position` in the impression table is log order. Never use it as a feature.
- Popularity and any other train-derived statistic must be computed from `fit` only.
- Raw 28-label emotion scores are saved, so changing the 28 to 6 mapping never needs a re-run of the model.
- Labelled chunks are tied to model, max length, chunk size and text hash; changed settings are refused.
