# Beyond Keywords: project notes for Claude

MSc Data Analytics practicum (NCI). Question: does emotional congruence between a user's click history and a news item
improve click ranking? Data: MIND-small (news stands in for ads). Planned 8-model ablation, each with/without emotion:
M0 popularity, M1 TF-IDF, M2 Sentence-BERT, M3 = M2+emotion, M4 LightGBM, M5 = M4+emotion, M6 NRMS, M7 = M6+emotion.
No ranking model is built yet; do not build one unless asked.

## Data (data/processed/, built by `python scripts/run_base.py`)
- news.parquet: news_id, category, subcategory, title, abstract, text (title + ". " + abstract). 65,238 items, 0 empty titles,
  3,415 (5.23%) empty abstracts.
- impressions.parquet: imp_key, part, user_id, time, news_id, label, position. part = fit (train) / valid (tune) / dev (test once).
- news_emotion_raw28.parquet + news_emotion.parquet: "combined" baseline (GoEmotions on `text`, max_length 128, 6 classes).
  Never overwrite these, emotion_chunks/, or annotation/manual_sample.csv (6-class hand labels, filled by the user).

## Emotion layer (supervisor-approved design)
- Soft probabilities everywhere, never a hard label as a feature.
- Title and abstract scored separately, combined with TITLE_WEIGHT 0.3 / ABSTRACT_WEIGHT 0.6 normalised (1/3, 2/3).
  Empty abstract -> title only. Never score an empty string. Combine the raw 28 GoEmotions scores first, then derive features.
- Sources (ablation): A = GoEmotions (SamLowe/roberta-base-go_emotions); B = NRC EmoLex + NRC VAD v2.1.
- Prefixes: go_*, go_dist_* (8 emotions, sums to 1; uniform only at exact zero sum), go_emotion_mass (sum of the 8 raw
  scores; congruence should be weighted/filtered by it), nrc_* (8 emotion shares + nrc_positive/nrc_negative separate,
  coverage, has_hits), vad_* (mean over matched tokens in [-1, 1], vad_coverage), go_vad_* (probability-weighted VAD of
  the 27 non-neutral label words; GO_VAD_OVERRIDES in config for missing words).
- Lexicon fields with no hits: renormalise the weights over fields with hits; none -> 0 and has_hits = 0.
- NRC -> 6 classes (validation only): anger + disgust -> anger, no hits -> neutral, trust/anticipation ignored,
  ties broken by seeded random choice (SEED), tie rate reported.
- Expect high neutral and partial lexicon coverage; report honestly, do not force emotions.
- Variants (Phase 4/5): go_title_* and go_abs_* from the saved title/abstract raw28 files (no re-run; go_abs_* = NaN
  without abstract); Phase 5 compares title vs abstract vs weighted on fit only. Configured weights stay unchanged.
- Phase 5 test (user-approved): one-sample t-test on per-impression clicked-minus-unclicked differences + user-level
  t-test, Holm. Not Wilcoxon: biased here (93/101 false positives with labels shuffled within impressions).
  No popularity control in Phase 5 (limitation; handled by LightGBM M4 vs M5).
- go_pct_*: percentile rank of each go_* emotion across all items (item-level, no clicks), because label floors differ.
  go_dist_* is kept but unreliable at low go_emotion_mass; go_emotion_mass can exceed 1 (sum of independent sigmoids).

## Engineering rules
- Logic in src/, thin scripts in scripts/ (import _bootstrap), all constants in src/config.py, provenance via src/provenance.py.
- Model labelling is resumable: chunk folders with a settings.json guard and atomic .tmp -> replace writes (label_corpus).
  New runs use new chunk folders and new output files.
- Do not change src/data.py, src/metrics.py, src/evaluate.py unless needed (explain why).
- No leakage: features are item-level text only; click analyses use part == "fit" only; never look at or tune on dev.
- Every new function gets a test with small fakes (fake classifier, tiny lexicon files in tmp_path); no internet/GPU/real lexicons.
- NRC files live in data/external/nrc/ (git-ignored, research-only licence). Never download or commit them.
- Windows, PowerShell, Python 3.11 venv (.venv), RTX 3050 6GB; use pick_device("auto").
- Short code comments. Do not commit or push; suggest a commit message instead.
- Work in phases and stop for review after each one; ask when the request conflicts with the code.

## Phases
0 plan, 1 config + lexicon loaders, 2 weighted GoEmotions, 3 NRC + go_vad features, 4 merged feature table + diagnostics,
5 signal analysis on fit (Wilcoxon, Holm), 6 hand-label validation per source. Stop after 6.
