# Beyond Keywords: project notes for Claude

MSc Data Analytics practicum (NCI). Question: does emotional congruence between a user's click history and a news item
improve click ranking? Data: MIND-small (news stands in for ads). Planned 8-model ablation, each with/without emotion:
M0 popularity, M1 TF-IDF, M2 Sentence-BERT, M3 = M2+emotion, M4 LightGBM, M5 = M4+emotion, M6 NRMS, M7 = M6+emotion.
M0 and M1 are done (valid + dev once each); do not build further models unless asked.

## Research direction
Recommenders match content context (words/meaning). Hypothesis: clicks are also driven by emotional context, so content
context + emotion improves recommendation (ads/news/similar content). Wording rule: write "emotional context inferred
from the user's recent reading", never "the user's emotional state".

## Data (data/processed/, built by `python scripts/run_base.py`)
- news.parquet: news_id, category, subcategory, title, abstract, text (title + ". " + abstract). 65,238 items, 0 empty titles,
  3,415 (5.23%) empty abstracts.
- impressions.parquet: imp_key, part, user_id, time, news_id, label, position. part = fit (train) / valid (tune) / dev (test once).
- news_emotion_raw28.parquet + news_emotion.parquet: "combined" baseline (GoEmotions on `text`, max_length 128, 6 classes).
  Never overwrite these or emotion_chunks/. Hand labelling is cancelled (D45); its material is archived in results/archive/annotation/.

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
- Congruence (script 31, shared code src.congruence.congruence_features): profile from behaviors.history only
  (decay 0.9, last 50, cold start = corpus mean). Always report raw, *_generic, *_personal (raw - corpus-mean profile,
  pre-declared) and the stricter *_otheruser / *_userspecific control (raw - random other user's profile).
- go_pct_*: percentile rank of each go_* emotion across all items (item-level, no clicks), because label floors differ.
  go_dist_* is kept but unreliable at low go_emotion_mass; go_emotion_mass can exceed 1 (sum of independent sigmoids).

- Emotion spec: config/emotion_spec_v3.json (confirmed on SemEval, D48/D49; v1 and v2 kept for the record; v3 differs from v2 only in
  labelling). Title share 0.5, labels at t = 0.05 are for reporting only. M3/M5/M7 must use exactly its features/fusion
  (news_emotion_features_v2.parquet). Any change to its hashed content = new file v4 + decision entry + test hash (tests/test_emotion_spec.py).

- Reporting labels: v2b at t = 0.05 (no approval/disapproval), chosen on SemEval trial (D47); v2 and v2c stay for reference.
- SemEval-2007 (scripts 14, 29, 38, src/semeval.py): trial selects, test reports. Headlines only: it validates the title part of the scoring.
- True recency (script 35) uses earlier log clicks for analysis only; M2-M7 use the history column only (Q09 decided, D45).
- M5-all is a pre-declared robustness variant with all 19 item candidates (D37).

## Engineering rules
- Decisions: docs/decisions.md (every non-trivial choice, with evidence part). Progress: docs/progress.md.
  Open questions: docs/open_questions.md. Data rules: fit trains, valid chooses, dev once per model (--final-dev).
- Logic in src/, thin scripts in scripts/ (import _bootstrap), all constants in src/config.py, provenance via src/provenance.py.
  Scripts are numbered by stage: 1x data, 2x emotion, 3x analysis, 4x models. Start at docs/START_HERE.md.
- Model labelling is resumable: chunk folders with a settings.json guard and atomic .tmp -> replace writes (label_corpus).
  New runs use new chunk folders and new output files.
- Do not change src/data.py, src/metrics.py, src/evaluate.py unless needed (explain why).
- No leakage: features are item-level text only; click analyses use fit, replication and settings use valid; never load or tune on dev.
- Every new function gets a test with small fakes (fake classifier, tiny lexicon files in tmp_path); no internet/GPU/real lexicons.
- NRC files live in data/external/nrc/ (git-ignored, research-only licence). Never download or commit them.
- Windows, PowerShell, Python 3.11 venv (.venv), RTX 3050 6GB; use pick_device("auto").
- `python scripts/run_base.py` runs the whole pipeline incrementally (src/pipeline.py). Every new script must be
  registered there as a Step with its data inputs and outputs (code deps are found from its src imports).
- Short code comments. Never push. Commit only when the user asks (autonomous runs commit locally on their own branch).
- Work in phases and stop for review after each one; ask when the request conflicts with the code.

## Phases
Done: base pipeline, emotion layer v1 and v2, click signal, congruence, context windows, true recency, M0, M1, cleanup, SemEval validation.
Next: M2, M3.
