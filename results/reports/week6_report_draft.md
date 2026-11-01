# Week 6 Report

## Summary of research activities
- Hand labelling was cancelled. Validated the emotion scores on an external human benchmark instead: SemEval-2007 Task 14 Affective Text
  (Strapparava & Mihalcea, 2007), 250 trial and 1,000 test news headlines. All rules were written down before scoring.
- GoEmotions scores follow human ratings on the test headlines. Spearman: valence 0.72, sadness 0.60, anger 0.52, fear 0.46, joy 0.45. Surprise
  is weak (0.19).
- GoEmotions is stronger than NRC EmoLex on every emotion except fear, where the two overlap (valence 0.72 vs 0.50).
- The emotion specification passed the pre-declared check (Spearman at least 0.2 for valence and at least 3 of 4 main emotions).
- The trial set chose label rule v2b (no approval or disapproval). On test: kappa 0.30, fair agreement, against 0.22 for the old rule. Rules
  v2, v2b and v2c cannot be separated statistically. The labels stay for reporting only, so model features did not change.
- Neutral share: humans 65%, our rule 61% on the same headlines, 52% on MIND titles and 39% on MIND title + abstract. The human figure also depends
  on the cutoff (16% to 65%), so neutral is partly a definition.
- Decided that models M2 to M7 use the standard MIND history column only. Recency may come later as a labelled extension.
- Rebuilt the full pipeline without GPU scoring: all 76 earlier result files stayed identical.
- Learned: emotion scores measure something real in headlines, but exact labels are much less reliable than soft scores, which supports the choice
  of soft features.
- Limits: the check covers 2007 headlines only, not abstracts, and there are no human labels on MIND. It says nothing about clicks yet.
- Built M2 (Sentence-BERT) and M3 (M2 + emotion, late fusion). All rules were written down before computing, and each model ran once on the final test.
- Final test AUC: popularity 0.589, TF-IDF 0.574, M2 0.625, M3 0.628. Meaning-based matching clearly beats both baselines.
- M3 beats M2 by +0.0029 AUC (95% CI 0.0023 to 0.0035, dz 0.03), close to the validation gain of +0.0031. The gain is real but very small, and no top-of-list
  metric improved significantly on the final test.
- The best emotion score was personal congruence with weight 0.2. Mood repair never beat M2, so clicks follow the match, not the opposite.
- Checks passed: weight 0 on emotion reproduces M2 exactly, and shuffled emotion scores never beat M2.
- Where it helps (descriptive): more for long histories (21+ items) and news and music clicks; negative for sports and finance. Emotion is linked to topic,
  so these are not causes.
- Next: M4 and M5 (LightGBM with and without emotion), to see whether emotion adds more inside a richer model.

## Supervisor Feedback
_(to be added after the meeting)_
