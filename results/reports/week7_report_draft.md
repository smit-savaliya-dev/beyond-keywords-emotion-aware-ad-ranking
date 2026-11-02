# Week 7 Report

## Summary of research activities
- Checked the M2 encoder against the sentence-transformers package on 500 items per text variant: minimum cosine 0.99999999999, so M2 and M3 stand.
- Made six results charts for the supervisor, and resolved the spec conflict: spec v4 names the personal score for fusion, with nothing else changed.
- Built M4 (LightGBM ranker with popularity, content and context) and M5 (M4 + emotion). All rules were written down before training. Click-based features use only earlier days, and a test proves that a row's own
  impression never contributes.
- Final test AUC: popularity 0.589, TF-IDF 0.574, M2 0.625, M3 0.628, M4 0.674, M5 0.671. M4 is the best model so far (M4 minus M2 +0.049).
- Adding emotion to M4 did not help on the final test: M5 minus M4 -0.0030 (95% CI -0.0042 to -0.0018). The pre-declared question is answered no. M4 varies by 0.0037 between seeds on validation, so this reads as no gain, not as harm.
- On validation emotion looked like a clear win (+0.014). Before any final test run, a control gave every item another item's emotion scores: that meaningless version still gained +0.008. Trees were partly recognising individual
  items from their scores.
- Validation shared 71% of its candidates with training and the final test only 19%. On validation the whole gain sat on items already seen in training.
- Real emotion scores still beat the meaningless ones on the final test (+0.0044), so they carry some information, but not enough to beat M4.
- Learned: continuous item scores can act as an item ID, and a validation day close to training can reward that. Across the project the only positive emotion result on the final test is the small M3 gain (+0.003).
- Limits: one seed on the final test, no human labels on MIND, and the congruence-only control is not like for like with M4.
- Next: M6 and M7 (NRMS with and without emotion), with the seen and unseen item split pre-declared from the start.

## Supervisor Feedback
_(to be added after the meeting)_
