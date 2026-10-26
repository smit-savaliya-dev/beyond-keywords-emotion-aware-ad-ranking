# Week 5 Report

## Summary of research activities
- Tested emotional congruence on 126,695 training impressions: clicked news is closer to the emotions of the user's
  own click history (cosine dz 0.20). After removing what all users share (a random other user's profile), a
  user-specific effect remains (dz 0.14).
- The personal congruence effect replicates on the next day (validation, 30,270 impressions): dz 0.24 on both days.
  No sign of mood repair: every personal congruence measure is positive.
- Item-level emotion effects are less stable: 9 of 25 emotion features failed the next-day check (e.g. overall
  emotionality +0.10 on training days, -0.04 on the validation day).
- Built two baselines without emotion. Validation AUC: popularity 0.574, TF-IDF 0.590 (TF-IDF better on all 10
  metrics). Final test, run once: popularity 0.589, TF-IDF 0.574; TF-IDF better only on top-k metrics
  (HR@10 0.671 vs 0.660).
- Froze a provisional emotion specification before building any emotion model: 12 item features and 4 congruence
  features, chosen by a rule written down before checking the validation day.
- Neutral share depends strongly on the threshold: 24% to 93% of items are neutral for thresholds 0.05 to 0.5
  (current default: 81%).
- Built a labelling tool and guidelines for the 180-item hand-label check, plus a 30-item blind re-label for
  intra-rater reliability.
- Learned: emotion seems to matter as a personal match with the user's history, not as one "clickable" emotion, and
  popularity is a strong baseline on the test day.
- Next: build M2 (Sentence-BERT) and M3 (Sentence-BERT + emotion congruence, weight chosen on validation), and finish
  the hand labels to confirm the emotion specification.

## Supervisor Feedback
_(to be added after the meeting)_
