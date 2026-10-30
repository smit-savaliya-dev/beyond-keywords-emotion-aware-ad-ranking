# Base v2 report (2026-10-30)

Branch `base-v2` (local commits only, never pushed), built on `week5-autonomous`. 137 tests pass. The pipeline is up to date
except step 06, which waits for the hand labels. Only fit and valid were used; dev was never loaded. The rules were committed
before any computing (commit 7ccfe5e). Hand-label sheet unchanged.

## Summary for the professor
1. Recommenders today match words and meaning. This work tests whether emotional context adds to that.
2. We changed how emotion is labelled, which cuts neutral news from 81% to 26%. The classifier itself did not change, so
   this is a looser definition of emotion, and hand labels must confirm it.
3. Clicked news is closer to the emotional context inferred from the user's recent reading, and this holds on the next day.
   The strictest version, which removes what all users share, is dz 0.14 on both periods: small but consistent.
4. A long reading history works better than the last few items, but newer items are not better than slightly older ones.
   This looks like a stable preference rather than a short-term effect. True recency could not be tested with this data.
5. There is no simple "emotion that gets clicked": 7 of 20 emotion features reverse direction on the next day. The
   personal match is what stays stable.
6. Nothing yet shows better ranking. The next step is to test whether emotion adds value beyond content (M2 vs M3).

## Chosen setting and why
**Title share 0.5, abstract share 0.5, t = 0.05** (rule D30, spec v2 provisional).
- Criterion 1: neutral 25.6%, inside the preferred limit of 30%. 8 settings passed in that tier.
- Criterion 2: valid personal congruence dz 0.2388, above the gate of 0.2134 (and above the current 0.2371).
- Criterion 3: best min(fit, valid) = 0.2388. Next: share 0.6 at 0.2381, share 1/3 at 0.2371, share 0.7 at 0.2362.
- t was set by the tie rule: the congruence measure uses soft vectors, so it depends on the weight and not on t. Higher t
  wins; t = 0.05 is the highest value that stays within 30%.
- Shuffled labels: 0 false positives on fit and on valid.

**Be careful with this result:**
- The weights are effectively tied. Their gaps (under 0.003) are smaller than one confidence interval (about +/-0.011).
- Title-only is clearly weaker for congruence (0.224 fit, 0.217 valid vs 0.246, 0.239), and reaches 30.4% neutral only at the
  lowest t.

## Neutral before and after
| Rule | Neutral % | Joy | Sadness | Anger | Fear | Surprise |
|---|---|---|---|---|---|---|
| Original (combined text, argmax) | 75.9 | 10.2 | 6.1 | 0.7 | 0.6 | 6.5 |
| v1 (1/3 title, argmax) | 81.1 | 8.7 | 5.3 | 0.6 | 0.4 | 4.0 |
| **v2 (title 0.5, t = 0.05)** | **25.6** | **34.5** | **18.9** | **5.6** | **1.9** | **13.5** |

What this does and does not mean:
- A low t reduces neutral by definition. 55.5% of items left neutral, and for 91.6% of those the classifier's own neutral score
  is still above 0.5. Among all items labelled emotional, 75.0% have a neutral score above 0.5.
- "Joy" is 51% approval and 15% admiration; "anger" is 60% disapproval. These shares come partly from the 28-to-6 mapping.
- Labels are for reporting and validation only. Model features stay soft, so congruence does not depend on them.

## Full grid: neutral and class shares for every (weight, t)
| Title share | t | Neutral | Joy | Sadness | Anger | Fear | Surprise | Pass |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 0.03 | 30.4 | 32.3 | 17.7 | 6.8 | 1.9 | 10.7 |  |
| 1.00 | 0.05 | 43.5 | 23.8 | 16.3 | 5.3 | 1.7 | 9.5 |  |
| 1.00 | 0.075 | 52.3 | 18.5 | 14.8 | 4.2 | 1.5 | 8.7 |  |
| 1.00 | 0.1 | 58.1 | 15.1 | 13.8 | 3.5 | 1.3 | 8.1 |  |
| 1.00 | 0.15 | 65.7 | 11.2 | 12.0 | 2.6 | 1.1 | 7.4 |  |
| 1.00 | 0.2 | 70.9 | 8.8 | 10.6 | 2.0 | 1.0 | 6.8 |  |
| 1.00 | 0.3 | 77.8 | 6.0 | 8.4 | 1.2 | 0.7 | 5.8 |  |
| 0.70 | 0.03 | 15.2 | 41.1 | 20.2 | 7.0 | 2.1 | 14.3 | yes |
| 0.70 | 0.05 | 29.6 | 31.5 | 18.7 | 5.6 | 1.9 | 12.7 | yes |
| 0.70 | 0.075 | 41.3 | 24.3 | 16.9 | 4.4 | 1.7 | 11.4 |  |
| 0.70 | 0.1 | 49.4 | 19.6 | 15.3 | 3.6 | 1.5 | 10.6 |  |
| 0.70 | 0.15 | 60.9 | 13.6 | 12.8 | 2.4 | 1.2 | 9.2 |  |
| 0.70 | 0.2 | 69.3 | 9.9 | 10.7 | 1.7 | 0.9 | 7.5 |  |
| 0.70 | 0.3 | 81.2 | 5.0 | 7.6 | 0.8 | 0.5 | 4.9 |  |
| 0.60 | 0.03 | 13.5 | 42.5 | 20.3 | 6.9 | 2.1 | 14.8 | yes |
| 0.60 | 0.05 | 27.3 | 33.2 | 18.9 | 5.6 | 1.9 | 13.1 | yes |
| 0.60 | 0.075 | 39.3 | 25.7 | 17.1 | 4.4 | 1.7 | 11.8 |  |
| 0.60 | 0.1 | 47.4 | 21.0 | 15.6 | 3.6 | 1.5 | 10.9 |  |
| 0.60 | 0.15 | 59.0 | 14.8 | 13.1 | 2.4 | 1.2 | 9.6 |  |
| 0.60 | 0.2 | 67.2 | 10.9 | 11.1 | 1.7 | 0.9 | 8.3 |  |
| 0.60 | 0.3 | 80.5 | 5.9 | 7.6 | 0.7 | 0.5 | 4.8 |  |
| 0.50 | 0.03 | 12.3 | 43.6 | 20.2 | 6.8 | 2.1 | 15.1 | yes |
| **0.50** | **0.05** | **25.6** | 34.5 | 18.9 | 5.6 | 1.9 | 13.5 | yes |
| 0.50 | 0.075 | 37.6 | 27.1 | 17.2 | 4.4 | 1.7 | 11.9 |  |
| 0.50 | 0.1 | 46.0 | 22.2 | 15.7 | 3.6 | 1.5 | 11.0 |  |
| 0.50 | 0.15 | 57.7 | 15.9 | 13.2 | 2.5 | 1.2 | 9.6 |  |
| 0.50 | 0.2 | 66.3 | 11.8 | 11.1 | 1.7 | 0.9 | 8.2 |  |
| 0.50 | 0.3 | 79.3 | 6.8 | 7.6 | 0.7 | 0.5 | 5.1 |  |
| 0.33 | 0.03 | 11.0 | 45.3 | 19.7 | 6.6 | 2.0 | 15.4 | yes |
| 0.33 | 0.05 | 23.8 | 36.4 | 18.5 | 5.6 | 1.8 | 13.9 | yes |
| 0.33 | 0.075 | 36.1 | 29.0 | 17.0 | 4.5 | 1.6 | 12.0 |  |
| 0.33 | 0.1 | 44.8 | 23.9 | 15.5 | 3.6 | 1.4 | 10.8 |  |
| 0.33 | 0.15 | 57.3 | 17.2 | 12.9 | 2.5 | 1.1 | 9.0 |  |
| 0.33 | 0.2 | 66.5 | 13.0 | 10.6 | 1.7 | 0.8 | 7.2 |  |
| 0.33 | 0.3 | 79.4 | 7.5 | 7.1 | 0.8 | 0.5 | 4.8 |  |

Column "pass" = passes (1) and (2) in the tier used (neutral <= 30%). Personal dz per weight, fit / valid: title-only 0.2240 /
0.2173; 0.7 0.2474 / 0.2362; 0.6 0.2474 / 0.2381; 0.5 0.2463 / 0.2388; 1/3 0.2419 / 0.2371.

## Short-term vs long-term context
| Window | Raw dz fit / valid | Personal dz fit / valid | User-specific dz fit / valid |
|---|---|---|---|
| **Last 50, decay 0.9 (long-term)** | 0.191 / 0.167 | 0.246 / 0.239 | **0.138 / 0.137** |
| Last 5 | 0.176 / 0.149 | 0.200 / 0.195 | 0.112 / 0.107 |
| Last 3 | 0.163 / 0.136 | 0.167 / 0.166 | 0.094 / 0.094 |
| Last 1 | 0.117 / 0.096 | 0.097 / 0.102 | 0.057 / 0.062 |

- All 12 features are significant at both levels on both parts; shuffled labels 0 of 12 on both.
- Long-term matters most on both parts: short minus long, user-specific dz, is -0.081, -0.044, -0.026 on fit and -0.075,
  -0.043, -0.030 on valid (all Holm-significant).
- Same-size control, users with at least 10 history items: last 5 vs previous 5 is 0.111 vs 0.123 on fit (older is better,
  p Holm 0.003) and 0.113 vs 0.118 on valid (not significant). So the gain does not come from recency.
- Limitation: history is one list per user from before the log period, so "last 1" is not the click just before the impression.

## The 8 charts (`results/figures/story/`, explained in `docs/figures_guide.md`)
1. **Emotion mix:** neutral drops from 81% to 26%, by a looser rule rather than a better classifier.
2. **Neutral vs threshold:** the threshold moves neutral far more than the title weight.
3. **Title vs abstract intensity:** abstracts score higher (median 0.116 vs 0.068).
4. **Emotion by category:** food and drink is 58% joy; sadness leads only in news (28%).
5. **Valence-arousal map:** categories differ on average, but items within a category vary much more.
6. **Clicked vs not clicked:** 7 of 20 features flip sign on the next day; fear is stable.
7. **Congruence:** the strict user-specific effect is almost identical on both periods (0.138, 0.137).
8. **Short vs long context:** longer history is stronger, but newer items are not better than slightly older ones.

## Spec v2 (provisional; v1 kept)
- 13 item features: go_pct_excitement, joy, optimism, nervousness, annoyance, fear; go_vad_valence, go_vad_dominance;
  vad_valence; nrc_trust; go_dist_optimism, nervousness, fear.
- 4 congruence features: cong_pct_cos, cong_pct_cos_personal, cong_six_negkl_personal, cong_valence_absdist_personal.
- 17 of 26 candidates kept (D32). All sources use the same weight. Script 06 now validates v2 labels and can revert them.

## Issues found and fixed
1. **A wrong claim of mine, now corrected.** I said neutral rose under weighting because abstracts are more neutral than titles.
   That is false: on the same items, neutral is 79.5% for titles, 76.2% for abstracts and 84.0% for their mean. The cause is
   dilution by averaging (D36, methodology notes corrected).
2. **A drafted chart title was false** ("titles reach higher intensity"). Caught when the data came in; fixed.
3. **A drafted chart caption was unverified.** Replaced by numbers computed in the script.
4. **Long-vs-short was confounded by profile size.** Added a same-size control before computing (D31).
5. **Conflict over intensity:** you asked to keep it as a feature, but D32 rejects it because its sign flips (+0.149, -0.036). It
   stays in the feature table and is left out of the M5 list (D34, Q05).
6. **Examples could break your blinding.** The 60 examples exclude your 180 items (Q06).
7. Chart label collisions and overlapping ticks fixed after inspecting each render. Long shell commands failed to parse
   (nothing ran); edits were moved into script files.

## Open questions (`docs/open_questions.md`)
- **Q01** Headline user-specific instead of personal? Chosen: report both, headline the stricter one.
- **Q02** Run M5 on all 19 item candidates as a variant? Chosen: no, unless pre-declared.
- **Q03** Percentiles include dev-only items (no labels or clicks). Chosen: kept, noted.
- **Q04** Are the t = 0.05 labels acceptable? Chosen: kept for reporting; the hand labels decide.
- **Q05** Should intensity be a separate M5 variant? Chosen: not in the list; the column is kept.
- **Q06** Recommended: finish your 180 labels before reading `emotion_examples_v2.csv`.
- **Q07** Should a real recency test use time-ordered log clicks? Chosen: not done; it needs your approval and a pre-declaration.

## Next steps for you
1. Label the 180 items first: `python scripts/label_tool.py`, then read `results/tables/emotion_examples_v2.csv`.
2. Create the re-label sheet with `python scripts/label_tool.py --make-relabel`; label it a day later with `--relabel`.
3. Run `python scripts/run_base.py`: it reports kappa for v2 labels against v1 and confirms, bumps or holds spec v2.
4. Review `git log week5-autonomous..base-v2` and merge if you agree.
