# Methodology notes

Short justifications for the main methodological choices. Decision IDs (Dxx) refer to `docs/decisions.md`.

## 1. Train / valid / dev split
| Part | Source | Dates (2019) | Impressions | Candidate rows | Use |
|---|---|---|---|---|---|
| fit | MIND-small train, all days but the last | 9-13 Nov | 126,695 | 4,621,015 | train models, compute click statistics, signal analyses |
| valid | MIND-small train, last day | 14 Nov | 30,270 | 1,222,429 | choose settings (grids, alpha, feature replication) |
| dev | official MIND-small dev | 15 Nov | 73,152 | 2,740,998 | final test, once per model with frozen settings |

- **Official dev as the test set** keeps results comparable with published MIND work, which reports on this split.
- **Time-based validation (last train day)** mirrors deployment: settings are chosen on a day that comes after the
  training days and before the test day, so no future information is used.
- **A random 80/20 split would leak the future**: training rows from later days would inform predictions for earlier
  ones, and popularity statistics would include clicks that happen after the scored impression.
- Click statistics only use earlier parts: valid is scored with fit statistics, dev with fit + valid (M0, D21).
- The pipeline enforces this: data loaders read only the requested part (D18), analysis loaders refuse dev, and
  `--final-dev` refuses a second dev run or unfrozen settings (D19).

## 2. Title and abstract weighting
- Title and abstract are scored separately with GoEmotions (title max 128 tokens, abstract max 256: 0.48% of
  abstracts exceed 128, 0.055% exceed 256; D03).
- Combined score = (0.3 x title + 0.6 x abstract) / 0.9, i.e. **exactly 1/3 title and 2/3 abstract**. Items without an
  abstract (3,415, 5.23%) use the title alone (weight 1.0). The raw 28 scores are combined first, then all features
  are derived (D01, D02).
- Honest note: on the full corpus, the dominant class is neutral for 75.9% of items with the combined text and for
  81.1% with the weighted scores, because abstracts are more neutral than titles. The weighting did not reduce
  neutrality; percentile features (section 4) and thresholds (Phase 4) address this instead.
- Title and abstract agree only moderately on the same item (Spearman 0.27-0.56 per emotion), and in the click
  analysis the title-only percentile features carried slightly more signal than abstract-only (fit).

## 3. Why the Wilcoxon test was replaced
- Each impression gives one difference: mean(feature | clicked) - mean(feature | not clicked), with 1-2 clicks
  against ~35 non-clicks. For right-skewed features a single draw usually lies below the mean of many draws, so the
  differences are not symmetric around 0 even with no effect, which violates the Wilcoxon assumption.
- Evidence (fit): with labels shuffled within impressions, Wilcoxon flagged **93 of 101** features as significant;
  the one-sample t-test on the same differences flagged **0** (user level 1, both levels 0). The t-test is valid by
  the central limit theorem with ~10^5 impressions and matches the effect size dz = mean / sd.
- Primary test since then: t-test + Holm, plus a user-level t-test (users averaged), significant only if both agree
  in sign (D09, D10). Wilcoxon is kept as a robustness column.

## 4. Percentile calibration (go_pct_*)
- GoEmotions labels have different floors on neutral text: median raw score annoyance 0.0074, optimism 0.0051,
  joy 0.0048, desire 0.0028, curiosity 0.0020, nervousness 0.0009. 69.8% of items have total emotion mass < 0.1,
  where the normalised distribution go_dist_* mostly reflects these floors.
- go_pct_* = percentile rank of each emotion across all items (item-level, no clicks). It removes the floor
  differences. On fit it carried more click signal than raw scores (e.g. fear dz 0.017 raw vs 0.162 percentile);
  this was observed after the features were defined, not used to define them (D08).
- Limitation: ranks are computed over all 65,238 items including dev-only items. No labels or clicks are involved,
  so this is not leakage, but it is transductive (Q03).

## 5. GoEmotions vs NRC lexicons
Spearman correlation on the same items (items with at least one EmoLex emotion word, n = 60,109; VAD n = 65,226):

| GoEmotions | NRC | Spearman |
|---|---|---|
| go_vad_valence | vad_valence | 0.551 |
| go_fear | nrc_fear | 0.420 |
| go_nervousness | nrc_fear | 0.360 |
| go_excitement | nrc_joy | 0.256 |
| go_annoyance | nrc_anger | 0.250 |
| go_vad_dominance | vad_dominance | 0.241 |
| go_joy | nrc_joy | 0.220 |
| go_optimism | nrc_anticipation | 0.210 |
| go_emotionality | nrc_coverage | 0.095 |
| go_vad_arousal | vad_arousal | -0.010 |

- **Agree:** valence and negative emotions (fear) moderately. **Disagree:** arousal (unrelated) and "how emotional"
  an item is (0.095). The two sources measure different things, which makes the A/B source ablation informative.
- EmoLex on news is partly topical: the most frequent matched words are police (fear, trust), trump (surprise),
  county, school and president (trust). 28.3% of titles, 13.9% of abstracts and 7.9% of items (weighted) have no
  EmoLex emotion word. Topic words were not filtered (D07).
- NRC -> 6 classes (validation only): anger + disgust -> anger, no hits -> neutral, trust and anticipation not mapped,
  ties by seeded random choice; its agreement with hand labels is reported once the labels exist.

## 6. Congruence and personalisation
- Profile = recency-weighted mean (decay 0.9, newest weight 1, last 50) of history-item vectors from the `history`
  column only; cold start (1.95% of fit impressions) = corpus mean (D11).
- `*_personal` = measure(user profile) - measure(corpus mean), pre-declared. Because histories are past clicks, every
  profile is shifted towards "typically clicked news", so a random other user's profile also predicts clicks
  (cong_pct_cos_otheruser dz 0.089 on fit). The stricter `*_userspecific` = measure(user) - measure(random other user)
  removes that shared shift: dz 0.136 (fit), still significant at both levels (D13, Q01).
- Out-of-time check on valid: personal congruence replicated (0.242 -> 0.237), while several item-level emotion
  effects flipped sign (e.g. go_emotionality +0.103 -> -0.041), so item emotion effects depend on the day's news (D28).
