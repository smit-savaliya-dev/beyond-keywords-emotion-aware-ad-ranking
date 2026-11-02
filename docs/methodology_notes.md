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
  81.1% with the weighted scores. The cause is dilution by averaging: when title and abstract peak on different emotions,
  the mean weakens both peaks while the neutral score stays high. Abstracts are not more neutral than titles: on the same
  items, argmax neutral is 79.5% for titles, 76.2% for abstracts and 84.0% for their plain mean. (An earlier version of
  this note gave the wrong reason; corrected 2026-10-30.) Spec v2 addresses this with a threshold on intensity.
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
  ties by seeded random choice; its agreement with human ratings on SemEval-2007 is in section 7.

## 6. Congruence and personalisation
- Profile = recency-weighted mean (decay 0.9, newest weight 1, last 50) of history-item vectors from the `history`
  column only; cold start (1.95% of fit impressions) = corpus mean (D11).
- `*_personal` = measure(user profile) - measure(corpus mean), pre-declared. Because histories are past clicks, every
  profile is shifted towards "typically clicked news", so a random other user's profile also predicts clicks
  (cong_pct_cos_otheruser dz 0.089 on fit). The stricter `*_userspecific` = measure(user) - measure(random other user)
  removes that shared shift: dz 0.136 (fit), still significant at both levels (D13, Q01).
- Out-of-time check on valid: personal congruence replicated (0.242 -> 0.237), while several item-level emotion
  effects flipped sign (e.g. go_emotionality +0.103 -> -0.041), so item emotion effects depend on the day's news (D28).

## 7. External validation on SemEval-2007
- Why: hand labelling was cancelled (D45), so there are **no direct human labels on MIND headlines**. SemEval-2007 Task 14 Affective Text
  (Strapparava & Mihalcea, 2007) gives human emotion ratings (0-100) and valence (-100 to 100) for 1,250 news headlines.
- Design, pre-declared before scoring (D46-D48): trial (250) only selects the labelling rule, test (1,000) only reports. Threshold-free
  correlations are primary, because the model features are continuous and labels are for reporting only. Gold class: anger = max(anger,
  disgust), argmax of 5 emotions, neutral below 50, which is the README's coarse-grained cutoff. Sensitivity cutoffs 25 and 30.
- Same scoring as MIND: same model, max length 128, same lexicons, same no-hits rule, same go_vad formula.
- **Title-only limitation:** SemEval has no abstracts, so this validates the title part of the scoring only. The MIND features are half
  title, half abstract. The abstract half has no human check. For the same reason the like-for-like neutral figure for MIND is the chosen rule on
  titles alone (52.4%), not the 39.0% of title + abstract. That title-only figure was added after the pre-declaration and is descriptive.
- **Domain gap, 2007 vs 2019:** SemEval headlines are 12 years older than MIND, from different outlets and with different topics. The classifier
  was trained on Reddit comments, so it is outside its home domain in both cases. These results support the scores on news headlines; they
  do not transfer perfectly to MIND.
- What it does not test: click behaviour, congruence, or features without a human rating (trust, go_dist_*). go_pct_* are percentile ranks of
  scores, so a single-label Spearman result carries over to them on headlines; features built on title + abstract are only partly covered.
- Accuracy is misleading here: 65.0% of test headlines are neutral at the official cutoff, so the all-neutral reference reaches 65.0% accuracy,
  more than any rule. Kappa corrects for this, and the all-neutral reference has kappa 0.
- The neutral level depends on the human cutoff as well: 65.0% at 50, 24.2% at 30, 16.0% at 25.

## 8. M2 and M3
- **M2:** all-mpnet-base-v2, pretrained and never fitted, so encoding dev items leaks nothing. Loaded through `transformers` with mean pooling and L2
  normalisation, the stack in the model's own config files, at a pinned revision (D50). User vector = mean of the last 50 history embeddings, score
  = cosine, cold start = 0. Four configs chosen on valid; title with mean aggregation won (D51, D57).
- **M3, late fusion:** final = alpha * norm(M2) + (1 - alpha) * norm(emotion), min-max within each impression, so the two scores share a scale
  without any label. Min-max uses only the candidates in the list. A constant score normalises to 0.
- **Emotion variants (D52):** cosine (primary), personal (cosine minus the corpus-mean profile's cosine), user-specific (minus the mean cosine with
  10 other users' fit profiles), mood repair (1 - cosine). The spec's title-only variant is reported but not eligible. Selection covers 4 x 11 = 44
  configs by valid AUC, and ties go to the primary variant and the alpha closest to M2 (D53).
- **Why valid overstates a selected model:** the best of 44 configs is optimistic on the data that chose it. The dev run with frozen settings is the claim.
- **Order:** M3 was frozen before M2 ran on dev, so no dev number existed while any choice was open (D57).
- **Checks (D54):** alpha = 1.0 reproduces M2 exactly on every metric; all configs score identical impressions; permuting the emotion score within each
  impression must not beat M2 at any alpha.
- **Test (D55):** paired t-test on per-impression AUC, all metrics with Holm, user-level robustness. dz is reported because a large sample makes tiny
  differences significant. With 73,152 impressions a gain of 0.003 is significant, so significance alone says little about usefulness.
- **Limits:** late fusion with one global alpha cannot tell M3 when emotion helps. `personal` is 0 for cold-start users, so those impressions keep M2's
  (tied) order. RQ5 conditions on the clicked item and uses labels of history items, so it describes rather than explains; emotion is linked to topic,
  so its category and profile rows may reflect topic.

## 9. M4 and M5
- **Why:** M2 and M3 ask whether emotion helps a content model. M4 and M5 ask the production question: does emotion still add anything to a model that already combines
  popularity, content and context? A gradient-boosted ranker is the usual way to do that, and it can learn category-specific effects from fit, which a single alpha cannot.
- **Model:** LightGBM LGBMRanker with lambdarank, one group per impression, deterministic mode, fixed thread count. M4 has 9 features: popularity, M1 and M2 scores, category,
  subcategory, user-category affinity, history length, title length and impression size. Position is never a feature; it is not even stored in the feature table. M5 adds the
  spec v4 emotion features (13 item + 4 congruence). M5-all adds all 19 item candidates instead, as a robustness check on valid only.
- **Leakage rule (D64):** a click-derived feature uses only days strictly before the row's day. Only popularity is click-derived. A fit row on day d uses fit days before d, valid uses all
  fit days, dev uses fit + valid. Day 1 gets a fixed constant 0.04, because a prior computed from fit would include the row's own impression, and a constant cannot rank within a list.
  Tests flip an impression's labels and prove that its own day and earlier days do not change, while later days do, so the test has power. On valid and dev the feature must also
  equal the saved M0 score. On fit, popularity is computed from fewer days than on valid, so training sees weaker popularity than scoring does: a known limit of any strictly-past feature.
- **Sampling:** training keeps all clicks and up to 4 non-clicks per click within each impression, seeded. Impression size is counted before sampling, so the feature means the same on all parts.
  Valid and dev always score all candidates. Lambdarank sees only pairs inside an impression, so sampling changes the pairs, not the scale of the problem.
- **Tuning:** the same 8 configs for M4, M5 and M5-all, up to 2000 rounds with patience 50. Early stopping uses mean per-impression AUC through a custom metric, because LightGBM's own AUC is
  global, not per impression. A fairness check reports M5 at M4's hyperparameters, so M5 cannot gain only from a different search result. The seed also drives the negative sampling, because
  without bagging LightGBM ignores its seed and three seeds would falsely show zero variability.
- **Controls:** a shuffled-emotion refit (D67d) shows what M5 can gain from noise through early stopping on valid. Both valid and dev are in the comparison tables, but valid chose the settings and the
  stopping round, so dev is the claim.
- **Importance:** gain and mean absolute contribution are computed on valid only. Features that carry the same information share credit, so a low rank means little unique use, not that the feature
  has no link to clicks.
- **Item identity, the main lesson of M5:** continuous item scores can act as an item ID. One emotion feature alone separates almost every item (6,099 distinct values on 6,144 valid items), so
  trees can learn which items were clicked on fit. That is not leakage, because fit clicks are past data, but it is not emotion either, and M4 has no such ID to compare with. The first shuffled
  control could not see this, because shuffling within an impression breaks the item-to-score link. The right control keeps identity and removes meaning: each item consistently takes another
  item's emotion vector (D69). On valid that control kept about half of the gain.
- **Why valid misled here:** valid is the day after fit, so 70.8% of its candidates were shown on fit days. Dev comes two days after fit, so only 19.4% were. Tuning and early stopping on valid therefore reward
  memorising items, and the reward disappears on dev. This affects any model with item-level continuous features and matters for M6 and M7. A system retrained daily might keep part of this gain,
  but that gain would be item popularity, not emotion.
- **What compares like with like:** on valid the controls are read against M4 at M5's hyperparameters and the same seed. On dev only M4 with its own hyperparameters exists, so the control rows there differ in
  hyperparameters and stopping round too. The dev comparison is also one seed, and its confidence interval covers impressions, not training variation.

