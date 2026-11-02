# Guide to the charts

The eight story charts are in `results/figures/story/` and use one style. The six results charts for the supervisor are at the end. On charts 6, 7 and 8, the blue circle is fit (9-13 Nov) and
the orange square is valid (14 Nov, the next day). "dz" is an effect size: how far clicked items differ from unclicked
items in the same list, in standard deviations. 0 means no difference, 0.1-0.2 is small but real. A hollow marker means the
result was not significant at both the impression level and the user level. Dev data was never used.

## 1. Emotion mix, old vs new (`1_emotion_mix.png`)
- **Shows:** the share of each emotion label under four rules: the original combined text, v1, v2, and v2b (v2 without the
  stance labels approval and disapproval).
- **Why it matters:** neutral falls from 81% to 26% in v2, or 39% in v2b. The v2b bar is the more careful figure, because
  approval and disapproval express an opinion, not an emotion. The SemEval-2007 selection kept v2b (D47, D49).
- **Read with care:** the classifier did not change. Only the rule for what counts as emotion changed. For 91.6% of the
  items that left neutral, the classifier's own neutral score is still above 0.5.

## 2. Neutral share vs threshold (`2_neutral_vs_threshold.png`)
- **Shows:** how the neutral share changes with the threshold t, one line per title weight. The ring marks the chosen setting.
- **Why it matters:** the threshold moves neutral far more than the title weight. Neutral is largely a choice of cut-off, not
  a fixed fact about the news. The dashed line (v2b) sits higher because stance labels no longer count as emotion.

## 3. Title vs abstract intensity (`3_title_vs_abstract_intensity.png`)
- **Shows:** for the same items, how many titles and abstracts reach each level of emotion score.
- **Why it matters:** abstracts reach higher scores (median 0.116 vs 0.068), probably because longer text gives the
  classifier more to pick up. That does not make abstracts more useful for clicks: see `docs/reports/base_v2_report.md`.

## 4. Emotion by category (`4_emotion_by_category.png`)
- **Shows:** the emotion mix inside the 10 largest categories. Each row adds up to 100%.
- **Why it matters:** emotion is linked to topic. Food and drink is 58% joy; news is the only category where sadness leads
  (28%). This is why congruence might partly be topic matching, which M2 vs M3 will test.
- **Read with care:** "joy" includes approval and admiration, so it is broader than the everyday word.

## 5. Valence-arousal map (`5_valence_arousal_map.png`)
- **Shows:** the average position of each category on negative-to-positive (valence) and calm-to-excited (arousal). The bars
  show where the middle half of its items fall.
- **Why it matters:** categories differ on average (news is the most negative at -0.05, food and drink the most positive at
  0.41), but the long bars show that items inside one category vary far more than the categories differ from each other.

## 6. Clicked vs not clicked effects (`6_clicked_vs_not_effects.png`)
- **Shows:** for 20 emotion features, how different clicked items are from unclicked items, on fit and on the next day.
- **Why it matters:** 7 of 20 features flip sign between the two periods (marked "sign flips"). Fear is stable; overall
  emotionality and intensity are not. There is no simple "emotion that always gets clicks".

## 7. Congruence effects (`7_congruence_effects.png`)
- **Shows:** how much closer clicked items are to the emotional context inferred from the user's recent reading.
  Raw is the plain match. Personal removes the average news item. User-specific removes what a random other user's history
  also predicts, so it is the strictest version.
- **Why it matters:** this is the core of the research question. The effect appears on both periods, and the strict version
  is 0.138 and 0.137, almost identical. It is an association, not yet proof that ranking improves.

## 8. Short-term vs long-term context (`8_short_vs_long_context.png`)
- **Shows (left):** the user-specific effect for the last 1, 3, 5 and 50 items of reading history.
- **Shows (right):** the last 5 items against the 5 just before them, so both profiles have the same size.
- **Why it matters:** longer history is stronger, but the right panel shows newer items are not better than slightly older
  ones. So the gain does not come from recency: it looks like a stable preference, measured better with more items.
- **Read with care:** the history list stops before the log period, so "last 1" is not the click just before the impression.

# Results charts for the supervisor

All six are in `results/figures/results/`. Every caption is computed from the saved tables, so its numbers always match the results. Bars around a
dot are 95% confidence intervals: if a bar stays clear of zero, the difference is unlikely to be chance. Dev is the final test, run once per model.

## 1. Model ladder (`1_model_ladder.png`)
- **Shows:** final test AUC for each model, with or without emotion. AUC is the chance that a clicked item is ranked above an unclicked one in the same
  list; 0.5 is random.
- **Why it matters:** this is the whole project on one chart. New models appear on it automatically.
- **Read with care:** each model's bar shows the uncertainty of that model alone. To compare two models use the paired tests, which are far more precise.

## 2. Gain from emotion per metric (`2_m3_gain_by_metric.png`)
- **Shows:** M3 minus M2 on all 10 metrics, on valid and on dev. A hollow marker means not significant after Holm correction.
- **Why it matters:** AUC measures the whole list, while the other metrics measure the top of it. Emotion improves the first, and on dev not clearly the second.
- **Read with care:** a hollow marker can have a bar that clears zero, because Holm is stricter than a single interval: it allows for testing 10 metrics at once. Valid is
  the set that chose M3, so it is expected to look better than dev.

## 3. Alpha curve (`3_alpha_ablation.png`)
- **Shows:** valid AUC as the blend moves from emotion alone (alpha 0) to M2 alone (alpha 1), one line per emotion score. The dashed line is M2.
- **Why it matters:** only the personal score rises clearly above M2, and only near alpha 0.8. Mood repair stays below M2. Emotion works as a small correction, not as a ranker.
- **Read with care:** this is the set that chose the setting, so the peak is optimistic.

## 4. Where emotion helps, by category (`4_m3_gain_by_category.png`)
- **Shows:** M3 minus M2 AUC on dev, split by the category of the clicked item.
- **Why it matters:** the small average hides larger effects with opposite signs: up for music and news, down for sports and finance.
- **Read with care:** descriptive only. Emotion is linked to topic, and the split uses the click itself, so this does not show why. Nothing was chosen from it.

## 5. Gain by history length (`5_m3_gain_by_history.png`)
- **Shows:** the same gain, split by how many history items the user has.
- **Why it matters:** the gain is largest with 21 or more items, where the emotional profile is best measured. Without history it is exactly 0, because the personal score is constant.
- **Read with care:** descriptive only, and the bars of the two middle groups overlap.

## 6. Emotion validation (`6_emotion_validation.png`)
- **Shows:** how well our emotion scores track human ratings on 1,000 SemEval-2007 news headlines, for GoEmotions and NRC EmoLex. The dashed line is the pre-declared
  minimum of 0.2.
- **Why it matters:** this is the only human check of the scores. GoEmotions passes on anger, fear, joy, sadness and valence, and beats NRC everywhere except fear, where the
  intervals overlap. There is no group score for disgust, because disgust is part of the anger group.
- **Read with care:** the data are headlines from 2007, so this tests the title scoring, not the abstracts. Surprise is weak for every source.

