# Guide to the charts

The eight story charts are in `results/figures/story/` and use one style. A ninth, on external validation, is at the end. On charts 6, 7 and 8, the blue circle is fit (9-13 Nov) and
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

## External validation (`results/figures/diagnostics/semeval2007_correlations.png`)
The story folder already holds 8 charts, so this one sits in diagnostics.
- **Shows:** how well our emotion scores track human ratings on the 1,000 SemEval-2007 test headlines, with 95% intervals. Blue circles
  are GoEmotions group scores, green triangles single GoEmotions labels, orange squares NRC EmoLex. The bottom row is valence, where blue
  is go_vad_valence and orange is NRC VAD. The dashed line is the pre-declared minimum of 0.2. Dev and MIND clicks are not involved.
- **Why it matters:** this is the only human check of the scores. GoEmotions passes on anger, fear, joy, sadness and valence, and beats NRC
  everywhere except fear, where the intervals overlap. There is no group score for disgust, because disgust is part of the anger group.
- **Read with care:** the data are headlines from 2007, so this tests the title scoring, not the abstracts. Surprise is weak for every source.

