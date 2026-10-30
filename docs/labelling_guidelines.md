# Labelling guidelines (6 classes)

You label `annotation/manual_sample.csv` (180 MIND news items) with `python scripts/label_tool.py`.
The tool shows the title and abstract only. Never look at model output before or while labelling.

## The six classes
| Key | Class | Use when the text conveys... |
|---|---|---|
| 1 | joy | happiness, success, celebration, pride, love, relief, amusement, hope or excitement about something good |
| 2 | sadness | loss, death, grief, disappointment, regret, loneliness |
| 3 | anger | outrage, blame, conflict, disgust, injustice, frustration or annoyance |
| 4 | fear | threat, danger, risk, worry, anxiety, alarm about what may happen |
| 5 | surprise | something unexpected, shocking or astonishing (positive or negative), or strong curiosity ("you won't believe") |
| 6 | neutral | factual information with no clear emotion |

## Rules
1. **Label what the text conveys, not your own reaction.** "Stock market closes 0.3% lower" is neutral even if you
   own shares. "Family devastated after house fire" is sadness even if you feel nothing about it.
2. **Neutral when there is no clear emotion.** A topic that is often emotional is not enough: "City council votes on
   police budget" is neutral; "Furious residents demand police budget cuts" is anger. Recipes, schedules, product
   lists, scores without emotional framing and how-to articles are usually neutral.
3. **If mixed, choose the strongest emotion.** If two are equally strong, use the one in the title.
4. **Use the title and abstract together.** If they disagree, label the overall impression of both.
5. **Do not overthink.** Aim for 10-20 seconds per item; use `s` to skip a hard item and come back to it at the end.
6. **Be consistent.** Read these rules again before each session. Do not change your interpretation halfway through.

## Re-labelling (intra-rater reliability)
After finishing all 180, run `python scripts/label_tool.py --make-relabel`. It creates `annotation/relabel_sample.csv`
with 30 random items (seeded). Label it with `python scripts/label_tool.py --relabel` **at least one day later**,
without looking at your first labels. The agreement between your two passes (intra-rater kappa) is reported by
`scripts/38_analysis_label_validation.py`.
