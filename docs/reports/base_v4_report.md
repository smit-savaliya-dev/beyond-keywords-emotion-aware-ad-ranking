# Base v4 report (2026-10-30)

Branch `base-v4` (local commits only, never pushed). Dev was not loaded or run: the only dev runs are still the two from 26 October. MIND was
not re-scored on the GPU. No model was built.

## Summary for the professor
1. Hand labelling was replaced by a published human benchmark: 1,000 news headlines rated by people (SemEval-2007).
2. Our emotion scores follow human ratings well: 0.72 for positive vs negative tone, and 0.45 to 0.60 for anger, fear, joy and sadness.
   Surprise is weak (0.19).
3. The classifier is clearly better than the word-list method, except on fear, where the two are close.
4. Exact emotion labels agree with humans only fairly (kappa 0.30), so model features stay as soft scores and labels stay for reporting.
5. This check covers headlines from 2007, not abstracts, and it says nothing about clicks. Whether emotion improves ranking is still open
   until M2 vs M3.

## Data
| Item | Count |
|---|---|
| SemEval-2007 Task 14 trial, selection only | 250 headlines |
| SemEval-2007 Task 14 test, reporting only | 1,000 headlines |
| Emotion and valence gold ids matching the headlines | all |

Citation: Strapparava, C. and Mihalcea, R. (2007). SemEval-2007 Task 14: Affective Text. Proceedings of SemEval-2007.
The README states the coarse-grained cutoff of 50, so D46 used it.

## Threshold-free correlations (test, primary)
Spearman with 95% bootstrap CI; Pearson in `results/tables/semeval2007_validation.csv`.

| Gold rating | GoEmotions group | GoEmotions single label | NRC |
|---|---|---|---|
| Anger | 0.523 (0.476, 0.563) | 0.521 (0.472, 0.564) | 0.313 (0.256, 0.368) |
| Disgust | no group | 0.377 (0.319, 0.432) | 0.168 (0.100, 0.239) |
| Fear | 0.457 (0.405, 0.507) | 0.488 (0.438, 0.540) | 0.407 (0.347, 0.462) |
| Joy | 0.452 (0.398, 0.505) | 0.368 (0.310, 0.426) | 0.219 (0.158, 0.280) |
| Sadness | 0.595 (0.548, 0.638) | 0.618 (0.574, 0.659) | 0.333 (0.283, 0.384) |
| Surprise | 0.192 (0.131, 0.251) | 0.090 (0.029, 0.155) | 0.019 (-0.044, 0.080) |
| Valence | go_vad 0.721 (0.684, 0.753) | | NRC VAD 0.501 (0.454, 0.547) |

Grouping helps joy and surprise most. NRC has an emotion word in 65.4% of test headlines, and VAD covers 99.7%. Gold ratings are mostly near
zero, so Spearman is well above Pearson for some emotions (anger 0.52 vs 0.33).

## Labelling-rule choice and kappa on test
Six rules were eligible on trial (MIND neutral at or below 40%). v2b at t 0.05 had the highest trial kappa, so it was chosen without needing the
overlap rule. Its trial interval was wide, so the trial set could not separate the rules. Test results at the official cutoff:

| Rule | Kappa (95% CI) | Accuracy | Macro-F1 | Neutral predicted |
|---|---|---|---|---|
| v1 | 0.216 (0.163, 0.270) | 0.646 | 0.323 | 81.9% |
| v2, t 0.05 | 0.276 (0.231, 0.320) | 0.542 | 0.375 | 49.7% |
| **v2b, t 0.05 (chosen)** | **0.296 (0.246, 0.343)** | **0.599** | **0.393** | **61.1%** |
| v2c, t 0.05 | 0.304 (0.255, 0.354) | 0.609 | 0.398 | 62.3% |
| NRC EmoLex | 0.121 (0.087, 0.158) | 0.428 | 0.229 | 48.3% |
| All neutral (reference) | 0.000 | 0.650 | 0.131 | 100% |

Humans: 65.0% neutral. Kappa is "fair" for all GoEmotions rules. All-neutral has the best accuracy, which is why accuracy should not be read
alone here.

Per-class recall, v2b at t 0.05 (gold items in brackets): joy 0.45 (110), sadness 0.73 (100), anger 0.24 (25), fear 0.20 (74), surprise 0.15 (41),
neutral 0.69 (650). No class falls under the 20-item flag; anger is the smallest. Excluding approval and disapproval lowers joy recall from
0.65 to 0.45 and raises neutral recall from 0.57 to 0.69. Confusion matrices are in `semeval2007_validation_meta.json`.

Sensitivity to the human cutoff: at 25 and 30, v2 leads (0.263 and 0.281) over v2b (0.225 and 0.244) with overlapping intervals, and v1 is clearly
worse (0.120 and 0.136). So v2, v2b and v2c are not separated, while the old v1 rule falls behind at lower cutoffs (at 50 its interval
still touches v2b's). v2b stays because it was pre-selected on trial; test results did not change it (Q11).

## Verdict
- **Which rule agrees best:** v2c is highest at the official cutoff by 0.008, and v2 at the lower cutoffs, but no difference is resolved.
  v2b stays the chosen rule.
- **Is the MIND neutral level supported?** Partly. On headlines the chosen rule gives 61.1% neutral against 65.0% for humans. On MIND titles
  alone it gives 52.4%. The 39.0% for title + abstract has no human check, because SemEval has no abstracts. The human figure itself moves with
  the cutoff: 16.0% at 25, 24.2% at 30, 65.0% at 50. Matching shares does not mean matching items, since kappa stays fair.
- **Spec check (D48):** **CONFIRM** for GoEmotions: valence 0.721, and all 4 main emotions pass. NRC also gives CONFIRM (0.501; 0.313, 0.407, 0.219, 0.333).
  The bar of 0.2 is modest; the point is that both pass, and GoEmotions by a wide margin.
- **Spec files:** the chosen rule differs from the spec v2 labelling, so D47 required `config/emotion_spec_v3.json`. Only labelling and the
  confirmation rule changed; a new test enforces that features and fusion match v2. Spec v2 changed only in its unhashed status fields.

## What changed
- Hand-label sheet (hash unchanged, 0 of 180 labelled), tool, guidelines and tests moved to `results/archive/annotation/`. Nothing active depends
  on `annotation/`. Script 27 reads the archived ids, so its example table is unchanged.
- New: `src/semeval.py`, `src/agreement.py` (statistics kept from `src/annotation.py`), scripts 14, 29, 38, spec v3, a SemEval section in
  `show_results.py`, one chart in `results/figures/diagnostics/` (the story folder already holds 8).
- D45-D48 were committed before scoring; D49 records the results. Q04, Q06, Q08, Q09 closed; Q10, Q11 opened.
- `--rebuild`: exit 0, GPU steps held, 76 of 76 snapshot files identical. 164 tests pass.

## Issues fixed
1. Python could not verify the university server's certificate chain. The download uses `curl`, which verifies it through Windows, plus a
   pinned file hash. Verification was never turned off.
2. Gold scores came back as a read-only array, which a test caught.
3. The constant-input check used std == 0, which float noise can miss. It now uses the value range.
4. MY NEAR-MISS: comparing 39.0% on MIND with the human 65.0% was not like for like, because SemEval is headline-only. Added the title-only figure
   52.4%, labelled descriptive because it came after the pre-declaration.
5. The rule selection exposed that spec v2 labels with plain v2, not v2b, so a status-only update was not allowed. Spec v3 was written.

## Limits
- Title scoring only; the abstract half of MIND features has no human check (Q10).
- 2007 vs 2019 headlines, different outlets and topics.
- Tests measurement, not clicks, congruence or ranking. Trust and go_dist_* have no human rating here.
- No direct human labels on MIND headlines (D45).

## Open questions
Q10 (abstracts unchecked; accepted as a stated limitation) and Q11 (v2, v2b, v2c are not separated; v2b kept). See `docs/open_questions.md`.
