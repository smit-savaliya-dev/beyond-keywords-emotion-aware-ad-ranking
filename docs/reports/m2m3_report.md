# M2 and M3 report (2026-11-01)

Branch `models-m2m3` (local commits only, never pushed). 185 tests pass. Dev chose nothing. M2 and M3 each ran on dev once, and M0 and M1 did not run
again. MIND emotion was not re-scored. M4 and M5 were not started.

## Summary for the professor
1. Meaning-based matching works far better than word matching: Sentence-BERT reaches AUC 0.625 on the final test, against 0.574 for TF-IDF and 0.589 for
   popularity.
2. Adding the emotional context inferred from the user's recent reading improves AUC by 0.003 on the final test, and the gain held up from the
   validation day (0.003 there too).
3. That gain is real but very small (effect size 0.03), and it appears in AUC only: no top-of-list metric improved significantly on the final test.
4. People tend to click items that match their usual emotional mix, not items that counter it: the "mood repair" version never beat the plain model.
5. The gain is uneven: it is largest for users with long histories and for news and music clicks, and it is negative for sports and finance. These are
   descriptions, not causes, and emotion is linked to topic.

## M2 grid (valid)
| Text | Aggregation | AUC | MRR | nDCG@10 |
|---|---|---|---|---|
| **Title** | **Mean (chosen)** | **0.6468** | **0.2998** | **0.3807** |
| Title | Decay 0.9 | 0.6396 | 0.2940 | 0.3749 |
| Title + abstract | Mean | 0.6449 | 0.2999 | 0.3819 |
| Title + abstract | Decay 0.9 | 0.6392 | 0.2949 | 0.3759 |

M2 vs M1 on valid: AUC +0.0568 (95% CI 0.0531, 0.0606), dz 0.171, all 10 metrics significant with Holm. Mean beats decay, in line with the earlier finding that
long-term reading matters more than recency.

## M3 grid (valid AUC)
Chosen by D53: **personal, alpha 0.8** (frozen:M3_sbert_emotion:5abb38af3d). M2 alone is 0.6468.

| Alpha | Cosine (primary) | Personal | User-specific | Mood repair | Title-only (not eligible) |
|---|---|---|---|---|---|
| 0.0 | 0.5450 | 0.5668 | 0.5504 | 0.4550 | 0.5474 |
| 0.2 | 0.5776 | 0.5920 | 0.5773 | 0.4977 | 0.5804 |
| 0.4 | 0.6087 | 0.6184 | 0.6060 | 0.5555 | 0.6100 |
| 0.5 | 0.6246 | 0.6325 | 0.6226 | 0.5881 | 0.6254 |
| 0.6 | 0.6370 | 0.6437 | 0.6359 | 0.6159 | 0.6372 |
| 0.7 | 0.6433 | 0.6483 | 0.6428 | 0.6316 | 0.6431 |
| 0.8 | 0.6465 | **0.6499** | 0.6463 | 0.6402 | 0.6465 |
| 0.9 | 0.6475 | 0.6493 | 0.6471 | 0.6449 | 0.6474 |
| 1.0 | 0.6468 | 0.6468 | 0.6468 | 0.6468 | 0.6468 |

Full grid with all metrics and alphas: `results/tables/m3_fusion_grid_valid.csv`.

**Alpha curve:** every line rises towards M2 and flattens near alpha 0.8 to 0.9, so emotion helps only as a small correction. Emotion alone scores 0.545 to 0.567
(about what popularity scores on valid, 0.574, at best); M2 carries most of the ranking. Personal is the only variant clearly above M2. Cosine gains 0.0007 at best, and
user-specific never gets above M2 by more than 0.0003 at any alpha shown. Mood repair is below M2 at every alpha below 1 and below 0.5 when alone, so
the opposite of the match does not predict clicks. Title-only tracks cosine closely. Chart: `results/figures/diagnostics/m3_alpha_curve.png`, in
diagnostics because replacing a story chart would have changed existing files.

## M3 vs M2
| Part | AUC M2 | AUC M3 | Difference (95% CI) | dz | User-level (95% CI) | Metrics significant (Holm) |
|---|---|---|---|---|---|---|
| Valid | 0.6468 | 0.6499 | +0.0031 (0.0022, 0.0040) | 0.038 | +0.0030 (0.0020, 0.0040) | 7 of 10 |
| **Dev** | **0.6253** | **0.6282** | **+0.0029 (0.0023, 0.0035)** | **0.034** | **+0.0028 (0.0021, 0.0035)** | **1 of 10** |

Other dev metrics: MRR +0.0011 (Holm p 0.059), nDCG@10 +0.0010 (0.064), nDCG@5 +0.0008 (0.516); precision, recall and hit rate at 5 and 10 are all under +0.001
and not significant.

**RQ3 verdict (D55):** supported on the pre-declared primary metric, because the dev AUC gain is positive with Holm p < 0.05. Read with its size: dz 0.034 is far below
the 0.1 usually called small, and it does not reach the top of the list. With 73,152 impressions, significance alone says little. On valid, 7 of 10 metrics passed
but valid is the selection set, so dev is the claim.

## Comparison with M0 and M1 (dev)
| Model | AUC | MRR | nDCG@10 | HR@10 |
|---|---|---|---|---|
| M0 popularity | 0.5890 | 0.2749 | 0.3506 | 0.6599 |
| M1 TF-IDF | 0.5744 | 0.2759 | 0.3554 | 0.6710 |
| M2 Sentence-BERT | 0.6253 | 0.2979 | 0.3850 | 0.7116 |
| M3 M2 + emotion | 0.6282 | 0.2990 | 0.3860 | 0.7124 |

M3 vs popularity +0.0392 (0.0364, 0.0420), dz 0.101. M3 vs TF-IDF +0.0538 (0.0512, 0.0564), dz 0.150. M2 vs popularity +0.0363 (0.0335, 0.0391); M2 vs TF-IDF +0.0509
(0.0483, 0.0535). All 10 metrics are significant in these four pairs. The bar of popularity is cleared, by M2 alone: almost all of that margin comes from the content model.

## Where emotion helps, RQ5 (dev, descriptive, M3 minus M2 AUC)
| Breakdown | Group | Impressions | Mean difference (95% CI) | dz |
|---|---|---|---|---|
| Clicked category | News | 17,230 | +0.0161 (0.0148, 0.0175) | 0.179 |
| | Sports | 18,009 | -0.0083 (-0.0096, -0.0069) | -0.091 |
| | Lifestyle | 13,008 | -0.0012 (-0.0025, 0.0001) | -0.016 |
| | Finance | 2,416 | -0.0089 (-0.0123, -0.0055) | -0.104 |
| | Music | 2,575 | +0.0203 (0.0166, 0.0241) | 0.210 |
| | TV | 3,528 | -0.0031 (-0.0054, -0.0008) | -0.045 |
| | Other | 16,386 | +0.0048 (0.0036, 0.0060) | 0.061 |
| Dominant profile emotion | Joy | 29,916 | -0.0002 (-0.0011, 0.0008) | -0.002 |
| | Sadness | 29,612 | +0.0065 (0.0055, 0.0075) | 0.074 |
| | Surprise | 8,800 | +0.0023 (0.0005, 0.0041) | 0.027 |
| | Anger | 1,207 | -0.0001 (-0.0050, 0.0048) | -0.001 |
| | Fear | 717 | +0.0061 (-0.0007, 0.0130) | 0.065 |
| | Neutral | 686 | -0.0006 (-0.0075, 0.0063) | -0.007 |
| | Cold start | 2,214 | 0.0000 | 0.000 |
| Known history items | 0 | 2,214 | 0.0000 | 0.000 |
| | 1-5 | 10,768 | +0.0021 (0.0003, 0.0038) | 0.022 |
| | 6-20 | 25,435 | +0.0018 (0.0007, 0.0028) | 0.020 |
| | 21+ | 34,735 | +0.0042 (0.0033, 0.0051) | 0.050 |

- Inside clicked categories the effects are larger than the overall gain and have opposite signs, so the 0.003 average hides real variation (Q12).
- 68% of the total gain comes from users with 21+ known history items. Cold-start impressions change by exactly 0, because `personal` is constant
  without history, so the gain does not come from them.
- Users whose profile leans to sadness gain; joy-leaning users do not. These are often news readers, so this row may be topic, not emotion.
- Conditioned on the clicked item, and emotion is linked to topic: none of this is causal, and no setting was changed from it.

**Examples** (`m3_where_it_helps_examples_dev.csv`, lists with 10+ candidates; anecdotes only). Helped: a user who read several crime and death stories clicked a
stabbing story (AUC 0.18 to 0.73), a clean tone match. Hurt: lists where a user with mixed or tense reading clicked a light item such as a stately-home yard sale or
Black Friday deals (0.70 to 0.20, 0.82 to 0.36), which matching the emotional mix pushed down. Some large swings are not obviously emotional, so these are
for discussion, not evidence.

## Checks
| Check | Result |
|---|---|
| Alpha 1.0 reproduces M2 (D54a) | Pass: identical metrics in all 4 variants, max difference 0.0 |
| Identical impressions in all configs (D54b) | Pass |
| Shuffled emotion does not beat M2 (D54c) | Pass: best mean difference -0.0009 (alpha 0.9), none positive |
| One dev run per model | Pass: 4 `final-dev` rows, M0 to M3 |
| Dev files unchanged by the rebuild | Pass: 8 of 8 hashes match |
| `--rebuild` | Exit 0, GPU steps held. M2 and M3 valid outputs identical to before, frozen hashes unchanged |
| Snapshot | 75 of 76 identical. `summary.csv` differs only by the 4 new M2 and M3 rows; its 4 original rows match with difference 0 |
| Tests | 185 pass |

## Spec note
Variants 2 and 3 were your instruction, not spec v3, and spec v3 lists user-specific scores as analysis controls (D52). I followed the instruction and left the
spec file unchanged. This does not affect the result: the chosen variant, personal, is in the spec's congruence features. The spec's title-only variant is reported but
was not eligible. The spec's `emotion_score` still names cosine, so M7 should decide this explicitly.

## Issues fixed
1. MY BUG: `profile_cosine` reused the 500,000-row emotion chunk for 768-dimension vectors, needing about 3 GB per array, so the first rebuild stopped at step 44 with a
   memory error. Chunk is now 50,000; M2 valid scores are identical to before (difference 0.0).
2. `sentence-transformers` is not installed. The project avoids new heavy dependencies, so the same stack runs through `transformers`, pinned to a model revision (D50).
3. The alpha chart's legend overlapped its caption. Fixed and checked.
4. The laptop was closed during the second rebuild. The same process paused and resumed, it was not restarted, so the run still counts as one rebuild.
5. Order tightened: M3 was frozen before the M2 dev run, so no dev number existed while any choice was open (D57).

## Open questions
Q12: a category-aware alpha is tempting but would use dev patterns; not done. Q13: is an AUC-only gain enough for RQ3 in the dissertation? Reported as stated in D55.
