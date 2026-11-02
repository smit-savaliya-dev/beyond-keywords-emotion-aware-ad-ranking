# M4 and M5 report (2026-11-02)

Branch `results-m4m5` (local commits only, never pushed). 219 tests pass. Dev chose nothing. M4, M5 and two pre-declared controls each ran on dev once; M0 to M3 did not run again.
MIND emotion was not re-scored. NRMS (M6, M7) was not started.

## Summary for the professor
1. A boosted ranker that combines popularity, content and context is the best model so far: AUC 0.674 on the final test, against 0.628 for M3 and 0.589 for popularity.
2. Adding emotion to it did not help on the final test: AUC 0.671 against 0.674. The pre-declared question is answered no.
3. On the validation day emotion looked like a clear win (+0.014), but that was misleading. Most of it disappeared when emotion scores were replaced by another item's scores, so the model was mainly
   recognising individual news items, not using emotion.
4. This matters for how the study reads: validation shared 71% of its items with training, while the final test shared only 19%, so memorising items paid off on one and not the other.
5. Across the project, the only positive emotion result on the final test is still the small M3 gain (+0.003). Emotional context inferred from recent reading adds at most a little beyond content
   and popularity.

## Encoder check
Our transformers-based all-mpnet-base-v2 embeddings against the sentence-transformers package, 500 random items per text variant:

| Text | Min cosine | Mean cosine |
|---|---|---|
| Title | 0.9999999999992 | 0.9999999999999 |
| Title + abstract | 0.9999999999988 | 0.9999999999999 |

Threshold 0.999: pass. Nothing was re-encoded and no dev run was repeated (D61).

## The 6 results charts (`results/figures/results/`)
| Chart | Takeaway |
|---|---|
| 1 Model ladder | M4 is the highest at 0.674; each step up comes from content and popularity, not from emotion. |
| 2 M3 gain per metric | M3 gains +0.0029 AUC on dev; 1 of 10 metrics survives Holm on dev, 7 of 10 on valid. |
| 3 Alpha curve | Only the personal score rises clearly above M2 (0.6499 vs 0.6468), and only near alpha 0.8. |
| 4 M3 gain by category | Descriptive: from +0.0203 for music to -0.0089 for finance, around +0.0029 overall. |
| 5 M3 gain by history | Descriptive: +0.0042 with 21+ items, +0.0021 with 1-5, exactly 0 without history. |
| 6 Emotion validation | Spearman with human ratings: valence 0.72 for GoEmotions, 0.50 for NRC; main emotions 0.45 to 0.59. |

The alpha curve and the SemEval chart moved here from diagnostics, so only one copy exists. Explanations are in `docs/figures_guide.md`.

## M4 and M5 grids (valid AUC, best round in brackets)
| Leaves | Rate | Min child | M4 | M5 | M5-all |
|---|---|---|---|---|---|
| 31 | 0.05 | 50 | **0.6745** (149) | 0.6875 (154) | 0.6852 (172) |
| 31 | 0.05 | 200 | 0.6729 (87) | 0.6877 (173) | 0.6837 (60) |
| 31 | 0.10 | 50 | 0.6727 (35) | 0.6855 (67) | 0.6868 (95) |
| 31 | 0.10 | 200 | 0.6734 (116) | 0.6870 (73) | 0.6826 (84) |
| 63 | 0.05 | 50 | 0.6744 (134) | 0.6866 (97) | 0.6853 (103) |
| 63 | 0.05 | 200 | 0.6742 (105) | **0.6887** (81) | **0.6881** (164) |
| 63 | 0.10 | 50 | 0.6733 (35) | 0.6852 (54) | 0.6839 (71) |
| 63 | 0.10 | 200 | 0.6708 (25) | 0.6841 (77) | 0.6836 (36) |

Each fit used 899,931 sampled training rows. All other metrics are in `results/tables/m4m5_tuning_valid.csv`.

**Chosen:** M4 31 leaves, rate 0.05, min child 50, 149 rounds (frozen:M4_lightgbm:11ba13ceeb). M5 63 leaves, rate 0.05, min child 200, 81 rounds (frozen:M5_lightgbm_emotion:74aeee3004). M5-all the same config as M5, 164
rounds, valid only. **Fairness check:** M5 at M4's hyperparameters reaches 0.6875, +0.0130 (0.0108, 0.0152) on valid, so M5's valid gain did not come from a different search result.

## Seed variability (valid AUC)
| Model | Seed 42 | Seed 43 | Seed 44 | Spread | Rounds |
|---|---|---|---|---|---|
| M4 | 0.6745 | 0.6746 | 0.6782 | 0.0037 | 149, 49, 54 |
| M5 | 0.6887 | 0.6886 | 0.6877 | 0.0010 | 81, 109, 108 |
| M5-all | 0.6881 | 0.6888 | 0.6854 | 0.0034 | 164, 195, 96 |

The seed drives the negative sampling. M4's spread (0.0037) is larger than the M5 - M4 difference on dev (0.0030), and the confidence intervals below cover impressions, not training variation. So read the dev result as "no gain", not as "emotion hurts".

## Results with 95% CI
| Comparison | Part | AUC difference | dz | User-level | Metrics significant |
|---|---|---|---|---|---|
| **M5 - M4 (primary)** | Valid | +0.0142 (0.0121, 0.0163) | 0.076 | +0.0140 (0.0116, 0.0164) | 10 of 10 |
| **M5 - M4 (primary)** | **Dev** | **-0.0030 (-0.0042, -0.0018)** | **-0.017** | **-0.0034 (-0.0048, -0.0020)** | **5 of 10, all negative** |
| M5 - M3 | Valid | +0.0388 (0.0360, 0.0417) | 0.154 | +0.0380 (0.0348, 0.0413) | 10 of 10 |
| M5 - M3 | Dev | +0.0427 (0.0409, 0.0446) | 0.169 | +0.0422 (0.0401, 0.0442) | 10 of 10 |
| M4 - M2 | Valid | +0.0277 (0.0247, 0.0307) | 0.105 | +0.0270 (0.0237, 0.0304) | 10 of 10 |
| M4 - M2 | Dev | +0.0486 (0.0468, 0.0505) | 0.193 | +0.0484 (0.0463, 0.0504) | 10 of 10 |
| M5-all - M5 | Valid | -0.0006 (-0.0020, 0.0007) | -0.005 | -0.0008 (-0.0024, 0.0007) | 0 of 10 |

M5 - M4 on dev, other metrics: nDCG@5 -0.0021, P@5 -0.0008, R@5 -0.0048, HR@5 -0.0037 are significantly negative; MRR -0.0009 and nDCG@10 -0.0011 are not significant.

| Model | Dev AUC | MRR | nDCG@10 | HR@10 |
|---|---|---|---|---|
| M0 popularity | 0.5890 | 0.2749 | 0.3506 | 0.6599 |
| M1 TF-IDF | 0.5744 | 0.2759 | 0.3554 | 0.6710 |
| M2 Sentence-BERT | 0.6253 | 0.2979 | 0.3850 | 0.7116 |
| M3 = M2 + emotion | 0.6282 | 0.2990 | 0.3860 | 0.7124 |
| **M4 LightGBM** | **0.6739** | **0.3245** | **0.4182** | **0.7509** |
| M5 = M4 + emotion | 0.6709 | 0.3236 | 0.4172 | 0.7494 |

**Verdict (D68):** the production-paradigm question is **not supported**. M5 - M4 on dev is negative. M4 > M2 held on valid and dev, and M4 stays below the leakage limit.

## Why valid and dev disagree
Valid M5 - M4 was five times the M3 gain, so before any dev run I tested why. One emotion feature alone separates almost every item (6,099 distinct values on 6,144 valid items), so trees can learn item identity from fit clicks. This
was not in your brief. I added the controls below before any M4 or M5 dev run (D69), and D68 stayed the verdict rule.

**Gain vs M4 at the same hyperparameters, valid, mean of 3 seeds:**
| Model | Gain (range) | What it tests |
|---|---|---|
| M5, real emotion | +0.0141 (0.0136 to 0.0146) | Reference |
| M4 + item emotion features only | +0.0130 (0.0124 to 0.0136) | Items that can act as an ID |
| Swapped vectors, meaning destroyed | +0.0076 (0.0050 to 0.0100) | Identity without emotion |
| M4 + congruence only | +0.0021 (0.0015 to 0.0024) | Cannot fingerprint an item alone |

**M5 - M4 AUC by whether the candidate was shown on fit days (pre-declared):**
| Part | Candidates shown in fit | On those | On unseen candidates |
|---|---|---|---|
| Valid | 70.8% | +0.0145 (0.0118, 0.0172) | +0.0013 (-0.0019, 0.0045) |
| Dev | 19.4% | -0.0016 (-0.0041, 0.0008) | -0.0035 (-0.0049, -0.0022) |

On valid, the whole gain sits on items the model had already seen. Dev has few of them.

**Controls on dev (one seed):**
| Comparison | AUC difference | Metrics significant |
|---|---|---|
| M5 - swapped vectors | +0.0044 (0.0029, 0.0058) | 2 of 10 (AUC, nDCG@10) |
| Swapped vectors - M4 | -0.0074 (-0.0085, -0.0062) | 6 of 10, all negative |
| Congruence only - M4 | -0.0030 (-0.0043, -0.0018) | 4 of 10, mixed |

- The D69 rule is met in the letter: real emotion vectors beat meaningless ones, so emotional meaning carries some information. But there is no gain over M4 to credit to it, so this says emotion is less harmful than noise, not that it helps.
- Congruence only is mixed: AUC falls, while P@5 (+0.0010), P@10 (+0.0005) and R@10 (+0.0027) rise significantly. This is not like for like, because it stopped at 29 rounds with other hyperparameters than M4's 149 (Q14).
- My first control, the shuffled-emotion refit, passed (shuffled - M4 -0.0001, CI -0.0012 to 0.0010) but could not see this problem, because shuffling within an impression breaks the item-to-score link. It was the wrong control for this risk.

## Feature importance (M5, valid only)
| Family | Features | Share of mean contribution | Share of gain |
|---|---|---|---|
| Content | 5 | 58.6% | 54.6% |
| Emotion | 17 | 22.0% | 29.3% |
| Popularity | 1 | 13.0% | 8.6% |
| Context | 3 | 6.4% | 7.6% |

By contribution the top features are m2_score, subcategory, pop_ctr, user_cat_affinity, then go_vad_dominance, the strongest emotion feature, rank 5 of 26. The best congruence feature, cong_pct_cos_personal, ranks 8; the other three congruence features rank 24, 25 and 26. This overstates
emotion, because item-level scores can act as an ID and valid is the day that rewards it. Chart: `results/figures/diagnostics/m5_feature_importance.png`, in diagnostics because the results folder is at its maximum of 6.

## Where it differs, RQ5 for M5 - M4 (dev, descriptive)
| Breakdown | Group | Impressions | Difference (95% CI) |
|---|---|---|---|
| Clicked category | TV | 3,528 | +0.0355 (0.0307, 0.0404) |
| | News | 17,230 | +0.0161 (0.0131, 0.0191) |
| | Lifestyle | 13,008 | +0.0098 (0.0067, 0.0129) |
| | Other | 16,386 | +0.0084 (0.0065, 0.0102) |
| | Sports | 18,009 | -0.0257 (-0.0281, -0.0232) |
| | Finance | 2,416 | -0.0739 (-0.0803, -0.0674) |
| | Music | 2,575 | -0.0953 (-0.1031, -0.0875) |
| Known history items | 0 | 2,214 | -0.0008 (-0.0110, 0.0094) |
| | 1-5 | 10,768 | -0.0010 (-0.0050, 0.0029) |
| | 6-20 | 25,435 | -0.0038 (-0.0060, -0.0016) |
| | 21+ | 34,735 | -0.0031 (-0.0046, -0.0016) |

Category effects are large and opposite, and they differ from M3's (music is M3's best and M5's worst). M5 and M4 have different hyperparameters, and this conditions on the clicked item, so it describes the two models, not emotion.

## Checks
| Check | Result |
|---|---|
| Own impression never contributes (test) | Pass; the test also proves later days do change |
| Position or label in a feature list (test) | Never; position is not stored in the feature table |
| Features equal saved M0, M1, M2 scores | Pass: largest gap 3e-8 on valid and on dev |
| Day 1 popularity | One constant value, as designed |
| Identical impressions in all configs | Pass |
| M4 below the leakage limit | Pass: valid 0.6745 |
| One dev run per model | Pass: 8 `final-dev` rows, M5-all has none |
| Dev files unchanged by the rebuild | Pass: 16 of 16 hashes match |
| `--rebuild` | Exit 0, GPU steps held, 5 of 5 boosters byte-identical, frozen hashes unchanged, M2 to M5 valid results identical |
| Snapshot | 75 of 76 identical. `summary.csv` differs only by new rows; its 4 original rows differ by 0 |
| Tests | 219 pass |

## Issues fixed
1. **Spec count:** your brief said 12 item features; spec v4 has 13, and the spec is what M5 must use (D65).
2. **Day 1 prior:** a prior computed from fit would include the row's own impression, so day 1 uses a fixed constant, 0.04 (D64).
3. **Seeds:** LightGBM without bagging ignores its seed, so three seeds would have shown fake zero variability. The seed now also drives the negative sampling (D66).
4. **Early stopping:** LightGBM's AUC is global, so I wrote a per-impression metric that matches the harness, proved by a test.
5. **MY ERROR, rounding:** the sadness Spearman is 0.5946, so 0.59. The Week 6 draft and base-v4 report said 0.60, a double rounding from 0.595. Both are corrected.
6. **Colour clash:** in chart 2 dev first used the blue circle reserved for fit. It is now a green triangle.
7. **Chart layout:** caption and legend overlaps on chart 5 and the importance chart were fixed after looking at each render.
8. **Renumbering:** the controls had to run before the comparison, so the 5x scripts moved by one. Controls are 50, compare 51, importance 52, ladder 53, charts 54.

## Decisions to check
- **Two extra dev runs.** You asked me to decide about dev runs for M5-all (no, D65). The item-swap and congruence-only controls are extra dev runs you did not ask for. I added them before any M4 or M5 dev run because the M5 gain could not otherwise be read. They chose nothing.
- **One added fit.** The shuffled-emotion refit (D67d) was also my addition.

## Open questions
- **Q14:** a clean congruence-only comparison on dev needs M4 at M5's hyperparameters on dev. Not run, because it would be decided after seeing dev.
- **Q15:** M6 and M7 will be tuned on the same valid, which rewards item memory. I recommend pre-declaring the fresh-item split for M7 from the start.
