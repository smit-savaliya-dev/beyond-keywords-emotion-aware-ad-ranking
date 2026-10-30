# Week 5 autonomous run: final report (2026-10-26)

Branch `week5-autonomous` (local commits only, never pushed). Tests: 127 passing (90 at the start of this run).
`python scripts/run_base.py` reports every step up to date; step 06 waits for the hand labels.

## 1. Completed, phase by phase
| Phase | Result | Commit |
|---|---|---|
| 1 Congruence | Shared, chunked congruence code in `src/congruence.py` (reused by M3/M5/M7); valence distance added; `*_excess` renamed `*_personal`; shuffled-user control added; decision log back-filled | c4599fd |
| 2 Labelling | `scripts/label_tool.py` + `docs/labelling_guidelines.md`; 30-item blind re-label sample; no item labelled by me | 01c5d47 |
| 3 Baselines | `src/stats.py`, run log, dev guard; M0 popularity and M1 TF-IDF tuned on valid, then run once each on dev | d1b342c |
| 4 Validation | Script 06 rewritten (5 sources, per-class recall, thresholds, intra-rater, spec verdict), tested end to end with fakes; neutral share per threshold computed now | 4b330f3 |
| 5 Emotion spec | Pre-declared selection rule, valid replication, `config/emotion_spec_v1.json` (PROVISIONAL) with hash-locked test | 71ec356 |
| 6 Docs | `docs/methodology_notes.md`, EDA axis label fixed, README run order, week 5 draft | see git log |

## 2. Key results

### Congruence (fit, 126,695 impressions, 46,012 users; cold start 1.95%)
dz = mean / sd of the per-impression clicked-minus-unclicked difference. "Both" = Holm-significant at impression and
user level with the same sign. Shuffled labels: **0 of 25** features significant (max null |dz| 0.0049).

| Measure | Role | dz impr. | dz user | Both | Warm users only |
|---|---|---|---|---|---|
| cosine go_pct (main) | raw | +0.200 | +0.266 | yes | +0.201 |
| | personal (raw - corpus mean) | +0.242 | +0.291 | yes | +0.244 |
| | other user's profile (control) | +0.089 | +0.127 | yes | +0.088 |
| | user-specific (raw - other user) | +0.136 | +0.175 | yes | +0.140 |
| -KL 6-class | raw | -0.040 | -0.056 | yes | -0.039 |
| | personal | +0.117 | +0.134 | yes | +0.119 |
| | user-specific | +0.079 | +0.099 | yes | +0.080 |
| valence distance | raw | -0.033 | -0.043 | yes | -0.030 |
| | personal | +0.157 | +0.189 | yes | +0.159 |
| | user-specific | +0.073 | +0.098 | yes | +0.075 |
| cosine 6-class (variant) | raw / personal / user-specific | -0.084 / +0.131 / +0.071 | -0.116 / +0.158 / +0.095 | yes | -0.082 / +0.132 / +0.072 |
| -KL go_pct (exploratory) | raw / personal | +0.160 / +0.225 | +0.216 / +0.270 | yes | +0.160 / +0.227 |

- **Direction: congruence-seeking** for every personal and user-specific measure. No mood repair. The negative raw
  values (-KL 6-class, valence distance, 6-class cosine) come from the item's own emotion: the corpus-mean and
  other-user controls are even more negative. (The CSV's `direction` column applies the sign rule to every row; only
  the personal / user-specific rows answer the congruence vs mood-repair question.)
- About 40% of the pre-declared `personal` effect is shared by all users (other-user control dz 0.089), so the
  stricter `userspecific` measure is the honest headline (Q01).
- Valid replication (30,270 impressions): personal cosine 0.242 -> 0.237, personal -KL 0.117 -> 0.092, personal
  valence 0.158 -> 0.105.

### M0 and M1 (no emotion)
| Model | Part | AUC | MRR | nDCG@5 | nDCG@10 | P@5 | P@10 | R@5 | R@10 | HR@5 | HR@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 popularity (a=10) | valid | 0.5740 | 0.2347 | 0.2415 | 0.3026 | 0.0880 | 0.0688 | 0.3562 | 0.5328 | 0.4163 | 0.6098 |
| M1 TF-IDF (title+abstract, mean) | valid | 0.5900 | 0.2765 | 0.2923 | 0.3497 | 0.1029 | 0.0751 | 0.4046 | 0.5700 | 0.4751 | 0.6504 |
| M0 popularity | dev | 0.5890 | 0.2749 | 0.2900 | 0.3506 | 0.1028 | 0.0759 | 0.4077 | 0.5832 | 0.4762 | 0.6599 |
| M1 TF-IDF | dev | 0.5744 | 0.2759 | 0.2933 | 0.3554 | 0.1048 | 0.0776 | 0.4159 | 0.5949 | 0.4849 | 0.6710 |

Same impression count for both models on each part (valid 30,270; dev 73,152). Both above random on valid and dev.
Unseen candidates: 29.2% on valid (fit statistics), 21.3% on dev (fit + valid statistics).

### Paired M1 - M0 (t-test, Holm across 10 metrics)
| Part | AUC | MRR | nDCG@5 | nDCG@10 | HR@10 |
|---|---|---|---|---|---|
| valid | +0.0160 [0.0116, 0.0204] * | +0.0419 * | +0.0508 * | +0.0470 * | +0.0406 * |
| dev | -0.0146 [-0.0175, -0.0117] * | +0.0010 (n.s.) | +0.0032 (n.s.) | +0.0048 * | +0.0111 * |

On dev, popularity wins on AUC; TF-IDF is slightly better on P/R/HR@5/10 and nDCG@10. Not leakage: dev popularity
uses the valid day (one day before dev), which lowers the unseen share and adds recency.

### Neutral share on the full corpus (65,238 items; argmax baseline 81.1%)
| Raw max sigmoid t | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| neutral % | 23.8 | 44.8 | 66.5 | 79.4 | 86.8 | 92.6 |

| Max percentile t (secondary) | 0.80 | 0.85 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|
| neutral % | 24.6 | 34.5 | 49.3 | 70.1 | 92.7 |

Kappa per threshold is computed by script 06 once the labels exist (sensitivity only; t is not chosen by kappa).

## 3. Issues found and fixed
1. **Personalisation overstated by the pre-declared control.** The corpus-mean control leaves a shared "typical
   clicked news" shift; a random other user's profile still gives dz 0.089. Added `*_otheruser` / `*_userspecific`
   (D13), kept `*_personal` unchanged.
2. **Pipeline re-ran everything after any config change** (both GPU steps included). It now depends on the individual
   constants each step imports (D16); `--adopt` can no longer skip a forced step. Tests added.
3. **Dev was loaded when scoring valid**: `load_truth` and script 03 read all parts. Both now read only the requested
   part (D15, D18); analysis loaders refuse dev.
4. **Cosmetic**: M1 dev log line said "valid auc". Fixed; saved metrics were correct, so dev was not re-run.
5. **Syntax error from a scripted edit** in script 06 (newline inside an f-string), caught by the end-to-end test.
6. **Not a bug**: M0 beats M1 on dev AUC (see above). Several item-level emotion effects flip between fit and valid
   (e.g. emotionality +0.103 -> -0.041): treated as a finding, handled by the pre-declared replication rule.

## 4. Decisions and open questions
All choices are in `docs/decisions.md` (D01-D29), with alternatives, reasons and the data part used. The key ones:
D09 (t-test instead of Wilcoxon), D13 (other-user control, added after seeing results), D21/D23 (frozen M0/M1
settings), D24 (thresholds declared before running), D27 (feature selection rule declared before the valid check),
D28 (spec v1), D29 (post-label confirmation rule).
Open (`docs/open_questions.md`): Q01 headline `userspecific` vs `personal`; Q02 a full-candidate M5 robustness variant;
Q03 percentiles ranked over all items including dev-only items (label-free).

## 5. What you need to do next
1. Read `docs/labelling_guidelines.md`, then label in short sessions (saves after every key, resumes automatically):
   `python scripts/label_tool.py`   (keys 1 joy, 2 sadness, 3 anger, 4 fear, 5 surprise, 6 neutral, s skip, b back, q quit)
2. When all 180 are done: `python scripts/label_tool.py --make-relabel`
3. At least one day later, without looking at your first labels: `python scripts/label_tool.py --relabel`
4. Then run `python scripts/run_base.py` (or `python scripts/06_validate_emotion_labels.py`). It completes Phase 4
   (kappa per source with CI, per-class recall, threshold curve, intra-rater kappa) and prints the spec verdict:
   CONFIRM (set `status` to confirmed), BUMP (write `config/emotion_spec_v2.json`) or HOLD.
5. Review the branch `week5-autonomous` (`git log main..week5-autonomous`) and merge it if you agree.

## 6. Week 5 report draft
See `results/reports/week5_report_draft.md`.
