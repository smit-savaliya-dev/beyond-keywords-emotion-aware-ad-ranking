# Progress (week 5 autonomous run)

Read this first when resuming. Latest work: branch `base-v2` (local only), section "base-v2" at the end of this file.
Earlier: branch `week5-autonomous`, Phases 1-6 of the 2026-10-26 instruction.

| Phase | Status | Commit |
|---|---|---|
| 0 State check | done | - |
| 1 Congruence (valence distance, personal, controls, decision log) | done | see git log |
| 2 Labelling tool + guidelines | done | see git log |
| 3 M0 popularity, M1 TF-IDF, stats, runs log | done (dev run once each) | see git log |
| 4 Label validation (code + fakes; neutral share per threshold) | code done; real run waits for labels | see git log |
| 5 Emotion spec v1 (provisional) | done | see git log |
| 6 Docs, EDA label fix, README, week 5 draft | done | see git log |

## Phase 1 (2026-10-26)
- Shared function `src.congruence.congruence_features` (chunked, float32) so M3/M5/M7 reuse the same code;
  measures declared in `CONGRUENCE_MEASURES` (config).
- Added valence distance -|profile - item| on go_vad_valence; renamed `*_excess` to `*_personal`.
- Key numbers (fit, 126,695 impressions, 46,012 users, cold start 1.95%), dz per impression / user, all significant at
  both levels (Holm), shuffled labels 0/25 false positives (max null |dz| 0.0049):
  - cong_pct_cos raw 0.200 / 0.266; personal 0.242 / 0.291; other-user 0.089 / 0.127; user-specific 0.136 / 0.175.
  - cong_six_cos raw -0.084; personal 0.131; user-specific 0.071.
  - cong_six_negkl raw -0.040; personal 0.117; user-specific 0.079.
  - cong_valence_absdist raw -0.033; personal 0.158; user-specific 0.073.
  - Warm users only: practically identical (e.g. pct cos raw 0.202, user-specific 0.140).
  - Direction: congruence-seeking for every personal and user-specific measure; the negative raw 6-class and valence
    values come from the item's own emotion (the generic and other-user controls are even more negative), not mood repair.
- Issues found and fixed:
  1. The pre-declared generic control (corpus mean) overstates personalisation: the other-user profile also gives
     dz 0.089. Added a shuffled-user control (D13, Q01).
  2. Pipeline: any change in src/config.py re-ran every step, including the GPU steps. Now each step depends only on
     the constants it imports (D16), with tests.
  3. `--adopt` could skip a step that was explicitly forced. Fixed, with a test.
  4. Script 03 loaded dev by default; now valid only (D15). Harness check on valid: random AUC 0.4985, oracle 1.0.

## Phase 2 (2026-10-26)
- `docs/labelling_guidelines.md`; `scripts/label_tool.py` (logic in `src/labelling.py`): one item at a time, title +
  abstract only, keys 1-6 / s / b / q, atomic save after every answer, timestamped backup in `annotation/backups/`
  before the first change of each session, resumes at the first unlabelled item.
- `--make-relabel` (30 seeded items, only when the sheet is complete, never regenerated) and `--relabel` (warns if
  < 24 h since creation). 6 tests on temp CSVs. Real sheet checked read-only (hash unchanged); nothing was labelled.

## Phase 3 (2026-10-26)
- `src/stats.py` (paired t-test, 95% CI, dz, Holm; Wilcoxon robustness only; Holm moved here from click_signal),
  `src/experiments.py` (runs_log.csv, frozen settings, dev guard), `src/baselines.py`, scripts 20/21/22.
- Valid (30,270 impressions): M0 a=10 AUC 0.5740 (a=1 0.5725, a=100 0.5734), 29.2% of candidates unseen in fit.
  M1 best = title+abstract, no sublinear, mean: AUC 0.5900 (8 configs 0.5728-0.5900). M1 - M0: every metric better,
  AUC +0.016 [0.012, 0.020], nDCG@5 +0.051, MRR +0.042 (Holm significant).
- Frozen settings recorded in decisions.md (D21, D23), then dev ran ONCE per model (73,152 impressions, same count):
  M0 AUC 0.5890 (21.3% unseen in fit+valid), M1 AUC 0.5744. M1 - M0 on dev: AUC -0.0146 [-0.0175, -0.0117]
  (M0 better), MRR +0.001 (n.s.), nDCG@5 +0.003 (n.s.), nDCG@10 +0.005, P/R/HR@5/10 small but significant gains.
- Checks: harness random AUC 0.4985 (valid), both models above random on valid and dev, no leakage warning,
  rerunning --tune reproduced the same frozen hashes, Phase 5 table bit-identical after moving holm.
- Issues found and fixed:
  1. `load_truth` read every part before filtering (dev in memory while scoring valid); now filtered at read (D18).
  2. M1 dev log line said "valid auc" (hard-coded label); fixed. Saved metrics were correct, so no dev rerun.
- Not a bug: M0 beats M1 on dev AUC. Popularity statistics for dev include the valid day (one extra day, lower
  unseen share), which gives a strong recency signal; dev labels were never used.

## Phase 4 (2026-10-26)
- Thresholds declared in config BEFORE the first run (D24). `src/validation.py`, script 16 (item-level, runs now),
  script 06 rewritten (5 sources, per-class recall with < 5 flag, threshold kappa curve with bootstrap CI,
  intra-rater kappa from the re-label sheet, figure). End-to-end test runs script 06 on a fake project in tmp_path;
  NO fake labels were written to any real output. The pipeline step 06 waits until the 180 labels exist.
- Neutral share on the full corpus (65,238 items): raw t=0.05 23.8%, 0.1 44.8%, 0.2 66.5%, 0.3 79.4%, 0.4 86.8%,
  0.5 92.6%; percentile t=0.80 24.6%, 0.85 34.5%, 0.90 49.3%, 0.95 70.1%, 0.99 92.7%. Argmax baseline 81.1%.
- Limitation found: the percentile view cannot produce sadness (D25), documented in the metadata.

## Phase 5 (2026-10-26)
- Selection rule pre-declared (D27) BEFORE the valid replication; script 17 (valid only) applied it mechanically.
- 25 candidates -> 16 kept (12 item, 4 congruence). Dropped: go_pct_curiosity, go_pct_desire, go_emotionality,
  go_vad_arousal, nrc_anticipation, vad_dominance, cong_six_negkl, cong_valence_absdist (sign flips on valid),
  go_pct_annoyance (valid CI includes 0). Personal congruence replicated: pct 0.242 -> 0.237, six negKL 0.117 ->
  0.092, valence 0.158 -> 0.105 (fit -> valid dz). Shuffled-label null on valid 0/25.
- `config/emotion_spec_v1.json` (status provisional), `src/emotion_spec.py` (hash excluding status, validation,
  confirmation rule D29 as code); script 06 prints and saves the verdict. Test locks the v1 hash.
- Issue found and fixed: a scripted edit put a raw newline inside an f-string in script 06; caught by the end-to-end
  test, fixed by hand.

## Phase 6 (2026-10-26)
- `docs/methodology_notes.md` (split, 1/3 : 2/3 weighting, Wilcoxon replacement, percentile calibration, GoEmotions vs
  NRC agreement with numbers, congruence controls); EDA clicks x-axis now "Number of clicked articles in one
  impression" (figures regenerated by the pipeline, only step 07 ran); README run order covers every script;
  `results/reports/week5_report_draft.md`; final report `docs/week5_run_report.md`.
- Final state: 127 tests pass; `run_base.py` up to date except step 06 (waits for labels).

## Resume here
Everything planned is done. Remaining work needs the student: hand labels (then script 06 confirms or bumps the
emotion spec), review/merge of branch `week5-autonomous`, then M2/M3.

## base-v2 (2026-10-30)
| Step | Status |
|---|---|
| Pre-declaration D30-D32 (committed before computing) | done |
| 1 Reduce neutral, spec v2 | done |
| 2 Short-term vs long-term context | done |
| 3 Eight story charts + figures guide | done |
| Final report | done (`docs/base_v2_report.md`) |

### Step 1
- Grid reproduces v1 exactly (share 1/3: fit 0.2419, valid 0.2371). Chosen by D30: title share 0.5, t = 0.05.
- Neutral 25.6% (v1 argmax 81.1%, combined 75.9%). Joy 34.5, sadness 18.9, surprise 13.5, anger 5.6, fear 1.9.
- Personal congruence fit 0.2463, valid 0.2388 (gate 0.2134). Shuffled labels: 0 on fit and on valid.
- D32 on v2 features: 17 of 26 candidates kept; go_intensity fails (sign flips). Spec v2 frozen, v1 kept.
- New files only: news_emotion_raw28_v2, news_emotion_labels_v2, news_emotion_features_v2 (v1 outputs untouched).
- Issues: (1) a long shell heredoc failed to parse (nothing ran), edits moved to a script file. (2) The weights are
  statistically tied (D33). (3) Low t is a relabelling, not more detection (Q04). Hand-label sheet hash unchanged.

### Step 2
- Long-term wins on fit and valid (user-specific dz 0.138 / 0.137); last 1 is weakest (0.057 / 0.062). All 12 features
  significant at both levels on both parts; shuffled labels 0 of 12.
- Control on users with >= 10 history items: last 5 does not beat the previous 5, so the win comes from size and stable
  preference, not recency (D35). True recency cannot be tested with the history column (Q07).

### Step 3 and final
- `src/story_charts.py` (one style, palette validated), `scripts/34_story_charts.py`, `docs/figures_guide.md`, exactly 8 PNGs
  in `results/figures/story/`, each rendered and inspected.
- Issues found while checking the charts: (1) a drafted chart 3 title said titles are more intense; the data says abstracts
  are (median 0.116 vs 0.068), so the title was fixed. (2) That exposed a wrong statement in `docs/methodology_notes.md`
  ("abstracts are more neutral than titles"); the real cause of the higher v1 neutral share is dilution by averaging. Fixed.
  (3) A drafted chart 1 caption was replaced by computed numbers. (4) Label collisions and tick overlap fixed.

## Resume here
base-v2 is complete. Waiting for the student: hand labels (then script 06 confirms, bumps or holds spec v2), review of the
open questions, review and merge of `base-v2`, then M2/M3.

## base-v3 (2026-10-30)
Branch `base-v3`, built from the base-v2 tip (759d563) because main does not contain it (Q08). Pre-declarations D37-D40 were
committed before computing. `annotation/` is excluded from every commit while the student labels.

| Step | Status |
|---|---|
| 1 Stance-excluded labels v2b, v2c | done |
| 2 True recency | done |
| 3 Percentile robustness | done (max dz change 0.00022, D43) |
| 4 Cleanup and restructure | done |
| 5 Terminal summary, --rebuild | done |
| Final report | done (`docs/reports/base_v3_report.md`) |

### Step 1
- At t = 0.05: v2 neutral 25.6 (joy 34.5, sadness 18.9, anger 5.6, fear 1.9, surprise 13.5); v2b 39.0 (20.7, 19.8, 2.8, 1.9, 15.8);
  v2c 42.4 (20.9, 20.3, 2.8, 2.0, 11.6). Anger is now 82% annoyance. Script 06 compares v1, v2, v2b, v2c when labels exist.
- Issue: the end-to-end test took 3 minutes after adding the variants (1,000 bootstraps x 32 thresholds). Added --n-boot; the
  test uses 30, the real run keeps 1,000.

### Step 2
- Coverage: 63.68% of fit and 86.82% of valid impressions have an earlier log click.
  1.12% / 0.846% of candidates removed as repeats. Shuffled labels 0 of 18 on both parts.
- Residual dz fit / valid: last 1 0.054 / 0.017, last 3 0.067 / 0.023, last 5 0.068 / 0.033; rule D39 says 'adds'. Long-term stays stronger.
  Same-size controls n.s. Placebo (D41, post-hoc) tested a shared-trend explanation and rejected it. Outcome D42; Q09 left undecided.

### Step 3
- Train-only ECDFs: max |dz change| 0.00022 (limit 0.02), no change in significance. Current features kept (D43).

### Step 4
- `docs/cleanup_plan.md` lists every file. Scripts grouped: 1x data, 2x emotion, 3x analysis, 4x models. Figures split into story/ and
  diagnostics/; 3 superseded figures archived; reports moved to docs/reports/; `docs/START_HERE.md` added.
- Removed: corpus_mean, load_processed, scripts/09_check_lexicons.py, the plotting code of the 3 archived figures, notebooks/.
- Merged: TABLES (19 copies), lazy value views, Holm family, part loader, window congruence, chart colours.
- Issues found and fixed:
  1. src/eda.py and the v1 congruence script read dev rows into memory. Both now read fit only; no result changed.
  2. A pre-cleanup bug of mine: the recency category table used per-category Holm for one robustness column. Fixed before the snapshot.
  3. The snapshot tool missed changes inside array columns; found by its own test, rewritten.
  4. Nothing tested the pipeline definition, so a NameError got through. Added tests/test_run_base.py.
  5. A dry run showed the GPU steps would start after the renames. Added a hold.
  6. MY ERROR: the first rebuild stopped at step 30 (two archived figures still listed as outputs). The rerun used --from 30, which did not
     carry the hold, so steps 20 and 21 ran for about 36 s. They loaded the model on the GPU, reloaded 97 saved chunks and scored 0 texts; the 7
     output files are byte-identical. GPU steps with existing outputs now never start unless named with --force-step, on any run.
  7. MY ERROR: entries D37-D43 were first dated 2026-11-01; the real date is 2026-10-30. Corrected.

### Step 5 and final check
- `scripts/show_results.py` (logic in `src/summary.py`, saved tables only) is the last pipeline step. `--rebuild` skips GPU scoring.
- One uninterrupted `--rebuild`: exit 0, 23 steps ran, 76 of 76 snapshot files identical, 163 tests pass, no dev run.

## Resume here
base-v3 is complete. Waiting for the student: fix the merge state (Q08), hand labels, Q09, then M2/M3.

## 2026-10-30 base-v4 step 1: hand labelling archived, SemEval tooling
- Moved the sheet (0 of 180 labelled), tool, guidelines, sheet maker and their tests to `results/archive/annotation/` (D45). `src/annotation.py` became `src/agreement.py` (statistics only). The pipeline, script 38 and spec code no longer depend on `annotation/*.csv`. Script 27 reads the archived ids so `emotion_examples_v2.csv` stays the same.
- SemEval-2007 Task 14 downloaded with verified TLS; sha256 pinned in config; 250 + 1,000 headlines, ids match the gold files. New: `src/semeval.py`, scripts 14 (download), 29 (scoring), 38 (validation).
- Fixed while testing: gold scores came back as a read-only array; the constant-input guard used std == 0, which float noise can miss.
