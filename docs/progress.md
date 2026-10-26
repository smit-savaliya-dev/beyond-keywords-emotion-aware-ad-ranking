# Progress (week 5 autonomous run)

Read this first when resuming. Branch: `week5-autonomous` (local only, never pushed). Plan: Phases 1-6 of the
2026-10-26 instruction, a test run and a local commit after each phase.

| Phase | Status | Commit |
|---|---|---|
| 0 State check | done | - |
| 1 Congruence (valence distance, personal, controls, decision log) | done | see git log |
| 2 Labelling tool + guidelines | done | see git log |
| 3 M0 popularity, M1 TF-IDF, stats, runs log | done (dev run once each) | see git log |
| 4 Label validation (code + fakes; neutral share per threshold) | code done; real run waits for labels | see git log |
| 5 Emotion spec v1 (provisional) | done | see git log |
| 6 Docs, EDA label fix, README, week 5 draft | pending | |

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
