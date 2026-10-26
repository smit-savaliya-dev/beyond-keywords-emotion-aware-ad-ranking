# Progress (week 5 autonomous run)

Read this first when resuming. Branch: `week5-autonomous` (local only, never pushed). Plan: Phases 1-6 of the
2026-10-26 instruction, a test run and a local commit after each phase.

| Phase | Status | Commit |
|---|---|---|
| 0 State check | done | - |
| 1 Congruence (valence distance, personal, controls, decision log) | done | see git log |
| 2 Labelling tool + guidelines | pending | |
| 3 M0 popularity, M1 TF-IDF, stats, runs log | pending | |
| 4 Label validation (code + fakes; neutral share per threshold) | pending | |
| 5 Emotion spec v1 (provisional) | pending | |
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
