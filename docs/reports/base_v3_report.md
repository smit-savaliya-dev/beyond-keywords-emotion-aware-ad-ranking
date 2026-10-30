# Base v3 report (2026-10-30)

Branch `base-v3` (local commits only, never pushed). 163 tests pass. Only fit and valid were analysed. Dev was not run again: the only dev
runs are still the two from 26 October. The hand-label sheet is unchanged. No model was built.

**Action needed first (Q08):** base-v2 was reported as merged, but `main` and `origin/main` are both still at a916a50. `main` was not touched.
`base-v3` was built from the base-v2 tip (759d563), so merging `base-v3` brings in all reviewed work.

## Summary for the professor
1. Removing opinion labels (approval, disapproval) is the honest way to count emotion: neutral is 39%, not 26%. Hand labels will decide
   which labelling to keep. Model features do not depend on this choice.
2. We tested recency properly, using each user's earlier clicks in the log. Recent reading adds a little beyond long-term preference.
3. The long-term profile still matters more, and the order of recent clicks does not matter: the very last click is no more useful than
   the one before it. So this is about what someone read this week, not a momentary effect.
4. Results are stable: changing how percentiles are computed moves effect sizes by at most 0.0002.
5. The project is now organised and fully reproducible: every step except GPU scoring was rerun and all 76 result files matched.
6. Nothing yet shows better ranking. The next step is M2 vs M3: does emotion add value beyond content?

## Steps done
| Step | Result |
|---|---|
| Pre-declaration | D37-D40 committed before computing |
| 1 Labels v2b, v2c | Added to charts 1 and 2 and to the hand-label comparison |
| 2 True recency | Small real addition, long-term stays stronger (D42) |
| 3 Percentile robustness | Max dz change 0.00022, features kept (D43) |
| 4 Cleanup | Scripts grouped, duplicates merged, snapshot identical |
| 5 Summary | `scripts/show_results.py`, `--rebuild`, last pipeline step |

## v2 vs v2b at t = 0.05 (% of items)
| Rule | Neutral | Joy | Sadness | Anger | Fear | Surprise |
|---|---|---|---|---|---|---|
| v1 | 81.1 | 8.7 | 5.3 | 0.6 | 0.4 | 4.0 |
| v2 | 25.6 | 34.5 | 18.9 | 5.6 | 1.9 | 13.5 |
| **v2b, no approval or disapproval** | **39.0** | **20.7** | **19.8** | **2.8** | **1.9** | **15.8** |
| v2c, also no realization | 42.4 | 20.9 | 20.3 | 2.8 | 2.0 | 11.6 |

Neutral % by threshold:
| Variant | 0.03 | 0.05 | 0.075 | 0.1 | 0.15 | 0.2 | 0.3 |
|---|---|---|---|---|---|---|---|
| v2 | 12.3 | 25.6 | 37.6 | 46.0 | 57.7 | 66.3 | 79.3 |
| v2b | 26.5 | 39.0 | 48.3 | 54.5 | 63.0 | 69.7 | 80.6 |
| v2c | 32.6 | 42.4 | 50.1 | 55.5 | 63.6 | 70.0 | 80.7 |

Removing stance mainly affects joy and anger: anger is now 82% annoyance rather than 60% disapproval. Script 38 will compare v1, v2, v2b and v2c
against the hand labels. No variant is chosen yet.

## True recency: does very recent emotional context add anything beyond long-term?
**Yes, a small amount. Long-term still matters more, and the order of recent clicks does not matter.**

Coverage: 63.7% of fit and 86.8% of valid impressions have an earlier log click (28,604 and 16,933 users). Repeated candidates were removed
(1.1% and 0.8%). Shuffled labels gave 0 of 18 false positives on both parts.

| Context, user-specific dz | Fit | Valid |
|---|---|---|
| Long-term profile | 0.139 | 0.143 |
| Last 5 log clicks | 0.090 | 0.085 |
| Last 3 log clicks | 0.087 | 0.073 |
| Last 1 log click | 0.065 | 0.056 |

| Added beyond long-term, residual dz | Fit | Valid | Rule D39 |
|---|---|---|---|
| Last 1 | 0.054 | 0.017 | adds |
| Last 3 | 0.067 | 0.023 | adds |
| Last 5 | 0.068 | 0.033 | adds |

- Long-term adds more in the other direction: long-term given last 5 is 0.078 on fit and 0.083 on valid.
- Same-size controls (last k vs previous k, k = 1, 3, 5): no significant difference on either part, so order does not matter.
- Placebo (post-hoc, D41): I suspected the residual was only a shared trend. Replacing the user's clicks with another user's clicks at nearly
  the same time gives a negative residual, 0.08 to 0.11 below the real one, so this explanation is rejected.
- Caution: on valid the residual is close to 0 inside news, sports and finance. Active users only; recent clicks may also share a topic.
- Whether models may use log clicks is left open (Q09).

## Percentile robustness (Q03)
Train-only ECDFs on 51,282 items: max |dz change| 0.00022 (limit 0.02), mean percentile shift 0.0011, no change in significance. Current
features stay.

## Cleanup summary
- **Removed:** `corpus_mean`, `load_processed`, `09_check_lexicons.py`, the plotting code of the 3 archived figures, `notebooks/`.
- **Merged:** `TABLES` (19 copies), lazy value views, Holm family logic, part loader, window congruence helper, chart colours.
- **Archived:** 3 superseded figures in `results/archive/figures/`, with a README.
- **Moved:** 9 files to `results/figures/diagnostics/`, 3 reports to `docs/reports/`.
- **Renamed:** 27 scripts by stage. Old-to-new table in `docs/cleanup_plan.md`, which lists every file.
- **Kept on purpose:** v1 tables at their paths and spec files unchanged, because hashes pin them. Spec text still names old scripts.

```
scripts/  1x data · 2x emotion · 3x analysis · 4x models · run_base, show_results, label_tool, snapshot_results, check_goemotions
docs/     START_HERE · decisions · methodology_notes · figures_guide · open_questions · progress · cleanup_plan · reports/
results/  tables/ · figures/story/ (8 charts) · figures/diagnostics/ · frozen/ · archive/
```

## Snapshot equality check
Taken before the cleanup: 76 files (tables, frozen settings, processed parquet and JSON). After the cleanup, one uninterrupted `--rebuild` ran 23
steps and passed the official MIND counts check. **76 of 76 files are identical**, floats within 1e-9; run timestamps are ignored. Spec v1 and v2
evidence hashes still match. The run log is append-only and was excluded. The two GPU steps were held and `--final-dev` was never run.

## Issues found and fixed
**My own errors**
1. The first rebuild stopped at step 30 because two archived figures were still listed as outputs. The wrapper reported exit 0, so I reran with
   the real exit code captured.
2. My rerun with `--from 30` did not carry the GPU hold, so steps 20 and 21 ran for about 36 seconds. They loaded the model on the GPU, reloaded
   97 saved chunks and scored 0 texts; all 7 outputs are byte-identical. This should not have happened. GPU steps with existing outputs now
   never start unless named with `--force-step`.
3. I dated entries D37-D43 as 2026-11-01; the real date is 2026-10-30. Corrected.
4. The recency category table used per-category Holm for one robustness column. Fixed before the snapshot.

**Found by checks**
5. `src/eda.py` and the v1 congruence script read dev rows into memory. Both now read fit only; no result changed.
6. The snapshot tool missed changes inside array columns; its own test found this, and it was rewritten.
7. Nothing tested the pipeline definition, so a `NameError` got through. Added `tests/test_run_base.py`.
8. The end-to-end test took 3 minutes; added `--n-boot`.

## Decisions (`docs/decisions.md`)
D37 M5-all variant, D38 v2b and v2c, D39 recency rule, D40 percentile rule, D41 placebo (post-hoc), D42 recency outcome, D43 percentile outcome,
D44 cleanup. M5-all is declared but not built.

## Open questions (`docs/open_questions.md`)
- **Q08 (new):** merge state of main, see the top of this report.
- **Q09 (new, not decided):** may models use within-log earlier clicks?
- **Q04, Q06:** on hold until your labels.
- Q01, Q02, Q03, Q05, Q07 are resolved.

## Exact commands
```powershell
python scripts/run_base.py --dry-run     # what would run and why
python scripts/run_base.py               # run what is out of date, then print the summary
python scripts/run_base.py --rebuild     # rerun everything except GPU scoring (about 15 minutes)
python scripts/show_results.py           # only the summary, from saved tables
python scripts/snapshot_results.py --check   # prove the results still match the snapshot
```
