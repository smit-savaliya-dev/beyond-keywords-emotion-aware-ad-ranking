# Start here

## What this project is
MSc Data Analytics practicum (National College of Ireland). Recommenders match content context (words and meaning). This project
tests whether emotional context also helps: does the match between a news item and the emotional context inferred from the user's
recent reading improve click ranking? Data: MIND-small (Microsoft News). Design: 8 models, M0 to M7, each with and without emotion.

Status: the data, emotion features, analyses, the two baselines M0 and M1, and the external validation on SemEval-2007 headlines are
done. M2 to M7 are not built yet. Hand labelling was cancelled (D45), so there are no direct human labels on MIND headlines.

## How to run
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_base.py --dry-run     # what would run and why; runs nothing
python scripts/run_base.py               # run whatever is out of date, then print the summary
python scripts/show_results.py           # only the summary, from saved tables
python scripts/run_base.py --rebuild     # rerun everything except GPU scoring; dev runs are never in the pipeline
```
Put MIND-small in `data/raw/train/` and `data/raw/dev/`, and the NRC lexicons in `data/external/nrc/` (never commit them).
Step 14 downloads SemEval-2007 into `data/external/semeval2007/` (git-ignored, needs `curl`).

## Where things live
| Path | What |
|---|---|
| `scripts/1x_data_*` | download, prepare, harness check, EDA, SemEval-2007 download |
| `scripts/2x_emotion_*` | emotion scoring and features, SemEval headline scoring |
| `scripts/3x_analysis_*` | click signal, congruence, context, charts, SemEval validation |
| `scripts/4x_model_*` | baselines: `--tune` is in the pipeline, `--final-dev` is run once by hand |
| `scripts/` (unnumbered) | `run_base.py`, `show_results.py`, `snapshot_results.py`, `check_goemotions.py` (optional) |
| `src/` | all logic; `src/config.py` holds every constant |
| `config/` | frozen emotion specifications (hash-locked by a test) |
| `results/tables/` | every result table; `summary.csv` holds model metrics, `runs_log.csv` every run |
| `results/figures/story/` | the 8 key charts |
| `results/figures/diagnostics/` | EDA and diagnostic figures |
| `results/frozen/` | frozen model settings |
| `results/archive/` | superseded results; `annotation/` holds the cancelled hand-label sheet and tool |
| `data/processed/` | parquet tables (git-ignored) |

## Reading order
1. This file.
2. `docs/reports/base_v4_report.md`: the latest results in simple language.
3. `docs/figures_guide.md`: the 8 charts and the SemEval chart explained.
4. `docs/methodology_notes.md`: why the method is what it is.
5. `docs/decisions.md`: every decision, with alternatives and evidence.
6. `docs/open_questions.md`: what still needs a decision.
7. `docs/progress.md`: what was done and which issues were fixed. `docs/cleanup_plan.md`: old-to-new script names.

## Rules that protect the results
- fit trains and analyses, valid chooses settings, dev is the final test: once per model, with settings frozen first.
- Analysis scripts never read dev. Click statistics use only parts earlier than the one being scored.
- Selection rules are written in `docs/decisions.md` and committed before computing.
- `position` is log order: never a feature.
- Write "emotional context inferred from the user's recent reading", never "the user's emotional state".
