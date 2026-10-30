# Start here

## What this project is
MSc Data Analytics practicum (National College of Ireland). Recommenders match content context (words and meaning). This project
tests whether emotional context also helps: does the match between a news item and the emotional context inferred from the user's
recent reading improve click ranking? Data: MIND-small (Microsoft News). Design: 8 models, M0 to M7, each with and without emotion.

Status: the data, emotion features, analyses, and the two baselines M0 and M1 are done. M2 to M7 are not built yet. The hand-label
check is waiting for the student's labels.

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

Hand labels: `python scripts/label_tool.py` (guidelines: `docs/labelling_guidelines.md`).

## Where things live
| Path | What |
|---|---|
| `scripts/1x_data_*` | download, prepare, harness check, EDA, annotation sheet |
| `scripts/2x_emotion_*` | emotion scoring and features |
| `scripts/3x_analysis_*` | click signal, congruence, context, charts, label validation |
| `scripts/4x_model_*` | baselines: `--tune` is in the pipeline, `--final-dev` is run once by hand |
| `scripts/` (unnumbered) | `run_base.py`, `show_results.py`, `label_tool.py`, `snapshot_results.py`, `check_goemotions.py` (optional) |
| `src/` | all logic; `src/config.py` holds every constant |
| `config/` | frozen emotion specifications (hash-locked by a test) |
| `results/tables/` | every result table; `summary.csv` holds model metrics, `runs_log.csv` every run |
| `results/figures/story/` | the 8 key charts |
| `results/figures/diagnostics/` | EDA and diagnostic figures |
| `results/frozen/` | frozen model settings |
| `results/archive/` | superseded results |
| `data/processed/` | parquet tables (git-ignored) |
| `annotation/` | the hand-label sheet |

## Reading order
1. This file.
2. `docs/reports/base_v3_report.md`: the latest results in simple language.
3. `docs/figures_guide.md`: the 8 charts explained.
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
