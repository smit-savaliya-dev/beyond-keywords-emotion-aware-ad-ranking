"""Run log, frozen model settings and the once-only dev guard shared by every model script."""
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from .config import DECISIONS, FROZEN_DIR, RESULTS, RUNS_LOG
from .provenance import run_info


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_run(model, part, mode, config, metrics, note="", path=RUNS_LOG):
    """One row per evaluated run: model, part, mode, config (JSON + hash), metrics, timestamp, git commit."""
    info = run_info()
    row = {"timestamp_utc": _now(), "model": model, "part": part, "mode": mode,
           "config": json.dumps(config, sort_keys=True), "config_hash": config_hash(config),
           **{k: round(float(v), 6) for k, v in dict(metrics).items()},
           "git_commit": info.get("git_commit"), "git_dirty": info.get("git_dirty"), "note": note}
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.concat([pd.read_csv(path), pd.DataFrame([row])], ignore_index=True) if path.exists() \
        else pd.DataFrame([row])
    tmp = path.with_name(path.name + ".tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(path)
    return row


def save_frozen(model, config, evidence, frozen_dir=FROZEN_DIR):
    frozen_dir.mkdir(parents=True, exist_ok=True)
    doc = {"model": model, "config": config, "config_hash": config_hash(config), "chosen_on": "valid",
           "evidence": evidence, "frozen_utc": _now(), "run_info": run_info()}
    path = frozen_dir / f"{model}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(path)
    return doc


def load_frozen(model, frozen_dir=FROZEN_DIR):
    path = frozen_dir / f"{model}.json"
    if not path.exists():
        raise SystemExit(f"no frozen settings for {model}: run the script with --tune first")
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_marker(model, cfg_hash):
    """The exact text docs/decisions.md must contain before a dev run."""
    return f"frozen:{model}:{cfg_hash}"


def dev_already_run(model, results=RESULTS, runs_log=RUNS_LOG):
    summary = results / "tables" / "summary.csv"
    if summary.exists():
        s = pd.read_csv(summary)
        if ((s["model"] == model) & (s["part"] == "dev")).any():
            return True
    if runs_log.exists():
        r = pd.read_csv(runs_log)
        if ((r["model"] == model) & (r["part"] == "dev")).any():
            return True
    return False


def check_final_dev(model, rerun_reason=None, frozen_dir=FROZEN_DIR, decisions=DECISIONS, results=RESULTS,
                    runs_log=RUNS_LOG):
    """Refuse a dev run unless settings are frozen, recorded in the decision log, and dev was not run before
    (a rerun needs an explicit reason, which is appended to the decision log). -> frozen settings."""
    frozen = load_frozen(model, frozen_dir)
    marker = frozen_marker(model, frozen["config_hash"])
    text = decisions.read_text(encoding="utf-8") if decisions.exists() else ""
    if marker not in text:
        raise SystemExit(f"record the frozen settings in {decisions.name} first; the entry must contain '{marker}'")
    if dev_already_run(model, results, runs_log):
        if not rerun_reason:
            raise SystemExit(f"{model} already has a dev result. Dev runs once per model; rerun only with "
                             '--rerun-dev "<reason>" (logged in runs_log.csv and docs/decisions.md)')
        with open(decisions, "a", encoding="utf-8") as f:
            f.write(f"| R-{model}-{_now()} | {_now()[:10]} | Re-ran {model} on dev ({marker}) | - | "
                    f"{rerun_reason.replace('|', '/')} | dev re-run, logged automatically |\n")
    return frozen


def load_rows(processed, parts, columns):
    """Impression rows of the given parts only (other parts are never read)."""
    rows = pd.read_parquet(processed / "impressions.parquet", columns=["part", *columns],
                           filters=[("part", "in", list(parts))])
    found = set(rows["part"].astype(str))
    if not found <= set(parts):
        raise ValueError(f"unexpected parts loaded: {sorted(found - set(parts))}")
    return rows


def pick_best(summaries, metric):
    """Index of the highest `metric` among grid summaries; ties go to the first in grid order."""
    return max(range(len(summaries)), key=lambda i: (summaries[i][metric], -i))
