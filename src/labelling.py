"""Terminal hand-labelling: one item at a time (title + abstract only, never model output), save after every answer."""
import json
import shutil
import textwrap
from datetime import datetime, timezone

import numpy as np

from .annotation import read_sheet, sheet_problems
from .config import RELABEL_MIN_HOURS, RELABEL_N, SEED
from .emotion import SIX_CLASSES

KEYMAP = {str(i + 1): c for i, c in enumerate(SIX_CLASSES)}
COLUMNS = ["sample_id", "news_id", "title", "abstract", "your_label"]
HELP = "  ".join(f"{k}={c}" for k, c in KEYMAP.items()) + "  s=skip  b=back  q=quit"


def load_sheet(path):
    sheet = read_sheet(path)
    missing = set(COLUMNS) - set(sheet.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns {sorted(missing)}")
    if sheet["sample_id"].duplicated().any():
        raise ValueError(f"{path.name} has duplicate sample_id values")
    return sheet


def save_sheet(sheet, path):
    tmp = path.with_name(path.name + ".tmp")
    sheet.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)  # atomic: an interrupted save never leaves a half-written sheet


def backup_sheet(path, backup_dir, now=None):
    now = now or datetime.now(timezone.utc)
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"{path.stem}.{now.strftime('%Y%m%d-%H%M%S')}{path.suffix}"
    shutil.copy2(path, dest)
    return dest


def next_unlabelled(sheet, after=-1):
    """First unlabelled row after `after`, wrapping around; None if every row is labelled."""
    empty = np.flatnonzero((sheet["your_label"] == "").to_numpy())
    if not len(empty):
        return None
    later = empty[empty > after]
    return int(later[0] if len(later) else empty[0])


def render(sheet, pos, width=100):
    row = sheet.iloc[pos]
    done = int((sheet["your_label"] != "").sum())
    head = f"item {pos + 1}/{len(sheet)}   labelled {done}/{len(sheet)}   (sample_id {row['sample_id']})"
    lines = ["", "=" * width, head]
    if row["your_label"]:
        lines.append(f"your current label: {row['your_label']}")
    lines += ["", "TITLE:"] + textwrap.wrap(row["title"], width)
    lines += ["", "ABSTRACT:"] + (textwrap.wrap(row["abstract"], width) or ["(no abstract)"])
    lines += ["", HELP]
    return "\n".join(lines)


def run_session(path, read_key, out=print, backup_dir=None, now=None):
    """Label `path` interactively. read_key() returns one key. -> {'labelled_now', 'labelled_total', 'backup'}."""
    sheet = load_sheet(path)
    pos = next_unlabelled(sheet)
    history, backup, labelled_now = [], None, 0
    if pos is None:
        out("Every item is already labelled.")
    while pos is not None:
        out(render(sheet, pos))
        key = (read_key() or "").strip().lower()[:1]
        if key in KEYMAP:
            if backup is None and backup_dir is not None:
                backup = backup_sheet(path, backup_dir, now)
            sheet.iat[pos, sheet.columns.get_loc("your_label")] = KEYMAP[key]
            save_sheet(sheet, path)
            labelled_now += 1
            history.append(pos)
            pos = next_unlabelled(sheet, pos)
            if pos is None:
                out("\nAll items labelled. Thank you.")
        elif key == "s":
            history.append(pos)
            nxt = next_unlabelled(sheet, pos)
            if nxt == pos:
                out("Only this item is left unlabelled.")
            pos = nxt
        elif key == "b":
            if history:
                pos = history.pop()
            else:
                out("Nothing to go back to.")
        elif key in ("q", "\x03"):
            out("Saved. Run the tool again to continue where you stopped.")
            break
        else:
            out(f"Unknown key {key!r}. {HELP}")
    total = int((sheet["your_label"] != "").sum())
    return {"labelled_now": labelled_now, "labelled_total": total, "of": len(sheet), "backup": backup}


def make_relabel_sample(sheet_path, out_path, n=RELABEL_N, seed=SEED, now=None):
    """Seeded random subset of a COMPLETE sheet with empty labels, for blind re-labelling a day later."""
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists; it is never regenerated")
    sheet = load_sheet(sheet_path)
    problems = sheet_problems(sheet)
    if problems:
        raise ValueError(f"finish {sheet_path.name} first: {problems[0]}")
    sub = sheet.sample(n=min(n, len(sheet)), random_state=seed)[COLUMNS].copy()
    sub["your_label"] = ""
    save_sheet(sub.reset_index(drop=True), out_path)
    now = now or datetime.now(timezone.utc)
    meta = {"created_utc": now.isoformat(timespec="seconds"), "source": sheet_path.name, "n": len(sub), "seed": seed,
            "min_hours_before_relabel": RELABEL_MIN_HOURS}
    meta_path(out_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return sub


def meta_path(out_path):
    return out_path.with_name(out_path.stem + "_meta.json")


def hours_since_created(out_path, now=None):
    meta = json.loads(meta_path(out_path).read_text(encoding="utf-8"))
    created = datetime.fromisoformat(meta["created_utc"])
    return ((now or datetime.now(timezone.utc)) - created).total_seconds() / 3600
