"""Results snapshot: copy tables and emotion outputs, then prove a later rebuild reproduces them (floats within 1e-9).
Volatile metadata (run info, timestamps) is ignored; everything else must match."""
import json
import shutil

import numpy as np
import pandas as pd

from .provenance import sha256_file

TOL = 1e-9
IGNORED_KEYS = {"run_info", "frozen_utc", "timestamp_utc", "raw_sha256"}
IGNORED_FILES = {"runs_log.csv"}  # append-only: checked by prefix instead


def snapshot_files(roots):
    """{relative name: path} of csv/json/parquet files under each (label, folder)."""
    files = {}
    for label, folder in roots:
        for path in sorted(folder.glob("*")):
            if path.suffix in (".csv", ".json", ".parquet") and path.is_file():
                files[f"{label}/{path.name}"] = path
    return files


def key_numbers(path):
    """Shape and numeric column sums of a table: a readable fingerprint next to the hash."""
    if path.suffix == ".json":
        return {"keys": len(json.loads(path.read_text(encoding="utf-8")))}
    t = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    nums = t.select_dtypes("number")
    return {"rows": int(len(t)), "columns": int(t.shape[1]),
            "numeric_sums": {c: round(float(nums[c].sum()), 6) for c in list(nums.columns)[:40]}}


def take(roots, copy_dir):
    """Copy every file to copy_dir and return the manifest {name: {sha256, key numbers}}."""
    copy_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, path in snapshot_files(roots).items():
        dest = copy_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        manifest[name] = {"sha256": sha256_file(path), **key_numbers(path)}
    return manifest


def _same_value(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        keys = (set(a) | set(b)) - IGNORED_KEYS
        return all(k in a and k in b and _same_value(a[k], b[k]) for k in keys)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return (a != a and b != b) or abs(a - b) <= TOL
    return a == b


def _cell(e):
    return tuple(e) if isinstance(e, (np.ndarray, list)) else e


def _same_frame(a, b):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(x):
            u, v = x.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64)
            if (np.isnan(u) != np.isnan(v)).any() or np.nanmax(np.abs(u - v), initial=0.0) > TOL:
                return False
        else:
            for e, f in zip(x.astype(object), y.astype(object)):
                e, f = _cell(e), _cell(f)
                if e != f and not (e != e and f != f):
                    return False
    return True


def same_file(old, new):
    """Equal after ignoring volatile metadata; floats within TOL."""
    if sha256_file(old) == sha256_file(new):
        return True
    if old.suffix == ".json":
        return _same_value(json.loads(old.read_text(encoding="utf-8")), json.loads(new.read_text(encoding="utf-8")))
    read = pd.read_csv if old.suffix == ".csv" else pd.read_parquet
    return _same_frame(read(old), read(new))


def compare(manifest, copy_dir, roots):
    """-> list of problems (empty = every snapshot file is reproduced). New files are allowed."""
    current = snapshot_files(roots)
    problems = []
    for name in manifest:
        base = name.split("/", 1)[1]
        if base in IGNORED_FILES:
            continue
        if name not in current:
            problems.append(f"missing: {name}")
        elif not same_file(copy_dir / name, current[name]):
            problems.append(f"different: {name}")
    return problems
