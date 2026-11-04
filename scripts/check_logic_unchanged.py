"""Prove the tidy-up changed no logic: every .py file has the same normalised syntax tree as at the base commit.

  python scripts/check_logic_unchanged.py --take    # once, on a clean tree: record the base commit and a hash per file
  python scripts/check_logic_unchanged.py --check   # compare today's files with the base commit (old names mapped to new)

Docstrings and comments are ignored. Renames, deleted files and the few reviewed exceptions are listed in
results/archive/tidy_renames.json.
"""
import argparse
import json
import subprocess
import sys

import _bootstrap  # noqa: F401
from src.config import RESULTS, ROOT
from src.logic_check import differences, digest, normalised_statements

BASELINE = RESULTS / "archive" / "tidy_ast_baseline.json"
RENAMES = RESULTS / "archive" / "tidy_renames.json"


def git(*args):
    """Run a git command in the project root and return its output."""
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout


def python_files():
    """Every tracked .py file."""
    return sorted(p for p in git("ls-files", "*.py").split("\n") if p)


def main():
    """Record the baseline (--take) or compare the files with it (--check)."""
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--take", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.take:
        if git("status", "--porcelain", "--", "*.py").strip():
            sys.exit("commit or stash .py changes first: the baseline must describe the base commit")
        files = {p: digest(normalised_statements((ROOT / p).read_text(encoding="utf-8"))) for p in python_files()}
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps({"base_commit": git("rev-parse", "HEAD").strip(), "files": files}, indent=1), encoding="utf-8")
        print(f"[ok] {len(files)} files at {git('rev-parse', '--short', 'HEAD').strip()} -> {BASELINE.relative_to(ROOT)}")
        return

    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    plan = json.loads(RENAMES.read_text(encoding="utf-8")) if RENAMES.exists() else {}
    renames, deleted = plan.get("renames", {}), plan.get("deleted", [])
    allowed, added_ok = plan.get("allowed", {}), plan.get("added", {})
    modules = {o[len("src/"):-3]: n[len("src/"):-3] for o, n in renames.items() if o.startswith("src/")}
    stems = {o.split("/")[-1][:-3]: n.split("/")[-1][:-3] for o, n in renames.items() if o.startswith("scripts/")}
    stems.update({o: n for o, n in renames.items() if not o.startswith("scripts/")})  # path strings such as "src/old.py"
    problems, checked = [], 0
    for old, recorded in base["files"].items():
        old_source = git("show", f"{base['base_commit']}:{old}")
        if digest(normalised_statements(old_source)) != recorded:
            problems.append(f"baseline mismatch for {old}")
            continue
        if old in deleted:
            continue
        new = renames.get(old, old)
        if not (ROOT / new).exists():
            problems.append(f"missing: {old} -> {new} (not listed as deleted)")
            continue
        diff = differences(old_source, (ROOT / new).read_text(encoding="utf-8"), modules, stems, allowed.get(new, ()), added_ok.get(new, ()))
        checked += 1
        problems += [f"logic differs in {new}: {d}" for d in diff]
    added = sorted(set(python_files()) - {renames.get(o, o) for o in base["files"]})
    for p in problems:
        print(" -", p)
    print(f"{checked} files compared ({len(renames)} renamed, {len(deleted)} deleted); {len(problems)} problems")
    print(f"new files (not in the baseline): {', '.join(added) or 'none'}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
