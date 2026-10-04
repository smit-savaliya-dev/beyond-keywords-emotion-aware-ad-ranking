"""One command for the whole base: tests -> data -> harness check -> EDA. Stops at the first failure.

  python scripts/run_base.py
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-official-check", action="store_true")
    args = ap.parse_args()
    extra = ["--skip-official-check"] if args.skip_official_check else []
    steps = [
        ("Unit tests", ["-m", "pytest", "-q"]),
        ("Download MIND-small (skips if present)", ["scripts/01_download_mind.py"]),
        ("Parse and verify data", ["scripts/02_prepare_mind.py", *extra]),
        ("Evaluation harness sanity (valid + dev)", ["scripts/03_harness_sanity.py"]),
        ("EDA figures", ["scripts/07_eda.py"]),
    ]
    start = time.time()
    for i, (name, cmd) in enumerate(steps, 1):
        print(f"\n===== [{i}/{len(steps)}] {name} =====", flush=True)
        if subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode != 0:
            raise SystemExit(f"\nSTOPPED at step {i}: {name}")
    print(f"\nBASE OK ({time.time() - start:.0f}s). Figures: results/figures/")


if __name__ == "__main__":
    main()
