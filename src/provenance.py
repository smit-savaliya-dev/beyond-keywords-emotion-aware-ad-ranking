import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata

from .config import ROOT

PACKAGES = ("pandas", "numpy", "scipy", "scikit-learn", "pyarrow", "torch", "transformers")


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def run_info():
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    for pkg in PACKAGES:
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return info
