"""Download SemEval-2007 Task 14 Affective Text (Strapparava & Mihalcea, 2007) and verify its checksum.

Reads:  the SemEval archive (internet, needs curl; sha256 checked).
Writes: data/external/semeval2007/ (git-ignored).
Run:    python scripts/01_05_download_semeval.py
"""
import _bootstrap  # noqa: F401
from src.config import SEMEVAL_DIR, SEMEVAL_SPLITS
from src.semeval import download_archive, load_semeval, split_files


def main():
    """Download and unpack the archive if it is missing, then check the files."""
    if not all(p.exists() for s in SEMEVAL_SPLITS for p in split_files(SEMEVAL_DIR, s)):
        print(f"[download] {download_archive(SEMEVAL_DIR).name}")
    data = load_semeval(SEMEVAL_DIR)
    print({s: int((data["split"] == s).sum()) for s in SEMEVAL_SPLITS}, "headlines, ids match the gold files")


if __name__ == "__main__":
    main()
