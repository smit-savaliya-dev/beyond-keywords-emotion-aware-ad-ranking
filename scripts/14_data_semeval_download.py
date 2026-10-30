"""Download SemEval-2007 Task 14 Affective Text (Strapparava & Mihalcea, 2007) into data/external/semeval2007/ and verify it.

  python scripts/14_data_semeval_download.py
"""
import _bootstrap  # noqa: F401
from src.config import SEMEVAL_DIR, SEMEVAL_SPLITS
from src.semeval import download_archive, load_semeval, split_files


def main():
    if not all(p.exists() for s in SEMEVAL_SPLITS for p in split_files(SEMEVAL_DIR, s)):
        print(f"[download] {download_archive(SEMEVAL_DIR).name}")
    data = load_semeval(SEMEVAL_DIR)
    print({s: int((data["split"] == s).sum()) for s in SEMEVAL_SPLITS}, "headlines, ids match the gold files")


if __name__ == "__main__":
    main()
