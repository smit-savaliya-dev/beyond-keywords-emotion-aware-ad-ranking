"""Download and unzip MIND-small into data/raw/ (skips what is already there)."""
import urllib.request
import zipfile

import _bootstrap  # noqa: F401
from src.config import MIND_URLS, RAW


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for split, url in MIND_URLS.items():
        out_dir = RAW / split
        zip_path = RAW / f"MINDsmall_{split}.zip"
        if list(out_dir.rglob("behaviors.tsv")):
            print(f"[skip] {split}: already unzipped")
            continue
        if not zip_path.exists():
            print(f"[download] {url}")
            urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
        if not list(out_dir.rglob("behaviors.tsv")):
            raise SystemExit(f"{split}: behaviors.tsv missing after unzip")
        print(f"[ok] {split}")


if __name__ == "__main__":
    main()
