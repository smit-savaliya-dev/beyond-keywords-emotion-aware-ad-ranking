"""All paths and fixed settings live here, so every script uses the same values."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"              # unzipped MIND files (never edit these)
PROCESSED = ROOT / "data" / "processed"  # parquet files we create
RESULTS = ROOT / "results"

MIND_URLS = {
    "train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
    "dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
}

# MIND files have NO header row, so we name the columns ourselves.
NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract",
             "url", "title_entities", "abstract_entities"]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"     # e.g. 11/15/2019 8:55:22 AM

SEED = 42          # used for random baselines and metric tie-breaking
K_VALUES = (5, 10) # K for nDCG@K, Precision@K, Recall@K, HitRate@K
