from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
ANNOTATION = ROOT / "annotation"

MIND_URLS = {
    "train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
    "dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
}

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract",
             "url", "title_entities", "abstract_entities"]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"

SEED = 42
K_VALUES = (5, 10)

EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
EMOTION_MAX_LENGTH = 128

# Official MIND-small sizes: catches partial downloads and parsing bugs.
OFFICIAL_COUNTS = {"news_items": 65238, "train_impressions": 156965, "dev_impressions": 73152}
