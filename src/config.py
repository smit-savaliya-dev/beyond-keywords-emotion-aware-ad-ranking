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
EMOTION_MAX_LENGTH = 128  # combined `text` baseline (news_emotion.parquet)

# Weighted emotion layer: title and abstract scored separately, then combined.
TITLE_WEIGHT = 0.3
ABSTRACT_WEIGHT = 0.6  # normalised by the sum -> 1/3 title, 2/3 abstract; empty abstract -> title only
TITLE_MAX_LENGTH = 128
ABSTRACT_MAX_LENGTH = 256  # 0.48% of abstracts exceed 128 tokens, 0.055% exceed 256 (token_lengths.json)
TITLE_BATCH_SIZE = 64
ABSTRACT_BATCH_SIZE = 64  # smoke: batch 32 x 256 tokens peaked at 0.73 of 6 GB

# GoEmotions features: name -> source labels (2+ labels are combined by noisy-OR).
GO_FEATURES = {
    "curiosity": ["curiosity"],
    "desire": ["desire"],
    "excitement": ["excitement"],
    "joy": ["joy", "amusement"],
    "optimism": ["optimism"],
    "nervousness": ["nervousness"],  # anxiety
    "annoyance": ["annoyance"],      # frustration
    "fear": ["fear"],
}

# NRC lexicons: research-only licence, never commit (data/ is git-ignored).
EXTERNAL = ROOT / "data" / "external"
NRC_DIR = EXTERNAL / "nrc"
EMOLEX_PATH = NRC_DIR / "NRC-Emotion-Lexicon" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
VAD_PATH = NRC_DIR / "NRC-VAD-Lexicon-v2.1" / "NRC-VAD-Lexicon-v2.1.txt"
EMOLEX_URL = "https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm"
VAD_URL = "https://saifmohammad.com/WebPages/nrc-vad.html"
NRC_EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
NRC_SENTIMENTS = ["negative", "positive"]
VAD_DIMS = ["valence", "arousal", "dominance"]
TOKEN_PATTERN = r"[a-z]+"
# GoEmotions label -> VAD term, used only if the label word itself is missing from the VAD lexicon.
GO_VAD_OVERRIDES = {}

# Official MIND-small sizes: catches partial downloads and parsing bugs.
OFFICIAL_COUNTS = {"news_items": 65238, "train_impressions": 156965, "dev_impressions": 73152}

# Diagnostics: (GoEmotions feature, lexicon feature, rows used) for cross-source correlations.
CROSS_SOURCE_PAIRS = [
    ("go_joy", "nrc_joy", "nrc_has_hits"),
    ("go_excitement", "nrc_joy", "nrc_has_hits"),
    ("go_fear", "nrc_fear", "nrc_has_hits"),
    ("go_nervousness", "nrc_fear", "nrc_has_hits"),
    ("go_annoyance", "nrc_anger", "nrc_has_hits"),
    ("go_optimism", "nrc_anticipation", "nrc_has_hits"),
    ("go_emotionality", "nrc_coverage", None),
    ("go_vad_valence", "vad_valence", "vad_has_hits"),
    ("go_vad_arousal", "vad_arousal", "vad_has_hits"),
    ("go_vad_dominance", "vad_dominance", "vad_has_hits"),
]
LOW_MASS_THRESHOLD = 0.1  # go_emotion_mass below this: go_dist_* mostly reflects label floors

# Phase 5 signal analysis (fit only, descriptive).
SIGNAL_ALPHA = 0.05
SIGNAL_TOP_CATEGORIES = 6  # by fit clicks
SIGNAL_CORE_FEATURES = (
    [f"go_{e}" for e in GO_FEATURES] + ["go_neutral", "go_emotionality", "go_emotion_mass"]
    + [f"go_vad_{d}" for d in VAD_DIMS]
    + [f"nrc_{e}" for e in NRC_EMOTIONS] + ["nrc_positive", "nrc_negative", "nrc_coverage"]
    + [f"vad_{d}" for d in VAD_DIMS]
)
