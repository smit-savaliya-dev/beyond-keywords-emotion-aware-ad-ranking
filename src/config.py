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

# Congruence (fit only): user profile from the HISTORY column, never from impression clicks.
CONGRUENCE_DECAY = 0.9
CONGRUENCE_MAX_HISTORY = 50
CONGRUENCE_EPS = 1e-6  # smoothing for negative KL

# Validation only: NRC shares -> 6 classes (trust, anticipation not mapped; no mapped hits -> neutral).
NRC_SIX_MAP = {"joy": ["nrc_joy"], "sadness": ["nrc_sadness"], "anger": ["nrc_anger", "nrc_disgust"],
               "fear": ["nrc_fear"], "surprise": ["nrc_surprise"]}
# (vector version, measure, status). Personal = measure(user profile) - measure(corpus-mean profile).
CONGRUENCE_MEASURES = [
    ("pct", "cos", "main"),                       # 8 go_pct_* percentiles, cosine
    ("six", "cos", "variant"),                    # weighted 6-class distribution, cosine
    ("six", "negkl", "variant"),                  # -KL(item || profile) on the 6-class distribution
    ("valence", "absdist", "variant"),            # -|profile go_vad_valence - item go_vad_valence|
    ("pct", "negkl", "exploratory, not pre-declared"),
]
CONGRUENCE_CHUNK = 500_000  # candidate rows per chunk

# Hand labelling (scripts/label_tool.py). Keys 1-6 follow SIX_CLASSES order: joy, sadness, anger, fear, surprise, neutral.
LABEL_SHEET = ANNOTATION / "manual_sample.csv"
RELABEL_SHEET = ANNOTATION / "relabel_sample.csv"
LABEL_BACKUPS = ANNOTATION / "backups"
RELABEL_N = 30
RELABEL_MIN_HOURS = 24

# Model runs: run log, frozen settings (written by --tune), decision log checked by --final-dev.
RUNS_LOG = RESULTS / "tables" / "runs_log.csv"
FROZEN_DIR = RESULTS / "frozen"
DECISIONS = ROOT / "docs" / "decisions.md"
METRIC_COLUMNS = ["auc", "mrr", "ndcg@5", "ndcg@10", "p@5", "p@10", "r@5", "r@10", "hr@5", "hr@10"]

# Baselines (choose settings on valid by SELECTION_METRIC; dev once with frozen settings).
SELECTION_METRIC = "auc"
LEAKAGE_AUC_WARN = 0.75  # a no-emotion baseline above this on valid suggests leakage
M0_NAME, M1_NAME = "M0_popularity", "M1_tfidf"
M0_SMOOTHING_GRID = [1, 10, 100]  # a in (clicks + a*p) / (shown + a)
M1_GRID = {"text": ["title", "text"], "sublinear_tf": [False, True], "aggregation": ["mean", "decay"]}  # 8 configs
TFIDF_MIN_DF = 2                   # fixed, not tuned
TFIDF_STOP_WORDS = "english"       # fixed, not tuned

# Threshold sensitivity (pre-declared before the first run, 2026-10-26). Non-neutral class if score > t, else neutral.
THRESHOLDS_RAW = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]       # max raw sigmoid over the 27 non-neutral labels (weighted)
THRESHOLDS_PCT = [0.80, 0.85, 0.90, 0.95, 0.99]        # max go_pct_* over the 8 emotions (secondary view)
MIN_CLASS_COUNT = 5                                   # per-class results with fewer hand labels are flagged unreliable

# Emotion spec v1 candidates (D27). Selection: fit both-level significance + same-sign replication on valid.
SPEC_ITEM_CANDIDATES = ([f"go_pct_{e}" for e in GO_FEATURES] + ["go_emotionality", "go_vad_valence", "go_vad_dominance",
                        "go_vad_arousal", "vad_valence", "nrc_trust", "nrc_anticipation"]
                        + ["vad_dominance", "go_dist_optimism", "go_dist_nervousness", "go_dist_fear"])  # D27 add-ons
SPEC_CONGRUENCE_CANDIDATES = ["cong_pct_cos", "cong_pct_cos_personal", "cong_six_negkl", "cong_six_negkl_personal",
                              "cong_valence_absdist", "cong_valence_absdist_personal"]
EMOTION_SPEC_DIR = ROOT / "config"
EMOTION_SPEC_VERSION = 2  # bump (new file emotion_spec_v2.json) for any change to the hashed content

# Emotion layer v2 (D30): grid on saved raw scores; selection rule constants.
V2_TITLE_SHARES = [1.0, 0.7, 0.6, 0.5, 1 / 3]  # title share; abstract share = 1 - share; 1/3 = current v1 weighting
V2_T_GRID = [0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
V2_NEUTRAL_PREFER, V2_NEUTRAL_MAX = 30.0, 40.0  # % neutral: preferred tier, then hard limit
V2_VALID_TOLERANCE = 0.9                        # valid personal dz must stay >= this share of the current value
V2_CURRENT_VALID_DZ = 0.23715                   # cong_pct_cos_personal on valid, v1 (emotion_spec_selection.csv)
V2_MAIN_MEASURE = [("pct", "cos", "main")]
V2_EXAMPLES_PER_CLASS = 10
V2_ITEM_EXTRA = ["go_intensity"]                # candidate added to the v1 item list for spec v2 (D32)

# Context windows (D31). skip = newest items dropped first; size = items kept; decay 1.0 = plain mean.
CONTEXT_WINDOWS = {
    "long_50": {"skip": 0, "size": 50, "decay": 0.9},
    "short_5": {"skip": 0, "size": 5, "decay": 1.0},
    "short_3": {"skip": 0, "size": 3, "decay": 1.0},
    "short_1": {"skip": 0, "size": 1, "decay": 1.0},
}
CONTEXT_CONTROL = {"older_5": {"skip": 5, "size": 5, "decay": 1.0}}  # same size as short_5, older items
CONTEXT_CONTROL_MIN_HISTORY = 10

# Story charts (base-v2 Step 3).
STORY_DIR = FIGURES / "story"
STORY_TOP_CATEGORIES = 10
STORY_CHARTS = ["1_emotion_mix", "2_neutral_vs_threshold", "3_title_vs_abstract_intensity", "4_emotion_by_category",
                "5_valence_arousal_map", "6_clicked_vs_not_effects", "7_congruence_effects", "8_short_vs_long_context"]

# M5-all robustness variant (D37): all 19 item candidates; go_intensity stays out (Q05).
M5_ROBUSTNESS_ITEM_FEATURES = list(SPEC_ITEM_CANDIDATES)

# Reporting-label variants (D38): GoEmotions labels ignored for emotion presence and the dominant class.
LABEL_VARIANTS = {"v2": [], "v2b": ["approval", "disapproval"], "v2c": ["approval", "disapproval", "realization"]}

RECENCY_WINDOWS = [1, 3, 5]  # true recency (D39): last k earlier log clicks, no decay
PERCENTILE_MAX_DZ_CHANGE = 0.02  # D40: larger max |dz change| goes to open_questions.md

# Results snapshot (tool: scripts/snapshot_results.py). Copies are git-ignored; the manifest is tracked.
TABLES = RESULTS / "tables"
SNAPSHOT_COPY = ROOT / "data" / "snapshots" / "pre_cleanup"
SNAPSHOT_MANIFEST = RESULTS / "archive" / "pre_cleanup_snapshot.json"
