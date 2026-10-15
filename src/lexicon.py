"""NRC EmoLex and NRC VAD loaders plus the tokeniser used for lexicon matching."""
import re

import numpy as np
import pandas as pd

from .config import (EMOLEX_PATH, EMOLEX_URL, NRC_DIR, NRC_EMOTIONS, NRC_SENTIMENTS, TOKEN_PATTERN, VAD_DIMS,
                     VAD_PATH, VAD_URL)
from .provenance import sha256_file

EMOLEX_COLUMNS = NRC_EMOTIONS + NRC_SENTIMENTS
_TOKEN_RE = re.compile(TOKEN_PATTERN)


def tokenize(text):
    """Lowercase regex word tokens for unigram lexicon lookup.

    Limitations: no negation handling ("not happy" counts as happy), no lemmatisation or stemming,
    apostrophes and hyphens split words ("trump's" -> "trump", "s"), multi-word entries are not matched.
    """
    return _TOKEN_RE.findall(str(text).lower())


def _require(path, what, url):
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}.\nDownload it from {url} (research-only licence), unzip it under "
            f"{NRC_DIR} so that this exact file exists, and never commit it.")


def _read_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        return [line.rstrip("\r\n").split("\t") for line in f if line.strip()]


def load_emolex(path=EMOLEX_PATH, log=print):
    """-> (DataFrame indexed by word with 0/1 int8 columns EMOLEX_COLUMNS, info). Long or wide format."""
    _require(path, "NRC Emotion Lexicon (EmoLex)", EMOLEX_URL)
    rows = _read_rows(path)
    head = [c.strip().lower() for c in rows[0]]
    if set(EMOLEX_COLUMNS) <= set(head):
        fmt = "wide"
        df = pd.DataFrame(rows[1:], columns=head)
        df = df.rename(columns={head[0]: "word"})[["word", *EMOLEX_COLUMNS]]
        long = df.melt(id_vars="word", var_name="emotion", value_name="flag")
    else:
        fmt = "long"
        if any(len(r) != 3 for r in rows):
            raise ValueError(f"{path.name}: expected 3 tab-separated columns (word, emotion, 0/1)")
        if not rows[0][2].strip().isdigit():
            rows = rows[1:]  # header line
        long = pd.DataFrame(rows, columns=["word", "emotion", "flag"])
    long = long.apply(lambda s: s.str.strip())
    long["word"] = long["word"].str.lower()
    long = long[long["word"] != ""]
    if not long["flag"].isin(["0", "1"]).all():
        raise ValueError(f"{path.name}: association flags other than 0/1 found")
    found = set(long["emotion"])
    if found != set(EMOLEX_COLUMNS):
        raise ValueError(f"{path.name}: emotions {sorted(found)} differ from expected {EMOLEX_COLUMNS}")
    long = long.drop_duplicates()
    if long.duplicated(["word", "emotion"]).any():
        raise ValueError(f"{path.name}: conflicting flags for the same word and emotion")
    table = long.pivot(index="word", columns="emotion", values="flag")
    if table.isna().any().any():
        raise ValueError(f"{path.name}: some words lack a flag for some emotions")
    table = table[EMOLEX_COLUMNS].astype("int8")
    table.columns.name = None
    info = {"path": str(path), "sha256": sha256_file(path), "format": fmt, "n_words": int(len(table)),
            "n_words_with_emotion": int((table[NRC_EMOTIONS].sum(axis=1) > 0).sum())}
    log(f"EmoLex: {path.name}: {fmt} format, {info['n_words']} words, "
        f"{info['n_words_with_emotion']} with at least one of the 8 emotions")
    return table, info


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_vad(path=VAD_PATH, log=print):
    """-> (DataFrame indexed by term with VAD_DIMS in [-1, 1], 0 = neutral; info). Scale detected from the values."""
    _require(path, "NRC VAD Lexicon", VAD_URL)
    rows = _read_rows(path)
    if any(len(r) != 4 for r in rows):
        raise ValueError(f"{path.name}: expected 4 tab-separated columns (term, valence, arousal, dominance)")
    has_header = not all(_is_number(x) for x in rows[0][1:])
    if has_header:
        names = [c.strip().lower() for c in rows[0][1:]]
        if names != VAD_DIMS:
            raise ValueError(f"{path.name}: header columns {names} differ from {VAD_DIMS}")
        rows = rows[1:]
    df = pd.DataFrame(rows, columns=["term", *VAD_DIMS])
    df["term"] = df["term"].str.strip().str.lower()
    try:
        values = df[VAD_DIMS].astype(float)
    except ValueError as e:
        raise ValueError(f"{path.name}: non-numeric VAD score ({e})") from None
    lo, hi = float(values.min().min()), float(values.max().max())
    if lo < 0 and lo >= -1 and hi <= 1:
        scale, rescaled = "-1..1", False
    elif lo >= 0 and hi <= 1:
        scale, rescaled = "0..1", True
        values = 2 * values - 1
    else:
        raise ValueError(f"{path.name}: VAD scores span {lo}..{hi}, neither 0..1 nor -1..1")
    df[VAD_DIMS] = values
    df = df.drop_duplicates()
    if df["term"].duplicated().any():
        raise ValueError(f"{path.name}: conflicting scores for the same term")
    table = df.set_index("term")[VAD_DIMS].astype(np.float64)
    n_multi = int(table.index.str.contains(" ").sum())
    info = {"path": str(path), "sha256": sha256_file(path), "header": has_header, "scale": scale,
            "rescaled_to_minus1_1": rescaled, "n_terms": int(len(table)),
            "n_unigrams": int(len(table)) - n_multi, "n_multiword": n_multi}
    log(f"VAD: {path.name}: header={has_header}, detected scale {scale}"
        f"{' (rescaled 2x-1 to -1..1)' if rescaled else ' (kept)'}, {info['n_terms']} terms "
        f"({info['n_unigrams']} unigrams, {n_multi} multi-word, multi-word not matched)")
    return table, info


def resolve_terms(words, vocab, overrides):
    """Map each word to itself if in vocab, else to overrides[word]. -> (mapping, overridden words)."""
    vocab = set(vocab)
    mapping, used, missing = {}, [], []
    for w in words:
        if w in vocab:
            mapping[w] = w
        elif overrides.get(w) in vocab:
            mapping[w] = overrides[w]
            used.append(w)
        else:
            missing.append(w)
    if missing:
        raise KeyError(f"terms missing from the lexicon with no usable override: {missing}; "
                       "add them to GO_VAD_OVERRIDES in src/config.py")
    return mapping, used
