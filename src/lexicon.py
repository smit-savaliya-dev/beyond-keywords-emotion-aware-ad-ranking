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


STAT_COLUMNS = (["n_tokens", "emo_tokens", "pos_tokens", "neg_tokens", "vad_tokens"]
                + [f"n_{e}" for e in NRC_EMOTIONS] + [f"sum_{d}" for d in VAD_DIMS])
SHARE_COLUMNS = [f"nrc_{e}" for e in NRC_EMOTIONS]
VAD_COLUMNS = [f"vad_{d}" for d in VAD_DIMS]
RENORM_COLUMNS = SHARE_COLUMNS + VAD_COLUMNS          # undefined without hits: weights renormalised over fields with hits
PLAIN_COLUMNS = ["nrc_positive", "nrc_negative", "nrc_coverage", "vad_coverage"]  # token rates: plain field weights
LEXICON_COLUMNS = (SHARE_COLUMNS + ["nrc_positive", "nrc_negative", "nrc_coverage", "nrc_has_hits"]
                   + VAD_COLUMNS + ["vad_coverage", "vad_has_hits"])


def lexicon_counts(texts, emolex, vad):
    """Per-text token counts: EmoLex emotion/sentiment hits and VAD sums over matched unigrams (repeats count)."""
    emo = emolex[NRC_EMOTIONS].to_numpy()
    keep = emolex[EMOLEX_COLUMNS].to_numpy().sum(axis=1) > 0
    emo_map = {w: (row, int(p), int(n)) for w, row, p, n, k in
               zip(emolex.index, emo, emolex["positive"], emolex["negative"], keep) if k}
    uni = vad[~vad.index.str.contains(" ")]
    vad_map = dict(zip(uni.index, uni[VAD_DIMS].to_numpy()))
    out = np.zeros((len(texts), len(STAT_COLUMNS)))
    k = len(NRC_EMOTIONS)
    for i, text in enumerate(texts):
        tokens = tokenize(text)
        row = out[i]
        row[0] = len(tokens)
        for t in tokens:
            hit = emo_map.get(t)
            if hit is not None:
                flags, p, n = hit
                row[1] += flags.any()
                row[2] += p
                row[3] += n
                row[5:5 + k] += flags
            v = vad_map.get(t)
            if v is not None:
                row[4] += 1
                row[5 + k:] += v
    return pd.DataFrame(out, columns=STAT_COLUMNS)


def field_lexicon_features(counts):
    """Counts -> one field's features. Shares and VAD means are NaN when the field has no hits."""
    c = counts
    n = c["n_tokens"].to_numpy()
    safe = lambda x: np.where(x > 0, x, 1.0)
    emo = c[[f"n_{e}" for e in NRC_EMOTIONS]].to_numpy()
    total = emo.sum(axis=1)
    shares = np.where(total[:, None] > 0, emo / safe(total)[:, None], np.nan)
    out = pd.DataFrame(shares, columns=SHARE_COLUMNS)
    out["nrc_positive"] = c["pos_tokens"].to_numpy() / safe(n)
    out["nrc_negative"] = c["neg_tokens"].to_numpy() / safe(n)
    out["nrc_coverage"] = c["emo_tokens"].to_numpy() / safe(n)
    out["nrc_has_hits"] = (total > 0).astype("int8")
    m = c["vad_tokens"].to_numpy()
    sums = c[[f"sum_{d}" for d in VAD_DIMS]].to_numpy()
    out[VAD_COLUMNS] = np.where(m[:, None] > 0, sums / safe(m)[:, None], np.nan)
    out["vad_coverage"] = m / safe(n)
    out["vad_has_hits"] = (m > 0).astype("int8")
    return out


def combine_lexicon_fields(title, abstract, has_abstract, title_weight):
    """Weighted title/abstract lexicon features (rows aligned; abstract rows ignored where has_abstract is False).

    Shares and VAD means: weights renormalised over fields with hits, 0 when neither field has hits.
    Token rates (positive, negative, coverage): plain weights; no abstract -> title only.
    """
    has_abstract = np.asarray(has_abstract, dtype=bool)
    wt = np.where(has_abstract, title_weight, 1.0)
    wa = 1.0 - wt
    out = pd.DataFrame(index=title.index)
    for col in PLAIN_COLUMNS:
        a = np.where(has_abstract, abstract[col].to_numpy(), 0.0)
        out[col] = wt * title[col].to_numpy() + wa * a
    groups = [(SHARE_COLUMNS, "nrc_has_hits"), (VAD_COLUMNS, "vad_has_hits")]
    for cols, flag in groups:
        t_ok = title[flag].to_numpy() > 0
        a_ok = has_abstract & (abstract[flag].to_numpy() > 0)
        w_t, w_a = wt * t_ok, wa * a_ok
        norm = w_t + w_a
        t_vals = np.nan_to_num(title[cols].to_numpy())
        a_vals = np.nan_to_num(abstract[cols].to_numpy())
        vals = (w_t[:, None] * t_vals + w_a[:, None] * a_vals) / np.where(norm > 0, norm, 1.0)[:, None]
        out[cols] = np.where(norm[:, None] > 0, vals, 0.0)
        out[flag] = (t_ok | a_ok).astype("int8")
    return out[LEXICON_COLUMNS]


def top_matched_words(texts, vocab, n=30):
    """Most frequent tokens found in `vocab` (to spot lexicon artefacts)."""
    from collections import Counter
    vocab = set(vocab)
    counter = Counter(t for text in texts for t in tokenize(text) if t in vocab)
    return pd.DataFrame(counter.most_common(n), columns=["word", "count"])


def nrc_to_six(nrc, seed):
    """NRC share table -> 6-class label per row (validation only). Mapped scores per NRC_SIX_MAP; all zero (no hits,
    or trust/anticipation only) -> neutral; ties between top classes -> seeded random choice. -> (labels, tied)."""
    from .config import NRC_SIX_MAP
    classes = list(NRC_SIX_MAP)
    scores = np.column_stack([nrc[cols].to_numpy(dtype=np.float64).sum(axis=1) for cols in NRC_SIX_MAP.values()])
    top = scores.max(axis=1)
    is_top = np.isclose(scores, top[:, None], rtol=0, atol=1e-12) & (top[:, None] > 0)
    tied = is_top.sum(axis=1) > 1
    rng = np.random.default_rng(seed)
    labels = np.full(len(scores), "neutral", dtype=object)
    for i in np.flatnonzero(top > 0):
        choices = np.flatnonzero(is_top[i])
        labels[i] = classes[choices[0] if len(choices) == 1 else rng.choice(choices)]
    return pd.Series(labels, index=nrc.index), pd.Series(tied, index=nrc.index)
