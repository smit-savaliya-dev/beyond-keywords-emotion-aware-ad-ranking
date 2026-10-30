"""User emotion profiles from click HISTORY only, and item-profile congruence (cosine, negative KL)."""
import numpy as np

from .config import CONGRUENCE_DECAY, CONGRUENCE_EPS, CONGRUENCE_MAX_HISTORY


def user_profiles(histories, vectors, fallback, decay=CONGRUENCE_DECAY, max_items=CONGRUENCE_MAX_HISTORY):
    """histories: per profile, item row indices oldest -> newest (-1 = unknown item, skipped).
    Recency-weighted mean of the last `max_items` known items, weight decay**k (k = 0 newest).
    Profiles with no known items get `fallback` (corpus mean). -> (profiles, n_items_used)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    rows, cols, weights = [], [], []
    for g, h in enumerate(histories):
        h = np.asarray(h, dtype=np.int64)
        h = h[h >= 0][-max_items:]
        if len(h):
            rows.append(np.full(len(h), g))
            cols.append(h)
            weights.append(decay ** np.arange(len(h) - 1, -1, -1, dtype=np.float64))
    n = len(histories)
    out = np.tile(np.asarray(fallback, dtype=np.float64), (n, 1))
    used = np.zeros(n, dtype=np.int64)
    if rows:
        r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(weights)
        wsum = np.bincount(r, weights=w, minlength=n)
        acc = np.stack([np.bincount(r, weights=w * vectors[c, j], minlength=n) for j in range(vectors.shape[1])], 1)
        has = wsum > 0
        out[has] = acc[has] / wsum[has, None]
        used = np.bincount(r, minlength=n)
    return out, used


def cosine_rows(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    denom = na * nb
    return np.where(denom > 0, (a * b).sum(axis=1) / np.where(denom > 0, denom, 1.0), 0.0)


def neg_kl_rows(item, profile, eps=CONGRUENCE_EPS):
    """-KL(item || profile), both smoothed by eps and renormalised to sum 1. 0 = identical, more negative = less alike."""
    p = np.asarray(item, dtype=np.float64) + eps
    q = np.asarray(profile, dtype=np.float64) + eps
    p /= p.sum(axis=1, keepdims=True)
    q /= q.sum(axis=1, keepdims=True)
    return -(p * np.log(p / q)).sum(axis=1)


def neg_abs_distance(item, profile):
    """-|profile - item| summed over dimensions (one dimension for valence). 0 = identical."""
    return -np.abs(np.asarray(item, dtype=np.float64) - np.asarray(profile, dtype=np.float64)).sum(axis=1)


MEASURE_FUNCS = {"cos": cosine_rows, "negkl": neg_kl_rows, "absdist": neg_abs_distance}


def history_index_lists(histories, item_index):
    """Per row: history news ids -> row positions in item_index (-1 = unknown), oldest -> newest."""
    lengths = np.fromiter((len(h) for h in histories), dtype=np.int64, count=len(histories))
    if lengths.sum() == 0:
        return [np.empty(0, dtype=np.int64) for _ in lengths]
    flat = np.concatenate([np.asarray(h, dtype=object) for h in histories if len(h)]).astype(str)
    pos = item_index.get_indexer(flat)
    return np.split(pos, np.cumsum(lengths)[:-1])


def window_histories(histories, skip=0, size=None):
    """Known history items per row after dropping the newest `skip`, keeping the newest `size` of the rest."""
    out = []
    for h in histories:
        h = np.asarray(h, dtype=np.int64)
        h = h[h >= 0]
        if skip:
            h = h[:-skip]
        out.append(h if size is None else h[-size:])
    return out


def congruence_features(groups, item_idx, histories, vectors, measures, chunk_size=None, shuffle_seed=None,
                        decay=CONGRUENCE_DECAY, max_items=CONGRUENCE_MAX_HISTORY):
    """Candidate-level congruence for each (version, measure, status) in `measures`.

    groups: profile row per candidate; item_idx: item row per candidate; histories: per profile row, item rows;
    vectors: {version: (n_items, d)}. Cold start uses the corpus mean (mean over all items, no clicks).
    With shuffle_seed: also _otheruser (profile of a random other profile row) and _userspecific (raw - _otheruser),
    a control that keeps the population's typical profile but breaks the link to the actual user.
    -> (features {name: float32 array}, n_history_items_used per profile row)."""
    from .config import CONGRUENCE_CHUNK
    chunk_size = chunk_size or CONGRUENCE_CHUNK
    groups, item_idx = np.asarray(groups), np.asarray(item_idx)
    profiles, means, used = {}, {}, None
    for version in {v for v, _, _ in measures}:
        mat = np.asarray(vectors[version], dtype=np.float64)
        if mat.ndim == 1:
            mat = mat[:, None]
        vectors = {**vectors, version: mat}
        means[version] = mat.mean(axis=0)
        profiles[version], used = user_profiles(histories, mat, means[version], decay, max_items)
    out, n = {}, len(item_idx)
    n_rows = len(histories)
    other = None if shuffle_seed is None else np.random.default_rng(shuffle_seed).permutation(n_rows)
    for version, measure, _ in measures:
        fn, mat, prof = MEASURE_FUNCS[measure], vectors[version], profiles[version]
        raw, gen = np.empty(n, dtype=np.float32), np.empty(n, dtype=np.float32)
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            item = mat[item_idx[s:e]]
            raw[s:e] = fn(item, prof[groups[s:e]])
            gen[s:e] = fn(item, np.broadcast_to(means[version], item.shape))
        name = f"cong_{version}_{measure}"
        out[name], out[f"{name}_generic"], out[f"{name}_personal"] = raw, gen, raw - gen
        if other is not None:
            oth = np.empty(n, dtype=np.float32)
            for s in range(0, n, chunk_size):
                e = min(s + chunk_size, n)
                oth[s:e] = fn(mat[item_idx[s:e]], prof[other[groups[s:e]]])
            out[f"{name}_otheruser"], out[f"{name}_userspecific"] = oth, raw - oth
    return out, used
