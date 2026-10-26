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
