"""M0 popularity (smoothed CTR) and M1 TF-IDF (history profile cosine). No emotion."""
import numpy as np
import pandas as pd
from scipy import sparse

from .config import CONGRUENCE_DECAY, CONGRUENCE_MAX_HISTORY, NEWS_COLS, RAW


# ---------- M0 popularity ----------

def item_ctr(stats_rows):
    """stats_rows: news_id, label of the statistics parts only. -> (per-item clicks/impressions, global CTR)."""
    g = stats_rows.groupby(stats_rows["news_id"].astype(str), observed=True)["label"].agg(clicks="sum", shown="size")
    p = float(stats_rows["label"].sum() / len(stats_rows))
    return g, p


def popularity_scores(candidates, ctr, p, a):
    """Beta-smoothed CTR (clicks + a*p) / (shown + a); items never shown in the statistics parts get p."""
    if a <= 0:
        raise ValueError("smoothing strength a must be positive")
    smoothed = (ctr["clicks"] + a * p) / (ctr["shown"] + a)
    ids = candidates["news_id"].astype(str)
    return ids.map(smoothed).fillna(p).to_numpy(dtype=np.float64)


def unseen_share(candidates, ctr):
    return float((~candidates["news_id"].astype(str).isin(ctr.index)).mean())


# ---------- M1 TF-IDF ----------

def train_news_ids(raw=RAW):
    """News ids listed in the TRAIN split's news.tsv (label-free corpus for fitting the vectorizer)."""
    from .data import read_tsv
    return set(read_tsv(raw / "train" / "news.tsv", NEWS_COLS)["news_id"])


def fit_tfidf(fit_texts, all_texts, sublinear_tf, min_df=2, stop_words="english"):
    """Fit on fit_texts only, transform all_texts. Rows are L2-normalised (empty text -> zero row)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(sublinear_tf=sublinear_tf, min_df=min_df, stop_words=stop_words, dtype=np.float32)
    vec.fit(fit_texts)
    return vec, vec.transform(all_texts).tocsr()


def profile_weights(histories, n_items, aggregation, decay=CONGRUENCE_DECAY, max_items=CONGRUENCE_MAX_HISTORY):
    """Sparse (n_profiles x n_items) weights over the last max_items known history items (oldest -> newest).
    aggregation 'mean': equal weights; 'decay': decay**k with k = 0 (weight 1) for the newest. Rows sum to 1."""
    if aggregation not in ("mean", "decay"):
        raise ValueError(f"unknown aggregation {aggregation!r}")
    rows, cols, vals = [], [], []
    for r, h in enumerate(histories):
        h = np.asarray(h, dtype=np.int64)
        h = h[h >= 0][-max_items:]
        if not len(h):
            continue
        w = np.ones(len(h)) if aggregation == "mean" else decay ** np.arange(len(h) - 1, -1, -1, dtype=np.float64)
        rows.append(np.full(len(h), r))
        cols.append(h)
        vals.append(w / w.sum())
    if not rows:
        return sparse.csr_matrix((len(histories), n_items))
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(histories), n_items))


def cosine_scores(profiles, items, groups, item_idx, chunk_size=200_000):
    """cosine(profiles[groups[i]], items[item_idx[i]]) per candidate, in sparse chunks; 0 if either vector is empty."""
    p_norm = np.sqrt(np.asarray(profiles.multiply(profiles).sum(axis=1))).ravel()
    i_norm = np.sqrt(np.asarray(items.multiply(items).sum(axis=1))).ravel()
    out = np.zeros(len(groups), dtype=np.float64)
    for s in range(0, len(groups), chunk_size):
        g, it = groups[s:s + chunk_size], item_idx[s:s + chunk_size]
        dot = np.asarray(profiles[g].multiply(items[it]).sum(axis=1)).ravel()
        denom = p_norm[g] * i_norm[it]
        out[s:s + chunk_size] = np.where(denom > 0, dot / np.where(denom > 0, denom, 1.0), 0.0)
    return out


def unique_profiles(behaviors):
    """behaviors: imp_key, user_id, history. -> (profile id per impression, history per unique (user, history))."""
    key = behaviors["user_id"].astype(str) + "|" + behaviors["history"].map(lambda h: " ".join(h))
    codes, uniques = pd.factorize(key)
    first = pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()
    return codes, behaviors["history"].to_numpy()[first]
