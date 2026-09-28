"""Per-impression ranking metrics, matching the official MIND definitions.

Every metric is computed for ONE impression (labels + scores of its candidates).
evaluate.py averages them over impressions and keeps the per-impression values
for paired significance tests.
"""
import numpy as np
from scipy.stats import rankdata


def auc(labels, scores):
    """Chance a clicked item scores higher than a non-clicked one. Ties count as 0.5 (same as sklearn)."""
    labels = np.asarray(labels, dtype=float)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores)  # average rank for ties
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ranked_labels(labels, scores, rng=None):
    """Labels sorted by score (highest first). Ties are broken randomly so that
    no model gets free credit from the order items were logged in."""
    rng = rng if rng is not None else np.random.default_rng(0)
    scores = np.asarray(scores, dtype=float)
    tiebreak = rng.random(len(scores))
    order = np.lexsort((tiebreak, -scores))  # last key = primary: score descending
    return np.asarray(labels, dtype=float)[order]


def impression_metrics(labels, scores, ks=(5, 10), rng=None):
    labels = np.asarray(labels, dtype=float)
    y = ranked_labels(labels, scores, rng)
    n_click = y.sum()
    ranks = np.arange(1, len(y) + 1)
    discounts = 1.0 / np.log2(ranks + 1)
    ideal = np.sort(labels)[::-1]

    out = {
        "auc": auc(labels, scores),
        # official MIND MRR: average of 1/rank over ALL clicked items
        "mrr": (y / ranks).sum() / n_click if n_click else np.nan,
    }
    for k in ks:
        dcg = ((2 ** y[:k] - 1) * discounts[:k]).sum()
        idcg = ((2 ** ideal[:k] - 1) * discounts[:k]).sum()
        hits = y[:k].sum()
        out[f"ndcg@{k}"] = dcg / idcg if idcg else np.nan
        out[f"p@{k}"] = hits / k              # offline CTR of a K-slot display
        out[f"r@{k}"] = hits / n_click if n_click else np.nan
        out[f"hr@{k}"] = float(hits > 0)      # impression-level engagement
    return out
