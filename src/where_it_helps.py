"""RQ5, descriptive: where one model differs from another on dev (D56, D68). Nothing here selects or changes a model.

Inputs: per-impression AUC differences, histories and item labels. Outputs: breakdown tables and example impressions.
"""
import numpy as np
import pandas as pd

from .config import RQ5_MIN_CANDIDATES, RQ5_N_EXAMPLES, RQ5_TOP_CATEGORIES
from .congruence import user_profiles
from .emotion import SIX_CLASSES
from .stats import paired_test

EMOTIONS = [c for c in SIX_CLASSES if c != "neutral"]
HISTORY_BINS = [(0, 0, "0"), (1, 5, "1-5"), (6, 20, "6-20"), (21, None, "21+")]


def history_bin(n):
    """History-length bin name for each count."""
    n = np.asarray(n)
    out = np.empty(len(n), dtype=object)
    for low, high, name in HISTORY_BINS:
        out[(n >= low) & (n <= (n.max() if high is None else high))] = name
    return out


def dominant_emotion(histories, labels):
    """Per profile: argmax of decay-weighted emotion shares over history labels. neutral items carry no weight.
    -> (dominant class, known history items used)."""
    onehot = np.stack([(np.asarray(labels) == e).astype(np.float64) for e in EMOTIONS], axis=1)
    profiles, used = user_profiles(histories, onehot, np.zeros(len(EMOTIONS)))
    out = np.array(EMOTIONS, dtype=object)[profiles.argmax(axis=1)]
    out[profiles.sum(axis=1) == 0] = "neutral"
    out[used == 0] = "cold start"
    return out, used


def top_categories(clicked_ids, category_of, n=RQ5_TOP_CATEGORIES):
    """The n categories with most clicks (called on fit, so dev does not choose them)."""
    return pd.Series(clicked_ids).map(category_of).value_counts().head(n).index.tolist()


def clicked_category(truth, category_of, top):
    """Per impression: category of the first clicked item by position, or 'other' outside `top`. truth: position order."""
    first = truth[truth["label"] == 1].drop_duplicates("imp_key").set_index("imp_key")["news_id"]
    cat = first.map(category_of)
    return cat.where(cat.isin(top), "other")


def breakdown(diff, labels, name, order=None):
    """diff: per-impression M3 - M2 AUC; labels: aligned groups. -> n, mean, 95% CI, dz per group (no significance claims)."""
    t = pd.DataFrame({"diff": np.asarray(diff, dtype=np.float64), "group": np.asarray(labels)})
    rows = []
    for g in order or sorted(t["group"].unique()):
        d = t.loc[t["group"] == g, "diff"].to_numpy()
        if len(d):
            r = paired_test(np.zeros(len(d)), d)
            rows.append({"breakdown": name, "group": g, "n": r["n"], "mean_diff": r["mean_diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                         "dz": r["dz"], "pct_m3_better": r["pct_b_better"]})
    return pd.DataFrame(rows)


def example_impressions(diff, sizes, n=RQ5_N_EXAMPLES, min_candidates=RQ5_MIN_CANDIDATES):
    """diff, sizes: Series indexed by imp_key. -> (n most helped, n most hurt), long lists only, ties by imp_key."""
    t = pd.DataFrame({"diff": diff, "size": sizes})
    t = t[t["size"] >= min_candidates].sort_index(kind="mergesort").sort_values("diff", kind="mergesort")
    return t.index[::-1][:n].tolist(), t.index[:n].tolist()
