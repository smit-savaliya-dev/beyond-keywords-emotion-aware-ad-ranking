"""True recency (D39): emotional context from the user's clicks in EARLIER impressions of the log, strictly before each impression.
Analysis only; only fit and valid are ever read. Standard MIND protocol uses the history column alone (see Q09)."""
import numpy as np
import pandas as pd

LOG_PARTS = ("fit", "valid")


def load_log(processed):
    """Impression rows of fit and valid only (dev is never read). -> (click rows, one row per impression)."""
    rows = pd.read_parquet(processed / "impressions.parquet", columns=["imp_key", "part", "user_id", "time", "news_id", "label"],
                           filters=[("part", "in", list(LOG_PARTS))])
    if not set(rows["part"].astype(str)) <= set(LOG_PARTS):
        raise ValueError("rows of other parts loaded")
    rows = rows.assign(imp_key=rows["imp_key"].astype(str), user_id=rows["user_id"].astype(str),
                       news_id=rows["news_id"].astype(str))
    clicks = rows.loc[rows["label"] == 1, ["user_id", "time", "news_id"]].reset_index(drop=True)
    impressions = rows[["imp_key", "user_id", "time"]].drop_duplicates("imp_key").set_index("imp_key")
    return clicks, impressions


def earlier_click_lists(imp_users, imp_times, click_users, click_times, click_items):
    """For each impression: item rows the same user clicked at a strictly earlier time, oldest -> newest."""
    order = np.argsort(np.asarray(click_times), kind="stable")
    times, users, items = np.asarray(click_times)[order], np.asarray(click_users)[order], np.asarray(click_items)[order]
    by_user = {}
    for u, idx in pd.Series(np.arange(len(users))).groupby(users).indices.items():
        by_user[u] = (times[idx], items[idx])
    out = []
    empty = np.empty(0, dtype=np.int64)
    for user, t in zip(imp_users, imp_times):
        hit = by_user.get(user)
        out.append(empty if hit is None else hit[1][:np.searchsorted(hit[0], t, side="left")])
    return out


def repeated_candidates(groups, item_idx, lists):
    """True for candidates this user already clicked earlier: such an item matches itself, not an emotional context."""
    n_items = int(item_idx.max()) + 1 if len(item_idx) else 1
    seen = [np.asarray(h, dtype=np.int64) for h in lists]
    lengths = np.fromiter((len(h) for h in seen), dtype=np.int64, count=len(seen))
    if lengths.sum() == 0:
        return np.zeros(len(groups), dtype=bool)
    keys = np.repeat(np.arange(len(seen), dtype=np.int64), lengths) * n_items + np.concatenate(seen)
    return np.isin(np.asarray(groups, dtype=np.int64) * n_items + item_idx, keys)


def within_slope(groups, x, y):
    """Slope of y on x after removing each impression's mean (the comparison happens inside impressions)."""
    groups = np.asarray(groups)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n = np.bincount(groups)
    safe = np.where(n > 0, n, 1)
    dx = x - (np.bincount(groups, weights=x) / safe)[groups]
    dy = y - (np.bincount(groups, weights=y) / safe)[groups]
    denom = float(dx @ dx)
    return float(dx @ dy) / denom if denom > 0 else 0.0
