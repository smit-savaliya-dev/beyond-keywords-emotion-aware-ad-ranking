"""Emotion layer v2: title-weight / threshold grid on saved raw scores, selection rule (D30), labels and intensity.

Labels are for reporting and validation only; model features stay soft (intensity is a continuous feature).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .click_signal import load_part_impressions
from .config import V2_NEUTRAL_MAX, V2_NEUTRAL_PREFER, V2_VALID_TOLERANCE
from .congruence import history_index_lists
from .emotion import GOEMOTIONS_LABELS, LABEL_TO_GROUP, SIX_CLASSES
from .emotion_features import combine_fields

NON_NEUTRAL = [label for label in GOEMOTIONS_LABELS if label != "neutral"]
_NN_GROUP = np.array([LABEL_TO_GROUP[label] for label in NON_NEUTRAL], dtype=object)
_NN_LABEL = np.array(NON_NEUTRAL, dtype=object)


def combine_share(title_raw, abstract_raw, share):
    """Raw 28 scores at title share `share` (1.0 = title only). Items without an abstract keep their title scores."""
    if share >= 1:
        out = title_raw[["news_id", *GOEMOTIONS_LABELS]].astype({c: np.float64 for c in GOEMOTIONS_LABELS})
        return out.assign(has_abstract=title_raw["news_id"].isin(abstract_raw["news_id"]).to_numpy())
    return combine_fields(title_raw, abstract_raw, share, 1 - share)


def intensity_table(raw, exclude=()):
    """Per item: intensity (max raw score over non-neutral labels, minus `exclude`), its label and 6-class group."""
    keep = [i for i, label in enumerate(NON_NEUTRAL) if label not in set(exclude)]
    scores = raw[[NON_NEUTRAL[i] for i in keep]].to_numpy(dtype=np.float64)
    best = scores.argmax(axis=1)
    return pd.DataFrame({"news_id": raw["news_id"].to_numpy(), "intensity": scores[np.arange(len(scores)), best],
                         "top_label": _NN_LABEL[keep][best], "group": _NN_GROUP[keep][best]})


def ecdf_percentiles(scores, reference):
    """(n, d) scores -> percentile rank in (0, 1] against the rows where `reference` is True, per column.
    Equals pandas rank(pct=True, method='average') for reference rows; other rows use the same ECDF."""
    scores = np.asarray(scores, dtype=np.float64)
    ref = np.sort(scores[np.asarray(reference, dtype=bool)], axis=0)
    out = np.empty_like(scores)
    n = len(ref)
    for j in range(scores.shape[1]):
        left = np.searchsorted(ref[:, j], scores[:, j], side="left")
        right = np.searchsorted(ref[:, j], scores[:, j], side="right")
        out[:, j] = np.minimum((left + right + 1) / 2 / n, 1.0)
    return out


def threshold_labels(table, t):
    """6-class group if intensity >= t, else neutral."""
    return np.where(table["intensity"].to_numpy() >= t, table["group"].to_numpy(), "neutral")


def class_shares(labels):
    """% of items per class, in SIX_CLASSES order."""
    s = pd.Series(labels).value_counts(normalize=True).reindex(SIX_CLASSES, fill_value=0)
    return {f"pct_{c}": 100 * float(v) for c, v in s.items()}


def select_setting(grid, current_valid_dz):
    """D30. grid columns: share, t, pct_neutral, fit_dz, valid_dz. -> (chosen row or None, tier, passing rows).
    Criteria: neutral tier, valid dz gate, max of min(fit, valid); ties at 4 decimals go to the higher t."""
    gate = V2_VALID_TOLERANCE * current_valid_dz
    g = grid.assign(min_dz=grid[["fit_dz", "valid_dz"]].min(axis=1))
    g = g[g["valid_dz"] >= gate]
    for tier in (V2_NEUTRAL_PREFER, V2_NEUTRAL_MAX):
        passing = g[g["pct_neutral"] <= tier]
        if len(passing):
            best = passing["min_dz"].round(4).max()
            top = passing[passing["min_dz"].round(4) == best].sort_values(["t", "share"], ascending=False)
            return top.iloc[0], tier, passing
    return None, None, g.iloc[0:0]


def example_rows(labels, news, exclude_ids, n, seed):
    """n random items per class (fewer if a class is small), never items in exclude_ids (hand-label sheet)."""
    t = labels.merge(news[["news_id", "title", "abstract"]], on="news_id", validate="one_to_one")
    t = t[~t["news_id"].isin(set(exclude_ids))]
    parts = []
    for c in SIX_CLASSES:
        sub = t[t["label"] == c]
        parts.append(sub.sample(n=min(n, len(sub)), random_state=seed))
    cols = ["label", "news_id", "title", "abstract", "intensity", "top_label", "neutral_score"]
    return pd.concat(parts, ignore_index=True)[cols]


@dataclass
class PartContext:
    """Candidates of one analysis part mapped to item rows and impression (profile) rows."""
    groups: np.ndarray        # impression row per candidate
    group_users: np.ndarray   # user code per impression row
    labels: np.ndarray        # click label per candidate
    item_idx: np.ndarray      # item row per candidate
    histories: list           # item rows per impression row, oldest -> newest
    n_impressions: int
    imp_keys: pd.Index = None # impression key per impression row


def load_part_context(processed, part, item_index):
    """fit or valid only (dev is refused by the loader). History comes from the behaviors table, never from clicks."""
    imps = load_part_impressions(processed, part)
    groups, imp_keys = pd.factorize(imps["imp_key"].astype(str))
    users = pd.Series(pd.factorize(imps["user_id"].astype(str))[0]).groupby(groups).first().to_numpy()
    idx = item_index.get_indexer(imps["news_id"].astype(str))
    if (idx < 0).any():
        raise ValueError(f"{part} candidates without emotion vectors")
    beh = pd.read_parquet(processed / "behaviors.parquet", columns=["imp_key", "part", "history"],
                          filters=[("part", "==", part)]).set_index("imp_key").reindex(imp_keys)
    if beh["history"].isna().any():
        raise ValueError(f"{part} impressions without a behaviors row")
    return PartContext(groups, users, imps["label"].to_numpy(), idx,
                       history_index_lists(beh["history"].to_numpy(), item_index), len(imp_keys), imp_keys)
