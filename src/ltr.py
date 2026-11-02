"""M4 and M5: LightGBM ranker features, the time-safe popularity feature, negative sampling and training (D63-D67)."""
import hashlib
import itertools
import json
import shutil
import time

import numpy as np
import pandas as pd

from .baselines import cosine_scores, fit_tfidf, item_ctr, load_part, popularity_scores, profile_weights, train_news_ids
from .click_signal import permute_within_groups
from .config import (LTR_CHECK_TOLERANCE, LTR_CONGRUENCE_MEASURES, LTR_DAY1_PRIOR, LTR_GRID, LTR_MAX_ROUNDS, LTR_MODEL_DIR, LTR_RUN_DIR, LTR_NEG_PER_CLICK, LTR_PATIENCE, LTR_STATS_PARTS, LTR_THREADS,
                     M0_NAME, M1_NAME, M2_NAME, M4_CATEGORICAL, M4_FEATURES, M4_NAME, M5_ALL_NAME, M5_NAME, M5_ROBUSTNESS_ITEM_FEATURES,
                     PROCESSED, RESULTS, TFIDF_MIN_DF, TFIDF_STOP_WORDS)
from .congruence import congruence_features, history_index_lists, user_profiles
from .emotion import GOEMOTIONS_LABELS, collapse
from .emotion_features import GO_EMOTIONS
from .emotion_spec import load_spec
from .experiments import load_frozen, load_rows
from .sbert import load_embeddings, profile_cosine


def spec_features(spec=None):
    """The spec's item and congruence features: (item list, congruence list)."""
    spec = spec or load_spec()
    return list(spec["item_features_m5"]), list(spec["congruence_features"])


def model_features(model, spec=None):
    items, cong = spec_features(spec)
    return {M4_NAME: list(M4_FEATURES), M5_NAME: [*M4_FEATURES, *items, *cong],
            M5_ALL_NAME: [*M4_FEATURES, *M5_ROBUSTNESS_ITEM_FEATURES, *cong]}[model]


def emotion_columns(model, spec=None):
    return [c for c in model_features(model, spec) if c not in M4_FEATURES]


def past_popularity(cands, stats, a, prior=LTR_DAY1_PRIOR):
    """The M0 smoothed CTR per row, from stats rows on days strictly before the row's day (D64). No earlier data -> the fixed prior.
    cands: news_id, day. stats: news_id, day, label."""
    days = cands["day"].to_numpy()
    out = np.empty(len(cands), dtype=np.float64)
    for d in np.unique(days):
        rows = days == d
        past = stats[stats["day"].to_numpy() < d]
        if past.empty:
            out[rows] = prior
        else:
            ctr, p = item_ctr(past)
            out[rows] = popularity_scores(cands[rows], ctr, p, a)
    return out


def user_category_affinity(groups, item_idx, histories, item_codes, n_codes):
    """Share of the last 50 known history items in the candidate's category; 0 without history."""
    item_codes = np.asarray(item_codes)
    profiles, _ = user_profiles(histories, np.eye(n_codes)[item_codes], np.zeros(n_codes), decay=1.0)
    return profiles[np.asarray(groups), item_codes[np.asarray(item_idx)]]


def impression_sizes(keys):
    """Candidates per impression, for every row. Always called on all candidates, before any sampling."""
    return pd.Series(np.asarray(keys)).groupby(np.asarray(keys)).transform("size").to_numpy()


def category_codes(values):
    """Stable integer codes: sorted over the full news table, so every part shares them."""
    return pd.Categorical(values, categories=sorted(pd.unique(values))).codes


def sample_negatives(keys, labels, per_click, seed):
    """Mask of training rows: all clicks plus up to per_click non-clicks per click within each impression, seeded. Row order is kept."""
    labels = np.asarray(labels)
    groups = pd.factorize(np.asarray(keys))[0]
    t = pd.DataFrame({"g": groups, "y": labels, "r": np.random.default_rng(seed).random(len(labels))})
    clicks = t["y"].groupby(t["g"]).transform("sum").to_numpy()
    neg = labels == 0
    rank = np.zeros(len(labels))
    rank[neg] = t.loc[neg, "r"].groupby(t.loc[neg, "g"]).rank(method="first").to_numpy()
    return ~neg | (rank <= per_click * clicks)


def group_sizes(keys):
    """Rows per impression in row order; refuses an impression split into two blocks, which would break lambdarank groups."""
    keys = np.asarray(keys)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    if len(set(keys[starts].tolist())) != len(starts):
        raise ValueError("impression rows are not consecutive")
    return np.diff(np.r_[starts, len(keys)])


def mean_impression_auc(groups, labels, scores):
    """Mean per-impression AUC, ties count one half; same value as the harness, fast enough to run every boosting round."""
    groups, labels = np.asarray(groups), np.asarray(labels, dtype=np.float64)
    ranks = pd.Series(np.asarray(scores, dtype=np.float64)).groupby(groups).rank(method="average").to_numpy()
    n = np.bincount(groups).astype(np.float64)
    pos = np.bincount(groups, weights=labels)
    rank_sum = np.bincount(groups, weights=ranks * labels)
    ok = (pos > 0) & (pos < n)
    return float(((rank_sum[ok] - pos[ok] * (pos[ok] + 1) / 2) / (pos[ok] * (n[ok] - pos[ok]))).mean())


def shuffle_within_impressions(frame, columns, seed):
    """A copy with the columns permuted within each impression, jointly; everything else untouched (D67d)."""
    groups = pd.factorize(frame["imp_key"].to_numpy())[0]
    out = frame.copy()
    for c in columns:
        out[c] = permute_within_groups(groups, frame[c].to_numpy(), seed)
    return out


def run_id(features, params, seed, tag, data_sha):
    text = json.dumps({"features": features, "params": params, "seed": seed, "tag": tag, "data": data_sha}, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def file_sha(*paths):
    h = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def grid_configs():
    keys = list(LTR_GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(LTR_GRID[k] for k in keys))]


def fit_or_load(train, valid, features, params, seed, tag, data_sha, resume, run_dir=LTR_RUN_DIR):
    """One run, always read back from its saved file, so a resumed run and a fresh run score the same booster. -> (booster, info)."""
    import lightgbm as lgb
    run_dir.mkdir(parents=True, exist_ok=True)
    rid = run_id(features, params, seed, tag, data_sha)
    path, meta = run_dir / f"{rid}.txt", run_dir / f"{rid}.json"
    if resume and path.exists() and meta.exists():
        return lgb.Booster(model_file=str(path)), json.loads(meta.read_text(encoding="utf-8"))
    start = time.time()
    booster, best, rows, auc = train_ranker(train, valid, features, params, seed)
    tmp = path.with_name(path.name + ".tmp")
    booster.save_model(str(tmp), num_iteration=best)
    tmp.replace(path)  # atomic: an interrupted run never leaves a half-written model
    info = {"run_id": rid, "best_iteration": best, "train_rows": rows, "valid_auc": auc, "seconds": time.time() - start, "path": str(path)}
    meta.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return lgb.Booster(model_file=str(path)), info


def model_path(model, model_dir=LTR_MODEL_DIR):
    return model_dir / f"{model}.txt"


def save_model_copy(info, model, model_dir=LTR_MODEL_DIR):
    """Keep the chosen booster under the model's name. -> its sha256, stored in the frozen config."""
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(info["path"], model_path(model, model_dir))
    return file_sha(model_path(model, model_dir))


def load_model(model, expected_sha, model_dir=LTR_MODEL_DIR):
    """The frozen booster; refuses a file that is not the one recorded when the settings were frozen."""
    import lightgbm as lgb
    path = model_path(model, model_dir)
    if file_sha(path) != expected_sha:
        raise SystemExit(f"{path.name} is not the frozen model")
    return lgb.Booster(model_file=str(path))


def check_saved_scores(frame, part, tolerance=LTR_CHECK_TOLERANCE, results=RESULTS):
    """D67b: pop_ctr, m1_score and m2_score must equal the saved M0, M1 and M2 scores of the part. -> largest gap per feature."""
    gaps = {}
    for model, column in ((M0_NAME, "pop_ctr"), (M1_NAME, "m1_score"), (M2_NAME, "m2_score")):
        saved = pd.read_parquet(results / "scores" / f"{model}__{part}.parquet")
        merged = frame[["imp_key", "news_id", column]].merge(saved, on=["imp_key", "news_id"], validate="one_to_one")
        if len(merged) != len(frame):
            raise SystemExit(f"{model}: saved scores do not cover the {part} rows")
        gaps[column] = float(np.abs(merged[column].to_numpy(np.float64) - merged["score"].to_numpy()).max())
    if max(gaps.values()) > tolerance:
        raise SystemExit(f"{part} features do not reproduce M0, M1 and M2 (D67b): {gaps}")
    return gaps


def train_ranker(train, valid, features, params, seed, max_rounds=LTR_MAX_ROUNDS, patience=LTR_PATIENCE):
    """One LGBMRanker: sampled train rows, early stopping on mean per-impression valid AUC. -> (booster, best iteration, rows, valid AUC)."""
    import lightgbm as lgb
    keep = sample_negatives(train["imp_key"].to_numpy(), train["label"].to_numpy(), LTR_NEG_PER_CLICK, seed)
    tr_keys = train["imp_key"].to_numpy()[keep]
    va_keys = valid["imp_key"].to_numpy()
    va_groups, va_labels = pd.factorize(va_keys)[0], valid["label"].to_numpy()

    def imp_auc(_y_true, y_pred):
        return "imp_auc", mean_impression_auc(va_groups, va_labels, y_pred), True

    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=max_rounds, random_state=seed, n_jobs=LTR_THREADS, deterministic=True,
                           force_row_wise=True, metric="None", verbose=-1, **params)
    model.fit(train.loc[keep, features], train["label"].to_numpy()[keep], group=group_sizes(tr_keys), eval_X=(valid[features],), eval_y=(va_labels,),
              eval_group=[group_sizes(va_keys)], eval_metric=imp_auc, categorical_feature=[c for c in features if c in M4_CATEGORICAL],
              callbacks=[lgb.early_stopping(patience, verbose=False)])
    return model.booster_, int(model.best_iteration_), int(keep.sum()), float(model.best_score_["valid_0"]["imp_auc"])


def build_features(part, processed=PROCESSED):
    """One row per candidate of a part, in harness order. Labels are kept only as the training target; position is never stored."""
    truth, groups, histories, _ = load_part(part, processed)
    beh = pd.read_parquet(processed / "behaviors.parquet", columns=["imp_key", "part", "time", "history_len"], filters=[("part", "==", part)])
    if set(beh["part"].astype(str)) != {part}:
        raise ValueError("behaviors of other parts loaded")
    beh = beh.set_index(beh["imp_key"].astype(str))
    news = pd.read_parquet(processed / "news.parquet", columns=["news_id", "category", "subcategory", "title", "text"])
    index = pd.Index(news["news_id"])
    item_idx = index.get_indexer(truth["news_id"])
    if (item_idx < 0).any():
        raise ValueError("candidates missing from news.parquet")
    hist_rows = history_index_lists(histories, index)
    keys = truth["imp_key"].to_numpy()
    day = pd.to_datetime(beh["time"].reindex(keys).to_numpy()).normalize()
    out = pd.DataFrame({"imp_key": keys, "news_id": truth["news_id"].to_numpy(), "label": truth["label"].to_numpy(dtype=np.int8), "day": day})

    stats = load_rows(processed, LTR_STATS_PARTS[part], ["news_id", "label", "time"])
    stats = pd.DataFrame({"news_id": stats["news_id"].astype(str).to_numpy(), "label": stats["label"].to_numpy(),
                          "day": pd.to_datetime(stats["time"]).dt.normalize().to_numpy()})
    out["pop_ctr"] = past_popularity(out, stats, load_frozen(M0_NAME)["config"]["a"]).astype(np.float32)
    del stats

    m1 = load_frozen(M1_NAME)["config"]
    is_train = news["news_id"].isin(train_news_ids()).to_numpy()
    texts = news[m1["text"]].fillna("")
    _, X = fit_tfidf(texts[is_train], texts, m1["sublinear_tf"], TFIDF_MIN_DF, TFIDF_STOP_WORDS)
    U = profile_weights(hist_rows, len(news), m1["aggregation"]) @ X
    out["m1_score"] = cosine_scores(U, X, groups, item_idx).astype(np.float32)
    del X, U

    m2 = load_frozen(M2_NAME)["config"]
    E = load_embeddings(m2["text"], news["news_id"], processed)
    out["m2_score"] = profile_cosine(profile_weights(hist_rows, len(news), m2["aggregation"]) @ E, E, groups, item_idx).astype(np.float32)
    del E

    cats = category_codes(news["category"].astype(str).to_numpy())
    subs = category_codes(news["subcategory"].astype(str).to_numpy())
    out["category"] = cats[item_idx].astype(np.int16)
    out["subcategory"] = subs[item_idx].astype(np.int16)
    out["user_cat_affinity"] = user_category_affinity(groups, item_idx, hist_rows, cats, int(cats.max()) + 1).astype(np.float32)
    out["history_len"] = beh["history_len"].reindex(keys).to_numpy(dtype=np.float32)
    out["title_len"] = news["title"].str.split().str.len().to_numpy(dtype=np.float32)[item_idx]
    out["imp_size"] = impression_sizes(keys).astype(np.float32)

    items, cong_names = spec_features()
    columns = list(dict.fromkeys([*items, *M5_ROBUSTNESS_ITEM_FEATURES]))
    pct = [f"go_pct_{e}" for e in GO_EMOTIONS]
    feats = pd.read_parquet(processed / "news_emotion_features_v2.parquet", columns=["news_id", *dict.fromkeys([*columns, *pct, "go_vad_valence"])])
    raw = pd.read_parquet(processed / "news_emotion_raw28_v2.parquet", columns=["news_id", *GOEMOTIONS_LABELS])
    if feats["news_id"].tolist() != news["news_id"].tolist() or raw["news_id"].tolist() != news["news_id"].tolist():
        raise ValueError("emotion tables are not aligned with news.parquet")
    vectors = {"pct": feats[pct].to_numpy(np.float64), "six": collapse(raw[GOEMOTIONS_LABELS].to_numpy()),
               "valence": feats["go_vad_valence"].to_numpy(np.float64)}
    cong, _ = congruence_features(groups, item_idx, hist_rows, vectors, LTR_CONGRUENCE_MEASURES)
    for c in columns:
        out[c] = feats[c].to_numpy(np.float32)[item_idx]
    for c in cong_names:
        out[c] = cong[c]
    return out
