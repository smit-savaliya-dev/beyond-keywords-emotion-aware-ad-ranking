"""Hand-label validation: per-source agreement, threshold sensitivity and intra-rater reliability."""
import numpy as np
import pandas as pd

from .annotation import agreement, compare_sources, read_sheet, sheet_problems
from .config import GO_FEATURES, MIN_CLASS_COUNT, SEED, THRESHOLDS_PCT, THRESHOLDS_RAW
from .emotion import GOEMOTIONS_LABELS, LABEL_TO_GROUP, SIX_CLASSES, build_emotion_table
from .emotion_features import GO_EMOTIONS
from .lexicon import nrc_to_six

NON_NEUTRAL = [label for label in GOEMOTIONS_LABELS if label != "neutral"]
PCT_GROUP = {e: LABEL_TO_GROUP[labels[0]] for e, labels in GO_FEATURES.items()}  # e.g. curiosity -> surprise


def threshold_labels_raw(raw28, t):
    """Class of the strongest non-neutral GoEmotions label if its sigmoid > t, else neutral."""
    scores = raw28[NON_NEUTRAL].to_numpy(dtype=np.float64)
    best = scores.argmax(axis=1)
    groups = np.array([LABEL_TO_GROUP[NON_NEUTRAL[j]] for j in best], dtype=object)
    return np.where(scores.max(axis=1) > t, groups, "neutral")


def threshold_labels_pct(go_pct, t):
    """Class of the highest go_pct_* emotion if its percentile > t, else neutral."""
    cols = [f"go_pct_{e}" for e in GO_EMOTIONS]
    scores = go_pct[cols].to_numpy(dtype=np.float64)
    best = scores.argmax(axis=1)
    groups = np.array([PCT_GROUP[GO_EMOTIONS[j]] for j in best], dtype=object)
    return np.where(scores.max(axis=1) > t, groups, "neutral")


def threshold_corpus_shares(raw28, go_pct):
    """Item-level (no labels): class shares on the full corpus for every pre-declared threshold."""
    rows = []
    for kind, ts, fn, data in (("raw_max_sigmoid", THRESHOLDS_RAW, threshold_labels_raw, raw28),
                               ("pct_max_percentile", THRESHOLDS_PCT, threshold_labels_pct, go_pct)):
        for t in ts:
            lab = pd.Series(fn(data, t))
            shares = lab.value_counts(normalize=True).reindex(SIX_CLASSES, fill_value=0)
            rows.append({"score": kind, "threshold": t, **{f"share_{c}": round(100 * float(v), 2)
                                                           for c, v in shares.items()}})
    return pd.DataFrame(rows)


def per_class_table(y_true, results):
    """Per source and class: hand-label count, recall, flag when the count is below MIN_CLASS_COUNT."""
    counts = pd.Series(y_true).value_counts().reindex(SIX_CLASSES, fill_value=0)
    rows = []
    for source, res in results.items():
        conf = res["confusion"].to_numpy()
        for i, c in enumerate(SIX_CLASSES):
            n = int(conf[i].sum())
            rows.append({"source": source, "class": c, "n_hand_labels": int(counts[c]),
                         "recall": float(conf[i, i] / n) if n else np.nan,
                         "n_predicted": int(conf[:, i].sum()),
                         "reliable": bool(counts[c] >= MIN_CLASS_COUNT)})
    return pd.DataFrame(rows)


def intra_rater(main_sheet, relabel_sheet):
    """Agreement between the first and second labelling of the same items. None until the re-label sheet is done."""
    if relabel_sheet is None or sheet_problems(relabel_sheet):
        return None
    m = relabel_sheet[["sample_id", "your_label"]].merge(main_sheet[["sample_id", "your_label"]], on="sample_id",
                                                        suffixes=("_second", "_first"), validate="one_to_one")
    if len(m) != len(relabel_sheet):
        raise ValueError("re-label sheet contains sample_ids that are not in the main sheet")
    return agreement(m["your_label_first"], m["your_label_second"])


def load_sources(processed, news_ids):
    """6-class predictions per source for the sampled items, plus score tables for the threshold views."""
    ids = pd.Index(news_ids)

    def table(name):
        t = pd.read_parquet(processed / name).set_index("news_id").reindex(ids)
        if t.isna().all(axis=1).any():
            raise ValueError(f"some sampled news ids are missing from {name}")
        return t

    combined, weighted = table("news_emotion.parquet"), table("news_emotion_weighted.parquet")
    title = build_emotion_table(table("news_emotion_raw28_title.parquet").reset_index()).set_index("news_id")
    nrc = table("news_emotion_nrc_features.parquet")
    nrc_labels, nrc_tied = nrc_to_six(nrc, SEED)
    preds = {"goemotions_combined": combined["dominant_emotion"].to_numpy(),
             "goemotions_weighted": weighted["dominant_emotion"].to_numpy(),
             "goemotions_title": title["dominant_emotion"].to_numpy(),
             "nrc_emolex_6class": nrc_labels.to_numpy(),
             "reference_all_neutral": np.array(["neutral"] * len(ids), dtype=object)}
    extra = {"raw28_weighted": table("news_emotion_raw28_weighted.parquet"),
             "go_pct": table("news_emotion_features.parquet"), "nrc": nrc, "nrc_tied": nrc_tied,
             "combined": combined, "weighted": weighted}
    return preds, extra


def run_validation(sheet_path, processed, relabel_path=None, n_boot=1000):
    """Everything script 06 reports. Raises if the main sheet is not completely labelled."""
    sheet = read_sheet(sheet_path)
    problems = sheet_problems(sheet)
    if problems:
        raise ValueError("; ".join(problems))
    y = sheet["your_label"].to_numpy()
    preds, extra = load_sources(processed, sheet["news_id"])
    summary, results = compare_sources(y, preds, n_boot=n_boot)
    per_class = per_class_table(y, results)

    rows = []
    for kind, ts, fn, data in (("raw_max_sigmoid", THRESHOLDS_RAW, threshold_labels_raw, extra["raw28_weighted"]),
                               ("pct_max_percentile", THRESHOLDS_PCT, threshold_labels_pct, extra["go_pct"])):
        for t in ts:
            res = agreement(y, fn(data, t), n_boot=n_boot)
            lab = pd.Series(fn(data, t))
            rows.append({"score": kind, "threshold": t, "kappa": res["kappa"], "kappa_ci_low": res["kappa_ci95"][0],
                         "kappa_ci_high": res["kappa_ci95"][1], "accuracy": res["accuracy"],
                         "macro_f1": res["macro_f1"], "sample_pct_neutral": 100 * float((lab == "neutral").mean())})
    sensitivity = pd.DataFrame(rows)

    relabel = read_sheet(relabel_path) if relabel_path is not None and relabel_path.exists() else None
    intra = intra_rater(sheet, relabel)
    counts = pd.Series(y).value_counts().reindex(SIX_CLASSES, fill_value=0)
    nrc = extra["nrc"]
    nrc_info = {"tie_rate_pct": round(100 * float(extra["nrc_tied"].mean()), 2),
                "pct_no_hits_to_neutral": round(100 * float((nrc["nrc_has_hits"] == 0).mean()), 2)}
    return {"sheet": sheet, "summary": summary, "results": results, "per_class": per_class,
            "sensitivity": sensitivity, "intra": intra, "label_counts": counts, "nrc_info": nrc_info,
            "extra": extra}


def plot_threshold_panels(table, value, ylabel, path, title, subtitle, ci=None):
    """One panel per score type: `value` against the threshold (optionally with CI columns)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surface, color = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
    kinds = list(dict.fromkeys(table["score"]))
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.2 * len(kinds), 3.8), squeeze=False)
    fig.patch.set_facecolor(surface)
    for ax, kind in zip(axes[0], kinds):
        t = table[table["score"] == kind]
        if ci:
            ax.fill_between(t["threshold"], t[ci[0]], t[ci[1]], color=color, alpha=0.15, linewidth=0)
        ax.plot(t["threshold"], t[value], color=color, linewidth=2, marker="o", markersize=6)
        for x, v in zip(t["threshold"], t[value]):
            ax.annotate(f"{v:.2f}" if abs(v) < 1.5 else f"{v:.0f}", (x, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=ink2)
        ax.set_title(kind.replace("_", " "), loc="left", fontsize=10, color=ink)
        ax.set_xlabel("threshold t (non-neutral if score > t)", fontsize=8.5, color=ink2)
        ax.set_ylabel(ylabel, fontsize=8.5, color=ink2)
        ax.set_xticks(t["threshold"])
        ax.grid(axis="y", color=grid, linewidth=0.6)
        ax.tick_params(length=0, colors=ink2, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, color=ink)
    fig.text(0.01, 0.885, subtitle, fontsize=8.5, color=ink2, ha="left")
    fig.subplots_adjust(top=0.74, wspace=0.28)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=surface)
    plt.close(fig)
