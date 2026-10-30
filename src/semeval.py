"""External validation on SemEval-2007 Task 14 Affective Text (Strapparava & Mihalcea, 2007): loading, gold classes, rules, verdict.
Headlines only: this validates the title part of the MIND scoring. Trial selects, test reports (D46-D48)."""
import hashlib
import html
import re
import subprocess
import tarfile

import numpy as np
import pandas as pd
from scipy import stats as st

from .agreement import compare_sources
from .config import (LABEL_VARIANTS, MIN_CLASS_COUNT, SEED, SEMEVAL_CUTOFF, SEMEVAL_MIN_MAIN_EMOTIONS, SEMEVAL_MIN_RHO,
                     SEMEVAL_NEUTRAL_LIMIT, SEMEVAL_REFERENCE_RULE, SEMEVAL_SENSITIVITY_CUTOFFS, SEMEVAL_SHA256, SEMEVAL_SPLITS,
                     SEMEVAL_URL, SIGNAL_ALPHA)
from .emotion import EMOTION_GROUPS, SIX_CLASSES, build_emotion_table
from .emotion_v2 import intensity_table, threshold_labels
from .lexicon import combine_lexicon_fields, field_lexicon_features, lexicon_counts, nrc_to_six

GOLD_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
GOLD_CLASSES = [c for c in SIX_CLASSES if c != "neutral"]
MAIN_EMOTIONS = ["anger", "fear", "joy", "sadness"]


def download_archive(dest):
    """Fetch the official archive with curl (the server's certificate chain fails in Python's TLS, curl verifies it through the
    OS store), check its hash and extract it with the safe tar filter."""
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / SEMEVAL_URL.rsplit("/", 1)[1]
    subprocess.run(["curl", "-sS", "-L", "--fail", "--max-time", "120", SEMEVAL_URL, "-o", str(path)], check=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SEMEVAL_SHA256:
        raise ValueError(f"unexpected archive hash {digest}")
    with tarfile.open(path) as t:
        t.extractall(dest, filter="data")
    return path


def split_files(root, split):
    folder = root / f"AffectiveText.{split}"
    stem = f"affectivetext_{split}"
    return folder / f"{stem}.xml", folder / f"{stem}.emotions.gold", folder / f"{stem}.valence.gold"


def parse_headlines(path):
    """id and text per <instance>. Parsed by pattern: the SGML file is not guaranteed to be valid XML."""
    pairs = re.findall(r'<instance id="(\d+)">(.*?)</instance>', path.read_text(encoding="latin-1"), re.S)
    return pd.DataFrame({"id": [int(i) for i, _ in pairs], "text": [html.unescape(t).strip() for _, t in pairs]})


def _gold(path, columns):
    return pd.read_csv(path, sep=r"\s+", header=None, names=["id", *columns])


def load_split(root, split):
    xml, emotions, valence = split_files(root, split)
    t = parse_headlines(xml).merge(_gold(emotions, GOLD_EMOTIONS), on="id", validate="one_to_one", how="outer", indicator="a")
    t = t.merge(_gold(valence, ["valence"]), on="id", validate="one_to_one", how="outer", indicator="b")
    if (t["a"] != "both").any() or (t["b"] != "both").any():
        raise ValueError(f"{split}: headline and gold ids differ")
    t = t.drop(columns=["a", "b"]).sort_values("id").reset_index(drop=True)
    if t["text"].str.strip().eq("").any():
        raise ValueError(f"{split}: empty headline")
    if not t[GOLD_EMOTIONS].stack().between(0, 100).all() or not t["valence"].between(-100, 100).all():
        raise ValueError(f"{split}: gold scores out of range")
    return t.assign(split=split, news_id=[f"{split}_{i}" for i in t["id"]])


def load_semeval(root, sizes=SEMEVAL_SPLITS):
    frames = []
    for split, n in sizes.items():
        t = load_split(root, split)
        if len(t) != n:
            raise ValueError(f"{split}: {len(t)} headlines, expected {n}")
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def gold_classes(gold, cutoff, seed=SEED):
    """D46: anger = max(anger, disgust); argmax over the 5 emotions, ties at random (seeded); neutral if the max < cutoff."""
    scores = gold[["joy", "sadness", "anger", "fear", "surprise"]].to_numpy(dtype=np.float64, copy=True)
    scores[:, 2] = np.maximum(scores[:, 2], gold["disgust"].to_numpy(dtype=np.float64))
    top = scores.max(axis=1)
    rng = np.random.default_rng(seed)
    labels = np.empty(len(scores), dtype=object)
    for i in range(len(scores)):
        hits = np.flatnonzero(scores[i] == top[i])
        labels[i] = "neutral" if top[i] < cutoff else GOLD_CLASSES[hits[0] if len(hits) == 1 else rng.choice(hits)]
    return labels


def group_scores(raw, exclude=LABEL_VARIANTS["v2b"]):
    """Sum of the raw GoEmotions scores per 6-class group (no neutral), without the excluded labels."""
    return pd.DataFrame({c: raw[[label for label in EMOTION_GROUPS[c] if label not in set(exclude)]].sum(axis=1).to_numpy()
                         for c in GOLD_CLASSES})


def rule_labels(raw, rule, t=None):
    """Labels of one rule: v1 = argmax of the 6-class collapse; v2, v2b, v2c = their threshold rule at t."""
    if rule == "v1":
        return build_emotion_table(raw)["dominant_emotion"].to_numpy()
    return threshold_labels(intensity_table(raw, LABEL_VARIANTS[rule]), t)


def correlations(x, y, n_boot=1000, seed=SEED):
    """Pearson and Spearman with percentile bootstrap 95% CIs (same resamples for both)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": len(x), "pearson": np.nan, "pearson_low": np.nan, "pearson_high": np.nan, "spearman": np.nan,
                "spearman_low": np.nan, "spearman_high": np.nan, "spearman_p": 1.0}
    rng = np.random.default_rng(seed)
    pearson, spearman = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            pearson.append(np.corrcoef(x[i], y[i])[0, 1])
            spearman.append(np.corrcoef(st.rankdata(x[i]), st.rankdata(y[i]))[0, 1])
    rho = st.spearmanr(x, y)
    return {"n": len(x), "pearson": float(np.corrcoef(x, y)[0, 1]), "pearson_low": float(np.percentile(pearson, 2.5)),
            "pearson_high": float(np.percentile(pearson, 97.5)), "spearman": float(rho.statistic),
            "spearman_low": float(np.percentile(spearman, 2.5)), "spearman_high": float(np.percentile(spearman, 97.5)),
            "spearman_p": float(rho.pvalue)}


def select_rule(table, reference):
    """D47. table: rule, t, kappa, kappa_ci_low, kappa_ci_high of eligible rules on trial. Highest kappa wins, unless its CI overlaps
    the reference rule's CI, then the reference stays. -> (rule, t, reason)."""
    best = table.loc[table["kappa"].idxmax()]
    ref = table[(table["rule"] == reference[0]) & (table["t"] == reference[1])].iloc[0]
    if best["rule"] == ref["rule"] and best["t"] == ref["t"]:
        return ref["rule"], float(ref["t"]), "the reference rule has the highest kappa"
    if best["kappa_ci_low"] <= ref["kappa_ci_high"] and ref["kappa_ci_low"] <= best["kappa_ci_high"]:
        return ref["rule"], float(ref["t"]), f"{best['rule']} at t {best['t']} has the highest kappa, but its CI overlaps the reference"
    return best["rule"], float(best["t"]), "highest kappa, and its CI does not overlap the reference"


def spec_check(valence, emotions):
    """D48. valence: {spearman, spearman_p}; emotions: {main emotion: {spearman, spearman_p}}. -> CONFIRM or HOLD with reasons."""
    passes = lambda r: r["spearman"] >= SEMEVAL_MIN_RHO and r["spearman_p"] < SIGNAL_ALPHA
    failed = [e for e in MAIN_EMOTIONS if not passes(emotions[e])]
    valence_ok = passes(valence)
    n_ok = len(MAIN_EMOTIONS) - len(failed)
    ok = valence_ok and n_ok >= SEMEVAL_MIN_MAIN_EMOTIONS
    return {"verdict": "CONFIRM" if ok else "HOLD", "valence_passes": bool(valence_ok), "emotions_passing": n_ok,
            "emotions_failing": failed}


def per_class_table(y_true, results, min_count):
    """Per source and class: gold count, recall, predictions, and a flag for classes with few gold items."""
    counts = pd.Series(y_true).value_counts().reindex(SIX_CLASSES, fill_value=0)
    rows = []
    for source, res in results.items():
        conf = res["confusion"].to_numpy()
        for i, c in enumerate(SIX_CLASSES):
            n = int(conf[i].sum())
            rows.append({"source": source, "class": c, "n_gold": int(counts[c]), "recall": float(conf[i, i] / n) if n else np.nan,
                         "n_predicted": int(conf[:, i].sum()), "reliable": bool(counts[c] >= min_count)})
    return pd.DataFrame(rows)


def headline_lexicon(texts, emolex, vad):
    """NRC + VAD on headlines: the MIND title-only rule, so no hits -> 0 and has_hits = 0."""
    t = field_lexicon_features(lexicon_counts(list(texts), emolex, vad))
    return combine_lexicon_fields(t, t, np.zeros(len(t), dtype=bool), 1.0)


def _name(rule, t):
    return rule if t is None else f"{rule}_t{t}"


def select_on_trial(trial, raw, grid, n_boot=1000):
    """Eligible rules (MIND neutral <= limit) against trial gold, then D47. -> (table, rule, t, reason)."""
    eligible = grid[grid["pct_neutral"] <= SEMEVAL_NEUTRAL_LIMIT].reset_index(drop=True)
    gold = gold_classes(trial, SEMEVAL_CUTOFF)
    preds = {_name(r["variant"], r["t"]): rule_labels(raw, r["variant"], r["t"]) for _, r in eligible.iterrows()}
    summary, _ = compare_sources(gold, preds, n_boot=n_boot)
    table = summary.assign(rule=eligible["variant"], t=eligible["t"], mind_pct_neutral=eligible["pct_neutral"])
    rule, t, reason = select_rule(table, SEMEVAL_REFERENCE_RULE)
    return table, rule, t, reason


def correlation_table(test, raw, lex, gv, n_boot=1000):
    """Threshold-free, on test: GoEmotions single label and group sum, NRC share, each against the gold score; valence."""
    group = group_scores(raw)
    rows = []
    for e in GOLD_EMOTIONS:
        scores = {"goemotions_label": raw[e], "nrc_emolex": lex[f"nrc_{e}"]}
        if e in group:
            scores["goemotions_group"] = group[e]
        rows += [{"gold": e, "source": s, **correlations(x, test[e], n_boot)} for s, x in scores.items()]
    for s, x in (("go_vad_valence", gv["go_vad_valence"]), ("vad_valence", lex["vad_valence"])):
        rows.append({"gold": "valence", "source": s, **correlations(x, test["valence"], n_boot)})
    return pd.DataFrame(rows)


def _verdict(corr, valence_source, emotion_source):
    get = lambda gold, source: corr[(corr["gold"] == gold) & (corr["source"] == source)].iloc[0]
    return spec_check(get("valence", valence_source), {e: get(e, emotion_source) for e in MAIN_EMOTIONS})


def agreement_tables(test, raw, nrc_labels, chosen, n_boot=1000):
    """Each rule against test gold at the official cutoff and the sensitivity cutoffs. chosen = (rule, t).
    -> (long table, per-class table, {source: confusion matrix} at the official cutoff)."""
    rules = [("v1", None), ("v2", 0.05), ("v2b", 0.05), ("v2c", 0.05)]
    rules += [chosen] if chosen not in rules else []
    preds = {_name(*r): rule_labels(raw, *r) for r in rules}
    preds |= {"nrc_emolex": nrc_labels, "all_neutral": np.full(len(test), "neutral", dtype=object)}
    frames = []
    for cutoff in [SEMEVAL_CUTOFF, *SEMEVAL_SENSITIVITY_CUTOFFS]:
        gold = gold_classes(test, cutoff)
        summary, results = compare_sources(gold, preds, n_boot=n_boot)
        frames.append(summary.assign(cutoff=cutoff, pct_gold_neutral=100 * float((gold == "neutral").mean())))
        if cutoff == SEMEVAL_CUTOFF:
            per_class = per_class_table(gold, results, MIN_CLASS_COUNT)
            confusion = {s: r["confusion"].to_dict() for s, r in results.items()}
    table = pd.concat(frames, ignore_index=True)
    return table.assign(chosen=table["source"] == _name(*chosen)), per_class, confusion


def run_validation(data, raw, lex, gv, grid, n_boot=1000):
    """D46-D48. All inputs row-aligned. Trial only selects the rule; everything reported comes from test."""
    is_trial = (data["split"] == "trial").to_numpy()
    pick = lambda t, mask: t[mask].reset_index(drop=True)
    selection, rule, t, reason = select_on_trial(pick(data, is_trial), pick(raw, is_trial), grid, n_boot)
    test, raw_test, lex_test, gv_test = (pick(x, ~is_trial) for x in (data, raw, lex, gv))
    corr = correlation_table(test, raw_test, lex_test, gv_test, n_boot)
    nrc_labels, tied = nrc_to_six(lex_test, SEED)
    table, per_class, confusion = agreement_tables(test, raw_test, nrc_labels.to_numpy(), (rule, t), n_boot)
    chosen_row = table[table["chosen"] & (table["cutoff"] == SEMEVAL_CUTOFF)].iloc[0]
    mind = grid[(grid["variant"] == rule) & (grid["t"] == t)].iloc[0]
    return {"selection": selection, "correlations": corr, "agreement": table, "per_class": per_class, "confusion": confusion,
            "chosen": {"rule": rule, "t": t, "reason": reason},
            "verdict_goemotions": _verdict(corr, "go_vad_valence", "goemotions_group"),
            "verdict_nrc": _verdict(corr, "vad_valence", "nrc_emolex"),
            "neutral": {"gold_test_pct": float(chosen_row["pct_gold_neutral"]), "chosen_test_pct": float(chosen_row["pct_pred_neutral"]),
                        "chosen_mind_pct": float(mind["pct_neutral"])},
            "nrc_tie_rate": float(tied.mean()), "nrc_hit_rate": float(lex_test["nrc_has_hits"].mean()),
            "vad_hit_rate": float(lex_test["vad_has_hits"].mean())}
