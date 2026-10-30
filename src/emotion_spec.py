"""Versioned emotion specification for M3/M5/M7 (config/emotion_spec_v<N>.json) and its post-label confirmation rule."""
import hashlib
import json

import numpy as np
import pandas as pd

from .config import CONGRUENCE_MEASURES, EMOTION_SPEC_DIR, EMOTION_SPEC_VERSION

UNHASHED = ("status", "confirmation_log")  # may change without a version bump; everything else may not
KAPPA_MIN_UPPER = 0.2    # a source whose kappa 95% CI upper bound is below this is not trusted
INTRA_MIN_UPPER = 0.4    # labels whose intra-rater kappa CI upper bound is below this are too unreliable to act on


def spec_path(version=EMOTION_SPEC_VERSION, spec_dir=EMOTION_SPEC_DIR):
    return spec_dir / f"emotion_spec_v{version}.json"


def spec_hash(spec):
    core = {k: v for k, v in spec.items() if k not in UNHASHED}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode("utf-8")).hexdigest()


def load_spec(version=EMOTION_SPEC_VERSION, spec_dir=EMOTION_SPEC_DIR):
    spec = json.loads(spec_path(version, spec_dir).read_text(encoding="utf-8"))
    if spec.get("version") != version:
        raise ValueError(f"spec file v{version} declares version {spec.get('version')}")
    return spec


def validate_spec(spec, feature_columns):
    """Every listed item feature must exist in the feature table; congruence features must come from declared
    measures (raw or _personal). Returns a list of problems (empty = valid)."""
    problems = [f"unknown item feature {f}" for f in spec["item_features_m5"] if f not in feature_columns]
    measures = {f"cong_{v}_{m}" for v, m, _ in CONGRUENCE_MEASURES}
    for f in spec["congruence_features"]:
        base = f[: -len("_personal")] if f.endswith("_personal") else f
        if base not in measures:
            problems.append(f"unknown congruence feature {f}")
    if spec["fusion"]["emotion_score"] not in spec["congruence_features"]:
        problems.append("fusion emotion_score must be one of the congruence features")
    overlap = set(spec["item_features_m5"]) & set(spec.get("excluded", {}))
    problems += [f"{f} is both included and excluded" for f in sorted(overlap)]
    return problems


def confirmation_actions(spec, summary, intra=None):
    """Apply the pre-declared post-label rule. summary: per-source table from scripts/06 (source, kappa_ci_low,
    kappa_ci_high). -> (verdict 'confirm' | 'bump' | 'hold', list of actions)."""
    ci = summary.set_index("source")[["kappa_ci_low", "kappa_ci_high"]]
    if intra is not None and intra["kappa_ci95"][1] < INTRA_MIN_UPPER:
        return "hold", [f"intra-rater kappa CI upper {intra['kappa_ci95'][1]:.3f} < {INTRA_MIN_UPPER}: labels too "
                        "unreliable to change the spec; keep v1 provisional and record this"]
    actions = []
    for source, features in spec["confirmation_rule"]["drop_if_source_untrusted"].items():
        upper = float(ci.loc[source, "kappa_ci_high"])
        if upper < KAPPA_MIN_UPPER:
            kept = [f for f in features if f in spec["item_features_m5"] + spec["congruence_features"]]
            if kept:
                actions.append(f"drop {kept}: {source} kappa CI upper {upper:.3f} < {KAPPA_MIN_UPPER}")
    rule = spec["confirmation_rule"]
    primary = rule.get("primary_source", "goemotions_weighted")
    w, t = ci.loc[primary], ci.loc["goemotions_title"]
    if float(t["kappa_ci_low"]) > float(w["kappa_ci_high"]):
        actions.append(f"make title the primary text variant: title kappa CI lies entirely above {primary}")
    if "revert_labels_to" in rule and float(w["kappa_ci_high"]) < float(ci.loc[rule["revert_labels_to"], "kappa_ci_low"]):
        actions.append(f"revert the labelling rule: {primary} kappa CI lies entirely below {rule['revert_labels_to']}")
    return ("bump" if actions else "confirm"), actions


def replication_selection(fit, valid):
    """D27: keep if significant at both levels on fit and, on valid, same sign with a 95% CI excluding 0.
    fit/valid: signal tables indexed by feature. -> selection table with keep and reason."""
    t = pd.DataFrame({"fit_dz": fit["dz"], "fit_user_dz": fit["user_dz"], "fit_significant_both": fit["significant_both"],
                      "valid_dz": valid["dz"], "valid_ci_low": valid["dz_ci_low"], "valid_ci_high": valid["dz_ci_high"],
                      "valid_user_dz": valid["user_dz"], "valid_significant_both": valid["significant_both"]})
    same_sign = np.sign(t["fit_dz"]) == np.sign(t["valid_dz"])
    ci_ok = (t["valid_ci_low"] > 0) | (t["valid_ci_high"] < 0)
    t["keep"] = t["fit_significant_both"].astype(bool) & same_sign & ci_ok
    t["reason"] = np.select([~t["fit_significant_both"].astype(bool), ~same_sign, ~ci_ok],
                            ["not significant at both levels on fit", "sign flips on valid",
                             "valid 95% CI includes 0"], "replicated on valid")
    return t
