import json
import re

import numpy as np
import pandas as pd
import pytest

from src.config import EMOTION_SPEC_DIR, EMOTION_SPEC_VERSION
from src.emotion_spec import (confirmation_actions, load_spec, replication_selection, spec_hash, spec_path,
                              validate_spec)

# Hash of the hashed content of every released spec version. Changing a released spec without a version bump
# breaks this test on purpose: write config/emotion_spec_v<N+1>.json instead and add its hash here.
KNOWN_SPEC_HASHES = {1: "b2a9ca38e355c6efa858a4f1ed0ab7bb4500bcc50d54bff23711ae8c897bf7f0"}


def test_released_specs_unchanged_without_version_bump():
    files = sorted(EMOTION_SPEC_DIR.glob("emotion_spec_v*.json"))
    assert files, "no emotion spec found"
    for f in files:
        version = int(re.search(r"_v(\d+)\.json$", f.name).group(1))
        spec = json.loads(f.read_text(encoding="utf-8"))
        assert spec["version"] == version, f"{f.name} declares version {spec['version']}"
        assert version in KNOWN_SPEC_HASHES, f"new spec {f.name}: add its hash to KNOWN_SPEC_HASHES"
        assert spec_hash(spec) == KNOWN_SPEC_HASHES[version], f"{f.name} changed without a version bump"
    assert spec_path().exists() and EMOTION_SPEC_VERSION == max(KNOWN_SPEC_HASHES)


def test_status_change_does_not_change_hash_but_content_does():
    spec = load_spec()
    h = spec_hash(spec)
    assert spec_hash({**spec, "status": "confirmed"}) == h
    assert spec_hash({**spec, "item_features_m5": spec["item_features_m5"][:-1]}) != h


def test_validate_spec_catches_unknown_and_inconsistent_entries():
    spec = load_spec()
    cols = set(spec["item_features_m5"])
    assert validate_spec(spec, cols) == []
    bad = {**spec, "item_features_m5": spec["item_features_m5"] + ["go_magic"],
           "congruence_features": ["cong_pct_cos_otheruser"], "fusion": {**spec["fusion"], "emotion_score": "x"}}
    problems = validate_spec(bad, cols)
    assert any("go_magic" in p for p in problems) and any("otheruser" in p for p in problems)
    assert any("emotion_score" in p for p in problems)


def summary(weighted=(0.3, 0.5), title=(0.3, 0.5), nrc=(0.1, 0.3)):
    rows = [("goemotions_weighted", *weighted), ("goemotions_title", *title), ("nrc_emolex_6class", *nrc),
            ("goemotions_combined", 0.3, 0.5), ("reference_all_neutral", 0.0, 0.0)]
    return pd.DataFrame(rows, columns=["source", "kappa_ci_low", "kappa_ci_high"])


def test_confirmation_rule_confirm_drop_title_and_hold():
    spec = load_spec()
    assert confirmation_actions(spec, summary()) == ("confirm", [])
    verdict, actions = confirmation_actions(spec, summary(nrc=(-0.05, 0.15)))
    assert verdict == "bump" and "nrc_trust" in actions[0]
    verdict, actions = confirmation_actions(spec, summary(weighted=(0.0, 0.18)))
    assert verdict == "bump" and any("cong_six_negkl_personal" in a for a in actions)
    assert any("title" in a for a in actions)                         # title CI (0.3) above weighted upper (0.18)
    verdict, actions = confirmation_actions(spec, summary(nrc=(-0.05, 0.15)), {"kappa_ci95": [0.1, 0.35]})
    assert verdict == "hold"


def test_replication_selection_rule():
    fit = pd.DataFrame({"dz": [0.2, 0.1, 0.1, 0.05], "user_dz": [0.2, 0.1, 0.1, 0.05],
                        "significant_both": [True, True, True, False]}, index=list("abcd"))
    valid = pd.DataFrame({"dz": [0.1, -0.05, 0.01, 0.1], "dz_ci_low": [0.05, -0.08, -0.01, 0.05],
                          "dz_ci_high": [0.15, -0.02, 0.03, 0.15], "user_dz": [0.1, -0.05, 0.0, 0.1],
                          "significant_both": [True, True, False, True]}, index=list("abcd"))
    t = replication_selection(fit, valid)
    assert t["keep"].tolist() == [True, False, False, False]
    assert t["reason"].tolist() == ["replicated on valid", "sign flips on valid", "valid 95% CI includes 0",
                                    "not significant at both levels on fit"]
