import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.emotion import GOEMOTIONS_LABELS, SIX_CLASSES, build_emotion_table
from src.emotion_features import GO_EMOTIONS
from src.lexicon import SHARE_COLUMNS
from src.validation import (PCT_GROUP, intra_rater, per_class_table, threshold_corpus_shares, threshold_labels_pct,
                            threshold_labels_raw)

ROOT = Path(__file__).resolve().parents[1]
IDX = {l: i for i, l in enumerate(GOEMOTIONS_LABELS)}


def raw_rows(spec):
    """spec: list of {label: score}; other labels 0.01."""
    rows = np.full((len(spec), 28), 0.01)
    for r, d in enumerate(spec):
        for k, v in d.items():
            rows[r, IDX[k]] = v
    return rows


def test_threshold_labels_raw_uses_non_neutral_max_and_group():
    raw = pd.DataFrame(raw_rows([{"grief": 0.6, "neutral": 0.9}, {"curiosity": 0.15}, {"anger": 0.04}]),
                       columns=GOEMOTIONS_LABELS)
    assert threshold_labels_raw(raw, 0.1).tolist() == ["sadness", "surprise", "neutral"]
    assert threshold_labels_raw(raw, 0.5).tolist() == ["sadness", "neutral", "neutral"]   # neutral score ignored


def test_threshold_labels_pct_and_groups():
    assert PCT_GROUP["curiosity"] == "surprise" and PCT_GROUP["nervousness"] == "fear"
    assert "sadness" not in PCT_GROUP.values()                                             # documented limitation
    pct = pd.DataFrame([[0.5] * 8, [0.5] * 7 + [0.97]], columns=[f"go_pct_{e}" for e in GO_EMOTIONS])
    assert threshold_labels_pct(pct, 0.9).tolist() == ["neutral", PCT_GROUP[GO_EMOTIONS[-1]]]


def test_threshold_corpus_shares_rows_and_sums():
    raw = pd.DataFrame(raw_rows([{"joy": 0.3}, {"fear": 0.08}]), columns=GOEMOTIONS_LABELS)
    pct = pd.DataFrame([[0.99] + [0.1] * 7, [0.1] * 8], columns=[f"go_pct_{e}" for e in GO_EMOTIONS])
    t = threshold_corpus_shares(raw, pct)
    assert len(t) == 11 and set(t["score"]) == {"raw_max_sigmoid", "pct_max_percentile"}
    share_cols = [f"share_{c}" for c in SIX_CLASSES]
    assert np.allclose(t[share_cols].sum(axis=1), 100)
    assert t.loc[(t["score"] == "raw_max_sigmoid") & (t["threshold"] == 0.05), "share_neutral"].item() == 0


def test_per_class_table_flags_rare_classes():
    from src.annotation import agreement
    y = ["neutral"] * 6 + ["joy"] * 5 + ["fear"]
    res = {"s": agreement(y, ["neutral"] * 12, n_boot=20)}
    t = per_class_table(y, res).set_index("class")
    assert t.loc["neutral", "recall"] == 1.0 and t.loc["joy", "recall"] == 0.0
    assert t.loc["joy", "reliable"] and not t.loc["fear", "reliable"] and np.isnan(t.loc["anger", "recall"])


def test_intra_rater_matches_on_sample_id():
    main = pd.DataFrame({"sample_id": ["1", "2", "3"], "your_label": ["joy", "neutral", "fear"]})
    second = pd.DataFrame({"sample_id": ["3", "1"], "your_label": ["fear", "joy"]})
    assert intra_rater(main, second)["accuracy"] == 1.0
    assert intra_rater(main, second.assign(your_label=["", "joy"])) is None
    with pytest.raises(ValueError):
        intra_rater(main, pd.DataFrame({"sample_id": ["9"], "your_label": ["joy"]}))


@pytest.fixture
def fake_project(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    ids = [f"N{i}" for i in range(n)]
    proc = tmp_path / "processed"
    proc.mkdir()
    raw = pd.DataFrame(rng.uniform(0, 0.6, (n, 28)), columns=GOEMOTIONS_LABELS)
    raw.insert(0, "news_id", ids)
    raw.to_parquet(proc / "news_emotion_raw28_title.parquet", index=False)
    raw.assign(has_abstract=True).to_parquet(proc / "news_emotion_raw28_weighted.parquet", index=False)
    for f in ("news_emotion.parquet", "news_emotion_weighted.parquet"):
        build_emotion_table(raw).to_parquet(proc / f, index=False)
    nrc = pd.DataFrame(rng.dirichlet(np.ones(8), n), columns=SHARE_COLUMNS)
    nrc.insert(0, "news_id", ids)
    nrc["nrc_has_hits"] = 1
    nrc.to_parquet(proc / "news_emotion_nrc_features.parquet", index=False)
    pct = pd.DataFrame(rng.uniform(0, 1, (n, 8)), columns=[f"go_pct_{e}" for e in GO_EMOTIONS])
    pct.insert(0, "news_id", ids)
    pct.to_parquet(proc / "news_emotion_features.parquet", index=False)
    pd.DataFrame({"news_id": ids, "intensity": rng.uniform(0, 0.6, n), "group": rng.choice(SIX_CLASSES[:5], n),
                  "label": rng.choice(SIX_CLASSES, n)}).to_parquet(proc / "news_emotion_labels_v2.parquet", index=False)
    variants = {"news_id": ids}
    for v in ("v2b", "v2c"):
        variants.update({f"intensity_{v}": rng.uniform(0, 0.6, n), f"group_{v}": rng.choice(SIX_CLASSES[:5], n),
                         f"label_{v}": rng.choice(SIX_CLASSES, n)})
    pd.DataFrame(variants).to_parquet(proc / "news_emotion_labels_variants.parquet", index=False)
    labels = rng.choice(SIX_CLASSES, n)
    sheet = pd.DataFrame({"sample_id": [str(i + 1) for i in range(n)], "news_id": ids,
                          "title": [f"title {i}" for i in range(n)], "abstract": ["abs"] * n, "your_label": labels})
    sheet.to_csv(tmp_path / "sheet.csv", index=False, encoding="utf-8-sig")
    sheet.iloc[:10].to_csv(tmp_path / "relabel.csv", index=False, encoding="utf-8-sig")
    return tmp_path


def run_06(p, sheet="sheet.csv"):
    cmd = [sys.executable, str(ROOT / "scripts" / "38_analysis_label_validation.py"), "--sheet", str(p / sheet),
           "--relabel-sheet", str(p / "relabel.csv"), "--processed", str(p / "processed"),
           "--out-dir", str(p / "out"), "--fig-dir", str(p / "fig"), "--n-boot", "30"]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def test_script_06_end_to_end_with_fakes(fake_project):
    r = run_06(fake_project)
    assert r.returncode == 0, r.stderr
    out = fake_project / "out"
    src = pd.read_csv(out / "emotion_validation_sources.csv")
    assert src["source"].tolist() == ["goemotions_combined", "goemotions_weighted", "goemotions_title",
                                      "goemotions_v2", "goemotions_v2b", "goemotions_v2c", "nrc_emolex_6class",
                                      "reference_all_neutral"]
    sens = pd.read_csv(out / "emotion_threshold_sensitivity.csv")
    assert len(sens) == 32 and sens["kappa_ci_low"].le(sens["kappa_ci_high"]).all()
    assert set(sens["score"]) >= {"v2_intensity", "v2b_intensity", "v2c_intensity"}
    meta = json.loads((out / "emotion_validation_sources.json").read_text(encoding="utf-8"))
    assert meta["intra_rater"]["accuracy"] == 1.0 and meta["n_labelled"] == 40
    assert meta["emotion_spec_confirmation"]["verdict"] in {"confirm", "bump", "hold"}
    assert "Emotion spec v2" in r.stdout
    assert len(pd.read_csv(out / "emotion_validation_per_class.csv")) == 8 * 6
    assert (fake_project / "fig" / "emotion_threshold_kappa.png").stat().st_size > 0


def test_script_06_refuses_incomplete_sheet(fake_project):
    s = pd.read_csv(fake_project / "sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    s.loc[3, "your_label"] = ""
    s.to_csv(fake_project / "partial.csv", index=False, encoding="utf-8-sig")
    r = run_06(fake_project, "partial.csv")
    assert r.returncode == 1 and "Fix the sheet first" in r.stdout and not (fake_project / "out").exists()
