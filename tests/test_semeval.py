import io
import tarfile

import numpy as np
import pandas as pd
import pytest

import src.semeval as sm
from src.emotion import GOEMOTIONS_LABELS


def write_split(root, split, rows):
    """rows: (id, text, anger, disgust, fear, joy, sadness, surprise, valence)"""
    folder = root / f"AffectiveText.{split}"
    folder.mkdir(parents=True)
    body = "\n".join(f'<instance id="{r[0]}">{r[1]}</instance>' for r in rows)
    (folder / f"affectivetext_{split}.xml").write_text(f'<corpus task="affective text">\n{body}\n</corpus>\n', encoding="latin-1")
    (folder / f"affectivetext_{split}.emotions.gold").write_text("".join(" ".join(map(str, (r[0], *r[2:8]))) + "\n" for r in rows))
    (folder / f"affectivetext_{split}.valence.gold").write_text("".join(f"{r[0]} {r[8]}\n" for r in rows))


ROWS = [(1, "Bomb &amp; fear", 10, 0, 70, 0, 20, 5, -60), (2, "A win", 0, 0, 0, 80, 0, 10, 70)]


def gold_frame(rows):
    return pd.DataFrame(rows, columns=sm.GOLD_EMOTIONS)


def raw_frame(n, values):
    """values: {label: array}; other labels 0.01."""
    t = pd.DataFrame({label: np.full(n, 0.01) for label in GOEMOTIONS_LABELS})
    for label, v in values.items():
        t[label] = v
    t.insert(0, "news_id", [str(i) for i in range(n)])
    return t


def test_load_split_parses_and_unescapes(tmp_path):
    write_split(tmp_path, "trial", ROWS)
    t = sm.load_split(tmp_path, "trial")
    assert t["text"].tolist() == ["Bomb & fear", "A win"]
    assert t["news_id"].tolist() == ["trial_1", "trial_2"] and t.loc[1, "valence"] == 70


def test_load_split_rejects_id_mismatch_and_bad_range(tmp_path):
    write_split(tmp_path, "trial", ROWS)
    gold = tmp_path / "AffectiveText.trial" / "affectivetext_trial.valence.gold"
    gold.write_text("1 -60\n3 70\n")
    with pytest.raises(ValueError, match="ids differ"):
        sm.load_split(tmp_path, "trial")
    gold.write_text("1 -60\n2 170\n")
    with pytest.raises(ValueError, match="out of range"):
        sm.load_split(tmp_path, "trial")


def test_load_semeval_checks_counts(tmp_path):
    write_split(tmp_path, "trial", ROWS)
    assert len(sm.load_semeval(tmp_path, {"trial": 2})) == 2
    with pytest.raises(ValueError, match="expected 3"):
        sm.load_semeval(tmp_path, {"trial": 3})


def test_gold_classes_cutoff_disgust_and_ties():
    g = gold_frame([[10, 60, 0, 0, 0, 0], [0, 0, 49, 0, 0, 0], [0, 0, 50, 0, 0, 0], [0, 0, 0, 70, 70, 0]])
    labels = sm.gold_classes(g, 50)
    assert labels[:3].tolist() == ["anger", "neutral", "fear"]
    assert labels[3] in ("joy", "sadness")
    assert sm.gold_classes(g, 50).tolist() == labels.tolist()  # seeded
    assert sm.gold_classes(g, 25)[1] == "fear"


def test_group_scores_excludes_stance_labels():
    raw = raw_frame(1, {"anger": 0.5, "disapproval": 0.9, "joy": 0.4, "approval": 0.9})
    g = sm.group_scores(raw)
    assert g.loc[0, "anger"] == pytest.approx(0.5 + 0.01 + 0.01)  # anger, annoyance, disgust; no disapproval
    assert "neutral" not in g.columns
    assert g.loc[0, "joy"] == pytest.approx(0.4 + 10 * 0.01)      # 11 labels without approval


def test_rule_labels():
    raw = raw_frame(2, {"neutral": [0.9, 0.9], "approval": [0.2, 0.02], "fear": [0.04, 0.06]})
    assert sm.rule_labels(raw, "v1").tolist() == ["neutral", "neutral"]
    assert sm.rule_labels(raw, "v2", 0.05).tolist() == ["joy", "fear"]
    assert sm.rule_labels(raw, "v2b", 0.05).tolist() == ["neutral", "fear"]


def test_correlations_values_and_flat_input():
    x = np.arange(50.0)
    r = sm.correlations(x, 2 * x, n_boot=20)
    assert r["pearson"] == pytest.approx(1) and r["spearman"] == pytest.approx(1) and r["spearman_p"] < 1e-6
    flat = sm.correlations(np.ones(5), x[:5], n_boot=20)
    assert np.isnan(flat["spearman"]) and flat["spearman_p"] == 1.0


def rules(rows):
    return pd.DataFrame(rows, columns=["rule", "t", "kappa", "kappa_ci_low", "kappa_ci_high"])


def test_select_rule_all_three_branches():
    ref = ("v2b", 0.05)
    assert sm.select_rule(rules([("v2", 0.05, 0.2, 0.1, 0.3), ("v2b", 0.05, 0.3, 0.2, 0.4)]), ref)[:2] == ("v2b", 0.05)
    assert sm.select_rule(rules([("v2", 0.05, 0.35, 0.3, 0.45), ("v2b", 0.05, 0.3, 0.2, 0.4)]), ref)[:2] == ("v2b", 0.05)
    assert sm.select_rule(rules([("v2", 0.03, 0.6, 0.5, 0.7), ("v2b", 0.05, 0.3, 0.2, 0.4)]), ref)[:2] == ("v2", 0.03)


def good(rho=0.3, p=0.001):
    return {"spearman": rho, "spearman_p": p}


def test_spec_check_rule():
    ok = {e: good() for e in sm.MAIN_EMOTIONS}
    assert sm.spec_check(good(), ok)["verdict"] == "CONFIRM"
    three = {**ok, "fear": good(0.19)}
    r = sm.spec_check(good(), three)
    assert r["verdict"] == "CONFIRM" and r["emotions_failing"] == ["fear"]
    two = {**three, "joy": good(0.5, 0.2)}
    assert sm.spec_check(good(), two)["verdict"] == "HOLD"
    assert sm.spec_check(good(0.1), ok) == {"verdict": "HOLD", "valence_passes": False, "emotions_passing": 4, "emotions_failing": []}


def test_download_checks_hash_and_extracts(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        info = tarfile.TarInfo("AffectiveText.trial/README")
        info.size = 2
        t.addfile(info, io.BytesIO(b"hi"))
    monkeypatch.setattr(sm.subprocess, "run", lambda cmd, check: open(cmd[cmd.index("-o") + 1], "wb").write(buf.getvalue()))
    with pytest.raises(ValueError, match="unexpected archive hash"):
        sm.download_archive(tmp_path)
    import hashlib
    monkeypatch.setattr(sm, "SEMEVAL_SHA256", hashlib.sha256(buf.getvalue()).hexdigest())
    sm.download_archive(tmp_path)
    assert (tmp_path / "AffectiveText.trial" / "README").read_text() == "hi"


def test_headline_lexicon_no_hits_is_zero_not_nan():
    emolex = pd.DataFrame({c: [0] for c in sm.field_lexicon_features.__globals__["EMOLEX_COLUMNS"]}, index=["fear"])
    emolex["fear"] = 1
    vad = pd.DataFrame({"valence": [-0.5], "arousal": [0.5], "dominance": [0.0]}, index=["fear"])
    t = sm.headline_lexicon(["fear grows", "nothing here"], emolex, vad)
    assert t.loc[0, "nrc_fear"] == 1 and t.loc[0, "vad_valence"] == -0.5
    assert t.loc[1, "nrc_fear"] == 0 and t.loc[1, "nrc_has_hits"] == 0 and not t.isna().any().any()


def fake_study(n=120):
    rng = np.random.default_rng(1)
    joy, fear = rng.uniform(0, 100, n), rng.uniform(0, 100, n)
    data = pd.DataFrame({"anger": joy * 0, "disgust": joy * 0, "fear": fear, "joy": joy, "sadness": fear / 2, "surprise": joy / 2,
                         "valence": joy - fear, "split": ["trial"] * 40 + ["test"] * (n - 40), "news_id": [str(i) for i in range(n)]})
    data["anger"] = 100 - joy
    raw = raw_frame(n, {"joy": joy / 100, "fear": fear / 100, "anger": (100 - joy) / 100, "sadness": fear / 200})
    lex = pd.DataFrame({f"nrc_{e}": rng.uniform(0, 1, n) for e in sm.GOLD_EMOTIONS + ["trust", "anticipation"]})
    lex = lex.assign(vad_valence=rng.uniform(-1, 1, n), nrc_has_hits=1, vad_has_hits=1)
    gv = pd.DataFrame({"go_vad_valence": (joy - fear) / 100})
    grid = pd.DataFrame({"variant": ["v2", "v2b", "v2b", "v2c"], "t": [0.05, 0.05, 0.075, 0.05], "pct_neutral": [25.6, 39.0, 48.3, 42.4]})
    return data, raw, lex, gv, grid


def test_run_validation_keeps_trial_out_of_the_report():
    data, raw, lex, gv, grid = fake_study()
    v = sm.run_validation(data, raw, lex, gv, grid, n_boot=20)
    assert set(v["selection"]["source"]) == {"v2_t0.05", "v2b_t0.05"}           # neutral limit applied
    assert v["correlations"]["n"].eq(80).all() and v["agreement"]["n"].eq(80).all()
    assert v["verdict_goemotions"]["verdict"] == "CONFIRM" and v["verdict_nrc"]["verdict"] == "HOLD"
    assert sorted(v["agreement"]["cutoff"].unique()) == [25, 30, 50]
    a = v["agreement"].set_index(["source", "cutoff"])
    assert a.loc[("all_neutral", 50), "kappa"] == 0 and a["chosen"].sum() == 3
    assert not v["per_class"]["reliable"].all() or v["per_class"]["n_gold"].min() >= 20
    # changing trial gold must not change anything reported from test, except through the rule choice
    other = data.copy()
    other.loc[data["split"] == "trial", sm.GOLD_EMOTIONS] = 0
    w = sm.run_validation(other, raw, lex, gv, grid, n_boot=20)
    pd.testing.assert_frame_equal(v["correlations"], w["correlations"])
