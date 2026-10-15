import numpy as np
import pytest

from src.config import NRC_EMOTIONS, NRC_SENTIMENTS, VAD_DIMS
from src.lexicon import EMOLEX_COLUMNS, load_emolex, load_vad, resolve_terms, tokenize

QUIET = dict(log=lambda *_: None)
ALL10 = NRC_EMOTIONS + NRC_SENTIMENTS


def _emolex_long(path, assoc, header=False, crlf=True):
    lines = ["word\temotion\tassociation"] if header else []
    for word, emos in assoc.items():
        lines += [f"{word}\t{e}\t{int(e in emos)}" for e in sorted(ALL10)]
    path.write_text(("\r\n" if crlf else "\n").join(lines) + "\n", encoding="utf-8", newline="")
    return path


ASSOC = {"abandon": {"fear", "sadness", "negative"}, "happy": {"joy", "trust", "positive"}, "table": set(),
         "nan": {"anger"}}


def test_tokenize():
    assert tokenize("Trump's WIN: 3 cheers, well-known!") == ["trump", "s", "win", "cheers", "well", "known"]
    assert tokenize("") == [] and tokenize(None) == ["none"]


@pytest.mark.parametrize("header", [False, True])
def test_load_emolex_long(tmp_path, header):
    table, info = load_emolex(_emolex_long(tmp_path / "emo.txt", ASSOC, header=header), **QUIET)
    assert list(table.columns) == EMOLEX_COLUMNS and table.dtypes.eq("int8").all()
    assert set(table.index) == {"abandon", "happy", "table", "nan"}       # "nan" stays a word
    assert table.loc["abandon", "fear"] == 1 and table.loc["abandon", "joy"] == 0
    assert table.loc["table"].sum() == 0
    assert info["format"] == "long" and info["n_words"] == 4 and info["n_words_with_emotion"] == 3


def test_load_emolex_wide(tmp_path):
    cols = ["English Word", *ALL10, "French Word"]
    lines = ["\t".join(cols), "\t".join(["Happy", *[str(int(e in ASSOC["happy"])) for e in ALL10], "heureux"])]
    p = tmp_path / "wide.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    table, info = load_emolex(p, **QUIET)
    assert info["format"] == "wide" and table.loc["happy", "trust"] == 1 and table.loc["happy", "anger"] == 0


def test_load_emolex_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="data.external.nrc|never commit"):
        load_emolex(tmp_path / "missing.txt", **QUIET)
    bad = tmp_path / "bad.txt"
    bad.write_text("happy\tjoy\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="0/1"):
        load_emolex(bad, **QUIET)
    bad.write_text("happy\tjoy\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differ from expected"):
        load_emolex(bad, **QUIET)
    p = _emolex_long(tmp_path / "dup.txt", {"happy": {"joy"}})
    p.write_text(p.read_text(encoding="utf-8") + "happy\tjoy\t0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        load_emolex(p, **QUIET)


def _vad(path, rows, header="term\tvalence\tarousal\tdominance"):
    lines = ([header] if header else []) + ["\t".join(map(str, r)) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_vad_v2_scale_kept(tmp_path):
    p = _vad(tmp_path / "v2.txt", [("happy", 0.9, 0.4, 0.3), ("sad", -0.8, -0.3, -0.5), ("a bit", -0.1, -0.2, 0.0)])
    table, info = load_vad(p, **QUIET)
    assert info["scale"] == "-1..1" and not info["rescaled_to_minus1_1"] and info["header"]
    assert info["n_unigrams"] == 2 and info["n_multiword"] == 1
    assert table.loc["sad", "valence"] == pytest.approx(-0.8) and list(table.columns) == VAD_DIMS


@pytest.mark.parametrize("header", ["Word\tValence\tArousal\tDominance", None])
def test_load_vad_v1_rescaled(tmp_path, header):
    p = _vad(tmp_path / "v1.txt", [("Happy", 1.0, 0.5, 0.75), ("sad", 0.0, 0.25, 0.5)], header=header)
    table, info = load_vad(p, **QUIET)
    assert info["scale"] == "0..1" and info["rescaled_to_minus1_1"] and info["header"] == (header is not None)
    assert np.allclose(table.loc["happy"], [1.0, 0.0, 0.5])
    assert np.allclose(table.loc["sad"], [-1.0, -0.5, 0.0])


def test_load_vad_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="NRC VAD"):
        load_vad(tmp_path / "missing.txt", **QUIET)
    with pytest.raises(ValueError, match="neither"):
        load_vad(_vad(tmp_path / "a.txt", [("x", 3.0, 0.1, 0.1)]), **QUIET)
    with pytest.raises(ValueError, match="non-numeric"):
        load_vad(_vad(tmp_path / "b.txt", [("x", 0.1, 0.1, 0.1), ("y", "high", 0.1, 0.1)]), **QUIET)
    with pytest.raises(ValueError, match="4 tab-separated"):
        load_vad(_vad(tmp_path / "c.txt", [("x", 0.1, 0.1)]), **QUIET)
    with pytest.raises(ValueError, match="conflicting"):
        load_vad(_vad(tmp_path / "d.txt", [("x", 0.1, 0.1, 0.1), ("x", 0.2, 0.1, 0.1)]), **QUIET)


def test_resolve_terms_overrides():
    mapping, used = resolve_terms(["joy", "nervousness"], {"joy", "nervous"}, {"nervousness": "nervous"})
    assert mapping == {"joy": "joy", "nervousness": "nervous"} and used == ["nervousness"]
    with pytest.raises(KeyError, match="nervousness"):
        resolve_terms(["nervousness"], {"joy"}, {"nervousness": "nervous"})
