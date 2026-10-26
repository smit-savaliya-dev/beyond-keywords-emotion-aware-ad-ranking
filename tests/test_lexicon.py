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


from src.config import GO_VAD_OVERRIDES  # noqa: E402
from src.lexicon import (LEXICON_COLUMNS, SHARE_COLUMNS, combine_lexicon_fields, field_lexicon_features,  # noqa: E402
                         lexicon_counts, top_matched_words)


@pytest.fixture
def lexicons(tmp_path):
    emolex, _ = load_emolex(_emolex_long(tmp_path / "emo.txt", ASSOC), **QUIET)
    vad, _ = load_vad(_vad(tmp_path / "v.txt", [("happy", 0.8, 0.4, 0.2), ("sad", -0.6, -0.2, -0.4),
                                                ("of", 0.0, 0.0, 0.0), ("very happy", 1.0, 1.0, 1.0)]), **QUIET)
    return emolex, vad


def test_lexicon_counts_and_field_features(lexicons):
    emolex, vad = lexicons
    texts = ["Happy, happy table of abandon!", "sad", "", "very happy"]
    counts = lexicon_counts(texts, emolex, vad)
    assert counts.loc[0, "n_tokens"] == 5 and counts.loc[0, "emo_tokens"] == 3      # table has no association
    assert counts.loc[0, "n_joy"] == 2 and counts.loc[0, "n_fear"] == 1 and counts.loc[0, "pos_tokens"] == 2
    assert counts.loc[0, "vad_tokens"] == 3                                         # happy x2, of
    f = field_lexicon_features(counts)
    # associations: happy x2 -> joy, trust (4); abandon -> fear, sadness (2)
    assert f.loc[0, "nrc_joy"] == pytest.approx(2 / 6) and f.loc[0, "nrc_fear"] == pytest.approx(1 / 6)
    assert f.loc[0, SHARE_COLUMNS].sum() == pytest.approx(1.0)
    assert f.loc[0, "nrc_coverage"] == pytest.approx(3 / 5) and f.loc[0, "nrc_negative"] == pytest.approx(1 / 5)
    assert f.loc[0, "vad_valence"] == pytest.approx(1.6 / 3) and f.loc[0, "vad_coverage"] == pytest.approx(3 / 5)
    assert f.loc[1, "nrc_has_hits"] == 0 and f.loc[1, SHARE_COLUMNS].isna().all()  # "sad" not in fake EmoLex
    assert f.loc[1, "vad_valence"] == pytest.approx(-0.6)
    assert f.loc[2, "nrc_coverage"] == 0 and f.loc[2, "vad_has_hits"] == 0          # empty text: no division by 0
    assert f.loc[3, "vad_valence"] == pytest.approx(0.8)                            # multi-word entry not matched


def test_combine_lexicon_fields_renormalises_and_falls_back(lexicons):
    emolex, vad = lexicons
    title = field_lexicon_features(lexicon_counts(["happy", "table", "table", "abandon"], emolex, vad))
    abstract = field_lexicon_features(lexicon_counts(["abandon", "abandon", "", "happy"], emolex, vad))
    out = combine_lexicon_fields(title, abstract, [True, True, False, False], 1 / 3)
    assert list(out.columns) == LEXICON_COLUMNS
    assert out.loc[0, "nrc_joy"] == pytest.approx(1 / 3 * 0.5) and out.loc[0, "nrc_fear"] == pytest.approx(2 / 3 * 0.5)
    assert out.loc[1, "nrc_fear"] == pytest.approx(0.5)              # title has no hits -> abstract weight 1
    assert out.loc[1, "nrc_coverage"] == pytest.approx(2 / 3)        # plain weights: 1/3 * 0 + 2/3 * 1
    assert out.loc[2, "nrc_has_hits"] == 0 and (out.loc[2, SHARE_COLUMNS] == 0).all()
    assert out.loc[2, "vad_valence"] == 0 and out.loc[2, "vad_has_hits"] == 0
    assert out.loc[3, "nrc_fear"] == pytest.approx(0.5) and out.loc[3, "nrc_joy"] == 0   # abstract ignored
    assert not out.isna().any().any()


def test_top_matched_words():
    top = top_matched_words(["Happy happy sad", "happy"], {"happy", "sad"}, n=1)
    assert top.to_dict("records") == [{"word": "happy", "count": 3}]


def test_go_vad(lexicons):
    from src.emotion import GOEMOTIONS_LABELS
    from src.emotion_features import GO_VAD_COLUMNS, GO_VAD_LABELS, go_vad
    import pandas as pd
    words = {l: (0.1 * (i % 5) - 0.2, 0.0, 0.5) for i, l in enumerate(GO_VAD_LABELS)}
    words["joy"], words["sadness"] = (1.0, 0.6, 0.4), (-1.0, -0.2, -0.6)
    vad = pd.DataFrame(words.values(), index=list(words), columns=VAD_DIMS)
    raw = pd.DataFrame(np.zeros((3, 28)), columns=GOEMOTIONS_LABELS)
    raw.insert(0, "news_id", ["A", "B", "C"])
    raw.loc[0, "joy"] = 0.6
    raw.loc[0, "sadness"] = 0.2
    raw.loc[1, "neutral"] = 0.99                                      # neutral ignored -> all zero -> 0
    out, used = go_vad(raw, vad, {})
    assert list(out.columns) == ["news_id", *GO_VAD_COLUMNS] and used == []
    assert out.loc[0, "go_vad_valence"] == pytest.approx((0.6 - 0.2) / 0.8)
    assert out.loc[0, "go_vad_dominance"] == pytest.approx((0.6 * 0.4 - 0.2 * 0.6) / 0.8)
    assert (out.loc[1, GO_VAD_COLUMNS] == 0).all()
    renamed = vad.rename(index={"nervousness": "nervous"})
    out2, used2 = go_vad(raw, renamed, {"nervousness": "nervous"})
    assert used2 == ["nervousness"] and np.allclose(out2[GO_VAD_COLUMNS], out[GO_VAD_COLUMNS])
    with pytest.raises(KeyError, match="nervousness"):
        go_vad(raw, renamed, GO_VAD_OVERRIDES)


def test_nrc_to_six_mapping_neutral_and_ties():
    import pandas as pd
    from src.lexicon import nrc_to_six
    rows = [
        {"nrc_joy": 0.6, "nrc_trust": 0.4},                       # joy
        {"nrc_anger": 0.2, "nrc_disgust": 0.3, "nrc_fear": 0.4},  # anger + disgust = 0.5 > fear
        {},                                                       # no hits -> neutral
        {"nrc_trust": 0.5, "nrc_anticipation": 0.5},              # trust/anticipation only -> neutral
        {"nrc_sadness": 0.5, "nrc_surprise": 0.5},                # tie
    ]
    nrc = pd.DataFrame(rows, columns=SHARE_COLUMNS).fillna(0.0)
    labels, tied = nrc_to_six(nrc, seed=0)
    assert labels.tolist()[:4] == ["joy", "anger", "neutral", "neutral"]
    assert labels[4] in {"sadness", "surprise"} and tied.tolist() == [False, False, False, False, True]
    assert labels.equals(nrc_to_six(nrc, seed=0)[0])                                     # seeded
    picks = {nrc_to_six(nrc.iloc[[4] * 40].reset_index(drop=True), seed=1)[0].nunique()}
    assert picks == {2}                                                                   # random, not first-class
