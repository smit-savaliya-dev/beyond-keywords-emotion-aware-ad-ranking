from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.labelling import (hours_since_created, load_sheet, make_relabel_sample, next_unlabelled, render,
                           run_session)

T0 = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)


def sheet_file(tmp_path, labels=("", "", ""), name="sheet.csv"):
    df = pd.DataFrame({"sample_id": [str(i + 1) for i in range(len(labels))],
                       "news_id": [f"N{i}" for i in range(len(labels))],
                       "title": [f"Café headline {i}" for i in range(len(labels))],
                       "abstract": ["An abstract.", "", "Another one."][:len(labels)] + [""] * (len(labels) - 3),
                       "your_label": list(labels)})
    path = tmp_path / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def keys(*seq):
    it = iter(seq)
    return lambda: next(it)


def labels(path):
    return load_sheet(path)["your_label"].tolist()


def test_labels_saved_after_every_answer_and_backup_once(tmp_path):
    path = sheet_file(tmp_path)
    seen = []

    def read_key():
        seen.append(labels(path))                     # state on disk before each key press
        return ["1", "6", "q"][len(seen) - 1]

    res = run_session(path, read_key, out=lambda *_: None, backup_dir=tmp_path / "bk", now=T0)
    assert seen == [["", "", ""], ["joy", "", ""], ["joy", "neutral", ""]]
    assert labels(path) == ["joy", "neutral", ""] and res["labelled_now"] == 2
    backups = list((tmp_path / "bk").iterdir())
    assert len(backups) == 1 and load_sheet(backups[0])["your_label"].tolist() == ["", "", ""]
    assert load_sheet(path)["title"][0] == "Café headline 0"          # text and encoding preserved


def test_resume_skip_back_and_unknown_key(tmp_path):
    path = sheet_file(tmp_path, labels=("sadness", "", ""))
    out = []
    run_session(path, keys("x", "s", "4", "b", "3", "q"), out=out.append, backup_dir=None)
    # starts at item 2 (first unlabelled); x ignored; s skips to 3; 4 = fear on 3; b back to 3; 3 = anger on 3
    assert labels(path) == ["sadness", "", "anger"]
    assert any("Unknown key" in o for o in out) and "item 2/3" in out[0]


def test_quit_without_answer_writes_nothing_and_no_backup(tmp_path):
    path = sheet_file(tmp_path)
    before = path.read_bytes()
    res = run_session(path, keys("q"), out=lambda *_: None, backup_dir=tmp_path / "bk")
    assert path.read_bytes() == before and res["backup"] is None and not (tmp_path / "bk").exists()


def test_finishes_when_all_labelled(tmp_path):
    path = sheet_file(tmp_path)
    out = []
    res = run_session(path, keys("1", "2", "5"), out=out.append)
    assert labels(path) == ["joy", "sadness", "surprise"] and res["labelled_total"] == 3
    assert "All items labelled" in out[-1]
    assert run_session(path, keys(), out=out.append)["labelled_now"] == 0


def test_next_unlabelled_wraps_and_render_has_no_model_output(tmp_path):
    sheet = load_sheet(sheet_file(tmp_path, labels=("", "joy", "")))
    assert next_unlabelled(sheet, 0) == 2 and next_unlabelled(sheet, 2) == 0
    text = render(sheet, 1)
    assert "your current label: joy" in text and "(no abstract)" in text
    assert "dominant" not in text and "score" not in text


def test_make_relabel_sample_requires_complete_sheet_and_never_overwrites(tmp_path):
    incomplete = sheet_file(tmp_path, labels=("joy", "", "fear"))
    with pytest.raises(ValueError, match="finish"):
        make_relabel_sample(incomplete, tmp_path / "re.csv", n=2)
    full = sheet_file(tmp_path, labels=("joy", "neutral", "fear"), name="full.csv")
    out = tmp_path / "re.csv"
    sub = make_relabel_sample(full, out, n=2, seed=0, now=T0)
    assert len(sub) == 2 and labels(out) == ["", ""]
    assert set(load_sheet(out)["sample_id"]) <= {"1", "2", "3"}
    assert make_relabel_sample(full, tmp_path / "re2.csv", n=2, seed=0)["sample_id"].tolist() == \
        sub["sample_id"].tolist()                                             # seeded
    with pytest.raises(FileExistsError):
        make_relabel_sample(full, out, n=2)
    assert hours_since_created(out, now=T0 + timedelta(hours=30)) == pytest.approx(30.0)
