import json
import sys

import pytest

from src.pipeline import HashCache, Pipeline, Step, StepFailed, code_deps

COPY = """import sys
from pathlib import Path
src, dst, log = map(Path, sys.argv[1:4])
with open(log, "a") as f:
    f.write(dst.name + "\\n")
dst.write_text(src.read_text().upper() + "!")
"""


@pytest.fixture
def proj(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("from .b import x\n")
    (tmp_path / "src" / "b.py").write_text("x = 1\n")
    (tmp_path / "src" / "unused.py").write_text("y = 2\n")
    (tmp_path / "copy.py").write_text("from src.a import x\n" + COPY)
    (tmp_path / "fail.py").write_text("import sys\nsys.exit(3)\n")
    (tmp_path / "in.txt").write_text("hello")
    return tmp_path


def steps(p, **kw):
    log = p / "runs.log"
    return [Step("s1", [p / "copy.py", p / "in.txt", p / "mid.txt", log], inputs=[p / "in.txt"],
                 outputs=[p / "mid.txt"], script=p / "copy.py", **kw),
            Step("s2", [p / "copy.py", p / "mid.txt", p / "out.txt", log], inputs=[p / "mid.txt"],
                 outputs=[p / "out.txt"], script=p / "copy.py")]


def runs(p):
    return (p / "runs.log").read_text().split() if (p / "runs.log").exists() else []


def pipe(p, st=None):
    return Pipeline(st or steps(p), p, p / ".state", log=lambda *_: None)


def test_code_deps_follows_absolute_and_relative_imports(proj):
    deps = [d.name for d in code_deps(proj / "copy.py", proj)]
    assert sorted(deps) == ["a.py", "b.py", "copy.py"]


def test_skips_when_nothing_changed(proj):
    pipe(proj).run()
    assert runs(proj) == ["mid.txt", "out.txt"] and (proj / "out.txt").read_text() == "HELLO!!"
    report = pipe(proj).run()
    assert [a for _, a, _, _ in report] == ["skip", "skip"] and len(runs(proj)) == 2


def test_input_change_propagates_downstream(proj):
    pipe(proj).run()
    (proj / "in.txt").write_text("bye")
    pipe(proj).run()
    assert runs(proj)[2:] == ["mid.txt", "out.txt"] and (proj / "out.txt").read_text() == "BYE!!"


def test_identical_rewrite_does_not_rerun_downstream(proj):
    pipe(proj).run()
    pipe(proj).run(force={"s1"})                    # s1 rewrites mid.txt with identical content
    assert runs(proj)[2:] == ["mid.txt"]


def test_code_dependency_change_reruns(proj):
    pipe(proj).run()
    (proj / "src" / "unused.py").write_text("y = 3\n")
    assert len(runs(proj)) == 2 and pipe(proj).status(steps(proj)[0])[0] == "skip"
    (proj / "src" / "b.py").write_text("x = 2\n")
    action, reason = pipe(proj).status(steps(proj)[0])
    assert action == "run" and "src/b.py" in reason


def test_modified_or_missing_output_reruns(proj):
    pipe(proj).run()
    (proj / "out.txt").write_text("tampered")
    assert pipe(proj).status(steps(proj)[1]) == ("run", "output modified: out.txt")
    (proj / "out.txt").unlink()
    assert pipe(proj).status(steps(proj)[1])[1] == "missing out.txt"


def test_argument_change_reruns(proj):
    pipe(proj).run()
    st = steps(proj)
    st[0].cmd = [*st[0].cmd, "--flag"]
    assert pipe(proj, st).status(st[0]) == ("run", "arguments changed")


def test_dry_run_runs_nothing_and_flags_downstream(proj):
    pipe(proj).run()
    (proj / "in.txt").write_text("new")
    report = pipe(proj).run(dry_run=True)
    assert [a for _, a, _, _ in report] == ["would run", "would run"]
    assert "upstream" in report[1][2] and len(runs(proj)) == 2


def test_failure_stops_and_writes_no_stamp(proj):
    st = [Step("bad", [proj / "fail.py"], outputs=[proj / "x.txt"]), *steps(proj)]
    with pytest.raises(StepFailed) as e:
        pipe(proj, st).run()
    assert e.value.report[-1][1] == "FAILED" and runs(proj) == []
    assert not (proj / ".state" / "bad.json").exists()


def test_create_only_and_condition(proj):
    (proj / "sheet.csv").write_text("my labels")
    st = [Step("sheet", [proj / "fail.py"], outputs=[proj / "sheet.csv"], create_only=True),
          Step("validate", [proj / "fail.py"], condition=lambda: "waiting for labels")]
    report = pipe(proj, st).run(force={"sheet"})
    assert [(n, a) for n, a, _, _ in report] == [("sheet", "skip"), ("validate", "wait")]
    assert (proj / "sheet.csv").read_text() == "my labels"


def test_adopt_records_existing_outputs_without_running(proj):
    (proj / "mid.txt").write_text("existing")
    (proj / "out.txt").write_text("existing")
    report = pipe(proj).run(adopt=True)
    assert [a for _, a, _, _ in report] == ["skip", "skip"] and runs(proj) == []
    assert json.loads((proj / ".state" / "s1.json").read_text())["how"] == "adopted existing outputs"
    assert pipe(proj).status(steps(proj)[0]) == ("skip", "up to date")


def test_hash_cache_reuses_and_detects_change(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    c = HashCache(tmp_path / "cache.json")
    h = c.sha(f)
    c.save()
    assert HashCache(tmp_path / "cache.json").sha(f) == h
    f.write_bytes(b"abcd")
    assert HashCache(tmp_path / "cache.json").sha(f) != h and c.sha(tmp_path / "missing") is None


def test_config_names_found_in_absolute_relative_and_nested_imports(proj):
    from src.pipeline import config_names
    (proj / "src" / "c.py").write_text("from .config import A, B\ndef f():\n    from .config import C\n")
    (proj / "s.py").write_text("from src.config import D\n")
    assert config_names([proj / "src" / "c.py", proj / "s.py"], proj) == {"A", "B", "C", "D"}
    (proj / "w.py").write_text("from src.config import *\n")
    assert config_names([proj / "w.py"], proj) is None


def test_only_used_config_constants_trigger_rerun(proj):
    cfg = proj / "src" / "config.py"
    cfg.write_text("USED = 1\nOTHER = 1\n")
    (proj / "src" / "b.py").write_text("from .config import USED\nx = USED\n")
    pipe(proj).run()
    cfg.write_text("USED = 1\nOTHER = 2\nNEW = 3\n")                 # unrelated constants change
    assert pipe(proj).status(steps(proj)[0]) == ("skip", "up to date")
    cfg.write_text("USED = 5\nOTHER = 2\nNEW = 3\n")
    assert pipe(proj).status(steps(proj)[0]) == ("run", "changed: src/config.py:USED")


def test_adopt_never_skips_a_forced_step(proj):
    (proj / "mid.txt").write_text("existing")
    (proj / "out.txt").write_text("existing")
    pipe(proj).run(adopt=True, force={"s1"})
    assert runs(proj)[0] == "mid.txt"
