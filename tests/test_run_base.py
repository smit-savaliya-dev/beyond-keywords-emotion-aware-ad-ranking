import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_steps():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("run_base", ROOT / "scripts" / "run_base.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_steps(False)


def test_every_step_script_exists_and_steps_are_in_number_order():
    steps = load_steps()
    numbers = [int(s.name.split("_")[0]) for s in steps]
    assert numbers == sorted(numbers) and len(set(numbers)) == len(numbers)
    for s in steps:
        if s.script is not None:
            assert Path(s.script).exists(), s.name


def test_pipeline_never_runs_a_dev_final():
    for s in load_steps():
        assert "--final-dev" not in [str(c) for c in s.cmd], s.name


def test_only_the_two_scoring_steps_need_the_gpu():
    assert {s.name for s in load_steps() if s.gpu} == {"20_emotion_label_combined", "21_emotion_score_fields"}


def test_hand_label_sheet_is_never_regenerated():
    sheet = [s for s in load_steps() if s.name == "14_data_annotation_sheet"][0]
    assert sheet.create_only


def test_outputs_of_a_step_are_inputs_of_later_steps_only():
    steps = load_steps()
    made_by = {str(p): i for i, s in enumerate(steps) for p in s.outputs}
    for i, s in enumerate(steps):
        for p in s.inputs:
            assert made_by.get(str(p), -1) < i or made_by.get(str(p)) == i, f"{s.name} reads {p} before it is made"


def test_no_step_outputs_an_archived_file():
    archived = {p.name for p in (ROOT / "results" / "archive" / "figures").glob("*")}
    for s in load_steps():
        assert not archived & {Path(p).name for p in s.outputs}, s.name
