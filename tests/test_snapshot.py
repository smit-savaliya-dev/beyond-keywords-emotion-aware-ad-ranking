import json

import numpy as np
import pandas as pd

from src.snapshot import compare, same_file, snapshot_files, take


def setup(tmp_path):
    folder = tmp_path / "tables"
    folder.mkdir()
    pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]}).to_csv(folder / "t.csv", index=False)
    (folder / "m.json").write_text(json.dumps({"v": 0.5, "run_info": {"timestamp": 1}}), encoding="utf-8")
    pd.DataFrame({"id": ["A"], "h": [np.array(["N1", "N2"])], "x": [0.1]}).to_parquet(folder / "p.parquet")
    pd.DataFrame({"model": ["M0"]}).to_csv(folder / "runs_log.csv", index=False)
    return [("tables", folder)], folder


def test_take_then_compare_is_clean(tmp_path):
    roots, folder = setup(tmp_path)
    manifest = take(roots, tmp_path / "copy")
    assert set(snapshot_files(roots)) == set(manifest) and manifest["tables/t.csv"]["rows"] == 2
    assert compare(manifest, tmp_path / "copy", roots) == []


def test_tiny_float_noise_and_volatile_metadata_are_ignored(tmp_path):
    roots, folder = setup(tmp_path)
    manifest = take(roots, tmp_path / "copy")
    pd.DataFrame({"a": [1.0 + 1e-11, 2.0], "b": ["x", "y"]}).to_csv(folder / "t.csv", index=False)
    (folder / "m.json").write_text(json.dumps({"v": 0.5, "run_info": {"timestamp": 2}}), encoding="utf-8")
    pd.DataFrame({"model": ["M0", "M1"]}).to_csv(folder / "runs_log.csv", index=False)
    assert compare(manifest, tmp_path / "copy", roots) == []


def test_real_changes_are_reported(tmp_path):
    roots, folder = setup(tmp_path)
    manifest = take(roots, tmp_path / "copy")
    pd.DataFrame({"a": [1.0 + 1e-6, 2.0], "b": ["x", "y"]}).to_csv(folder / "t.csv", index=False)
    (folder / "m.json").write_text(json.dumps({"v": 0.6}), encoding="utf-8")
    pd.DataFrame({"id": ["A"], "h": [np.array(["N1", "N3"])], "x": [0.1]}).to_parquet(folder / "p.parquet")
    assert sorted(compare(manifest, tmp_path / "copy", roots)) == ["different: tables/m.json", "different: tables/p.parquet",
                                                                   "different: tables/t.csv"]
    (folder / "t.csv").unlink()
    assert "missing: tables/t.csv" in compare(manifest, tmp_path / "copy", roots)


def test_same_file_text_columns(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame({"s": ["x"]}).to_csv(a, index=False)
    pd.DataFrame({"s": ["z"]}).to_csv(b, index=False)
    assert not same_file(a, b)
