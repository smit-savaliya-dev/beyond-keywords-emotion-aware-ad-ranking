import numpy as np
import pandas as pd

from src import story_charts as sc
from src.emotion import SIX_CLASSES


def dots(rows, significant=True):
    return pd.DataFrame([{"row": r, "part": part, "dz": 0.1 + i / 50, "dz_ci_low": 0.08, "dz_ci_high": 0.2,
                          "significant_both": significant and part == "fit"}
                         for i, r in enumerate(rows) for part in ("fit", "valid")])


def test_all_eight_charts_write_files(tmp_path):
    mix = pd.DataFrame(np.full((2, 6), 100 / 6), index=["old", "new"], columns=SIX_CLASSES)
    grid = pd.DataFrame([{"share": s, "t": t, "pct_neutral": 100 * t + s} for s in (1.0, 0.5) for t in (0.05, 0.1)])
    heat = pd.DataFrame(np.full((3, 6), 16.0), index=list("abc"), columns=SIX_CLASSES)
    va = pd.DataFrame({"category": ["a", "b"], "valence": [0.1, 0.2], "arousal": [0.1, 0.2], "v_low": [0.0, 0.1],
                       "v_high": [0.2, 0.3], "a_low": [0.0, 0.1], "a_high": [0.2, 0.3]})
    sc.emotion_mix(mix, tmp_path / "1.png", "t", "c")
    sc.neutral_curve(grid, 0.5, 0.05, tmp_path / "2.png", "t", "c")
    sc.intensity_compare(np.linspace(0.01, 0.9, 50), np.linspace(0.02, 0.9, 50), 0.05, tmp_path / "3.png", "t", "c")
    sc.category_heatmap(heat, tmp_path / "4.png", "t", "c")
    sc.valence_arousal_map(va, {"a": (6, -13)}, tmp_path / "5.png", "t", "c")
    sc.clicked_effects(dots(["x", "y"]), ["x"], tmp_path / "6.png", "t", "c")
    sc.congruence_effects(dots(["raw", "personal"]), tmp_path / "7.png", "t", "c")
    sc.context_effects(dots(["last 5", "last 1"]), dots(["last 5", "previous 5"], False), tmp_path / "8.png", "t", "c")
    assert sorted(p.name for p in tmp_path.glob("*.png")) == [f"{i}.png" for i in range(1, 9)]
    assert all(p.stat().st_size > 0 for p in tmp_path.glob("*.png"))
