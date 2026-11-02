import pytest
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
    sc.neutral_curve(grid, 0.5, 0.05, tmp_path / "2.png", "t", "c", extra=("v2b", grid[grid["share"] == 0.5]))
    sc.intensity_compare(np.linspace(0.01, 0.9, 50), np.linspace(0.02, 0.9, 50), 0.05, tmp_path / "3.png", "t", "c")
    sc.category_heatmap(heat, tmp_path / "4.png", "t", "c")
    sc.valence_arousal_map(va, {"a": (6, -13)}, tmp_path / "5.png", "t", "c")
    sc.clicked_effects(dots(["x", "y"]), ["x"], tmp_path / "6.png", "t", "c")
    sc.congruence_effects(dots(["raw", "personal"]), tmp_path / "7.png", "t", "c")
    sc.context_effects(dots(["last 5", "last 1"]), dots(["last 5", "previous 5"], False), tmp_path / "8.png", "t", "c")
    assert sorted(p.name for p in tmp_path.glob("*.png")) == [f"{i}.png" for i in range(1, 9)]
    assert all(p.stat().st_size > 0 for p in tmp_path.glob("*.png"))


def test_semeval_correlations_chart_renders(tmp_path):
    from src.story_charts import semeval_correlations
    gold = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    rows = [(g, source, 0.3, 0.2, 0.4) for g in gold for source in ("goemotions_label", "nrc_emolex")]
    rows += [(g, "goemotions_group", 0.4, 0.3, 0.5) for g in gold if g != "disgust"]
    rows += [("valence", "go_vad_valence", 0.5, 0.4, 0.6), ("valence", "vad_valence", 0.3, 0.2, 0.4)]
    corr = pd.DataFrame(rows, columns=["gold", "source", "spearman", "spearman_low", "spearman_high"])
    path = tmp_path / "c.png"
    semeval_correlations(corr, path, "t", "c")
    assert path.stat().st_size > 1000


def fake_result_tables(t):
    import json
    from src.config import METRIC_COLUMNS
    pd.DataFrame({"model": ["M0_popularity", "M3_sbert_emotion"], "label": ["M0 popularity", "M3 = M2 + emotion"], "emotion": [False, True], "part": "dev",
                  "n": 100, "auc": [0.59, 0.63], "ci_low": [0.58, 0.62], "ci_high": [0.60, 0.64]}).to_csv(t / "model_ladder.csv", index=False)
    rows = [{"model_a": "M2_sbert", "model_b": "M3_sbert_emotion", "part": p, "metric": m, "mean_diff": 0.002, "ci_low": 0.001, "ci_high": 0.003,
             "significant_holm": m == "auc"} for p in ("valid", "dev") for m in METRIC_COLUMNS]
    pd.DataFrame(rows).to_csv(t / "m2m3_comparison.csv", index=False)
    grid = [{"variant": v, "alpha": a, "auc": 0.6 + a / 100} for v in ("cosine", "personal", "user_specific", "mood_repair", "title_only") for a in (0.0, 0.5, 1.0)]
    pd.DataFrame(grid).to_csv(t / "m3_fusion_grid_valid.csv", index=False)
    (t / "m3_fusion_meta.json").write_text(json.dumps({"chosen": {"variant": "personal", "alpha": 0.8}, "valid_auc": 0.65, "m2_valid_auc": 0.64}))
    rq = [{"breakdown": "clicked category", "group": g, "n": 10, "mean_diff": d, "ci_low": d - 0.001, "ci_high": d + 0.001} for g, d in (("news", 0.01), ("sports", -0.01))]
    rq += [{"breakdown": "known history items", "group": g, "n": 10, "mean_diff": d, "ci_low": d, "ci_high": d} for g, d in (("0", 0.0), ("1-5", 0.002), ("21+", 0.004))]
    pd.DataFrame(rq).to_csv(t / "m3_where_it_helps_dev.csv", index=False)
    gold = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    sem = [{"kind": "correlation", "gold": g, "source": s, "n": 1000, "spearman": 0.3, "spearman_low": 0.2, "spearman_high": 0.4} for g in gold
           for s in ("goemotions_label", "nrc_emolex")]
    sem += [{"kind": "correlation", "gold": g, "source": "goemotions_group", "n": 1000, "spearman": 0.4, "spearman_low": 0.3, "spearman_high": 0.5} for g in gold if g != "disgust"]
    sem += [{"kind": "correlation", "gold": "valence", "source": s, "n": 1000, "spearman": 0.5, "spearman_low": 0.4, "spearman_high": 0.6} for s in ("go_vad_valence", "vad_valence")]
    pd.DataFrame(sem).to_csv(t / "semeval2007_validation.csv", index=False)


def test_build_charts_makes_every_chart_with_real_numbers_in_captions(tmp_path):
    from src.config import RESULT_CHARTS
    from src.results_charts import build_charts
    fake_result_tables(tmp_path)
    out = tmp_path / "figs"
    caps = build_charts(tmp_path, out)
    assert list(caps) == RESULT_CHARTS and len(RESULT_CHARTS) <= 6
    assert all((out / f"{n}.png").stat().st_size > 1000 for n in RESULT_CHARTS)
    assert "1 of 10" in caps["2_m3_gain_by_metric"] and "M3 0.630" in caps["1_model_ladder"]
    assert "Descriptive" in caps["4_m3_gain_by_category"] and "+0.0100 (news)" in caps["4_m3_gain_by_category"]


def test_results_charts_read_no_impression_or_behavior_data():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for f in ("src/results_charts.py", "scripts/53_results_charts.py"):
        text = (root / f).read_text(encoding="utf-8")
        assert "impressions.parquet" not in text and "behaviors.parquet" not in text and "per_impression" not in text


def test_ladder_table_keeps_order_and_skips_missing_models(tmp_path):
    from src.results_tables import ladder_table
    pd.DataFrame({"auc": [0.6, 0.7, 0.8]}).to_parquet(tmp_path / "M3_sbert_emotion__dev.parquet")
    pd.DataFrame({"auc": [0.5, 0.6, 0.7]}).to_parquet(tmp_path / "M0_popularity__dev.parquet")
    t = ladder_table(tmp_path)
    assert t["model"].tolist() == ["M0_popularity", "M3_sbert_emotion"] and t["emotion"].tolist() == [False, True]
    assert t.loc[1, "auc"] == pytest.approx(0.7) and t.loc[1, "ci_low"] < 0.7 < t.loc[1, "ci_high"]


def test_importance_chart_renders(tmp_path):
    from src.results_charts import importance_chart
    t = pd.DataFrame({"feature": ["a", "b", "c", "d"], "family": ["popularity", "content", "context", "emotion"], "mean_abs_contrib": [0.4, 0.3, 0.2, 0.1]})
    importance_chart(t, tmp_path / "i.png", "t", "c")
    assert (tmp_path / "i.png").stat().st_size > 1000


def test_compare_pairs_and_merge_part(tmp_path):
    import json
    from src.results_tables import compare_pairs, merge_part
    (tmp_path / "data_summary.json").write_text(json.dumps({"parts": {"valid": {"impressions": 4}}}))
    pd.DataFrame({"imp_key": list("abcd"), "part": "valid", "user_id": ["u", "u", "v", "v"]}).to_parquet(tmp_path / "behaviors.parquet")
    from src.config import METRIC_COLUMNS
    base = {m: [0.5, 0.6, 0.7, 0.8] for m in METRIC_COLUMNS}
    pd.DataFrame({"imp_key": list("abcd"), **base}).to_parquet(tmp_path / "A__valid.parquet")
    up = {m: [0.6, 0.7, 0.75, 0.9] for m in METRIC_COLUMNS}
    pd.DataFrame({"imp_key": list("abcd"), **up}).to_parquet(tmp_path / "B__valid.parquet")
    comp, user = compare_pairs(tmp_path, [("A", "B")], "valid", tmp_path)
    assert len(comp) == 10 and comp["mean_diff"].iloc[0] == pytest.approx(0.0875) and user["users"].iloc[0] == 2
    path = tmp_path / "t.csv"
    merge_part(path, pd.DataFrame({"part": ["valid"], "x": [1]}), "valid")
    merge_part(path, pd.DataFrame({"part": ["dev"], "x": [2]}), "dev")
    merge_part(path, pd.DataFrame({"part": ["valid"], "x": [3]}), "valid")
    assert sorted(pd.read_csv(path)["x"]) == [2, 3]

