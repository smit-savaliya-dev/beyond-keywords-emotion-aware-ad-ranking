import ast
import json
from pathlib import Path

import pandas as pd

from src.summary import build_summary

ROOT = Path(__file__).resolve().parents[1]


def fake_tables(t):
    pd.DataFrame({"rule": ["v1", "v2", "v2b"], "joy": [8.7, 34.5, 20.7], "sadness": [5.3, 18.9, 19.8], "anger": [0.6, 5.6, 2.8],
                  "fear": [0.4, 1.9, 1.9], "surprise": [4.0, 13.5, 15.8], "neutral": [81.1, 25.6, 39.0]}).to_csv(t / "story_emotion_mix.csv", index=False)
    pd.DataFrame({"feature": ["go_pct_fear", "go_x"], "group": ["item", "item"], "fit_dz": [0.166, 0.1], "valid_dz": [0.168, -0.04],
                  "keep": [True, False], "reason": ["replicated on valid", "sign flips on valid"]}).to_csv(t / "emotion_spec_v2_selection.csv", index=False)
    rows = [{"feature": f, "part": p, "dz": 0.2, "significant_both": True}
            for f in ("cong_pct_cos", "cong_pct_cos_personal", "cong_pct_cos_userspecific") for p in ("fit", "valid")]
    pd.DataFrame(rows).to_csv(t / "emotion_signal_v2.csv", index=False)
    pd.DataFrame([{"sample": "all", "role": "user-specific", "window": w, "part": p, "dz": 0.1}
                  for w in ("long_50", "short_5", "short_3", "short_1") for p in ("fit", "valid")]).to_csv(t / "context_windows_signal.csv", index=False)
    pd.DataFrame([{"sample": "earlier click >= 1", "role": "user-specific", "window": w, "part": p, "dz": 0.09}
                  for w in ("long_50", "log_5", "log_3", "log_1") for p in ("fit", "valid")]).to_csv(t / "true_recency_signal.csv", index=False)
    (t / "true_recency_meta.json").write_text(json.dumps({"adds_beyond_long_term": {
        "log_1": {"adds_beyond_long_term": True, "fit_dz": 0.05, "valid_dz": 0.02}}}), encoding="utf-8")
    pd.DataFrame({"model": ["M0_popularity"], "part": ["dev"], "auc": [0.589], "mrr": [0.27], "ndcg@5": [0.29], "ndcg@10": [0.35],
                  "hr@10": [0.66]}).to_csv(t / "summary.csv", index=False)


def test_summary_covers_every_section(tmp_path):
    fake_tables(tmp_path)
    (tmp_path / "data_summary.json").write_text(json.dumps({"news_items": 65238, "empty_abstract_pct": 5.23, "parts": {
        "fit": {"impressions": 126695, "candidate_rows": 4621015, "users": 46012}}}), encoding="utf-8")
    text = build_summary(tmp_path, tmp_path)
    for needle in ("65,238", "81.1", "39.0", "go_pct_fear", "1 of 2 candidates flip", "user-specific", "last 50", "last 1 log click",
                   "adds", "M0_popularity", "0.5890"):
        assert needle in text, needle


def test_summary_survives_missing_tables(tmp_path):
    assert build_summary(tmp_path, tmp_path).count("not available") >= 5


def test_summary_module_reads_no_impression_or_behavior_data():
    for f in ("src/summary.py", "scripts/show_results.py"):
        source = (ROOT / f).read_text(encoding="utf-8")
        strings = {n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert not any(s.endswith(("impressions.parquet", "behaviors.parquet")) for s in strings), f
        assert "read_parquet" not in source, f


def test_semeval_section(tmp_path):
    import json
    from src.summary import semeval_lines
    rows = [{"kind": "correlation", "gold": g, "source": s, "spearman": 0.3} for g in ("anger", "fear", "joy", "sadness", "surprise")
            for s in ("goemotions_group", "nrc_emolex")]
    rows += [{"kind": "correlation", "gold": "valence", "source": s, "spearman": 0.4} for s in ("go_vad_valence", "vad_valence")]
    rows.append({"kind": "agreement", "cutoff": 50, "source": "v2b_t0.05", "kappa": 0.2, "kappa_ci_low": 0.1, "kappa_ci_high": 0.3,
                 "pct_pred_neutral": 30.0, "chosen": True})
    pd.DataFrame(rows).to_csv(tmp_path / "semeval2007_validation.csv", index=False)
    meta = {"n": {"test": 1000}, "official_cutoff": 50, "neutral": {"gold_test_pct": 60.0, "chosen_test_pct": 55.0, "chosen_mind_title_only_pct": 45.0, "chosen_mind_pct": 39.0},
            "verdict_goemotions": {"verdict": "CONFIRM"}, "verdict_nrc": {"verdict": "HOLD"}}
    (tmp_path / "semeval2007_validation_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    text = "\n".join(semeval_lines(tmp_path))
    assert "chosen on trial" in text and "GoEmotions CONFIRM, NRC HOLD" in text and "60.0%" in text
    assert semeval_lines(tmp_path / "missing") == ["  not available"]
