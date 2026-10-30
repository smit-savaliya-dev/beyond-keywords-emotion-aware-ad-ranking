"""Terminal summary of the project, built only from saved result tables. Impression data, including dev, is never read:
dev metrics come from results/tables/summary.csv, written once by the --final-dev runs."""
import json

import pandas as pd

WIDTH = 78
ROLE_NAMES = {"cong_pct_cos": "raw match", "cong_pct_cos_personal": "personal", "cong_pct_cos_userspecific": "user-specific"}
WINDOWS = {"long_50": "last 50, decay 0.9", "short_5": "last 5", "short_3": "last 3", "short_1": "last 1"}
LOG_WINDOWS = {"long_50": "long-term profile", "log_5": "last 5 log clicks", "log_3": "last 3 log clicks", "log_1": "last 1 log click"}


def _section(title):
    return ["", title, "-" * WIDTH]


def _read(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else pd.read_csv(path)


def data_lines(processed):
    s = _read(processed / "data_summary.json")
    if s is None:
        return ["  not available"]
    lines = [f"  {s['news_items']:,} news items, {s['empty_abstract_pct']}% without an abstract"]
    for part, v in s["parts"].items():
        lines.append(f"  {part:6s} {v['impressions']:>7,} impressions  {v['candidate_rows']:>9,} candidates  {v['users']:>6,} users")
    return lines


def neutral_lines(tables):
    mix = _read(tables / "story_emotion_mix.csv")
    if mix is None:
        return ["  not available"]
    lines = [f"  {'rule':34s}{'neutral':>8s}{'joy':>7s}{'sadness':>9s}{'anger':>7s}{'fear':>6s}{'surprise':>10s}"]
    for _, r in mix.iterrows():
        lines.append(f"  {r.iloc[0]:34s}{r['neutral']:7.1f}%{r['joy']:7.1f}{r['sadness']:9.1f}{r['anger']:7.1f}{r['fear']:6.1f}{r['surprise']:10.1f}")
    return lines


def stable_signal_lines(tables, n=6):
    t = _read(tables / "emotion_spec_v2_selection.csv")
    if t is None:
        return ["  not available"]
    t = t[(t["group"] == "item") & t["keep"]].copy()
    t["strength"] = t[["fit_dz", "valid_dz"]].abs().min(axis=1)
    lines = [f"  {'feature':28s}{'fit dz':>9s}{'valid dz':>10s}"]
    for _, r in t.sort_values("strength", ascending=False).head(n).iterrows():
        lines.append(f"  {r['feature']:28s}{r['fit_dz']:+9.3f}{r['valid_dz']:+10.3f}")
    dropped = _read(tables / "emotion_spec_v2_selection.csv")
    flips = int((dropped["reason"] == "sign flips on valid").sum())
    lines.append(f"  {flips} of {len(dropped)} candidates flip sign on the next day")
    return lines


def congruence_lines(tables):
    t = _read(tables / "emotion_signal_v2.csv")
    if t is None:
        return ["  not available"]
    p = t[t["feature"].isin(ROLE_NAMES)].pivot(index="feature", columns="part", values="dz")
    both = t[t["feature"].isin(ROLE_NAMES)].groupby("feature")["significant_both"].all()
    lines = [f"  {'measure':20s}{'fit dz':>9s}{'valid dz':>10s}{'both levels':>13s}"]
    for f, name in ROLE_NAMES.items():
        lines.append(f"  {name:20s}{p.loc[f, 'fit']:9.3f}{p.loc[f, 'valid']:10.3f}{'yes' if both[f] else 'no':>13s}")
    return lines


def window_lines(tables):
    lines = ["  history column (user-specific dz):"]
    t = _read(tables / "context_windows_signal.csv")
    if t is None:
        lines.append("    not available")
    else:
        p = t[(t["sample"] == "all") & (t["role"] == "user-specific")].pivot(index="window", columns="part", values="dz")
        lines += [f"    {name:22s}fit {p.loc[w, 'fit']:.3f}   valid {p.loc[w, 'valid']:.3f}" for w, name in WINDOWS.items()]
    lines.append("  true recency, impressions with an earlier log click (user-specific dz):")
    t = _read(tables / "true_recency_signal.csv")
    m = _read(tables / "true_recency_meta.json")
    if t is None or m is None:
        lines.append("    not available")
        return lines
    s = t[t["sample"] == "earlier click >= 1"]
    p = s[s["role"] == "user-specific"].pivot(index="window", columns="part", values="dz")
    lines += [f"    {name:22s}fit {p.loc[w, 'fit']:.3f}   valid {p.loc[w, 'valid']:.3f}" for w, name in LOG_WINDOWS.items()]
    lines.append("  added beyond long-term (residual dz):")
    for w, r in m["adds_beyond_long_term"].items():
        lines.append(f"    {w:22s}fit {r['fit_dz']:.3f}   valid {r['valid_dz']:.3f}   {'adds' if r['adds_beyond_long_term'] else 'no'}")
    return lines


def baseline_lines(tables):
    t = _read(tables / "summary.csv")
    if t is None:
        return ["  not available"]
    lines = [f"  {'model':16s}{'part':7s}{'AUC':>8s}{'MRR':>8s}{'nDCG@5':>8s}{'nDCG@10':>9s}{'HR@10':>8s}"]
    for _, r in t.sort_values(["part", "model"], ascending=[False, True]).iterrows():
        lines.append(f"  {r['model']:16s}{r['part']:7s}{r['auc']:8.4f}{r['mrr']:8.4f}{r['ndcg@5']:8.4f}{r['ndcg@10']:9.4f}{r['hr@10']:8.4f}")
    return lines


def semeval_lines(tables):
    t = _read(tables / "semeval2007_validation.csv")
    m = _read(tables / "semeval2007_validation_meta.json")
    if t is None or m is None:
        return ["  not available"]
    c = t[t["kind"] == "correlation"]
    lines = [f"  {m['n']['test']:,} expert-rated news headlines (test split); this validates the title part of the scoring", "",
             f"  {'Spearman with human rating':30s}{'GoEmotions':>12s}{'NRC':>8s}"]
    get = lambda gold, source: c[(c["gold"] == gold) & (c["source"] == source)]["spearman"].iloc[0]
    for e in ("anger", "fear", "joy", "sadness", "surprise"):
        lines.append(f"  {e:30s}{get(e, 'goemotions_group'):12.3f}{get(e, 'nrc_emolex'):8.3f}")
    lines += [f"  {'valence':30s}{get('valence', 'go_vad_valence'):12.3f}{get('valence', 'vad_valence'):8.3f}", ""]
    a = t[(t["kind"] == "agreement") & (t["cutoff"] == m["official_cutoff"])]
    lines.append(f"  {'label agreement':30s}{'kappa':>12s}{'95% CI':>18s}{'neutral':>9s}")
    for _, r in a.iterrows():
        mark = " (chosen on trial)" if r["chosen"] else ""
        ci = f"{r['kappa_ci_low']:.3f} to {r['kappa_ci_high']:.3f}"
        lines.append(f"  {r['source']:30s}{r['kappa']:12.3f}{ci:>18s}{r['pct_pred_neutral']:8.1f}%{mark}")
    lines += [f"  human neutral share {m['neutral']['gold_test_pct']:.1f}%, chosen rule on MIND {m['neutral']['chosen_mind_pct']:.1f}%",
              f"  spec check (D48): GoEmotions {m['verdict_goemotions']['verdict']}, NRC {m['verdict_nrc']['verdict']}"]
    return lines


def build_summary(tables, processed):
    lines = ["=" * WIDTH, "BEYOND KEYWORDS: RESULTS SUMMARY", "=" * WIDTH]
    for title, body in (("Data", data_lines(processed)), ("Neutral share and emotion mix (% of items)", neutral_lines(tables)),
                        ("Most stable click signals (clicked vs not, same list)", stable_signal_lines(tables)),
                        ("Emotional congruence with the user's recent reading", congruence_lines(tables)),
                        ("Which context window matters", window_lines(tables)), ("External validation: SemEval-2007 headlines", semeval_lines(tables)),
                        ("Baselines (no emotion)", baseline_lines(tables))):
        lines += _section(title) + body
    lines += ["", "dz: effect size in SD units; 0.1-0.2 is small but real. Associations only, not ranking gains.", "=" * WIDTH]
    return "\n".join(lines)
