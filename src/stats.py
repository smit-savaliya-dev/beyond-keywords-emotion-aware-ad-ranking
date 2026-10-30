"""Paired model comparison on per-impression metrics, and Holm correction."""
import numpy as np
import pandas as pd
from scipy import stats as st


def holm(pvalues):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def paired_test(a, b):
    """b - a on paired values (NaN pairs dropped): mean diff with t-based 95% CI, dz, paired t-test p;
    Wilcoxon p as a robustness column only."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = b[ok] - a[ok]
    n = len(d)
    mean = float(d.mean()) if n else np.nan
    sd = float(d.std(ddof=1)) if n > 1 else np.nan
    if n > 1 and sd > 0:
        half = float(st.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        t = st.ttest_1samp(d, 0.0)
        t_stat, p = float(t.statistic), float(t.pvalue)
        dz = mean / sd
    else:
        half, t_stat, p, dz = 0.0, np.nan, 1.0, 0.0
    nz = d[d != 0]
    p_w = float(st.wilcoxon(nz).pvalue) if len(nz) else 1.0
    return {"n": n, "mean_a": float(a[ok].mean()) if n else np.nan, "mean_b": float(b[ok].mean()) if n else np.nan,
            "mean_diff": mean, "ci_low": mean - half, "ci_high": mean + half, "dz": dz, "t_stat": t_stat, "p": p,
            "p_wilcoxon": p_w, "pct_b_better": 100 * float((d > 0).mean()) if n else np.nan}


def dz_difference(diffs_a, diffs_b):
    """dz(a) - dz(b) on paired per-impression differences: each divided by its own sd, then a one-sample t-test.
    NaN pairs are dropped. -> mean difference with t-based 95% CI and p."""
    a, b = np.asarray(diffs_a, dtype=np.float64), np.asarray(diffs_b, dtype=np.float64)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    z = a / a.std(ddof=1) - b / b.std(ddof=1)
    half = float(st.t.ppf(0.975, len(z) - 1) * z.std(ddof=1) / np.sqrt(len(z)))
    return {"n": len(z), "dz_diff": float(z.mean()), "ci_low": float(z.mean() - half), "ci_high": float(z.mean() + half),
            "p": float(st.ttest_1samp(z, 0.0).pvalue)}


def compare_models(per_a, per_b, metrics, name_a="A", name_b="B", alpha=0.05):
    """Per-impression tables of two models on the same part -> one row per metric (B - A), Holm across metrics."""
    for name, t in ((name_a, per_a), (name_b, per_b)):
        if t["imp_key"].duplicated().any():
            raise ValueError(f"duplicate impressions in {name}")
    if set(per_a["imp_key"]) != set(per_b["imp_key"]):
        raise ValueError(f"{name_a} and {name_b} were scored on different impressions")
    m = per_a.merge(per_b, on="imp_key", suffixes=("_a", "_b"), validate="one_to_one")
    rows = [{"model_a": name_a, "model_b": name_b, "metric": k, **paired_test(m[f"{k}_a"], m[f"{k}_b"])}
            for k in metrics]
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy())
    out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"].to_numpy())
    out["significant_holm"] = out["p_holm"] < alpha
    return out
