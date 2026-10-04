"""EDA figures for MIND-small: six individual captioned PNGs plus one combined overview."""
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PART_ORDER = ["fit", "valid", "dev"]
PART_LABEL = {"fit": "Fit (train)", "valid": "Valid (train, last day)", "dev": "Dev (test)"}
PART_COLOR = {"fit": "#1f4e79", "valid": "#2e75b6", "dev": "#bf8f00"}
CAND_CLIP, HIST_CLIP = 120, 100

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.edgecolor": "#444444", "axes.labelcolor": "#222222", "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
})


def load(processed):
    news = pd.read_parquet(processed / "news.parquet")
    beh = pd.read_parquet(processed / "behaviors.parquet", columns=["part", "history_len"])
    imps = pd.read_parquet(processed / "impressions.parquet", columns=["part", "imp_key", "label"])
    summary = json.loads((processed / "data_summary.json").read_text(encoding="utf-8"))
    return news, beh, imps, summary


def per_impression_counts(imps):
    g = imps.groupby(["part", "imp_key"], observed=True)["label"].agg(n="size", clicks="sum")
    return g.reset_index()


def panel_part_sizes(ax, summary):
    parts = [p for p in PART_ORDER if p in summary["parts"]]
    counts = [summary["parts"][p]["impressions"] for p in parts]
    bars = ax.bar([PART_LABEL[p] for p in parts], counts, color=[PART_COLOR[p] for p in parts],
                 edgecolor="white", width=0.6)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Impressions")
    ax.tick_params(axis="x", labelsize=9)


def panel_candidates_hist(ax, counts):
    data = counts.loc[counts["part"].astype(str) == "fit", "n"]
    ax.hist(data.clip(upper=CAND_CLIP), bins=np.arange(0, CAND_CLIP + 5, 4), color="#1f4e79",
            edgecolor="white", linewidth=0.3)
    ax.axvline(data.mean(), color="#bf8f00", linestyle="--", linewidth=1.3, label=f"mean = {data.mean():.1f}")
    ax.axvline(data.median(), color="#548235", linestyle=":", linewidth=1.3, label=f"median = {data.median():.0f}")
    ax.set_xlabel(f"Candidates shown (values above {CAND_CLIP} grouped in last bar)")
    ax.set_ylabel("Impressions")
    ax.set_yscale("log")
    ax.legend(fontsize=8.5, frameon=False)
    ax.text(0.98, 0.95, f"max = {int(data.max())}", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#555555")


def panel_clicks_bar(ax, counts):
    data = counts.loc[counts["part"].astype(str) == "fit", "clicks"].clip(upper=4)
    share = data.value_counts(normalize=True).sort_index() * 100
    vals = [share.get(i, 0) for i in range(1, 5)]
    bars = ax.bar(["1", "2", "3", "4+"], vals, color="#2e75b6", edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Clicked items in the impression")
    ax.set_ylabel("Share of impressions (%)")


def panel_categories(ax, news):
    vc = news["category"].value_counts().sort_values(ascending=True).tail(10)
    ax.barh(vc.index, vc.to_numpy(), color="#1f4e79", edgecolor="white")
    ax.set_xlabel("Number of news items")


def panel_history_hist(ax, beh, summary):
    data = beh.loc[beh["part"].astype(str) == "fit", "history_len"]
    ax.hist(data.clip(upper=HIST_CLIP), bins=np.arange(0, HIST_CLIP + 4, 3), color="#1f4e79",
            edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.axvline(0.5, color="#c00000", linestyle="--", linewidth=1.2)
    ax.text(0.98, 0.95, f"cold start (no history): {summary['parts']['fit']['cold_start_pct']:.2f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#c00000")
    ax.set_xlabel(f"Items in history (values above {HIST_CLIP} grouped in last bar)")
    ax.set_ylabel("Impressions")


def panel_integrity(ax, summary):
    ax.axis("off")
    parts = [p for p in PART_ORDER if p in summary["parts"]]
    invalid = " / ".join(str(summary["parts"][p]["impressions_without_click"]
                             + summary["parts"][p]["impressions_without_nonclick"]) for p in parts)
    rows = [
        ("News items (train+dev, deduplicated)", f"{summary['news_items']:,}"),
        ("Empty abstract", f"{summary['empty_abstract_pct']:.2f}%"),
        ("Candidate IDs missing from news table", str(summary["missing_candidate_ids"])),
        ("History IDs missing from news table", str(summary["missing_history_ids"])),
        (f"Invalid impressions ({' / '.join(PART_LABEL[p].split()[0] for p in parts)})", invalid),
    ]
    y = 0.92
    for label, value in rows:
        ax.text(0.0, y, label, fontsize=10.5, transform=ax.transAxes, color="#333333")
        ax.text(0.0, y - 0.09, value, fontsize=13, transform=ax.transAxes, fontweight="bold", color="#1f4e79")
        y -= 0.24


PANELS = [
    ("01_part_sizes", "A. Impressions per data part",
     "Impressions in each split. 'Fit' trains the models, 'valid' (the last day of train) tunes alpha and "
     "hyperparameters, 'dev' is scored only once, at the end."),
    ("02_candidates_per_impression", "B. Candidates per impression (fit)",
     "Items ranked in one impression. The distribution is right-skewed, so every metric is computed per "
     "impression and then averaged, not pooled across all rows."),
    ("03_clicks_per_impression", "C. Clicks per impression (fit)",
     "Most impressions have one click; a minority have two or more. This caps the best possible MRR below "
     "1.0 and keeps Precision@K low."),
    ("04_categories", "D. Top 10 news categories",
     "News volume for the ten largest categories. Shows the data is not dominated by one topic, which matters "
     "for the per-category breakdown of results."),
    ("05_history_length", "E. User click-history length (fit)",
     "Past clicks each user brings into an impression. Users with no history (cold start) fall back to the "
     "corpus-average emotion profile."),
    ("06_integrity", "F. Data integrity checks",
     "Confirms the parsed data is clean before modelling: no missing news IDs, and every impression has at "
     "least one click and one non-click."),
]
DRAW = {
    "01_part_sizes": lambda ax, c: panel_part_sizes(ax, c["summary"]),
    "02_candidates_per_impression": lambda ax, c: panel_candidates_hist(ax, c["counts"]),
    "03_clicks_per_impression": lambda ax, c: panel_clicks_bar(ax, c["counts"]),
    "04_categories": lambda ax, c: panel_categories(ax, c["news"]),
    "05_history_length": lambda ax, c: panel_history_hist(ax, c["beh"], c["summary"]),
    "06_integrity": lambda ax, c: panel_integrity(ax, c["summary"]),
}


def save_individual(name, title, caption, ctx, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 4.5) if name == "06_integrity" else (6.5, 5.6))
    DRAW[name](ax, ctx)
    wrapped = textwrap.wrap(caption, 78)
    bottom = 0.14 + 0.035 * len(wrapped)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0.02, bottom, 0.98, 0.94])
    fig.text(0.5, bottom - 0.045, "\n".join(wrapped), ha="center", va="top", fontsize=9, color="#333333")
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def make_overview(ctx, out_path):
    summary = ctx["summary"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (name, title, _) in zip(axes.flat, PANELS):
        DRAW[name](ax, ctx)
        ax.set_title(title)
    fig.suptitle("MIND-small: Exploratory Data Analysis", fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.965, f"{summary['news_items']:,} news items · "
             f"{sum(s['impressions'] for s in summary['parts'].values()):,} impressions · "
             f"{summary['categories']} categories", ha="center", fontsize=10.5, color="#444444")
    fig.text(0.01, 0.01, "Source: MIND-small (Wu et al., 2020), msnews.github.io", fontsize=8, color="#888888")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(processed, out_dir, log=print):
    news, beh, imps, summary = load(processed)
    ctx = {"news": news, "beh": beh, "summary": summary, "counts": per_impression_counts(imps)}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, title, caption in PANELS:
        path = save_individual(name, title, caption, ctx, out_dir)
        log(f"{title}\n  -> {path.name}\n  {caption}\n")
    make_overview(ctx, out_dir / "eda_overview.png")
    stats = {k: v for k, v in summary.items() if k not in ("raw_sha256", "run_info")}
    (out_dir / "eda_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"[done] 6 PNGs + eda_overview.png in {out_dir}")
