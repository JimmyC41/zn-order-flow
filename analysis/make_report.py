#!/usr/bin/env python3
"""Create a compact report from diagnostics CSVs, without reading raw market data."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from residual_io import VALIDATION_DIGEST, new_directory, sha256

MODELS = ("constant_rate", "time_of_day", "time_of_day_news", "self_exciting")
NAMES = {"constant_rate": "Constant rate", "time_of_day": "Time of day",
         "time_of_day_news": "Time of day + news", "self_exciting": "Refined Hawkes"}
COLORS = {"constant_rate": "#858c95", "time_of_day": "#bd8c47",
          "time_of_day_news": "#9873a9", "self_exciting": "#176b93"}
FILES = ("pooled_residual_summary.csv", "daily_residual_summary.csv",
         "pooled_qq.csv", "lag_correlation_summary.csv")


def fmt(x, precision=4):
    if pd.isna(x):
        return "undefined"
    if np.isposinf(x):
        return "infinite"
    if np.isneginf(x):
        return "negative infinite"
    return f"{x:.{precision}f}"


def make_report(root):
    root = Path(root)
    with new_directory(root / "report") as out:
        render_report(root, out)


def render_report(root, out):
    manifest = root / "run_manifest.json"
    audit = json.loads(manifest.read_text()) if manifest.exists() else {}
    loaded, hashes = {}, {}
    for name in FILES:
        raw = (root / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if manifest.exists() and hashes[name] != audit.get("output_sha256", {}).get(name):
            raise ValueError(f"Diagnostic table does not match run manifest: {name}")
        loaded[name] = pd.read_csv(BytesIO(raw))
    pooled = loaded[FILES[0]].set_index("model")
    daily, qq, lag = loaded[FILES[1]], loaded[FILES[2]], loaded[FILES[3]]
    if set(pooled.index) != set(MODELS) or not pooled.index.is_unique:
        raise ValueError("Expected exactly the four named arrival models")
    if set(daily.model) != set(MODELS):
        raise ValueError("Daily summaries do not contain the expected four models")
    if not daily.day.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("Daily summaries require YYYY-MM-DD dates")
    if not daily.day.between("2026-05-01", "2026-08-31").all():
        raise ValueError("Report input is restricted to May-August 2026")
    if daily.duplicated(["model", "day"]).any():
        raise ValueError("Duplicate model/day summary")
    days = set(daily.loc[daily.model == MODELS[0], "day"])
    for model in MODELS:
        md = daily[daily.model == model]
        if set(md.day) != days or int(md.n.sum()) != int(pooled.loc[model, "n"]):
            raise ValueError("Daily and pooled samples do not reconcile")
    if pooled.n.nunique() != 1:
        raise ValueError("Model comparisons must use the same gap count")
    if daily.pivot(index="day", columns="model", values="n").nunique(axis=1).ne(1).any():
        raise ValueError("Every model must use the same number of gaps in each session")
    status = audit.get("evidence_type", "unverified")
    if status == "actual_validation":
        # Manifests from the first release recorded the verified Parquet file hash.
        if "ordered_residuals_sha256" in audit:
            verified = audit["ordered_residuals_sha256"] == VALIDATION_DIGEST
        else:
            verified = audit.get("input_sha256") == "36a5834b066ace375431e2e4478a7855731333dd2a8ec2efaf7d254bfad89422"
        if not verified:
            status = "unverified"
    synthetic = status == "synthetic"
    empirical = status == "actual_validation"
    banner = ("SYNTHETIC DEMONSTRATION - NOT ZN RESULTS" if synthetic else
              "VALIDATION DIAGNOSTICS" if empirical else "UNVERIFIED INPUT DIAGNOSTICS")
    rows = []
    for model in MODELS:
        p = pooled.loc[model]
        dl = lag[(lag.model == model) & (lag.lag == 1)]
        if len(dl) != 1:
            raise ValueError(f"Missing or duplicate lag-1 summary for {model}")
        r = dl.iloc[0]
        rows.append({"model": NAMES[model], "gap_count": int(p.n), "mean": p["mean"],
                     "sample_variance": p.variance, "ks_exp1": p.ks_exp1,
                     "ad_exp1": p.ad_statistic_exp1, "ad_per_gap": p.ad_per_residual,
                     "ad_status": p.ad_status, "nonpositive_count": int(p.nonpositive_count),
                     "P_z_gt_3_percent": 100*p.tail_gt_3_fraction,
                     "P_z_gt_5_percent": 100*p.tail_gt_5_fraction,
                     "median_daily_lag1_correlation": r["median"],
                     "p05_daily_lag1_correlation": r.p05,
                     "p95_daily_lag1_correlation": r.p95})
    table = pd.DataFrame(rows)
    table.insert(0, "evidence_status", status)
    table.to_csv(out / "report_table.csv", index=False, float_format="%.10g")
    baseline, refined = pooled.loc["constant_rate"], pooled.loc["self_exciting"]
    rd = daily[daily.model == "self_exciting"].sort_values("day")
    rl = lag[lag.model == "self_exciting"].sort_values("lag")
    r1 = rl[rl.lag == 1].iloc[0]
    n = int(refined.n)
    if "not_requested" in (refined.ad_status, baseline.ad_status):
        ad_sentence = "Anderson-Darling diagnostics were not requested for this run. "
    elif np.isfinite(baseline.ad_statistic_exp1) and np.isfinite(refined.ad_statistic_exp1):
        ad_sentence = (f"Their fixed-Exp(1) AD statistics were {fmt(refined.ad_statistic_exp1, 1)} "
                       f"and {fmt(baseline.ad_statistic_exp1, 1)}, respectively. ")
    else:
        ad_sentence = (f"The full-sample AD statistic is {fmt(refined.ad_statistic_exp1)} for the refined model; "
                       f"{int(refined.nonpositive_count):,} nonpositive residuals are retained and flagged. ")
    introduction = ("Using frozen arrival models, we examined" if empirical else
                    "For this synthetic example, we examined" if synthetic else
                    "For this unverified export, we examined")
    text = (
        f"{banner}\n\n"
        f"{introduction} {n:,} time-rescaled gaps across "
        f"{len(days)} May-August 2026 sessions. The refined model's pooled KS distance "
        f"was {fmt(refined.ks_exp1)}, compared with {fmt(baseline.ks_exp1)} for the "
        f"constant-rate baseline. {ad_sentence}"
        f"The refined residual mean and variance were {fmt(refined['mean'], 3)} and "
        f"{fmt(refined.variance, 3)}, against unit-exponential benchmarks of one. "
        f"Its fractions exceeding 3 and 5 were {100*refined.tail_gt_3_fraction:.2f}% and "
        f"{100*refined.tail_gt_5_fraction:.2f}%, versus 4.98% and 0.67%.\n\n"
        f"Across sessions, refined KS had median {rd.ks_exp1.median():.4f} "
        f"and range {rd.ks_exp1.min():.4f}-{rd.ks_exp1.max():.4f}. "
        f"Median within-session lag-1 correlation was {fmt(r1['median'])}, with a "
        f"fifth-95th percentile range of {fmt(r1.p05)}-{fmt(r1.p95)}. "
        "Marginal calibration and serial dependence are separate requirements: "
        "improved distributional fit does not establish independent residuals. "
        "These supplementary validation diagnostics use no refitting or test-period "
        "evaluation. AD, KS and correlations are descriptive; no generic iid "
        "critical values or formal acceptance claim are applied.\n"
    )
    if not empirical:
        text += "\nThese figures describe the supplied input. The four-model diagnostic sample used in the team report has a separately verified fingerprint.\n"
    (out / "report_insert.txt").write_text(text)
    notes = (
        "Table note. All four models use the same exported gaps. Pooled statistics "
        "weight every gap equally; daily medians and percentile ranges weight sessions "
        "equally. The exponential reference has fixed mean one: no residual-scale "
        "fitting or adjustment is applied. Variance uses ddof=1. A2/n is a descriptive "
        "weighted CDF discrepancy, not a p-value. Infinite AD values are preserved. "
        "Daily percentile ranges are not confidence intervals. The export omits the "
        "first interval from session open and the censored final interval. Correlation "
        "lags count successive exported residual intervals, not milliseconds or seconds. "
        "Any session-kind comparisons are descriptive, not causal news effects or "
        "significance claims; the small number of Fed-decision sessions limits interpretation.\n"
    )
    (out / "table_note.txt").write_text(notes)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    omitted = 0
    for model in MODELS:
        g = qq[qq.model == model].sort_values("probability")
        good = (g.exp1_quantile > 0) & (g.empirical_quantile > 0)
        omitted += int((~good).sum())
        axes[0].loglog(g.loc[good, "exp1_quantile"], g.loc[good, "empirical_quantile"],
                       "o-", ms=3, lw=1.4, color=COLORS[model], label=NAMES[model])
    reference = qq.exp1_quantile[qq.exp1_quantile > 0]
    axes[0].plot([reference.min(), reference.max()], [reference.min(), reference.max()],
                 "--", color="#252b31", lw=1, label="Exp(1) reference")
    axes[0].set(xlabel="Exp(1) reference quantile", ylabel="Empirical rescaled gap",
                title="A  Marginal QQ (log scales)")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    for model in ("constant_rate", "self_exciting"):
        d = daily[daily.model == model].sort_values("day")
        axes[1].plot(pd.to_datetime(d.day), d.ks_exp1, ".-", ms=3, lw=1,
                     label=NAMES[model], color=COLORS[model])
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[1].set_xlim(pd.Timestamp("2026-05-01"), pd.Timestamp("2026-08-31"))
    axes[1].set(xlabel="2026 validation session", ylabel="KS distance to Exp(1)",
                title="B  Day-by-day marginal calibration")
    axes[1].legend(fontsize=8, frameon=False)
    axes[2].fill_between(rl.lag.to_numpy(), rl.p05.to_numpy(), rl.p95.to_numpy(),
                         color=COLORS["self_exciting"], alpha=.18, label="Daily 5th-95th percentiles")
    axes[2].plot(rl.lag, rl["median"], "o-", ms=4, color=COLORS["self_exciting"], label="Daily median")
    axes[2].axhline(0, ls="--", color="#52575d", lw=.8)
    axes[2].set_xscale("log")
    axes[2].set_xticks(rl.lag, [str(int(v)) for v in rl.lag])
    axes[2].set(xlabel="Lag in residuals (within session)", ylabel="Pearson correlation",
                title="C  Refined residual dependence")
    axes[2].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle(f"{banner}: {n:,} gaps, {len(days)} sessions", fontsize=13, y=.99)
    foot = "Fixed Exp(1) reference. Daily percentile bands describe variation across sessions; they are not confidence intervals."
    if omitted:
        foot += f" QQ log plot omits {omitted} nonpositive empirical quantile points; all residuals remain in the tables."
    fig.text(.02, .025, foot, fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, .08, 1, .94))
    figure_name = ("residual_validation_summary.png" if empirical else
                   "SYNTHETIC_residual_validation_summary.png" if synthetic else
                   "UNVERIFIED_residual_validation_summary.png")
    fig.savefig(out / figure_name, dpi=250)
    plt.close(fig)
    caption = (
        f"Figure. {banner.lower()} for {n:,} time-rescaled gaps from {len(days)} sessions. "
        "Panel A compares pooled residual quantiles with the fixed Exp(1) reference "
        "on logarithmic axes. Panel B shows session-level KS distances for the "
        "constant-rate and refined models. Panel C shows the median and fifth-95th "
        "percentile range of within-session residual correlations for the refined "
        "model; no pair crosses a session boundary and these ranges are not "
        "confidence intervals. Lags count residual intervals, not units of clock time. "
        f"Nonpositive plotted quantiles omitted from panel A: {omitted}; "
        "all observations remain in the numerical diagnostics.\n"
    )
    (out / "figure_caption.txt").write_text(caption)
    provenance = {"evidence_status": status, "source_manifest": manifest.name if manifest.exists() else None,
                  "figure_file": figure_name,
                  "input_sha256": hashes,
                  "report_script_sha256": sha256(Path(__file__)),
                  "nonpositive_qq_quantiles_omitted_in_log_plot": omitted}
    (out / "report_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    make_report(args.results_dir)
