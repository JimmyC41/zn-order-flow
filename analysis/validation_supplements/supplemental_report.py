#!/usr/bin/env python3
"""Reconcile frozen validation quoting proxies and describe bounded sensitivities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import audit
from export_validation_diagnostic import new_output_directory, validate_export_manifest, verify_export_input


ARMS = ("all", "news_only", "news_plus_1ms", "news_plus_100ms", "news_plus_1000ms")
SIZES = ("size <= 1", "1 < size < 100", "size >= 100")


def reconcile(results, official_path):
    arms = pd.read_csv(results / "arms.csv")
    comparisons = pd.read_csv(results / "comparisons.csv")
    official = pd.read_csv(official_path, index_col=0)
    common = arms[arms.scenario == "common_horizons"].set_index("arm")
    cmp = comparisons[(comparisons.scenario == "common_horizons") &
                      (comparisons.reference == "all")].set_index("arm")
    rows = []
    for label, arm in [("always quote", "all")] + [
        (f"news windows and busy periods, {ms} ms reaction", f"news_plus_{ms}ms")
        for ms in audit.REACTIONS
    ]:
        actual = common.loc[arm]
        metrics = {
            "share of orders quoted": actual.event_retention,
            "kept per contract quoted": actual.pooled_capture_ticks_per_contract,
            "kept per contract, standard error across days": actual.equal_day_standard_error,
            "gain over always quote, per contract": 0. if arm == "all" else cmp.loc[arm].paired_equal_day_gain,
            "gain over always quote, standard error across days": 0. if arm == "all" else cmp.loc[arm].paired_equal_day_standard_error,
        }
        for name, value in metrics.items():
            expected = official.loc[label, name]
            difference = value - expected
            rows.append({"arm": arm, "official_row": label, "metric": name,
                         "official_value": expected, "audit_value": value,
                         "audit_minus_official": difference,
                         "matches_absolute_tolerance_1e_10": bool(abs(difference) <= 1e-10)})
    return pd.DataFrame(rows)


def size_sensitivity(input_file, model):
    """Fixed size bands on the shared complete-horizon sample, read in chunks.

    This is a descriptive decomposition, not a newly selected quoting policy.
    Uses the same parser/eligibility as the already-validated original audit.
    """
    thresholds = audit.frozen_thresholds(model)
    totals = {}
    wanted = set(audit.REQUIRED) | {f"back_{h}" for h in audit.ORIGINAL_HORIZONS} | {"earned_vwap"}
    for chunk in pd.read_csv(input_file, usecols=lambda c: c in wanted,
                             dtype={"day": "string", "news_window": "string"},
                             float_precision="round_trip", chunksize=200_000):
        frame = audit.prepare_events(chunk)
        valid = frame[["earned", *(f"back_{h}" for h in audit.ORIGINAL_HORIZONS),
                       *(f"seen_{ms}" for ms in audit.REACTIONS)]].notna().all(axis=1)
        f = frame.loc[valid]
        masks = {"all": pd.Series(True, index=f.index), "news_only": ~f.news_window}
        masks.update({f"news_plus_{ms}ms": ~f.news_window & (f[f"seen_{ms}"] <= thresholds[ms])
                      for ms in audit.REACTIONS})
        bins = {SIZES[0]: f["size"] <= 1, SIZES[1]: (f["size"] > 1) & (f["size"] < 100),
                SIZES[2]: f["size"] >= 100}
        for measurement in ("earned", "earned_vwap"):
            eligible = f.earned_vwap.notna() if measurement == "earned_vwap" else pd.Series(True, index=f.index)
            for arm, on in masks.items():
                for size_name, band in bins.items():
                    selected = f.loc[on & band & eligible]
                    value = selected[measurement] - selected.back_10
                    key = (measurement, arm, size_name)
                    record = totals.setdefault(key, {"events": 0, "volume": 0., "capture_tick_contracts": 0.,
                                                     "capture_sum_over_events": 0.})
                    record["events"] += len(selected)
                    record["volume"] += selected["size"].sum()
                    record["capture_tick_contracts"] += (value * selected["size"]).sum()
                    record["capture_sum_over_events"] += value.sum()
    rows = []
    for (measurement, arm, size_name), record in totals.items():
        n, volume = record["events"], record["volume"]
        rows.append({"scenario": "common_horizons", "earned_column": measurement, "arm": arm,
                     "size_band": size_name, **record,
                     "pooled_capture_ticks_per_contract": record["capture_tick_contracts"] / volume if volume else np.nan,
                     "event_mean_capture_ticks": record["capture_sum_over_events"] / n if n else np.nan})
    table = pd.DataFrame(rows)
    for (measurement, arm), g in table.groupby(["earned_column", "arm"]):
        table.loc[g.index, "event_share_within_arm"] = g.events / g.events.sum()
        table.loc[g.index, "volume_share_within_arm"] = g.volume / g.volume.sum()
    return table


def fmt(value, digits=6):
    return "undefined" if not np.isfinite(value) else f"{value:.{digits}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--official", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--export-manifest", required=True, type=Path)
    args = p.parse_args()
    out = args.results / "interpretation"
    if out.exists():
        raise FileExistsError(f"Refuse to overwrite {out}")
    manifest = json.loads(args.export_manifest.read_text())
    validate_export_manifest(manifest)
    metadata = json.loads((args.results / "metadata.json").read_text())
    if (metadata["data_kind"] != "validation" or metadata["input_events"] != manifest["n_orders"]
            or metadata["input_days"] != manifest["days"] or metadata["common_horizon_columns"] !=
            [f"back_{h}" for h in audit.ORIGINAL_HORIZONS]):
        raise ValueError("Expected all orders, sessions and original horizons from the validation export")
    for filename, path, key in (("validation_events.csv.gz", args.input, "input_sha256"),
                                ("frozen_model.json", args.model, "model_sha256")):
        digest = audit.sha256(path)
        if digest != metadata[key]:
            raise ValueError(f"{filename} differs from the completed audit")
        verify_export_input(manifest, filename, digest)
    # Older audit folders predate table checksums.
    if "table_sha256" in metadata:
        for name in ("arms", "comparisons", "samples", "sensitivity", "lag_diagnostics"):
            filename = name + ".csv"
            if audit.sha256(args.results / filename) != metadata["table_sha256"].get(filename):
                raise ValueError(f"{filename} differs from the completed audit")
    check = reconcile(args.results, args.official)
    matches = int(check.matches_absolute_tolerance_1e_10.sum())
    comparison_note = f"{matches} of {len(check)} official table values match within 1e-10."
    model = json.loads(args.model.read_text())
    sizes = size_sensitivity(args.input, model)
    arms = pd.read_csv(args.results / "arms.csv")
    comparisons = pd.read_csv(args.results / "comparisons.csv")
    samples = pd.read_csv(args.results / "samples.csv").set_index("scenario")
    sensitivity = pd.read_csv(args.results / "sensitivity.csv")
    lags = pd.read_csv(args.results / "lag_diagnostics.csv").set_index("scenario")
    common = arms[arms.scenario == "common_horizons"].set_index("arm")
    news_cmp = comparisons[(comparisons.scenario == "common_horizons") &
                           (comparisons.reference == "news_only")].set_index("arm")
    table = common.copy()
    for col in ("pooled_ratio_difference", "pooled_difference_ci_low", "pooled_difference_ci_high",
                "paired_equal_day_gain", "paired_equal_day_ci_low", "paired_equal_day_ci_high",
                "paired_positive_volume_days"):
        table[f"increment_vs_news_only_{col}"] = news_cmp[col]
    table["mean_retained_event_size"] = table.retained_volume / table.retained_events
    lines = ["Supplementary validation of quoting rules", "",
             "This is a historical capture proxy, not executed P&L. No fitting, threshold tuning, or test-month evaluation.",
             f"Validation export: {manifest['n_orders']:,} orders, {len(manifest['days'])} sessions.",
             comparison_note,
             "The official kept-per-contract level is pooled, while its gain and standard error summarize daily ratios; the reconciliation respects this distinction.", "",
             f"Common 1/5/10/30/60-second sample: {int(samples.loc['common_horizons','events']):,} events; 10-second-only sample: {int(samples.loc['10s_only','events']):,}.",
             "All policies share the same eligible events within each scenario.", ""]
    for arm in ARMS:
        row = common.loc[arm]
        lines.append(f"{arm}: retains {row.event_retention:.2%} of events and {row.volume_retention:.2%} of volume; "
                     f"pooled capture {fmt(row.pooled_capture_ticks_per_contract)} ticks/contract "
                     f"[95% IID-day percentile interval {fmt(row.pooled_ci_low)}, {fmt(row.pooled_ci_high)}]; "
                     f"equal-day mean {fmt(row.equal_day_mean_ticks_per_contract)}.")
        if arm.startswith("news_plus_"):
            c = news_cmp.loc[arm]
            lines.append(f"  Additional to news-only: pooled difference {fmt(c.pooled_ratio_difference)} "
                         f"[{fmt(c.pooled_difference_ci_low)}, {fmt(c.pooled_difference_ci_high)}]; "
                         f"paired equal-day gain {fmt(c.paired_equal_day_gain)} "
                         f"[{fmt(c.paired_equal_day_ci_low)}, {fmt(c.paired_equal_day_ci_high)}], "
                         f"paired days={int(c.paired_positive_volume_days)}.")
    lines += ["", "Measurement and sample checks:"]
    for _, row in sensitivity.iterrows():
        lines.append(f"{row.question}, {row.arm}, {row.candidate_scenario} minus {row.reference_scenario}: "
                     f"pooled change {fmt(row.pooled_change_candidate_minus_reference)}; "
                     f"equal-day change {fmt(row.equal_day_change_candidate_minus_reference)} ticks/contract.")
    lines += ["", "Fixed size-band decomposition of the always-quote common-horizon sample:"]
    for _, row in sizes[(sizes.arm == "all") & (sizes.earned_column == "earned")].iterrows():
        lines.append(f"{row.size_band}: {row.event_share_within_arm:.2%} of events / {row.volume_share_within_arm:.2%} of volume; "
                     f"event-mean capture {fmt(row.event_mean_capture_ticks)}, pooled per-contract capture {fmt(row.pooled_capture_ticks_per_contract)}.")
    lag = lags.loc["common_horizons"]
    lines += ["", f"10-second match overshoot (not elapsed horizon): p50 {lag.overshoot_p50_ms:.3f} ms, "
              f"p95 {lag.overshoot_p95_ms:.3f} ms, p99 {lag.overshoot_p99_ms:.3f} ms, max {lag.overshoot_max_ms:.3f} ms.",
              "Future marks are first trade-sampled pretrade quotes at/after the target, not quote-only updates at the exact wall-clock horizon.", "",
              "Intervals resample days assuming IID days; they are pointwise, conditional on the fixed model/thresholds, and omit parameter/selection uncertainty and multiplicity adjustment. Cross-day dependence can invalidate coverage.",
              "Fixed size bands are descriptive decomposition, not a proposed fitted trading rule. VWAP matched comparisons change only the cluster-price measurement on identical retained events.",
              "Fees, queue position/fill probability, own quoting decisions, inventory and execution costs are absent. Positive proxy capture is not evidence of executable profitability."]
    provenance = {"all_official_cells_reproduced": matches == len(check), "official_cells": len(check),
                  "maximum_absolute_reconciliation_difference": float(check.audit_minus_official.abs().max()),
                  "input_sha256": metadata["input_sha256"], "model_sha256": metadata["model_sha256"],
                  "export_manifest_sha256": audit.sha256(args.export_manifest),
                  "official_table_sha256": audit.sha256(args.official),
                  "report_script_sha256": audit.sha256(__file__),
                  "size_bands_fixed": SIZES,
                  "scope": "May-August validation only; frozen thresholds; historical proxy, not P&L"}
    with new_output_directory(out) as stage:
        check.to_csv(stage / "official_reconciliation.csv", index=False)
        table.to_csv(stage / "news_only_increment_table.csv", index=True)
        sizes.to_csv(stage / "size_sensitivity.csv", index=False)
        (stage / "findings.txt").write_text("\n".join(lines) + "\n")
        (stage / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"{comparison_note} See {out / 'official_reconciliation.csv'} for the comparison.")


if __name__ == "__main__":
    main()
