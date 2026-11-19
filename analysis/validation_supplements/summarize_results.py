#!/usr/bin/env python3
"""Review complete 80-session scores; no model fitting or new forecasting."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import os
import tempfile

from export_validation_diagnostic import validate_export_manifest, verify_export_input


def validate_score_source(results, manifest, scores):
    validate_export_manifest(manifest)
    provenance = json.loads((results / "provenance.json").read_text())
    if provenance.get("input_label") != "validation_export":
        raise ValueError("Complete validation summaries require empirical validation scores")
    if provenance.get("validation_range") != ["2026-05-01", "2026-08-31"]:
        raise ValueError("Incorrect forecast validation dates")
    if (sorted(provenance.get("horizons_s", [])) != [1, 10, 600]
            or sorted(provenance.get("delays_s", [])) != [0.001, 0.1, 1.0]):
        raise ValueError("Unexpected forecast horizons or delays")

    # The original release recorded its scores checksum separately. Subsequent
    # runs record the checksum and order count beside the input hashes.
    original_scores = "0b2dfc4e6a3ca36573f4895b00b07b6f090de06b48b1096ca2554f29827b0e67"
    checksum = hashlib.sha256(scores.read_bytes()).hexdigest()
    if checksum != provenance.get("daily_scores_sha256", original_scores):
        raise ValueError("Daily scores differ from the recorded forecast output")
    if "n_orders" in provenance and provenance["n_orders"] != manifest["n_orders"]:
        raise ValueError("Forecast order count differs from the validation export")
    hashes = provenance.get("input_sha256", {})
    verify_export_input(manifest, "validation_orders.csv.gz", hashes.get("orders"))
    for role, exported, original in (
        ("model", "frozen_model.json", "output/model.json"),
        ("config", "frozen_config.json", "analysis/config.json"),
        ("news", "news_releases.csv", "data/news_releases.csv"),
    ):
        digest = hashes.get(role)
        # The runner also accepts the original repository JSONs. The exporter
        # reserializes those JSONs, so their two file hashes can differ.
        if not digest or digest != manifest["source_hashes"][original]:
            verify_export_input(manifest, exported, digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--export-manifest", type=Path, required=True)
    args = parser.parse_args()
    source = args.results / "daily_scores.csv"
    manifest = json.loads(args.export_manifest.read_text())
    validate_score_source(args.results, manifest, source)
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    days = manifest["days"]
    if len(days) != 80 or len(rows) != 80 * 27:
        raise ValueError("Expected a complete 80-session, nine-grid, three-forecast run")
    lookup = {}
    for row in rows:
        key = (row["day"], int(row["horizon_s"]), float(row["delay_ms"]), row["forecast"])
        if key in lookup:
            raise ValueError("Duplicate score key")
        for name in ("mse", "mae", "mean_forecast_count", "mean_observed_count"):
            row[name] = float(row[name])
            if not math.isfinite(row[name]) or row[name] < 0:
                raise ValueError("Nonfinite or negative score")
        row["n_windows"] = int(row["n_windows"])
        if key[1] not in (1, 10, 600) or row["n_windows"] != 24000 // key[1] - 1:
            raise ValueError("Unexpected window count")
        lookup[key] = row
    methods = ("stationary_profile", "direct_history_integral", "conditional_mean")
    expected = {(day, h, delay, method) for day in days for h in (1, 10, 600)
                for delay in (1., 100., 1000.) for method in methods}
    if set(lookup) != expected:
        raise ValueError("Missing or unexpected sessions, grids or forecasts")
    comparison, paired = [], []
    for h in (1, 10, 600):
        for delay in (1., 100., 1000.):
            scores = {m: [lookup[(d, h, delay, m)] for d in days] for m in methods}
            for i, day in enumerate(days):
                reference = lookup[(day, h, 1.0, "stationary_profile")]
                if any(scores[m][i]["mean_observed_count"] != reference["mean_observed_count"]
                       for m in methods):
                    raise ValueError("Comparators use different observed targets")
                if scores["stationary_profile"][i]["mse"] != reference["mse"]:
                    raise ValueError("The same profile baseline must score identically across delays")
            mean_mse = {m: statistics.mean(r["mse"] for r in scores[m]) for m in methods}
            gains = [a["mse"] - b["mse"] for a, b in
                     zip(scores["stationary_profile"], scores["conditional_mean"])]
            for i, day in enumerate(days):
                paired.append({"day": day, "horizon_s": h, "delay_ms": delay,
                               "stationary_minus_conditional_mse": gains[i]})
            comparison.append({"horizon_s": h, "delay_ms": delay, "days": 80,
                "windows_per_day": scores["conditional_mean"][0]["n_windows"],
                "stationary_profile_mean_mse": mean_mse["stationary_profile"],
                "conditional_mean_mse": mean_mse["conditional_mean"],
                "direct_history_integral_mean_mse": mean_mse["direct_history_integral"],
                "paired_mean_mse_gain_vs_stationary": statistics.mean(gains),
                "mse_reduction_vs_stationary_percent": 100*statistics.mean(gains)/mean_mse["stationary_profile"],
                "conditional_lower_mse_days_vs_stationary": sum(x > 0 for x in gains),
                "paired_gain_sd_across_days": statistics.stdev(gains),
                "mse_reduction_vs_direct_history_percent": 100*(mean_mse["direct_history_integral"]-mean_mse["conditional_mean"])/mean_mse["direct_history_integral"]})
    outputs = {"comparison.csv": comparison, "paired_daily_mse.csv": paired}
    for name in [*outputs, "comparison.tsv", "review.json"]:
        if (args.results/name).exists():
            raise FileExistsError(f"Refusing to replace {name}")
    compact = [{"Horizon (s)": r["horizon_s"], "Delay (ms)": int(r["delay_ms"]),
                "Profile MSE": f'{r["stationary_profile_mean_mse"]:.6g}',
                "Conditional MSE": f'{r["conditional_mean_mse"]:.6g}',
                "MSE reduction (%)": f'{r["mse_reduction_vs_stationary_percent"]:.6g}'}
               for r in comparison]
    audit = {"status": "COMPLETE_80_SESSION_SCORE_REVIEW", "score_rows": len(rows),
             "source_scores_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "export_manifest_sha256": hashlib.sha256(args.export_manifest.read_bytes()).hexdigest(),
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "interpretation": "Equal-day paired MSE comparisons, descriptive validation results. Tiny changes are not evidence of a usable edge. No PnL, threshold tuning, refitting or independent-day significance claim."}
    with tempfile.TemporaryDirectory(dir=args.results) as scratch:
        stage = Path(scratch)
        for name, data in outputs.items():
            with (stage / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        with (stage / "comparison.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(compact[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(compact)
        (stage / "review.json").write_text(json.dumps(audit, indent=2)+"\n")
        published = []
        try:
            for name in [*outputs, "comparison.tsv", "review.json"]:
                path = args.results / name
                os.link(stage / name, path)
                published.append(path)
        except BaseException:
            for path in published:
                path.unlink()
            raise
    print(f"Forecast comparisons saved to {args.results / 'comparison.tsv'}")


if __name__ == "__main__":
    main()
