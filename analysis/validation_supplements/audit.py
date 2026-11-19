"""Independent, frozen-threshold validation audit. No fitting or test-period use."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from export_validation_diagnostic import new_output_directory, sha256

REACTIONS = (1, 100, 1000)
ORIGINAL_HORIZONS = (1, 5, 10, 30, 60)
REQUIRED = ("day", "size", "earned", "back_10", "news_window", "seen_1", "seen_100", "seen_1000")


def input_column(name):
    return name in {*REQUIRED, "event_id", "cluster_id", "earned_vwap", "lag_10_ms", "lag10ms", "back60"} or bool(re.fullmatch(r"back_\d+", name))


def strict_bool(value):
    """Avoid bool('false') silently treating a news-free event as news."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("true", "1"):
            return True
        if value in ("false", "0"):
            return False
    elif isinstance(value, (int, float, np.number)) and np.isfinite(value):
        if value in (0, 1):
            return bool(value)
    raise ValueError(f"news_window must be true/false or 1/0; received {value!r}")


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        raise ValueError("Input contains no events.")
    if not events.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed.")
    frame = events.loc[:, [c for c in events if input_column(c)]].copy()
    for alias, canonical in (("back60", "back_60"), ("lag10ms", "lag_10_ms"), ("cluster_id", "event_id")):
        if alias in frame:
            if canonical in frame:
                raise ValueError(f"Supply only one of {alias} and {canonical}.")
            frame = frame.rename(columns={alias: canonical})
    missing = set(REQUIRED) - set(frame)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    # Require unambiguous session dates, not timestamps whose timezone conversion might change the day.
    dates = frame.day.astype(str)
    if not dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("day must contain session dates in YYYY-MM-DD format, without times.")
    parsed = pd.to_datetime(dates, format="%Y-%m-%d", errors="raise")
    if not parsed.between("2026-05-01", "2026-08-31").all():
        raise ValueError("Only validation dates 2026-05-01 through 2026-08-31 are permitted; no automatic truncation.")
    frame["day"] = dates
    frame["news_window"] = frame.news_window.map(strict_bool)
    numeric = {"size", "earned", "earned_vwap", "lag_10_ms", *(f"seen_{ms}" for ms in REACTIONS)}
    numeric |= {col for col in frame if re.fullmatch(r"back_\d+", col)}
    for col in sorted(numeric & set(frame)):
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
        if np.isinf(frame[col].to_numpy()).any():
            raise ValueError(f"{col} contains infinity.")
    if frame["size"].isna().any() or (frame["size"] <= 0).any():
        raise ValueError("size must be finite and strictly positive.")
    for ms in REACTIONS:
        if (frame[f"seen_{ms}"] < 0).any():
            raise ValueError(f"seen_{ms} must be nonnegative when observed.")
    if "lag_10_ms" in frame and (frame.lag_10_ms < 0).any():
        raise ValueError("lag_10_ms is nonnegative MATCH MINUS TARGET overshoot in milliseconds.")
    if "event_id" in frame and (frame.event_id.isna().any() or frame.duplicated(["day", "event_id"]).any()):
        raise ValueError("If supplied, event_id must be nonmissing and unique within each day.")
    return frame.reset_index(drop=True)


def frozen_thresholds(model: dict) -> dict[int, float]:
    thresholds = {}
    for ms in REACTIONS:
        try:
            value = model["quoting"][str(ms)]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Frozen model lacks quoting threshold {ms} ms.") from exc
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid quoting threshold for {ms} ms.") from exc
        if isinstance(value, bool) or not np.isfinite(numeric) or numeric < 0:
            raise ValueError(f"Invalid quoting threshold for {ms} ms.")
        thresholds[ms] = numeric
    return thresholds


def ratio(num, den):
    return np.divide(num, den, out=np.full(np.shape(num), np.nan, dtype=float), where=np.asarray(den) > 0)


def ci(draws):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return np.nan, np.nan, 0
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi), len(draws)


def mean_se(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) >= 2 else np.nan


def daily_table(frame, on, days, earned_col):
    selected = frame.loc[on, ["day", "size", earned_col, "back_10"]].copy()
    selected["value"] = selected["size"] * (selected[earned_col] - selected.back_10)
    daily = selected.groupby("day").agg(events=("size", "size"), volume=("size", "sum"), value=("value", "sum"))
    daily = daily.reindex(days, fill_value=0)
    daily["capture"] = ratio(daily.value.values, daily.volume.values)
    return daily


def summarize_arm(daily, total_events, total_volume, draws, rng):
    num, den = daily.value.to_numpy(), daily.volume.to_numpy()
    observed = daily.capture.dropna().to_numpy()
    pool_boot, mean_boot = [], []
    if len(daily) >= 2 and len(observed) >= 2:
        idx = rng.integers(0, len(daily), size=(draws, len(daily)))
        pool_boot = ratio(num[idx].sum(axis=1), den[idx].sum(axis=1))
    if len(observed) >= 2:
        idx = rng.integers(0, len(observed), size=(draws, len(observed)))
        mean_boot = observed[idx].mean(axis=1)
    plo, phi, pn = ci(pool_boot)
    dlo, dhi, dn = ci(mean_boot)
    return {
        "retained_events": int(daily.events.sum()), "event_retention": daily.events.sum() / total_events,
        "retained_volume": den.sum(), "volume_retention": den.sum() / total_volume,
        "eligible_days": len(daily), "positive_volume_days": len(observed), "zero_volume_days": int((den == 0).sum()),
        "pooled_capture_ticks_per_contract": float(ratio(num.sum(), den.sum())),
        "pooled_ci_low": plo, "pooled_ci_high": phi, "pooled_valid_bootstrap_draws": pn,
        "equal_day_mean_ticks_per_contract": observed.mean() if len(observed) else np.nan,
        "equal_day_standard_error": mean_se(observed),
        "equal_day_ci_low": dlo, "equal_day_ci_high": dhi, "equal_day_valid_bootstrap_draws": dn,
        "total_capture_proxy_tick_contracts": num.sum(),
    }


def compare_arms(arm, reference, draws, rng):
    # Same eligible-day resample for both ratios; keep zero-volume days in the pooled estimand.
    n = len(arm)
    av, aq = arm.value.to_numpy(), arm.volume.to_numpy()
    bv, bq = reference.value.to_numpy(), reference.volume.to_numpy()
    pool_boot = []
    if n >= 2 and (aq > 0).sum() >= 2 and (bq > 0).sum() >= 2:
        idx = rng.integers(0, n, size=(draws, n))
        pool_boot = ratio(av[idx].sum(axis=1), aq[idx].sum(axis=1)) - ratio(bv[idx].sum(axis=1), bq[idx].sum(axis=1))
    gains = (arm.capture - reference.capture).dropna().to_numpy()
    mean_boot = []
    if len(gains) >= 2:
        idx = rng.integers(0, len(gains), size=(draws, len(gains)))
        mean_boot = gains[idx].mean(axis=1)
    plo, phi, pn = ci(pool_boot)
    dlo, dhi, dn = ci(mean_boot)
    return {
        "eligible_days": n, "paired_positive_volume_days": len(gains), "unpaired_days": n - len(gains),
        "pooled_ratio_difference": float(ratio(av.sum(), aq.sum()) - ratio(bv.sum(), bq.sum())),
        "pooled_difference_ci_low": plo, "pooled_difference_ci_high": phi, "pooled_valid_bootstrap_draws": pn,
        "paired_equal_day_gain": gains.mean() if len(gains) else np.nan,
        "paired_equal_day_standard_error": mean_se(gains),
        "paired_equal_day_ci_low": dlo, "paired_equal_day_ci_high": dhi, "paired_equal_day_valid_bootstrap_draws": dn,
    }


def run_audit(events, model, *, data_kind, bootstrap=2000, seed=3666):
    if data_kind not in ("synthetic", "validation"):
        raise ValueError("data_kind must explicitly be synthetic or validation.")
    if not isinstance(bootstrap, int) or bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap draws; 2000 or more is recommended.")
    frame = prepare_events(events)
    thresholds = frozen_thresholds(model)
    required_valid = ["earned", "back_10", *(f"seen_{ms}" for ms in REACTIONS)]
    valid = frame[required_valid].notna().all(axis=1)
    if not valid.any():
        raise ValueError("No common eligible 10-second events remain.")
    horizons = sorted((c for c in frame if re.fullmatch(r"back_\d+", c)), key=lambda c: int(c.split('_')[1]))
    selections = [("10s_only", valid)]
    if len(horizons) > 1:
        selections.append(("common_horizons", valid & frame[horizons].notna().all(axis=1)))
    scenarios = [(name, mask, "earned") for name, mask in selections]
    if "earned_vwap" in frame:
        for name, mask in selections:
            matched = mask & frame.earned_vwap.notna()
            scenarios.extend([(name + "_vwap_matched_first", matched, "earned"), (name + "_vwap_matched_vwap", matched, "earned_vwap")])
    summaries, comparisons, daily_rows, samples, lags = [], [], [], [], []
    rng = np.random.default_rng(seed)
    for scenario, keep, earned_col in scenarios:
        current = frame.loc[keep]
        samples.append({"data_kind": data_kind, "scenario": scenario, "earned_column": earned_col,
                        "events": len(current), "volume": current["size"].sum(), "days": current.day.nunique(),
                        "events_excluded_from_input": len(frame) - len(current), "event_share_of_input": len(current) / len(frame),
                        "days_excluded_from_input": frame.day.nunique() - current.day.nunique()})
        if current.empty:
            continue
        days = sorted(current.day.unique())
        masks = {"all": pd.Series(True, index=current.index), "news_only": ~current.news_window}
        masks.update({f"news_plus_{ms}ms": ~current.news_window & (current[f"seen_{ms}"] <= thresholds[ms]) for ms in REACTIONS})
        daily = {name: daily_table(current, on, days, earned_col) for name, on in masks.items()}
        for arm_name, table in daily.items():
            row = {"data_kind": data_kind, "scenario": scenario, "arm": arm_name,
                   **summarize_arm(table, len(current), current["size"].sum(), bootstrap, rng)}
            summaries.append(row)
            daily_rows.append(table.reset_index().assign(data_kind=data_kind, scenario=scenario, arm=arm_name))
            for reference in ("all", "news_only"):
                if arm_name != reference:
                    comparisons.append({"data_kind": data_kind, "scenario": scenario, "arm": arm_name, "reference": reference,
                                        **compare_arms(table, daily[reference], bootstrap, rng)})
        if "lag_10_ms" in current:
            values = current.lag_10_ms.dropna()
            lags.append({"data_kind": data_kind, "scenario": scenario, "eligible_events": len(current),
                         "lag_observed_events": len(values), "lag_missing_events": int(current.lag_10_ms.isna().sum()),
                         **{f"overshoot_{key}_ms": float(values.quantile(q)) if len(values) else np.nan
                            for key, q in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1))}})
    tables = {"arms": pd.DataFrame(summaries), "comparisons": pd.DataFrame(comparisons),
              "daily": pd.concat(daily_rows, ignore_index=True), "samples": pd.DataFrame(samples),
              "lag_diagnostics": pd.DataFrame(lags)}
    sensitivity = []
    scenario_pairs = [("common_horizons", "10s_only", "horizon_sample")]
    scenario_pairs += [(name + "_vwap_matched_vwap", name + "_vwap_matched_first", "first_price_vs_vwap") for name, _ in selections]
    for candidate, reference, question in scenario_pairs:
        left = tables["arms"].loc[tables["arms"].scenario == candidate].set_index("arm")
        right = tables["arms"].loc[tables["arms"].scenario == reference].set_index("arm")
        for arm_name in left.index.intersection(right.index):
            a, b = left.loc[arm_name], right.loc[arm_name]
            sensitivity.append({"data_kind": data_kind, "question": question, "arm": arm_name,
                                "candidate_scenario": candidate, "reference_scenario": reference,
                                "candidate_events": int(a.retained_events), "reference_events": int(b.retained_events),
                                "pooled_change_candidate_minus_reference": a.pooled_capture_ticks_per_contract - b.pooled_capture_ticks_per_contract,
                                "equal_day_change_candidate_minus_reference": a.equal_day_mean_ticks_per_contract - b.equal_day_mean_ticks_per_contract,
                                "interpretation": "Descriptive measurement/sample sensitivity; no CI and no causal claim."})
    tables["sensitivity"] = pd.DataFrame(sensitivity)
    metadata = {
        "data_kind": data_kind,
        "warning": "SYNTHETIC example data" if data_kind == "synthetic" else "May–August 2026 validation data",
        "input_events": len(frame), "input_days": sorted(frame.day.unique()),
        "scenario_days": {name: sorted(frame.loc[mask, "day"].unique()) for name, mask, _ in scenarios},
        "input_missing_counts": {c: int(frame[c].isna().sum()) for c in required_valid + [c for c in horizons if c != "back_10"]},
        "common_horizon_columns": horizons,
        "all_original_horizons_supplied": set(f"back_{h}" for h in ORIGINAL_HORIZONS).issubset(frame.columns),
        "frozen_thresholds": thresholds, "reaction_ms": list(REACTIONS),
        "ignored_model_quoting_keys": sorted(set(model["quoting"]) - {str(ms) for ms in REACTIONS}),
        "bootstrap_draws_requested": bootstrap, "bootstrap_seed": seed,
        "inference": "The pointwise 95% intervals resample whole days in pairs, assuming independent days and fixed model parameters and thresholds. Pooled comparisons use all eligible days. Daily means require positive volume, and paired daily gains require volume in both policies. Intervals require at least two relevant days. The tables give valid draw counts after excluding zero denominators.",
        "scope": "Within each scenario all policies use the same eligible orders and dates. Retention is relative to eligible orders or contract volume. A missing intensity excludes the order from every policy. The standalone audit accepts partial validation samples.",
        "measurement": "Historical capture per contract is the volume-weighted mean of earned minus back_10, in ticks. It excludes fees, queue fills and inventory costs. The lag_10_ms column measures how far after the target the trade-sampled quote arrived.",
        "sample_caveat": "To reproduce the common-horizon sample, start with all orders before filtering and supply markouts at 1, 5, 10, 30 and 60 seconds. With partial input, common_horizons covers only the supplied orders and horizons.",
    }
    return tables, metadata


def read_events(path):
    # Frozen thresholds can be only a few ulps above one. The default pandas
    # parser can move an exactly-equal CSV value across the <= boundary.
    return pd.read_csv(path, usecols=input_column,
                       dtype={"day": "string", "news_window": "string"},
                       float_precision="round_trip")


def write_results(output_dir, tables, metadata):
    with new_output_directory(output_dir) as destination:
        for name, table in tables.items():
            table.to_csv(destination / f"{name}.csv", index=False)
        metadata = {**metadata, "table_sha256": {p.name: sha256(p) for p in destination.glob("*.csv")}}
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        (destination / "READ_ME_FIRST.txt").write_text(
            metadata["warning"] + "\n\n" + "\n\n".join(
                metadata[k] for k in ("scope", "inference", "measurement", "sample_caveat")
            ) + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV export before global horizon dropna")
    parser.add_argument("--model", required=True, type=Path, help="Existing frozen model.json; quoting keys 1/100/1000 required")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing paths are refused")
    parser.add_argument("--data-kind", required=True, choices=("synthetic", "validation"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3666)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist; original outputs are never overwritten.")
    try:
        input_digest, model_digest = sha256(args.input), sha256(args.model)
        events = read_events(args.input)
        model = json.loads(args.model.read_text())
        tables, metadata = run_audit(events, model, data_kind=args.data_kind, bootstrap=args.bootstrap, seed=args.seed)
        if (input_digest, model_digest) != (sha256(args.input), sha256(args.model)):
            raise ValueError("Inputs changed while computing the quoting statistics")
        metadata.update({"input_file": args.input.name, "input_sha256": input_digest,
                         "model_file": args.model.name, "model_sha256": model_digest,
                         "script_sha256": sha256(__file__),
                         "csv_float_precision": "round_trip",
                         "threshold_precision": "CSV parsing preserves exported float values at the frozen <= threshold boundaries."})
        write_results(args.output_dir, tables, metadata)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(metadata["warning"])
    print(f"Quote comparisons saved to {args.output_dir}")


if __name__ == "__main__":
    main()
