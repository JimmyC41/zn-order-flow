#!/usr/bin/env python3
"""Validation-only diagnostics for the official four-model wide residual export.

Fixed Exp(1) reference; no parameter estimation, p-values, or test-month access.
One model column at a time, with all lag pairs formed inside individual sessions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

from residual_io import VALIDATION_DIGEST, new_directory, sha256

START, END = "2026-05-01", "2026-08-31"
SOURCE_COMMIT = "57a2ad0e558321a3155effbbaeaa98908470302e"
META = ("session_date_ny", "session_kind", "order_in_session")
MODELS = ("constant_rate", "time_of_day", "time_of_day_news", "self_exciting")
COLUMNS = {m: f"rescaled_gap_{m}" for m in MODELS}
KINDS = ("none", "data release", "fed decision")
LAGS = (1, 2, 5, 10, 20, 50)
TAIL_LEVELS = (1., 2., 3., 5., 10.)
QQ_PROBS = (.001, .005, .01, .025, .05, .1, .25, .5, .75, .9, .95, .975, .99, .995, .999)
OFFICIAL_ROWS, OFFICIAL_SESSIONS = 2_413_945, 80


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=columns, float_precision="round_trip", dtype={k: str for k in META[:2]})
    raise ValueError("Input must be the official wide .parquet or .csv export")


def input_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        return pq.read_schema(path).names
    if path.suffix.lower() == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    raise ValueError("Input must be .parquet or .csv")


@dataclass(frozen=True)
class Session:
    day: str
    kind: str
    start: int
    stop: int


def prepare_metadata(frame: pd.DataFrame) -> tuple[list[Session], np.ndarray | None, dict]:
    """Validate full sequence, then provide one shared ordering for every model."""
    if frame.empty:
        raise ValueError("Input contains no residuals")
    if not set(META).issubset(frame.columns):
        raise ValueError(f"Missing metadata columns: {set(META) - set(frame.columns)}")
    days = frame.session_date_ny
    kinds = frame.session_kind
    if days.isna().any() or kinds.isna().any():
        raise ValueError("Missing session date or kind; rows are never silently dropped")
    unique_days = pd.Series(days.unique())
    if not unique_days.map(lambda x: isinstance(x, str)).all():
        raise ValueError("session_date_ny must be ISO date strings, not timestamps")
    if not unique_days.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("session_date_ny must be exactly YYYY-MM-DD")
    pd.to_datetime(unique_days, format="%Y-%m-%d", errors="raise")
    outside = unique_days[~unique_days.between(START, END)].tolist()
    if outside:
        raise ValueError(f"Validation only ({START} to {END}); found {outside[:5]}")
    if not kinds.isin(KINDS).all():
        raise ValueError(f"session_kind must be one of {KINDS}")
    source_order = frame.order_in_session
    if pd.api.types.is_bool_dtype(source_order):
        raise ValueError("order_in_session cannot be boolean")
    order_float = pd.to_numeric(source_order, errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(order_float).all() or (order_float < 2).any()
            or (order_float > 2**53 - 1).any()
            or (order_float != np.floor(order_float)).any()):
        raise ValueError("order_in_session must be finite exact integers >= 2 and <= 2**53-1")
    order = order_float.astype(np.int64)
    categories = sorted(unique_days.tolist())
    codes = pd.Categorical(days, categories=categories).codes
    kind_codes = pd.Categorical(kinds, categories=KINDS).codes
    already_sorted = bool(np.all((codes[1:] > codes[:-1]) |
                                 ((codes[1:] == codes[:-1]) & (order[1:] > order[:-1]))))
    permutation = None if already_sorted else np.lexsort((order, codes))
    if permutation is not None:
        codes, kind_codes, order = codes[permutation], kind_codes[permutation], order[permutation]
    bounds = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(codes)]
    sessions = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        day = categories[codes[start]]
        if order[start] != 2 or np.any(np.diff(order[start:stop]) != 1):
            raise ValueError(f"{day}: order_in_session must start at 2 with no duplicates or gaps")
        if np.any(kind_codes[start:stop] != kind_codes[start]):
            raise ValueError(f"{day}: session_kind changes within a session")
        sessions.append(Session(day, KINDS[kind_codes[start]], int(start), int(stop)))
    return sessions, permutation, {
        "input_rows": len(frame), "sessions": len(sessions),
        "source_rows_reordered": permutation is not None,
        "first_day": sessions[0].day, "last_day": sessions[-1].day,
        "session_kind_counts": {k: sum(s.kind == k for s in sessions) for k in KINDS},
        "official_expected_rows": OFFICIAL_ROWS, "official_expected_sessions": OFFICIAL_SESSIONS,
        "matches_official_expected_counts": len(frame) == OFFICIAL_ROWS and len(sessions) == OFFICIAL_SESSIONS,
        "first_exported_order": 2,
    }


def validate_residuals(values: pd.Series | np.ndarray, expected_rows: int) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        raise ValueError("Residuals cannot be boolean")
    z = np.asarray(values, dtype=float)
    if z.ndim != 1 or len(z) != expected_rows:
        raise ValueError("Residual column length differs from metadata")
    if not np.isfinite(z).all():
        raise ValueError("Non-finite residuals; missing values are never silently dropped")
    return z


def summarize(z: np.ndarray, include_ad: bool = False) -> dict:
    """Exact fixed-CDF KS and optional unadjusted AD, retaining nonpositives."""
    a = np.sort(z)
    n = len(a)
    if not n:
        raise ValueError("Cannot summarize empty residuals")
    cdf = -np.expm1(-np.maximum(a, 0.))
    ranks = np.arange(1, n + 1, dtype=float)
    ks = max(np.max(ranks / n - cdf), np.max(cdf - (ranks - 1.) / n))
    nonpositive = int(np.count_nonzero(a <= 0))
    ad, status = np.nan, "not_requested"
    if include_ad:
        if nonpositive:
            ad, status = np.inf, "infinite_nonpositive_residual"
        else:
            # Fixed Exp(1): log F(z)=log(-expm1(-z)); log survival(z)=-z.
            # No fitted scale, eps clipping, jitter, deletion, or 1-exp(-z).
            terms = (2. * ranks - 1.) / n * (np.log(-np.expm1(-a)) - a[::-1])
            ad = float(-n - np.sum(terms))
            status = "finite" if np.isfinite(ad) else "numeric_overflow"
    result = {
        "n": n, "mean": float(z.mean()),
        "variance": float(z.var(ddof=1)) if n > 1 else np.nan,
        "minimum": float(a[0]), "maximum": float(a[-1]),
        "zero_count": int(np.count_nonzero(a == 0)),
        "negative_count": int(np.count_nonzero(a < 0)),
        "nonpositive_count": nonpositive,
        "ties_count": int(np.count_nonzero(a[1:] == a[:-1])),
        "ks_exp1": float(ks), "ad_statistic_exp1": ad,
        "ad_per_residual": ad / n, "ad_status": status,
    }
    for level in TAIL_LEVELS:
        name = f"tail_gt_{level:g}"
        count = int(np.count_nonzero(z > level))
        ref = float(np.exp(-level))
        result.update({f"{name}_count": count, f"{name}_fraction": count / n,
                       f"{name}_exp1_reference": ref, f"{name}_fraction_minus_reference": count / n - ref})
    return result


def correlation_from_moments(n: int, xx: float, yy: float, xy: float) -> tuple[float, str]:
    if n < 2:
        return np.nan, "fewer_than_two_pairs"
    if xx <= 0. or yy <= 0.:
        return np.nan, "zero_pair_variance"
    value = float(xy / np.sqrt(xx) / np.sqrt(yy))
    if not np.isfinite(value):
        return np.nan, "numeric_overflow"
    return value, "ok"


@dataclass
class PairMoments:
    """Merge centered pair moments without concatenating all session pairs."""
    n: int = 0
    mean_x: float = 0.
    mean_y: float = 0.
    xx: float = 0.
    yy: float = 0.
    xy: float = 0.
    within_xx: float = 0.
    within_yy: float = 0.
    within_xy: float = 0.

    def add(self, x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
        n = len(x)
        if n == 0:
            return np.nan, "fewer_than_two_pairs"
        # Shift before taking means so constant decimal values have exactly zero variance.
        xc, yc = x - x[0], y - y[0]
        offset_x, offset_y = float(xc.mean()), float(yc.mean())
        mx, my = x[0] + offset_x, y[0] + offset_y
        xc -= offset_x
        yc -= offset_y
        xx, yy, xy = float(xc @ xc), float(yc @ yc), float(xc @ yc)
        total = self.n + n
        dx, dy = mx - self.mean_x, my - self.mean_y
        weight = self.n * n / total
        self.xx += xx + dx * dx * weight
        self.yy += yy + dy * dy * weight
        self.xy += xy + dx * dy * weight
        self.mean_x += dx * n / total
        self.mean_y += dy * n / total
        self.n = total
        self.within_xx += xx
        self.within_yy += yy
        self.within_xy += xy
        return correlation_from_moments(n, xx, yy, xy)


def diagnose_model(model: str, z: np.ndarray, sessions: list[Session], include_ad: bool) -> dict[str, list[dict]]:
    daily, lag_daily, lag_pooled = [], [], []
    for s in sessions:
        daily.append({"model": model, "day": s.day, "session_kind": s.kind,
                      **summarize(z[s.start:s.stop], include_ad)})
    pooled = [{"model": model, "days": len(sessions), "weighting": "one_weight_per_residual",
               **summarize(z, include_ad)}]
    qq = [{"model": model, "probability": p, "empirical_quantile": empirical,
           "exp1_quantile": -np.log1p(-p), "empirical_minus_exp1": empirical + np.log1p(-p)}
          for p, empirical in zip(QQ_PROBS, np.quantile(z, QQ_PROBS, method="linear"))]
    for lag in LAGS:
        moments = PairMoments()
        correlations = []
        for s in sessions:
            a = z[s.start:s.stop]
            x, y = a[:-lag], a[lag:]
            r, status = moments.add(x, y)
            lag_daily.append({"model": model, "day": s.day, "session_kind": s.kind,
                              "lag": lag, "pairs": len(x), "correlation": r, "status": status})
            if np.isfinite(r):
                correlations.append(r)
        pooled_r, pooled_status = correlation_from_moments(moments.n, moments.xx, moments.yy, moments.xy)
        centered_r, centered_status = correlation_from_moments(
            moments.n, moments.within_xx, moments.within_yy, moments.within_xy)
        row = {"model": model, "lag": lag, "pairs": moments.n,
               "pooled_pair_correlation": pooled_r, "pooled_status": pooled_status,
               "session_centered_pair_correlation": centered_r, "centered_status": centered_status,
               "days_defined": len(correlations), "days_total": len(sessions),
               "equal_day_mean": float(np.mean(correlations)) if correlations else np.nan}
        for q, label in ((.05, "p05"), (.5, "median"), (.95, "p95")):
            row[label] = float(np.quantile(correlations, q)) if correlations else np.nan
        lag_pooled.append(row)
    return {"daily_residual_summary.csv": daily, "pooled_residual_summary.csv": pooled,
            "pooled_qq.csv": qq, "daily_lag_correlations.csv": lag_daily,
            "lag_correlation_summary.csv": lag_pooled}


def stability_tables(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Equal-day distributions and paired daily model improvements, not CIs."""
    stability = []
    for model, group in daily.groupby("model", sort=False):
        for kind in ("all", *KINDS):
            g = group if kind == "all" else group[group.session_kind == kind]
            if g.empty:
                continue
            row = {"model": model, "session_kind": kind, "days": len(g), "n": int(g.n.sum()),
                   "days_with_nonpositive": int((g.nonpositive_count > 0).sum()),
                   "days_with_infinite_ad": int(np.isinf(g.ad_statistic_exp1).sum())}
            for metric in ("ks_exp1", "mean", "variance", "ad_per_residual"):
                vals = g[metric].to_numpy()
                row[f"{metric}_equal_day_mean"] = float(np.mean(vals))
                # np.quantile interpolation involving infinity gives NaN. Quantiles of
                # AD are left undefined when any value is infinite, explicitly counted.
                for q, label in ((.05, "p05"), (.5, "median"), (.95, "p95")):
                    row[f"{metric}_{label}"] = float(np.quantile(vals, q)) if np.isfinite(vals).all() else np.nan
                row[f"{metric}_minimum"] = float(np.min(vals))
                row[f"{metric}_maximum"] = float(np.max(vals))
            stability.append(row)
    comparisons = []
    base = daily[daily.model == MODELS[0]].set_index("day")
    for model in MODELS[1:]:
        alt = daily[daily.model == model].set_index("day")
        for day in base.index:
            b, a = base.loc[day], alt.loc[day]
            comparisons.append({"model": model, "reference_model": MODELS[0], "day": day,
                                "session_kind": a.session_kind, "n": int(a.n),
                                "ks_reference_minus_model": b.ks_exp1 - a.ks_exp1,
                                "ks_model_lower": bool(a.ks_exp1 < b.ks_exp1)})
    return {"daily_stability_summary.csv": pd.DataFrame(stability),
            "daily_model_comparisons.csv": pd.DataFrame(comparisons)}


def run(input_path: Path, output_dir: Path, *, include_ad: bool = False,
        synthetic: bool = False, model_path: Path | None = None,
        source_audit_path: Path | None = None, require_official_counts: bool = False,
        report: bool = False) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}; choose a new path")
    columns = input_columns(input_path)
    required = set(META) | set(COLUMNS.values())
    if not required.issubset(columns):
        raise ValueError(f"Missing official wide-schema columns: {sorted(required - set(columns))}")
    initial_hash = sha256(input_path)
    meta = read_columns(input_path, list(META))
    sessions, permutation, audit = prepare_metadata(meta)
    del meta
    if require_official_counts and not audit["matches_official_expected_counts"]:
        raise ValueError(f"Expected exactly {OFFICIAL_ROWS} gaps and {OFFICIAL_SESSIONS} sessions")
    content = hashlib.sha256()
    # Contiguous order numbers from 2 are checked above. Session counts therefore
    # identify every row key, even when a CSV or Parquet stores rows out of order.
    for session in sessions:
        content.update(f"{session.day},{session.kind},{session.stop - session.start}\n".encode("ascii"))
    records: dict[str, list[dict]] = {}
    for model, column in COLUMNS.items():
        values = read_columns(input_path, [column])[column]
        z = validate_residuals(values, audit["input_rows"])
        if permutation is not None:
            z = z[permutation]
        content.update((column + "\n").encode("ascii"))
        content.update(np.ascontiguousarray(z, dtype="<f8"))
        for name, rows in diagnose_model(model, z, sessions, include_ad).items():
            records.setdefault(name, []).extend(rows)
        del values, z
    if sha256(input_path) != initial_hash:
        raise RuntimeError("Input changed during analysis; no results written")
    tables = {name: pd.DataFrame(rows) for name, rows in records.items()}
    tables.update(stability_tables(tables["daily_residual_summary.csv"]))
    if synthetic:
        evidence_type = "synthetic"
    elif content.hexdigest() == VALIDATION_DIGEST:
        evidence_type = "actual_validation"
    else:
        evidence_type = "unverified"
    label = {"synthetic": "SYNTHETIC", "actual_validation": "ZN VALIDATION",
             "unverified": "UNVERIFIED INPUT"}[evidence_type]
    audit.update({
        "evidence_status": label, "evidence_type": evidence_type,
        "ordered_residuals_sha256": content.hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": input_path.name, "input_sha256": initial_hash,
        "input_bytes": input_path.stat().st_size, "input_columns": columns,
        "script_sha256": sha256(Path(__file__)),
        "schema_source_commit": SOURCE_COMMIT,
        "schema_commit_is_not_proof_of_input_provenance": True,
        "model_file": model_path.name if model_path else None,
        "model_file_sha256": sha256(model_path) if model_path else None,
        "source_audit_file": source_audit_path.name if source_audit_path else None,
        "source_audit_sha256": sha256(source_audit_path) if source_audit_path else None,
        "validation_start": START, "validation_end": END,
        "models": COLUMNS, "lags": LAGS, "tail_levels": TAIL_LEVELS,
        "qq_probabilities": QQ_PROBS, "anderson_darling_requested": include_ad,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__, "python_version": sys.version,
        "p_values_computed": False, "models_fitted": False, "rows_removed": 0,
        "limitations": [
            "Reference is fixed Exp(mean=1); no parameters or model variants are selected here.",
            "Pooled marginal diagnostics weight residuals equally, so longer sessions carry more weight.",
            "Daily stability summaries give each observed session equal weight; day spreads are not confidence intervals.",
            "Lag pairs never cross sessions; first exported gap ends at order 2 and no origin gap is available.",
            "Pooled pair correlation may reflect between-session levels; session-centered correlation removes each paired-vector mean.",
            "Session-centered correlation is still pair/variance weighted, not an equal-day mean.",
            "Undefined daily correlations are blank, with reasons and defined-day counts; they are not imputed as zero.",
            "AD is unadjusted descriptive A-squared; A-squared/n is normalized descriptive discrepancy, not a p-value.",
            "Positive exact ties retain finite AD when representable; zero/negative values make fixed-Exp1 AD infinite.",
            "No clipping, jittering, truncating or deletion of residuals; strict upper-tail frequencies use z > threshold.",
            "No iid p-values: arrival dependence, parameter estimation and model selection are not calibrated here.",
            "Small marginal distance and zero linear autocorrelation do not establish iid residuals or a correct model.",
            "Calendar labels identify whole news days, not event windows or causal news effects.",
            "Expected row/session totals are a cross-check, not proof that the underlying export is authentic or complete.",
            "AD quantiles are blank if any value is infinite; infinite-day counts and extrema remain explicit.",
        ],
        "method_reference": "https://www.itl.nist.gov/div898/handbook/eda/section3/eda35e.htm",
    })
    with new_directory(output_dir) as stage:
        for filename, table in tables.items():
            table.insert(0, "evidence_status", label)
            table.to_csv(stage / filename, index=False, float_format="%.12g")
        audit["output_sha256"] = {name: sha256(stage / name) for name in tables}
        (stage / "run_manifest.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
        if report:
            from make_report import make_report
            make_report(stage)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Official seven-column wide Parquet/CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--include-ad", action="store_true", help="Add fixed Exp(1) descriptive Anderson-Darling statistics")
    parser.add_argument("--synthetic", action="store_true", help="Label outputs explicitly as synthetic, not empirical")
    parser.add_argument("--model-file", type=Path, help="Hash frozen model.json for provenance; never fit or modify")
    parser.add_argument("--source-audit", type=Path, help="Hash source/export audit for provenance")
    parser.add_argument("--require-official-counts", action="store_true", help="Require official 2,413,945 rows / 80 sessions")
    parser.add_argument("--report", action="store_true", help="Also run make_report.py on generated small tables")
    args = parser.parse_args()
    audit = run(args.input, args.output_dir, include_ad=args.include_ad, synthetic=args.synthetic,
                model_path=args.model_file, source_audit_path=args.source_audit,
                require_official_counts=args.require_official_counts, report=args.report)
    print(f"Residual checks for {audit['input_rows']:,} gaps ({audit['evidence_status']}) "
          f"across {audit['sessions']} sessions saved to {args.output_dir}")


if __name__ == "__main__":
    main()
