"""Numerical and sequencing checks using synthetic sessions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import residual_pipeline as rp
from make_report import make_report


def wide(days=("2026-05-04", "2026-08-28"), lengths=(4, 4)):
    rows = []
    for j, (day, n) in enumerate(zip(days, lengths)):
        for i in range(n):
            row = {"session_date_ny": day, "session_kind": rp.KINDS[j % 3], "order_in_session": i + 2}
            for k, col in enumerate(rp.COLUMNS.values()):
                row[col] = .3 + .1 * i + .5 * j + .2 * k
            rows.append(row)
    return pd.DataFrame(rows)


class DistributionTests(unittest.TestCase):
    def test_fixed_exp1_ad_analytic_ties_and_scale(self):
        z = np.array([np.log(2), np.log(2)])
        result = rp.summarize(z, True)
        self.assertAlmostEqual(result["ks_exp1"], .5)
        self.assertAlmostEqual(result["ad_statistic_exp1"], -2 + 4 * np.log(2))
        self.assertAlmostEqual(result["ad_per_residual"], -1 + 2 * np.log(2))
        self.assertEqual(result["ties_count"], 1)
        self.assertEqual(result["ad_status"], "finite")
        self.assertNotAlmostEqual(rp.summarize(2 * z, True)["ad_statistic_exp1"], result["ad_statistic_exp1"])

    def test_ad_matches_direct_formula_and_stable_tails(self):
        z = np.array([5., .2, 1., .75])
        a = np.sort(z)
        n = len(z)
        direct = -n - sum((2 * i - 1) / n * (np.log(1 - np.exp(-a[i - 1]))
                        + np.log(np.exp(-a[n - i]))) for i in range(1, n + 1))
        self.assertAlmostEqual(rp.summarize(z, True)["ad_statistic_exp1"], direct, places=12)
        extreme = rp.summarize(np.array([1e-300, 1000.]), True)
        self.assertTrue(np.isfinite(extreme["ad_statistic_exp1"]))
        self.assertEqual(extreme["ad_status"], "finite")

    def test_nonpositive_never_dropped_or_clipped(self):
        result = rp.summarize(np.array([-1., 0., .5, 3., 5.]), True)
        self.assertEqual(result["n"], 5)
        self.assertEqual(result["negative_count"], 1)
        self.assertEqual(result["zero_count"], 1)
        self.assertEqual(result["nonpositive_count"], 2)
        self.assertEqual(result["minimum"], -1)
        self.assertTrue(np.isinf(result["ad_statistic_exp1"]))
        self.assertEqual(result["ad_status"], "infinite_nonpositive_residual")
        self.assertEqual(result["tail_gt_3_count"], 1)  # strict greater-than, not >=
        self.assertEqual(result["tail_gt_5_count"], 0)

    def test_ad_optional_and_sample_variance(self):
        result = rp.summarize(np.array([1., 3.]))
        self.assertTrue(np.isnan(result["ad_statistic_exp1"]))
        self.assertEqual(result["ad_status"], "not_requested")
        self.assertEqual(result["variance"], 2.)
        self.assertTrue(np.isnan(rp.summarize(np.array([1.]))["variance"]))


class SchemaTests(unittest.TestCase):
    def test_dates_reject_test_fit_and_bad_format(self):
        for day in ("2026-09-01", "2026-10-31", "2026-04-30", "2026-5-01", "2026-05-32", None):
            frame = wide(days=(day,), lengths=(3,))
            with self.subTest(day=day), self.assertRaises((ValueError, TypeError)):
                rp.prepare_metadata(frame)

    def test_sort_shared_permutation_preserves_order(self):
        frame = wide().iloc[[6, 5, 7, 4, 2, 0, 3, 1]].reset_index(drop=True)
        sessions, perm, audit = rp.prepare_metadata(frame)
        self.assertTrue(audit["source_rows_reordered"])
        self.assertEqual([(s.day, s.start, s.stop) for s in sessions],
                         [("2026-05-04", 0, 4), ("2026-08-28", 4, 8)])
        np.testing.assert_array_equal(frame.iloc[perm].order_in_session, [2, 3, 4, 5, 2, 3, 4, 5])
        np.testing.assert_allclose(frame[rp.COLUMNS["self_exciting"]].to_numpy()[perm],
                                   wide()[rp.COLUMNS["self_exciting"]])

    def test_incomplete_duplicate_fractional_or_kind_mismatch_rejected(self):
        cases = []
        for values in ([1, 2, 3, 4], [2, 3, 5, 6], [2, 3, 3, 4], [2, 3.5, 4, 5], [2, 3, np.inf, 5]):
            f = wide(days=("2026-05-04",), lengths=(4,))
            f["order_in_session"] = values
            cases.append(f)
        f = wide(); f.loc[1, "session_kind"] = "fed decision"; cases.append(f)
        f = wide(); f.loc[1, "session_kind"] = "false"; cases.append(f)
        for f in cases:
            with self.subTest(frame=f.to_dict()), self.assertRaises(ValueError):
                rp.prepare_metadata(f)

    def test_nonfinite_residuals_fail_instead_of_filter(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rp.validate_residuals(np.array([1., value]), 2)
        with self.assertRaises(ValueError):
            rp.validate_residuals(np.array([True, False]), 2)


class SessionLagTests(unittest.TestCase):
    def test_no_session_bridge_and_level_vs_centered_correlation(self):
        sessions = [rp.Session("2026-05-04", "none", 0, 4), rp.Session("2026-05-05", "none", 4, 8)]
        z = np.array([1., 2., 1., 0., 101., 102., 101., 100.])
        tables = rp.diagnose_model("self_exciting", z, sessions, True)
        lag1 = tables["lag_correlation_summary.csv"][0]
        self.assertEqual(lag1["pairs"], 6)  # naive concatenation incorrectly gives 7
        self.assertAlmostEqual(lag1["session_centered_pair_correlation"], 0.)
        self.assertAlmostEqual(lag1["equal_day_mean"], 0.)
        self.assertGreater(lag1["pooled_pair_correlation"], .99)
        for row in tables["daily_lag_correlations.csv"][:2]:
            self.assertEqual(row["pairs"], 3)
            self.assertAlmostEqual(row["correlation"], 0.)

    def test_repeated_decimal_has_zero_variance(self):
        m = rp.PairMoments()
        r, status = m.add(np.full(3, 0.1), np.full(3, 0.3))
        self.assertTrue(np.isnan(r))
        self.assertEqual(status, "zero_pair_variance")
        self.assertEqual(m.within_xx, 0)
        self.assertEqual(m.within_yy, 0)

    def test_moment_merge_matches_direct_pair_arrays(self):
        rng = np.random.default_rng(187)
        arrays = [rng.exponential(scale=scale, size=n) for scale, n in ((1., 300), (7., 61), (.2, 504))]
        moments = rp.PairMoments()
        lag = 5
        for a in arrays:
            moments.add(a[:-lag], a[lag:])
        x, y = np.concatenate([a[:-lag] for a in arrays]), np.concatenate([a[lag:] for a in arrays])
        direct = np.corrcoef(x, y)[0, 1]
        pooled, _ = rp.correlation_from_moments(moments.n, moments.xx, moments.yy, moments.xy)
        self.assertAlmostEqual(pooled, direct, places=13)
        x = np.concatenate([a[:-lag] - a[:-lag].mean() for a in arrays])
        y = np.concatenate([a[lag:] - a[lag:].mean() for a in arrays])
        centered, _ = rp.correlation_from_moments(moments.n, moments.within_xx, moments.within_yy, moments.within_xy)
        self.assertAlmostEqual(centered, np.corrcoef(x, y)[0, 1], places=13)

    def test_short_constant_days_undefined_not_zero(self):
        z = np.array([1., 1., 1., 2.])
        sessions = [rp.Session("2026-05-04", "none", 0, 3), rp.Session("2026-05-05", "none", 3, 4)]
        tables = rp.diagnose_model("self_exciting", z, sessions, False)
        first = tables["lag_correlation_summary.csv"][0]
        self.assertEqual(first["days_defined"], 0)
        self.assertEqual(first["days_total"], 2)
        self.assertTrue(np.isnan(first["equal_day_mean"]))
        self.assertEqual(tables["daily_lag_correlations.csv"][0]["status"], "zero_pair_variance")
        self.assertEqual(tables["daily_lag_correlations.csv"][1]["status"], "fewer_than_two_pairs")


class IntegrationTests(unittest.TestCase):
    def test_wide_parquet_all_models_hashes_synthetic_label_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path, output = tmp / "input.parquet", tmp / "results"
            frame = wide(lengths=(3, 9))
            frame.loc[0, rp.COLUMNS["self_exciting"]] = 0.
            frame.to_parquet(path, index=False)
            manifest = rp.run(path, output, include_ad=True, synthetic=True)
            self.assertEqual(manifest["input_sha256"], rp.sha256(path))
            self.assertEqual(manifest["evidence_type"], "synthetic")
            self.assertFalse(manifest["models_fitted"])
            self.assertFalse(manifest["p_values_computed"])
            self.assertEqual(manifest["rows_removed"], 0)
            saved = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(saved["input_rows"], 12)
            pooled = pd.read_csv(output / "pooled_residual_summary.csv")
            self.assertEqual(pooled.model.tolist(), list(rp.MODELS))
            self.assertTrue(pooled.evidence_status.str.startswith("SYNTHETIC").all())
            self.assertTrue(np.isinf(pooled.set_index("model").loc["self_exciting", "ad_statistic_exp1"]))
            for name, digest in manifest["output_sha256"].items():
                self.assertEqual(rp.sha256(output / name), digest)
            stability = pd.read_csv(output / "daily_stability_summary.csv")
            all_days = stability[(stability.model == "constant_rate") & (stability.session_kind == "all")].iloc[0]
            expected_daily_mean = (frame.iloc[:3][rp.COLUMNS["constant_rate"]].mean()
                                   + frame.iloc[3:][rp.COLUMNS["constant_rate"]].mean()) / 2
            self.assertAlmostEqual(all_days.mean_equal_day_mean, expected_daily_mean)
            self.assertNotAlmostEqual(all_days.mean_equal_day_mean, pooled.iloc[0]["mean"])
            with self.assertRaises(FileExistsError):
                rp.run(path, output, synthetic=True)

    def test_content_fingerprint_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            frame = wide().sample(frac=1, random_state=21)
            frame.loc[1, rp.COLUMNS["constant_rate"]] = np.nextafter(0.1, 1.0)
            frame.to_csv(tmp / "in.csv", index=False)
            frame.sort_values(["session_date_ny", "order_in_session"]).to_parquet(tmp / "in.parquet")
            a = rp.run(tmp / "in.csv", tmp / "a")
            b = rp.run(tmp / "in.parquet", tmp / "b")
            self.assertEqual(a["evidence_type"], "unverified")
            self.assertEqual(a["ordered_residuals_sha256"], b["ordered_residuals_sha256"])
            with patch.object(rp, "VALIDATION_DIGEST", a["ordered_residuals_sha256"]):
                self.assertEqual(rp.run(tmp / "in.csv", tmp / "c")["evidence_type"], "actual_validation")
                frame.loc[1, rp.COLUMNS["constant_rate"]] = 0.1
                frame.to_csv(tmp / "changed.csv", index=False)
                self.assertEqual(rp.run(tmp / "changed.csv", tmp / "d")["evidence_type"], "unverified")

    def test_report_rejects_changed_tables_and_allows_clean_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            frame = wide()
            frame.to_parquet(tmp / "input.parquet", index=False)
            output = tmp / "results"
            rp.run(tmp / "input.parquet", output, synthetic=True)
            table = output / "pooled_qq.csv"
            original = table.read_bytes()
            table.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                make_report(output)
            self.assertFalse((output / "report").exists())
            table.write_bytes(original)
            make_report(output)
            self.assertTrue((output / "report/SYNTHETIC_residual_validation_summary.png").exists())
            meta = json.loads((output / "report/report_provenance.json").read_text())
            self.assertEqual(meta["source_manifest"], "run_manifest.json")

    def test_failed_combined_report_leaves_clean_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wide().to_parquet(tmp / "input.parquet", index=False)
            with patch("make_report.make_report", side_effect=OSError("write failed")), self.assertRaises(OSError):
                rp.run(tmp / "input.parquet", tmp / "results", report=True)
            self.assertFalse((tmp / "results").exists())

    def test_bad_schema_counts_or_nonfinite_leave_no_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path, output = tmp / "input.parquet", tmp / "results"
            frame = wide()
            frame.drop(columns=[rp.COLUMNS["time_of_day"]]).to_parquet(path)
            with self.assertRaises(ValueError):
                rp.run(path, output)
            self.assertFalse(output.exists())
            frame.to_parquet(path)
            with self.assertRaises(ValueError):
                rp.run(path, output, require_official_counts=True)
            self.assertFalse(output.exists())
            frame.loc[4, rp.COLUMNS["self_exciting"]] = np.nan
            frame.to_parquet(path)
            with self.assertRaises(ValueError):
                rp.run(path, output)
            self.assertFalse(output.exists())

    def test_csv_and_parquet_equal_after_reordering(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            frame = wide().sample(frac=1, random_state=18)
            frame.to_csv(tmp / "input.csv", index=False)
            frame.to_parquet(tmp / "input.parquet", index=False)
            for extension in ("csv", "parquet"):
                rp.run(tmp / f"input.{extension}", tmp / extension, include_ad=True, synthetic=True)
            for name in ("pooled_residual_summary.csv", "daily_lag_correlations.csv"):
                pd.testing.assert_frame_equal(pd.read_csv(tmp / "csv" / name), pd.read_csv(tmp / "parquet" / name))


if __name__ == "__main__":
    unittest.main()
