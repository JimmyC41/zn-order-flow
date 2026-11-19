"""Tests use constructed examples only; none are ZN observations."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from audit import prepare_events, run_audit, write_results


def events():
    return pd.DataFrame({
        "day": ["2026-05-04", "2026-05-04", "2026-05-05", "2026-05-05"],
        "size": [1000., 1., 1., 1.], "earned": [1., 2., -1., 3.], "back_10": [0., 0., 0., 0.],
        "news_window": [False, True, False, True],
        "seen_1": [1., 1., 1., 1.], "seen_100": [1., 1., 1., 1.], "seen_1000": [1., 1., 1., 1.],
    })


MODEL = {"quoting": {"1": 1.1, "100": 1.1, "1000": 1.1, "10": 999.}}


def run(frame):
    return run_audit(frame, MODEL, data_kind="synthetic", bootstrap=200, seed=7)


def arm(tables, name, scenario="10s_only"):
    table = tables["arms"]
    return table[(table.arm == name) & (table.scenario == scenario)].iloc[0]


class AuditTests(unittest.TestCase):
    def test_unequal_day_volume_estimands_are_distinct(self):
        frame = events().iloc[[0, 2]].copy()
        tables, _ = run(frame)
        row = arm(tables, "all")
        self.assertAlmostEqual(row.pooled_capture_ticks_per_contract, 999 / 1001)
        self.assertAlmostEqual(row.equal_day_mean_ticks_per_contract, 0.)
        self.assertAlmostEqual(row.equal_day_standard_error, 1.)
        # The equal-day SE must not be presented as the pooled estimator's SE.
        self.assertNotIn("pooled_standard_error", tables["arms"].columns)

    def test_zero_retained_day_is_missing_ratio_not_zero(self):
        frame = events()
        frame.loc[frame.day == "2026-05-05", ["seen_1", "seen_100", "seen_1000"]] = 4.
        tables, _ = run(frame)
        row = arm(tables, "news_plus_1ms")
        self.assertEqual(row.zero_volume_days, 1)
        self.assertEqual(row.positive_volume_days, 1)
        self.assertTrue(np.isnan(row.pooled_ci_low))
        daily = tables["daily"]
        zero = daily[(daily.arm == "news_plus_1ms") & (daily.day == "2026-05-05")].iloc[0]
        self.assertEqual(zero.volume, 0)
        self.assertTrue(np.isnan(zero.capture))
        pair = tables["comparisons"]
        pair = pair[(pair.arm == "news_plus_1ms") & (pair.reference == "news_only")].iloc[0]
        self.assertEqual(pair.paired_positive_volume_days, 1)
        self.assertTrue(np.isnan(pair.paired_equal_day_standard_error))
        self.assertTrue(np.isnan(pair.paired_equal_day_ci_low))

    def test_all_rejected_arm_has_undefined_capture_and_pair(self):
        frame = events()
        frame[["seen_1", "seen_100", "seen_1000"]] = 4.
        tables, _ = run(frame)
        row = arm(tables, "news_plus_1ms")
        self.assertEqual(row.retained_events, 0)
        self.assertEqual(row.event_retention, 0)
        self.assertTrue(np.isnan(row.pooled_capture_ticks_per_contract))
        self.assertTrue(np.isnan(row.equal_day_mean_ticks_per_contract))
        pairs = tables["comparisons"]
        self.assertTrue((pairs.loc[pairs.arm == "news_plus_1ms", "paired_positive_volume_days"] == 0).all())

    def test_missing_horizon_does_not_delete_ten_second_only(self):
        frame = events()
        frame["back60"] = [0., 0., np.nan, 0.]
        tables, metadata = run(frame)
        self.assertEqual(arm(tables, "all").retained_events, 4)
        self.assertEqual(arm(tables, "all", "common_horizons").retained_events, 3)
        self.assertEqual(metadata["common_horizon_columns"], ["back_10", "back_60"])
        self.assertFalse(metadata["all_original_horizons_supplied"])

    def test_every_provided_horizon_participates_in_common_sample(self):
        frame = events()
        for horizon in (1, 5, 30, 60):
            frame[f"back_{horizon}"] = 0.
        frame.loc[0, "back_5"] = np.nan
        tables, metadata = run(frame)
        self.assertEqual(arm(tables, "all", "common_horizons").retained_events, 3)
        self.assertTrue(metadata["all_original_horizons_supplied"])

    def test_false_strings_are_parsed_and_ambiguous_values_rejected(self):
        frame = events()
        frame["news_window"] = ["false", "True", "0", "1"]
        tables, _ = run(frame)
        self.assertEqual(arm(tables, "news_only").retained_events, 2)
        for invalid in ("no", "yes", "", None, 2):
            frame.loc[0, "news_window"] = invalid
            with self.assertRaises(ValueError):
                prepare_events(frame)

    def test_test_and_fit_dates_fail_even_when_other_values_missing(self):
        for bad_date in ("2026-09-01", "2026-10-01", "2026-04-30", "2025-12-01"):
            frame = events()
            frame.loc[0, "day"] = bad_date
            frame.loc[0, "back_10"] = np.nan
            with self.assertRaisesRegex(ValueError, "Only validation"):
                run(frame)

    def test_missing_signal_uses_same_population_for_all_arms(self):
        frame = events()
        frame.loc[0, "seen_100"] = np.nan
        tables, _ = run(frame)
        self.assertEqual(arm(tables, "all").retained_events, 3)
        self.assertAlmostEqual(arm(tables, "news_only").event_retention, 1 / 3)

    def test_paired_bootstrap_is_zero_for_identical_rules(self):
        tables, _ = run(events())
        pairs = tables["comparisons"]
        row = pairs[(pairs.arm == "news_plus_1ms") & (pairs.reference == "news_only")].iloc[0]
        for field in ("pooled_ratio_difference", "pooled_difference_ci_low", "pooled_difference_ci_high",
                      "paired_equal_day_gain", "paired_equal_day_ci_low", "paired_equal_day_ci_high"):
            self.assertEqual(row[field], 0.)
        self.assertEqual(row.pooled_valid_bootstrap_draws, 200)

    def test_vwap_comparison_uses_identical_events(self):
        frame = events()
        frame["earned_vwap"] = frame.earned + 2.
        frame.loc[0, "earned_vwap"] = np.nan
        tables, _ = run(frame)
        first = arm(tables, "all", "10s_only_vwap_matched_first")
        vwap = arm(tables, "all", "10s_only_vwap_matched_vwap")
        self.assertEqual(first.retained_events, vwap.retained_events)
        self.assertEqual(first.retained_events, 3)
        self.assertAlmostEqual(vwap.pooled_capture_ticks_per_contract - first.pooled_capture_ticks_per_contract, 2.)

    def test_lag_units_and_missing_are_reported(self):
        frame = events()
        frame["lag10ms"] = [0., 1000., 2000., np.nan]
        tables, _ = run(frame)
        lag = tables["lag_diagnostics"].iloc[0]
        self.assertEqual(lag.overshoot_p50_ms, 1000.)
        self.assertEqual(lag.overshoot_max_ms, 2000.)
        self.assertEqual(lag.lag_missing_events, 1)
        frame.loc[0, "lag10ms"] = -1
        with self.assertRaises(ValueError):
            run(frame)

    def test_frozen_thresholds_are_required_and_obsolete_key_ignored(self):
        _, metadata = run(events())
        self.assertEqual(metadata["ignored_model_quoting_keys"], ["10"])
        with self.assertRaisesRegex(ValueError, "1000 ms"):
            run_audit(events(), {"quoting": {"1": 1, "100": 1}}, data_kind="synthetic")

    def test_output_never_overwrites_and_synthetic_label_persists(self):
        tables, metadata = run(events())
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result"
            write_results(output, tables, metadata)
            self.assertIn("SYNTHETIC", (output / "READ_ME_FIRST.txt").read_text())
            self.assertTrue((tables["arms"].data_kind == "synthetic").all())
            with self.assertRaises(FileExistsError):
                write_results(output, tables, metadata)

    def test_exporter_cluster_id_alias_enables_duplicate_checks(self):
        frame = events()
        frame["cluster_id"] = [0, 1, 0, 1]  # IDs may restart each session.
        prepared = prepare_events(frame)
        self.assertIn("event_id", prepared)
        frame.loc[1, "cluster_id"] = 0
        with self.assertRaisesRegex(ValueError, "unique within each day"):
            prepare_events(frame)


if __name__ == "__main__":
    unittest.main()
