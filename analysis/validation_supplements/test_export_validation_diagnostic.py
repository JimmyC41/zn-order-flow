"""Regression tests for frozen source aggregation, including licensed data locally.

Run: python test_export_validation_diagnostic.py
Only aggregate results are printed. The optional real-data test is skipped if the
four official validation files have not been installed under data/databento/ZN_tbbo.
"""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
sys.path.insert(0, str(SRC / 'analysis'))
import flow
from export_validation_diagnostic import (
    EXPORT_FILES, day_frame, document_digest, new_output_directory,
    validate_export_manifest, verify_export_input,
)

CFG = json.loads((SRC / 'analysis/config.json').read_text())
flow.configure(CFG['session_open'], CFG['session_close'], CFG['bucket_seconds'])


class ConstantArrival:
    shape = np.ones(40)
    def base(self, kind):
        return self.shape
    def rate_seen(self, t, kind, delay):
        return np.ones(len(t))


def synthetic(delta, sides=(1, 1)):
    epoch = pd.Timestamp('2026-05-04T12:20:00Z').value
    return pd.DataFrame({
        'day': [pd.Timestamp('2026-05-04').date()] * 2,
        'ts': np.array([epoch, epoch + delta], dtype=np.int64),
        't': [0., delta / 1e9], 'sgn': list(sides), 'size': [2., 3.],
        'px': [110., 110.015625], 'mid': [110., 110.],
        'bid': [109.9921875] * 2, 'ask': [110.0078125] * 2,
        'bsz': [10.] * 2, 'asz': [10.] * 2})


class FrozenGroupingTests(unittest.TestCase):
    def test_near_boundary_matches_frozen_source_not_integer_diff(self):
        tr = synthetic(99999)
        self.assertEqual(tr.ts.diff().iloc[1], 99999)
        self.assertGreater(tr.groupby('day').ts.diff().iloc[1], 100000)
        orders = flow.to_orders(tr, 100000)
        self.assertEqual(len(orders), 2)
        frame = day_frame(tr, orders, ConstantArrival(), None, CFG)
        np.testing.assert_array_equal(frame.ts_ns, orders.ts)
        np.testing.assert_array_equal(frame['size'], orders['size'])
        self.assertEqual(frame['size'].sum(), tr['size'].sum())

    def test_opposite_side_ties_remain_separate(self):
        tr = synthetic(0, (1, -1))
        orders = flow.to_orders(tr, 100000)
        frame = day_frame(tr, orders, ConstantArrival(), None, CFG)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.ts_ns.iloc[0], frame.ts_ns.iloc[1])
        self.assertTrue(frame.back_1.isna().all())

    def test_guard_rejects_incompatible_order_definition(self):
        tr = synthetic(99999)
        # Integer-difference grouping would merge these two events.
        incompatible = flow.to_orders(tr, 200000)
        with self.assertRaisesRegex(ValueError, 'Source aggregation differs'):
            day_frame(tr, incompatible, ConstantArrival(), None, CFG)

    def test_vwap_and_target_use_correct_trade_quote_and_sign(self):
        # The first two fills belong to one buy order. At exactly the
        # 10-second target two sell-side prints share a timestamp.
        tr = synthetic(50_000)
        tail = synthetic(0, (-1, -1))
        tail["ts"] = tr.ts.iloc[0] + 10_000_000_000
        tail["t"] = 10.
        tail["mid"] = [110.03125, 110.0625]
        tr = pd.concat([tr, tail], ignore_index=True)
        orders = flow.to_orders(tr, 100_000)
        out = day_frame(tr, orders, ConstantArrival(), None, CFG)
        first = out.iloc[0]
        self.assertEqual(first.earned, 0)
        self.assertAlmostEqual(first.earned_vwap, 3/5)
        self.assertEqual(first.back_10, 2)
        self.assertEqual(first.lag_10_ms, 0)
        self.assertEqual(first.match_10_ns, tr.ts.iloc[2])
        self.assertAlmostEqual(first.lag_5_ms, 5000)
        self.assertTrue(pd.isna(out.iloc[1].back_10))

    def test_missing_fill_price_invalidates_only_vwap(self):
        tr = synthetic(50_000)
        tr.loc[1, "px"] = np.nan
        result = day_frame(tr, flow.to_orders(tr, 100_000), ConstantArrival(), None, CFG)
        self.assertEqual(result.iloc[0].earned, 0)
        self.assertTrue(pd.isna(result.iloc[0].earned_vwap))

    def test_news_window_boundaries(self):
        times = np.array([599., 600., 1799., 1800., 20399., 20400.])
        tr = pd.concat([synthetic(0).iloc[:1]] * len(times), ignore_index=True)
        tr["t"] = times
        tr["ts"] = tr.ts.iloc[0] + (times * 1e9).astype(np.int64)
        orders = flow.to_orders(tr, 100_000)
        for kind, expected in (("data release", [0, 1, 1, 0, 0, 0]),
                               ("fed decision", [0, 0, 0, 0, 0, 1]),
                               (None, [0] * 6)):
            out = day_frame(tr, orders, ConstantArrival(), kind, CFG)
            np.testing.assert_array_equal(out.news_window, expected)

    def test_publication_rolls_back_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "export"
            with self.assertRaises(RuntimeError):
                with new_output_directory(dest) as work:
                    (work / "incomplete").write_text("partial")
                    raise RuntimeError("interrupted")
            self.assertFalse(dest.exists())
            with new_output_directory(dest) as work:
                (work / "complete").write_text("done")
            self.assertTrue((dest / "complete").is_file())

    def test_manifest_rejects_wrong_calendar_and_mixed_files(self):
        # Substitute a short calendar fingerprint so this fixture needs no
        # licensed files or machine-specific paths.
        days = pd.bdate_range("2026-05-01", periods=80).strftime("%Y-%m-%d").tolist()
        import hashlib
        calendar = hashlib.sha256("\n".join(days).encode()).hexdigest()
        sources = {"output/model.json": "a" * 64}
        raw = {"data.parquet": "b" * 64}
        doc = {"schema_version": 2, "status": "REAL_VALIDATION_EXPORT", "days": days,
               "n_orders": 2_414_025, "n_residuals": 2_413_945, "source_hashes": sources, "raw_sha256": raw,
               "output_sha256": {f: "a" * 64 for f in EXPORT_FILES}}
        with patch("export_validation_diagnostic.CALENDAR_DIGEST", calendar), patch(
                "export_validation_diagnostic.SOURCE_DIGEST", document_digest(sources)), patch(
                "export_validation_diagnostic.RAW_FILES_DIGEST", document_digest(raw)):
            verify_export_input(doc, "validation_events.csv.gz", "a" * 64)
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify_export_input(doc, "validation_events.csv.gz", "b" * 64)
            with self.assertRaisesRegex(ValueError, "Parquet"):
                validate_export_manifest({**doc, "raw_sha256": {"data.parquet": "c" * 64}})
            with self.assertRaisesRegex(ValueError, "calendar"):
                validate_export_manifest({**doc, "days": list(reversed(days))})
        with self.assertRaisesRegex(ValueError, "Unrecognized legacy"):
            validate_export_manifest({"status": "REAL_VALIDATION_EXPORT"})

    @unittest.skipUnless((SRC / 'data/databento/ZN_tbbo/20260501_20260601.parquet').is_file(),
                         'licensed validation files unavailable')
    def test_real_may_six_source_alignment(self):
        paths = [str(p) for p in (SRC / 'data/databento/ZN_tbbo').glob('*.parquet')]
        skip = flow.skipped_days(paths)
        tr, _ = flow.load_trades(paths, '2026-05-06', '2026-05-06', skip)
        tr = tr.reset_index(drop=True)
        orders = flow.to_orders(tr, 100000)
        exact = tr.ts.diff()
        old_boundaries = exact.isna() | (exact > 100000) | (tr.sgn != tr.sgn.shift())
        self.assertEqual(len(orders), 22883)
        self.assertEqual(int(old_boundaries.sum()), 22882)
        frame = day_frame(tr, orders, ConstantArrival(), None, CFG)
        np.testing.assert_array_equal(frame.ts_ns, orders.ts)
        np.testing.assert_array_equal(frame['size'], orders['size'])
        self.assertEqual(len(frame), 22883)


if __name__ == '__main__':
    unittest.main(verbosity=2)
