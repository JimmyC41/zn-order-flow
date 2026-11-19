"""Regression: CSV parsing must preserve frozen <= threshold policy decisions."""
from io import StringIO
import unittest

import numpy as np
import pandas as pd

import audit


class CsvPrecisionTests(unittest.TestCase):
    def test_adjacent_floats_and_equality_keep_frozen_decisions(self):
        thresholds = {1: 1.1137661531031582, 100: 1.00001953321778, 1000: 1.0000000000900007}
        for ms, threshold in thresholds.items():
            values = np.array([np.nextafter(threshold, -np.inf), threshold,
                               np.nextafter(threshold, np.inf)])
            f = pd.DataFrame({"day": ["2026-05-04"] * 3, "size": 1., "earned": .5,
                              "back_10": .25, "news_window": False,
                              "seen_1": 0., "seen_100": 0., "seen_1000": 0.})
            f[f"seen_{ms}"] = values
            parsed = audit.read_events(StringIO(f.to_csv(index=False)))
            np.testing.assert_array_equal(parsed[f"seen_{ms}"].to_numpy(), values)
            model = {"quoting": {str(k): v for k, v in thresholds.items()}}
            tables, _ = audit.run_audit(parsed, model, data_kind="synthetic", bootstrap=100)
            arms = tables["arms"].set_index("arm")
            self.assertEqual(arms.loc[f"news_plus_{ms}ms", "retained_events"], 2)


if __name__ == "__main__":
    unittest.main()
