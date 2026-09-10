import unittest

import numpy as np
import pandas as pd

from etf_universe.data_quality import repair_symbol_history


class PriceQualityTests(unittest.TestCase):
    def test_two_for_one_split_is_back_adjusted(self):
        frame = pd.DataFrame({
            "symbol": ["159560"] * 5,
            "date": pd.date_range("2026-09-04", periods=5),
            "open": [2.20, 2.22, 2.18, 1.10, 1.11],
            "high": [2.22, 2.24, 2.20, 1.12, 1.13],
            "low": [2.18, 2.20, 2.16, 1.08, 1.09],
            "close": [2.20, 2.22, 2.18, 1.10, 1.11],
            "volume": [1e6] * 5,
        })
        repaired, anomalies = repair_symbol_history(frame)
        self.assertEqual(len(anomalies), 1)
        self.assertTrue(anomalies[0]["repaired"])
        self.assertEqual(anomalies[0]["repair_confidence"], "已外部确认")
        self.assertFalse(anomalies[0]["manual_review_required"])
        self.assertTrue(anomalies[0]["confirmation_source"])
        self.assertAlmostEqual(repaired.iloc[2].close, 1.09)
        self.assertAlmostEqual(repaired.iloc[3].close / repaired.iloc[2].close - 1, 1.10 / 1.09 - 1)

    def test_unexplained_extreme_jump_is_isolated(self):
        frame = pd.DataFrame({
            "symbol": ["X"] * 3, "date": pd.date_range("2026-01-01", periods=3),
            "open": [1.0, 1.0, .61], "high": [1.0, 1.0, .62],
            "low": [1.0, 1.0, .60], "close": [1.0, 1.0, .61], "volume": [1, 1, 1],
        })
        repaired, anomalies = repair_symbol_history(frame)
        self.assertFalse(anomalies[0]["repaired"])
        self.assertTrue(anomalies[0]["isolated"])
        self.assertEqual(anomalies[0]["repair_confidence"], "待人工核验")
        self.assertTrue(np.isnan(repaired.iloc[2].close))


if __name__ == "__main__":
    unittest.main()
