import tempfile
import unittest
from pathlib import Path

import pandas as pd

from simulation.etf_universe.config import PaperConfig
from simulation.etf_universe.history import SignalHistory


class SignalHistoryTests(unittest.TestCase):
    def test_weekly_frozen_snapshot_does_not_change_midweek(self):
        with tempfile.TemporaryDirectory() as root:
            history = SignalHistory(Path(root) / "signals.db")
            history.cache_validation("2026-09-11", pd.DataFrame([{"strategy": "trend", "validation_level": "合格"}]))
            self.assertEqual({"trend": "合格"}, history.frozen_validation("2026-09-14"))
            history.cache_validation("2026-09-14", pd.DataFrame([{"strategy": "trend", "validation_level": "未通过"}]))
            history.cache_validation("2026-09-15", pd.DataFrame([{"strategy": "trend", "validation_level": "强"}]))
            self.assertEqual({"trend": "合格"}, history.frozen_validation("2026-09-16"))
            self.assertEqual({"trend": "强"}, history.frozen_validation("2026-09-21"))

    def test_delayed_label_and_no_future_feature(self):
        with tempfile.TemporaryDirectory() as root:
            history = SignalHistory(Path(root) / "signals.db")
            dates = pd.bdate_range("2026-08-03", periods=12).strftime("%Y-%m-%d").tolist()
            lane = pd.DataFrame([{"symbol": "159001", "name": "测试", "action": "继续观察",
                                  "return_3d": .02, "return_5d": .03, "distance_ma20": .02,
                                  "validated_trend_support": 1, "trend_support": 1,
                                  "future_10d_return": 99}])
            history.record_day(dates[0], {"trend": lane}, "震荡", "", PaperConfig())
            frame = pd.DataFrame({"date": dates, "close": [1] + [1 + .01 * i for i in range(1, 12)]})
            history.fill_forward({"159001": frame}, dates[4])
            with history.connect() as db:
                self.assertIsNone(db.execute("SELECT future_10d_return FROM signal_outcomes").fetchone()[0])
                self.assertNotIn("future_10d_return", [r[1] for r in db.execute("PRAGMA table_info(signal_feature_history)")])
            history.fill_forward({"159001": frame}, dates[-1])
            self.assertEqual(1, history.counts()["missed_opportunity"])

    def test_false_probe_requires_real_fill(self):
        with tempfile.TemporaryDirectory() as root:
            history = SignalHistory(Path(root) / "signals.db")
            days = pd.bdate_range("2026-08-03", periods=12).strftime("%Y-%m-%d").tolist()
            lane = pd.DataFrame([{"symbol": "159001", "action": "继续观察", "return_3d": .02,
                                  "return_5d": .03, "distance_ma20": .02,
                                  "validated_trend_support": 1, "trend_support": 1}])
            history.record_day(days[0], {"trend": lane}, "震荡", "", PaperConfig())
            frame = pd.DataFrame({"date": days, "close": [1.0] + [.9] * 11})
            history.fill_forward({"159001": frame}, days[-1])
            self.assertEqual({}, history.counts())
            with history.connect() as db:
                db.execute("UPDATE signal_outcomes SET future_10d_return=NULL WHERE code='159001'")
            history.mark_executed([{"side": "buy", "signal_date": days[0], "code": "159001", "entry_type": "trend"}])
            history.fill_forward({"159001": frame}, days[-1])
            self.assertEqual(1, history.counts()["false_probe"])


if __name__ == "__main__":
    unittest.main()
