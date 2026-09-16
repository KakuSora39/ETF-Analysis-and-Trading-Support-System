import tempfile
import unittest
from pathlib import Path

import pandas as pd

from simulation.etf_universe.config import PaperConfig
from simulation.etf_universe.daily import run_day
from simulation.etf_universe.strategy_policy import progressive_entry


class ProgressivePaperTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        days = pd.bdate_range("2026-08-03", periods=65)
        self.t, self.next = [d.date().isoformat() for d in days[-2:]]
        self.frames = {"159001": pd.DataFrame({
            "date": days, "open": 1.0, "high": 1.02, "low": .98,
            "close": 1.0, "volume": 10000,
        })}

    def lane(self, **changes):
        row = {"symbol": "159001", "name": "测试ETF", "action": "继续观察",
               "analysis_rank": 1, "return_3d": .02, "return_5d": .03,
               "distance_ma20": .02, "validated_trend_support": 1,
               "trend_support": 2, "validated_trend_strategies": "趋势动量"}
        row.update(changes)
        return {"trend": pd.DataFrame([row]), "rebound": pd.DataFrame(), "defense": pd.DataFrame()}

    def test_probe_then_add_and_isolation(self):
        lanes = self.lane()
        a = run_day("progressive", self.t, lanes, self.frames, self.root)
        self.assertEqual("probe_entry", a["pending_orders"][0]["stage"])
        self.assertEqual([], a["trade_log"])
        self.assertEqual([], run_day("conservative", self.t, lanes, self.frames, self.root)["pending_orders"])
        confirmed = self.lane(action="可以关注买入")
        b = run_day("progressive", self.next, confirmed, self.frames, self.root)
        self.assertEqual("add_position", b["pending_orders"][0]["stage"])
        self.assertEqual(.2, b["positions"]["159001"]["target_ratio"])
        self.assertEqual(1, len(b["trade_log"]))

    def test_overheated_does_not_add(self):
        cfg = PaperConfig()
        action, ratio, _ = progressive_entry({"opportunity_lane": "trend", "action": "暂不追高",
                                               "distance_ma20": .1}, True, cfg)
        self.assertEqual(("hold", 0.0), (action, ratio))

    def test_rebound_requires_more_than_score(self):
        cfg = PaperConfig()
        row = {"opportunity_lane": "rebound", "stopping_score": 3,
               "validated_rebound_support": 1, "recent_3d_no_new_low": True,
               "ma5_up": True}
        self.assertEqual("probe_entry", progressive_entry(row, False, cfg)[0])
        row["latest_broke_recent_low"] = True
        self.assertEqual("watch", progressive_entry(row, False, cfg)[0])


if __name__ == "__main__":
    unittest.main()
