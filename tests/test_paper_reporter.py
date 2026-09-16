import unittest

from simulation.etf_universe.reporter import compare


class PaperReporterTests(unittest.TestCase):
    def test_comparison(self):
        base = {"mode": "conservative", "strategy_version": "conservative-v1", "initial_capital": 100000,
                "cash": 100000, "positions": {}, "nav_history": [], "trade_log": []}
        advanced = {**base, "mode": "progressive", "strategy_version": "progressive-v1",
                    "cash": 80000, "positions": {"159001": {"shares": 20000, "entry_price": 1,
                                                       "total_cost": 20000}},
                    "nav_history": [{"total_value": 101000, "drawdown": -.01, "daily_return": .01}],
                    "trade_log": [{"side": "buy", "realized_pnl": 0}]}
        result = compare(base, advanced, {"159001": 1.1})
        self.assertAlmostEqual(.02, result["return_gap"])
        self.assertEqual(1, result["trade_count_gap"])
        self.assertAlmostEqual(2000, result["unrealized_pnl_gap"])
        self.assertAlmostEqual(-.01, result["progressive"]["max_drawdown"])


if __name__ == "__main__":
    unittest.main()
