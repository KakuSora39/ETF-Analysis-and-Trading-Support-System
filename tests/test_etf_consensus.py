import unittest

import numpy as np
import pandas as pd

from etf_universe.consensus import run_consensus


class ConsensusTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2024-01-02", periods=280)
        rng = np.random.default_rng(7)
        self.symbols = ["510001", "510002", "510003", "510004", "510005"]
        self.data = {}
        for index, symbol in enumerate(self.symbols):
            drift = .0015 - index * .0005
            close = 2 * np.cumprod(1 + rng.normal(drift, .008 + index*.001, len(dates)))
            frame = pd.DataFrame({
                "date": dates, "symbol": symbol, "close": close,
                "open": close*.999, "high": close*1.01, "low": close*.99,
                "volume": 10_000_000 * (1 + rng.uniform(-.1, .1, len(dates))),
            })
            frame["pct_chg"] = frame.close.pct_change(fill_method=None)
            self.data[symbol] = frame
        benchmark_close = 4000*np.cumprod(1+rng.normal(.0004, .006, len(dates)))
        self.benchmark = pd.DataFrame({"date": dates, "close": benchmark_close})
        self.candidates = pd.DataFrame({
            "symbol": self.symbols, "name": [f"ETF{i}" for i in range(5)],
            "avg_amount": np.arange(5)*1_000_000+30_000_000,
            "aum_yuan": 500_000_000, "fund_type": "equity", "index_id": "",
        })

    def test_votes_and_family_normalization(self):
        result = run_consensus(self.data, self.benchmark, self.candidates,
                               strategy_top_n=3, focus_top_n=4)
        self.assertFalse(result.votes.empty)
        self.assertEqual(len(result.focus), 4)
        self.assertTrue(result.ranking.family_votes.is_monotonic_decreasing)
        self.assertTrue((result.ranking.family_votes <= result.ranking.strategy_votes).all())
        self.assertEqual(set(result.focus.symbol).issubset(self.symbols), True)

    def test_every_known_strategy_is_audited(self):
        strategy_root = __import__("pathlib").Path(__file__).resolve().parents[1] / "strategies"
        result = run_consensus(self.data, self.benchmark, self.candidates, strategy_root=strategy_root)
        status = result.status.set_index("strategy")
        for name in ("momentum_rotation", "pair_trading", "neural_momentum", "asset_allocation"):
            self.assertIn(name, status.index)
        self.assertEqual(status.loc["pair_trading", "status"], "not_applicable")
        self.assertEqual(status.loc["momentum_rotation", "status"], "voted")
        expected = {p.name for p in strategy_root.iterdir()
                    if p.is_dir() and not p.name.startswith("__") and any(p.glob("*.py"))}
        self.assertEqual(set(status.index), expected)

    def test_invalid_consensus_limits(self):
        with self.assertRaises(ValueError):
            run_consensus(self.data, self.benchmark, self.candidates, strategy_top_n=0)


if __name__ == "__main__":
    unittest.main()
