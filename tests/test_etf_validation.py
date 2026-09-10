import unittest

import numpy as np
import pandas as pd

from etf_universe.consensus import run_consensus
from etf_universe.validation import backtest_strategies, build_buy_analysis, validation_confidence


class ValidationTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2023-01-02", periods=756)
        self.symbols = ["510001", "510002", "510003", "510004"]
        self.data = {}
        for index, symbol in enumerate(self.symbols):
            # Deterministic paths make the expected trend and risk ordering stable.
            returns = .0012-index*.00025 + .004*np.sin(np.arange(len(dates))/(7+index))
            close = 2*np.cumprod(1+returns)
            self.data[symbol] = pd.DataFrame({
                "date": dates, "symbol": symbol, "open": close*(1-returns/2),
                "high": close*1.005, "low": close*.995, "close": close,
                "volume": 10_000_000*(1+.1*np.sin(np.arange(len(dates))/5)),
                "pct_chg": pd.Series(close).pct_change(fill_method=None),
            })
        benchmark_close = 4000*np.cumprod(1+.0003+.003*np.sin(np.arange(len(dates))/11))
        self.benchmark = pd.DataFrame({"date": dates, "close": benchmark_close})
        self.candidates = pd.DataFrame({
            "symbol": self.symbols, "name": [f"ETF{i}" for i in range(4)],
            "avg_amount": 50_000_000, "aum_yuan": 500_000_000,
            "fund_type": "equity", "index_id": "",
        })

    def test_backtest_metrics_and_buy_analysis(self):
        consensus = run_consensus(self.data, self.benchmark, self.candidates,
                                  strategy_top_n=3, focus_top_n=4)
        backtests = backtest_strategies(self.data, self.benchmark, self.symbols,
                                        backtest_days=252, rebalance_days=5)
        self.assertEqual(set(backtests.strategy), set(__import__("etf_universe.consensus", fromlist=["EVALUATORS"]).EVALUATORS))
        self.assertIn("passed", backtests)
        self.assertTrue((backtests.loc[backtests.status == "completed", "days"] >= 250).all())
        self.assertTrue((backtests.loc[backtests.status == "completed", "max_drawdown"] <= 0).all())
        self.assertTrue((backtests.loc[backtests.status == "completed", "validation_confidence"] == "较高").all())
        analysis, votes = build_buy_analysis(consensus, backtests, 3, min_buy_families=1,
                                             min_buy_mean_sharpe=-10)
        self.assertEqual(set(analysis.symbol), set(self.symbols))
        self.assertTrue(set(votes.strategy).issubset(set(backtests.loc[backtests.passed, "strategy"])))
        self.assertTrue(set(analysis.decision).issubset({"可考虑买入", "继续观察", "暂不买入"}))

    def test_invalid_and_short_backtest(self):
        self.assertEqual(validation_confidence(93), "低")
        self.assertEqual(validation_confidence(180), "中")
        self.assertEqual(validation_confidence(251), "较高")
        with self.assertRaises(ValueError):
            backtest_strategies(self.data, self.benchmark, self.symbols, backtest_days=20)
        short = {symbol: frame.tail(300).reset_index(drop=True) for symbol, frame in self.data.items()}
        with self.assertRaises(ValueError):
            backtest_strategies(short, self.benchmark.tail(300).reset_index(drop=True), self.symbols)

    def test_pre_listing_missing_rows_are_not_strategy_errors(self):
        data = {symbol: frame.copy() for symbol, frame in self.data.items()}
        for column in ("open", "high", "low", "close", "volume", "pct_chg"):
            data[self.symbols[-1]].loc[:99, column] = np.nan
        result = backtest_strategies(data, self.benchmark, self.symbols, backtest_days=252)
        self.assertFalse((result.status == "error").any())

    def test_too_new_symbol_is_excluded_from_validation_votes(self):
        data = {symbol: frame.copy() for symbol, frame in self.data.items()}
        for column in ("open", "high", "low", "close", "volume", "pct_chg"):
            data[self.symbols[-1]].loc[:499, column] = np.nan
        consensus = run_consensus(data, self.benchmark, self.candidates,
                                  strategy_top_n=3, focus_top_n=4)
        backtests = backtest_strategies(data, self.benchmark, self.symbols, backtest_days=252)
        analysis, votes = build_buy_analysis(consensus, backtests, 3)
        self.assertNotIn(self.symbols[-1], set(votes.symbol))
        self.assertFalse(analysis.set_index("symbol").loc[self.symbols[-1], "backtest_eligible"])


if __name__ == "__main__":
    unittest.main()
