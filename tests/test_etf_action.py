import unittest

import numpy as np
import pandas as pd

from etf_universe.action import analyze_actions
from etf_universe.presentation import build_markdown_report, chinese_frame


def market(prices):
    dates = pd.date_range("2026-01-01", periods=len(prices), freq="D")
    values = np.asarray(prices, dtype=float)
    return pd.DataFrame({
        "date": dates, "open": values, "high": values * 1.002,
        "low": values * 0.998, "close": values,
        "volume": np.linspace(100, 130, len(values)),
        "amount": np.linspace(1e7, 1.3e7, len(values)),
    })


def analysis(symbol="A"):
    return pd.DataFrame([{
        "analysis_rank": 1, "symbol": symbol, "name": "示例ETF",
        "backtest_eligible": True, "validated_family_votes": 2,
        "validated_strategy_votes": 2, "validated_strategies": "momentum_ma_etf;volume_price",
        "mean_backtest_sharpe": 1.08,
    }])


class ActionTests(unittest.TestCase):
    def test_reasonable_uptrend_can_be_buy_candidate(self):
        prices = np.r_[np.linspace(90, 100, 55), np.linspace(100, 102, 10)]
        result = analyze_actions(analysis(), {"A": market(prices)}, "2026-03-06")
        self.assertEqual(result.iloc[0].action, "可以关注买入")
        self.assertIn("20 日均线高于 60 日均线", result.iloc[0].action_reason)

    def test_fast_rise_is_not_chased(self):
        prices = np.r_[np.linspace(90, 100, 59), np.linspace(100, 115, 6)]
        result = analyze_actions(analysis(), {"A": market(prices)}, "2026-03-06")
        self.assertEqual(result.iloc[0].action, "暂不追高")

    def test_no_validated_support_means_no_participation(self):
        source = analysis()
        source["validated_family_votes"] = 0
        result = analyze_actions(source, {"A": market(np.linspace(90, 105, 65))}, "2026-03-06")
        self.assertEqual(result.iloc[0].action, "暂不参与")

    def test_chinese_report_and_columns(self):
        result = analyze_actions(analysis(), {"A": market(np.linspace(90, 105, 65))}, "2026-03-06")
        translated = chinese_frame(result)
        self.assertIn("行动建议", translated.columns)
        self.assertIn("量价配合", translated.iloc[0]["通过验证且当前支持的策略"])
        report = build_markdown_report(result, "2026-03-06")
        self.assertIn("风险收益表现", report)
        self.assertIn("行动建议", report)


if __name__ == "__main__":
    unittest.main()
