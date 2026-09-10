import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from etf_universe.holdings import HoldingExitConfig, analyze_holdings, load_holdings, write_holdings_template


def history(symbol: str, closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": pd.bdate_range("2026-05-01", periods=len(closes)), "symbol": symbol,
        "open": closes, "high": closes * 1.01, "low": closes * .99, "close": closes,
        "amount": 30_000_000,
    })


class HoldingTests(unittest.TestCase):
    def setUp(self):
        self.trend = history("510001", np.r_[np.linspace(1, 1.25, 78), 1.08, 1.07])
        self.rebound = history("510002", np.r_[np.linspace(1.3, .95, 78), .93, .90])
        self.defense = history("510003", 1 + .01 * np.sin(np.arange(80) / 5))
        self.as_of = self.trend.date.iloc[-1].date().isoformat()
        self.lanes = {
            "trend": pd.DataFrame([{"symbol": "510001", "validated_trend_support": 0, "validated_trend_strategies": "", "trading_risk_level": "高", "trading_risk_reason": "趋势转弱"}]),
            "rebound": pd.DataFrame([{"symbol": "510002", "validated_rebound_support": 0, "validated_rebound_strategies": "", "trading_risk_level": "高", "trading_risk_reason": "反弹转弱"}]),
            "defense": pd.DataFrame([{"symbol": "510003", "validated_defense_support": 1, "validated_defense_strategies": "低波动", "trading_risk_level": "低", "trading_risk_reason": "波动正常"}]),
        }

    def test_three_lanes_use_different_exit_logic(self):
        holdings = pd.DataFrame([
            {"symbol": "510001", "name": "趋势ETF", "buy_date": self.trend.date.iloc[-20], "buy_price": 1.15, "shares": 100, "opportunity_type": "趋势", "entry_strategies": "趋势动量", "entry_reason": "趋势向上"},
            {"symbol": "510002", "name": "反弹ETF", "buy_date": self.rebound.date.iloc[-5], "buy_price": .91, "shares": 100, "opportunity_type": "反弹", "entry_strategies": "超跌回归", "entry_reason": "超跌止跌"},
            {"symbol": "510003", "name": "防守ETF", "buy_date": self.defense.date.iloc[-20], "buy_price": 1.0, "shares": 100, "opportunity_type": "防守", "entry_strategies": "低波动", "entry_reason": "低波动"},
        ])
        tracking, signals = analyze_holdings(
            holdings, {"510001": self.trend, "510002": self.rebound, "510003": self.defense},
            self.lanes, self.as_of,
        )
        states = tracking.set_index("symbol").exit_advice.to_dict()
        self.assertEqual(states["510001"], "趋势破坏退出")
        self.assertEqual(states["510002"], "反弹失败退出")
        self.assertEqual(states["510003"], "继续持有")
        self.assertEqual(set(signals.symbol), {"510001", "510002"})

    def test_template_and_empty_file_handling(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "holdings.json"
            self.assertTrue(load_holdings(target).empty)
            write_holdings_template(target)
            with self.assertRaises(ValueError):
                load_holdings(target)

    def test_rebound_position_uses_time_stop_when_repair_does_not_arrive(self):
        rebound = history("510004", np.r_[np.linspace(1.3, 1.0, 75), .99, .98, .9805, .981, .982])
        as_of = rebound.date.iloc[-1].date().isoformat()
        holdings = pd.DataFrame([{
            "symbol": "510004", "name": "反弹时间测试", "buy_date": rebound.date.iloc[-5],
            "buy_price": .98, "shares": 100, "opportunity_type": "反弹",
            "entry_strategies": "超跌回归", "entry_reason": "超跌后短期修复",
            "entry_reference_low": .90,
        }])
        lanes = {"trend": pd.DataFrame(), "rebound": pd.DataFrame([{
            "symbol": "510004", "validated_rebound_support": 1,
            "validated_rebound_strategies": "超跌回归", "trading_risk_level": "中",
            "trading_risk_reason": "中期仍弱",
        }]), "defense": pd.DataFrame()}
        tracking, signals = analyze_holdings(
            holdings, {"510004": rebound}, lanes, as_of,
            HoldingExitConfig(rebound_max_days=1),
        )
        self.assertEqual(tracking.iloc[0].exit_advice, "时间止损退出")
        self.assertEqual(signals.iloc[0].symbol, "510004")

    def test_defense_position_exits_when_volatility_worsens_and_support_disappears(self):
        volatile = history("510005", np.r_[np.ones(60), np.tile([.8, 1.2], 10)])
        as_of = volatile.date.iloc[-1].date().isoformat()
        holdings = pd.DataFrame([{
            "symbol": "510005", "name": "防守风险测试", "buy_date": volatile.date.iloc[-10],
            "buy_price": 1.2, "shares": 100, "opportunity_type": "defensive",
            "entry_strategies": "低波动", "entry_reason": "低波动持有",
            "entry_volatility": .10,
        }])
        tracking, signals = analyze_holdings(
            holdings, {"510005": volatile},
            {"trend": pd.DataFrame(), "rebound": pd.DataFrame(), "defense": pd.DataFrame()},
            as_of,
        )
        self.assertEqual(tracking.iloc[0].exit_advice, "风险恶化退出")
        self.assertEqual(signals.iloc[0].symbol, "510005")

    def test_documented_manual_format_aliases_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "holdings.json"
            target.write_text('{"holdings":[{"code":"510300","entry_date":"2026-05-01","entry_price":4.0,"quantity":100,"entry_type":"trend","active":true}]}', encoding="utf-8")
            result = load_holdings(target)
            self.assertEqual(result.iloc[0].symbol, "510300")
            self.assertEqual(result.iloc[0].opportunity_type, "趋势")


if __name__ == "__main__":
    unittest.main()
