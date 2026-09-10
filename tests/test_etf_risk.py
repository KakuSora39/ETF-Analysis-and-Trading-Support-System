import unittest

import numpy as np
import pandas as pd

from etf_universe.action import analyze_actions, attach_position_metrics
from etf_universe.risk import RiskConfig, assess_etf_risks, check_etf_risk


def prices(amount=30_000_000):
    close = np.r_[np.linspace(90, 100, 55), np.linspace(100, 102, 10)]
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=65), "open": close,
        "high": close * 1.002, "low": close * 0.998, "close": close,
        "volume": 1000, "amount": amount,
    })


def candidate(fund_type="equity"):
    return pd.DataFrame([{
        "analysis_rank": 1, "symbol": "A", "name": "示例ETF", "fund_type": fund_type,
        "backtest_eligible": True, "validated_family_votes": 2,
        "validated_strategy_votes": 2, "validated_strategies": "momentum_ma_etf;volume_price",
        "mean_backtest_sharpe": 1.0,
    }])


class RiskTests(unittest.TestCase):
    def test_equity_liquidity_and_unavailable_premium_are_explicit(self):
        metrics = attach_position_metrics(candidate(), {"A": prices()}, "2026-03-06")
        result = assess_etf_risks(metrics)
        self.assertEqual(result.iloc[0].etf_type, "普通股票ETF")
        self.assertEqual(result.iloc[0].liquidity_status, "流动性正常")
        self.assertEqual(result.iloc[0].premium_status, "数据不可用")
        self.assertFalse(result.iloc[0].premium_confirmation_needed)
        self.assertEqual(result.iloc[0].etf_risk_level, "中")
        structured = check_etf_risk(result.iloc[0])
        self.assertEqual(structured["risk_level"], "medium")
        self.assertEqual(set(structured), {
            "risk_level", "risk_tags", "warnings", "details", "etf_type", "special_risk_note",
            "liquidity_status", "liquidity_reason", "price_risk_status", "price_risk_reason",
            "premium_rate", "premium_status", "premium_reason", "announcement_status",
            "announcement_warnings", "premium_confirmation_needed", "etf_risk_level",
            "structural_risk_level", "structural_risk_reason",
            "risk_force_observe", "risk_blocking", "risk_block_reason",
        })

    def test_cross_border_requires_premium_confirmation_before_buy(self):
        metrics = attach_position_metrics(candidate("qdii_equity"), {"A": prices()}, "2026-03-06")
        risks = assess_etf_risks(metrics)
        result = analyze_actions(risks)
        self.assertEqual(risks.iloc[0].etf_type, "跨境ETF / QDII")
        self.assertTrue(risks.iloc[0].premium_confirmation_needed)
        self.assertEqual(risks.iloc[0].structural_risk_level, "评估不完整")
        self.assertEqual(result.iloc[0].action, "继续观察")
        self.assertIn("溢价", result.iloc[0].action_reason)

    def test_weak_liquidity_blocks_buy(self):
        metrics = attach_position_metrics(candidate(), {"A": prices(5_000_000)}, "2026-03-06")
        risks = assess_etf_risks(metrics, RiskConfig(min_recent_amount=20_000_000))
        result = analyze_actions(risks)
        self.assertEqual(risks.iloc[0].liquidity_status, "流动性偏弱")
        self.assertTrue(risks.iloc[0].risk_blocking)
        self.assertEqual(result.iloc[0].action, "暂不参与")

    def test_known_high_premium_blocks_buy(self):
        metrics = attach_position_metrics(candidate("commodity"), {"A": prices()}, "2026-03-06")
        metrics["premium_rate"] = 0.05
        risks = assess_etf_risks(metrics)
        self.assertEqual(risks.iloc[0].premium_status, "溢价偏高")
        self.assertTrue(risks.iloc[0].risk_blocking)

    def test_hong_kong_connect_equity_has_medium_structural_risk(self):
        item = candidate("equity")
        item["name"] = "港股通央企红利ETF"
        metrics = attach_position_metrics(item, {"A": prices()}, "2026-03-06")
        risk = assess_etf_risks(metrics).iloc[0]
        self.assertEqual(risk.etf_type, "港股通股票ETF")
        self.assertEqual(risk.structural_risk_level, "中")
        self.assertFalse(risk.premium_confirmation_needed)
        self.assertIn("港股交易制度", risk.special_risk_note)

        upstream_qdii = candidate("qdii_equity")
        upstream_qdii["name"] = "港股通科技ETF"
        overridden = assess_etf_risks(
            attach_position_metrics(upstream_qdii, {"A": prices()}, "2026-03-06")
        ).iloc[0]
        self.assertEqual(overridden.etf_type, "港股通股票ETF")


if __name__ == "__main__":
    unittest.main()
