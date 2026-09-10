import unittest

import numpy as np
import pandas as pd

from etf_universe.consensus import ConsensusResult
from etf_universe.lane_actions import add_stopping_confirmation, build_lane_actions
from etf_universe.market_regime import detect_market_regime
from etf_universe.opportunities import build_lane_shortlist
from etf_universe.theme import build_theme_summary, deduplicate_summary_lanes
from etf_universe.validation import validate_rebound_strategies


class LaneTests(unittest.TestCase):
    def test_similar_trend_strategies_count_as_one_group(self):
        status = pd.DataFrame([
            {"strategy": "momentum_rotation", "status": "voted"},
            {"strategy": "dual_momentum", "status": "voted"},
            {"strategy": "contrarian_reversion", "status": "voted"},
            {"strategy": "low_vol_rotation", "status": "voted"},
        ])
        votes = pd.DataFrame([
            {"strategy": strategy, "symbol": "A", "points": 3, "rank": 1}
            for strategy in status.strategy
        ])
        candidates = pd.DataFrame([{"symbol": "A", "name": "示例", "avg_amount": 3e7}])
        consensus = ConsensusResult(status, votes, pd.DataFrame(), pd.DataFrame())
        result, denominators = build_lane_shortlist(consensus, candidates)
        self.assertEqual(result.iloc[0].trend_support, 1)
        self.assertEqual(denominators, {"trend": 1, "rebound": 1, "defense": 1})
        self.assertEqual(result.iloc[0].trend_active_strategy_denominator, 2)
        self.assertEqual(result.iloc[0].trend_active_family_denominator, 1)

    def test_market_regime_changes_rebound_threshold(self):
        strong = pd.DataFrame({"close": np.linspace(100, 150, 300)})
        weak = pd.DataFrame({"close": np.linspace(150, 90, 300)})
        self.assertEqual(detect_market_regime(strong).rebound_confirmation_required, 3)
        self.assertEqual(detect_market_regime(weak).rebound_confirmation_required, 4)

    def test_range_benchmark_with_dispersed_etfs_is_structural(self):
        benchmark = pd.DataFrame({"close": 100 + np.sin(np.arange(300) / 8)})
        market = {}
        for index, total_return in enumerate(np.linspace(-.25, .25, 40)):
            market[str(index)] = pd.DataFrame({
                "close": 100 * np.exp(np.linspace(0, np.log1p(total_return), 30)),
            })
        regime = detect_market_regime(benchmark, market)
        self.assertIn(regime.broad_code, {"range", "range_strong", "range_weak"})
        self.assertEqual(regime.structure_code, "dispersed")
        self.assertEqual(regime.structure_label, "明显分化")
        self.assertIn("明显分化", regime.label)
        self.assertGreaterEqual(regime.dispersion_signals, 2)

    def test_stopping_confirmation_is_independent_of_ma20_ma60(self):
        row = pd.DataFrame([{
            "recent_3d_no_new_low": True, "return_3d": .02, "above_ma5": True,
            "ma5_up": True, "rsi_recovering": False,
        }])
        result = add_stopping_confirmation(row)
        self.assertEqual(result.iloc[0].stopping_score, 4)

    def test_weak_market_rebound_needs_four_points_but_not_bullish_ma(self):
        weak = detect_market_regime(pd.DataFrame({"close": np.linspace(150, 90, 300)}))
        row = pd.DataFrame([{
            "symbol": "A", "name": "反弹示例", "in_trend_shortlist": False,
            "in_rebound_shortlist": True, "in_defense_shortlist": False,
            "validated_rebound_support": 2, "rebound_active_denominator": 4,
            "validated_rebound_strategies": "mean_reversion;contrarian_reversion",
            "recent_3d_no_new_low": True, "return_3d": .02, "above_ma5": True,
            "ma5_up": True, "rsi_recovering": False, "ma20": 90, "ma60": 100,
            "risk_blocking": False, "premium_confirmation_needed": False,
        }])
        result = build_lane_actions(row, weak)["rebound"]
        self.assertEqual(result.iloc[0].stopping_score, 4)
        self.assertEqual(result.iloc[0].action, "较高优先级反弹候选")
        row["ma5_up"] = False
        result = build_lane_actions(row, weak)["rebound"]
        self.assertEqual(result.iloc[0].action, "继续观察")

    def test_rebound_persistence_can_veto_three_point_trial(self):
        regime = detect_market_regime(pd.DataFrame({"close": 100 + np.sin(np.arange(300) / 8)}))
        row = pd.DataFrame([{
            "symbol": "A", "name": "反弹示例", "in_trend_shortlist": False,
            "in_rebound_shortlist": True, "in_defense_shortlist": False,
            "validated_rebound_support": 1, "rebound_active_denominator": 4,
            "validated_rebound_strategies": "mean_reversion",
            "recent_3d_no_new_low": True, "recent_3d_non_new_low_days": 3,
            "return_3d": .02, "above_ma5": True, "ma5_up": False,
            "rsi_recovering": False, "recent_2d_sharp_decline": False,
            "latest_broke_recent_low": True, "risk_blocking": False,
            "premium_confirmation_needed": False,
        }])
        result = build_lane_actions(row, regime)["rebound"]
        self.assertEqual(result.iloc[0].stopping_score, 3)
        self.assertEqual(result.iloc[0].action, "继续观察")
        self.assertEqual(result.iloc[0].current_state, "反弹持续性不足")

    def test_summary_theme_dedup_keeps_two_representatives(self):
        rebound = pd.DataFrame([
            {"symbol": "A", "name": "半导体设备ETF", "index_id": "IDX:A", "action": "高优先级反弹候选", "validated_rebound_support": 3, "avg_amount": 3e8},
            {"symbol": "B", "name": "科创半导体ETF", "index_id": "IDX:B", "action": "较高优先级反弹候选", "validated_rebound_support": 2, "avg_amount": 2e8},
            {"symbol": "C", "name": "芯片ETF", "index_id": "IDX:C", "action": "小仓反弹试错", "validated_rebound_support": 1, "avg_amount": 1e8},
        ])
        lanes = {"trend": pd.DataFrame(), "rebound": rebound, "defense": pd.DataFrame()}
        result = deduplicate_summary_lanes(lanes)
        self.assertEqual(result["rebound"].symbol.tolist(), ["A", "B"])
        self.assertEqual(set(result["rebound"].summary_theme), {"半导体芯片"})

    def test_defense_theme_summary_has_one_representative_and_alternative(self):
        defense = pd.DataFrame([
            {"symbol": "513920", "name": "央企红利ETF", "action": "防守候选", "validated_defense_support": 2, "avg_amount": 3e8},
            {"symbol": "520990", "name": "港股通央企红利ETF", "action": "防守候选", "validated_defense_support": 1, "avg_amount": 2e8},
        ])
        result = build_theme_summary({"trend": pd.DataFrame(), "rebound": pd.DataFrame(), "defense": defense})
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].representative_symbol, "513920")
        self.assertIn("520990", result.iloc[0].alternative_etfs)

    def test_rebound_validation_uses_forward_returns(self):
        dates = pd.bdate_range("2025-01-01", periods=180)
        data = {}
        for index, symbol in enumerate(("A", "B", "C")):
            close = 100 + 8 * np.sin(np.arange(180) / (5 + index)) + index
            data[symbol] = pd.DataFrame({
                "date": dates, "close": close, "open": close,
                "high": close * 1.01, "low": close * .99, "volume": 1e6,
            })
        result = validate_rebound_strategies(data, list(data), backtest_days=120)
        self.assertEqual(len(result), 4)
        self.assertIn("forward_10d_win_rate", result)
        self.assertGreater(result.loc[result.strategy == "mean_reversion", "signal_count"].iloc[0], 0)
        self.assertTrue(set(result.validation_level) <= {"强", "合格", "弱优势", "未通过"})

    def test_consensus_displays_valid_models_separately_from_all_groups(self):
        regime = detect_market_regime(pd.DataFrame({"close": np.linspace(100, 110, 300)}))
        row = pd.DataFrame([{
            "symbol": "A", "name": "趋势示例", "in_trend_shortlist": True,
            "in_rebound_shortlist": False, "in_defense_shortlist": False,
            "validated_trend_support": 2, "validated_trend_strategy_support": 2,
            "trend_validated_family_denominator": 2, "trend_validated_strategy_denominator": 2,
            "trend_support": 2, "trend_active_family_denominator": 10,
            "validated_trend_strategies": "rsi_trend_rotation;bollinger_rotation",
            "trend_mean_sharpe": 1.0, "position_data_sufficient": True,
            "backtest_eligible": True, "risk_blocking": False, "risk_force_observe": False,
            "premium_confirmation_needed": False, "ma20": 101, "ma60": 100,
            "return_20d": .03, "distance_ma60": .02, "return_5d": .01,
            "distance_ma20": .01, "distance_20d_high": -.03, "volatility_20d": .2,
        }])
        result = build_lane_actions(row, regime)["trend"].iloc[0]
        self.assertEqual(result.effective_strategy_consensus, "2 / 2 个历史有效策略支持")
        self.assertEqual(result.effective_family_consensus, "2 / 2 个历史有效策略家族支持")
        self.assertEqual(result.total_family_coverage, "2 / 10 个活跃策略家族支持")
        self.assertEqual(result.consensus_label, "强共识")

    def test_weak_edge_rebound_only_enters_radar(self):
        regime = detect_market_regime(pd.DataFrame({"close": 100 + np.sin(np.arange(300) / 8)}))
        row = pd.DataFrame([{
            "symbol": "A", "name": "弱优势反弹", "in_trend_shortlist": False,
            "in_rebound_shortlist": True, "in_defense_shortlist": False,
            "validated_rebound_support": 0, "validated_rebound_strategy_support": 0,
            "weak_rebound_support": 2, "weak_rebound_strategies": "mean_reversion;contrarian_reversion",
            "recent_3d_no_new_low": True, "recent_3d_non_new_low_days": 3,
            "return_3d": .02, "above_ma5": True, "ma5_up": False,
            "rsi_recovering": False, "recent_2d_sharp_decline": False,
            "latest_broke_recent_low": False, "risk_blocking": False,
            "premium_confirmation_needed": False,
        }])
        result = build_lane_actions(row, regime)["rebound"].iloc[0]
        self.assertEqual(result.stopping_score, 3)
        self.assertEqual(result.action, "反弹观察")
        self.assertNotIn("小仓反弹试错", result.action)


if __name__ == "__main__":
    unittest.main()
