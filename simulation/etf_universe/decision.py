"""综合模拟盘行动决策。只读取当日已知三赛道结果。"""
from __future__ import annotations

import math
import pandas as pd

from etf_universe.holdings import HoldingExitConfig, analyze_holdings, EXIT_ACTIONS

ENTRY_ACTIONS = {"可以关注买入", "小仓反弹试错", "较高优先级反弹候选", "高优先级反弹候选"}
LANE_TYPES = {"trend": "趋势", "rebound": "反弹", "defense": "防守"}


def conservative_entry(row: dict) -> tuple[str, float, str]:
    """稳定基准版只接受当前正式行动层给出的可执行信号。"""
    action = str(row.get("action", ""))
    if action not in ENTRY_ACTIONS or row.get("risk_blocking", False) or row.get("premium_confirmation_needed", False):
        return "watch", 0.0, "正式行动层未给出可执行买入信号"
    if row.get("opportunity_lane") == "defense":
        return "watch", 0.0, "防守候选尚未达到正式买点"
    return "entry", 1.0, str(row.get("action_reason", action))


def current_holding_advice(position: dict, frames: dict[str, pd.DataFrame], lanes: dict[str, pd.DataFrame],
                           as_of: str, time_stop_days: int = 8) -> tuple[str, str]:
    """复用人工持仓模块的分赛道退出检查，不读取人工持仓文件。"""
    row = {
        "symbol": position["symbol"], "name": position.get("name", ""),
        "buy_date": position["entry_date"], "buy_price": position["entry_price"],
        "shares": position["shares"], "peak_price": position.get("peak_price", position["entry_price"]),
        "opportunity_type": LANE_TYPES[position["entry_type"]],
        "entry_strategies": position.get("strategy_support_at_entry", []),
        "entry_reason": position.get("entry_reason", ""),
        "entry_reference_low": position.get("entry_metrics", {}).get("reference_low"),
        "entry_volatility": position.get("entry_metrics", {}).get("volatility_20d"),
    }
    config = HoldingExitConfig(rebound_max_days=time_stop_days)
    try:
        tracking, _ = analyze_holdings(pd.DataFrame([row]), frames, lanes, as_of, config)
    except (KeyError, ValueError, TypeError):
        return "持有观察", "当日历史行情或持仓指标不足，暂不能可靠判断退出"
    if tracking.empty:
        return "持有观察", "当日持仓分析无结果"
    result = tracking.iloc[0]
    return str(result["exit_advice"]), str(result["exit_reason"])
