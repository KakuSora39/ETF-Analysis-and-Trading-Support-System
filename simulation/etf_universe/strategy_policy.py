"""Experimental staged entries, evaluated only from the current signal row."""
from __future__ import annotations

import math

from .decision import conservative_entry


LABELS = {
    "no_position": "暂不参与", "watch": "观察", "probe_entry": "小仓试错",
    "add_position": "确认加仓", "hold": "继续持有", "reduce": "减仓", "exit": "退出",
}


def _number(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except (ValueError, TypeError):
        return default


def progressive_entry(row: dict, held: bool, config) -> tuple[str, float, str]:
    """Return stage, fraction of this symbol's budget, and an auditable reason."""
    if row.get("risk_blocking") or row.get("premium_confirmation_needed") or row.get("trading_risk_level") == "高":
        return ("hold" if held else "watch"), 0.0, "结构或交易风险阻止新增仓位"
    lane = row.get("opportunity_lane")
    action = str(row.get("action", ""))
    if lane == "trend":
        distance = _number(row, "distance_ma20", 1.0)
        r5 = _number(row, "return_5d")
        if action == "暂不追高" or distance > .08 or r5 > .10:
            return ("hold" if held else "watch"), 0.0, "短期过热，暂停新增仓位"
        formal, _, _ = conservative_entry(row)
        if formal == "entry":
            return ("add_position" if held else "probe_entry"), config.confirmed_position_ratio, "趋势正式确认"
        early = (action in {"继续观察", "等待回调"}
                 and _number(row, "return_3d") > 0
                 and r5 > 0
                 and distance <= .04
                 and _number(row, "validated_trend_support") >= 1
                 and _number(row, "trend_support") >= 1)
        if early and not held:
            return "probe_entry", config.probe_position_ratio, "短期强度改善且策略支持的早期趋势"
    elif lane == "rebound":
        score = _number(row, "stopping_score")
        support = _number(row, "validated_rebound_support") + _number(row, "weak_rebound_support")
        safe = (support >= 1 and not row.get("latest_broke_recent_low", False)
                and not row.get("recent_2d_sharp_decline", False)
                and row.get("recent_3d_no_new_low", False)
                and row.get("ma5_up", False))
        if safe and score >= 5 and row.get("rebound_persistence_pass", False):
            return ("add_position" if held else "probe_entry"), config.confirmed_position_ratio, "反弹5/5且持续性通过"
        if safe and score >= 3 and not held:
            ratio = config.probe_position_ratio * (.5 if score == 3 else 1.0)
            return "probe_entry", ratio, f"反弹止跌{int(score)}/5且风险门槛通过"
    elif lane == "defense":
        if (action in {"防守候选", "稳健持有候选"}
                and _number(row, "validated_defense_support") >= 1
                and _number(row, "volatility_20d", 1.0) <= .30):
            return ("add_position" if held else "probe_entry"), (config.confirmed_position_ratio if held else config.probe_position_ratio), "低波动且防守策略支持"
    return ("hold" if held else "watch"), 0.0, "尚未达到分阶段入场条件"
