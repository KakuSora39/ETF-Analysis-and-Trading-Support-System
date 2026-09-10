"""Independent action rules for trend, rebound, and defensive opportunities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .action import ActionConfig, analyze_actions
from .market_regime import MarketRegime


def _attach_consensus_and_trading_risk(frame: pd.DataFrame, lane: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    def numbers(column: str) -> pd.Series:
        values = result[column] if column in result else pd.Series(0, index=result.index)
        return pd.to_numeric(values, errors="coerce").fillna(0).astype(int)

    valid_support = numbers(f"validated_{lane}_support")
    valid_family_total = numbers(
        f"{lane}_validated_family_denominator"
        if f"{lane}_validated_family_denominator" in result else f"{lane}_validated_denominator"
    )
    valid_strategy_support = numbers(f"validated_{lane}_strategy_support")
    valid_strategy_total = numbers(f"{lane}_validated_strategy_denominator")
    total_support = numbers(f"{lane}_support")
    total_groups = numbers(
        f"{lane}_active_family_denominator"
        if f"{lane}_active_family_denominator" in result else f"{lane}_active_denominator"
    )
    result["effective_strategy_consensus"] = valid_strategy_support.astype(str) + " / " + valid_strategy_total.astype(str) + " 个历史有效策略支持"
    result["effective_family_consensus"] = valid_support.astype(str) + " / " + valid_family_total.astype(str) + " 个历史有效策略家族支持"
    result["effective_consensus"] = result["effective_family_consensus"]
    result["total_family_coverage"] = total_support.astype(str) + " / " + total_groups.astype(str) + " 个活跃策略家族支持"
    ratios = valid_support / valid_family_total.replace(0, np.nan)
    result["consensus_label"] = np.select(
        [valid_family_total.eq(0), valid_support.eq(0), valid_support.eq(1), ratios.eq(1), ratios.ge(.6)],
        ["暂无有效模型", "无有效支持", "单一家族信号", "强共识", "较强共识"],
        default="有一定支持",
    )
    if lane == "rebound":
        high = result.action.isin(["暂不参与", "继续观察", "反弹观察"]) | ~result.get("rebound_persistence_pass", False).fillna(False)
        medium = ~high
        reasons = np.where(
            high, "中期趋势可能仍弱，或止跌与持续性确认不足。",
            "短期修复得到确认，但反弹机会仍有较高波动和再次走弱风险。",
        )
    elif lane == "trend":
        high = result.action.isin(["暂不参与", "暂不追高"])
        medium = result.action.isin(["等待回调", "继续观察"])
        reasons = np.select(
            [high, medium],
            ["趋势偏弱、价格异常或短期过热，当前交易风险较高。", "趋势存在，但位置、波动或历史表现仍需确认。"],
            default="趋势和当前位置较协调，仍需承担正常市场波动。",
        )
    else:
        high = result.action.eq("暂不参与")
        medium = result.action.eq("防守候选")
        reasons = np.select(
            [high, medium],
            ["缺少有效防守模型支持，或波动与结构风险不适合参与。", "具备防守支持，但波动或回撤仍需观察。"],
            default="波动和回撤相对温和。",
        )
    result["trading_risk_level"] = np.select([high, medium], ["高", "中"], default="低")
    result["trading_risk_reason"] = reasons
    return result


def add_stopping_confirmation(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the transparent five-point stopping confirmation for rebound trades."""
    result = frame.copy()
    conditions = {
        "stop_no_new_low": result["recent_3d_no_new_low"].fillna(False).astype(bool),
        "stop_positive_3d": result["return_3d"].fillna(-np.inf) > 0,
        "stop_above_ma5": result["above_ma5"].fillna(False).astype(bool),
        "stop_ma5_up": result["ma5_up"].fillna(False).astype(bool),
        "stop_rsi_recovering": result["rsi_recovering"].fillna(False).astype(bool),
    }
    for column, values in conditions.items():
        result[column] = values
    result["stopping_score"] = sum(values.astype(int) for values in conditions.values())
    result["stopping_signals"] = result.apply(
        lambda row: "；".join([
            text for column, text in (
                ("stop_no_new_low", "最近3日未再创新低"), ("stop_positive_3d", "最近3日累计上涨"),
                ("stop_above_ma5", "价格站上5日均价"), ("stop_ma5_up", "5日均价开始向上"),
                ("stop_rsi_recovering", "RSI从低位回升"),
            ) if row[column]
        ]) or "尚未出现明确止跌信号", axis=1,
    )
    non_new_low_days = pd.to_numeric(
        result.get("recent_3d_non_new_low_days", result["recent_3d_no_new_low"].astype(int) * 3),
        errors="coerce",
    ).fillna(0)
    result["persistence_non_new_low"] = non_new_low_days >= 2
    result["persistence_no_sharp_decline"] = ~result.get(
        "recent_2d_sharp_decline", pd.Series(False, index=result.index)
    ).fillna(False).astype(bool)
    result["persistence_holds_ma5"] = result["above_ma5"] | result["ma5_up"]
    result["persistence_not_broken_low"] = ~result.get(
        "latest_broke_recent_low", pd.Series(False, index=result.index)
    ).fillna(False).astype(bool)
    persistence_columns = (
        "persistence_non_new_low", "persistence_no_sharp_decline",
        "persistence_holds_ma5", "persistence_not_broken_low",
    )
    result["rebound_persistence_pass"] = result[list(persistence_columns)].all(axis=1)
    result["rebound_persistence_reason"] = result.apply(
        lambda row: "；".join([
            text for column, text in (
                ("persistence_non_new_low", "最近3日中不足2日守住前期低点"),
                ("persistence_no_sharp_decline", "最近2日连续明显下跌"),
                ("persistence_holds_ma5", "价格与5日均线重新转弱"),
                ("persistence_not_broken_low", "最新价重新跌破近期低点"),
            ) if not row[column]
        ]) or "短期修复在最近几日仍有延续", axis=1,
    )
    return result


def _trend_actions(frame: pd.DataFrame, regime: MarketRegime, min_support: int, min_sharpe: float) -> pd.DataFrame:
    result = frame[frame.in_trend_shortlist].copy()
    if result.empty:
        return result
    result["analysis_rank"] = np.arange(1, len(result) + 1)
    result["validated_family_votes"] = result["validated_trend_support"]
    result["validated_strategies"] = result["validated_trend_strategies"]
    result["backtest_eligible"] = True
    result = analyze_actions(result, config=ActionConfig(min_buy_families=max(min_support, regime.trend_support_required)))
    weak_history = result.action.eq("可以关注买入") & result.trend_mean_sharpe.fillna(-np.inf).lt(min_sharpe)
    result.loc[weak_history, "action"] = "继续观察"
    result.loc[weak_history, "current_state"] = "当前位置合理但历史表现不足"
    result.loc[weak_history, "action_reason"] = f"当前位置满足趋势条件，但有效策略的平均风险收益表现低于 {min_sharpe:g}，继续观察。"
    result["opportunity_type"] = "趋势机会"
    return _attach_consensus_and_trading_risk(result, "trend")


def _rebound_actions(frame: pd.DataFrame, regime: MarketRegime) -> pd.DataFrame:
    selected = frame[frame.in_rebound_shortlist].copy()
    if selected.empty:
        return selected
    result = add_stopping_confirmation(selected)
    rows = []
    support_required = 2 if regime.code == "weak" else 1
    confirmation_required = regime.rebound_confirmation_required
    for _, source in result.iterrows():
        row = source.to_dict()
        support, score = int(row.get("validated_rebound_support", 0)), int(row["stopping_score"])
        weak_support = int(row.get("weak_rebound_support", 0))
        if row.get("risk_blocking", False):
            action, state, reason = "暂不参与", "额外风险较高", f"ETF特有风险检查未通过：{row.get('risk_block_reason', '')}"
        elif support == 0 and weak_support > 0:
            action, state = "反弹观察", "弱优势模型发现超跌修复"
            reason = (
                f"有 {weak_support} 个弱优势反弹家族提示该ETF，止跌确认 {score}/5。"
                "弱优势模型只用于发现和观察，不能单独触发小仓反弹试错；反弹也不代表趋势反转。"
            )
        elif support == 0:
            action, state, reason = "暂不参与", "缺少有效超跌策略", "当前超跌信号没有通过反弹专用历史验证。"
        elif score <= 1:
            action, state, reason = "暂不参与", "仍在弱势", f"止跌确认仅 {score}/5，跌得较多但尚未看到足够的停跌迹象。"
        elif score == 2:
            action, state, reason = "继续观察", "出现修复迹象", f"止跌确认 {score}/5，已经出现少量短期修复迹象，但还不足以进行反弹试错。"
        elif support < support_required or score < confirmation_required:
            action, state, reason = "继续观察", "反弹条件尚未满足环境门槛", f"止跌确认 {score}/5；{regime.label}要求至少 {confirmation_required}/5，且需要 {support_required} 类有效超跌支持。中期趋势可能仍弱，这不代表趋势反转。"
        elif not row.get("rebound_persistence_pass", False):
            action, state = "继续观察", "反弹持续性不足"
            reason = f"止跌确认虽然达到 {score}/5，但持续性检查未通过：{row.get('rebound_persistence_reason', '')}。先观察修复能否延续，不把单日大涨当成趋势反转。"
        elif row.get("premium_confirmation_needed", False):
            action, state, reason = "继续观察", "折溢价尚未确认", "已经出现止跌信号，但这类ETF需要先确认实时折溢价。"
        elif score == 3:
            action, state = "小仓反弹试错", "反弹条件初步成立"
            reason = f"止跌确认 {score}/5，且短期修复具有延续。可以作为高风险的小仓反弹试错，但中期趋势可能仍弱，不代表趋势反转。"
        elif score == 4:
            action, state = "较高优先级反弹候选", "反弹确认较强"
            reason = f"止跌确认 {score}/5，且持续性检查通过，可作为较高优先级反弹候选。它仍属于短期修复机会，不代表趋势已经反转。"
        else:
            action, state = "高优先级反弹候选", "反弹确认充分"
            reason = f"止跌确认 {score}/5，且持续性检查通过，可作为高优先级反弹候选。中期趋势可能仍弱，反弹确认不等于趋势反转。"
        row.update(action=action, current_state=state, action_reason=reason, opportunity_type="超跌反弹")
        rows.append(row)
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    action_order = {
        "高优先级反弹候选": 0, "较高优先级反弹候选": 1,
        "小仓反弹试错": 2, "反弹观察": 3, "继续观察": 4, "暂不参与": 5,
    }
    output["_action_order"] = output.action.map(action_order).fillna(5)
    output = output.sort_values(
        ["_action_order", "stopping_score", "validated_rebound_support"],
        ascending=[True, False, False],
    ).drop(columns="_action_order").reset_index(drop=True)
    return _attach_consensus_and_trading_risk(output, "rebound")


def _defense_actions(frame: pd.DataFrame, regime: MarketRegime) -> pd.DataFrame:
    rows = []
    for _, source in frame[frame.in_defense_shortlist].iterrows():
        row = source.to_dict()
        support = int(row.get("validated_defense_support", 0))
        valid_total = int(row.get("defense_validated_denominator", 0))
        if valid_total == 0:
            action, state = "暂不参与", "暂无有效防守策略"
            reason = "当前没有历史验证达到合格等级的防守策略，系统暂时无法可靠评价防守机会。"
        elif row.get("premium_confirmation_needed", False):
            action, state = "暂不参与", "折溢价尚未确认"
            reason = "该特殊品种缺少可靠折溢价数据，结构风险评估不完整，暂不提升为防守行动候选。"
        elif row.get("risk_blocking", False) or support < regime.defense_support_required:
            action, state = "暂不参与", "防守依据不足"
            reason = f"{regime.label}要求至少 {regime.defense_support_required} 类有效防守支持，当前为 {support} 类，或存在阻断风险。"
        elif row.get("volatility_20d", np.inf) <= .30 and row.get("max_drawdown_60d", -1) >= -.15:
            action, state = "稳健持有候选", "波动和回撤相对温和"
            reason = "当前波动率和近60日回撤相对温和，并有通过验证的防守策略支持。"
        else:
            action, state = "防守候选", "具备防守支持"
            reason = "有通过验证的低波动或风险收益策略支持，但波动和回撤仍需继续观察。"
        row.update(action=action, current_state=state, action_reason=reason, opportunity_type="防守机会")
        rows.append(row)
    return _attach_consensus_and_trading_risk(pd.DataFrame(rows), "defense")


def build_lane_actions(frame: pd.DataFrame, regime: MarketRegime, min_trend_support: int = 2, min_trend_sharpe: float = 0.5):
    """Return three separately judged opportunity frames."""
    return {
        "trend": _trend_actions(frame, regime, min_trend_support, min_trend_sharpe),
        "rebound": _rebound_actions(frame, regime),
        "defense": _defense_actions(frame, regime),
    }
