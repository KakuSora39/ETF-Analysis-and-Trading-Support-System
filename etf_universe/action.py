"""Transparent current-position checks applied after strategy validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ActionConfig:
    min_buy_families: int = 2
    max_normal_volatility: float = 0.45
    max_buy_distance_ma20: float = 0.04
    max_buy_return_5d: float = 0.05
    weak_distance_ma60: float = -0.03


def _pct(current: float, base: float) -> float:
    return float(current / base - 1) if np.isfinite(base) and base > 0 else np.nan


def _fmt(value: float) -> str:
    return "数据不足" if pd.isna(value) else f"{value:+.1%}"


def _position_metrics(frame: pd.DataFrame, as_of: str) -> dict:
    frame = frame.copy()
    if frame.empty or "close" not in frame:
        return {"position_data_sufficient": False}
    if "date" in frame:
        frame = frame[pd.to_datetime(frame["date"]) <= pd.Timestamp(as_of)]
    frame = frame.sort_values("date") if "date" in frame else frame
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if len(close) < 60:
        return {"position_data_sufficient": False}

    current = float(close.iloc[-1])
    ma20 = float(close.iloc[-20:].mean())
    ma60 = float(close.iloc[-60:].mean())
    ma5 = float(close.iloc[-5:].mean())
    previous_ma5 = float(close.iloc[-6:-1].mean())
    high_source = pd.to_numeric(frame.get("high", frame.get("close")), errors="coerce").dropna()
    high20 = float(high_source.iloc[-20:].max())
    returns = close.iloc[-21:].pct_change(fill_method=None).dropna()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    previous_low = float(close.iloc[-8:-3].min())
    recent_low = float(close.iloc[-3:].min())
    recent_non_new_low_days = int((close.iloc[-3:] >= previous_low).sum())
    last_two_returns = close.pct_change(fill_method=None).iloc[-2:]
    recent_2d_sharp_decline = bool(len(last_two_returns) == 2 and (last_two_returns <= -0.02).all())
    latest_broke_recent_low = bool(current < float(close.iloc[-6:-1].min()))
    window60 = close.iloc[-60:]
    drawdown = window60 / window60.cummax() - 1

    activity_column = "amount" if "amount" in frame and pd.to_numeric(frame["amount"], errors="coerce").notna().sum() >= 20 else "volume"
    activity = pd.to_numeric(frame.get(activity_column), errors="coerce").dropna()
    activity_change = np.nan
    if len(activity) >= 20 and float(activity.iloc[-20:-5].mean()) > 0:
        activity_change = float(activity.iloc[-5:].mean() / activity.iloc[-20:-5].mean() - 1)

    return {
        "position_data_sufficient": True,
        "current_price": current,
        "return_1d": _pct(current, float(close.iloc[-2])) if len(close) >= 2 else np.nan,
        "return_3d": _pct(current, float(close.iloc[-4])) if len(close) >= 4 else np.nan,
        "return_5d": _pct(current, float(close.iloc[-6])) if len(close) >= 6 else np.nan,
        "return_20d": _pct(current, float(close.iloc[-21])) if len(close) >= 21 else np.nan,
        "ma20": ma20,
        "ma60": ma60,
        "ma5": ma5,
        "ma5_up": bool(ma5 > previous_ma5),
        "above_ma5": bool(current > ma5),
        "recent_3d_no_new_low": bool(recent_low >= previous_low),
        "recent_3d_non_new_low_days": recent_non_new_low_days,
        "recent_2d_sharp_decline": recent_2d_sharp_decline,
        "latest_broke_recent_low": latest_broke_recent_low,
        "rsi14": float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else np.nan,
        "rsi_recovering": bool(pd.notna(rsi.iloc[-4]) and pd.notna(rsi.iloc[-1]) and rsi.iloc[-1] > rsi.iloc[-4] and rsi.iloc[-5:].min() < 40),
        "close_above_previous": bool(current > float(close.iloc[-2])),
        "rebound_from_20d_low": _pct(current, float(close.iloc[-20:].min())),
        "distance_ma20": _pct(current, ma20),
        "distance_ma60": _pct(current, ma60),
        "distance_20d_high": _pct(current, high20),
        "volatility_20d": float(returns.std() * np.sqrt(252)) if len(returns) > 1 else np.nan,
        "max_drawdown_60d": float(drawdown.min()),
        "activity_change_5d": activity_change,
        "activity_basis": "成交额" if activity_column == "amount" else "成交量",
        "recent_avg_amount": float(pd.to_numeric(frame["amount"], errors="coerce").dropna().iloc[-5:].mean()) if "amount" in frame and pd.to_numeric(frame["amount"], errors="coerce").notna().sum() >= 5 else np.nan,
        "consecutive_up_days": _consecutive_up_days(close),
    }


def _consecutive_up_days(close: pd.Series) -> int:
    changes = close.pct_change(fill_method=None).dropna()
    count = 0
    for value in reversed(changes.tolist()):
        if value <= 0:
            break
        count += 1
    return count


def attach_position_metrics(analysis: pd.DataFrame, data: dict[str, pd.DataFrame], as_of: str) -> pd.DataFrame:
    """Calculate the shared price/liquidity metrics once for risk and action checks."""
    rows = []
    for _, source in analysis.iterrows():
        values = source.to_dict()
        values.update(_position_metrics(data.get(str(source["symbol"]), pd.DataFrame()), as_of))
        rows.append(values)
    return pd.DataFrame(rows)


def _judge(row: pd.Series, config: ActionConfig) -> tuple[str, str, str]:
    if not row.get("position_data_sufficient", False):
        return "暂不参与", "数据不足", "有效日线不足 60 个交易日，暂时无法判断趋势和当前位置。"
    if not row.get("backtest_eligible", False) or row.get("validated_family_votes", 0) == 0:
        return "暂不参与", "策略依据不足", "没有通过历史验证且当前支持这只 ETF 的策略，暂不参与。"
    if row.get("risk_blocking", False):
        return "暂不参与", "存在额外风险", f"ETF 特有风险检查未通过：{row.get('risk_block_reason', '存在需要先确认的风险')}"

    trend_up = row.ma20 > row.ma60 and row.return_20d > 0
    weak = row.distance_ma60 < config.weak_distance_ma60 or (row.ma20 <= row.ma60 and row.return_20d <= 0)
    hot_flags = [
        row.return_5d >= 0.06,
        row.distance_ma20 >= 0.06,
        row.distance_ma60 >= 0.12,
        row.distance_20d_high >= -0.01,
    ]
    very_hot = row.return_5d >= 0.10 or row.distance_ma20 >= 0.10 or sum(hot_flags) >= 2
    stretched = row.return_5d >= 0.05 or row.distance_ma20 >= 0.04
    buy_position = (
        trend_up
        and row.validated_family_votes >= config.min_buy_families
        and -0.02 <= row.distance_ma20 <= config.max_buy_distance_ma20
        and -0.03 <= row.return_5d <= config.max_buy_return_5d
        and row.volatility_20d <= config.max_normal_volatility
    )

    if weak:
        action, state = "暂不参与", "趋势偏弱"
        reason = f"价格相对 60 日均线为 {_fmt(row.distance_ma60)}，中期趋势尚未转强，当前不适合参与。"
    elif very_hot:
        action, state = "暂不追高", "趋势较强但短期偏热"
        reason = f"趋势虽然较强，但近 5 日涨跌幅为 {_fmt(row.return_5d)}，价格距离 20 日均线 {_fmt(row.distance_ma20)}，且距离 20 日高点 {_fmt(row.distance_20d_high)}，当前位置追入风险偏高。"
    elif row.get("risk_force_observe", False):
        action, state = "继续观察", "出现异常波动"
        reason = f"ETF 特有风险检查发现异常价格波动：{row.get('price_risk_reason', '')}。先观察价格恢复稳定。"
    elif buy_position:
        if row.get("premium_confirmation_needed", False):
            action, state = "继续观察", "位置合理但溢价未确认"
            reason = f"趋势与价格位置较合理，但这类 ETF 的实时折溢价数据不可用，买入前需要先确认场内价格没有明显溢价。"
        else:
            action, state = "可以关注买入", "趋势向上且位置合理"
            reason = f"20 日均线高于 60 日均线，近 5 日涨跌幅为 {_fmt(row.return_5d)}，价格距离 20 日均线 {_fmt(row.distance_ma20)}，趋势与当前位置均较合理。"
    elif trend_up and stretched:
        action, state = "等待回调", "趋势较强但位置偏高"
        reason = f"中期趋势向上，但近 5 日涨跌幅为 {_fmt(row.return_5d)}，价格距离 20 日均线 {_fmt(row.distance_ma20)}，更适合等待价格回到均线附近。"
    else:
        action, state = "继续观察", "方向或买点尚未确认"
        reason = f"已有有效策略支持，但当前趋势或价格位置尚未同时满足条件；近 20 日涨跌幅 {_fmt(row.return_20d)}，距离 20 日均线 {_fmt(row.distance_ma20)}。"
    return action, state, reason


def analyze_actions(
    analysis: pd.DataFrame,
    data: dict[str, pd.DataFrame] | None = None,
    as_of: str | None = None,
    config: ActionConfig | None = None,
) -> pd.DataFrame:
    """Attach current market metrics and an explainable action to each ETF."""
    config = config or ActionConfig()
    prepared = analysis if "position_data_sufficient" in analysis else attach_position_metrics(analysis, data or {}, str(as_of))
    rows = []
    for _, source in prepared.iterrows():
        values = source.to_dict()
        action, state, explanation = _judge(pd.Series(values), config)
        values.update(action=action, current_state=state, action_reason=explanation)
        rows.append(values)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    action_order = {"可以关注买入": 0, "等待回调": 1, "继续观察": 2, "暂不追高": 3, "暂不参与": 4}
    result["_action_order"] = result.action.map(action_order)
    result = result.sort_values(["_action_order", "analysis_rank"]).drop(columns="_action_order").reset_index(drop=True)
    result["action_rank"] = np.arange(1, len(result) + 1)
    return result
