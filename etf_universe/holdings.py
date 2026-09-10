"""持仓读取与分机会类型卖出检查。

本模块只生成建议，不发送真实或模拟交易指令。手工持仓使用一个轻量
JSON 文件；已有 simulation/output/state_*.json 可按需作为模拟持仓读入。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .action import attach_position_metrics
from .lane_actions import add_stopping_confirmation


EXIT_ACTIONS = {
    "减仓锁利", "止盈退出", "止损退出", "趋势破坏退出",
    "反弹失败退出", "时间止损退出", "风险恶化退出",
}


@dataclass(frozen=True)
class HoldingExitConfig:
    trend_stop_loss: float = -0.12
    rebound_stop_loss: float = -0.08
    defense_stop_loss: float = -0.10
    trailing_activation: float = 0.05
    trailing_high_activation: float = 0.10
    trailing_pullback: float = -0.05
    trailing_high_pullback: float = -0.07
    rebound_max_days: int = 8
    rebound_target_return: float = 0.08
    defense_max_volatility: float = 0.35


def _normalize_type(value: str) -> str:
    text = str(value or "").strip()
    lower = text.lower()
    if "反弹" in text or "超跌" in text or lower == "rebound":
        return "反弹"
    if "防守" in text or "低波" in text or "风险收益" in text or lower in ("defense", "defensive"):
        return "防守"
    return "趋势"


def load_holdings(path: str | Path, simulation_dir: str | Path | None = None) -> pd.DataFrame:
    """读取人工持仓，并可兼容读取项目已有的模拟盘单持仓 JSON。"""
    rows: list[dict[str, Any]] = []
    source = Path(path)
    if source.exists():
        raw = json.loads(source.read_text(encoding="utf-8-sig"))
        items = raw.get("holdings", []) if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            raise ValueError("持仓文件必须是数组，或包含 holdings 数组的对象")
        for item in items:
            if not isinstance(item, dict):
                continue
            rows.append({**item, "holding_source": item.get("holding_source", "人工持仓")})
    if simulation_dir:
        for state_path in sorted(Path(simulation_dir).glob("state_*.json")):
            try:
                raw = json.loads(state_path.read_text(encoding="utf-8-sig"))
                pos = raw.get("position", {})
                if not pos.get("symbol") or int(pos.get("shares", 0)) <= 0:
                    continue
                strategy = raw.get("strategy_name") or state_path.stem.removeprefix("state_")
                rows.append({
                    "symbol": str(pos.get("symbol")), "name": pos.get("name", ""),
                    "buy_date": pos.get("buy_date", ""), "buy_price": pos.get("avg_cost", 0),
                    "shares": pos.get("shares", 0), "peak_price": pos.get("highest_price", 0),
                    "opportunity_type": pos.get("opportunity_type") or _normalize_type(strategy),
                    "entry_market_regime": pos.get("entry_market_regime", "未记录"),
                    "entry_strategies": pos.get("entry_strategies") or [strategy],
                    "entry_reason": pos.get("entry_reason", "旧模拟盘未保存完整买入理由"),
                    "holding_source": f"模拟盘：{strategy}",
                })
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    aliases = {
        "code": "symbol", "entry_date": "buy_date", "entry_price": "buy_price",
        "quantity": "shares", "entry_type": "opportunity_type",
    }
    for source_name, target_name in aliases.items():
        if target_name not in result and source_name in result:
            result[target_name] = result[source_name]
    if "active" in result:
        active = result["active"].map(lambda value: value if isinstance(value, bool) else str(value).strip().lower() not in ("false", "0", "否", "no"))
        result = result[active].copy()
    if result.empty:
        return pd.DataFrame()
    required = {"symbol", "buy_date", "buy_price", "shares"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"持仓记录缺少字段：{', '.join(sorted(missing))}")
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result["buy_date"] = pd.to_datetime(result["buy_date"], errors="coerce")
    result["buy_price"] = pd.to_numeric(result["buy_price"], errors="coerce")
    result["shares"] = pd.to_numeric(result["shares"], errors="coerce").fillna(0).astype(int)
    invalid = result["buy_date"].isna() | result["buy_price"].le(0) | result["shares"].le(0)
    if invalid.any():
        codes = "、".join(result.loc[invalid, "symbol"].astype(str).tolist())
        raise ValueError(f"持仓的买入日期、买入价或数量无效：{codes}")
    defaults = {
        "opportunity_type": "趋势", "entry_reason": "未记录",
        "entry_market_regime": "未记录", "entry_strategies": "",
    }
    for column, default in defaults.items():
        if column not in result:
            result[column] = default
    result["opportunity_type"] = result["opportunity_type"].map(_normalize_type)
    result["entry_reason"] = result["entry_reason"].fillna("未记录")
    result["entry_market_regime"] = result["entry_market_regime"].fillna("未记录")
    result["entry_strategies"] = result["entry_strategies"].map(
        lambda value: "；".join(value) if isinstance(value, list) else str(value or "")
    )
    return result.reset_index(drop=True)


def write_holdings_template(path: str | Path) -> Path:
    """写出可直接编辑的人工持仓模板，不覆盖已有文件。"""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"持仓文件已存在：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    template = {"version": 1, "holdings": [{
        "code": "示例：510300", "name": "示例：沪深300ETF", "entry_date": "YYYY-MM-DD",
        "entry_price": 0, "quantity": 0, "entry_type": "trend/rebound/defensive", "active": True,
        "entry_market_regime": "买入时市场环境", "entry_strategies": ["买入时支持策略"],
        "entry_reason": "当初为什么买", "entry_reference_low": None,
        "entry_volatility": None,
    }]}
    target.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _history_details(frame: pd.DataFrame, buy_date: pd.Timestamp) -> dict:
    if frame is None or frame.empty:
        return {"holding_data_sufficient": False}
    history = frame.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date")
    close = pd.to_numeric(history["close"], errors="coerce")
    bought = history[history.date >= buy_date].copy()
    if bought.empty or close.notna().sum() < 60:
        return {"holding_data_sufficient": False}
    bought_close = pd.to_numeric(bought["close"], errors="coerce").dropna()
    latest_ma20 = float(close.tail(20).mean())
    previous_ma20 = float(close.iloc[-21:-1].mean()) if len(close) >= 21 else np.nan
    two_below = bool(len(close) >= 21 and (close.iloc[-2:] < latest_ma20).all())
    before_entry = history[history.date < buy_date].tail(20)
    reference_low = pd.to_numeric(before_entry.get("low", before_entry.get("close")), errors="coerce").min()
    trading_days = int(len(bought_close) - 1)
    return {
        "holding_data_sufficient": True,
        "history_peak_price": float(bought_close.max()),
        "two_days_below_ma20": two_below,
        "ma20_weakening": bool(np.isfinite(previous_ma20) and latest_ma20 < previous_ma20),
        "entry_reference_low_derived": float(reference_low) if np.isfinite(reference_low) else np.nan,
        "holding_trading_days": max(0, trading_days),
    }


def _support_lookup(lanes: dict[str, pd.DataFrame]) -> dict[tuple[str, str], tuple[int, str, str, str]]:
    output: dict[tuple[str, str], tuple[int, str, str, str]] = {}
    for lane, frame in lanes.items():
        if frame.empty:
            continue
        column = f"validated_{lane}_support"
        for _, row in frame.iterrows():
            output[(str(row["symbol"]), lane)] = (
                int(row.get(column, 0)), str(row.get(f"validated_{lane}_strategies", "")),
                str(row.get("trading_risk_level", "暂无可靠数据")),
                str(row.get("trading_risk_reason", "未进入今日赛道候选，风险结论有限")),
            )
    return output


def _entry_support_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return len([part for part in str(value or "").replace("；", ";").split(";") if part.strip()])


def analyze_holdings(
    holdings: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    lanes: dict[str, pd.DataFrame],
    as_of: str,
    config: HoldingExitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐只检查买入理由是否仍成立，返回持仓跟踪和卖出信号。"""
    if holdings.empty:
        return pd.DataFrame(), pd.DataFrame()
    cfg = config or HoldingExitConfig()
    metrics = attach_position_metrics(holdings, data, as_of)
    support = _support_lookup(lanes)
    defense_frame = lanes.get("defense", pd.DataFrame())
    best_defense_value = pd.to_numeric(
        defense_frame.get("validated_defense_support", pd.Series(dtype=float)), errors="coerce"
    ).max() if not defense_frame.empty else np.nan
    best_defense_support = int(best_defense_value) if pd.notna(best_defense_value) else 0
    rows = []
    for _, source in metrics.iterrows():
        row = source.to_dict()
        symbol = str(row["symbol"])
        row["opportunity_type"] = _normalize_type(row.get("opportunity_type", "趋势"))
        lane = {"趋势": "trend", "反弹": "rebound", "防守": "defense"}[row["opportunity_type"]]
        history = data.get(symbol, pd.DataFrame())
        row.update(_history_details(history, pd.Timestamp(row["buy_date"])))
        current = float(row.get("current_price", np.nan))
        buy_price = float(row["buy_price"])
        row["current_return"] = current / buy_price - 1 if np.isfinite(current) else np.nan
        stored_peak = pd.to_numeric(pd.Series([row.get("peak_price")]), errors="coerce").iloc[0]
        peak_price = max([value for value in (stored_peak, row.get("history_peak_price"), current) if pd.notna(value) and value > 0], default=np.nan)
        row["peak_return"] = peak_price / buy_price - 1 if np.isfinite(peak_price) else np.nan
        row["drawdown_from_peak"] = current / peak_price - 1 if np.isfinite(peak_price) else np.nan
        current_support, current_strategies, trading_risk, trading_risk_reason = support.get(
            (symbol, lane), (0, "", "暂无可靠数据", "未进入今日赛道候选，当前策略支持按0处理")
        )
        row["current_strategy_support"] = current_support
        row["current_support_strategies"] = current_strategies
        row["current_trading_risk"] = trading_risk
        row["current_trading_risk_reason"] = trading_risk_reason
        row["entry_strategy_support"] = _entry_support_count(row.get("entry_strategies"))
        row["entry_reason_still_valid"] = True

        if not row.get("position_data_sufficient", False) or not row.get("holding_data_sufficient", False):
            status, reason = "持有观察", "持仓行情不足60个有效交易日，暂时不能可靠判断退出条件。"
        elif lane == "trend":
            broken = row.get("two_days_below_ma20", False) or (
                current < row["ma20"] and current_support < row["entry_strategy_support"]
            ) or (row["ma20"] < row["ma60"] and row.get("ma20_weakening", False))
            if row["current_return"] <= cfg.trend_stop_loss:
                status, reason = "止损退出", f"当前亏损 {row['current_return']:.1%} 已触及趋势持仓安全阀。"
                row["entry_reason_still_valid"] = False
            elif broken:
                status, reason = "趋势破坏退出", "价格连续跌破20日均线，或均线与趋势策略支持同步转弱，原趋势买入理由已经失效。"
                row["entry_reason_still_valid"] = False
            elif row["peak_return"] >= cfg.trailing_high_activation and row["drawdown_from_peak"] <= cfg.trailing_high_pullback:
                status, reason = "减仓锁利", f"最高收益曾达 {row['peak_return']:.1%}，目前从高点回撤 {row['drawdown_from_peak']:.1%}，可减仓保护利润。"
            elif row["peak_return"] >= cfg.trailing_activation and row["drawdown_from_peak"] <= cfg.trailing_pullback:
                status, reason = "减仓锁利", f"移动止盈已启用，目前从持仓高点回撤 {row['drawdown_from_peak']:.1%}。"
            elif row.get("return_5d", 0) >= .10 or row.get("distance_ma20", 0) >= .10 or row.get("rsi14", 0) >= 80:
                status, reason = "减仓锁利", "短期涨幅、均线偏离或RSI显示明显过热，可部分锁定利润。"
            elif current < row["ma20"] or row.get("ma20_weakening", False) or current_support < row["entry_strategy_support"]:
                status, reason = "持有观察", "趋势出现一项转弱迹象，但尚未形成连续确认，暂不因单日波动退出。"
            elif current_support >= max(1, row["entry_strategy_support"]) and -.01 <= row.get("distance_ma20", np.nan) <= .02:
                status, reason = "可以考虑加仓", "原趋势理由仍成立，价格靠近20日均线且当前策略支持未减少；是否加仓仍由用户决定。"
            else:
                status, reason = "继续持有", "价格仍在趋势结构内，原买入理由尚未失效。"
        elif lane == "rebound":
            scored = add_stopping_confirmation(pd.DataFrame([row])).iloc[0]
            row.update(scored.to_dict())
            reference_low = pd.to_numeric(pd.Series([row.get("entry_reference_low")]), errors="coerce").iloc[0]
            if not np.isfinite(reference_low):
                reference_low = row.get("entry_reference_low_derived", np.nan)
            failed = (np.isfinite(reference_low) and current < reference_low) or row.get("latest_broke_recent_low", False) or (
                int(row.get("stopping_score", 0)) <= 1 and row["current_return"] < 0
            )
            if row["current_return"] <= cfg.rebound_stop_loss:
                status, reason = "止损退出", f"当前亏损 {row['current_return']:.1%} 已触及反弹试错安全阀。"
                row["entry_reason_still_valid"] = False
            elif failed:
                status, reason = "反弹失败退出", f"价格重新失守近期低点或止跌确认降至 {int(row.get('stopping_score', 0))}/5，原反弹理由失效。"
                row["entry_reason_still_valid"] = False
            elif current >= row["ma20"] or row.get("rsi14", 0) >= 55 or row.get("rebound_from_20d_low", 0) >= cfg.rebound_target_return:
                status, reason = "止盈退出", "价格已回到20日均线附近、RSI恢复中性或达到预设反弹幅度，短期修复目标基本完成。"
            elif row.get("holding_trading_days", 0) >= cfg.rebound_max_days:
                status, reason = "时间止损退出", f"买入后已过 {row['holding_trading_days']} 个交易日，仍未完成预期修复，反弹逻辑按时间失效。"
                row["entry_reason_still_valid"] = False
            elif not row.get("rebound_persistence_pass", False):
                status, reason = "持有观察", "止跌持续性转弱，但尚未失守前低；反弹持仓需要更紧密观察。"
            else:
                status, reason = "继续持有", "短期修复仍在延续，且尚未达到反弹目标或失败条件。"
        else:
            entry_vol = pd.to_numeric(pd.Series([row.get("entry_volatility")]), errors="coerce").iloc[0]
            volatility_bad = row.get("volatility_20d", 0) > cfg.defense_max_volatility or (
                np.isfinite(entry_vol) and row.get("volatility_20d", 0) > entry_vol * 1.5
            )
            row["better_defense_available"] = best_defense_support > current_support
            if row["current_return"] <= cfg.defense_stop_loss:
                status, reason = "止损退出", f"当前亏损 {row['current_return']:.1%} 已触及防守持仓安全阀。"
                row["entry_reason_still_valid"] = False
            elif volatility_bad and current_support == 0:
                status, reason = "风险恶化退出", "波动率明显恶化且防守策略支持消失，原低风险持有理由已经失效。"
                row["entry_reason_still_valid"] = False
            elif volatility_bad or current_support < row["entry_strategy_support"]:
                replacement = "；今日防守Sheet中存在支持度更高的替代候选" if row["better_defense_available"] else ""
                status, reason = "持有观察", f"波动或防守策略支持出现恶化，需要重新比较防守候选{replacement}。"
            else:
                status, reason = "继续持有", "当前风险特征没有明显恶化；防守持仓不因短期涨得慢而退出。"
        row["holding_status"] = status
        row["exit_advice"] = status
        row["exit_reason"] = reason
        rows.append(row)
    tracking = pd.DataFrame(rows)
    signals = tracking[tracking.exit_advice.isin(EXIT_ACTIONS)].copy()
    return tracking, signals


def exit_config_dict(config: HoldingExitConfig | None = None) -> dict:
    return asdict(config or HoldingExitConfig())
