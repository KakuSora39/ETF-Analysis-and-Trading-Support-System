"""ETF Universe 综合模拟盘每日 T+1 执行。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd

from simulation.framework.broker import SimBroker
from simulation.framework.engine import _check_limit_open, _check_suspended
from .account import PaperAccount
from .config import PaperConfig
from .decision import conservative_entry, current_holding_advice, EXIT_ACTIONS
from .strategy_policy import progressive_entry


def _row(frame: pd.DataFrame, day: str):
    if frame is None or frame.empty:
        return None
    match = frame[pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d").eq(day)]
    return match.iloc[-1] if not match.empty else None


def _price(row, field: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(field) if row is not None else None]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else 0.0


def _plain(value):
    """Convert pandas/numpy scalars before atomic JSON serialization."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        return None
    return value


def _order(mode: str, day: str, symbol: str, side: str, stage: str, **extra) -> dict:
    raw = f"{mode}|{day}|{symbol}|{side}|{stage}"
    return {"id": hashlib.sha256(raw.encode()).hexdigest()[:20], "signal_date": day,
            "symbol": symbol, "side": side, "stage": stage, "status": "pending", **extra}


def _execute(state: dict, order: dict, as_of: str, frames: dict[str, pd.DataFrame],
             broker: SimBroker, config: PaperConfig) -> None:
    if order["id"] in state["executed_order_ids"]:
        order["status"] = "executed"
        return
    symbol = order["symbol"]
    frame = frames.get(symbol)
    bar = _row(frame, as_of)
    if bar is None:
        order["status"] = "unfilled"
        order["reason"] = "执行日无可靠开盘行情，未成交"
        return
    # 若用户漏跑了首个后续交易日，不伪造过去的前向成交。
    next_dates = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    first_after = next_dates[next_dates.gt(order["signal_date"])].min()
    if first_after != as_of:
        order["status"] = "expired"
        order["reason"] = "错过首个后续交易日，订单作废，未回填历史成交"
        return
    opening = _price(bar, "open")
    prior = frame[next_dates.lt(as_of)]
    previous_close = _price(prior.iloc[-1], "close") if not prior.empty else 0.0
    if opening <= 0 or previous_close <= 0:
        order["status"] = "unfilled"
        order["reason"] = "开盘价或前收盘价无效"
        return
    suspended, reason = _check_suspended({symbol: {"volume": _price(bar, "volume")}}, symbol)
    limited, limit_reason = _check_limit_open(symbol, opening, previous_close)
    if suspended or (limited and ((order["side"] == "buy" and "涨停" in limit_reason) or (order["side"] == "sell" and "跌停" in limit_reason))):
        order["status"] = "unfilled"
        order["reason"] = reason or limit_reason
        return
    pos = state["positions"].get(symbol)
    if order["side"] == "buy":
        # Morning sizing cannot use today's close: use the opening auction for
        # existing holdings, then fall back to their recorded basis.
        equity = state["cash"] + sum(p["shares"] * (_price(_row(frames.get(s), as_of), "open") or p["entry_price"])
                                     for s, p in state["positions"].items())
        cap = equity * config.max_symbol_weight
        current_value = pos["shares"] * opening if pos else 0.0
        target = cap * float(order["target_ratio"])
        budget = min(max(0.0, target - current_value), state["cash"])
        fill = broker.quote_buy(opening, budget, state["cash"])
        if not fill.success:
            order["status"], order["reason"] = "unfilled", fill.reason
            return
        if pos is None:
            pos = {"symbol": symbol, "name": order.get("name", ""), "shares": 0,
                   "entry_type": order["entry_type"], "entry_reason": order.get("reason", ""),
                   "entry_date": as_of, "entry_price": 0.0, "entry_metrics": order.get("entry_metrics", {}),
                   "strategy_support_at_entry": order.get("strategies", []),
                   "market_regime_at_entry": order.get("market_regime", ""), "peak_price": fill.price,
                   "total_cost": 0.0, "target_ratio": 0.0}
            state["positions"][symbol] = pos
        pos["total_cost"] += fill.net_cost
        pos["shares"] += fill.shares
        pos["entry_price"] = pos["total_cost"] / pos["shares"]
        pos["peak_price"] = max(pos["peak_price"], fill.price)
        pos["target_ratio"] = max(pos.get("target_ratio", 0.0), float(order["target_ratio"]))
        state["cash"] -= fill.net_cost
        pnl = 0.0
    else:
        if pos is None or pos["shares"] <= 0:
            order["status"], order["reason"] = "cancelled", "已无持仓"
            return
        shares = pos["shares"] if order["stage"] == "exit" else max(100, int(pos["shares"] * .5 // 100) * 100)
        shares = min(shares, pos["shares"])
        basis = pos["total_cost"] * shares / pos["shares"]
        fill = broker.quote_sell(opening, shares, basis)
        if not fill.success:
            order["status"], order["reason"] = "unfilled", fill.reason
            return
        state["cash"] += fill.net_cost
        pos["shares"] -= shares
        pos["total_cost"] -= basis
        pnl = fill.pnl
        if pos["shares"] == 0:
            del state["positions"][symbol]
    state["executed_order_ids"].append(order["id"])
    order["status"] = "executed"
    state["trade_log"].append({"date": as_of, "signal_date": order["signal_date"], "order_id": order["id"],
                               "strategy_version": state["strategy_version"], "code": symbol,
                               "action": order["stage"], "entry_type": order.get("entry_type", pos.get("entry_type") if pos else ""),
                               "side": order["side"], "shares": fill.shares,
                               "price": fill.price, "amount": fill.amount, "commission": fill.commission,
                               "slippage_cost": abs(fill.price - opening) * fill.shares, "realized_pnl": round(pnl, 2),
                               "reason": order.get("reason", "")})


def run_day(mode: str, as_of: str, lanes: dict[str, pd.DataFrame], frames: dict[str, pd.DataFrame],
            output_dir: str | Path, config: PaperConfig | None = None, market_regime: str = "") -> dict:
    """先执行昨日订单，再根据今日收盘信号创建明日订单。重复日期不改状态。"""
    config = config or PaperConfig()
    account = PaperAccount(output_dir, mode, config)
    state = account.load_or_create(as_of)
    if state["last_processed_date"] >= as_of:
        return state
    broker = SimBroker(None, commission_rate=config.commission_rate, slippage=config.slippage)
    for order in state["pending_orders"]:
        if order["status"] == "pending" and order["signal_date"] < as_of:
            _execute(state, order, as_of, frames, broker, config)
    state["pending_orders"] = [o for o in state["pending_orders"] if o["status"] == "pending"]
    for symbol, pos in list(state["positions"].items()):
        bar = _row(frames.get(symbol), as_of)
        if bar is not None:
            pos["peak_price"] = max(pos["peak_price"], _price(bar, "close"))
        advice, reason = current_holding_advice(pos, frames, lanes, as_of, config.time_stop_days)
        if advice in EXIT_ACTIONS:
            stage = "reduce" if advice == "减仓锁利" else "exit"
            order = _order(mode, as_of, symbol, "sell", stage, reason=reason)
            if not any(o["symbol"] == symbol for o in state["pending_orders"]):
                state["pending_orders"].append(order)
    candidates = []
    for lane, frame in lanes.items():
        for _, source in frame.iterrows():
            row = source.to_dict()
            row["opportunity_lane"] = lane
            symbol = str(row["symbol"])
            if mode == "conservative":
                action, ratio, reason = conservative_entry(row)
                if symbol in state["positions"]:
                    continue
                stage = "entry"
            else:
                action, ratio, reason = progressive_entry(row, symbol in state["positions"], config)
                stage = action
            if action in {"entry", "probe_entry", "add_position"}:
                candidates.append((row, ratio, reason, stage))
    candidates.sort(key=lambda item: (item[0].get("analysis_rank", 999), -int(item[0].get("validated_family_votes", 0))))
    reserved = {o["symbol"] for o in state["pending_orders"] if o["side"] == "buy"}
    for row, ratio, reason, stage in candidates:
        symbol = str(row["symbol"])
        if any(o["symbol"] == symbol for o in state["pending_orders"]):
            continue
        if symbol not in state["positions"] and len(state["positions"]) + len(reserved) >= config.max_positions:
            continue
        if symbol in state["positions"] and stage != "add_position":
            continue
        if symbol in state["positions"] and state["positions"][symbol].get("target_ratio", 0) >= ratio:
            continue
        metrics = {key: row.get(key) for key in ("volatility_20d", "stopping_score", "recent_20d_low", "distance_ma20", "return_3d", "return_5d") if key in row}
        state["pending_orders"].append(_order(mode, as_of, symbol, "buy", stage, target_ratio=ratio,
            name=str(row.get("name", "")), entry_type=row["opportunity_lane"], reason=reason,
            strategies=[s for s in str(row.get(f"validated_{row['opportunity_lane']}_strategies", "")).split(";") if s],
            market_regime=market_regime, entry_metrics=_plain(metrics)))
        reserved.add(symbol)
    value = state["cash"]
    for symbol, pos in state["positions"].items():
        close = _price(_row(frames.get(symbol), as_of), "close")
        value += pos["shares"] * (close if close > 0 else pos["entry_price"])
    state["peak_value"] = max(state["peak_value"], value)
    previous = state["nav_history"][-1]["total_value"] if state["nav_history"] else state["initial_capital"]
    drawdown = value / state["peak_value"] - 1
    prior_max_drawdown = min((r["drawdown"] for r in state["nav_history"]), default=0.0)
    state["nav_history"].append({"date": as_of, "initial_capital": state["initial_capital"],
        "cash": round(state["cash"], 2), "total_value": round(value, 2), "peak_value": round(state["peak_value"], 2),
        "cumulative_return": value / state["initial_capital"] - 1, "daily_return": value / previous - 1,
        "drawdown": drawdown, "max_drawdown": min(prior_max_drawdown, drawdown), "positions": len(state["positions"]),
        "trade_count": len(state["trade_log"])})
    state["last_processed_date"] = as_of
    account.save(state)
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ETF Universe 双账户前向模拟盘")
    parser.add_argument("--db", type=Path, default=Path("data/etf_daily.db"))
    parser.add_argument("--analysis-dir", type=Path, help="单次 ETF Universe 分析输出目录")
    parser.add_argument("--analysis-root", type=Path, default=Path("data/universe"))
    parser.add_argument("--output-dir", type=Path, default=Path("simulation/output"))
    parser.add_argument("--config", type=Path, help="模拟盘 JSON 配置")
    args = parser.parse_args(argv)
    from etf_universe.data import load_strategy_market, _market_table
    from simulation.framework.state import StateManager
    from .history import SignalHistory
    from .reporter import append_excel, compare, export_accounts

    db_path = args.db.resolve()
    if not db_path.exists():
        parser.error(f"行情数据库不存在：{db_path}")
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as conn:
        table = _market_table(conn)
        latest = conn.execute(f"SELECT MAX(date) FROM {table}").fetchone()[0]
    if args.analysis_dir:
        analysis = args.analysis_dir
    else:
        candidates = sorted(args.analysis_root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        analysis = candidates[0].parent if candidates else None
    if analysis is None or not (analysis / "summary.json").exists():
        parser.error("没有完整 ETF Universe 分析结果；请先运行 python -m etf_universe")
    summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8-sig"))
    day = summary["as_of"]
    if day != latest:
        parser.error(f"分析行情日 {day} 与数据库最新日 {latest} 不一致；拒绝历史回填模拟成交")
    lane_names = ("trend", "rebound", "defense")
    missing = [name for name in lane_names if not (analysis / f"paper_lanes_{name}.csv").exists()]
    if missing:
        parser.error(f"缺少完整行动快照：{missing}；请重新运行 ETF Universe 分析")
    lanes = {name: pd.read_csv(analysis / f"paper_lanes_{name}.csv", dtype={"symbol": str},
                               encoding="utf-8-sig") for name in lane_names}
    config = PaperConfig.load(args.config)
    history = SignalHistory(args.output_dir / "signal_history.db")
    symbols = {str(symbol) for frame in lanes.values() for symbol in frame.get("symbol", [])}
    symbols |= history.pending_symbols()
    for mode in ("conservative", "progressive"):
        path = args.output_dir / f"state_etf_universe_{mode}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            symbols.update(saved.get("positions", {}))
            symbols.update(order["symbol"] for order in saved.get("pending_orders", []))
    frames, _, _ = load_strategy_market(db_path, list(symbols), day, 120) if symbols else ({}, None, [])
    states = {mode: run_day(mode, day, lanes, frames, args.output_dir, config,
                            summary.get("market_regime", "")) for mode in ("conservative", "progressive")}
    history.record_day(day, lanes, summary.get("market_regime", ""),
                       summary.get("broad_market_state", ""), config,
                       set(states["conservative"]["positions"]))
    history.mark_executed(states["progressive"]["trade_log"])
    validation_file = analysis / "strategy_backtests.csv"
    if validation_file.exists():
        history.cache_validation(day, pd.read_csv(validation_file, encoding="utf-8-sig"))
    history.fill_forward(frames, day)
    counts = history.counts()
    prices = {symbol: _price(_row(frame, day), "close") for symbol, frame in frames.items()}
    prices = {symbol: value for symbol, value in prices.items() if value > 0}
    result = compare(states["conservative"], states["progressive"], prices)
    result["as_of"] = day
    result["outcome_counts"] = counts
    StateManager.save_json_atomic(args.output_dir / "paper_summary.json", result)
    export_accounts(states, args.output_dir)
    workbook = analysis / summary.get("excel_report", "") if summary.get("excel_report") else None
    if workbook and workbook.is_file():
        append_excel(workbook, states, prices, day, counts)
    for mode, state in states.items():
        print(f"{mode}: 现金 {state['cash']:.2f}，持仓 {len(state['positions'])}，待执行订单 {len(state['pending_orders'])}")
    print(f"行情日 {day}；模拟盘状态：{args.output_dir}；Excel：{workbook if workbook and workbook.is_file() else '本次分析未生成'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
