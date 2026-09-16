"""Account and A/B statistics derived from durable account states."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pandas as pd


def account_stats(state: dict, prices: dict[str, float] | None = None) -> dict:
    prices = prices or {}
    positions = state["positions"]
    market_value = sum(p["shares"] * prices.get(symbol, p["entry_price"])
                       for symbol, p in positions.items())
    total = state["cash"] + market_value
    history = state["nav_history"]
    peak = max([state["initial_capital"]] + [r["total_value"] for r in history] + [total])
    max_drawdown = min([0.0] + [r["drawdown"] for r in history] + [total / peak - 1])
    realized = sum(t["realized_pnl"] for t in state["trade_log"] if t["side"] == "sell")
    unrealized = sum(p["shares"] * prices.get(symbol, p["entry_price"]) - p["total_cost"]
                     for symbol, p in positions.items())
    return {
        "mode": state["mode"], "strategy_version": state["strategy_version"],
        "initial_capital": state["initial_capital"], "cash": state["cash"],
        "market_value": market_value, "total_value": total,
        "cumulative_return": total / state["initial_capital"] - 1,
        "daily_return": history[-1]["daily_return"] if history else 0.0,
        "peak_value": peak, "current_drawdown": total / peak - 1,
        "max_drawdown": max_drawdown, "trade_count": len(state["trade_log"]),
        "position_count": len(positions), "cash_ratio": state["cash"] / total if total else 0.0,
        "realized_pnl": realized, "unrealized_pnl": unrealized,
    }


def compare(conservative: dict, progressive: dict, prices: dict[str, float] | None = None) -> dict:
    a, b = account_stats(conservative, prices), account_stats(progressive, prices)
    return {
        "conservative": a, "progressive": b,
        "return_gap": b["cumulative_return"] - a["cumulative_return"],
        "max_drawdown_gap": b["max_drawdown"] - a["max_drawdown"],
        "trade_count_gap": b["trade_count"] - a["trade_count"],
        "cash_ratio_gap": b["cash_ratio"] - a["cash_ratio"],
        "position_count_gap": b["position_count"] - a["position_count"],
        "realized_pnl_gap": b["realized_pnl"] - a["realized_pnl"],
        "unrealized_pnl_gap": b["unrealized_pnl"] - a["unrealized_pnl"],
    }


def _csv_atomic(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(suffix=".csv", dir=path.parent)
    os.close(handle)
    try:
        pd.DataFrame(rows, columns=columns).to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export_accounts(states: dict[str, dict], output_dir: str | Path) -> None:
    output = Path(output_dir)
    for mode, state in states.items():
        _csv_atomic(output / f"trades_etf_universe_{mode}.csv", state["trade_log"],
                    ["date", "signal_date", "order_id", "strategy_version", "code", "action", "entry_type",
                     "side", "shares", "price", "amount", "commission", "slippage_cost", "realized_pnl", "reason"])
        _csv_atomic(output / f"nav_etf_universe_{mode}.csv", state["nav_history"],
                    ["date", "initial_capital", "cash", "total_value", "peak_value", "cumulative_return",
                     "daily_return", "drawdown", "max_drawdown", "positions", "trade_count"])


def append_excel(workbook: str | Path, states: dict[str, dict], prices: dict[str, float],
                 day: str, outcome_counts: dict[str, int]) -> None:
    """Replace the four paper sheets on rerun; never duplicate rows."""
    workbook = Path(workbook)
    comparison = compare(states["conservative"], states["progressive"], prices)
    overview = pd.DataFrame([comparison[mode] for mode in ("conservative", "progressive")])
    positions = []
    trades = []
    for mode, state in states.items():
        for symbol, pos in state["positions"].items():
            positions.append({"账户": mode, "ETF代码": symbol, "ETF名称": pos.get("name", ""),
                              "份额": pos["shares"], "买入类型": pos["entry_type"],
                              "买入日期": pos["entry_date"], "买入理由": pos["entry_reason"],
                              "成本": pos["total_cost"], "当前价格": prices.get(symbol),
                              "当前市值": pos["shares"] * prices.get(symbol, pos["entry_price"])})
        trades.extend({"账户": mode, **trade} for trade in state["trade_log"] if trade["date"] == day)
    if not trades:
        trades = [{"说明": "今日无模拟交易"}]
    diff = {key: value for key, value in comparison.items() if key not in states}
    diff.update({"missed_opportunity_count": outcome_counts.get("missed_opportunity", 0),
                 "false_probe_count": outcome_counts.get("false_probe", 0)})
    sheets = {
        "模拟盘总览": overview, "模拟盘持仓": pd.DataFrame(positions or [{"说明": "当前无模拟持仓"}]),
        "模拟交易记录": pd.DataFrame(trades), "A_B模拟对比": pd.DataFrame([diff]),
    }
    with pd.ExcelWriter(workbook, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    from etf_universe.excel_report import _format_workbook
    _format_workbook(workbook)
