"""Comparable historical validation for current strategy votes."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .consensus import ConsensusResult, EVALUATORS
from .opportunities import STRATEGY_LANES


def validation_confidence(days: int | float | None) -> str:
    """Describe how much trust the validation window can reasonably carry."""
    value = int(days or 0)
    if value < 120:
        return "低"
    if value <= 250:
        return "中"
    return "较高"


def _metrics(equity: list[float], held_days: int, trades: int, benchmark_return: float):
    values = pd.Series([1.0] + equity, dtype=float)
    returns = values.pct_change(fill_method=None).dropna()
    days = len(returns)
    total_return = values.iloc[-1] - 1 if len(values) else 0.0
    annualized = (
        (values.iloc[-1] ** (252 / days) - 1) if days and values.iloc[-1] > 0 else -1.0
    )
    volatility = returns.std() * math.sqrt(252) if len(returns) > 1 else 0.0
    sharpe = (
        returns.mean() / returns.std() * math.sqrt(252)
        if len(returns) > 1 and returns.std() > 0
        else 0.0
    )
    drawdown = values / values.cummax() - 1
    max_drawdown = float(drawdown.min())
    calmar = annualized / abs(max_drawdown) if max_drawdown < 0 else 0.0
    return {
        "days": days,
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "benchmark_return": float(benchmark_return),
        "excess_return": float(total_return - benchmark_return),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "trades": int(trades),
        "exposure": float(held_days / days) if days else 0.0,
        "win_rate": float((returns > 0).mean()) if days else 0.0,
    }


def _equal_weight_return(
    data: dict[str, pd.DataFrame], symbols: list[str], start: int, end: int
):
    returns = []
    for symbol in symbols:
        frame = data.get(symbol)
        if frame is None or end >= len(frame):
            continue
        first = pd.to_numeric(
            pd.Series([frame.iloc[start]["close"]]), errors="coerce"
        ).iloc[0]
        last = pd.to_numeric(
            pd.Series([frame.iloc[end]["close"]]), errors="coerce"
        ).iloc[0]
        if np.isfinite(first) and np.isfinite(last) and first > 0:
            returns.append(last / first - 1)
    return float(np.mean(returns)) if returns else 0.0


def backtest_strategies(
    data: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    symbols: list[str],
    backtest_days: int = 504,
    rebalance_days: int = 5,
    one_way_cost: float = 0.0003,
    max_drawdown: float = 0.35,
    min_sharpe: float = 0.0,
) -> pd.DataFrame:
    """Validate every comparable selector under one execution convention.

    Signals use data through T-1 and changes execute at T open.  The current
    candidate list is intentionally held fixed, so this is a validation of the
    current shortlist rather than a point-in-time full-market backtest.
    """
    if backtest_days < 60 or rebalance_days < 1:
        raise ValueError(
            "backtest_days must be >=60 and rebalance_days must be positive"
        )
    if not 0 <= one_way_cost < 0.1 or not 0 < max_drawdown < 1:
        raise ValueError("invalid backtest cost or drawdown threshold")
    requested_symbols = [s for s in symbols if s in data]
    symbols = []
    for symbol in requested_symbols:
        frame = data[symbol]
        valid = pd.to_numeric(frame["open"], errors="coerce").gt(0) & pd.to_numeric(
            frame["close"], errors="coerce"
        ).gt(0)
        if int(valid.sum()) >= 311 and bool(valid.iloc[-1]):
            symbols.append(symbol)
    if not symbols:
        raise ValueError("重点候选中没有具备至少311个有效交易日的ETF")
    length = min(len(data[s]) for s in symbols)
    # Use a shared tradable calendar.  A current focus ETF may have listed after
    # the requested history start; its pre-listing NaNs are not strategy errors.
    common = np.ones(length, dtype=bool)
    for symbol in symbols:
        frame = data[symbol].iloc[:length]
        open_price = pd.to_numeric(frame["open"], errors="coerce").to_numpy()
        close_price = pd.to_numeric(frame["close"], errors="coerce").to_numpy()
        common &= (
            np.isfinite(open_price)
            & (open_price > 0)
            & np.isfinite(close_price)
            & (close_price > 0)
        )
    positions = np.flatnonzero(common)
    data = {
        symbol: data[symbol].iloc[positions].reset_index(drop=True)
        for symbol in symbols
    }
    if benchmark is not None and not benchmark.empty:
        benchmark = benchmark.iloc[positions].reset_index(drop=True)
    length = len(positions)
    warmup = 251
    start = max(warmup, length - backtest_days)
    end = length - 1
    if end - start + 1 < 60:
        raise ValueError("策略历史不足：至少需要约311个共同交易日")
    benchmark_return = _equal_weight_return(data, symbols, start, end)
    backtest_start_date = pd.Timestamp(data[symbols[0]].iloc[start]["date"]).date().isoformat()
    backtest_end_date = pd.Timestamp(data[symbols[0]].iloc[end]["date"]).date().isoformat()
    tested_symbols = ";".join(symbols)
    rows = []

    for name, evaluator in sorted(EVALUATORS.items()):
        cash, shares, holding = 1.0, 0.0, None
        target = None
        equity, held_days, trades = [], 0, 0
        error = ""
        for idx in range(start, end + 1):
            if (idx - start) % rebalance_days == 0:
                sliced = {s: data[s].iloc[:idx].copy() for s in symbols}
                sliced_benchmark = (
                    benchmark.iloc[:idx].copy()
                    if benchmark is not None
                    else pd.DataFrame()
                )
                try:
                    vote = evaluator(sliced, sliced_benchmark, 1)
                    target = vote.selections[0][0] if vote.selections else None
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    break

            if target != holding:
                if holding is not None:
                    sell_open = data[holding].iloc[idx]["open"]
                    if not np.isfinite(sell_open) or sell_open <= 0:
                        error = f"{holding} 在执行日缺少有效开盘价"
                        break
                    cash = shares * sell_open * (1 - one_way_cost)
                    shares = 0.0
                if target is not None:
                    buy_open = data[target].iloc[idx]["open"]
                    if not np.isfinite(buy_open) or buy_open <= 0:
                        target = None
                    else:
                        shares = cash / (buy_open * (1 + one_way_cost))
                        cash = 0.0
                holding = target
                trades += 1

            if holding is None:
                value = cash
            else:
                close = data[holding].iloc[idx]["close"]
                if not np.isfinite(close) or close <= 0:
                    error = f"{holding} 在估值日缺少有效收盘价"
                    break
                value = shares * close
                held_days += 1
            equity.append(float(value))

        if error:
            rows.append(
                {
                    "strategy": name,
                    "status": "error",
                    "passed": False,
                    "tested_symbols": tested_symbols,
                    "reason": error,
                    "backtest_start_date": backtest_start_date,
                    "backtest_end_date": backtest_end_date,
                    "one_way_cost": one_way_cost,
                    "slippage": 0.0,
                    "price_adjustment": "分析用前复权（含价格断层检测）",
                    "days": len(equity),
                    "validation_confidence": validation_confidence(len(equity)),
                }
            )
            continue
        metrics = _metrics(equity, held_days, trades, benchmark_return)
        passed = (
            metrics["total_return"] > 0
            and metrics["excess_return"] > 0
            and metrics["sharpe"] > min_sharpe
            and metrics["max_drawdown"] >= -max_drawdown
            and metrics["trades"] >= 2
        )
        failures = []
        if metrics["total_return"] <= 0:
            failures.append("收益非正")
        if metrics["excess_return"] <= 0:
            failures.append("未跑赢重点候选等权基准")
        if metrics["sharpe"] <= min_sharpe:
            failures.append(f"夏普不高于{min_sharpe:g}")
        if metrics["max_drawdown"] < -max_drawdown:
            failures.append(f"最大回撤超过{max_drawdown:.0%}")
        if metrics["trades"] < 2:
            failures.append("有效交易不足")
        quality = max(0.0, metrics["sharpe"]) * max(0.0, 1 + metrics["max_drawdown"])
        rows.append(
            {
                "strategy": name,
                "status": "completed",
                "passed": bool(passed),
                "quality_weight": float(quality),
                "tested_symbols": tested_symbols,
                "reason": ";".join(failures),
                "backtest_start_date": backtest_start_date,
                "backtest_end_date": backtest_end_date,
                "one_way_cost": one_way_cost,
                "slippage": 0.0,
                "price_adjustment": "分析用前复权（含价格断层检测）",
                "validation_confidence": validation_confidence(metrics["days"]),
                **metrics,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["passed", "sharpe", "total_return"],
            ascending=[False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_buy_analysis(
    consensus: ConsensusResult,
    backtests: pd.DataFrame,
    strategy_top_n: int,
    min_buy_families: int = 2,
    min_buy_mean_sharpe: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine validated strategy quality with today's votes."""
    passed = backtests[
        (backtests.get("passed", False) == True) & (backtests.status == "completed")
    ].copy()  # noqa: E712
    valid_votes = consensus.votes[consensus.votes.strategy.isin(passed.strategy)].copy()
    valid_votes = valid_votes[valid_votes.symbol.isin(consensus.focus.symbol)].copy()
    tested_by_strategy = (
        passed.set_index("strategy")["tested_symbols"]
        .fillna("")
        .map(lambda value: set(str(value).split(";")))
    )
    valid_votes = valid_votes[
        valid_votes.apply(
            lambda row: row["symbol"] in tested_by_strategy.get(row["strategy"], set()),
            axis=1,
        )
    ]
    if valid_votes.empty:
        result = consensus.focus.copy()
        tested_union = set()
        if "tested_symbols" in passed:
            for value in passed["tested_symbols"].dropna():
                tested_union.update(str(value).split(";"))
        result["backtest_eligible"] = result["symbol"].isin(tested_union)
        result["validated_family_votes"] = 0
        result["validated_strategy_votes"] = 0
        result["backtest_weighted_score"] = 0.0
        result["mean_backtest_sharpe"] = np.nan
        result["validated_strategies"] = ""
        result["decision"] = "暂不买入"
        result["decision_reason"] = "没有通过回测且当前选中该ETF的策略"
        return result, valid_votes

    quality = passed.set_index("strategy")["quality_weight"]
    sharpe = passed.set_index("strategy")["sharpe"]
    valid_votes["quality_weight"] = valid_votes.strategy.map(quality)
    valid_votes["backtest_sharpe"] = valid_votes.strategy.map(sharpe)
    valid_votes["weighted_points"] = (
        valid_votes["quality_weight"]
        * (strategy_top_n - valid_votes["rank"] + 1)
        / strategy_top_n
    )
    family_best = valid_votes.sort_values(
        ["weighted_points", "rank"], ascending=[False, True]
    ).drop_duplicates(["family", "symbol"])
    family = family_best.groupby("symbol").agg(
        validated_family_votes=("family", "nunique"),
        backtest_weighted_score=("weighted_points", "sum"),
    )
    raw = valid_votes.groupby("symbol").agg(
        validated_strategy_votes=("strategy", "nunique"),
        mean_backtest_sharpe=("backtest_sharpe", "mean"),
        validated_strategies=("strategy", lambda s: ";".join(sorted(set(s)))),
    )
    result = consensus.focus.merge(
        family, left_on="symbol", right_index=True, how="left"
    ).merge(raw, left_on="symbol", right_index=True, how="left")
    tested_union = (
        set().union(*tested_by_strategy.tolist()) if len(tested_by_strategy) else set()
    )
    result["backtest_eligible"] = result["symbol"].isin(tested_union)
    result[
        [
            "validated_family_votes",
            "validated_strategy_votes",
            "backtest_weighted_score",
        ]
    ] = result[
        [
            "validated_family_votes",
            "validated_strategy_votes",
            "backtest_weighted_score",
        ]
    ].fillna(
        0
    )
    result["validated_family_votes"] = result["validated_family_votes"].astype(int)
    result["validated_strategy_votes"] = result["validated_strategy_votes"].astype(int)
    result["validated_strategies"] = result["validated_strategies"].fillna("")
    buy = (result.validated_family_votes >= min_buy_families) & (
        result.mean_backtest_sharpe >= min_buy_mean_sharpe
    )
    watch = result.validated_family_votes > 0
    result["decision"] = np.select(
        [buy, watch], ["可考虑买入", "继续观察"], default="暂不买入"
    )
    result["decision_reason"] = np.select(
        [~result.backtest_eligible, buy, watch],
        [
            "有效历史不足311个交易日",
            "通过回测的当前信号达到家族数和平均夏普门槛",
            f"有通过回测的当前信号，但有效策略家族少于{min_buy_families}",
        ],
        default="通过回测的策略当前均未选中",
    )
    order = {"可考虑买入": 0, "继续观察": 1, "暂不买入": 2}
    result["_decision_order"] = result.decision.map(order)
    result = (
        result.sort_values(
            [
                "_decision_order",
                "validated_family_votes",
                "backtest_weighted_score",
                "consensus_rank",
            ],
            ascending=[True, False, False, True],
        )
        .drop(columns="_decision_order")
        .reset_index(drop=True)
    )
    result.insert(0, "analysis_rank", np.arange(1, len(result) + 1))
    return result, valid_votes


def validate_rebound_strategies(
    data: dict[str, pd.DataFrame],
    symbols: list[str],
    backtest_days: int = 504,
    signal_interval: int = 5,
    top_n: int = 3,
) -> pd.DataFrame:
    """Validate rebound signals by their 5/10-day outcomes instead of long holding returns."""
    symbols = [symbol for symbol in symbols if symbol in data and len(data[symbol]) >= 80]
    common_length = min((len(data[symbol]) for symbol in symbols), default=0)
    common_start = max(60, common_length - backtest_days)
    validation_start = (
        pd.Timestamp(data[symbols[0]].iloc[common_start]["date"]).date().isoformat()
        if symbols and common_length > common_start else None
    )
    validation_end = (
        pd.Timestamp(data[symbols[0]].iloc[common_length - 1]["date"]).date().isoformat()
        if symbols and common_length else None
    )
    validation_days = max(0, common_length - common_start)
    rows = []
    for strategy, evaluator in EVALUATORS.items():
        if STRATEGY_LANES.get(strategy) != "rebound":
            continue
        events = []
        length = min((len(data[symbol]) for symbol in symbols), default=0)
        start = max(60, length - backtest_days)
        for index in range(start, max(start, length - 10), signal_interval):
            sliced = {symbol: data[symbol].iloc[: index + 1].copy() for symbol in symbols}
            try:
                vote = evaluator(sliced, pd.DataFrame(), top_n)
            except Exception:
                continue
            for symbol, _ in vote.selections:
                frame = data.get(symbol)
                if frame is None or index + 10 >= len(frame):
                    continue
                entry = float(frame.iloc[index]["close"])
                after5 = float(frame.iloc[index + 5]["close"])
                after10 = float(frame.iloc[index + 10]["close"])
                lows = pd.to_numeric(frame.iloc[index + 1:index + 11].get("low", frame.iloc[index + 1:index + 11]["close"]), errors="coerce")
                if entry > 0 and np.isfinite([entry, after5, after10]).all() and lows.notna().any():
                    events.append((after5 / entry - 1, after10 / entry - 1, float(lows.min() / entry - 1)))
        if not events:
            rows.append({"strategy": strategy, "status": "completed", "passed": False,
                         "validation_level": "未通过", "signal_count": 0,
                         "backtest_start_date": validation_start, "backtest_end_date": validation_end,
                         "days": validation_days, "validation_confidence": validation_confidence(validation_days),
                         "price_adjustment": "分析用前复权（含价格断层检测）",
                         "reason": "历史区间没有可验证的超跌信号"})
            continue
        values = pd.DataFrame(events, columns=["forward_5d_return", "forward_10d_return", "adverse_excursion"])
        wins, losses = values.loc[values.forward_10d_return > 0, "forward_10d_return"], values.loc[values.forward_10d_return < 0, "forward_10d_return"]
        payoff = float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else np.nan
        metrics = {
            "signal_count": len(values), "forward_5d_mean": float(values.forward_5d_return.mean()),
            "forward_10d_mean": float(values.forward_10d_return.mean()),
            "forward_10d_win_rate": float((values.forward_10d_return > 0).mean()),
            "average_adverse_excursion": float(values.adverse_excursion.mean()), "payoff_ratio": payoff,
        }
        basic_edge = metrics["signal_count"] >= 5 and metrics["forward_10d_mean"] > 0 and metrics["forward_10d_win_rate"] >= 0.50 and metrics["average_adverse_excursion"] >= -0.08
        qualified = metrics["signal_count"] >= 10 and metrics["forward_10d_mean"] >= .01 and metrics["forward_10d_win_rate"] >= .55 and metrics["average_adverse_excursion"] >= -.08 and (pd.isna(payoff) or payoff >= 1.0)
        strong = metrics["signal_count"] >= 20 and metrics["forward_10d_mean"] >= .02 and metrics["forward_10d_win_rate"] >= .60 and metrics["average_adverse_excursion"] >= -.06 and (pd.isna(payoff) or payoff >= 1.5)
        level = "强" if strong else ("合格" if qualified else ("弱优势" if basic_edge else "未通过"))
        passed = level in ("强", "合格")
        failures = []
        if metrics["signal_count"] < 5: failures.append("有效信号少于5次")
        if metrics["forward_10d_mean"] <= 0: failures.append("信号后10日平均收益不为正")
        if metrics["forward_10d_win_rate"] < .50: failures.append("信号后10日上涨概率低于50%")
        if metrics["average_adverse_excursion"] < -.08: failures.append("平均最大不利波动超过8%")
        explanation = {
            "强": "历史信号在收益、上涨概率、样本量和不利波动方面均表现较强。",
            "合格": "历史信号达到可参与共识的基本强度，但仍需结合止跌确认。",
            "弱优势": "历史上存在轻微统计优势，但强度有限，不能单独作为买入依据。",
            "未通过": "历史信号未体现出足够且稳定的短期优势。",
        }[level]
        rows.append({"strategy": strategy, "status": "completed", "passed": passed,
                     "validation_level": level, "validation_explanation": explanation,
                     "backtest_start_date": validation_start, "backtest_end_date": validation_end,
                     "days": validation_days, "validation_confidence": validation_confidence(validation_days),
                     "price_adjustment": "分析用前复权（含价格断层检测）",
                     "reason": "；".join(failures), **metrics})
    order = {"强": 0, "合格": 1, "弱优势": 2, "未通过": 3}
    result = pd.DataFrame(rows)
    result["_level_order"] = result.validation_level.map(order).fillna(4)
    return result.sort_values(["_level_order", "forward_10d_mean"], ascending=[True, False], na_position="last").drop(columns="_level_order").reset_index(drop=True)
