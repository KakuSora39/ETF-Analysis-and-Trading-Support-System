"""Separate strategy votes into trend, rebound, and defensive opportunity lanes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .consensus import ConsensusResult


STRATEGY_LANES = {
    "momentum_rotation": "trend", "dual_momentum": "trend", "momentum_dual": "trend", "median_momentum": "trend",
    "momentum_ma_etf": "trend", "momentum_ma_filter": "trend", "momentum_vol_filter": "trend",
    "multi_period_momentum": "trend", "composite_momentum": "trend", "dual_ma_crossover": "trend",
    "macd_trend_rotation": "trend", "rsi_trend_rotation": "trend", "adx_trend_rotation": "trend",
    "bollinger_rotation": "trend", "donchian_breakout": "trend", "relative_strength": "trend",
    "volume_price": "trend", "vol_price_momentum": "trend",
    "bollinger_reversion": "rebound", "contrarian_reversion": "rebound", "mean_reversion": "rebound",
    "mean_reversion_rotation": "rebound",
    "sharpe_ranking": "defense", "sortino_ranking": "defense", "low_vol_rotation": "defense", "tail_risk": "defense",
}

# Closely related implementations contribute only one support group per ETF.
STRATEGY_GROUPS = {
    "momentum_rotation": "基础动量", "dual_momentum": "基础动量", "momentum_dual": "基础动量", "median_momentum": "基础动量",
    "momentum_ma_filter": "市场趋势过滤", "momentum_vol_filter": "市场趋势过滤",
    "momentum_ma_etf": "个体趋势过滤", "multi_period_momentum": "个体趋势过滤", "dual_ma_crossover": "个体趋势过滤",
    "composite_momentum": "复合趋势", "macd_trend_rotation": "MACD趋势", "rsi_trend_rotation": "RSI趋势",
    "adx_trend_rotation": "趋势强度", "bollinger_rotation": "通道强度", "donchian_breakout": "通道强度",
    "relative_strength": "相对强弱", "volume_price": "量价", "vol_price_momentum": "量价",
    "bollinger_reversion": "布林超跌", "contrarian_reversion": "跌幅反转", "mean_reversion": "均线偏离",
    "mean_reversion_rotation": "多指标超跌", "sharpe_ranking": "风险收益", "sortino_ranking": "风险收益",
    "low_vol_rotation": "低波动", "tail_risk": "尾部防守",
}

LANE_LABELS = {"trend": "趋势机会", "rebound": "超跌反弹", "defense": "防守机会"}


def lane_denominator_counts(status: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return explicit strategy and de-duplicated family counts per lane."""
    active = status[status.status == "voted"]
    result = {}
    for lane in LANE_LABELS:
        strategies = [s for s in active.strategy if STRATEGY_LANES.get(s) == lane]
        result[lane] = {
            "strategies": len(set(strategies)),
            "families": len({STRATEGY_GROUPS[s] for s in strategies}),
        }
    return result


def _support_table(votes: pd.DataFrame, status: pd.DataFrame, prefix: str = "") -> tuple[pd.DataFrame, dict]:
    denominators = {
        lane: len({STRATEGY_GROUPS[s] for s in status.loc[status.status == "voted", "strategy"] if STRATEGY_LANES.get(s) == lane})
        for lane in LANE_LABELS
    }
    data = votes[votes.strategy.isin(STRATEGY_LANES)].copy()
    if data.empty:
        return pd.DataFrame(columns=["symbol"]), denominators
    data["lane"] = data.strategy.map(STRATEGY_LANES)
    data["support_group"] = data.strategy.map(STRATEGY_GROUPS)
    grouped = data.groupby(["symbol", "lane"]).agg(
        support=("support_group", "nunique"),
        strategy_support=("strategy", "nunique"),
        points=("points", "sum"),
        strategies=("strategy", lambda values: ";".join(sorted(set(values)))),
    ).reset_index()
    pieces = []
    for lane in LANE_LABELS:
        part = grouped[grouped.lane == lane].drop(columns="lane").rename(columns={
            "support": f"{prefix}{lane}_support", "points": f"{prefix}{lane}_points",
            "strategy_support": f"{prefix}{lane}_strategy_support",
            "strategies": f"{prefix}{lane}_strategies",
        })
        pieces.append(part.set_index("symbol"))
    result = pd.concat(pieces, axis=1).reset_index()
    for lane in LANE_LABELS:
        support = f"{prefix}{lane}_support"
        result[support] = result.get(support, 0).fillna(0).astype(int)
        strategy_support = f"{prefix}{lane}_strategy_support"
        result[strategy_support] = result.get(strategy_support, 0).fillna(0).astype(int)
        result[f"{prefix}{lane}_ratio"] = result[support] / max(1, denominators[lane])
        result[f"{prefix}{lane}_strategies"] = result.get(f"{prefix}{lane}_strategies", "").fillna("")
    return result, denominators


def build_lane_shortlist(consensus: ConsensusResult, candidates: pd.DataFrame, top_n: int = 10):
    """Build an independent top-N for every lane before historical validation."""
    support, denominators = _support_table(consensus.votes, consensus.status)
    detailed_denominators = lane_denominator_counts(consensus.status)
    base = candidates.merge(support, on="symbol", how="left")
    for lane, denominator in denominators.items():
        base[f"{lane}_active_denominator"] = denominator
        base[f"{lane}_active_family_denominator"] = detailed_denominators[lane]["families"]
        base[f"{lane}_active_strategy_denominator"] = detailed_denominators[lane]["strategies"]
    selected = []
    lane_symbols = {}
    for lane in LANE_LABELS:
        support_col, points_col = f"{lane}_support", f"{lane}_points"
        for column, default in ((support_col, 0), (points_col, 0), (f"{lane}_strategies", ""), (f"{lane}_ratio", 0.0)):
            if column not in base:
                base[column] = default
            else:
                base[column] = base[column].fillna(default)
        base[support_col] = base[support_col].astype(int)
        lane_rows = base[base[support_col] > 0].sort_values(
            [support_col, points_col, "avg_amount", "symbol"], ascending=[False, False, False, True]
        ).head(top_n).copy()
        lane_rows["shortlist_lane"] = lane
        lane_symbols[lane] = set(lane_rows.symbol)
        selected.append(lane_rows)
    union_symbols = list(dict.fromkeys(pd.concat(selected).symbol.tolist())) if selected else []
    union = base[base.symbol.isin(union_symbols)].copy()
    for lane in LANE_LABELS:
        union[f"in_{lane}_shortlist"] = union.symbol.isin(lane_symbols.get(lane, set()))
    ratios = union[[f"{lane}_ratio" for lane in LANE_LABELS]].fillna(0)
    union["opportunity_type"] = ratios.idxmax(axis=1).str.replace("_ratio", "", regex=False).map(LANE_LABELS)
    return union.reset_index(drop=True), denominators


def validated_lane_support(
    shortlist: pd.DataFrame,
    consensus: ConsensusResult,
    valid_strategy_names: set[str],
) -> tuple[pd.DataFrame, dict]:
    votes = consensus.votes[
        consensus.votes.strategy.isin(valid_strategy_names) & consensus.votes.symbol.isin(shortlist.symbol)
    ].copy()
    status = consensus.status.copy()
    status["status"] = np.where(status.strategy.isin(valid_strategy_names), "voted", "invalid")
    detailed_denominators = lane_denominator_counts(status)
    support, denominators = _support_table(votes, status, prefix="validated_")
    result = shortlist.merge(support, on="symbol", how="left")
    for lane in LANE_LABELS:
        for suffix, default in (("support", 0), ("ratio", 0.0), ("strategies", "")):
            column = f"validated_{lane}_{suffix}"
            if column not in result:
                result[column] = default
            else:
                result[column] = result[column].fillna(default)
        result[f"validated_{lane}_support"] = result[f"validated_{lane}_support"].astype(int)
        result[f"{lane}_validated_denominator"] = denominators[lane]
        result[f"{lane}_validated_family_denominator"] = detailed_denominators[lane]["families"]
        result[f"{lane}_validated_strategy_denominator"] = detailed_denominators[lane]["strategies"]
    return result, denominators


def attach_weak_rebound_support(
    frame: pd.DataFrame,
    consensus: ConsensusResult,
    weak_strategy_names: set[str],
) -> pd.DataFrame:
    """Keep weak-edge rebound strategies as a radar without making them valid votes."""
    result = frame.copy()
    weak_names = {name for name in weak_strategy_names if STRATEGY_LANES.get(name) == "rebound"}
    votes = consensus.votes[
        consensus.votes.strategy.isin(weak_names) & consensus.votes.symbol.isin(result.symbol)
    ].copy()
    denominator = len({STRATEGY_GROUPS[name] for name in weak_names})
    strategy_denominator = len(weak_names)
    if votes.empty:
        result["weak_rebound_support"] = 0
        result["weak_rebound_strategy_support"] = 0
        result["weak_rebound_strategies"] = ""
    else:
        votes["support_group"] = votes.strategy.map(STRATEGY_GROUPS)
        grouped = votes.groupby("symbol").agg(
            weak_rebound_support=("support_group", "nunique"),
            weak_rebound_strategy_support=("strategy", "nunique"),
            weak_rebound_strategies=("strategy", lambda values: ";".join(sorted(set(values)))),
        )
        result = result.merge(grouped, left_on="symbol", right_index=True, how="left")
        result["weak_rebound_support"] = result["weak_rebound_support"].fillna(0).astype(int)
        result["weak_rebound_strategy_support"] = result["weak_rebound_strategy_support"].fillna(0).astype(int)
        result["weak_rebound_strategies"] = result["weak_rebound_strategies"].fillna("")
    result["weak_rebound_family_denominator"] = denominator
    result["weak_rebound_strategy_denominator"] = strategy_denominator
    return result


def attach_lane_validation_metrics(
    frame: pd.DataFrame,
    consensus: ConsensusResult,
    backtests: pd.DataFrame,
    rebound_validation: pd.DataFrame,
) -> pd.DataFrame:
    """Attach lane-appropriate historical measures for the currently supporting strategies."""
    result = frame.copy()
    long_passed = backtests[backtests.get("passed", False) == True].copy()  # noqa: E712
    sharpe = long_passed.set_index("strategy")["sharpe"].to_dict() if not long_passed.empty else {}
    rebound_passed = rebound_validation[rebound_validation.get("passed", False) == True].copy()  # noqa: E712
    forward10 = rebound_passed.set_index("strategy")["forward_10d_mean"].to_dict() if not rebound_passed.empty else {}
    win10 = rebound_passed.set_index("strategy")["forward_10d_win_rate"].to_dict() if not rebound_passed.empty else {}
    votes = consensus.votes[consensus.votes.symbol.isin(result.symbol)].copy()
    for lane in ("trend", "defense"):
        selected = votes[votes.strategy.map(STRATEGY_LANES) == lane].copy()
        selected["metric"] = selected.strategy.map(sharpe)
        means = selected.dropna(subset=["metric"]).groupby("symbol").metric.mean()
        result[f"{lane}_mean_sharpe"] = result.symbol.map(means)
    rebound = votes[votes.strategy.map(STRATEGY_LANES) == "rebound"].copy()
    rebound["forward10"] = rebound.strategy.map(forward10)
    rebound["win10"] = rebound.strategy.map(win10)
    result["rebound_forward_10d_mean"] = result.symbol.map(rebound.dropna(subset=["forward10"]).groupby("symbol").forward10.mean())
    result["rebound_forward_10d_win_rate"] = result.symbol.map(rebound.dropna(subset=["win10"]).groupby("symbol").win10.mean())
    return result
