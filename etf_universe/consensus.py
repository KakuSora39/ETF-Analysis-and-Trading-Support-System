"""Run comparable current-signal strategies and aggregate their ETF votes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import math

import numpy as np
import pandas as pd


@dataclass
class StrategyVote:
    strategy: str
    family: str
    status: str
    selections: list[tuple[str, float]]
    detail: str = ""
    eligible_count: int = 0


@dataclass
class ConsensusResult:
    status: pd.DataFrame
    votes: pd.DataFrame
    ranking: pd.DataFrame
    focus: pd.DataFrame


# Every current strategy directory is classified.  Entries with no evaluator
# still appear in strategy_status.csv with an explicit reason.
NON_COMPARABLE = {
    "alpha_etf_rotation": "需要已训练模型，模型的固定特征与201只候选池不兼容",
    "asset_allocation": "固定资产角色配置，不是全市场横截面选股策略",
    "combined": "组合了动量与配对交易，输出不能作为单ETF独立票",
    "cross_border": "只适用于预设跨境ETF池",
    "dca_timing": "定投择时策略，不输出横截面ETF排名",
    "gold_safe_haven": "黄金/风险资产固定角色切换，不适用于任意候选池",
    "hs300_ma_timing": "沪深300市场择时器，不输出ETF排名",
    "industry_momentum": "依赖预设行业分类，当前基础资料没有可靠行业标签",
    "lstm_etf_rotation": "需要针对固定ETF池训练的LSTM模型",
    "market_breadth": "市场宽度择时器，不输出ETF排名",
    "neural_momentum": "需要针对固定ETF池训练的神经网络模型",
    "pair_trading": "输出ETF配对与价差仓位，不是单ETF排名",
    "risk_switch_momentum": "使用固定风险资产角色，不能直接扩展到任意候选池",
    "risk_timing": "风险开关，不输出ETF排名",
    "sector_rotation": "依赖预设板块分类，当前基础资料没有可靠板块标签",
    "spread_reversion": "输出配对价差仓位，不是单ETF排名",
}

CONTROLLERS = {
    "adaptive_rotation": "已升级为市场环境控制器，不再作为普通策略投票",
}


def _valid(data: dict[str, pd.DataFrame], lookback: int, columns=("close",)):
    result = {}
    for symbol, frame in data.items():
        if len(frame) < lookback + 1:
            continue
        tail = frame.iloc[-(lookback + 1):]
        if all(column in tail and np.isfinite(pd.to_numeric(tail[column], errors="coerce")).all()
               for column in columns):
            result[symbol] = frame
    return result


def _momentum(data, window=20):
    values = {}
    for symbol, frame in _valid(data, window).items():
        values[symbol] = frame.iloc[-1]["close"] / frame.iloc[-window - 1]["close"] - 1
    return pd.Series(values, dtype=float)


def _returns(frame, window=20):
    return frame.iloc[-window - 1:]["close"].pct_change(fill_method=None).dropna()


def _pick(name, family, scores, top_n, detail="", ascending=False, predicate=None):
    scores = pd.Series(scores, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if predicate is not None:
        scores = scores[predicate(scores)]
    scores = scores.sort_values(ascending=ascending, kind="stable")
    selections = [(str(symbol), float(score)) for symbol, score in scores.head(top_n).items()]
    status = "voted" if selections else "abstained"
    if not selections and not detail:
        detail = "当前没有ETF满足该策略的入场条件"
    return StrategyVote(name, family, status, selections, detail, len(scores))


def _benchmark_ready(benchmark, days):
    return (benchmark is not None and len(benchmark) >= days + 1 and
            np.isfinite(pd.to_numeric(benchmark.iloc[-days - 1:]["close"], errors="coerce")).all())


def _eval_momentum(data, benchmark, top_n):
    return _pick("momentum_rotation", "20日动量", _momentum(data), top_n, "20日收益率降序")


def _eval_dual_momentum(data, benchmark, top_n):
    return _pick("dual_momentum", "20日动量", _momentum(data), top_n,
                 "仅保留绝对动量为正", predicate=lambda s: s > 0)


def _eval_momentum_dual(data, benchmark, top_n):
    return _pick("momentum_dual", "20日动量", _momentum(data), top_n,
                 "仅保留绝对动量为正", predicate=lambda s: s > 0)


def _eval_median(data, benchmark, top_n):
    scores = _momentum(data).sort_values(ascending=False, kind="stable")
    scores = scores.iloc[1:top_n + 1]
    return _pick("median_momentum", "20日动量", scores, top_n, "跳过动量第1名后选择")


def _eval_momentum_ma_etf(data, benchmark, top_n):
    scores = _momentum(data)
    keep = {}
    for symbol, frame in _valid(data, 60).items():
        if frame.iloc[-1]["close"] > frame.iloc[-60:]["close"].mean() and symbol in scores:
            keep[symbol] = scores[symbol]
    return _pick("momentum_ma_etf", "趋势过滤动量", keep, top_n, "价格高于自身MA60后按20日动量")


def _eval_momentum_ma_filter(data, benchmark, top_n):
    if not _benchmark_ready(benchmark, 250):
        return StrategyVote("momentum_ma_filter", "市场择时动量", "abstained", [], "沪深300历史不足250日")
    close = benchmark.iloc[-1]["close"]
    ma = benchmark.iloc[-250:]["close"].mean()
    if close <= ma:
        return StrategyVote("momentum_ma_filter", "市场择时动量", "abstained", [], "沪深300不在MA250上方")
    return _pick("momentum_ma_filter", "市场择时动量", _momentum(data), top_n, "沪深300在MA250上方")


def _eval_momentum_vol_filter(data, benchmark, top_n):
    if not _benchmark_ready(benchmark, 20):
        return StrategyVote("momentum_vol_filter", "市场择时动量", "abstained", [], "沪深300历史不足20日")
    vol = benchmark.iloc[-21:]["close"].pct_change(fill_method=None).dropna().std() * math.sqrt(252)
    if not np.isfinite(vol) or vol >= 0.30:
        return StrategyVote("momentum_vol_filter", "市场择时动量", "abstained", [], f"沪深300年化波动率{vol:.1%}未低于30%")
    return _pick("momentum_vol_filter", "市场择时动量", _momentum(data), top_n,
                 f"沪深30020日年化波动率{vol:.1%}")


def _eval_multi_period(data, benchmark, top_n):
    scores = {}
    for symbol, frame in _valid(data, 20).items():
        c = frame["close"]
        r5, r10, r20 = c.iloc[-1] / c.iloc[-6] - 1, c.iloc[-1] / c.iloc[-11] - 1, c.iloc[-1] / c.iloc[-21] - 1
        if r5 > 0 and r10 > 0 and r20 > 0:
            scores[symbol] = .2 * r5 + .3 * r10 + .5 * r20
    return _pick("multi_period_momentum", "多周期动量", scores, top_n, "5/10/20日动量同时为正")


def _zscore(series):
    series = pd.Series(series, dtype=float)
    sd = series.std()
    return (series - series.mean()) / sd if len(series.dropna()) >= 2 and sd > 0 else series * 0


def _eval_composite(data, benchmark, top_n):
    raw = {"trend": {}, "sharpe": {}, "quality": {}, "volume": {}}
    for symbol, frame in _valid(data, 60, ("close", "volume")).items():
        c, v = frame["close"], frame["volume"]
        raw["trend"][symbol] = .2 * (c.iloc[-1] / c.iloc[-6] - 1) + .5 * (c.iloc[-1] / c.iloc[-21] - 1) + .3 * (c.iloc[-1] / c.iloc[-61] - 1)
        ret = c.iloc[-21:].pct_change(fill_method=None).dropna()
        raw["sharpe"][symbol] = ret.mean() / ret.std() * math.sqrt(252) if ret.std() > 0 else 0
        raw["quality"][symbol] = np.mean([c.iloc[-1] / c.iloc[-w:].mean() - 1 for w in (5, 20, 60)])
        ratio = v.iloc[-1] / v.iloc[-20:].mean() if v.iloc[-20:].mean() > 0 else 1
        direction = np.sign(c.iloc[-1] / c.iloc[-2] - 1)
        raw["volume"][symbol] = direction * (ratio - 1)
    scores = .40 * _zscore(raw["trend"]) + .25 * _zscore(raw["sharpe"]) + .20 * _zscore(raw["quality"]) + .15 * _zscore(raw["volume"])
    return _pick("composite_momentum", "多因子趋势", scores, top_n, "趋势/夏普/均线质量/量价复合分")


def _eval_sharpe(data, benchmark, top_n):
    scores = {}
    for symbol, frame in _valid(data, 20).items():
        ret = _returns(frame)
        scores[symbol] = ret.mean() / ret.std() * math.sqrt(252) if ret.std() > 0 else 0
    return _pick("sharpe_ranking", "风险调整收益", scores, top_n, "20日年化夏普")


def _eval_sortino(data, benchmark, top_n):
    scores = {}
    for symbol, frame in _valid(data, 20).items():
        ret = _returns(frame)
        downside = ret[ret < 0].std()
        scores[symbol] = ret.mean() / downside * math.sqrt(252) if downside > 0 else (999.0 if ret.mean() > 0 else 0.0)
    return _pick("sortino_ranking", "风险调整收益", scores, top_n, "20日年化Sortino")


def _eval_low_vol(data, benchmark, top_n):
    scores = {}
    for symbol, frame in _valid(data, 20).items():
        scores[symbol] = _returns(frame).std() * math.sqrt(252)
    return _pick("low_vol_rotation", "低波动", scores, top_n, "20日年化波动率升序", ascending=True)


def _eval_dual_ma(data, benchmark, top_n):
    scores = {}
    for symbol, frame in _valid(data, 60).items():
        fast, slow = frame.iloc[-10:]["close"].mean(), frame.iloc[-60:]["close"].mean()
        if fast > slow:
            scores[symbol] = fast / slow - 1
    return _pick("dual_ma_crossover", "趋势过滤", scores, top_n, "MA10高于MA60，按均线距离")


def _eval_macd(data, benchmark, top_n):
    scores = {}
    for symbol, frame in _valid(data, 40).items():
        c = frame["close"]
        fast = c.ewm(span=12, adjust=False).mean()
        slow = c.ewm(span=26, adjust=False).mean()
        macd = fast - slow
        signal = macd.ewm(span=9, adjust=False).mean()
        if macd.iloc[-1] > 0:
            scores[symbol] = (macd.iloc[-1] / c.iloc[-1]) + (macd.iloc[-1] - signal.iloc[-1]) / c.iloc[-1]
    return _pick("macd_trend_rotation", "趋势指标", scores, top_n, "MACD为正，按MACD与柱线强度")


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _eval_rsi(data, benchmark, top_n):
    scores = {}
    for symbol, frame in _valid(data, 25).items():
        rsi = _rsi(frame["close"])
        value = rsi.iloc[-1]
        if np.isfinite(value) and 50 < value < 80:
            scores[symbol] = value + (value - rsi.iloc[-6])
    return _pick("rsi_trend_rotation", "趋势指标", scores, top_n, "RSI处于多头区间并考虑5日斜率")


def _eval_adx(data, benchmark, top_n):
    scores = {}
    for symbol, frame in _valid(data, 35, ("high", "low", "close")).items():
        high, low, close = frame["high"], frame["low"], frame["close"]
        up, down = high.diff(), -low.diff()
        plus_dm = up.where((up > down) & (up > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)
        tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr
        dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        if np.isfinite(adx) and adx >= 20 and plus_di.iloc[-1] > minus_di.iloc[-1]:
            scores[symbol] = adx + .3 * (plus_di.iloc[-1] - minus_di.iloc[-1]) + 20 * (close.iloc[-1] / close.iloc[-6] - 1)
    return _pick("adx_trend_rotation", "趋势强度", scores, top_n, "ADX>=20且DI+>DI-")


def _eval_bollinger(data, benchmark, top_n):
    scores = {}
    for symbol, frame in _valid(data, 30).items():
        c = frame.iloc[-30:]["close"]
        ma, sd = c.mean(), c.std()
        pct_b = (c.iloc[-1] - (ma - 1.5*sd)) / (3*sd) if sd > 0 else .5
        if pct_b < .2:
            scores[symbol] = .2 - pct_b
    return _pick("bollinger_reversion", "均值回归", scores, top_n, "30日布林带%B低于0.2")


def _eval_bollinger_rotation(data, benchmark, top_n):
    scores = {}
    for symbol, frame in _valid(data, 20).items():
        c = frame.iloc[-20:]["close"]
        sd = c.std()
        scores[symbol] = (c.iloc[-1]-(c.mean()-2*sd))/(4*sd) if sd > 0 else 0.5
    return _pick("bollinger_rotation", "布林强度", scores, top_n, "20日布林带位置由高到低")


def _eval_contrarian(data, benchmark, top_n):
    scores = _momentum(data)
    scores = scores[scores < -.02]
    return _pick("contrarian_reversion", "均值回归", -scores, top_n, "20日跌幅低于-2%，越弱分越高")


def _eval_mean_reversion(data, benchmark, top_n):
    scores = {}
    for symbol, frame in _valid(data, 30, ("close", "volume")).items():
        c = frame["close"]
        rsi = _rsi(c).iloc[-1]
        tail = c.iloc[-20:]
        sd = tail.std()
        pct_b = (c.iloc[-1] - (tail.mean()-2*sd))/(4*sd) if sd > 0 else .5
        if np.isfinite(rsi) and rsi < 35 and pct_b < .2:
            scores[symbol] = (35-rsi)/35 + (.2-pct_b)
    return _pick("mean_reversion_rotation", "均值回归", scores, top_n, "RSI<35且布林%B<0.2")


def _eval_simple_mean_reversion(data, benchmark, top_n):
    scores = {}
    for symbol, frame in _valid(data, 20).items():
        ma = frame.iloc[-20:]["close"].mean()
        divergence = frame.iloc[-1]["close"]/ma-1
        scores[symbol] = -divergence
    return _pick("mean_reversion", "均值回归", scores, top_n, "价格相对MA20偏离越低越优先")


def _eval_adaptive(data, benchmark, top_n):
    if not _benchmark_ready(benchmark, 60):
        return StrategyVote("adaptive_rotation", "市场自适应", "abstained", [], "沪深300历史不足60日")
    ratio = benchmark.iloc[-1]["close"]/benchmark.iloc[-60:]["close"].mean()-1
    if ratio > .03:
        return _pick("adaptive_rotation", "市场自适应", _momentum(data), top_n,
                     f"牛市模式：沪深300高于MA60 {ratio:.1%}", predicate=lambda s: s > 0)
    if ratio < -.03:
        return StrategyVote("adaptive_rotation", "市场自适应", "abstained", [],
                            f"熊市模式：沪深300低于MA60 {ratio:.1%}")
    scores = {}
    for symbol, frame in _valid(data, 20, ("close", "volume")).items():
        c = frame["close"]
        rsi = _rsi(c).iloc[-1]
        tail = c.iloc[-20:]
        sd = tail.std()
        pct_b = (c.iloc[-1]-(tail.mean()-2*sd))/(4*sd) if sd > 0 else .5
        if np.isfinite(rsi) and rsi < 35 and pct_b < 0:
            scores[symbol] = (35-rsi)/35-pct_b
    return _pick("adaptive_rotation", "市场自适应", scores, top_n,
                 f"震荡模式：沪深300相对MA60 {ratio:.1%}，寻找超卖")


def _eval_donchian(data, benchmark, top_n):
    scores = {}
    for symbol, frame in _valid(data, 60, ("high", "low", "close")).items():
        upper, lower, close = frame.iloc[-60:]["high"].max(), frame.iloc[-60:]["low"].min(), frame.iloc[-1]["close"]
        scores[symbol] = (close-lower)/(upper-lower) if upper > lower else 0
    return _pick("donchian_breakout", "突破", scores, top_n, "60日唐奇安通道位置")


def _eval_relative_strength(data, benchmark, top_n):
    valid = _valid(data, 20)
    if not valid:
        return _pick("relative_strength", "相对强度", {}, top_n)
    returns = pd.DataFrame({s: f.iloc[-21:]["close"].pct_change(fill_method=None).to_numpy() for s, f in valid.items()})
    mean = returns.mean(axis=1)
    persistence = returns.gt(mean, axis=0).mean()
    momentum = _momentum(valid)
    scores = momentum[persistence[persistence >= .60].index]
    return _pick("relative_strength", "相对强度", scores, top_n, "近20日跑赢候选池均值的天数>=60%")


def _eval_volume_price(data, benchmark, top_n):
    scores = {}
    mom = _momentum(data)
    for symbol, frame in _valid(data, 20, ("close", "volume")).items():
        ratio = frame.iloc[-5:]["volume"].mean()/frame.iloc[-20:]["volume"].mean()
        if ratio >= 1.20 and symbol in mom:
            scores[symbol] = mom[symbol]
    return _pick("volume_price", "量价", scores, top_n, "5日/20日均量>=1.2后按20日动量")


def _eval_vol_price_momentum(data, benchmark, top_n):
    scores = {}
    for symbol, frame in _valid(data, 20, ("close", "volume")).items():
        mom = frame.iloc[-1]["close"]/frame.iloc[-21]["close"]-1
        ratio = frame.iloc[-1]["volume"]/frame.iloc[-20:]["volume"].mean()
        scores[symbol] = mom * ratio
    return _pick("vol_price_momentum", "量价", scores, top_n, "20日动量乘当日量比")


def _eval_tail_risk(data, benchmark, top_n):
    tail = False
    detail = "无尾部风险，按20日动量"
    if _benchmark_ready(benchmark, 20):
        c = benchmark["close"]
        short_vol = c.iloc[-10:].pct_change(fill_method=None).dropna().std()*math.sqrt(252)
        long_vol = c.iloc[-20:].pct_change(fill_method=None).dropna().std()*math.sqrt(252)
        tail = c.iloc[-1]/c.iloc[-6]-1 < -.03 or short_vol > 1.5*long_vol
    if tail:
        detail = "触发尾部风险，改选20日最低波动"
        scores = {s: _returns(f).std()*math.sqrt(252) for s, f in _valid(data, 20).items()}
        return _pick("tail_risk", "尾部风险切换", scores, top_n, detail, ascending=True)
    return _pick("tail_risk", "尾部风险切换", _momentum(data), top_n, detail)


EVALUATORS: dict[str, Callable] = {
    "momentum_rotation": _eval_momentum,
    "dual_momentum": _eval_dual_momentum,
    "momentum_dual": _eval_momentum_dual,
    "median_momentum": _eval_median,
    "momentum_ma_etf": _eval_momentum_ma_etf,
    "momentum_ma_filter": _eval_momentum_ma_filter,
    "momentum_vol_filter": _eval_momentum_vol_filter,
    "multi_period_momentum": _eval_multi_period,
    "composite_momentum": _eval_composite,
    "sharpe_ranking": _eval_sharpe,
    "sortino_ranking": _eval_sortino,
    "low_vol_rotation": _eval_low_vol,
    "dual_ma_crossover": _eval_dual_ma,
    "macd_trend_rotation": _eval_macd,
    "rsi_trend_rotation": _eval_rsi,
    "adx_trend_rotation": _eval_adx,
    "bollinger_reversion": _eval_bollinger,
    "bollinger_rotation": _eval_bollinger_rotation,
    "contrarian_reversion": _eval_contrarian,
    "mean_reversion": _eval_simple_mean_reversion,
    "mean_reversion_rotation": _eval_mean_reversion,
    "donchian_breakout": _eval_donchian,
    "relative_strength": _eval_relative_strength,
    "volume_price": _eval_volume_price,
    "vol_price_momentum": _eval_vol_price_momentum,
    "tail_risk": _eval_tail_risk,
}

# These directory names are aliases or incomplete variants of an evaluator.
ALIASES = {
    "market_regime_rotation": ("regime", "市场状态控制仓位，不是独立横截面排名"),
}


def _strategy_directories(strategy_root: Path | None):
    if strategy_root is None or not strategy_root.exists():
        return sorted(set(EVALUATORS) | set(CONTROLLERS) | set(NON_COMPARABLE) | set(ALIASES))
    return sorted(p.name for p in strategy_root.iterdir()
                  if p.is_dir() and not p.name.startswith("__") and any(p.glob("*.py")))


def run_consensus(
    data: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    candidates: pd.DataFrame,
    strategy_top_n: int = 3,
    focus_top_n: int = 10,
    strategy_root: Path | None = None,
) -> ConsensusResult:
    if strategy_top_n < 1 or focus_top_n < 1:
        raise ValueError("strategy_top_n and focus_top_n must be positive")
    results = []
    for name in _strategy_directories(strategy_root):
        if name in EVALUATORS:
            try:
                result = EVALUATORS[name](data, benchmark, strategy_top_n)
            except Exception as exc:  # one broken legacy strategy must not erase all other votes
                result = StrategyVote(name, "unknown", "error", [], f"{type(exc).__name__}: {exc}")
        elif name in CONTROLLERS:
            result = StrategyVote(name, "market_controller", "controller", [], CONTROLLERS[name])
        elif name in NON_COMPARABLE:
            result = StrategyVote(name, "not_comparable", "not_applicable", [], NON_COMPARABLE[name])
        elif name in ALIASES:
            family, detail = ALIASES[name]
            result = StrategyVote(name, family, "not_applicable", [], detail)
        else:
            result = StrategyVote(name, "unclassified", "not_applicable", [], "尚未建立全市场当前信号适配器")
        results.append(result)

    status = pd.DataFrame([{
        "strategy": r.strategy, "family": r.family, "status": r.status,
        "selected_count": len(r.selections), "eligible_count": r.eligible_count, "detail": r.detail,
    } for r in results]).sort_values("strategy").reset_index(drop=True)
    vote_rows = []
    for result in results:
        for rank, (symbol, score) in enumerate(result.selections, 1):
            vote_rows.append({"strategy": result.strategy, "family": result.family, "rank": rank,
                              "symbol": symbol, "score": score,
                              "points": strategy_top_n-rank+1})
    votes = pd.DataFrame(vote_rows, columns=["strategy", "family", "rank", "symbol", "score", "points"])

    base_columns = [c for c in ("symbol", "name", "avg_amount", "aum_yuan", "fund_type", "index_id") if c in candidates]
    base = candidates[base_columns].drop_duplicates("symbol").set_index("symbol")
    if not votes.empty and "name" in candidates:
        names = candidates.drop_duplicates("symbol").set_index("symbol")["name"]
        votes.insert(4, "name", votes["symbol"].map(names).fillna(""))
    if votes.empty:
        ranking = base.reset_index()
        for column in ("family_votes", "family_points", "strategy_votes", "strategy_points"):
            ranking[column] = 0
        ranking["avg_rank"] = np.nan
    else:
        raw = votes.groupby("symbol").agg(strategy_votes=("strategy", "nunique"),
                                           strategy_points=("points", "sum"), avg_rank=("rank", "mean"))
        family_best = votes.sort_values("rank").drop_duplicates(["family", "symbol"])
        family = family_best.groupby("symbol").agg(family_votes=("family", "nunique"), family_points=("points", "sum"))
        ranking = base.join(family).join(raw).fillna({"family_votes": 0, "family_points": 0,
                                                      "strategy_votes": 0, "strategy_points": 0}).reset_index()
    for column in ("family_votes", "family_points", "strategy_votes", "strategy_points"):
        ranking[column] = ranking[column].astype(int)
    voted_families = max(1, status.loc[status.status == "voted", "family"].nunique())
    voted_strategies = max(1, int((status.status == "voted").sum()))
    ranking["family_agreement"] = ranking["family_votes"] / voted_families
    ranking["strategy_agreement"] = ranking["strategy_votes"] / voted_strategies
    ranking = ranking.sort_values(["family_votes", "family_points", "strategy_votes", "strategy_points", "avg_rank", "avg_amount", "symbol"],
                                  ascending=[False, False, False, False, True, False, True], na_position="last").reset_index(drop=True)
    ranking.insert(0, "consensus_rank", np.arange(1, len(ranking)+1))
    focus = ranking[ranking["family_votes"] > 0].head(focus_top_n).copy()
    return ConsensusResult(status=status, votes=votes, ranking=ranking, focus=focus)
