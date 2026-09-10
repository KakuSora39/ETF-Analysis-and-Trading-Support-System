"""Explainable market environment controller based on the existing benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketRegime:
    code: str
    label: str
    close: float
    ma60: float
    ma250: float
    distance_ma60: float
    distance_ma250: float
    volatility_20d: float
    explanation: str
    cross_section_count: int = 0
    return_5d_std: float = np.nan
    return_20d_std: float = np.nan
    top_bottom_20d_spread: float = np.nan
    advancing_ratio_20d: float = np.nan
    dispersion_signals: int = 0
    broad_code: str = "range"
    broad_label: str = "震荡"
    structure_code: str = "normal"
    structure_label: str = "正常分化"
    broad_explanation: str = ""
    structure_explanation: str = ""
    benchmark_return_20d: float = np.nan

    @property
    def rebound_confirmation_required(self) -> int:
        return 4 if self.code == "weak" else 3

    @property
    def trend_support_required(self) -> int:
        return 3 if self.code == "weak" else 2

    @property
    def defense_support_required(self) -> int:
        return 2 if self.code == "strong" else 1


def _market_structure(cross: dict) -> tuple[str, str, str]:
    count = cross["cross_section_count"]
    ratio = cross["advancing_ratio_20d"]
    if count < 20 or pd.isna(ratio):
        return "normal", "正常分化", "可比较ETF不足20只，市场结构暂按正常分化处理。"
    if ratio >= .70:
        return "broad_up", "普遍走强", f"近20日上涨ETF占比为{ratio:.1%}，多数ETF同步走强。"
    if ratio <= .30:
        return "broad_down", "普遍走弱", f"近20日上涨ETF占比仅{ratio:.1%}，多数ETF同步走弱。"
    if cross["dispersion_signals"] >= 2:
        return "dispersed", "明显分化", (
            f"近5日收益离散度{cross['return_5d_std']:.1%}，近20日收益离散度{cross['return_20d_std']:.1%}，"
            f"强弱两端平均收益差{cross['top_bottom_20d_spread']:.1%}；不同主题表现差异较大。"
        )
    return "normal", "正常分化", f"近20日上涨ETF占比为{ratio:.1%}，横截面差异未达到明显分化门槛。"


def _cross_section_metrics(candidate_market: dict[str, pd.DataFrame] | None) -> dict:
    returns_5d, returns_20d = [], []
    for frame in (candidate_market or {}).values():
        close = pd.to_numeric(frame.get("close"), errors="coerce").dropna()
        if len(close) >= 6 and close.iloc[-6] > 0:
            returns_5d.append(float(close.iloc[-1] / close.iloc[-6] - 1))
        if len(close) >= 21 and close.iloc[-21] > 0:
            returns_20d.append(float(close.iloc[-1] / close.iloc[-21] - 1))
    five = pd.Series(returns_5d, dtype=float)
    twenty = pd.Series(returns_20d, dtype=float)
    count = len(twenty)
    if count:
        edge_count = min(10, max(1, count // 4))
        ordered = twenty.sort_values()
        spread = float(ordered.tail(edge_count).mean() - ordered.head(edge_count).mean())
        advancing_ratio = float((twenty > 0).mean())
    else:
        spread = advancing_ratio = np.nan
    std5 = float(five.std(ddof=0)) if len(five) else np.nan
    std20 = float(twenty.std(ddof=0)) if count else np.nan
    signals = sum((
        bool(pd.notna(std5) and std5 >= 0.03),
        bool(pd.notna(std20) and std20 >= 0.07),
        bool(pd.notna(spread) and spread >= 0.15),
    ))
    return {
        "cross_section_count": count,
        "return_5d_std": std5,
        "return_20d_std": std20,
        "top_bottom_20d_spread": spread,
        "advancing_ratio_20d": advancing_ratio,
        "dispersion_signals": signals,
    }


def detect_market_regime(
    benchmark: pd.DataFrame,
    candidate_market: dict[str, pd.DataFrame] | None = None,
) -> MarketRegime:
    """Return independent benchmark direction and cross-sectional structure."""
    cross = _cross_section_metrics(candidate_market)
    structure_code, structure_label, structure_explanation = _market_structure(cross)
    close = pd.to_numeric(benchmark.get("close"), errors="coerce").dropna()
    if close.empty:
        broad = "没有可用的沪深300基准行情；大盘状态暂按震荡处理。"
        return MarketRegime(
            "range", f"震荡 · {structure_label}", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            f"{broad}{structure_explanation}", **cross, broad_code="range", broad_label="震荡",
            structure_code=structure_code, structure_label=structure_label,
            broad_explanation=broad, structure_explanation=structure_explanation,
        )
    current = float(close.iloc[-1])
    ma60 = float(close.iloc[-min(60, len(close)):].mean())
    ma250 = float(close.iloc[-min(250, len(close)):].mean())
    returns = close.iloc[-21:].pct_change(fill_method=None).dropna()
    volatility = float(returns.std() * math.sqrt(252))
    d60, d250 = current / ma60 - 1, current / ma250 - 1
    return20 = float(current / close.iloc[-21] - 1) if len(close) >= 21 and close.iloc[-21] > 0 else np.nan
    ma60_rising = len(close) >= 80 and ma60 > float(close.iloc[-80:-20].mean())
    if len(close) < 250:
        code, broad_label = "range", "震荡"
        broad_explanation = f"沪深300有效历史只有{len(close)}个交易日，不足250日；大盘状态暂按震荡处理。"
    elif d60 > 0.03 and d250 > 0 and ma60_rising and volatility < 0.30:
        code, broad_label = "strong", "强势"
        broad_explanation = "沪深300位于60日和250日均线上方，60日均线向上，近期波动未明显升高。"
    elif (d60 < -0.03 and d250 < 0) or volatility >= 0.30:
        code, broad_label = "weak", "弱势"
        broad_explanation = "沪深300位于关键均线下方，或近期波动明显偏高；趋势与反弹机会都需要更严格确认。"
    elif d60 >= 0 and (pd.isna(return20) or return20 >= 0):
        code, broad_label = "range_strong", "震荡偏强"
        broad_explanation = "沪深300尚未形成强趋势，但位于60日均线附近上方，近期表现偏强。"
    elif d60 < 0 and pd.notna(return20) and return20 < 0:
        code, broad_label = "range_weak", "震荡偏弱"
        broad_explanation = "沪深300尚未进入明确弱势，但位于60日均线附近下方，近期表现偏弱。"
    else:
        code, broad_label = "range", "震荡"
        broad_explanation = "沪深300围绕关键均线波动，近期没有形成明确方向。"
    label = f"{broad_label} · {structure_label}"
    explanation = f"{broad_explanation}{structure_explanation}"
    return MarketRegime(
        code, label, current, ma60, ma250, d60, d250, volatility, explanation, **cross,
        broad_code=code, broad_label=broad_label, structure_code=structure_code,
        structure_label=structure_label, broad_explanation=broad_explanation,
        structure_explanation=structure_explanation, benchmark_return_20d=return20,
    )
