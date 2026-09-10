"""ETF-specific risk checks that do not attempt to predict future returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    min_recent_amount: float = 20_000_000
    liquidity_drop: float = -0.40
    abnormal_return_1d: float = 0.08
    abnormal_return_5d: float = 0.15
    abnormal_distance_ma20: float = 0.10
    high_premium: float = 0.03


class AnnouncementRiskProvider(Protocol):
    """Stable announcement sources can implement this without changing risk rules."""

    def risk_warnings(self, symbol: str) -> list[str]: ...


TYPE_LABELS = {
    "equity": "普通股票ETF", "qdii_equity": "跨境ETF / QDII", "commodity": "商品ETF",
    "qdii_commodity": "跨境商品ETF / QDII", "bond": "债券ETF", "money": "货币ETF", "unknown": "其他特殊类型",
    "hong_kong_connect": "港股通股票ETF",
}

ANNOUNCEMENT_RISK_KEYWORDS = (
    "溢价风险", "交易价格明显高于基金份额参考净值", "停牌", "异常波动", "风险提示",
)


def _resolved_type(fund_type: str, name: str) -> str:
    key = str(fund_type or "unknown")
    text = str(name or "")
    if key in ("equity", "qdii_equity") and "港股通" in text:
        return "hong_kong_connect"
    return key


def _type_details(fund_type: str, name: str) -> tuple[str, str, str]:
    key = _resolved_type(fund_type, name)
    label = TYPE_LABELS.get(key, "其他特殊类型")
    if key == "hong_kong_connect":
        note = "基金通过港股通投资香港市场，除股票本身波动外，还存在人民币汇率、港股交易制度、交易日差异及港股通机制等额外风险。"
    elif key == "qdii_commodity":
        note = "同时具有跨境和商品属性：可能受海外休市、时差、汇率、场内折溢价、期货价格、合约换月和商品高波动影响。"
    elif key == "qdii_equity":
        note = "海外市场存在休市和时差差异，人民币汇率变化也会影响收益；场内价格还可能出现折溢价。"
    elif key == "commodity":
        note = "底层商品可能受期货价格、合约换月和商品本身高波动影响。"
    elif key == "bond":
        note = "债券ETF主要受利率和信用环境影响，不宜完全套用股票ETF的趋势解释。"
    elif key == "money":
        note = "货币ETF主要用于现金管理，不宜套用股票ETF的趋势和追高判断。"
    elif key == "unknown":
        note = "当前资料无法确认具体结构，参与前应查看基金合同和跟踪标的。"
    else:
        note = "未识别到跨境、商品、债券或货币等特殊结构。"
    return key, label, note


def _premium(row: pd.Series, special: bool, config: RiskConfig) -> tuple[float, str, str, bool]:
    premium = pd.to_numeric(pd.Series([row.get("premium_rate")]), errors="coerce").iloc[0]
    if pd.isna(premium):
        nav = pd.to_numeric(pd.Series([row.get("iopv", row.get("nav"))]), errors="coerce").iloc[0]
        price = pd.to_numeric(pd.Series([row.get("current_price")]), errors="coerce").iloc[0]
        if pd.notna(nav) and nav > 0 and pd.notna(price):
            premium = float(price / nav - 1)
    if pd.isna(premium):
        return np.nan, "数据不可用", "当前本地及公开缓存没有稳定、同一时点的IOPV或净值数据，未对折溢价作猜测。", special
    if premium >= config.high_premium:
        return float(premium), "溢价偏高", f"场内价格相对参考净值溢价约 {premium:.1%}，存在溢价回落风险。", False
    return float(premium), "溢价正常", f"场内价格相对参考净值的偏离约为 {premium:+.1%}。", False


def check_etf_risk(
    source: pd.Series | dict,
    config: RiskConfig | None = None,
    announcement_provider: AnnouncementRiskProvider | None = None,
) -> dict:
    """Return one structured, explainable ETF-specific risk result."""
    config = config or RiskConfig()
    row = source if isinstance(source, pd.Series) else pd.Series(source)
    fund_type, label, special_note = _type_details(row.get("fund_type", "unknown"), row.get("name", ""))
    premium_sensitive = fund_type in ("qdii_equity", "qdii_commodity", "commodity")
    special = premium_sensitive or fund_type == "hong_kong_connect"

    low_amount = pd.notna(row.get("recent_avg_amount")) and row.recent_avg_amount < config.min_recent_amount
    amount_drop = pd.notna(row.get("activity_change_5d")) and row.activity_change_5d <= config.liquidity_drop
    liquidity_weak = bool(low_amount or amount_drop)
    liquidity_status = "流动性偏弱" if liquidity_weak else "流动性正常"
    liquidity_reasons = []
    if low_amount: liquidity_reasons.append(f"最近5日平均成交额低于 {config.min_recent_amount / 1e4:.0f} 万元")
    if amount_drop: liquidity_reasons.append(f"近期成交活跃度下降 {abs(row.activity_change_5d):.1%}")
    if not liquidity_reasons: liquidity_reasons.append("最近成交额和成交活跃度未触发偏弱条件")

    abnormal_flags, position_flags = [], []
    if pd.notna(row.get("return_1d")) and abs(row.return_1d) >= config.abnormal_return_1d: abnormal_flags.append(f"单日涨跌达到 {row.return_1d:+.1%}")
    if pd.notna(row.get("return_5d")) and abs(row.return_5d) >= config.abnormal_return_5d: abnormal_flags.append(f"近5日涨跌达到 {row.return_5d:+.1%}")
    if pd.notna(row.get("distance_ma20")) and abs(row.distance_ma20) >= config.abnormal_distance_ma20: abnormal_flags.append(f"偏离20日均价 {row.distance_ma20:+.1%}")
    if row.get("consecutive_up_days", 0) >= 4 and row.get("return_5d", 0) >= 0.05: position_flags.append(f"连续快速上涨 {int(row.consecutive_up_days)} 日")
    if pd.notna(row.get("return_5d")) and row.return_5d >= 0.05: position_flags.append(f"近5日上涨 {row.return_5d:+.1%}")
    if pd.notna(row.get("distance_ma20")) and row.distance_ma20 >= 0.06: position_flags.append(f"高于20日均价 {row.distance_ma20:+.1%}")
    if pd.notna(row.get("distance_20d_high")) and row.distance_20d_high >= -0.01: position_flags.append("距离近20日最高价不足1%")
    price_flags = abnormal_flags + position_flags
    price_status = "价格波动异常" if abnormal_flags else ("短期位置偏高" if position_flags else "价格未见明显异常")

    premium, premium_status, premium_reason, confirmation_needed = _premium(row, premium_sensitive, config)
    premium_high = premium_status == "溢价偏高"
    severe_price = bool(abnormal_flags)
    unresolved_data = bool(row.get("has_unresolved_data_anomaly", False))
    blocking = liquidity_weak or premium_high
    incomplete = bool(confirmation_needed or unresolved_data)
    level = "高" if blocking or severe_price else ("中" if price_flags or special or fund_type in ("bond", "money", "unknown") else "低")
    structural_level = "评估不完整" if incomplete else ("高" if blocking else ("中" if special or fund_type in ("bond", "money", "unknown") else "低"))
    block_reasons = []
    if liquidity_weak: block_reasons.append("流动性偏弱")
    if premium_high: block_reasons.append(premium_reason)
    if unresolved_data: block_reasons.append("历史行情仍有未确认的价格断层")

    if announcement_provider is None:
        announcement_status, announcement_warnings = "暂无可靠数据", []
    else:
        announcement_warnings = announcement_provider.risk_warnings(str(row.get("symbol", "")))
        announcement_status = "发现风险提示" if announcement_warnings else "未发现风险提示"

    tags = [label, liquidity_status, price_status, premium_status]
    warnings = ([] if fund_type == "equity" else [special_note]) + price_flags
    if special or premium_high:
        warnings.append(premium_reason)
    if unresolved_data:
        warnings.append("历史行情存在尚未确认的极端跳变，相关技术指标可信度受限。")
    warnings.extend(announcement_warnings)
    details = {
        "recent_avg_amount": row.get("recent_avg_amount"), "activity_change_5d": row.get("activity_change_5d"),
        "return_1d": row.get("return_1d"), "return_5d": row.get("return_5d"),
        "distance_ma20": row.get("distance_ma20"), "distance_20d_high": row.get("distance_20d_high"),
        "premium_rate": premium,
    }
    return {
        "risk_level": {"低": "low", "中": "medium", "高": "high"}[level],
        "risk_tags": tags, "warnings": warnings, "details": details,
        "etf_type": label, "special_risk_note": special_note,
        "liquidity_status": liquidity_status, "liquidity_reason": "；".join(liquidity_reasons),
        "price_risk_status": price_status, "price_risk_reason": "；".join(price_flags) if price_flags else "近1日、5日涨跌及均线偏离未触发异常条件。",
        "premium_rate": premium, "premium_status": premium_status, "premium_reason": premium_reason,
        "announcement_status": announcement_status, "announcement_warnings": "；".join(announcement_warnings),
        # etf_risk_level is kept for old CSV/API compatibility; new reports use
        # the unambiguous structural_risk_level field.
        "premium_confirmation_needed": confirmation_needed, "etf_risk_level": level,
        "structural_risk_level": structural_level,
        "structural_risk_reason": "；".join(
            [text for text in (
                "流动性或折溢价触发阻断条件" if blocking else "",
                "跨境或商品品种缺少可靠折溢价数据" if confirmation_needed else "",
                "存在未确认的历史价格断层" if unresolved_data else "",
                special_note if special or fund_type in ("bond", "money", "unknown") else "",
            ) if text]
        ) or "流动性和产品结构未发现明显额外风险。",
        "risk_force_observe": severe_price,
        "risk_blocking": blocking or unresolved_data, "risk_block_reason": "；".join(block_reasons),
    }


def assess_etf_risks(
    metrics: pd.DataFrame,
    config: RiskConfig | None = None,
    announcement_provider: AnnouncementRiskProvider | None = None,
) -> pd.DataFrame:
    """Classify ETF structure and add liquidity, price, premium, and announcement checks."""
    config = config or RiskConfig()
    rows = []
    for _, source in metrics.iterrows():
        values = source.to_dict()
        risk_result = check_etf_risk(source, config, announcement_provider)
        values.update(risk_result)
        values["risk_result"] = risk_result
        rows.append(values)
    return pd.DataFrame(rows)
