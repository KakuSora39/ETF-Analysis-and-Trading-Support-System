"""ETF 历史价格断层检测与分析用复权表构建。

原始 ``etf_daily`` 保持不变，便于审计。分析统一读取
``etf_daily_adjusted``；典型份额拆分按常见比例回溯调整历史 OHLC，
无法解释的极端跳变只隔离异常日，不猜测价格。
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("open", "high", "low", "close")
COMMON_FACTORS = np.array([1 / n for n in range(2, 11)] + [float(n) for n in range(2, 11)])
QUALITY_ALGORITHM_VERSION = 2
EXTERNALLY_CONFIRMED_EVENTS = {
    ("159560", "2026-09-07"): "景顺长城基金关于芯片ETF份额拆分及相关业务安排的公开公告",
}


@dataclass(frozen=True)
class PriceQualitySummary:
    source_rows: int
    anomaly_count: int
    repaired_count: int
    isolated_count: int
    reused: bool = False


def _factor_for_ratio(ratio: float, tolerance: float = 0.12) -> tuple[float | None, float]:
    """Return a likely split factor and the residual market move."""
    if not np.isfinite(ratio) or ratio <= 0:
        return None, np.nan
    residuals = ratio / COMMON_FACTORS - 1.0
    index = int(np.argmin(np.abs(residuals)))
    residual = float(residuals[index])
    return (float(COMMON_FACTORS[index]), residual) if abs(residual) <= tolerance else (None, residual)


def repair_symbol_history(frame: pd.DataFrame, threshold: float = 0.20) -> tuple[pd.DataFrame, list[dict]]:
    """Build a continuous analysis series and return an auditable anomaly list."""
    result = frame.sort_values("date").copy().reset_index(drop=True)
    for column in PRICE_COLUMNS + ("volume",):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["is_adjusted"] = 0
    result["is_isolated"] = 0
    anomalies: list[dict] = []
    # Detect from raw closes.  Applying a later split must not hide an earlier event.
    raw_close = result["close"].copy()
    for index in range(1, len(result)):
        previous, current = raw_close.iloc[index - 1], raw_close.iloc[index]
        if not np.isfinite(previous) or not np.isfinite(current) or previous <= 0:
            continue
        ratio = float(current / previous)
        change = ratio - 1.0
        if abs(change) < threshold:
            continue
        factor, residual = _factor_for_ratio(ratio)
        split_like = factor is not None
        repaired = bool(split_like)
        isolated = bool(not split_like and abs(change) >= 0.35)
        method = "按常见份额调整比例回溯修正历史价格" if repaired else (
            "隔离异常交易日，等待外部数据或公告确认" if isolated else "仅记录，保留原始价格"
        )
        note = (
            f"检测到约 {factor:g} 倍价格基准变化；剔除份额因素后当日变化约 {residual:+.2%}"
            if repaired else "未匹配常见拆分/合并比例，不能自动认定为公司行为"
        )
        source_value = result.iloc[index].get("data_source")
        source_label = str(source_value) if pd.notna(source_value) and str(source_value).strip() else "历史库（未保存逐行来源）"
        symbol = str(result.iloc[index]["symbol"])
        event_date = pd.Timestamp(result.iloc[index]["date"]).date().isoformat()
        confirmation_source = EXTERNALLY_CONFIRMED_EVENTS.get((symbol, event_date), "")
        if confirmation_source:
            confidence, manual_review = "已外部确认", False
        elif repaired:
            confidence, manual_review = "自动规则修正", True
        else:
            confidence, manual_review = "待人工核验", True
        anomalies.append({
            "symbol": symbol, "date": str(result.iloc[index]["date"]),
            "previous_close": float(previous), "current_close": float(current), "daily_change": change,
            "suspected_corporate_action": repaired,
            "source": source_label,
            "repaired": repaired, "repair_method": method, "isolated": isolated,
            "factor": factor, "residual_return": residual if repaired else np.nan, "note": note,
            "repair_confidence": confidence, "confirmation_source": confirmation_source,
            "manual_review_required": manual_review,
        })
        if repaired:
            result.loc[: index - 1, list(PRICE_COLUMNS)] *= factor
            result.loc[: index - 1, "is_adjusted"] = 1
        elif isolated:
            result.loc[index, list(PRICE_COLUMNS)] = np.nan
            result.loc[index, "is_isolated"] = 1
    return result, anomalies


def prepare_adjusted_history(db_path: str | Path, force: bool = False) -> tuple[PriceQualitySummary, pd.DataFrame]:
    """Create or refresh the adjusted analysis table without overwriting raw history."""
    path = Path(db_path)
    with closing(sqlite3.connect(path)) as conn:
        source_rows, source_max, source_close_sum = conn.execute(
            "SELECT COUNT(*), MAX(date), ROUND(SUM(COALESCE(close,0)),6) FROM etf_daily"
        ).fetchone()
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not force and {"etf_daily_adjusted", "etf_price_anomalies", "etf_price_quality_state"} <= tables:
            state_columns = {row[1] for row in conn.execute("PRAGMA table_info(etf_price_quality_state)")}
            if "source_close_sum" not in state_columns:
                conn.execute("ALTER TABLE etf_price_quality_state ADD COLUMN source_close_sum REAL")
            if "algorithm_version" not in state_columns:
                conn.execute("ALTER TABLE etf_price_quality_state ADD COLUMN algorithm_version INTEGER")
            state = conn.execute("SELECT source_rows, source_max_date, source_close_sum, algorithm_version FROM etf_price_quality_state WHERE id=1").fetchone()
            adjusted_rows = conn.execute("SELECT COUNT(*) FROM etf_daily_adjusted").fetchone()[0]
            expected_state = (source_rows, source_max, source_close_sum, QUALITY_ALGORITHM_VERSION)
            if state == expected_state and adjusted_rows == source_rows:
                anomalies = pd.read_sql_query("SELECT * FROM etf_price_anomalies ORDER BY date,symbol", conn)
                return PriceQualitySummary(source_rows, len(anomalies), int(anomalies.repaired.sum()), int(anomalies.isolated.sum()), True), anomalies

        columns = {row[1] for row in conn.execute("PRAGMA table_info(etf_daily)")}
        selected = [
            column if column in columns else f"NULL AS {column}"
            for column in ("symbol", "date", "open", "high", "low", "close", "volume", "amount", "data_source", "price_adjustment")
        ]
        raw = pd.read_sql_query(
            f"SELECT {', '.join(selected)} FROM etf_daily ORDER BY symbol,date", conn
        )
        repaired_frames, anomaly_rows = [], []
        for _, group in raw.groupby("symbol", sort=False):
            repaired, anomalies = repair_symbol_history(group)
            repaired_frames.append(repaired)
            anomaly_rows.extend(anomalies)
        adjusted = pd.concat(repaired_frames, ignore_index=True) if repaired_frames else raw
        anomalies = pd.DataFrame(anomaly_rows)
        conn.execute("DROP INDEX IF EXISTS idx_etf_adjusted_symbol_date_new")
        conn.execute("DROP INDEX IF EXISTS idx_etf_adjusted_symbol_date")
        conn.execute("DROP TABLE IF EXISTS etf_daily_adjusted_new")
        adjusted.to_sql("etf_daily_adjusted_new", conn, index=False)
        conn.execute("DROP TABLE IF EXISTS etf_daily_adjusted")
        conn.execute("ALTER TABLE etf_daily_adjusted_new RENAME TO etf_daily_adjusted")
        conn.execute("CREATE UNIQUE INDEX idx_etf_adjusted_symbol_date ON etf_daily_adjusted(symbol,date)")
        conn.execute("DROP TABLE IF EXISTS etf_price_anomalies")
        if anomalies.empty:
            anomalies = pd.DataFrame(columns=[
                "symbol", "date", "previous_close", "current_close", "daily_change",
                "suspected_corporate_action", "source", "repaired", "repair_method", "isolated",
                "factor", "residual_return", "note",
                "repair_confidence", "confirmation_source", "manual_review_required",
            ])
        anomalies.to_sql("etf_price_anomalies", conn, index=False)
        conn.execute("CREATE INDEX idx_etf_anomalies_symbol_date ON etf_price_anomalies(symbol,date)")
        conn.execute("CREATE TABLE IF NOT EXISTS etf_price_quality_state (id INTEGER PRIMARY KEY, source_rows INTEGER, source_max_date TEXT, source_close_sum REAL, algorithm_version INTEGER)")
        state_columns = {row[1] for row in conn.execute("PRAGMA table_info(etf_price_quality_state)")}
        if "source_close_sum" not in state_columns:
            conn.execute("ALTER TABLE etf_price_quality_state ADD COLUMN source_close_sum REAL")
        if "algorithm_version" not in state_columns:
            conn.execute("ALTER TABLE etf_price_quality_state ADD COLUMN algorithm_version INTEGER")
        conn.execute(
            "INSERT OR REPLACE INTO etf_price_quality_state(id,source_rows,source_max_date,source_close_sum,algorithm_version) VALUES(1,?,?,?,?)",
            (source_rows, source_max, source_close_sum, QUALITY_ALGORITHM_VERSION),
        )
        conn.commit()
    return PriceQualitySummary(source_rows, len(anomalies), int(anomalies.repaired.sum()), int(anomalies.isolated.sum())), anomalies


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="扫描ETF历史价格断层并重建分析用复权表")
    parser.add_argument("--db", type=Path, default=Path("data/etf_daily.db"))
    parser.add_argument("--force", action="store_true", help="即使原始数据未变化也重新扫描")
    args = parser.parse_args(argv)
    summary, _ = prepare_adjusted_history(args.db, args.force)
    print(f"价格断层扫描完成：{summary.source_rows}条，异常{summary.anomaly_count}条，自动修正{summary.repaired_count}条，隔离{summary.isolated_count}条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
