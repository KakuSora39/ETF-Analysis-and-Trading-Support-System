"""Append-only signal snapshots and delayed forward outcomes."""
from __future__ import annotations

from pathlib import Path
import sqlite3
from contextlib import contextmanager

import pandas as pd

from .strategy_policy import progressive_entry
from .decision import conservative_entry


FEATURE_COLUMNS = (
    "date", "code", "name", "market_regime", "market_short_state", "lane", "action",
    "trend_support", "rebound_support", "defensive_support", "effective_strategy_support",
    "active_family_support", "return_3d", "return_5d", "return_20d", "distance_ma5",
    "distance_ma20", "distance_ma60", "ma20_slope", "volatility", "rsi", "volume_ratio",
    "relative_strength", "rebound_score", "structure_risk", "trading_risk",
)
FUTURE_COLUMNS = ("future_5d_return", "future_10d_return", "future_max_gain", "future_max_drawdown")


def _scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(value, (bool, int, float)):
        return float(value)
    return str(value)


def _feature(row: dict, day: str, lane: str, regime: str, short_state: str) -> dict:
    mapping = {
        "date": day, "code": row.get("symbol"), "name": row.get("name"),
        "market_regime": regime, "market_short_state": short_state, "lane": lane,
        "action": row.get("action"), "trend_support": row.get("trend_support"),
        "rebound_support": row.get("rebound_support"), "defensive_support": row.get("defense_support"),
        "effective_strategy_support": row.get(f"validated_{lane}_strategy_support"),
        "active_family_support": row.get(f"{lane}_support"), "return_3d": row.get("return_3d"),
        "return_5d": row.get("return_5d"), "return_20d": row.get("return_20d"),
        "distance_ma5": row.get("distance_ma5"), "distance_ma20": row.get("distance_ma20"),
        "distance_ma60": row.get("distance_ma60"), "ma20_slope": row.get("ma20_slope"),
        "volatility": row.get("volatility_20d"), "rsi": row.get("rsi14"),
        "volume_ratio": row.get("volume_ratio"), "relative_strength": row.get("relative_strength"),
        "rebound_score": row.get("stopping_score"), "structure_risk": row.get("structural_risk_level"),
        "trading_risk": row.get("trading_risk_level"),
    }
    return {key: _scalar(mapping[key]) for key in FEATURE_COLUMNS}


class SignalHistory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            definitions = ", ".join(f"{key} {'TEXT' if key in {'date','code','name','market_regime','market_short_state','lane','action','structure_risk','trading_risk'} else 'REAL'}" for key in FEATURE_COLUMNS)
            db.execute(f"CREATE TABLE IF NOT EXISTS signal_feature_history ({definitions}, PRIMARY KEY(date, code, lane))")
            db.execute("""CREATE TABLE IF NOT EXISTS signal_outcomes (
                signal_date TEXT, code TEXT, signal_type TEXT, conservative_action TEXT,
                progressive_action TEXT, future_5d_return REAL, future_10d_return REAL,
                future_max_gain REAL, future_max_drawdown REAL, classification TEXT,
                progressive_executed INTEGER DEFAULT 0,
                PRIMARY KEY(signal_date, code, signal_type))""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(signal_outcomes)")}
            if "progressive_executed" not in columns:
                db.execute("ALTER TABLE signal_outcomes ADD COLUMN progressive_executed INTEGER DEFAULT 0")
            db.execute("""CREATE TABLE IF NOT EXISTS strategy_validation_cache (
                week_key TEXT, as_of TEXT, strategy TEXT, validation_level TEXT,
                PRIMARY KEY(week_key, strategy))""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def record_day(self, day: str, lanes: dict[str, pd.DataFrame], regime: str,
                   short_state: str, config, conservative_positions: set[str] | None = None) -> None:
        conservative_positions = conservative_positions or set()
        placeholders = ",".join("?" for _ in FEATURE_COLUMNS)
        fields = ",".join(FEATURE_COLUMNS)
        with self.connect() as db:
            for lane, frame in lanes.items():
                for row in frame.to_dict("records"):
                    row["opportunity_lane"] = lane
                    values = _feature(row, day, lane, regime, short_state)
                    db.execute(f"INSERT OR IGNORE INTO signal_feature_history ({fields}) VALUES ({placeholders})",
                               tuple(values.values()))
                    conservative, _, _ = conservative_entry(row)
                    progressive, _, _ = progressive_entry(row, False, config)
                    if (conservative != "entry" and str(row["symbol"]) not in conservative_positions
                            and row.get("action") != "暂不参与"):
                        db.execute("""INSERT OR IGNORE INTO signal_outcomes
                            (signal_date, code, signal_type, conservative_action, progressive_action)
                            VALUES (?, ?, ?, ?, ?)""", (day, str(row["symbol"]), lane, conservative, progressive))

    def fill_forward(self, frames: dict[str, pd.DataFrame], as_of: str,
                     gain_threshold: float = .05, loss_threshold: float = -.05) -> None:
        """Only mature observations are labeled; this table is never read by decisions."""
        with self.connect() as db:
            pending = db.execute("""SELECT signal_date, code, signal_type, progressive_action, progressive_executed
                FROM signal_outcomes WHERE future_10d_return IS NULL""").fetchall()
            for signal_date, code, lane, action, executed in pending:
                frame = frames.get(code)
                if frame is None or frame.empty:
                    continue
                ordered = frame.sort_values("date")
                dates = pd.to_datetime(ordered["date"]).dt.strftime("%Y-%m-%d")
                first = ordered.loc[dates.eq(signal_date), "close"]
                future = ordered.loc[dates.gt(signal_date) & dates.le(as_of), "close"]
                if first.empty or len(future) < 10 or float(first.iloc[-1]) <= 0:
                    continue
                base = float(first.iloc[-1])
                returns = future.iloc[:10].astype(float) / base - 1
                five, ten = float(returns.iloc[4]), float(returns.iloc[9])
                maximum, minimum = float(returns.max()), float(returns.min())
                label = ("false_probe" if action == "probe_entry" and executed and (five <= loss_threshold or minimum <= loss_threshold)
                         else "missed_opportunity" if maximum >= gain_threshold else "")
                db.execute("""UPDATE signal_outcomes SET future_5d_return=?, future_10d_return=?,
                    future_max_gain=?, future_max_drawdown=?, classification=?
                    WHERE signal_date=? AND code=? AND signal_type=?""",
                    (five, ten, maximum, minimum, label, signal_date, code, lane))

    def mark_executed(self, trades: list[dict]) -> None:
        with self.connect() as db:
            for trade in trades:
                if trade.get("side") == "buy":
                    db.execute("""UPDATE signal_outcomes SET progressive_executed=1
                        WHERE signal_date=? AND code=? AND signal_type=?""",
                        (trade["signal_date"], trade["code"], trade.get("entry_type", "")))

    def counts(self) -> dict[str, int]:
        with self.connect() as db:
            return dict(db.execute("SELECT classification, count(*) FROM signal_outcomes WHERE classification != '' GROUP BY classification"))

    def pending_symbols(self) -> set[str]:
        with self.connect() as db:
            return {row[0] for row in db.execute("SELECT DISTINCT code FROM signal_outcomes WHERE future_10d_return IS NULL")}

    def cache_validation(self, day: str, report: pd.DataFrame) -> None:
        """Keep the latest observed validation per week for a future frozen policy.

        The formal analysis still computes validation daily. These snapshots are
        observational and are never read during today's order decision.
        """
        week = pd.Timestamp(day).strftime("%G-W%V")
        with self.connect() as db:
            for row in report.to_dict("records"):
                strategy = str(row.get("strategy", row.get("策略名称", "")))
                if not strategy:
                    continue
                level = str(row.get("validation_level", row.get("验证等级", row.get("是否通过历史验证", ""))))
                db.execute("""INSERT INTO strategy_validation_cache(week_key, as_of, strategy, validation_level)
                    VALUES (?, ?, ?, ?) ON CONFLICT(week_key, strategy) DO UPDATE SET
                    as_of=excluded.as_of, validation_level=excluded.validation_level
                    WHERE excluded.as_of >= strategy_validation_cache.as_of""", (week, day, strategy, level))

    def frozen_validation(self, day: str) -> dict[str, str]:
        """Return the completed previous trading week's last observed levels."""
        week = pd.Timestamp(day).strftime("%G-W%V")
        with self.connect() as db:
            prior = db.execute("SELECT MAX(week_key) FROM strategy_validation_cache WHERE week_key < ?", (week,)).fetchone()[0]
            if prior is None:
                return {}
            return dict(db.execute("SELECT strategy, validation_level FROM strategy_validation_cache WHERE week_key=?", (prior,)))
