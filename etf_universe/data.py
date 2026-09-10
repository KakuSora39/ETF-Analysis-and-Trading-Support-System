"""读取行情数据库和有日期的资料文件。"""

import json
from contextlib import closing
from pathlib import Path
import sqlite3

import pandas as pd

from .screen import METADATA_COLUMNS, FilterConfig, normalize_metadata


DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "etf_daily.db"


def _market_table(conn: sqlite3.Connection) -> str:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return "etf_daily_adjusted" if "etf_daily_adjusted" in tables else "etf_daily"


def load_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".json":
        frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8-sig")), columns=METADATA_COLUMNS)
    else:
        frame = pd.read_csv(path, dtype={"symbol": str, "index_id": str}, encoding="utf-8-sig")
    return normalize_metadata(frame)


def load_market(db_path: str | Path, config: FilterConfig, as_of: str | None = None):
    path = Path(db_path).resolve()
    # mode=ro 防止输错路径时 sqlite 静默创建空库。
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        table = _market_table(conn)
        if as_of is None:
            as_of = conn.execute(f"SELECT MAX(date) FROM {table}").fetchone()[0]
        if not as_of:
            raise ValueError("数据库没有 ETF 日线")
        as_of = pd.Timestamp(as_of).date().isoformat()
        dates = conn.execute(
            f"SELECT DISTINCT date FROM {table} WHERE date <= ? ORDER BY date DESC LIMIT ?",
            (as_of, config.history_days),
        ).fetchall()
        if not dates or dates[0][0] != as_of:
            raise ValueError(f"{as_of} 无市场行情，请指定有数据的交易日")
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        selected = [col for col in ("symbol", "date", "close", "volume", "amount") if col in columns]
        daily = pd.read_sql_query(
            f"SELECT {', '.join(selected)} FROM {table} WHERE date >= ? AND date <= ? ORDER BY date, symbol",
            conn, params=(dates[-1][0], as_of),
        )
        universe = pd.read_sql_query(
            "SELECT symbol, name, delisted_date FROM etf_list "
            "UNION ALL SELECT DISTINCT symbol, '' AS name, NULL AS delisted_date FROM etf_daily "
            "WHERE symbol NOT IN (SELECT symbol FROM etf_list) AND date <= ?",
            conn, params=(as_of,),
        )
    return daily, universe, as_of


def load_strategy_market(
    db_path: str | Path,
    symbols: list[str],
    as_of: str,
    history_days: int = 280,
    cache_path: str | Path | None = None,
    prefer_adjusted: bool = True,
):
    """Load an aligned history for current-signal strategy voting.

    The legacy database's ``volume`` column may mix source units.  When the
    universe cache contains verified turnover, the last liquidity window is
    converted to an implied share volume (amount / close).  Strategies that
    compare recent volume therefore receive a consistent unit.
    """
    if history_days < 61:
        raise ValueError("strategy history_days must be at least 61")
    path = Path(db_path).resolve()
    as_of = pd.Timestamp(as_of).date().isoformat()
    symbols = sorted({str(symbol).zfill(6) for symbol in symbols})
    if not symbols:
        return {}, pd.DataFrame(), []
    placeholders = ",".join("?" for _ in symbols)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        table = _market_table(conn) if prefer_adjusted else "etf_daily"
        dates = [row[0] for row in conn.execute(
            f"SELECT DISTINCT date FROM {table} WHERE date <= ? ORDER BY date DESC LIMIT ?",
            (as_of, history_days),
        ).fetchall()][::-1]
        if not dates or dates[-1] != as_of:
            raise ValueError(f"{as_of} has no ETF market data")
        etf_columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        market_select = ["symbol", "date"] + [
            column if column in etf_columns else f"NULL AS {column}"
            for column in ("open", "high", "low", "close", "volume")
        ]
        daily = pd.read_sql_query(
            f"SELECT {', '.join(market_select)} FROM {table} "
            f"WHERE symbol IN ({placeholders}) AND date >= ? AND date <= ? ORDER BY date,symbol",
            conn, params=(*symbols, dates[0], as_of),
        )
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "index_daily" in tables:
            benchmark = pd.read_sql_query(
                "SELECT date,open,high,low,close,volume FROM index_daily "
                "WHERE symbol='000300' AND date >= ? AND date <= ? ORDER BY date",
                conn, params=(dates[0], as_of),
            )
        else:
            benchmark = pd.DataFrame()

    calendar = pd.DatetimeIndex(pd.to_datetime(dates))
    daily["date"] = pd.to_datetime(daily["date"])
    for column in ("open", "high", "low", "close", "volume"):
        daily[column] = pd.to_numeric(daily[column], errors="coerce").astype(float)

    verified = pd.DataFrame()
    if cache_path is not None and Path(cache_path).exists():
        cache = Path(cache_path).resolve()
        with closing(sqlite3.connect(cache.as_uri() + "?mode=ro", uri=True)) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "amounts" in tables:
                verified = pd.read_sql_query(
                    f"SELECT symbol,date,amount FROM amounts WHERE symbol IN ({placeholders}) "
                    "AND date >= ? AND date <= ?",
                    conn, params=(*symbols, dates[0], as_of),
                )
    if not verified.empty:
        verified["date"] = pd.to_datetime(verified["date"])
        verified["amount"] = pd.to_numeric(verified["amount"], errors="coerce")
        daily = daily.merge(verified, on=["symbol", "date"], how="left")
        good = (daily["amount"] > 0) & (daily["close"] > 0)
        daily.loc[good, "volume"] = daily.loc[good, "amount"] / daily.loc[good, "close"]

    data = {}
    for symbol, frame in daily.groupby("symbol"):
        frame = frame.set_index("date").reindex(calendar)
        frame.index.name = "date"
        frame["date"] = calendar
        frame["symbol"] = symbol
        frame["pct_chg"] = frame["close"].pct_change(fill_method=None)
        data[symbol] = frame.reset_index(drop=True)

    if benchmark.empty:
        benchmark = pd.DataFrame(index=calendar, columns=["open", "high", "low", "close", "volume"])
        benchmark["date"] = calendar
    else:
        benchmark["date"] = pd.to_datetime(benchmark["date"])
        benchmark = benchmark.set_index("date").reindex(calendar)
        benchmark["date"] = calendar
        benchmark = benchmark.reset_index(drop=True)
    benchmark["pct_chg"] = pd.to_numeric(benchmark.get("close"), errors="coerce").pct_change(fill_method=None)
    return data, benchmark, [date.date().isoformat() for date in calendar]


def write_template(universe: pd.DataFrame, path: Path):
    """只预填代码和已有名称，不把今天伪装成历史资料日期。"""
    records = []
    for row in universe.sort_values("symbol").to_dict("records"):
        record = dict.fromkeys(METADATA_COLUMNS)
        record.update({"symbol": row["symbol"], "name": row.get("name", ""),
                       "delisted_date": row.get("delisted_date")})
        records.append(record)
    # 模板不覆盖用户已经维护的资料。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
