"""独立 SQLite 缓存；不改写原始价格、成交量或人工资料。"""

from contextlib import closing
from datetime import date, timedelta
import json
from pathlib import Path
import sqlite3

import pandas as pd

from .screen import METADATA_COLUMNS


class UniverseCache:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as conn, conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (
                    symbol TEXT NOT NULL, known_on TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(symbol, known_on));
                CREATE TABLE IF NOT EXISTS amounts (
                    symbol TEXT NOT NULL, date TEXT NOT NULL, amount REAL NOT NULL,
                    fetched_on TEXT NOT NULL, source TEXT NOT NULL,
                    PRIMARY KEY(symbol, date));
                CREATE TABLE IF NOT EXISTS amount_fetches (
                    symbol TEXT PRIMARY KEY, start TEXT NOT NULL, end TEXT NOT NULL,
                    fetched_on TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sync_runs (
                    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL);
            """)

    def connect(self):
        return sqlite3.connect(str(self.path), timeout=30)

    def metadata(self, cutoff):
        with closing(self.connect()) as conn:
            rows = conn.execute("SELECT payload FROM metadata WHERE known_on <= ? ORDER BY known_on", (cutoff,)).fetchall()
        return pd.DataFrame([json.loads(row[0]) for row in rows], columns=METADATA_COLUMNS)

    def refresh_plan(self, symbols, start, end, today, force=False, metadata_age=7):
        with closing(self.connect()) as conn:
            latest = dict(conn.execute("SELECT symbol, MAX(known_on) FROM metadata WHERE known_on <= ? GROUP BY symbol", (today,)))
            fetches = {row[0]: row[1:] for row in conn.execute("SELECT symbol,start,end,fetched_on FROM amount_fetches")}
        tasks = []
        for symbol in symbols:
            needs_meta = force or symbol not in latest or (date.fromisoformat(today) - date.fromisoformat(latest[symbol])).days >= metadata_age
            previous = fetches.get(symbol)
            amount_start = start
            needs_amount = True
            if not force and previous and previous[0] <= start:
                if previous[1] >= end and previous[2] >= end:
                    needs_amount = False
                elif previous[1] < end:
                    amount_start = max(start, (date.fromisoformat(previous[1]) + timedelta(days=1)).isoformat())
            if needs_meta or needs_amount:
                tasks.append((symbol, needs_meta, amount_start if needs_amount else None))
        return tasks

    def save(self, symbol, metadata, amounts, start, end, today):
        with closing(self.connect()) as conn, conn:
            if metadata is not None:
                conn.execute("INSERT OR REPLACE INTO metadata VALUES (?,?,?)", (
                    symbol, metadata["known_on"], json.dumps(metadata, ensure_ascii=False, allow_nan=False)))
            if amounts is not None:
                conn.executemany("INSERT OR REPLACE INTO amounts VALUES (?,?,?,?,?)", [
                    (symbol, row["date"], row["amount"], today, row.get("source", "eastmoney:kline:f57:CNY")) for row in amounts])
                # start 表示已尝试覆盖的窗口，包括无成交的日期。数据是否足够由筛选器判断。
                old = conn.execute("SELECT start,end FROM amount_fetches WHERE symbol=?", (symbol,)).fetchone()
                if old and start <= (date.fromisoformat(old[1]) + timedelta(days=1)).isoformat() and old[0] <= (date.fromisoformat(end) + timedelta(days=1)).isoformat():
                    start, end = min(start, old[0]), max(end, old[1])
                # 不把两个不相邻窗口之间的空白错误标记为已经采集。
                conn.execute("INSERT OR REPLACE INTO amount_fetches VALUES (?,?,?,?)", (symbol, start, end, today))

    def overlay_amounts(self, daily):
        result = daily.copy()
        with closing(self.connect()) as conn:
            amounts = pd.read_sql_query("SELECT symbol,date,amount FROM amounts WHERE date >= ? AND date <= ?", conn,
                                        params=(str(daily.date.min())[:10], str(daily.date.max())[:10]))
        result["date"] = pd.to_datetime(result["date"])
        amounts["date"] = pd.to_datetime(amounts["date"])
        result = result.merge(amounts.rename(columns={"amount": "_verified_amount"}), on=["symbol", "date"], how="left", validate="one_to_one")
        if "amount" in result:
            result["amount"] = result["_verified_amount"].combine_first(result["amount"])
        else:
            result["amount"] = result["_verified_amount"]
        return result.drop(columns="_verified_amount")

    def record_run(self, summary):
        with closing(self.connect()) as conn, conn:
            conn.execute("INSERT INTO sync_runs(created_at,payload) VALUES(datetime('now'),?)",
                         (json.dumps(summary, ensure_ascii=False),))
