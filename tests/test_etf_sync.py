import sqlite3
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from etf_sync.sync import ETFSync
from main import _describe_etf_sync


class _CurrentEngine:
    def __init__(self, symbols):
        self.symbols = symbols

    def get_etf_list_symbols(self):
        return self.symbols

    def get_all_etf_last_dates(self):
        return {symbol: "2099-01-01" for symbol in self.symbols}


class _LaggingEngine:
    def get_etf_list_symbols(self):
        return ["510300"]

    def get_all_etf_last_dates(self):
        return {"510300": "2026-09-10"}


class _FakeTencentSource:
    def __init__(self):
        self.source_count = {"tencent": 0, "sina": 0}
        self.active_source = "tencent"
        self.last_source = "tencent"

    def get_daily(self, code, days):
        self.source_count["tencent"] += 1
        frame = pd.DataFrame({
            "date": ["2026-09-11", "2026-09-14"],
            "close": [4.0, 4.1],
        })
        frame.attrs["adjustment"] = "qfq"
        frame.attrs["source"] = "tencent"
        return frame


class SyncFastPathTests(unittest.TestCase):
    def test_force_before_close_targets_previous_trading_day(self):
        sync = object.__new__(ETFSync)
        sync.is_trade_day = lambda value=None: (value or datetime.now().date()).weekday() < 5
        morning = sync._latest_completed_trade_date(datetime(2026, 9, 10, 9, 7))
        after_close = sync._latest_completed_trade_date(datetime(2026, 9, 10, 16, 0))
        self.assertEqual(morning.isoformat(), "2026-09-09")
        self.assertEqual(after_close.isoformat(), "2026-09-10")

    def test_force_skips_network_when_latest_day_is_already_covered(self):
        sync = object.__new__(ETFSync)
        sync.engine = _CurrentEngine([f"{index:06d}" for index in range(200)])
        result = sync.sync_etf_daily(force=True)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["error"], "already current")
        self.assertEqual(result["coverage_count"], 200)
        self.assertIn("无需扫描", _describe_etf_sync(result))

    def test_sync_summary_includes_records_and_target_date(self):
        text = _describe_etf_sync({
            "status": "ok", "etf_count": 3, "record_count": 6,
            "target_date": "2026-09-11", "is_trade_day": False,
        })
        self.assertEqual(text, "ETF 已同步 3 只，写入 6 条（截止 2026-09-11）")

    def test_normal_gate_targets_previous_day_but_weekend_can_catch_up(self):
        sync = object.__new__(ETFSync)
        sync.settings = SimpleNamespace(sync_after_hour=20, sync_after_minute=0)
        sync.is_trade_day = lambda value=None: (value or datetime.now().date()).weekday() < 5
        self.assertEqual(
            sync._latest_safe_sync_date(datetime(2026, 9, 10, 9, 0)).isoformat(),
            "2026-09-09",
        )
        self.assertEqual(
            sync._latest_safe_sync_date(datetime(2026, 9, 10, 20, 1)).isoformat(),
            "2026-09-10",
        )
        self.assertEqual(
            sync._latest_safe_sync_date(datetime(2026, 9, 12, 9, 0)).isoformat(),
            "2026-09-11",
        )

    def test_missing_history_is_written_before_gate_but_today_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "daily.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    "CREATE TABLE etf_daily (date TEXT, close REAL, symbol TEXT, "
                    "data_source TEXT, price_adjustment TEXT, PRIMARY KEY(symbol, date))"
                )
                conn.commit()
            finally:
                conn.close()
            sync = object.__new__(ETFSync)
            sync.settings = SimpleNamespace(
                db_path=str(db_path), start_date="2020-01-01",
                sync_after_hour=20, sync_after_minute=0,
            )
            sync.engine = _LaggingEngine()
            sync.tc_source = _FakeTencentSource()
            sync._sync_target = lambda force: (date(2026, 9, 11), True)

            result = sync.sync_etf_daily(force=False)

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["etf_count"], 1)
            conn = sqlite3.connect(db_path)
            try:
                dates = [row[0] for row in conn.execute("SELECT date FROM etf_daily ORDER BY date")]
            finally:
                conn.close()
            self.assertEqual(dates, ["2026-09-11"])


if __name__ == "__main__":
    unittest.main()
