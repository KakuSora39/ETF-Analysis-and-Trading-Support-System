import unittest
from datetime import datetime

from etf_sync.sync import ETFSync
from main import _describe_etf_sync


class _CurrentEngine:
    def __init__(self, symbols):
        self.symbols = symbols

    def get_etf_list_symbols(self):
        return self.symbols

    def get_all_etf_last_dates(self):
        return {symbol: "2099-01-01" for symbol in self.symbols}


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

    def test_time_gate_summary_is_not_reported_as_zero_etfs(self):
        text = _describe_etf_sync({
            "status": "skipped", "error": "time gate", "etf_count": 0,
            "is_trade_day": True,
        })
        self.assertEqual(text, "ETF 已跳过（未到同步时间）")


if __name__ == "__main__":
    unittest.main()
