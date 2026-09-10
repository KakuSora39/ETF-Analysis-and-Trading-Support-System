"""独立使用 unittest；不访问网络、不修改用户行情库。"""

from contextlib import closing, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from etf_universe import FilterConfig, screen_universe
from etf_universe.__main__ import main
from etf_universe.data import load_market, load_metadata, write_template
from etf_universe.screen import normalize_metadata
from etf_universe.cache import UniverseCache


class UniverseTests(unittest.TestCase):
    def setUp(self):
        self.cfg = FilterConfig()
        self.dates = pd.bdate_range("2025-01-02", periods=90)
        self.day = self.dates[-1].date().isoformat()
        self.symbols = [f"{510001 + i}" for i in range(10)]
        rng = np.random.default_rng(12)
        self.daily = pd.concat([
            pd.DataFrame({"symbol": symbol, "date": self.dates,
                          "close": 2 * np.cumprod(1 + rng.normal(.001, .01, len(self.dates))),
                          "amount": 40_000_000 + i * 1_000_000})
            for i, symbol in enumerate(self.symbols)
        ], ignore_index=True)
        self.meta = pd.DataFrame([
            {"symbol": symbol, "name": f"测试ETF{i}", "known_on": self.day,
             "listed_date": "2020-01-01", "fund_type": "equity", "aum_yuan": 500_000_000,
             "aum_date": self.day, "index_id": f"INDEX:{i}", "delisted_date": None}
            for i, symbol in enumerate(self.symbols)
        ])

    def screen(self, **kwargs):
        return screen_universe(kwargs.get("daily", self.daily), kwargs.get("metadata", self.meta),
                               self.day, kwargs.get("config", self.cfg), kwargs.get("universe"))

    def test_market_rank_and_top_n(self):
        result = self.screen()
        self.assertEqual(len(result.candidates), 10)
        self.assertEqual(len(result.selected), 3)
        self.assertTrue(result.ranking["momentum"].is_monotonic_decreasing)
        self.assertEqual(set(result.symbols), set(self.symbols))

    def test_all_basic_exclusions_have_reasons(self):
        self.meta.loc[0, "fund_type"] = "money"
        self.meta.loc[1, "fund_type"] = "bond"
        self.meta.loc[2, "listed_date"] = self.day
        self.meta.loc[3, "aum_yuan"] = self.cfg.min_aum - 1
        self.meta.loc[4, "fund_type"] = "unknown"
        self.meta.loc[5, "aum_date"] = "2024-01-01"
        self.daily.loc[self.daily.symbol == self.symbols[6], "amount"] = 1
        self.meta = self.meta[self.meta.symbol != self.symbols[7]]
        audit = self.screen().audit.set_index("symbol")
        for index, reason in enumerate(["货币ETF", "债券ETF", "上市不足60天", "基金规模过小",
                                        "基金类型缺失或未知", "基金规模过期", "20日成交额中位数过低", "缺少当时可用的基金资料"]):
            self.assertIn(reason, audit.loc[self.symbols[index], "reasons"])

    def test_threshold_equality_passes(self):
        self.meta["aum_yuan"] = self.cfg.min_aum
        self.meta["listed_date"] = (pd.Timestamp(self.day) - pd.Timedelta(days=180)).date().isoformat()
        self.daily["amount"] = self.cfg.min_avg_amount
        self.assertEqual(len(self.screen().candidates), 10)

    def test_same_index_keeps_more_liquid_before_ranking(self):
        self.meta.loc[:1, "index_id"] = "CSI:300"
        result = self.screen()
        row = result.audit.set_index("symbol").loc[self.symbols[0]]
        self.assertEqual(row["duplicate_of"], self.symbols[1])
        self.assertEqual(row["reasons"], "跟踪同一指数")
        self.assertNotIn(self.symbols[0], result.ranking.symbol.tolist())

    def test_correlation_only_deduplicates_similar_themes(self):
        self.meta["index_id"] = ""
        a = self.daily.symbol == self.symbols[0]
        b = self.daily.symbol == self.symbols[1]
        self.daily.loc[b, "close"] = self.daily.loc[a, "close"].to_numpy() * 3
        self.meta.loc[0, "name"] = "芯片ETF甲"
        self.meta.loc[1, "name"] = "半导体ETF乙"
        result = self.screen()
        row = result.audit.set_index("symbol").loc[self.symbols[0]]
        self.assertEqual(row["reasons"], "相似主题且收益相关性过高")
        self.assertEqual(row["duplicate_of"], self.symbols[1])

        self.meta.loc[0, "name"] = "油气ETF"
        self.meta.loc[1, "name"] = "船舶ETF"
        result = self.screen()
        self.assertIn(self.symbols[0], result.symbols)
        self.assertIn(self.symbols[1], result.symbols)

    def test_listing_age_creates_formal_and_observation_tiers(self):
        self.meta.loc[0, "listed_date"] = (pd.Timestamp(self.day) - pd.Timedelta(days=30)).date().isoformat()
        self.meta.loc[1, "listed_date"] = (pd.Timestamp(self.day) - pd.Timedelta(days=100)).date().isoformat()
        result = self.screen()
        audit = result.audit.set_index("symbol")
        self.assertEqual(audit.loc[self.symbols[0], "qualification"], "数据不足")
        self.assertEqual(audit.loc[self.symbols[1], "qualification"], "新ETF观察")
        self.assertNotIn(self.symbols[0], result.symbols)
        self.assertNotIn(self.symbols[1], result.symbols)
        self.assertEqual(set(result.new_etfs.symbol), {self.symbols[0], self.symbols[1]})

    def test_liquidity_uses_median_not_outlier_inflated_average(self):
        mask = self.daily.symbol == self.symbols[0]
        indices = self.daily[mask].tail(20).index
        self.daily.loc[indices, "amount"] = 1_000_000
        self.daily.loc[indices[-1], "amount"] = 500_000_000
        row = self.screen().audit.set_index("symbol").loc[self.symbols[0]]
        self.assertGreater(row["avg_amount"], self.cfg.min_avg_amount)
        self.assertLess(row["median_amount"], self.cfg.min_avg_amount)
        self.assertIn("20日成交额中位数过低", row["reasons"])

    def test_future_prices_and_metadata_cannot_change_past(self):
        baseline = self.screen()
        future = self.meta.copy()
        future["known_on"] = (pd.Timestamp(self.day) + pd.Timedelta(days=5)).date().isoformat()
        future["aum_yuan"] = 1
        prices = self.daily.tail(10).copy()
        prices["date"] = self.dates[-1] + pd.Timedelta(days=5)
        prices["symbol"] = self.symbols
        prices["close"] = 100000
        result = self.screen(daily=pd.concat([self.daily, prices]), metadata=pd.concat([self.meta, future]))
        pd.testing.assert_frame_equal(baseline.audit, result.audit)
        pd.testing.assert_frame_equal(baseline.selected, result.selected)

    def test_future_only_metadata_excluded(self):
        self.meta["known_on"] = (pd.Timestamp(self.day) + pd.Timedelta(days=1)).date().isoformat()
        result = self.screen()
        self.assertTrue(result.selected.empty)
        self.assertTrue(result.audit.reasons.str.contains("缺少当时可用").all())

    def test_current_analysis_records_separate_information_date(self):
        tomorrow = (pd.Timestamp(self.day) + pd.Timedelta(days=1)).date().isoformat()
        self.meta["known_on"] = tomorrow
        result = screen_universe(self.daily, self.meta, self.day, information_as_of=tomorrow)
        self.assertEqual(len(result.candidates), 10)
        self.assertEqual(result.as_of, self.day)
        self.assertEqual(result.information_as_of, tomorrow)
        self.assertTrue(self.screen().candidates.empty)

    def test_suspension_and_gaps_are_not_forward_filled(self):
        self.daily = self.daily[~((self.daily.symbol == self.symbols[0]) & (self.daily.date == self.dates[-1]))]
        self.daily.loc[(self.daily.symbol == self.symbols[1]) & (self.daily.date == self.dates[-1]), "amount"] = 0
        self.daily = self.daily[~((self.daily.symbol == self.symbols[2]) & (self.daily.date == self.dates[-40]))]
        audit = self.screen().audit.set_index("symbol")
        self.assertIn("当日无有效交易", audit.loc[self.symbols[0], "reasons"])
        self.assertIn("当日无有效交易", audit.loc[self.symbols[1], "reasons"])
        self.assertIn("相关性观察数据不足", audit.loc[self.symbols[2], "reasons"])

    def test_volume_unit_must_be_explicit(self):
        self.daily["volume"] = 300_000 / self.daily["close"]
        self.daily = self.daily.drop(columns="amount")
        self.assertTrue(self.screen().selected.empty)
        self.assertTrue(self.screen(config=replace(self.cfg, volume_unit="shares")).selected.empty)
        result = self.screen(config=replace(self.cfg, volume_unit="lots"))
        self.assertEqual(len(result.candidates), 10)
        self.assertTrue(result.candidates.amount_estimated.all())

    def test_real_amount_has_priority(self):
        self.daily["volume"] = 0
        result = self.screen(config=replace(self.cfg, volume_unit="lots"))
        self.assertEqual(len(result.candidates), 10)
        self.assertFalse(result.candidates.amount_estimated.any())

    def test_missing_and_nonfinite_metadata_fail_closed(self):
        self.meta.loc[0, "listed_date"] = None
        self.meta["aum_yuan"] = self.meta["aum_yuan"].astype(float)
        self.meta.loc[1, "aum_yuan"] = float("inf")
        self.meta.loc[2, "aum_date"] = None
        self.meta.loc[3, "known_on"] = "2024-01-01"
        self.meta.loc[3, "aum_date"] = "2024-01-01"
        audit = self.screen().audit.set_index("symbol")
        for index, reason in enumerate(["缺少上市日期", "缺少有效基金规模", "缺少有效基金规模", "基金资料过期"]):
            self.assertIn(reason, audit.loc[self.symbols[index], "reasons"])

    def test_latest_known_snapshot_replaces_old_values(self):
        older = self.meta.copy()
        older["known_on"] = self.dates[-2].date().isoformat()
        older["aum_date"] = older["known_on"]
        older["aum_yuan"] = 100_000_000
        result = self.screen(metadata=pd.concat([self.meta, older]))
        self.assertEqual(len(result.candidates), 10)

    def test_duplicate_prices_and_invalid_signal_date_raise(self):
        with self.assertRaises(ValueError):
            self.screen(daily=pd.concat([self.daily, self.daily.head(1)]))
        with self.assertRaises(ValueError):
            screen_universe(self.daily, self.meta, "2020-01-01")

    def test_delisting_uses_effective_date_and_no_data_is_audited(self):
        listing = pd.DataFrame([
            {"symbol": self.symbols[0], "name": "A", "delisted_date": self.day},
            {"symbol": self.symbols[1], "name": "B", "delisted_date": "2030-01-01"},
            {"symbol": "599999", "name": "C", "delisted_date": None},
        ])
        audit = self.screen(universe=listing).audit.set_index("symbol")
        self.assertIn("已退市", audit.loc[self.symbols[0], "reasons"])
        self.assertTrue(audit.loc[self.symbols[1], "eligible"])
        self.assertIn("缺少行情", audit.loc["599999", "reasons"])

    def test_invalid_config_and_metadata(self):
        for values in ({"top_n": 0}, {"min_aum": -1}, {"max_correlation": 1.1}, {"min_avg_amount": float("nan")}):
            with self.assertRaises(ValueError):
                FilterConfig(**values)
        with self.assertRaises(ValueError):
            normalize_metadata(pd.concat([self.meta, self.meta]))
        self.meta.loc[0, "aum_date"] = "2030-01-01"
        with self.assertRaises(ValueError):
            self.screen()

    def test_too_short_history_empty_instead_of_partial_windows(self):
        result = self.screen(daily=self.daily[self.daily.date >= self.dates[-10]])
        self.assertTrue(result.candidates.empty)

    def test_deterministic_ties_use_size_then_code(self):
        self.meta.loc[:2, "index_id"] = "same"
        self.daily["amount"] = 40_000_000
        self.meta.loc[0, "aum_yuan"] = 900_000_000
        self.meta.loc[1, "aum_yuan"] = 900_000_000
        self.assertIn(self.symbols[0], self.screen().symbols)
        self.assertNotIn(self.symbols[1], self.screen().symbols)

    def test_sqlite_cli_reports_and_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "test.db"
            with closing(sqlite3.connect(db)) as conn:
                frame = self.daily.copy()
                frame["date"] = frame.date.dt.strftime("%Y-%m-%d")
                frame.to_sql("etf_daily", conn, index=False)
                self.meta[["symbol", "name", "delisted_date"]].to_sql("etf_list", conn, index=False)
                conn.commit()
            metadata = root / "metadata.json"
            metadata.write_text(self.meta.to_json(orient="records", force_ascii=False), encoding="utf-8")
            daily, universe, day = load_market(db, self.cfg)
            self.assertEqual(day, self.day)
            self.assertEqual(daily.date.nunique(), self.cfg.history_days)
            template = root / "template.json"
            write_template(universe, template)
            with self.assertRaises(FileExistsError):
                write_template(universe, template)
            with redirect_stdout(io.StringIO()):
                status = main(["--db", str(db), "--metadata", str(metadata), "--output", str(root / "output")])
            self.assertEqual(status, 0)
            summary = json.loads(next((root / "output").glob("*/summary.json")).read_text(encoding="utf-8"))
            self.assertGreater(len(summary["selected_symbols"]), 0)
            self.assertLessEqual(len(summary["selected_symbols"]), 10)
            self.assertGreater(summary["voting_strategy_count"], 0)
            run_dir = next((root / "output").iterdir())
            self.assertTrue((run_dir / "strategy_votes.csv").exists())
            self.assertTrue((run_dir / "strategy_status.csv").exists())
            self.assertTrue((run_dir / "consensus.csv").exists())
            self.assertEqual(summary["universe_count"], 10)
            self.meta["fund_type"] = "money"
            metadata.write_text(self.meta.to_json(orient="records"), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--db", str(db), "--metadata", str(metadata), "--output", str(root / "empty")]), 2)
            with self.assertRaises(sqlite3.OperationalError):
                load_market(root / "missing.db", self.cfg)
            self.assertFalse((root / "missing.db").exists())

    def test_automatic_cache_cli_uses_real_amount_without_volume_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "test.db"
            with closing(sqlite3.connect(db)) as conn:
                frame = self.daily.drop(columns="amount").copy()
                frame["date"] = frame.date.dt.strftime("%Y-%m-%d")
                frame["volume"] = 0  # 不应该用于估算成交额。
                frame.to_sql("etf_daily", conn, index=False)
                self.meta[["symbol", "name", "delisted_date"]].to_sql("etf_list", conn, index=False)
                conn.commit()
            cache = UniverseCache(root / "etf_universe.db")
            for info in self.meta.to_dict("records"):
                amounts = self.daily[self.daily.symbol == info["symbol"]][["symbol", "date", "amount"]].copy()
                amounts["date"] = amounts.date.dt.strftime("%Y-%m-%d")
                cache.save(info["symbol"], info, amounts.to_dict("records"),
                           self.dates[0].date().isoformat(), self.day, self.day)
            with redirect_stdout(io.StringIO()):
                status = main(["--db", str(db), "--offline", "--as-of", self.day, "--output", str(root / "out")])
            self.assertEqual(status, 0)
            summary = json.loads(next((root / "out").glob("*/summary.json")).read_text(encoding="utf-8"))
            self.assertEqual(summary["candidate_count"], 10)
            self.assertEqual(summary["amount_estimated_count"], 0)


if __name__ == "__main__":
    unittest.main()
