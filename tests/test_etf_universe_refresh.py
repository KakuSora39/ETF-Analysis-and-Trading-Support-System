from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import pandas as pd
import requests

from etf_universe.cache import UniverseCache
from etf_universe.provider import classify_fund, parse_amounts, parse_profile, parse_exchange_amounts, PublicETFProvider, RateLimiter
from etf_universe.refresh import refresh_cache


TODAY = "2026-09-07"


def profile(symbol="510300", kind="指数型-股票", scale="948.72亿元（截止至：2026年06月30日）"):
    # 模拟源站未闭合的 td，确保使用 lxml 正确分隔相邻字段。
    return f"""<table><tr><th>基金全称</th><td>测试交易型开放式证券投资基金</td>
    <th>基金简称</th><td>测试ETF</td></tr><tr><th>基金代码</th><td>{symbol}（主代码）
    <th>基金类型</th><td>{kind}</td></tr><tr><th>净资产规模</th><td>{scale}</td>
    <th>跟踪标的</th><td>沪深300指数</td></tr></table>"""


class ProviderTests(unittest.TestCase):
    def test_profile_scale_date_code_and_tracking(self):
        info = parse_profile(profile(), "510300", TODAY)
        self.assertEqual(info["aum_yuan"], 948.72 * 1e8)
        self.assertEqual(info["aum_date"], "2026-06-30")
        self.assertEqual(info["known_on"], TODAY)
        self.assertIsNone(info["listed_date"])
        self.assertEqual(info["index_id"], "EM:沪深300指数")
        self.assertEqual(info["fund_type"], "equity")

    def test_scale_units_missing_and_foreign_currency(self):
        self.assertEqual(parse_profile(profile(scale="1,234.5万元（截止至：2026年06月30日）"), "510300", TODAY)["aum_yuan"], 12345000)
        for scale in ("---", "3亿美元（截止至：2026年06月30日）"):
            self.assertIsNone(parse_profile(profile(scale=scale), "510300", TODAY)["aum_yuan"])

    def test_error_page_and_wrong_symbol_rejected(self):
        for html in ("<html>稍后重试</html>", profile(symbol="159915")):
            with self.assertRaises(ValueError):
                parse_profile(html, "510300", TODAY)

    def test_fund_types_include_fixed_income_and_overseas(self):
        cases = [
            ("指数型-固收", "国债ETF", "国债指数", "bond"),
            ("QDII-债券", "债券ETF", "", "bond"),
            ("货币型-普通货币", "日利ETF", "", "money"),
            ("指数型-海外股票", "纳指ETF", "纳斯达克100指数", "qdii_equity"),
            ("指数型-其他", "黄金ETF", "黄金9999", "commodity"),
            ("QDII-商品", "原油ETF", "", "qdii_commodity"),
            ("指数型-其他", "未识别ETF", "", "unknown"),
        ]
        for kind, name, target, expected in cases:
            self.assertEqual(classify_fund(kind, name, target), expected)

    def test_amount_uses_f57_not_volume_or_close(self):
        payload = {"data": {"code": "510300", "klines": [
            "2026-09-04,4.637,4.616,4.672,4.599,8414655,3905672999.000,1.58",
            "2026-09-07,4.635,4.634,4.649,4.612,9930783,4602371534.000,0.80",
        ]}}
        rows = parse_amounts(payload, "510300", "2026-09-07", "2026-09-07")
        self.assertEqual(rows, [{"symbol": "510300", "date": TODAY, "amount": 4602371534.0}])

    def test_amount_invalid_payloads_fail(self):
        for data in (None, {"code": "159915"}, {"code": "510300", "klines": []},
                     {"code": "510300", "klines": ["2026-09-07,1,1,1,1,1,nan"]}):
            with self.assertRaises(ValueError):
                parse_amounts({"data": data}, "510300", "2026-09-01", TODAY)

    def test_exchange_amount_unit_and_source(self):
        payload = {"code": "510300", "kline": [
            [20260904, 4.637, 4.672, 4.599, 4.616, 841465543, 3905672999],
            [20260907, 4.635, 4.649, 4.612, 4.634, 993078300, 4602371534],
        ]}
        rows = parse_exchange_amounts(payload, "510300", TODAY, TODAY)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["amount"], 4602371534)
        self.assertEqual(rows[0]["source"], "sse:dayk:amount:CNY")
        with self.assertRaises(ValueError):
            parse_exchange_amounts(payload, "159915", TODAY, TODAY)
        payload["kline"][0].append(0)
        with self.assertRaises(ValueError):
            parse_exchange_amounts(payload, "510300", TODAY, TODAY)

    def test_service_unavailable_is_not_hammered(self):
        provider = PublicETFProvider(RateLimiter(0))
        self.addCleanup(provider.close)
        response = requests.Response()
        response.status_code = 503
        response.headers["Retry-After"] = "120"
        provider.session.get = Mock(return_value=response)
        with self.assertRaises(requests.HTTPError):
            provider.get("https://example.invalid/history")
        with self.assertRaises(requests.ConnectionError):
            provider.get("https://example.invalid/history")
        self.assertEqual(provider.session.get.call_count, 1)

    def test_transport_fallback_is_remembered(self):
        limiter = RateLimiter(0)
        provider = PublicETFProvider(limiter)
        self.addCleanup(provider.close)
        response = requests.Response()
        response.status_code = 200
        provider.session.get = Mock(side_effect=[requests.ConnectionError("连接失败"), response, response])
        with patch("etf_universe.provider.time.sleep"):
            provider.get("https://example.invalid/history", fallback_url="http://example.invalid/history")
            provider.get("https://example.invalid/history", fallback_url="http://example.invalid/history")
        self.assertEqual(provider.session.get.call_args_list[-1].args[0], "http://example.invalid/history")


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = UniverseCache(Path(self.temp.name) / "cache.db")

    def info(self, day=TODAY):
        result = parse_profile(profile(), "510300", day)
        result["listed_date"] = "2012-05-28"
        return result

    def save(self, day=TODAY):
        self.cache.save("510300", self.info(day), [{"symbol": "510300", "date": day, "amount": 123456}],
                        "2026-08-01", day, day)

    def test_metadata_history_is_append_only_across_dates(self):
        self.save("2026-09-04")
        self.save(TODAY)
        self.assertEqual(len(self.cache.metadata("2026-09-04")), 1)
        self.assertEqual(len(self.cache.metadata(TODAY)), 2)
        self.assertTrue(self.cache.metadata("2026-09-03").empty)

    def test_same_day_cache_and_incremental_amount(self):
        self.save()
        self.assertEqual(self.cache.refresh_plan(["510300"], "2026-08-10", TODAY, TODAY), [])
        self.assertEqual(self.cache.refresh_plan(["510300"], "2026-08-10", "2026-09-08", "2026-09-08"),
                         [("510300", False, "2026-09-08")])
        self.assertEqual(self.cache.refresh_plan(["510300"], "2026-08-10", TODAY, TODAY, force=True),
                         [("510300", True, "2026-08-10")])

    def test_metadata_refreshes_after_seven_days(self):
        self.save()
        plan = self.cache.refresh_plan(["510300"], "2026-08-10", "2026-09-14", "2026-09-14")
        self.assertTrue(plan[0][1])

    def test_overlay_is_date_aligned_and_does_not_change_price_volume(self):
        self.save()
        daily = pd.DataFrame({"symbol": ["510300", "510300", "159915"],
                              "date": [TODAY, "2026-09-04", TODAY], "close": [1, 2, 3], "volume": [10, 20, 30]})
        result = self.cache.overlay_amounts(daily)
        self.assertEqual(result.loc[0, "amount"], 123456)
        self.assertTrue(result.loc[1:, "amount"].isna().all())
        self.assertEqual(result.close.tolist(), daily.close.tolist())
        self.assertEqual(result.volume.tolist(), daily.volume.tolist())

    def test_refresh_keeps_partial_success_and_retries_failure(self):
        class FakeProvider:
            def metadata(self, symbol, day):
                raise ValueError("暂时无法取得资料")

            def amounts(self, symbol, start, end):
                return [{"symbol": symbol, "date": end, "amount": 30000000}]

            def close(self):
                pass

        with patch("etf_universe.refresh.china_today", return_value=TODAY), redirect_stdout(io.StringIO()):
            report = refresh_cache(self.cache, ["510300"], "2026-08-10", TODAY, provider_factory=FakeProvider)
        self.assertEqual(report["amounts_updated"], 1)
        self.assertEqual(report["metadata_updated"], 0)
        self.assertEqual(report["failures"][0]["stage"], "metadata")
        self.assertEqual(self.cache.refresh_plan(["510300"], "2026-08-10", TODAY, TODAY), [("510300", True, None)])

    def test_cached_refresh_does_not_call_network(self):
        self.save()

        def unexpected():
            raise AssertionError("不应发出网络请求")

        with patch("etf_universe.refresh.china_today", return_value=TODAY), redirect_stdout(io.StringIO()):
            report = refresh_cache(self.cache, ["510300"], "2026-08-10", TODAY, provider_factory=unexpected)
        self.assertEqual(report["scheduled"], 0)

    def test_disjoint_windows_do_not_claim_the_gap(self):
        self.save()
        self.cache.save("510300", None, [{"symbol": "510300", "date": "2026-10-09", "amount": 1}],
                        "2026-10-01", "2026-10-09", "2026-10-09")
        plan = self.cache.refresh_plan(["510300"], "2026-09-10", "2026-09-20", "2026-10-09")
        self.assertEqual(plan[0][2], "2026-09-10")

    def test_historical_refresh_does_not_fetch_current_metadata(self):
        calls = []

        class FakeProvider:
            def metadata(self, symbol, day):
                raise AssertionError("历史模式不应获取今天的资料")

            def amounts(self, symbol, start, end):
                calls.append(symbol)
                return [{"symbol": symbol, "date": end, "amount": 1}]

            def close(self):
                pass

        with patch("etf_universe.refresh.china_today", return_value=TODAY), redirect_stdout(io.StringIO()):
            report = refresh_cache(self.cache, ["510300"], "2026-08-10", "2026-08-31",
                                   provider_factory=FakeProvider, refresh_metadata=False)
        self.assertEqual(calls, ["510300"])
        self.assertEqual(report["metadata_updated"], 0)
        self.assertEqual(report["failures"], [])


if __name__ == "__main__":
    unittest.main()
