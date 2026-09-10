"""公开数据适配器：基金概况、上市日期和交易所日线真实成交额。"""

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import math
import re
import threading
import time
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import requests


def china_today():
    return datetime.now(timezone(timedelta(hours=8))).date().isoformat()


def parse_date(value):
    match = re.search(r"(\d{4})[年/-]?(\d{2})[月/-]?(\d{2})", str(value))
    if not match:
        return None
    return datetime.strptime("-".join(match.groups()), "%Y-%m-%d").date().isoformat()


def classify_fund(kind, full_name, target):
    # 不从投资范围正文分类（股票基金的投资范围也常包含债券）。
    if "货币" in kind:
        return "money"
    if any(word in kind for word in ("债券", "固收")):
        return "bond"
    overseas = any(word in kind for word in ("海外", "QDII"))
    if "股票" in kind:
        return "qdii_equity" if overseas else "equity"
    if any(word in kind for word in ("商品", "贵金属")) or (
        "其他" in kind and any(word in full_name + target for word in ("黄金", "白银", "原油", "商品", "豆粕", "有色金属期货", "能源化工"))
    ):
        return "qdii_commodity" if overseas else "commodity"
    return "unknown"


def parse_profile(html, symbol, known_on):
    # lxml 会修复源站未闭合的 td；html.parser 会把相邻字段嵌套。
    soup = BeautifulSoup(html, "lxml")
    fields = {th.get_text(strip=True): th.find_next_sibling("td").get_text(" ", strip=True)
              for th in soup.select("th") if th.find_next_sibling("td") is not None}
    if not fields.get("基金全称") or symbol not in fields.get("基金代码", ""):
        raise ValueError(f"{symbol} 基金概况缺失或代码不匹配")
    kind = fields.get("基金类型", "")
    target = re.sub(r"\s+", "", fields.get("跟踪标的", ""))
    if any(word in target for word in ("无跟踪", "暂无", "--")):
        target = ""
    scale = fields.get("净资产规模", "")
    value = re.search(r"([\d,.]+)\s*(亿|万)?元", scale)
    aum = None
    if value:
        # 外币金额不能直接当人民币元。
        if not any(word in scale for word in ("美元", "港元", "欧元", "日元")):
            aum = float(value[1].replace(",", "")) * {"亿": 1e8, "万": 1e4, None: 1}[value[2]]
    aum_date = parse_date(scale)
    if aum_date and aum_date > known_on:
        raise ValueError(f"{symbol} 基金规模统计日期晚于采集日")
    if aum is not None and not math.isfinite(aum):
        raise ValueError(f"{symbol} 基金规模不是有限数值")
    return {
        "symbol": symbol, "name": fields.get("基金简称", ""), "known_on": known_on,
        "listed_date": None, "fund_type": classify_fund(kind, fields["基金全称"], target),
        "aum_yuan": aum, "aum_date": aum_date,
        # 完整标的名称仅做空白归一，不把不同指数模糊合并；相关性补充去重。
        "index_id": f"EM:{target}" if target else "", "delisted_date": None,
        "source": f"https://fundf10.eastmoney.com/jbgk_{symbol}.html",
        "source_fund_type": kind, "source_index_name": target,
    }


def parse_amounts(payload, symbol, start, end):
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("code") != symbol:
        raise ValueError(f"{symbol} 日线接口未返回匹配的标的")
    rows = []
    for line in data.get("klines") or []:
        fields = line.split(",")
        if len(fields) < 7:
            raise ValueError(f"{symbol} 日线字段不足")
        day = parse_date(fields[0])
        amount = float(fields[6])  # f57：成交额（元），不是 f56 的成交手数。
        if day is None or not math.isfinite(amount) or amount < 0:
            raise ValueError(f"{symbol} 日线日期或成交额无效")
        if start <= day <= end:
            rows.append({"symbol": symbol, "date": day, "amount": amount})
    if not rows:
        raise ValueError(f"{symbol} 在请求区间没有成交额数据")
    if len({row["date"] for row in rows}) != len(rows):
        raise ValueError(f"{symbol} 成交额包含重复日期")
    return rows


def parse_exchange_amounts(payload, symbol, start, end):
    if payload.get("code") != symbol:
        raise ValueError(f"{symbol} 交易所行情代码不匹配")
    rows = []
    for fields in payload.get("kline") or []:
        if len(fields) != 7:
            raise ValueError(f"{symbol} 交易所行情字段变化")
        day, amount = parse_date(fields[0]), float(fields[6])
        if day is None or not math.isfinite(amount) or amount < 0:
            raise ValueError(f"{symbol} 交易所成交额无效")
        if start <= day <= end:
            rows.append({"symbol": symbol, "date": day, "amount": amount,
                         "source": "sse:dayk:amount:CNY"})
    if not rows or len({row["date"] for row in rows}) != len(rows):
        raise ValueError(f"{symbol} 交易所没有有效的区间成交额")
    return rows


class RateLimiter:
    def __init__(self, interval=0.12):
        self.interval = interval
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.preferred_endpoints = {}
        self.unavailable_until = {}

    def wait(self):
        with self.lock:
            delay = max(0, self.next_request - time.monotonic())
            self.next_request = max(self.next_request, time.monotonic()) + self.interval
        if delay:
            time.sleep(delay)


class PublicETFProvider:
    def __init__(self, limiter=None, timeout=15, attempts=3):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://fundf10.eastmoney.com/"})
        self.limiter = limiter or RateLimiter()
        self.timeout, self.attempts = timeout, attempts

    def close(self):
        self.session.close()

    def get(self, url, params=None, fallback_url=None, headers=None):
        for attempt in range(self.attempts):
            self.limiter.wait()
            try:
                endpoint = (fallback_url if attempt > 0 and fallback_url else
                            self.limiter.preferred_endpoints.get(url, url))
                host = urlparse(endpoint).netloc
                if time.monotonic() < self.limiter.unavailable_until.get(host, 0):
                    raise requests.ConnectionError(f"{host} 返回限流/服务不可用，暂缓请求")
                response = self.session.get(endpoint, params=params, timeout=self.timeout, headers=headers)
                if response.status_code in (429, 503):
                    retry_after = response.headers.get("Retry-After", "30")
                    try:
                        delay = float(retry_after) if retry_after.isdigit() else (
                            parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                    except (ValueError, TypeError):
                        delay = 30
                    delay = max(30, delay)
                    self.limiter.unavailable_until[host] = time.monotonic() + delay
                response.raise_for_status()
                response.encoding = "utf-8"
                self.limiter.preferred_endpoints[url] = endpoint
                return response
            except requests.RequestException:
                if time.monotonic() < self.limiter.unavailable_until.get(host, 0):
                    raise
                if attempt + 1 == self.attempts:
                    raise
                time.sleep(0.5 * (attempt + 1))

    @staticmethod
    def secid(symbol):
        if not re.fullmatch(r"\d{6}", symbol):
            raise ValueError("ETF 代码必须为六位数字")
        return f"{1 if symbol.startswith('5') else 0}.{symbol}"

    def metadata(self, symbol, known_on):
        info = parse_profile(self.get(f"https://fundf10.eastmoney.com/jbgk_{symbol}.html").text, symbol, known_on)
        data = self.get("https://push2delay.eastmoney.com/api/qt/stock/get", {
            "secid": self.secid(symbol), "fields": "f57,f58,f189",
        }).json().get("data")
        if not data or data.get("f57") != symbol:
            raise ValueError(f"{symbol} 上市日期接口代码不匹配")
        info["listed_date"] = parse_date(data.get("f189"))
        if not info["listed_date"] or info["listed_date"] > known_on:
            raise ValueError(f"{symbol} 未获得有效上市日期")
        info["listing_source"] = "eastmoney:f189"
        return info

    def amounts(self, symbol, start, end):
        market = "sh1" if symbol.startswith("5") else "sz1"
        count = max(30, (datetime.fromisoformat(china_today()) - datetime.fromisoformat(start)).days + 10)
        try:
            payload = self.get(f"https://yunhq.sse.com.cn:32042/v1/{market}/dayk/{symbol}", {
                "select": "date,open,high,low,close,volume,amount", "begin": -count, "end": -1,
            }, headers={"Referer": "https://www.sse.com.cn/"}).json()
            return parse_exchange_amounts(payload, symbol, start, end)
        except (requests.RequestException, ValueError):
            return self.eastmoney_amounts(symbol, start, end)

    def eastmoney_amounts(self, symbol, start, end):
        response = self.get("https://push2his.eastmoney.com/api/qt/stock/kline/get", {
            "secid": self.secid(symbol), "klt": "101", "fqt": "0",
            "beg": start.replace("-", ""), "end": end.replace("-", ""),
            "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        }, fallback_url="http://push2his.eastmoney.com/api/qt/stock/kline/get")
        rows = parse_amounts(response.json(), symbol, start, end)
        for row in rows:
            row["source"] = "eastmoney:kline:f57:CNY"
        return rows
