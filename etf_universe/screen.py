"""按信号日筛选，所有窗口只使用截至该日的数据。金额单位统一为元。"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .theme import themes_are_similar


METADATA_COLUMNS = [
    "symbol", "name", "known_on", "listed_date", "fund_type",
    "aum_yuan", "aum_date", "index_id", "delisted_date",
]
ALLOWED_TYPES = {"equity", "commodity", "qdii_equity", "qdii_commodity", "money", "bond"}


@dataclass(frozen=True)
class FilterConfig:
    liquidity_days: int = 20
    min_avg_amount: float = 10_000_000
    min_listing_days: int = 180
    min_observation_days: int = 60
    min_aum: float = 100_000_000
    max_metadata_age_days: int = 120
    correlation_days: int = 60
    max_correlation: float = 0.98
    momentum_days: int = 20
    top_n: int = 3
    # 旧库没有成交额；只有显式确认单位时才允许估算。
    volume_unit: str | None = None

    def __post_init__(self):
        for key in ("liquidity_days", "correlation_days", "momentum_days", "top_n"):
            value = getattr(self, key)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} 必须是正整数")
        if self.correlation_days < 2:
            raise ValueError("correlation_days 至少为 2")
        for key in ("min_avg_amount", "min_listing_days", "min_observation_days", "min_aum", "max_metadata_age_days"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} 必须是非负有限数值")
        if self.min_observation_days > self.min_listing_days:
            raise ValueError("min_observation_days 不能大于 min_listing_days")
        if not 0 < self.max_correlation <= 1:
            raise ValueError("max_correlation 必须在 (0, 1] 内")
        if self.volume_unit not in (None, "shares", "lots"):
            raise ValueError("volume_unit 必须是 shares、lots 或 None")

    @property
    def history_days(self):
        return max(self.liquidity_days, self.correlation_days + 1, self.momentum_days + 1)


@dataclass
class ScreenResult:
    as_of: str
    audit: pd.DataFrame
    candidates: pd.DataFrame
    ranking: pd.DataFrame
    selected: pd.DataFrame
    new_etfs: pd.DataFrame | None = None
    information_as_of: str = ""

    @property
    def symbols(self) -> list[str]:
        """供其他策略排名使用的完整候选池。"""
        return self.candidates["symbol"].tolist()


def _symbols(series):
    result = series.astype("string").str.strip()
    if not result.str.fullmatch(r"\d{6}").fillna(False).all():
        raise ValueError("symbol 必须是六位数字字符串")
    return result


def normalize_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """不猜测基金类别、上市时间或规模；缺失值交由筛选器记录。"""
    required = {"symbol", "known_on"}
    if not required.issubset(metadata.columns):
        raise ValueError("资料必须包含 symbol、known_on 列")
    meta = metadata.reindex(columns=METADATA_COLUMNS).copy()
    meta["symbol"] = _symbols(meta["symbol"])
    for key in ("known_on", "listed_date", "aum_date", "delisted_date"):
        raw = meta[key].replace("", None)
        parsed = pd.to_datetime(raw, errors="coerce")
        if (raw.notna() & parsed.isna()).any():
            raise ValueError(f"资料中的 {key} 存在无效日期")
        meta[key] = parsed.dt.normalize()
    if meta["known_on"].isna().any():
        raise ValueError("known_on 不允许为空，必须填写资料实际可获知日期")
    if meta.duplicated(["symbol", "known_on"]).any():
        raise ValueError("同一 symbol、known_on 存在重复资料")
    meta["aum_yuan"] = pd.to_numeric(meta["aum_yuan"], errors="coerce")
    for key in ("name", "fund_type", "index_id"):
        meta[key] = meta[key].fillna("").astype(str).str.strip()
    if (meta["aum_date"] > meta["known_on"]).any():
        raise ValueError("aum_date 不能晚于 known_on")
    return meta


def screen_universe(
    daily: pd.DataFrame,
    metadata: pd.DataFrame,
    as_of: str,
    config: FilterConfig | None = None,
    universe: pd.DataFrame | None = None,
    information_as_of: str | None = None,
) -> ScreenResult:
    """全市场 → 基础过滤 → 去重 → 候选 → 动量排名 → Top N。

    daily: symbol/date/close，成交额 amount（元）或 volume（需指定单位）。
    metadata: 按 known_on 保存的完整资料快照，见 METADATA_COLUMNS。
    universe: 可选的 etf_list，用于同时审计无行情及已退市的标的。
    as_of 必须是已有行情日。information_as_of 仅供当前分析（如周末）
    指定资料截止日；历史回测保持默认，禁止使用晚于信号日的资料。
    最早在行情日和资料截止日两者之后的下一交易日执行。
    """
    cfg = config or FilterConfig()
    cutoff = pd.Timestamp(as_of).normalize()
    if pd.isna(cutoff):
        raise ValueError("as_of 无效")
    info_cutoff = pd.Timestamp(information_as_of).normalize() if information_as_of else cutoff
    if pd.isna(info_cutoff) or info_cutoff < cutoff:
        raise ValueError("information_as_of 不能早于行情日期")
    if not {"symbol", "date", "close"}.issubset(daily.columns):
        raise ValueError("行情必须包含 symbol、date、close")
    prices = daily.copy()
    prices["symbol"] = _symbols(prices["symbol"])
    prices["date"] = pd.to_datetime(prices["date"], errors="raise").dt.normalize()
    prices = prices[prices["date"] <= cutoff]
    if prices.duplicated(["symbol", "date"]).any():
        raise ValueError("行情存在重复的 symbol、date")
    if prices.empty or cutoff not in prices["date"].values:
        raise ValueError(f"{as_of} 无市场行情，请使用有数据的交易日")
    calendar = pd.DatetimeIndex(sorted(prices["date"].unique()))[-cfg.history_days:]
    all_symbols = set(prices["symbol"])
    prices = prices[prices["date"].isin(calendar)].copy()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices.loc[~np.isfinite(prices["close"]) | (prices["close"] <= 0), "close"] = np.nan
    prices["amount"] = pd.to_numeric(prices.get("amount", pd.Series(index=prices.index, dtype=float)), errors="coerce")
    prices["amount_estimated"] = False
    if cfg.volume_unit is not None and "volume" in prices:
        missing = prices["amount"].isna()
        volume = pd.to_numeric(prices["volume"], errors="coerce")
        multiplier = 100 if cfg.volume_unit == "lots" else 1
        prices.loc[missing, "amount"] = prices["close"] * volume * multiplier
        prices.loc[missing, "amount_estimated"] = True
    prices.loc[~np.isfinite(prices["amount"]) | (prices["amount"] < 0), "amount"] = np.nan

    meta = normalize_metadata(metadata)
    meta = meta[meta["known_on"] <= info_cutoff].sort_values("known_on").drop_duplicates("symbol", keep="last").set_index("symbol")
    all_symbols.update(meta.index)
    universe_rows = {}
    if universe is not None:
        listing = universe.copy()
        listing["symbol"] = _symbols(listing["symbol"])
        if listing["symbol"].duplicated().any():
            raise ValueError("universe 存在重复代码")
        all_symbols.update(listing["symbol"])
        universe_rows = listing.set_index("symbol").to_dict("index")

    close = prices.pivot(index="date", columns="symbol", values="close").reindex(calendar)
    returns = close.pct_change(fill_method=None).tail(cfg.correlation_days)
    grouped = {symbol: group.set_index("date").reindex(calendar) for symbol, group in prices.groupby("symbol")}
    rows = []
    for symbol in sorted(all_symbols):
        reasons = []
        observation_reasons = []
        row = {"symbol": symbol, "name": universe_rows.get(symbol, {}).get("name", ""),
               "index_id": "", "avg_amount": np.nan, "median_amount": np.nan, "aum_yuan": np.nan,
               "momentum": np.nan, "amount_estimated": False, "duplicate_of": "",
               "fund_type": "", "listed_date": None, "listing_days": np.nan,
               "known_on": None, "aum_date": None, "history_days": 0,
               "data_completeness": np.nan, "qualification": "数据不足"}
        listing = universe_rows.get(symbol, {})
        delisted = pd.to_datetime(listing.get("delisted_date"), errors="coerce")
        if pd.notna(delisted) and delisted <= info_cutoff:
            reasons.append("已退市")
            observation_reasons.append("已退市")
        if symbol not in meta.index:
            reasons.append("缺少当时可用的基金资料")
            observation_reasons.append("缺少当时可用的基金资料")
        else:
            info = meta.loc[symbol]
            row.update({key: info[key] for key in ("name", "index_id", "aum_yuan", "fund_type",
                                                   "listed_date", "known_on", "aum_date")})
            if pd.notna(info["listed_date"]):
                row["listing_days"] = (info_cutoff - info["listed_date"]).days
                if row["listing_days"] < cfg.min_observation_days:
                    row["qualification"] = "数据不足"
                    reasons.append("上市不足60天，数据不足")
                elif row["listing_days"] < cfg.min_listing_days:
                    row["qualification"] = "新ETF观察"
                    reasons.append("新ETF观察（上市不足正式期限）")
                else:
                    row["qualification"] = "正式候选"
            if pd.notna(info["delisted_date"]) and info["delisted_date"] <= info_cutoff:
                reasons.append("已退市")
                observation_reasons.append("已退市")
            if info["fund_type"] in ("money", "bond"):
                type_reason = "货币ETF" if info["fund_type"] == "money" else "债券ETF"
                reasons.append(type_reason)
                observation_reasons.append(type_reason)
            elif info["fund_type"] not in ALLOWED_TYPES:
                reasons.append("基金类型缺失或未知")
                observation_reasons.append("基金类型缺失或未知")
            if pd.isna(info["listed_date"]):
                reasons.append("缺少上市日期")
                observation_reasons.append("缺少上市日期")
            elif info["listed_date"] > info_cutoff:
                reasons.append("尚未上市")
                observation_reasons.append("尚未上市")
            if (info_cutoff - info["known_on"]).days > cfg.max_metadata_age_days:
                reasons.append("基金资料过期")
                observation_reasons.append("基金资料过期")
            if not np.isfinite(info["aum_yuan"]) or info["aum_yuan"] <= 0 or pd.isna(info["aum_date"]):
                reasons.append("缺少有效基金规模")
                observation_reasons.append("缺少有效基金规模")
            else:
                if (info_cutoff - info["aum_date"]).days > cfg.max_metadata_age_days:
                    reasons.append("基金规模过期")
                    observation_reasons.append("基金规模过期")
                if info["aum_yuan"] < cfg.min_aum:
                    reasons.append("基金规模过小")
                    observation_reasons.append("基金规模过小")
        history = grouped.get(symbol)
        if history is None:
            reasons.append("缺少行情")
            if row["qualification"] != "数据不足":
                observation_reasons.append("缺少行情")
        else:
            row["history_days"] = int(history["close"].notna().sum())
            row["data_completeness"] = float(history["close"].notna().mean())
            amounts = history["amount"].tail(cfg.liquidity_days)
            row["amount_estimated"] = bool(history["amount_estimated"].tail(cfg.liquidity_days).fillna(False).any())
            if len(amounts) < cfg.liquidity_days or amounts.isna().any():
                reasons.append("成交额数据不足或单位未确认")
                if row["qualification"] != "数据不足":
                    observation_reasons.append("成交额数据不足或单位未确认")
            else:
                row["avg_amount"] = float(amounts.mean())
                row["median_amount"] = float(amounts.median())
                if row["median_amount"] < cfg.min_avg_amount:
                    reasons.append("20日成交额中位数过低")
                    observation_reasons.append("20日成交额中位数过低")
            if pd.isna(history["close"].iloc[-1]) or history["amount"].iloc[-1] <= 0:
                reasons.append("当日无有效交易")
                if row["qualification"] != "数据不足":
                    observation_reasons.append("当日无有效交易")
            elif pd.isna(history["amount"].iloc[-1]):
                reasons.append("当日成交额未知，无法确认交易有效性")
                if row["qualification"] != "数据不足":
                    observation_reasons.append("当日成交额未知，无法确认交易有效性")
            if len(returns) < cfg.correlation_days or returns[symbol].isna().any():
                reasons.append("相关性观察数据不足")
            elif returns[symbol].std() <= 1e-12:
                reasons.append("收益率无变化，无法判断重复度")
            momentum_prices = history["close"].tail(cfg.momentum_days + 1)
            if len(momentum_prices) == cfg.momentum_days + 1 and momentum_prices.notna().all():
                row["momentum"] = float(momentum_prices.iloc[-1] / momentum_prices.iloc[0] - 1)
        row["reasons"] = ";".join(dict.fromkeys(reasons))
        row["observation_reasons"] = ";".join(dict.fromkeys(observation_reasons))
        row["observation_eligible"] = row["qualification"] in ("数据不足", "新ETF观察") and not observation_reasons
        row["ranking_exclusion"] = "" if np.isfinite(row["momentum"]) else "动量观察数据不足"
        rows.append(row)

    audit = pd.DataFrame(rows).set_index("symbol", drop=False)
    # reset_index 避免 symbol 同时为索引名、列名导致 pandas 排序歧义。
    eligible = audit[audit["reasons"] == ""].reset_index(drop=True).sort_values(
        ["median_amount", "avg_amount", "aum_yuan", "history_days", "data_completeness", "symbol"],
        ascending=[False, False, False, False, False, True])
    correlations = returns[eligible["symbol"].tolist()].corr(min_periods=cfg.correlation_days)
    kept = []
    indices = {}
    for item in eligible.to_dict("records"):
        symbol, index_id = item["symbol"], item["index_id"]
        duplicate = indices.get(index_id) if index_id else None
        reason = "跟踪同一指数" if duplicate else ""
        if not duplicate:
            for retained in kept:
                value = correlations.at[symbol, retained]
                retained_row = audit.loc[retained]
                if (pd.notna(value) and value >= cfg.max_correlation and themes_are_similar(
                    item.get("name", ""), item.get("index_id", ""),
                    retained_row.get("name", ""), retained_row.get("index_id", ""),
                )):
                    duplicate, reason = retained, "相似主题且收益相关性过高"
                    break
        if duplicate:
            audit.loc[symbol, "duplicate_of"] = duplicate
            audit.loc[symbol, "reasons"] = reason
        else:
            kept.append(symbol)
            if index_id:
                indices[index_id] = symbol
    audit["eligible"] = audit["reasons"] == ""
    audit = audit.reset_index(drop=True)
    candidates = audit[audit["eligible"]].copy().reset_index(drop=True)
    new_etfs = audit[audit["observation_eligible"]].copy().sort_values(
        ["qualification", "median_amount", "aum_yuan"], ascending=[False, False, False]
    ).reset_index(drop=True)
    ranking = candidates.dropna(subset=["momentum"]).sort_values(
        ["momentum", "median_amount", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return ScreenResult(cutoff.date().isoformat(), audit, candidates, ranking, ranking.head(cfg.top_n).copy(), new_etfs, info_cutoff.date().isoformat())
