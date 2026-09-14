"""数据同步模块：ETF 及指数日线的全量/增量同步。

数据流向：
  - ETF 日线：TencentSource 主 → Sina 备选（双轨制自动切换）→ etf_daily 表
  - 指数日线：IndexDataSource（Sina API）→ index_daily 表
  - 中证2000：用 ETF 563000 日线代理（TencentSource 获取，存入 index_daily 表）

管线流程（run_full）：
  Phase 1: ETF 列表同步（akshare → etf_list 表）
  Phase 2: ETF 日线增量同步（双轨制数据源）
  Phase 3: 指数日线增量同步
  Phase 4: sync_log 记录 + WxPusher 推送
"""

from __future__ import annotations

import sqlite3
import time
from datetime import date, datetime, time as dt_time, timedelta
from typing import Optional

import pandas as pd

from etf_sync.config import Settings
from etf_sync.data_source import (
    IndexDataSource,
    TencentSource,
    get_etf_list,
    to_sina_code,
    to_tencent_code,
)
from etf_sync.engine import DataEngine
from etf_sync.logger import get_logger
from etf_sync.notify import push_error_alert, push_skip_notice, push_sync_summary

logger = get_logger(__name__)

# ── 指数配置（存入 index_daily 表） ──
# 前 4 个为真实指数，通过 Sina API 获取；
# 中证2000 用对应 ETF 日线代理（华夏中证2000ETF 563000）。
INDEX_CONFIG: dict[str, dict[str, str]] = {
    "000016": {"name": "上证50", "type": "index"},
    "000300": {"name": "沪深300", "type": "index"},
    "000905": {"name": "中证500", "type": "index"},
    "000852": {"name": "中证1000", "type": "index"},
    "563000": {"name": "中证2000（ETF代理）", "type": "etf_proxy"},
}


class ETFSync:
    """ETF 及指数数据同步管理器。

    管理数据源生命周期、增量同步策略、双轨制数据源切换跟踪。
    """

    def __init__(self, settings: Settings) -> None:
        """初始化 ETFSync。

        Args:
            settings: 系统配置（db_path, start_date, sync_after_hour 等）。
        """
        self.settings: Settings = settings
        self.engine = DataEngine(settings)
        self.tc_source = TencentSource()
        self.idx_source = IndexDataSource()

    # ════════════════════════════════════════════════════════════
    #  交易日判断
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def is_trade_day(check_date: date | None = None) -> bool:
        """判断指定日期是否为 A 股交易日。

        策略：
          1. 周末过滤（最快）
          2. chinese_calendar 节假日判断

        Args:
            check_date: 待检查日期，默认当天。

        Returns:
            True 表示交易日或无法确定（fail-open）。
        """
        if check_date is None:
            check_date = date.today()
        day_str: str = check_date.strftime("%Y-%m-%d")

        if check_date.weekday() >= 5:
            logger.info(f"is_trade_day: {day_str} 是周末")
            return False

        try:
            from chinese_calendar import is_workday
            result = is_workday(check_date)
            logger.info(
                f"is_trade_day: {day_str} -> {'交易日' if result else '节假日'}"
            )
            return result
        except ImportError:
            logger.warning("chinese_calendar 未安装，默认视为交易日")
            return True
        except Exception as e:
            logger.warning(f"chinese_calendar 异常（{e}），默认视为交易日")
            return True

    # ════════════════════════════════════════════════════════════
    #  时间门控
    # ════════════════════════════════════════════════════════════

    def _latest_completed_trade_date(self, now: datetime | None = None) -> date:
        """Return the latest trading day whose regular session has ended.

        Before 15:30 on a trading day, today's daily bar is not treated as
        complete even when ``--force`` bypasses the conservative 20:00 gate.
        """
        current = now or datetime.now()
        candidate = current.date()
        if self.is_trade_day(candidate) and current.time() < dt_time(15, 30):
            candidate -= timedelta(days=1)
        while not self.is_trade_day(candidate):
            candidate -= timedelta(days=1)
        return candidate

    def _latest_safe_sync_date(self, now: datetime | None = None) -> date:
        """返回普通同步在当前时刻允许写入的最近交易日。

        20:00 门控只保护今天尚未稳定的日线，不能阻止补齐更早的缺口。
        周末和节假日也可以补到最近一个交易日。
        """
        current = now or datetime.now()
        candidate = current.date()
        gate = dt_time(self.settings.sync_after_hour, self.settings.sync_after_minute)
        if self.is_trade_day(candidate) and current.time() < gate:
            candidate -= timedelta(days=1)
        while not self.is_trade_day(candidate):
            candidate -= timedelta(days=1)
        return candidate

    def _sync_target(self, force: bool, now: datetime | None = None) -> tuple[date, bool]:
        """确定同步截止日，并解释为什么可能暂缓今天的日线。"""
        current = now or datetime.now()
        today_is_trade_day = self.is_trade_day(current.date())
        target = self._latest_completed_trade_date(current) if force else self._latest_safe_sync_date(current)
        if not force:
            gate = dt_time(self.settings.sync_after_hour, self.settings.sync_after_minute)
            if today_is_trade_day and current.time() < gate:
                logger.info(
                    f"今日行情时间门控未到（{current.strftime('%H:%M')} < "
                    f"{gate.strftime('%H:%M')}），暂不写入今日K线；"
                    f"仍会补齐历史缺口至 {target.isoformat()}"
                )
            elif not today_is_trade_day:
                logger.info(f"当前为非交易日；仍会补齐历史缺口至 {target.isoformat()}")
        return target, today_is_trade_day

    # ════════════════════════════════════════════════════════════
    #  Phase 1: ETF 列表同步
    # ════════════════════════════════════════════════════════════

    def sync_etf_list(self) -> dict:
        """通过 akshare 获取全量 ETF 列表，对比本地 etf_list 表做增量更新。

        Returns:
            dict: {status, new_listed, delisted, total, error}
        """
        codes: list[str] = get_etf_list()
        if not codes:
            logger.error("sync_etf_list: 获取远程 ETF 列表失败")
            return {
                "status": "error",
                "new_listed": [],
                "delisted": [],
                "total": 0,
                "error": "akshare 返回空列表",
            }

        result: dict = self.engine.save_etf_list(codes)
        result["status"] = "ok"
        logger.info(
            f"ETF 列表同步完成: 新增 {len(result['new_listed'])} 只, "
            f"退市 {len(result['delisted'])} 只, "
            f"共 {result['total']} 只"
        )
        return result

    # ════════════════════════════════════════════════════════════
    #  Phase 2: ETF 日线增量同步（双轨制数据源）
    # ════════════════════════════════════════════════════════════

    def sync_etf_daily(self, force: bool = False, full_history: bool = False) -> dict:
        """增量同步 ETF 日线数据到 etf_daily 表。

        双轨制策略：
          1. 默认使用 Tencent 接口（主源）
          2. Tencent 失败自动切换到 Sina（备选），每 50 次尝试恢复 Tencent
          3. 返回结果中携带各数据源使用统计（tencent_count / sina_count）

        Args:
            force: 15:30 后允许把当天作为同步目标；补历史通常不需要。
            full_history: 回填模式；不使用最新交易日快速跳过。

        Returns:
            dict: {status, etf_count, tencent_count, sina_count, error}
        """
        # 时间门控只决定本次可写到哪一天，不再阻止补齐历史缺口。
        target_date, today_is_trade_day = self._sync_target(force)
        target_text = target_date.isoformat()

        # ── 获取待同步 ETF 列表 ──
        symbols: list[str] = self.engine.get_etf_list_symbols()
        if not symbols:
            logger.warning("sync_etf_daily: ETF 列表为空，尝试通过 akshare 获取")
            list_result = self.sync_etf_list()
            if list_result["status"] == "error":
                return {
                    "status": "error", "etf_count": 0,
                    "tencent_count": 0, "sina_count": 0,
                    "is_trade_day": today_is_trade_day, "error": "ETF 列表为空",
                }
            symbols = list_result.get("new_listed", [])
            if not symbols:
                symbols = self.engine.get_etf_symbols()

        if not symbols:
            logger.info("sync_etf_daily: 无待同步 ETF")
            return {
                "status": "skipped", "etf_count": 0,
                "tencent_count": 0, "sina_count": 0,
                "is_trade_day": today_is_trade_day, "error": "no symbols",
            }

        # ── 获取本地最新日期 ──
        last_dates: dict[str, str] = self.engine.get_all_etf_last_dates()
        logger.info(
            f"sync_etf_daily: 本地有 {len(last_dates)}/{len(symbols)} 只 ETF 的历史数据"
        )
        covered_count = sum(last_date >= target_text for last_date in last_dates.values())
        coverage_ratio = covered_count / len(symbols)
        if (
            not full_history
            and len(last_dates) == len(symbols)
            and coverage_ratio >= 0.995
        ):
            logger.info(
                f"sync_etf_daily: 最近完整交易日 {target_text} 已覆盖 "
                f"{covered_count}/{len(symbols)} 只 ETF（{coverage_ratio:.1%}），跳过全市场扫描"
            )
            return {
                "status": "skipped", "etf_count": 0,
                "tencent_count": 0, "sina_count": 0,
                "is_trade_day": today_is_trade_day, "error": "already current",
                "target_date": target_text, "coverage_count": covered_count,
                "symbol_count": len(symbols),
            }

        # ── 重置数据源追踪 ──
        self.tc_source.source_count = {"tencent": 0, "sina": 0}
        self.tc_source.active_source = "tencent"

        # ── 连接数据库，开始批量写入 ──
        conn: sqlite3.Connection = sqlite3.connect(self.settings.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        batch_buffer: list[dict] = []

        def _flush() -> int:
            nonlocal batch_buffer
            if not batch_buffer:
                return 0
            count = len(batch_buffer)
            cols = list(batch_buffer[0].keys())
            placeholders = ", ".join(f":{c}" for c in cols)
            cols_str = ", ".join(cols)
            conn.executemany(
                f"INSERT OR REPLACE INTO etf_daily ({cols_str}) VALUES ({placeholders})",
                batch_buffer,
            )
            conn.commit()
            batch_buffer.clear()
            return count

        def _buffer_row(record: dict) -> None:
            nonlocal batch_buffer
            batch_buffer.append(record)
            if len(batch_buffer) >= 500:
                _flush()

        # ── 逐只 ETF 同步 ──
        etf_count: int = 0
        total_new_records: int = 0
        consecutive_errors: int = 0
        max_consecutive_errors: int = 30
        log_batch_size: int = 100
        start_time: float = time.time()
        # 双轨制进度标识
        last_source_tag: str | None = None

        def _log_progress(processed: int) -> None:
            if processed % log_batch_size != 0 and processed != len(symbols):
                return
            elapsed = time.time() - start_time
            src = self.tc_source.source_count
            logger.info(
                f"ETF 扫描进度: [{processed}/{len(symbols)}] "
                f"取得数据 {etf_count} 只, 写入 {total_new_records} 条, "
                f"[腾讯{src['tencent']}/Sina{src['sina']}], {elapsed:.0f}s"
            )

        try:
            for i, sym in enumerate(symbols):
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        f"sync_etf_daily: 连续 {consecutive_errors} 次错误，终止"
                    )
                    break

                # 计算所需天数
                # 增量：按本地最新日期算；回填/新ETF：从 start_date 算起一次拉全
                # 腾讯前复权接口历史长度有限，但不会再用更长的未复权 Sina
                # 序列整段替换。数据连续性优先于单次回填长度。
                last_date = last_dates.get(sym)
                if last_date:
                    dt_last = datetime.strptime(last_date, "%Y-%m-%d")
                    days_needed = (datetime.now() - dt_last).days + 5
                    # 前复权历史会在拆分/分红后回溯变化。每次至少重取约90根，
                    # 让主源能够覆盖并改写近期旧价格，避免新旧复权口径拼接。
                    days_needed = max(days_needed, 90)
                else:
                    dt_start = datetime.strptime(
                        self.settings.start_date, "%Y-%m-%d"
                    )
                    days_needed = (datetime.now() - dt_start).days + 10
                    days_needed = max(days_needed, 30)
                days_needed = min(days_needed, 1800)

                tc_code: str = to_tencent_code(sym)
                df = self.tc_source.get_daily(tc_code, days=days_needed)

                if df is None or df.empty:
                    consecutive_errors += 1
                    if consecutive_errors <= 3:
                        logger.debug(f"sync_etf_daily {sym}: 无数据")
                    _log_progress(i + 1)
                    continue

                # 数据源可能返回今天的盘中或尚未稳定日线。门控前只写入
                # 上一完整交易日；此前遗漏的历史日期仍可正常补齐。
                df = df[df["date"].astype(str) <= target_text]
                if df.empty:
                    consecutive_errors = 0
                    _log_progress(i + 1)
                    continue

                source_adjustment = str(df.attrs.get("adjustment", "none"))
                # 腾讯为前复权数据，需要连同重叠区间一起覆盖；Sina 未声明
                # 复权，只允许补新日期，不能用它改写既有前复权历史。
                if last_date and source_adjustment != "qfq":
                    df = df[df["date"] > last_date]
                if df.empty:
                    consecutive_errors = 0
                    _log_progress(i + 1)
                    continue

                # 写入缓冲区
                df["symbol"] = sym
                df["data_source"] = str(df.attrs.get("source", self.tc_source.last_source or "unknown"))
                df["price_adjustment"] = source_adjustment
                for _, row in df.iterrows():
                    _buffer_row(row.to_dict())

                etf_count += 1
                total_new_records += len(df)
                consecutive_errors = 0

                # 数据源切换日志
                current_source = self.tc_source.last_source
                if current_source is not None and current_source != last_source_tag:
                    if last_source_tag is not None:
                        logger.info(
                            f"数据源切换: {last_source_tag} → {current_source}"
                        )
                    last_source_tag = current_source

                _log_progress(i + 1)

            _flush()

            elapsed = time.time() - start_time
            src = self.tc_source.source_count
            logger.info(
                f"ETF 日线同步完成: {etf_count}/{len(symbols)} 只, "
                f"写入 {total_new_records} 条, "
                f"[腾讯{src['tencent']}/Sina{src['sina']}], {elapsed:.0f}s"
            )
            conn.close()
            return {
                "status": "ok",
                "etf_count": etf_count,
                "tencent_count": src["tencent"],
                "sina_count": src["sina"],
                "is_trade_day": today_is_trade_day,
                "target_date": target_text,
                "record_count": total_new_records,
                "error": "",
            }

        except Exception as e:
            logger.error(f"sync_etf_daily 异常: {e}")
            _flush()
            conn.close()
            src = self.tc_source.source_count
            return {
                "status": "error",
                "etf_count": etf_count,
                "tencent_count": src["tencent"],
                "sina_count": src["sina"],
                "is_trade_day": today_is_trade_day,
                "target_date": target_text,
                "record_count": total_new_records,
                "error": str(e),
            }

    # ════════════════════════════════════════════════════════════
    #  Phase 3: 指数日线增量同步
    # ════════════════════════════════════════════════════════════

    def sync_index_daily(self, force: bool = False) -> dict:
        """同步指数日线数据到 index_daily 表。

        指数列表（INDEX_CONFIG）：
          - sh000016（上证50）、sh000300（沪深300）
          - sh000905（中证500）、sh000852（中证1000）
          - 563000（中证2000 ETF 代理，通过 TencentSource 取数据存入 index_daily）

        Args:
            force: 15:30 后允许把当天作为同步目标。

        Returns:
            dict: {status, index_count, error}
        """
        target_date, _ = self._sync_target(force)
        target_text = target_date.isoformat()

        conn: sqlite3.Connection = sqlite3.connect(self.settings.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        index_count: int = 0

        try:
            for code, info in INDEX_CONFIG.items():
                idx_type: str = info["type"]
                idx_name: str = info["name"]
                local_last: str | None = self.engine.get_index_last_date(code)
                if local_last and local_last >= target_text:
                    logger.info(f"sync_index_daily: {idx_name} 已覆盖 {target_text}，跳过")
                    continue

                if idx_type == "index":
                    sina_code: str = to_sina_code(code)
                    # 增量：按本地最新日期算；回填/无数据：从 start_date 拉全
                    if local_last:
                        dt_last = datetime.strptime(local_last, "%Y-%m-%d")
                        days_needed = (datetime.now() - dt_last).days + 5
                        days_needed = max(days_needed, 5)
                    else:
                        dt_start = datetime.strptime(
                            self.settings.start_date, "%Y-%m-%d"
                        )
                        days_needed = (datetime.now() - dt_start).days + 10
                        days_needed = max(days_needed, 30)
                    df = self.idx_source.get_daily(
                        sina_code, days=min(days_needed, 1800)
                    )
                elif idx_type == "etf_proxy":
                    tc_code: str = to_tencent_code(code)
                    # 增量优先（与指数同逻辑）；无本地数据才拉全量
                    if local_last:
                        dt_last = datetime.strptime(local_last, "%Y-%m-%d")
                        days_needed = (datetime.now() - dt_last).days + 5
                        days_needed = max(days_needed, 5)
                    else:
                        days_needed = 1800
                    df = self.tc_source.get_daily(
                        tc_code, days=min(days_needed, 1800)
                    )
                else:
                    logger.warning(f"sync_index_daily: 未知类型 {idx_type}（{code}）")
                    continue

                if df is None or df.empty:
                    logger.warning(f"sync_index_daily: {idx_name}（{code}）无数据")
                    continue

                # 普通同步在门控前不写入今天尚未稳定的指数日线。
                df = df[df["date"].astype(str) <= target_text]
                if df.empty:
                    logger.info(f"sync_index_daily: {idx_name} 没有可写入的已完成日线")
                    continue

                # 过滤增量部分
                if local_last:
                    df = df[df["date"] > local_last]
                if df.empty:
                    logger.info(f"sync_index_daily: {idx_name} 已是最新，跳过")
                    continue

                df["symbol"] = code
                n: int = self.engine.write_index_daily(df, conn)
                if n > 0:
                    index_count += 1
                    logger.info(
                        f"sync_index_daily: {idx_name} 写入 {n} 条 "
                        f"({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})"
                    )

            conn.commit()
            conn.close()
            logger.info(f"指数日线同步完成: {index_count} 个（含 ETF 代理）")
            return {"status": "ok", "index_count": index_count, "target_date": target_text, "error": ""}

        except Exception as e:
            logger.error(f"sync_index_daily 异常: {e}")
            conn.close()
            return {"status": "error", "index_count": index_count, "error": str(e)}

    # ════════════════════════════════════════════════════════════
    #  run_full：完整同步管线
    # ════════════════════════════════════════════════════════════

    def run_full(self) -> dict:
        """完整同步管线：ETF 列表 → ETF 日线 → 指数日线 → 日志 + 推送。

        各阶段说明：
          Phase 1: sync_etf_list() — akshare 获取全量 ETF 列表
          Phase 2: sync_etf_daily() — ETF 日线增量同步（双轨制数据源）
          Phase 3: sync_index_daily() — 增量指数同步
          Phase 4: sync_log 记录 + WxPusher 汇总推送

        Phase 2/Phase 3 返回 "skipped" 不终止管线。
        Phase 3 error 不影响整体状态（指数同步失败不影响 ETF 数据）。

        Returns:
            dict: {status, phases, error, duration_seconds}
        """
        t0: float = time.time()
        phases: dict[str, dict] = {}

        logger.info("=" * 60)
        logger.info("019 ETF 日线同步管线启动")
        logger.info(
            f"日期: {date.today().isoformat()}, "
            f"时间: {datetime.now().strftime('%H:%M:%S')}"
        )
        logger.info("=" * 60)

        # Phase 1: ETF 列表同步
        logger.info("Phase 1: ETF 列表同步")
        r1: dict = self.sync_etf_list()
        phases["etf_list"] = r1
        if r1.get("status") == "error":
            logger.error("Phase 1 失败，终止管线")
            error_msg = "ETF 列表同步失败"
            self.engine.log_sync(
                status="error", error_msg=error_msg,
                duration_seconds=time.time() - t0,
            )
            push_error_alert(self.settings, "Phase 1 ETF 列表同步", error_msg)
            return {
                "status": "error", "phases": phases,
                "error": error_msg, "duration_seconds": time.time() - t0,
            }

        # Phase 2: ETF 日线同步
        logger.info("Phase 2: ETF 日线增量同步")
        r2: dict = self.sync_etf_daily()
        phases["etf_daily"] = r2
        if r2.get("status") == "error":
            logger.error("Phase 2 失败，终止管线")
            error_msg = "ETF 日线同步失败"
            self.engine.log_sync(
                status="error", error_msg=error_msg,
                etf_count=r2.get("etf_count", 0),
                new_etf_count=len(r1.get("new_listed", [])),
                delisted_etf_count=len(r1.get("delisted", [])),
                duration_seconds=time.time() - t0,
            )
            push_error_alert(self.settings, "Phase 2 ETF 日线同步", error_msg)
            return {
                "status": "error", "phases": phases,
                "error": error_msg, "duration_seconds": time.time() - t0,
            }

        # Phase 3: 指数日线同步
        logger.info("Phase 3: 指数日线增量同步")
        r3: dict = self.sync_index_daily()
        phases["index_daily"] = r3

        # 汇总
        overall_status: str = "ok"
        error_msg: str = ""
        if r3.get("status") == "error":
            overall_status = "ok"
            error_msg = r3.get("error", "")
            logger.warning(f"指数同步失败（不影响 ETF 数据）: {error_msg}")

        elapsed: float = time.time() - t0
        self.engine.log_sync(
            status=overall_status,
            etf_count=r2.get("etf_count", 0),
            index_count=r3.get("index_count", 0),
            new_etf_count=len(r1.get("new_listed", [])),
            delisted_etf_count=len(r1.get("delisted", [])),
            is_trade_day=True,
            duration_seconds=elapsed,
            error_msg=error_msg,
        )

        result: dict = {
            "status": overall_status,
            "phases": phases,
            "error": error_msg,
            "duration_seconds": elapsed,
        }

        # 推送微信汇总
        push_sync_summary(self.settings, result)

        logger.info(f"管线完成: status={overall_status}, 耗时 {elapsed:.0f}s")
        logger.info(
            f"  ETF 列表: {r1.get('total', 0)} 只 "
            f"(新增 {len(r1.get('new_listed', []))}, "
            f"退市 {len(r1.get('delisted', []))})"
        )
        logger.info(f"  ETF 日线: 写入 {r2.get('etf_count', 0)} 只")
        logger.info(f"  指数日线: 写入 {r3.get('index_count', 0)} 个")

        return result
