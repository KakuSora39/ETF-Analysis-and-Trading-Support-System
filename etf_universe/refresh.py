"""有界并发、限速采集；逐只提交缓存，失败可重试且不会被静默忽略。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

from .provider import PublicETFProvider, RateLimiter, china_today


def refresh_cache(cache, symbols, start, end, *, workers=4, force=False, provider_factory=None, progress=print,
                  refresh_metadata=True):
    if not 1 <= workers <= 16:
        raise ValueError("workers 必须在 1 到 16 之间")
    today = china_today()
    if end > today:
        raise ValueError("不能采集未来交易日")
    tasks = cache.refresh_plan(symbols, start, end, today, force=force)
    if not refresh_metadata:
        tasks = [(symbol, False, amount_start) for symbol, _, amount_start in tasks if amount_start]
    summary = {"fetched_on": today, "start": start, "end": end, "requested": len(symbols),
               "scheduled": len(tasks), "metadata_updated": 0, "amounts_updated": 0, "failures": []}
    limiter = RateLimiter()
    factory = provider_factory or (lambda: PublicETFProvider(limiter))
    local = threading.local()
    clients = []
    clients_lock = threading.Lock()

    def fetch(task):
        symbol, needs_meta, amount_start = task
        if not hasattr(local, "client"):
            local.client = factory()
            with clients_lock:
                clients.append(local.client)
        client = local.client
        info, rows, errors = None, None, []
        if needs_meta:
            try:
                info = client.metadata(symbol, today)
            except Exception as exc:
                errors.append({"symbol": symbol, "stage": "metadata", "error": str(exc)[:250]})
        if amount_start:
            try:
                rows = client.amounts(symbol, amount_start, end)
            except Exception as exc:
                errors.append({"symbol": symbol, "stage": "amount", "error": str(exc)[:250]})
        return symbol, info, rows, errors

    progress(f"自动补充：{len(symbols)} 只 ETF，{len(tasks)} 只需要更新（{workers} 并发）", flush=True)
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(fetch, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            symbol, info, rows, errors = future.result()
            cache.save(symbol, info, rows, start, end, today)
            summary["metadata_updated"] += info is not None
            summary["amounts_updated"] += rows is not None
            summary["failures"].extend(errors)
            if done % 25 == 0 or done == len(tasks):
                progress(f"  {done}/{len(tasks)}：资料 {summary['metadata_updated']}，成交额 {summary['amounts_updated']}，失败 {len(summary['failures'])}，{time.monotonic()-started:.0f}s", flush=True)
    finally:
        # Ctrl+C 不再等待尚未开始的数千项任务，已完成的逐只缓存仍保留。
        pool.shutdown(wait=True, cancel_futures=True)
        for client in clients:
            client.close()
    cache.record_run(summary)
    return summary
