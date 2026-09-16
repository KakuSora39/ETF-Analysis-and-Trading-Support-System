"""多标的综合账户；持仓和订单均与人工持仓文件隔离。"""
from __future__ import annotations

import json
from pathlib import Path

from simulation.framework.state import StateManager
from .config import PaperConfig

VERSIONS = {"conservative": "conservative-v1", "progressive": "progressive-v1"}


class PaperAccount:
    def __init__(self, output_dir: str | Path, mode: str, config: PaperConfig):
        if mode not in VERSIONS:
            raise ValueError(f"未知账户：{mode}")
        self.path = Path(output_dir) / f"state_etf_universe_{mode}.json"
        self.mode = mode
        self.config = config

    def load_or_create(self, signal_date: str) -> dict:
        if not self.path.exists():
            state = {
                "schema_version": 1, "strategy_version": VERSIONS[self.mode],
                "mode": self.mode, "simulation_start_date": signal_date,
                "last_processed_date": "", "initial_capital": self.config.initial_capital,
                "cash": self.config.initial_capital, "positions": {}, "pending_orders": [],
                "executed_order_ids": [], "trade_log": [], "nav_history": [],
                "peak_value": self.config.initial_capital,
            }
            self.save(state)
            return state
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"模拟盘状态无法读取，请检查 {self.path}；未自动覆盖或重建") from exc
        if state.get("mode") != self.mode or state.get("strategy_version") != VERSIONS[self.mode]:
            raise ValueError("账户模式或策略版本与状态文件不匹配；请迁移旧状态或指定新目录")
        changed = False
        peak = state["initial_capital"]
        worst = 0.0
        for nav in state.get("nav_history", []):
            peak = max(peak, nav["total_value"])
            worst = min(worst, nav["total_value"] / peak - 1)
            for key, value in (("initial_capital", state["initial_capital"]),
                               ("peak_value", peak), ("max_drawdown", worst)):
                if key not in nav:
                    nav[key] = value
                    changed = True
        if changed:
            self.save(state)
        return state

    def save(self, state: dict) -> None:
        StateManager.save_json_atomic(self.path, state)
