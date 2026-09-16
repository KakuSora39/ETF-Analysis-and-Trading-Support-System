"""综合模拟盘参数。两个账户共享成交和风险预算参数。"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass(frozen=True)
class PaperConfig:
    initial_capital: float = 100000.0
    commission_rate: float = 0.0002
    slippage: float = 0.0001
    max_symbol_weight: float = 0.25
    max_positions: int = 4
    probe_position_ratio: float = 0.20
    confirmed_position_ratio: float = 0.60
    max_position_ratio: float = 1.00
    time_stop_days: int = 8
    strategy_validation_refresh: str = "weekly"

    def __post_init__(self):
        if self.initial_capital <= 0 or not 0 < self.max_symbol_weight <= 1:
            raise ValueError("初始资金和单只上限无效")
        if not 0 < self.probe_position_ratio < self.confirmed_position_ratio <= self.max_position_ratio <= 1:
            raise ValueError("分阶段仓位比例无效")
        if self.max_positions < 1 or self.time_stop_days < 1:
            raise ValueError("持仓数量或时间止损无效")
        if self.commission_rate < 0 or not 0 <= self.slippage < 1:
            raise ValueError("手续费或滑点无效")
        if self.strategy_validation_refresh not in ("daily", "weekly"):
            raise ValueError("策略验证刷新只能是 daily/weekly")

    @classmethod
    def load(cls, path: str | Path | None = None):
        if not path:
            return cls()
        values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        unknown = set(values) - set(asdict(cls()))
        if unknown:
            raise ValueError(f"未知模拟盘配置：{sorted(unknown)}")
        return cls(**values)
