"""StrategyRunner — resolves a registered backtest strategy for live use.

The backtest contract (see backtesting/strategies.py) is:
    fn(df, **params) -> DataFrame[index, "entry": bool, "exit": bool]
"""
from __future__ import annotations

from backtesting.strategies import STRATEGY_REGISTRY


class StrategyRunner:
    """Runs one registered strategy against a trailing window of bars."""

    def __init__(self, strategy_name: str) -> None:
        if strategy_name not in STRATEGY_REGISTRY:
            raise KeyError(
                f"Unknown strategy {strategy_name!r}. "
                f"Available: {list(STRATEGY_REGISTRY)}"
            )
        spec = STRATEGY_REGISTRY[strategy_name]
        self.name = strategy_name
        self._fn = spec["fn"]
        self.strategy_type = str(spec["type"])
        self.max_hold_days = spec.get("max_hold_days")
