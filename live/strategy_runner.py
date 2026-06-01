"""StrategyRunner — wraps a backtest strategy fn as a per-bar live signal.

The backtest contract (see backtesting/strategies.py) is:
    fn(df, **params) -> DataFrame[index, "entry": bool, "exit": bool]
Live trading only cares about the LAST bar: if its `entry` is True → BUY,
elif its `exit` is True → SELL, else HOLD.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from backtesting.strategies import STRATEGY_REGISTRY

Action = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class Signal:
    action: Action
    price: float
    symbol: str
    strategy: str


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

    def run(self, symbol: str, bars: pd.DataFrame) -> Signal:
        price = float(bars["close"].iloc[-1]) if len(bars) else 0.0
        try:
            signals = self._fn(bars)
        except Exception:
            return Signal("HOLD", price, symbol, self.name)
        if signals is None or len(signals) == 0:
            return Signal("HOLD", price, symbol, self.name)
        last = signals.iloc[-1]
        if bool(last.get("entry", False)):
            return Signal("BUY", price, symbol, self.name)
        if bool(last.get("exit", False)):
            return Signal("SELL", price, symbol, self.name)
        return Signal("HOLD", price, symbol, self.name)
