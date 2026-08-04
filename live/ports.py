"""Ports the Orchestrator depends on, so it can be tested with fakes."""
from __future__ import annotations

from typing import Optional, Protocol

import pandas as pd


class BarProvider(Protocol):
    def get_bars(self, symbol: str, lookback: int) -> Optional[pd.DataFrame]:
        """Trailing `lookback` daily OHLCV bars, lowercase columns, date index.

        May return None / empty when data is unavailable; callers must guard.
        """
        ...


class ExecutorPort(Protocol):
    def buy(self, *, symbol: str, qty: float, price: float, strategy: str,
            strategy_type: str, max_hold_days: Optional[int]) -> bool: ...

    def sell(self, *, symbol: str, qty: float, price: float, strategy: str) -> bool: ...
