"""Adapter mapping the ExecutorPort onto the existing executor.py.

BUY  -> place_market_order_with_time_exit (no stop-loss; time/signal exit).
SELL -> close_position (dedicated close path; cleaner than a SELL market order).
Isolates real-executor specifics here so the Orchestrator stays testable.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_MOMENTUM_BACKSTOP_DAYS = 252  # exit primarily on SELL signal; this is a safety cap


class LiveExecutorAdapter:
    def __init__(self, executor) -> None:
        self._ex = executor

    def buy(
        self,
        *,
        symbol: str,
        qty: float,
        price: float,
        strategy: str,
        strategy_type: str,
        max_hold_days: Optional[int],
    ) -> bool:
        hold = max_hold_days if max_hold_days else _MOMENTUM_BACKSTOP_DAYS
        res = self._ex.place_market_order_with_time_exit(
            symbol=symbol,
            qty=qty,
            max_hold_days=hold,
            side="BUY",
            strategy=strategy,
        )
        return res is not None

    def sell(
        self,
        *,
        symbol: str,
        qty: float,
        price: float,
        strategy: str,
    ) -> bool:
        # Use the dedicated close_position path — it handles the full
        # Alpaca close + journals the exit + clears the time-exit tracker.
        res = self._ex.close_position(symbol, reason=f"signal_exit:{strategy}")
        return res is not None
