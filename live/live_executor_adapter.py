"""Adapter mapping the ExecutorPort onto the existing executor.py.

BUY  -> place_market_order_with_time_exit (no stop-loss; time/signal exit).
SELL -> place_market_sell of the strategy's SPECIFIC qty.

MONEY GUARDRAIL — SELL must NEVER route through close_position. The three
horses share ONE paper account, so close_position would liquidate the whole
NET position for the symbol and silently clobber the other strategies' shares.
Selling the exact qty is what keeps per-strategy isolation intact. Do not
"simplify" sell back to close_position (see executor.place_market_sell).
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
        res = self._ex.place_market_sell(symbol=symbol, qty=qty, strategy=strategy)
        return res is not None
