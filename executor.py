#!/usr/bin/env python3
"""
Order executor — Alpaca trading client wrapper.

- Market orders with TIME-BASED exit for mean reversion (NO stop-loss)
- Market-hours enforcement (9:30 - 16:00 ET, Mon-Fri)
- Order logging with strategy attribution into the trade journal
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from live.market_calendar import is_trading_day

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import (
        ClosePositionRequest,
        LimitOrderRequest,
        MarketOrderRequest,
    )
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    from config import ALPACA_CONFIG
except ImportError:
    ALPACA_CONFIG = {"api_key": "", "secret_key": "", "paper": True}

try:
    from trade_journal import get_trade_journal
except ImportError:
    get_trade_journal = None


# NYSE regular session 9:30-16:00 ET. ``ZoneInfo`` tracks DST automatically, so
# a summer run no longer maps to the wrong wall-clock window (the old UTC-5
# approximation ran the gate 10:30-17:00 real ET from March-November).
_MARKET_OPEN_ET = time(9, 30)
_MARKET_CLOSE_ET = time(16, 0)
_ET = ZoneInfo("America/New_York")


def _now_et() -> datetime:
    return datetime.now(_ET)


def is_market_open(now: Optional[datetime] = None) -> bool:
    now = now or _now_et()
    # Reject full-day NYSE closures (weekends + holidays), not just weekends —
    # a DAY order submitted on a holiday queues to the next open and fills far
    # from the close the ledger recorded (silent fill drift).
    if not is_trading_day(now.date()):
        return False
    # ponytail: half-day early closes (13:00 ET) still read as open until 16:00
    # here; the 13:00-CST/14:00-ET cron would submit into a closed book on those
    # ~3 days/yr. Add early-close times to market_calendar if that bites.
    return _MARKET_OPEN_ET <= now.time() < _MARKET_CLOSE_ET


# ============================================================================
# Position tracker for time-based exits
# ============================================================================

class TimeExitTracker:
    """Tracks entry timestamps so mean-reversion positions can be closed by age."""

    def __init__(self):
        # symbol -> {"entry_time": datetime, "max_hold_days": int, "strategy": str}
        self._tracked: Dict[str, Dict[str, Any]] = {}

    def add(self, symbol: str, max_hold_days: int, strategy: str = "") -> None:
        self._tracked[symbol] = {
            "entry_time": datetime.utcnow(),
            "max_hold_days": max_hold_days,
            "strategy": strategy,
        }


# ============================================================================
# Executor
# ============================================================================

class Executor:
    """Alpaca trading wrapper. All methods are safe to call even when Alpaca
    credentials are missing — they log and return None instead of crashing."""

    def __init__(self):
        self._client = None
        self.time_exits = TimeExitTracker()
        if ALPACA_AVAILABLE and ALPACA_CONFIG.get("api_key") and ALPACA_CONFIG.get("secret_key"):
            try:
                self._client = TradingClient(
                    ALPACA_CONFIG["api_key"],
                    ALPACA_CONFIG["secret_key"],
                    paper=ALPACA_CONFIG.get("paper", True),
                )
                logger.info("Alpaca TradingClient ready (paper=%s)", ALPACA_CONFIG.get("paper", True))
            except Exception as e:
                logger.error(f"Alpaca TradingClient init failed: {e}")

    # ---------------------------------------------------------- account

    def get_account(self) -> Optional[Dict[str, Any]]:
        if not self._client:
            return None
        try:
            acc = self._client.get_account()
            return {
                "equity": float(acc.equity),
                "cash": float(acc.cash),
                "buying_power": float(acc.buying_power),
                "portfolio_value": float(acc.portfolio_value),
                "pattern_day_trader": getattr(acc, "pattern_day_trader", False),
            }
        except Exception as e:
            logger.error(f"get_account failed: {e}")
            return None

    def get_positions(self) -> List[Dict[str, Any]]:
        if not self._client:
            return []
        try:
            positions = self._client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"get_positions failed: {e}")
            return []

    # ---------------------------------------------------------- orders

    def place_market_order_with_time_exit(
        self,
        symbol: str,
        qty: float,
        max_hold_days: int,
        side: str = "BUY",
        strategy: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Mean reversion entry — market order, NO stop-loss. Time exit tracked locally."""
        if side.upper() != "BUY":
            raise ValueError(
                "place_market_order_with_time_exit is BUY-only; use place_market_sell for exits"
            )
        if not self._client:
            logger.warning(f"No Alpaca client — cannot place market order for {symbol}")
            return None
        if not is_market_open():
            logger.warning(f"Market closed — rejecting market order for {symbol}")
            return None
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=round(qty, 4),
                side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(req)
            self.time_exits.add(symbol, max_hold_days, strategy)
            self._journal({
                "timestamp": datetime.utcnow(),
                "symbol": symbol,
                "action": side.upper(),
                "strategy": strategy,
                "strategy_type": "mean_reversion",
                "reasoning": f"market order, time exit in {max_hold_days} days",
                "order_id": str(order.id),
            })
            return {"id": str(order.id), "symbol": symbol, "qty": qty, "type": "market_time_exit"}
        except Exception as e:
            logger.error(f"Market order failed for {symbol}: {e}")
            return None

    def place_market_sell(self, symbol: str, qty: float, strategy: str = "") -> Optional[Dict[str, Any]]:
        """Plain market SELL of a SPECIFIC qty (no stop-loss, no time-exit).

        Closes exactly ONE strategy's shares in the shared paper account so other
        strategies holding the same symbol are unaffected. This IS the exit, so it
        does NOT register a time-exit. Use this for strategy exit signals; use
        close_position only when liquidating the whole net account position.
        """
        if not self._client:
            logger.warning(f"No Alpaca client — cannot place market sell for {symbol}")
            return None
        if not is_market_open():
            logger.warning(f"Market closed — rejecting market sell for {symbol}")
            return None
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=round(qty, 4),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(req)
            self._journal({
                "timestamp": datetime.utcnow(),
                "symbol": symbol,
                "action": "SELL",
                "strategy": strategy,
                "strategy_type": "",
                "reasoning": "strategy exit signal — market sell of strategy qty",
                "order_id": str(order.id),
            })
            return {"id": str(order.id), "symbol": symbol, "qty": qty, "type": "market_sell"}
        except Exception as e:
            logger.error(f"Market sell failed for {symbol}: {e}")
            return None

    def close_all(self) -> int:
        if not self._client:
            return 0
        try:
            self._client.close_all_positions()
            return len(self.get_positions())
        except Exception as e:
            logger.error(f"close_all failed: {e}")
            return 0

    # ---------------------------------------------------------- helpers

    def _journal(self, entry: Dict[str, Any]) -> None:
        try:
            if get_trade_journal:
                get_trade_journal().write(entry)
        except Exception as e:
            logger.warning(f"trade journal write failed: {e}")


_executor: Optional[Executor] = None


def get_executor() -> Executor:
    global _executor
    if _executor is None:
        _executor = Executor()
    return _executor
