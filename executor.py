#!/usr/bin/env python3
"""
Order executor — Alpaca trading client wrapper.

- Bracket orders for breakout/momentum (entry + SL + TP)
- Market orders with TIME-BASED exit for mean reversion (NO stop-loss)
- PDT compliance tracking (3 day trades / rolling 5 days when equity < $25K)
- Market-hours enforcement (9:30 - 16:00 ET, Mon-Fri)
- Order logging with strategy attribution into the trade journal
"""

from __future__ import annotations

import logging
import os
from collections import deque
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import (
        ClosePositionRequest,
        LimitOrderRequest,
        MarketOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
    )
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    from config import ALPACA_CONFIG, RISK_CONFIG
except ImportError:
    ALPACA_CONFIG = {"api_key": "", "secret_key": "", "paper": True}
    RISK_CONFIG = {
        "max_open_positions": 4,
        "max_day_trades_per_week": 3,
        "pdt_equity_threshold": 25_000,
    }

try:
    from trade_journal import get_trade_journal
except ImportError:
    get_trade_journal = None


# US market hours expressed in UTC (NYSE 9:30 - 16:00 ET; ignores DST nuance —
# in production use exchange calendars). Approximate to ET == UTC-5/-4.
_MARKET_OPEN_ET = time(9, 30)
_MARKET_CLOSE_ET = time(16, 0)


def _now_et() -> datetime:
    # Approximation: use UTC-5 (EST) — DST ignored deliberately to keep this
    # dependency-free. Replace with `pandas_market_calendars` for production.
    return datetime.utcnow() - timedelta(hours=5)


def is_market_open(now: Optional[datetime] = None) -> bool:
    now = now or _now_et()
    if now.weekday() > 4:
        return False
    return _MARKET_OPEN_ET <= now.time() < _MARKET_CLOSE_ET


# ============================================================================
# PDT tracker
# ============================================================================

class PDTTracker:
    """Rolling 5-business-day day-trade counter. Day trade = open+close same day."""

    def __init__(self, max_per_window: int = 3, window_days: int = 5):
        self.max_per_window = max_per_window
        self.window_days = window_days
        self._day_trades: deque = deque()  # list of datetime when each occurred

    def record_round_trip(self, when: Optional[datetime] = None) -> None:
        when = when or datetime.utcnow()
        self._day_trades.append(when)
        self._prune(when)

    def count(self, now: Optional[datetime] = None) -> int:
        now = now or datetime.utcnow()
        self._prune(now)
        return len(self._day_trades)

    def can_day_trade(self, equity: float, now: Optional[datetime] = None) -> bool:
        if equity >= RISK_CONFIG.get("pdt_equity_threshold", 25_000):
            return True
        return self.count(now) < self.max_per_window

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(days=self.window_days)
        while self._day_trades and self._day_trades[0] < cutoff:
            self._day_trades.popleft()


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

    def remove(self, symbol: str) -> None:
        self._tracked.pop(symbol, None)

    def expired_symbols(self, now: Optional[datetime] = None) -> List[str]:
        now = now or datetime.utcnow()
        out: List[str] = []
        for symbol, info in list(self._tracked.items()):
            if (now - info["entry_time"]).days >= info["max_hold_days"]:
                out.append(symbol)
        return out


# ============================================================================
# Executor
# ============================================================================

class Executor:
    """Alpaca trading wrapper. All methods are safe to call even when Alpaca
    credentials are missing — they log and return None instead of crashing."""

    def __init__(self):
        self._client = None
        self.pdt = PDTTracker(
            max_per_window=RISK_CONFIG.get("max_day_trades_per_week", 3),
        )
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

    def place_bracket_order(
        self,
        symbol: str,
        qty: float,
        stop_loss: float,
        take_profit: float,
        side: str = "BUY",
        strategy: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Bracket order for breakout/momentum strategies."""
        if not self._client:
            logger.warning(f"No Alpaca client — cannot place bracket order for {symbol}")
            return None
        if not is_market_open():
            logger.warning(f"Market closed — rejecting bracket order for {symbol}")
            return None
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=round(qty, 4),
                side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                order_class="bracket",
                stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
                take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
            )
            order = self._client.submit_order(req)
            self._journal({
                "timestamp": datetime.utcnow(),
                "symbol": symbol,
                "action": side.upper(),
                "strategy": strategy,
                "strategy_type": "breakout",
                "entry_price": None,
                "reasoning": "bracket order",
                "order_id": str(order.id),
            })
            return {"id": str(order.id), "symbol": symbol, "qty": qty, "type": "bracket"}
        except Exception as e:
            logger.error(f"Bracket order failed for {symbol}: {e}")
            return None

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

    def check_time_exits(self) -> List[Dict[str, Any]]:
        """Close positions whose max_hold_days have elapsed."""
        closed: List[Dict[str, Any]] = []
        for symbol in self.time_exits.expired_symbols():
            result = self.close_position(symbol, reason="time_exit")
            if result:
                closed.append(result)
                self.time_exits.remove(symbol)
        return closed

    def close_position(self, symbol: str, reason: str = "") -> Optional[Dict[str, Any]]:
        if not self._client:
            return None
        if not is_market_open():
            logger.warning(f"Market closed — deferring close for {symbol}")
            return None
        try:
            self._client.close_position(symbol)
            self._journal({
                "timestamp": datetime.utcnow(),
                "symbol": symbol,
                "action": "SELL",
                "reasoning": reason or "manual close",
            })
            self.time_exits.remove(symbol)
            return {"symbol": symbol, "closed": True, "reason": reason}
        except Exception as e:
            logger.error(f"close_position failed for {symbol}: {e}")
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
