#!/usr/bin/env python3
"""
Live-trading wrappers for the mean-reversion strategies.

Reuses the pure functions in ``backtesting.strategies`` so the same logic
runs in backtest and in production. Returns signal dicts with the
``strategy_type='mean_reversion'`` field which tells the executor to use
TIME-BASED exits (no stop-loss) — this is research-backed and intentional.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from backtesting.strategies import strategy_confirmed_mr, strategy_rsi_mr_vix


def _build_dataframe(market_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Build an OHLCV DataFrame from a DataManager market_data dict."""
    history = market_data.get("price_history") or []
    volumes = market_data.get("volume_history") or []
    timestamps = market_data.get("timestamps") or []
    if not history or not volumes:
        return None

    n = min(len(history), len(volumes))
    closes = list(history[-n:])
    vols = list(volumes[-n:])
    if timestamps and len(timestamps) >= n:
        index = pd.DatetimeIndex(timestamps[-n:])
    else:
        # Synthesize a daily index ending today.
        index = pd.date_range(end=datetime.utcnow().date(), periods=n, freq="B")

    # Without OHLC we approximate open/high/low from close — sufficient for the
    # RSI(2) and reversal-candle filters at signal-emission time. Real OHLC is
    # used in backtests where the data has them.
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": vols,
        },
        index=index,
    )


def evaluate_rsi_mr(
    symbol: str,
    market_data: Dict[str, Any],
    *,
    vix_rank: Optional[float] = None,
    spy_uptrend: bool = True,
) -> Optional[Dict[str, Any]]:
    """Strategy A — RSI(2) Mean Reversion + VIX Filter.

    Returns a signal dict (action='BUY' / None) with strategy_type set so
    the executor uses time-based exit (max 10 days, no SL).
    """
    df = _build_dataframe(market_data)
    if df is None or len(df) < 220:
        return None

    if vix_rank is not None and vix_rank >= 50:
        return None
    if not spy_uptrend:
        return None

    sig = strategy_rsi_mr_vix(df)
    if sig.empty or "entry" not in sig.columns:
        return None
    if not bool(sig["entry"].iloc[-1]):
        return None

    last_close = float(df["close"].iloc[-1])
    return {
        "symbol": symbol,
        "action": "BUY",
        "strategy": "rsi_mr",
        "strategy_type": "mean_reversion",
        "entry_price": last_close,
        "max_hold_days": 10,
        "stop_loss": None,        # explicit: NO stop-loss for mean reversion
        "take_profit": None,
        "exit_rule": "rsi_2 > 70 OR 10 trading days elapsed",
        "reasoning": "RSI(2) oversold with VIX rank < 50 and SPY uptrend",
        "timestamp": datetime.utcnow(),
    }


def evaluate_confirmed_mr(
    symbol: str,
    market_data: Dict[str, Any],
    *,
    spy_uptrend: bool = True,
) -> Optional[Dict[str, Any]]:
    """Strategy B — Confirmed Mean Reversion."""
    df = _build_dataframe(market_data)
    if df is None or len(df) < 220:
        return None
    if not spy_uptrend:
        return None

    sig = strategy_confirmed_mr(df)
    if sig.empty or "entry" not in sig.columns:
        return None
    if not bool(sig["entry"].iloc[-1]):
        return None

    last_close = float(df["close"].iloc[-1])
    return {
        "symbol": symbol,
        "action": "BUY",
        "strategy": "confirmed_mr",
        "strategy_type": "mean_reversion",
        "entry_price": last_close,
        "max_hold_days": 7,
        "stop_loss": None,
        "take_profit": None,
        "exit_rule": "rsi_2 > 65 OR 7 trading days elapsed",
        "reasoning": "RSI(2) oversold with bullish reversal candle and SPY uptrend",
        "timestamp": datetime.utcnow(),
    }
