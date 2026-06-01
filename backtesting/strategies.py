#!/usr/bin/env python3
"""
Pure-function strategies operating on pandas DataFrames of OHLCV bars.

Each strategy returns a DataFrame indexed identically to the input with
boolean ``entry`` and ``exit`` columns. No API calls, no I/O — strategies
are deterministic and offline-testable.

Strategy contract:
    fn(df: DataFrame, **params) -> DataFrame[index, "entry": bool, "exit": bool]

Where df has columns: open, high, low, close, volume (lowercase).

Strategies expose ``STRATEGY_TYPE`` so the executor knows whether to use
time-based exits (mean_reversion / momentum) or bracket orders (breakout).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- indicators

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder's smoothing approximation via EWM
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_dn = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    mid = sma(series, period)
    sd = series.rolling(period, min_periods=1).std(ddof=0)
    return mid + std * sd, mid, mid - std * sd


# ------------------------------------------------------------ helper builders

def _empty_signals(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"entry": False, "exit": False},
        index=df.index,
    )


# --------------------------------------------------------------- strategies

def strategy_rsi_mr_vix(
    df: pd.DataFrame,
    vix_rank_series: Optional[pd.Series] = None,
    spy_close: Optional[pd.Series] = None,
    rsi_buy: int = 10,
    rsi_sell: int = 70,
    vix_threshold: float = 50.0,
    sma_short: int = 50,
    sma_long: int = 200,
) -> pd.DataFrame:
    """Strategy A — RSI(2) Mean Reversion + VIX Filter.

    Entry: RSI(2) < rsi_buy AND VIX rank < threshold AND SPY > 200d MA
           AND symbol's 50d MA > 200d MA (uptrend filter).
    Exit: RSI(2) > rsi_sell  (time-based exit handled by engine, max 10 days).
    """
    sig = _empty_signals(df)
    if df.empty or len(df) < sma_long:
        return sig

    r = rsi(df["close"], period=2)
    sym_short = sma(df["close"], sma_short)
    sym_long = sma(df["close"], sma_long)
    uptrend = sym_short > sym_long

    if vix_rank_series is not None:
        vix_ok = vix_rank_series.reindex(df.index).ffill() < vix_threshold
    else:
        vix_ok = pd.Series(True, index=df.index)

    if spy_close is not None and len(spy_close) >= sma_long:
        spy_aligned = spy_close.reindex(df.index).ffill()
        spy_ma = sma(spy_aligned, sma_long)
        spy_ok = spy_aligned > spy_ma
    else:
        spy_ok = pd.Series(True, index=df.index)

    sig["entry"] = (r < rsi_buy) & uptrend & vix_ok & spy_ok
    sig["exit"] = r > rsi_sell
    return sig


def strategy_confirmed_mr(
    df: pd.DataFrame,
    spy_close: Optional[pd.Series] = None,
    rsi_buy: int = 15,
    rsi_sell: int = 65,
    sma_long: int = 200,
) -> pd.DataFrame:
    """Strategy B — Confirmed Mean Reversion.

    Entry: RSI(2) < 15 AND close > open (bullish reversal) AND SPY > 200d MA.
    Exit: RSI(2) > 65 (engine enforces 7-day time exit).
    """
    sig = _empty_signals(df)
    if df.empty or len(df) < 30:
        return sig

    r = rsi(df["close"], period=2)
    bullish_candle = df["close"] > df["open"]

    if spy_close is not None and len(spy_close) >= sma_long:
        spy_aligned = spy_close.reindex(df.index).ffill()
        spy_ma = sma(spy_aligned, sma_long)
        spy_ok = spy_aligned > spy_ma
    else:
        spy_ok = pd.Series(True, index=df.index)

    sig["entry"] = (r < rsi_buy) & bullish_candle & spy_ok
    sig["exit"] = r > rsi_sell
    return sig


def strategy_momentum_rotation(
    df: pd.DataFrame,
    skip_recent_days: int = 21,
    lookback_days: int = 126,
) -> pd.DataFrame:
    """Strategy C — momentum signal generator (per-symbol view).

    The actual cross-sectional ranking and monthly rebalance is handled by the
    engine — this function exposes the *score* via a synthetic ``entry`` flag
    that fires when the symbol's trailing-momentum score is positive.
    The engine reads ``df["momentum_score"]`` for ranking.
    """
    sig = _empty_signals(df)
    if df.empty or len(df) < lookback_days + skip_recent_days + 2:
        return sig

    # Momentum = return from t-(lookback+skip) to t-skip, exclude last month.
    shifted = df["close"].shift(skip_recent_days)
    base = df["close"].shift(skip_recent_days + lookback_days)
    momentum_score = (shifted / base) - 1.0
    sig["momentum_score"] = momentum_score
    sig["entry"] = momentum_score > 0
    sig["exit"] = momentum_score < 0
    return sig


def strategy_ema_crossover(
    df: pd.DataFrame, fast: int = 10, slow: int = 50,
) -> pd.DataFrame:
    """Legacy strategy retained for baseline comparison."""
    sig = _empty_signals(df)
    if df.empty or len(df) < slow + 1:
        return sig
    f = ema(df["close"], fast)
    s = ema(df["close"], slow)
    cross_up = (f > s) & (f.shift(1) <= s.shift(1))
    cross_dn = (f < s) & (f.shift(1) >= s.shift(1))
    sig["entry"] = cross_up
    sig["exit"] = cross_dn
    return sig


def strategy_bollinger_breakout(
    df: pd.DataFrame, period: int = 20, std: float = 2.0,
) -> pd.DataFrame:
    """Legacy Bollinger breakout strategy retained for baseline comparison."""
    sig = _empty_signals(df)
    if df.empty or len(df) < period + 1:
        return sig
    upper, mid, lower = bollinger(df["close"], period, std)
    sig["entry"] = (df["close"].shift(1) <= upper.shift(1)) & (df["close"] > upper)
    sig["exit"] = df["close"] < mid
    return sig


# ---------------------------------------------------------- registry

STRATEGY_REGISTRY: Dict[str, Dict[str, object]] = {
    "rsi_mr": {
        "fn": strategy_rsi_mr_vix,
        "type": "mean_reversion",
        "max_hold_days": 10,
        "description": "RSI(2) Mean Reversion + VIX Filter",
    },
    "confirmed_mr": {
        "fn": strategy_confirmed_mr,
        "type": "mean_reversion",
        "max_hold_days": 7,
        "description": "RSI(2) Mean Reversion + Bullish Candle Confirmation",
    },
    "momentum_rotation": {
        "fn": strategy_momentum_rotation,
        "type": "momentum",
        "max_hold_days": None,  # rebalanced monthly
        "description": "6-month momentum, skip last month, monthly rebalance",
    },
    "ema_crossover": {
        "fn": strategy_ema_crossover,
        "type": "breakout",
        "max_hold_days": None,
        "description": "Legacy 10/50 EMA crossover",
    },
    "bollinger_breakout": {
        "fn": strategy_bollinger_breakout,
        "type": "breakout",
        "max_hold_days": None,
        "description": "Legacy Bollinger band breakout",
    },
}


def get_strategy(name: str) -> Dict[str, object]:
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name]
