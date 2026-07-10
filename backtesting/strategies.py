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
    sma_short: int = 50,
    sma_long: int = 200,
) -> pd.DataFrame:
    """Strategy B — Confirmed Mean Reversion.

    Entry: RSI(2) < 15 AND close > open (bullish reversal) AND SPY > 200d MA
           AND symbol's 50d MA > 200d MA (per-name uptrend filter).
    Exit: RSI(2) > 65 (engine enforces 7-day time exit).

    The per-name uptrend filter (the same one rsi_mr already had) is what stops
    this horse catching falling knives — the 06-25 audit traced its −9% bleed to
    buying oversold names that kept trending down with no trend gate.
    """
    sig = _empty_signals(df)
    if df.empty or len(df) < sma_long:
        return sig

    r = rsi(df["close"], period=2)
    bullish_candle = df["close"] > df["open"]
    uptrend = sma(df["close"], sma_short) > sma(df["close"], sma_long)

    if spy_close is not None and len(spy_close) >= sma_long:
        spy_aligned = spy_close.reindex(df.index).ffill()
        spy_ma = sma(spy_aligned, sma_long)
        spy_ok = spy_aligned > spy_ma
    else:
        spy_ok = pd.Series(True, index=df.index)

    sig["entry"] = (r < rsi_buy) & bullish_candle & uptrend & spy_ok
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


def strategy_donchian_breakout(
    df: pd.DataFrame, entry_lookback: int = 20, exit_lookback: int = 10,
) -> pd.DataFrame:
    """Strategy D — Donchian channel breakout (trend following).

    Entry: close breaks ABOVE the highest high of the prior ``entry_lookback``
           bars (a 20-day high breakout).
    Exit:  close breaks BELOW the lowest low of the prior ``exit_lookback`` bars
           (a 10-day low). No time stop — the channel itself carries the trend.

    Genuinely distinct from the other horses: momentum buys what already ran for
    6 months, mean-reversion buys what fell, this buys the instant price makes a
    new local high and rides until it makes a new local low. Channels use the
    PRIOR window (``.shift(1)``) so the break is measured against bars before
    today, never against today's own high/low.
    """
    sig = _empty_signals(df)
    if df.empty or len(df) < entry_lookback + 1:
        return sig
    upper = df["high"].rolling(entry_lookback).max().shift(1)
    lower = df["low"].rolling(exit_lookback).min().shift(1)
    sig["entry"] = df["close"] > upper
    sig["exit"] = df["close"] < lower
    return sig


def strategy_trend_pullback(
    df: pd.DataFrame, fast: int = 20, slow: int = 50,
) -> pd.DataFrame:
    """Strategy E — buy resolved pullbacks inside an intermediate uptrend.

    Thesis: the two dead mean-reversion horses (rsi_mr, confirmed_mr) *fought*
    the trend — they bought extreme oversold (RSI2<10/15) with no requirement
    that the pullback had turned back up, so they caught falling knives. This
    buys *with* the trend instead: only when the intermediate trend is up
    (SMA20 > SMA50) AND price has just reclaimed the fast line from below (a
    shallow dip that resolved upward), then rides until the trend itself breaks
    (close < SMA50). No extreme-oversold trigger, no counter-trend bet.

    Regime it expects to work in: intermediate/persistent UPTRENDS (the exact
    walk-forward OOS regime here, ~2024). It should sit out sustained
    downtrends because the SMA20>SMA50 gate is false there. Short lookbacks
    (20/50) so the signal is valid inside a 6-month OOS window — unlike a 200d
    regime line, which never accumulates enough bars per walk-forward window.
    """
    sig = _empty_signals(df)
    if df.empty or len(df) < slow + 1:
        return sig
    fast_ma = sma(df["close"], fast)
    slow_ma = sma(df["close"], slow)
    uptrend = fast_ma > slow_ma
    reclaim = (df["close"] > fast_ma) & (df["close"].shift(1) <= fast_ma.shift(1))
    sig["entry"] = reclaim & uptrend
    sig["exit"] = df["close"] < slow_ma
    return sig


def strategy_pullback_ride(
    df: pd.DataFrame, fast: int = 20, slow: int = 50, trail_lookback: int = 10,
) -> pd.DataFrame:
    """Strategy F — buy the contained dip, ride with a trailing-low stop.

    Thesis: trend_pullback (Field 02, idea 1) died for two named reasons in its
    postmortem: it entered *late* (bought only after price reclaimed SMA20, i.e.
    after the bounce already happened) and exited *early* (bailed on every touch
    of SMA50, cutting winners). Donchian tests the opposite entry (buy new highs,
    ride to an N-day low) and also had no edge. The untested cell is: buy the dip
    *itself* — while price is still under the fast line but held above the slow
    line inside an uptrend (weakness within strength, earlier than a reclaim) —
    and ride with a trailing-low stop instead of a fixed SMA50 exit, so winners
    run until a real trend break rather than every shallow pullback.

    Entry: SMA20 > SMA50 (uptrend) AND close <= SMA20 (in the dip, not extended)
           AND close > SMA50 (dip is contained above the slow line, not a knife).
    Exit:  close < lowest low of the prior ``trail_lookback`` bars (trend break),
           not the first SMA50 kiss.

    Regime it expects to work in: persistent uptrends that pull back shallowly to
    the fast MA. It sits out downtrends (SMA20>SMA50 false) and hard selloffs
    (close>SMA50 false). Short lookbacks (20/50/10) so every gate is valid inside
    the ~6-month walk-forward OOS window.
    """
    sig = _empty_signals(df)
    if df.empty or len(df) < slow + 1:
        return sig
    fast_ma = sma(df["close"], fast)
    slow_ma = sma(df["close"], slow)
    uptrend = fast_ma > slow_ma
    in_dip = (df["close"] <= fast_ma) & (df["close"] > slow_ma)
    trail_stop = df["low"].rolling(trail_lookback).min().shift(1)
    sig["entry"] = uptrend & in_dip
    sig["exit"] = df["close"] < trail_stop
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
    "momentum_news": {
        # Same momentum engine; the news-sentiment veto is applied as a
        # portfolio overlay in the orchestrator (StrategyConfig.news_overlay),
        # not in the per-bar signal, so the registry fn stays identical.
        "fn": strategy_momentum_rotation,
        "type": "momentum",
        "max_hold_days": None,
        "description": "Momentum rotation + AlphaVantage news-sentiment veto",
    },
    "trend_pullback": {
        "fn": strategy_trend_pullback,
        "type": "breakout",  # signal exit (close < SMA50), no time stop
        "max_hold_days": None,
        "description": "Buy resolved pullbacks in an SMA20>SMA50 uptrend",
    },
    "pullback_ride": {
        "fn": strategy_pullback_ride,
        "type": "breakout",  # signal exit (trailing-low break), no time stop
        "max_hold_days": None,
        "description": "Buy contained dip in SMA20>SMA50 uptrend, ride 10-day trailing-low stop",
    },
    "donchian_breakout": {
        "fn": strategy_donchian_breakout,
        "type": "breakout",
        "max_hold_days": None,  # exits on a 10-day-low break, not on time
        "description": "Donchian 20/10 channel breakout (trend following)",
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
