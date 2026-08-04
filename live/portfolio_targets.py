"""Cross-sectional target construction for the paper horse race.

The per-symbol backtest strategies only say BUY/SELL/HOLD for ONE name. Real
momentum and diversified mean-reversion are *portfolio* decisions: rank the whole
universe, then pick the top names. This module turns a universe of bars into
ranked candidate lists. It is pure (no I/O, no broker) and offline-testable.

    momentum_scores(bars)      -> {symbol: score}              (higher = stronger)
    momentum_top(bars, n)      -> [symbol, ...]                (top-n, score > 0)
    oversold_candidates(name, bars, exclude)
                               -> [(symbol, rsi2, price), ...] (most oversold first)
    exit_signals(name, bars, held)
                               -> [symbol, ...]                (held names to close)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from backtesting.strategies import (
    STRATEGY_REGISTRY,
    rsi,
    strategy_momentum_rotation,
)


def _usable(df: Optional[pd.DataFrame]) -> bool:
    return df is not None and len(df) > 0 and "close" in df.columns


def momentum_scores(bars_by_symbol: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Trailing-momentum score per symbol (6m return skipping the last month)."""
    scores: Dict[str, float] = {}
    for sym, df in bars_by_symbol.items():
        if not _usable(df):
            continue
        try:
            sig = strategy_momentum_rotation(df)
        except Exception:
            continue
        if "momentum_score" not in sig.columns or len(sig) == 0:
            continue
        val = sig["momentum_score"].iloc[-1]
        if pd.notna(val):
            scores[sym] = float(val)
    return scores


def momentum_top(
    bars_by_symbol: Dict[str, pd.DataFrame],
    n: int,
    *,
    sector_of: Optional[Callable[[str], str]] = None,
    max_per_sector: Optional[int] = None,
) -> List[str]:
    """Top-`n` symbols by momentum score, positive scores only (descending).

    When `sector_of` and `max_per_sector` are given, enforce a per-sector cap
    while filling the basket: walk names strongest-first, skip any whose sector
    already holds `max_per_sector` picks. Without a cap, behaviour is unchanged.
    This stops the basket from collapsing into one hot sector (e.g. 12/15 semis)
    — the difference between an edge and a leveraged single-theme bet.
    """
    scores = momentum_scores(bars_by_symbol)
    ranked = [s for s, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if score > 0]
    if sector_of is None or max_per_sector is None:
        return ranked[:n]
    picked: List[str] = []
    per_sector: Dict[str, int] = {}
    for sym in ranked:
        sec = sector_of(sym)
        if per_sector.get(sec, 0) >= max_per_sector:
            continue
        picked.append(sym)
        per_sector[sec] = per_sector.get(sec, 0) + 1
        if len(picked) >= n:
            break
    return picked


def oversold_candidates(
    strategy_name: str,
    bars_by_symbol: Dict[str, pd.DataFrame],
    exclude: Optional[set] = None,
) -> List[Tuple[str, float, float]]:
    """Symbols whose last bar fires this MR strategy's entry, ranked by RSI(2) asc.

    Returns (symbol, rsi2, last_price) tuples, most oversold first. `exclude` skips
    names already held so the caller never doubles down.
    """
    exclude = exclude or set()
    fn = STRATEGY_REGISTRY[strategy_name]["fn"]
    out: List[Tuple[str, float, float]] = []
    for sym, df in bars_by_symbol.items():
        if sym in exclude or not _usable(df):
            continue
        try:
            sig = fn(df)
        except Exception:
            continue
        if len(sig) == 0 or not bool(sig["entry"].iloc[-1]):
            continue
        r2 = rsi(df["close"], period=2).iloc[-1]
        price = float(df["close"].iloc[-1])
        if pd.isna(r2) or price <= 0:
            continue
        out.append((sym, float(r2), price))
    out.sort(key=lambda t: t[1])
    return out


def breakout_candidates(
    strategy_name: str,
    bars_by_symbol: Dict[str, pd.DataFrame],
    exclude: Optional[set] = None,
    entry_lookback: int = 20,
) -> List[Tuple[str, float, float]]:
    """Symbols whose last bar fires this breakout strategy's entry, ranked by
    breakout strength (how far close cleared the prior N-day high), strongest first.

    Returns (symbol, strength, last_price) tuples — same shape as
    ``oversold_candidates`` so the orchestrator fills slots identically. When
    more names break out than there are open slots, the ones that cleared the
    channel most decisively win. ``entry_lookback`` must match the strategy's
    entry window (20, the registry default).
    """
    exclude = exclude or set()
    fn = STRATEGY_REGISTRY[strategy_name]["fn"]
    out: List[Tuple[str, float, float]] = []
    for sym, df in bars_by_symbol.items():
        if sym in exclude or not _usable(df):
            continue
        if "high" not in df.columns or len(df) < entry_lookback + 1:
            continue
        try:
            sig = fn(df)
        except Exception:
            continue
        if len(sig) == 0 or not bool(sig["entry"].iloc[-1]):
            continue
        prior_high = float(df["high"].iloc[-(entry_lookback + 1):-1].max())
        price = float(df["close"].iloc[-1])
        if prior_high <= 0 or price <= 0:
            continue
        out.append((sym, price / prior_high - 1.0, price))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def exit_signals(
    strategy_name: str,
    bars_by_symbol: Dict[str, pd.DataFrame],
    held: set,
) -> List[str]:
    """Held symbols whose last bar fires this strategy's exit condition."""
    fn = STRATEGY_REGISTRY[strategy_name]["fn"]
    out: List[str] = []
    for sym in held:
        df = bars_by_symbol.get(sym)
        if not _usable(df):
            continue
        try:
            sig = fn(df)
        except Exception:
            continue
        if len(sig) and bool(sig["exit"].iloc[-1]):
            out.append(sym)
    return out


def event_candidates(
    strategy_name: str,
    bars_by_symbol: Dict[str, pd.DataFrame],
    exclude: Optional[set] = None,
) -> List[Tuple[str, float, float]]:
    """Símbolos cuya última barra dispara la entrada de una estrategia de evento.

    Sin ranking (rank=0.0): en el carril-event el calendario decide, no un score
    — todos los símbolos elegibles entran equal-weight el día del evento.
    """
    exclude = exclude or set()
    fn = STRATEGY_REGISTRY[strategy_name]["fn"]
    out: List[Tuple[str, float, float]] = []
    for sym, df in bars_by_symbol.items():
        if sym in exclude or not _usable(df):
            continue
        try:
            sig = fn(df)
        except Exception:
            continue
        if len(sig) == 0 or not bool(sig["entry"].iloc[-1]):
            continue
        price = float(df["close"].iloc[-1])
        if price > 0:
            out.append((sym, 0.0, price))
    return sorted(out)
