#!/usr/bin/env python3
"""
Momentum Rotation strategy — runs MONTHLY, separate from the daily scan.

Universe: S&P 500 large/mid cap.
Ranking: 6-month return excluding the most recent month (skip 21d, lookback 126d).
Action: Buy top 10 equal-weight on the 1st trading day of each month.
Filter: Only invest when SPY > 200-day MA, otherwise 100% cash.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def compute_momentum_score(
    closes: pd.Series,
    skip_recent_days: int = 21,
    lookback_days: int = 126,
) -> Optional[float]:
    """Score = return from t-(skip+lookback) to t-skip. None if insufficient data."""
    if closes is None or len(closes) < skip_recent_days + lookback_days + 1:
        return None
    end = closes.shift(skip_recent_days)
    start = closes.shift(skip_recent_days + lookback_days)
    score = (end / start) - 1.0
    last = score.iloc[-1]
    if pd.isna(last):
        return None
    return float(last)


def rank_universe(
    price_histories: Dict[str, pd.Series],
    skip_recent_days: int = 21,
    lookback_days: int = 126,
) -> List[Tuple[str, float]]:
    """Return [(symbol, score), ...] sorted descending by momentum score."""
    scored: List[Tuple[str, float]] = []
    for symbol, series in price_histories.items():
        score = compute_momentum_score(series, skip_recent_days, lookback_days)
        if score is None:
            continue
        scored.append((symbol, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def select_top_n(
    ranked: List[Tuple[str, float]],
    n: int = 10,
    require_positive: bool = True,
) -> List[str]:
    selected = [s for s, score in ranked if (not require_positive) or score > 0]
    return selected[:n]


def is_first_trading_day_of_month(d: Optional[date] = None) -> bool:
    """True if `d` (default today) is the first weekday of its month.

    Doesn't account for market holidays — caller can override by passing in
    the actual list of trading days. The bot's APScheduler should also be
    holiday-aware via the executor's market-hours check.
    """
    d = d or date.today()
    if d.weekday() >= 5:
        return False
    # Walk back day-by-day; if we cross into the previous month before hitting
    # a weekday, this is the first trading day.
    from datetime import timedelta
    prev = d - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev.month != d.month


def build_rebalance_orders(
    current_holdings: Dict[str, float],
    target_symbols: List[str],
    portfolio_value: float,
    max_positions: int = 10,
) -> List[Dict[str, Any]]:
    """Compute SELL/BUY orders to rebalance to equal-weight ``target_symbols``.

    Returns a list of action dicts (no execution). The executor consumes them.
    """
    orders: List[Dict[str, Any]] = []
    target_set = set(target_symbols[:max_positions])

    # Sells: anything held but not in target
    for symbol, value in current_holdings.items():
        if symbol not in target_set:
            orders.append({
                "symbol": symbol,
                "action": "SELL",
                "strategy": "momentum_rotation",
                "strategy_type": "momentum",
                "reasoning": "Removed from monthly momentum top-N",
                "timestamp": datetime.utcnow(),
            })

    # Buys / rebalances: target symbols equal-weighted
    if not target_set or portfolio_value <= 0:
        return orders
    target_dollar_per_position = portfolio_value / len(target_set)
    for symbol in target_set:
        existing = current_holdings.get(symbol, 0.0)
        delta = target_dollar_per_position - existing
        # Only act on meaningful drift (>5% of target weight)
        if abs(delta) < 0.05 * target_dollar_per_position:
            continue
        orders.append({
            "symbol": symbol,
            "action": "BUY" if delta > 0 else "SELL",
            "strategy": "momentum_rotation",
            "strategy_type": "momentum",
            "rebalance_dollars": abs(delta),
            "reasoning": f"Monthly momentum rebalance to ${target_dollar_per_position:.2f}",
            "timestamp": datetime.utcnow(),
        })
    return orders


def should_be_in_cash(spy_uptrend: bool) -> bool:
    """When SPY < 200-day MA, momentum strategy goes 100% to cash."""
    return not spy_uptrend
