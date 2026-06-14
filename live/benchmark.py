"""Buy-and-hold benchmark for the horse race.

A horse winning the race is not the same as beating the market. This computes
a passive buy-and-hold return for a reference symbol (SPY) from the race
inception date to the latest bar, so each strategy's return can be read as
*alpha* (return minus benchmark) instead of a raw number. No edge claim is
honest without it.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, Optional

import pandas as pd


def compute_benchmark(
    snapshot: Dict[str, pd.DataFrame],
    symbol: str,
    inception: date,
) -> Optional[dict]:
    """Buy-and-hold return for ``symbol`` from ``inception`` to the last bar.

    Anchors at the first bar dated on or after ``inception`` (so a weekend or
    holiday inception lands on the next real session). Returns ``None`` when the
    symbol is absent, has no bars, or has no bar on/after inception.
    """
    df = snapshot.get(symbol)
    if df is None or len(df) == 0:
        return None

    start_close: Optional[float] = None
    start_day: Optional[date] = None
    for ts, close in zip(df.index, df["close"]):
        day = ts.date() if hasattr(ts, "date") else ts
        if day >= inception:
            start_close = float(close)
            start_day = day
            break

    if start_close is None or start_close <= 0:
        return None

    end_close = float(df["close"].iloc[-1])
    return {
        "symbol": symbol,
        "return_pct": ((end_close / start_close) - 1.0) * 100.0,
        "start_close": start_close,
        "end_close": end_close,
        "start_date": str(start_day),
    }
