"""Shared fixtures: synthetic OHLCV bars for offline strategy/ledger tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _bars(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(closes, index=idx, dtype="float64")
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": pd.Series(1_000_000, index=idx, dtype="float64"),
        }
    )


@pytest.fixture
def rising_bars() -> pd.DataFrame:
    # 260 strictly rising bars — long enough for the 200d / momentum filters.
    return _bars([100.0 + i for i in range(260)])


@pytest.fixture
def oversold_then_bars() -> pd.DataFrame:
    # Long uptrend, then a sharp 5-day drop to force RSI(2) oversold on the last bar.
    base = [100.0 + i * 0.5 for i in range(255)]
    drop = [base[-1] * f for f in (0.96, 0.92, 0.88, 0.85, 0.82)]
    df = _bars(base + drop)
    # Make the final (oversold) bar a bullish reversal candle: open/low below the
    # close so confirmed_mr's `close > open` entry condition holds. The close
    # series is untouched, so RSI(2) stays deeply oversold (<15) on the last bar.
    last_close = float(df["close"].iloc[-1])
    df.iloc[-1, df.columns.get_loc("open")] = last_close * 0.99
    df.iloc[-1, df.columns.get_loc("low")] = last_close * 0.985
    return df
