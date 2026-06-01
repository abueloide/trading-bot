from __future__ import annotations

import pandas as pd

from live.yfinance_bars import YFinanceBars


def test_normalizes_columns_and_tails(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=300, freq="D")
    raw = pd.DataFrame({
        "Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 100,
        "Dividends": 0.0, "Stock Splits": 0.0,
    }, index=idx)

    class FakeTicker:
        def __init__(self, sym): pass
        def history(self, period, interval): return raw

    monkeypatch.setattr("live.yfinance_bars.yf.Ticker", FakeTicker)
    bars = YFinanceBars().get_bars("AAPL", 260)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert len(bars) == 260
    assert bars["close"].iloc[-1] == 1.5
