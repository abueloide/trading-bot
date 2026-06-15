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


def _multiindex_frame(symbols, periods=300):
    """Mimic yf.download(group_by='ticker'): MultiIndex (ticker, field) columns.

    yfinance returns this shape even for a SINGLE ticker, which is the case the
    batch parser used to mishandle.
    """
    idx = pd.date_range("2024-01-01", periods=periods, freq="D")
    cols = pd.MultiIndex.from_product([symbols, ["Open", "High", "Low", "Close", "Volume"]])
    data = {(s, f): (1.5 if f == "Close" else 1.0) for s in symbols for f in
            ["Open", "High", "Low", "Close", "Volume"]}
    return pd.DataFrame(data, index=idx, columns=cols)


def test_batch_single_ticker_multiindex(monkeypatch):
    """A 1-symbol batch (e.g. a benchmark-only refresh) must not be dropped.

    Modern yfinance returns MultiIndex columns even for one ticker; the old
    `else raw` branch handed that straight to _normalize, which rejected the
    tuple columns and silently returned an empty snapshot.
    """
    raw = _multiindex_frame(["SPY"])
    monkeypatch.setattr("live.yfinance_bars.yf.download", lambda **kw: raw)
    out = YFinanceBars().get_bars_batch(["SPY"], 260)
    assert "SPY" in out
    assert list(out["SPY"].columns) == ["open", "high", "low", "close", "volume"]
    assert out["SPY"]["close"].iloc[-1] == 1.5


def test_batch_multi_ticker_multiindex(monkeypatch):
    raw = _multiindex_frame(["SPY", "AAPL", "MSFT"])
    monkeypatch.setattr("live.yfinance_bars.yf.download", lambda **kw: raw)
    out = YFinanceBars().get_bars_batch(["SPY", "AAPL", "MSFT"], 260)
    assert set(out) == {"SPY", "AAPL", "MSFT"}
    assert len(out["AAPL"]) == 260
