"""yfinance-backed BarProvider: trailing daily OHLCV with lowercase columns."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YFinanceBars:
    def get_bars(self, symbol: str, lookback: int) -> Optional[pd.DataFrame]:
        # Fetch a bit more than lookback calendar days to clear weekends/holidays.
        period_days = int(lookback * 1.6) + 10
        try:
            raw = yf.Ticker(symbol).history(period=f"{period_days}d", interval="1d")
        except Exception as e:
            logger.warning("yfinance fetch failed for %s: %s", symbol, e)
            return None
        if raw is None or raw.empty:
            return None
        df = raw.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            logger.warning("yfinance returned unexpected columns for %s: %s", symbol, list(df.columns))
            return None
        return df[["open", "high", "low", "close", "volume"]].tail(lookback)
