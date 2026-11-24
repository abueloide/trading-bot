"""yfinance-backed BarProvider: trailing daily OHLCV with lowercase columns."""
from __future__ import annotations

import logging
import time
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
_REQUIRED = {"open", "high", "low", "close", "volume"}
_BATCH_CHUNK = 100  # symbols per yf.download call — one network round-trip per chunk.
# Retry backoff (seconds) for symbols still missing after the first pass. DNS
# blips (Could not resolve host: query2.finance.yahoo.com) are transient and
# usually clear within a few seconds; a real delisting just fails again, cheap
# since only the missing symbols (not the whole universe) are retried.
_RETRY_BACKOFFS_S = (5, 15)


def _normalize(raw: Optional[pd.DataFrame], lookback: int) -> Optional[pd.DataFrame]:
    """Rename to lowercase OHLCV, validate, drop empty rows, tail to lookback."""
    if raw is None or raw.empty:
        return None
    df = raw.rename(columns=_RENAME)
    if not _REQUIRED.issubset(df.columns):
        return None
    df = df[["open", "high", "low", "close", "volume"]].dropna(how="all")
    if df.empty:
        return None
    return df.tail(lookback)


def drop_in_progress_bars(
    snapshot: Dict[str, pd.DataFrame], today: date
) -> Dict[str, pd.DataFrame]:
    """Drop any trailing bar dated ``today`` or later.

    A daily run during market hours gets an in-progress (partial) bar for the
    live session: yfinance stamps it with today's date but the close has not
    settled. Strategies, the report, and the equity snapshot must act on settled
    closes only — a mid-session print taints return/alpha (e.g. the 2026-06-22
    snapshot stamped momentum at a media-session price). Symbols left empty after
    the drop are omitted (callers already guard on missing symbols).

    ponytail: drops today's bar unconditionally; a rare after-close manual run
    loses one day of recency (the next run recovers it), which a daily-close
    strategy never needs. Upgrade path if intraday runs ever matter: gate on a
    market-close clock instead of the calendar day.
    """
    out: Dict[str, pd.DataFrame] = {}
    for sym, df in snapshot.items():
        if df is not None and len(df) and df.index[-1].date() >= today:
            df = df.iloc[:-1]
        if df is not None and len(df):
            out[sym] = df
    return out


class YFinanceBars:
    def get_bars(self, symbol: str, lookback: int) -> Optional[pd.DataFrame]:
        # Fetch a bit more than lookback calendar days to clear weekends/holidays.
        period_days = int(lookback * 1.6) + 10
        try:
            raw = yf.Ticker(symbol).history(period=f"{period_days}d", interval="1d")
        except Exception as e:
            logger.warning("yfinance fetch failed for %s: %s", symbol, e)
            return None
        out = _normalize(raw, lookback)
        if out is None:
            logger.warning("yfinance returned no usable bars for %s", symbol)
        return out

    def _download_chunk(self, chunk: List[str], period_days: int, lookback: int,
                         out: Dict[str, pd.DataFrame], chunk_label: str) -> None:
        """Download one chunk and merge any usable symbols into ``out`` in place."""
        try:
            raw = yf.download(
                tickers=chunk,
                period=f"{period_days}d",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as e:
            logger.warning("batch download failed for chunk %s (%d syms): %s", chunk_label, len(chunk), e)
            return
        if raw is None or raw.empty:
            return
        # group_by="ticker" yields MultiIndex (ticker, field) columns — even
        # for a single ticker, so we always index by symbol when the columns
        # are MultiIndexed. A flat frame only maps to a lone-symbol chunk.
        multiindexed = isinstance(raw.columns, pd.MultiIndex)
        for sym in chunk:
            if multiindexed:
                try:
                    sub = raw[sym]
                except (KeyError, IndexError):
                    continue
            else:
                sub = raw if len(chunk) == 1 else None
            if sub is None:
                continue
            norm = _normalize(sub, lookback)
            if norm is not None:
                out[sym] = norm

    def get_bars_batch(self, symbols: List[str], lookback: int) -> Dict[str, pd.DataFrame]:
        """One snapshot of the whole universe: yf.download in chunks of ~100.

        Per-symbol fetching 500 names rate-limits Yahoo fast; batching collapses
        each chunk into a single round-trip. Symbols still missing after the
        first pass are retried (with backoff) — a transient DNS/network blip can
        otherwise silently strand a HELD symbol without a fresh mark. Symbols
        with no usable data after retries are omitted (callers already guard on
        missing).
        """
        period_days = int(lookback * 1.6) + 10
        out: Dict[str, pd.DataFrame] = {}
        unique = list(dict.fromkeys(symbols))  # dedupe, preserve order
        for i in range(0, len(unique), _BATCH_CHUNK):
            chunk = unique[i:i + _BATCH_CHUNK]
            self._download_chunk(chunk, period_days, lookback, out, chunk_label=str(i))

        missing = [s for s in unique if s not in out]
        for attempt, backoff in enumerate(_RETRY_BACKOFFS_S, start=1):
            if not missing:
                break
            logger.warning("retry %d/%d: %d symbols still missing bars, backing off %ds: %s",
                            attempt, len(_RETRY_BACKOFFS_S), len(missing), backoff, ", ".join(missing))
            time.sleep(backoff)
            for i in range(0, len(missing), _BATCH_CHUNK):
                chunk = missing[i:i + _BATCH_CHUNK]
                self._download_chunk(chunk, period_days, lookback, out, chunk_label=f"retry{attempt}-{i}")
            missing = [s for s in unique if s not in out]

        logger.info("batch bars: %d/%d symbols returned usable data", len(out), len(unique))
        return out


class CachedBars:
    """Serves a pre-fetched {symbol: DataFrame} snapshot — zero extra network.

    Lets the entrypoint download the whole universe once and reuse it for the
    orchestrator cycle, the rebalance-day check, and end-of-run marks.
    """

    def __init__(self, snapshot: Dict[str, pd.DataFrame]) -> None:
        self._snapshot = dict(snapshot)

    def get_bars(self, symbol: str, lookback: int) -> Optional[pd.DataFrame]:
        df = self._snapshot.get(symbol)
        return None if df is None else df.tail(lookback)

    def get_bars_batch(self, symbols: List[str], lookback: int) -> Dict[str, pd.DataFrame]:
        return {s: self._snapshot[s].tail(lookback)
                for s in symbols if self._snapshot.get(s) is not None}

    def snapshot(self) -> Dict[str, pd.DataFrame]:
        return self._snapshot
