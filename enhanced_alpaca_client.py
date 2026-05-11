#!/usr/bin/env python3
"""
Enhanced Alpaca API Client - Drop-in replacement for EnhancedBinanceClient.

Preserves the same return-shape interface as `EnhancedBinanceClient` so the
rest of the trading system can consume Alpaca data without modification.

Methods:
    get_enhanced_ticker(symbol)           -> snapshot dict
    get_enhanced_klines(symbol, interval) -> list of bar dicts
    get_order_book_enhanced(symbol)       -> dict with bids/asks (Level 1)
    get_api_stats()                       -> request stats
"""

import os
import time
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import (
        StockBarsRequest,
        StockLatestQuoteRequest,
        StockSnapshotRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed. EnhancedAlpacaClient will run in degraded mode.")


# Alpaca free-tier rate limit is ~200 req/min for IEX feed.
DEFAULT_RATE_LIMIT_REQUESTS_PER_MIN = 200
DEFAULT_TIMEOUT_SECONDS = 15

INTERVAL_MAP = {
    "1m": (1, TimeFrameUnit.Minute) if ALPACA_AVAILABLE else None,
    "5m": (5, TimeFrameUnit.Minute) if ALPACA_AVAILABLE else None,
    "15m": (15, TimeFrameUnit.Minute) if ALPACA_AVAILABLE else None,
    "30m": (30, TimeFrameUnit.Minute) if ALPACA_AVAILABLE else None,
    "1h": (1, TimeFrameUnit.Hour) if ALPACA_AVAILABLE else None,
    "4h": (4, TimeFrameUnit.Hour) if ALPACA_AVAILABLE else None,
    "1d": (1, TimeFrameUnit.Day) if ALPACA_AVAILABLE else None,
}


class EnhancedAlpacaClient:
    """Alpaca market-data client with the EnhancedBinanceClient interface."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_REQUESTS_PER_MIN,
    ):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self.rate_limit_per_minute = rate_limit_per_minute
        self.min_delay = 60.0 / max(rate_limit_per_minute, 1)
        self.last_request_time = 0.0
        self.request_history: deque = deque(maxlen=rate_limit_per_minute * 2)

        self.api_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "last_error": None,
        }

        self._client: Optional[Any] = None
        if ALPACA_AVAILABLE and self.api_key and self.secret_key:
            try:
                self._client = StockHistoricalDataClient(self.api_key, self.secret_key)
                logger.info("EnhancedAlpacaClient initialized")
            except Exception as e:  # pragma: no cover - depends on env
                logger.error(f"Failed to initialize Alpaca client: {e}")
                self._client = None
        else:
            logger.warning(
                "EnhancedAlpacaClient running without credentials/SDK; calls will return None."
            )

    # ------------------------------------------------------------------ helpers

    def _enhanced_rate_limit(self) -> None:
        now = time.time()
        cutoff = now - 60
        while self.request_history and self.request_history[0] < cutoff:
            self.request_history.popleft()

        if len(self.request_history) >= self.rate_limit_per_minute:
            sleep_for = 60 - (now - self.request_history[0])
            if sleep_for > 0:
                time.sleep(sleep_for)

        delta = time.time() - self.last_request_time
        if delta < self.min_delay:
            time.sleep(self.min_delay - delta)

        self.last_request_time = time.time()
        self.request_history.append(self.last_request_time)

    def _record_success(self, response_time: float) -> None:
        self.api_stats["total_requests"] += 1
        self.api_stats["successful_requests"] += 1
        n = self.api_stats["successful_requests"]
        prev = self.api_stats["avg_response_time"]
        self.api_stats["avg_response_time"] = (prev * (n - 1) + response_time) / n

    def _record_failure(self, error: str) -> None:
        self.api_stats["total_requests"] += 1
        self.api_stats["failed_requests"] += 1
        self.api_stats["last_error"] = error

    def _call_with_retry(self, fn, attempts: int = 3, backoff: float = 1.0):
        """Run an Alpaca SDK call with rate limiting and retry/backoff."""
        last_err = None
        for attempt in range(attempts):
            try:
                self._enhanced_rate_limit()
                start = time.time()
                result = fn()
                self._record_success(time.time() - start)
                return result
            except Exception as e:  # pragma: no cover - network dependent
                last_err = str(e)
                self._record_failure(last_err)
                if attempt < attempts - 1:
                    time.sleep(backoff * (2 ** attempt))
        logger.error(f"Alpaca request failed after {attempts} attempts: {last_err}")
        return None

    @staticmethod
    def _resolve_timeframe(interval: str) -> Optional["TimeFrame"]:
        if not ALPACA_AVAILABLE:
            return None
        spec = INTERVAL_MAP.get(interval.lower())
        if not spec:
            return TimeFrame.Day
        amount, unit = spec
        return TimeFrame(amount=amount, unit=unit)

    # --------------------------------------------------------------- public API

    def get_enhanced_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return a dict with the same shape as EnhancedBinanceClient.get_enhanced_ticker."""
        if not self._client:
            return None

        def _fetch():
            req = StockSnapshotRequest(symbol_or_symbols=symbol)
            return self._client.get_stock_snapshot(req)

        snap_map = self._call_with_retry(_fetch)
        if not snap_map or symbol not in snap_map:
            return None

        snap = snap_map[symbol]
        try:
            quote = snap.latest_quote
            trade = snap.latest_trade
            day_bar = snap.daily_bar
            prev_bar = snap.previous_daily_bar

            bid_price = float(getattr(quote, "bid_price", 0.0) or 0.0)
            ask_price = float(getattr(quote, "ask_price", 0.0) or 0.0)
            bid_qty = float(getattr(quote, "bid_size", 0.0) or 0.0)
            ask_qty = float(getattr(quote, "ask_size", 0.0) or 0.0)

            current_price = float(getattr(trade, "price", 0.0) or 0.0)
            if current_price <= 0 and day_bar is not None:
                current_price = float(getattr(day_bar, "close", 0.0) or 0.0)

            high_24h = float(getattr(day_bar, "high", current_price) or current_price)
            low_24h = float(getattr(day_bar, "low", current_price) or current_price)
            volume_24h = float(getattr(day_bar, "volume", 0.0) or 0.0)
            vwap = float(getattr(day_bar, "vwap", current_price) or current_price)
            trade_count = int(getattr(day_bar, "trade_count", 0) or 0)

            prev_close = (
                float(getattr(prev_bar, "close", current_price) or current_price)
                if prev_bar is not None
                else current_price
            )
            price_change = current_price - prev_close
            price_change_percent = (price_change / prev_close * 100) if prev_close else 0.0

            return {
                "symbol": symbol,
                "current_price": current_price,
                "price_change": price_change,
                "price_change_percent": price_change_percent,
                "high_24h": high_24h,
                "low_24h": low_24h,
                "volume_24h": volume_24h,
                "quote_volume_24h": volume_24h * vwap,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "bid_qty": bid_qty,
                "ask_qty": ask_qty,
                "trade_count": trade_count,
                "timestamp": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Error parsing Alpaca snapshot for {symbol}: {e}")
            return None

    def get_enhanced_klines(
        self, symbol: str, interval: str = "1d", limit: int = 200
    ) -> Optional[List[Dict[str, Any]]]:
        """Return a list of bar dicts compatible with the Binance kline shape."""
        if not self._client:
            return None

        timeframe = self._resolve_timeframe(interval)
        end = datetime.utcnow() - timedelta(minutes=16)  # respect IEX 15-min delay
        # Window scaled to interval to ensure we get `limit` bars even with weekends/holidays.
        if interval.endswith("m"):
            window = timedelta(minutes=int(interval[:-1]) * limit * 2)
        elif interval.endswith("h"):
            window = timedelta(hours=int(interval[:-1]) * limit * 2)
        else:
            # daily or unknown -> assume calendar days; pad for weekends/holidays
            window = timedelta(days=int(limit * 1.6) + 5)
        start = end - window

        def _fetch():
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=limit,
            )
            return self._client.get_stock_bars(req)

        bars = self._call_with_retry(_fetch)
        if not bars:
            return None

        try:
            df_or_dict = bars.df if hasattr(bars, "df") else None
            rows: List[Dict[str, Any]] = []

            if df_or_dict is not None and not df_or_dict.empty:
                # multi-index DataFrame: (symbol, timestamp)
                df = df_or_dict.reset_index()
                if "symbol" in df.columns:
                    df = df[df["symbol"] == symbol]
                for _, r in df.iterrows():
                    ts = r.get("timestamp")
                    if hasattr(ts, "to_pydatetime"):
                        ts = ts.to_pydatetime()
                    close_price = float(r.get("close", 0.0))
                    volume = float(r.get("volume", 0.0))
                    vwap = float(r.get("vwap", close_price) or close_price)
                    rows.append(
                        {
                            "open_time": ts,
                            "open": float(r.get("open", 0.0)),
                            "high": float(r.get("high", 0.0)),
                            "low": float(r.get("low", 0.0)),
                            "close": close_price,
                            "volume": volume,
                            "close_time": ts,
                            "quote_volume": volume * vwap,
                            "trade_count": int(r.get("trade_count", 0) or 0),
                            "taker_buy_volume": volume / 2.0,  # Alpaca has no maker/taker split
                            "taker_buy_quote": (volume * vwap) / 2.0,
                        }
                    )

            if rows:
                rows.sort(key=lambda x: x["open_time"])
                return rows[-limit:]

            # Fallback: dict-of-list shape some SDK versions return
            data = getattr(bars, "data", None)
            if isinstance(data, dict) and symbol in data:
                for bar in data[symbol]:
                    ts = getattr(bar, "timestamp", None)
                    close_price = float(getattr(bar, "close", 0.0))
                    volume = float(getattr(bar, "volume", 0.0))
                    vwap = float(getattr(bar, "vwap", close_price) or close_price)
                    rows.append(
                        {
                            "open_time": ts,
                            "open": float(getattr(bar, "open", 0.0)),
                            "high": float(getattr(bar, "high", 0.0)),
                            "low": float(getattr(bar, "low", 0.0)),
                            "close": close_price,
                            "volume": volume,
                            "close_time": ts,
                            "quote_volume": volume * vwap,
                            "trade_count": int(getattr(bar, "trade_count", 0) or 0),
                            "taker_buy_volume": volume / 2.0,
                            "taker_buy_quote": (volume * vwap) / 2.0,
                        }
                    )
                rows.sort(key=lambda x: x["open_time"])
                return rows[-limit:]

            return None
        except Exception as e:
            logger.error(f"Error parsing Alpaca bars for {symbol}: {e}")
            return None

    def get_order_book_enhanced(
        self, symbol: str, limit: int = 100
    ) -> Optional[Dict[str, Any]]:
        """Alpaca free tier exposes only Level 1 (best bid/ask).

        We synthesize a one-level book with the latest quote so downstream
        microstructure code keeps working — extra levels are not available.
        """
        if not self._client:
            return None

        def _fetch():
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            return self._client.get_stock_latest_quote(req)

        quote_map = self._call_with_retry(_fetch)
        if not quote_map or symbol not in quote_map:
            return None

        try:
            q = quote_map[symbol]
            bid_price = float(getattr(q, "bid_price", 0.0) or 0.0)
            ask_price = float(getattr(q, "ask_price", 0.0) or 0.0)
            bid_size = float(getattr(q, "bid_size", 0.0) or 0.0)
            ask_size = float(getattr(q, "ask_size", 0.0) or 0.0)

            return {
                "bids": [(bid_price, bid_size)] if bid_price > 0 else [],
                "asks": [(ask_price, ask_size)] if ask_price > 0 else [],
                "last_update_id": int(time.time() * 1000),
                "timestamp": datetime.now(),
                "depth_levels_available": 1,  # Alpaca free tier limitation
            }
        except Exception as e:
            logger.error(f"Error parsing Alpaca quote for {symbol}: {e}")
            return None

    def get_api_stats(self) -> Dict[str, Any]:
        stats = self.api_stats.copy()
        total = max(stats["total_requests"], 1)
        stats["success_rate"] = stats["successful_requests"] / total * 100
        stats["requests_per_minute"] = len(self.request_history)
        return stats
