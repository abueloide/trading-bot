#!/usr/bin/env python3
"""
Crypto Client — Binance API wrapper for the multi-market trading bot.

Follows the same interface pattern as EnhancedAlpacaClient so the signal
evaluator and executor can consume data from either market without changes.

Supports:
    - Binance Spot (mainnet + testnet)
    - Market data: ticker, klines, order book
    - Order execution: market, limit (paper via testnet)
    - 24/7 operation (no market hours restriction)

All external calls degrade gracefully — returns None/empty on failure.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from binance.client import Client as BinanceClient
    from binance.enums import (
        SIDE_BUY, SIDE_SELL,
        ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT,
        TIME_IN_FORCE_GTC,
    )
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    logger.warning("python-binance not installed — crypto client in degraded mode")


INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


class EnhancedCryptoClient:
    """Binance market-data and execution client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        testnet: bool = True,
    ):
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.secret_key = secret_key or os.getenv("BINANCE_SECRET_KEY", "")
        self.testnet = testnet
        self._client: Optional[Any] = None
        self._request_timestamps: deque = deque(maxlen=1200)
        self._init_client()

    def _init_client(self) -> None:
        if not BINANCE_AVAILABLE or not self.api_key:
            return
        try:
            self._client = BinanceClient(
                self.api_key,
                self.secret_key,
                testnet=self.testnet,
            )
            logger.info(f"Binance client initialized (testnet={self.testnet})")
        except Exception as e:
            logger.warning(f"Binance client init failed: {e}")

    def _rate_limit(self) -> None:
        """Simple rate limiter: max 1200 requests/minute (Binance limit)."""
        now = time.time()
        while self._request_timestamps and now - self._request_timestamps[0] > 60:
            self._request_timestamps.popleft()
        if len(self._request_timestamps) >= 1100:
            sleep_time = 60 - (now - self._request_timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._request_timestamps.append(time.time())

    # ----------------------------------------------------------- market data

    def get_enhanced_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price snapshot for a symbol (e.g., BTCUSDT)."""
        if not self._client:
            return None
        self._rate_limit()
        try:
            ticker = self._client.get_symbol_ticker(symbol=symbol)
            stats = self._client.get_ticker(symbol=symbol)
            return {
                "symbol": symbol,
                "price": float(ticker.get("price", 0)),
                "bid": float(stats.get("bidPrice", 0)),
                "ask": float(stats.get("askPrice", 0)),
                "volume_24h": float(stats.get("volume", 0)),
                "quote_volume_24h": float(stats.get("quoteVolume", 0)),
                "price_change_24h_pct": float(stats.get("priceChangePercent", 0)),
                "high_24h": float(stats.get("highPrice", 0)),
                "low_24h": float(stats.get("lowPrice", 0)),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Ticker fetch failed for {symbol}: {e}")
            return None

    def get_enhanced_klines(
        self, symbol: str, interval: str = "1d", limit: int = 252
    ) -> List[Dict[str, Any]]:
        """Get historical klines/candlesticks."""
        if not self._client:
            return []
        bi = INTERVAL_MAP.get(interval, "1d")
        self._rate_limit()
        try:
            raw = self._client.get_klines(symbol=symbol, interval=bi, limit=limit)
            bars = []
            for k in raw:
                bars.append({
                    "timestamp": datetime.utcfromtimestamp(k[0] / 1000),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            return bars
        except Exception as e:
            logger.warning(f"Klines fetch failed for {symbol}: {e}")
            return []

    def get_order_book_enhanced(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        """Get order book (bids/asks)."""
        if not self._client:
            return {"bids": [], "asks": []}
        self._rate_limit()
        try:
            book = self._client.get_order_book(symbol=symbol, limit=limit)
            return {
                "bids": [[float(p), float(q)] for p, q in book.get("bids", [])],
                "asks": [[float(p), float(q)] for p, q in book.get("asks", [])],
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Order book fetch failed for {symbol}: {e}")
            return {"bids": [], "asks": []}

    def get_price_history(self, symbol: str, days: int = 252) -> List[float]:
        """Convenience: returns list of close prices for signal evaluator."""
        bars = self.get_enhanced_klines(symbol, interval="1d", limit=days)
        return [b["close"] for b in bars]

    def get_volume_history(self, symbol: str, days: int = 252) -> List[float]:
        """Convenience: returns list of volumes for signal evaluator."""
        bars = self.get_enhanced_klines(symbol, interval="1d", limit=days)
        return [b["volume"] for b in bars]

    # ----------------------------------------------------------- execution

    def place_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> Optional[Dict]:
        """Place a market order. side = 'BUY' or 'SELL'."""
        if not self._client:
            logger.error("Cannot place order: Binance client not initialized")
            return None
        self._rate_limit()
        try:
            order = self._client.create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET if BINANCE_AVAILABLE else "MARKET",
                quantity=quantity,
            )
            logger.info(f"Crypto order placed: {side} {quantity} {symbol}")
            return order
        except Exception as e:
            logger.error(f"Order failed: {side} {quantity} {symbol}: {e}")
            return None

    def get_account_balance(self, asset: str = "USDT") -> float:
        """Get available balance for an asset."""
        if not self._client:
            return 0.0
        try:
            info = self._client.get_account()
            for b in info.get("balances", []):
                if b["asset"] == asset:
                    return float(b["free"])
            return 0.0
        except Exception as e:
            logger.warning(f"Balance fetch failed: {e}")
            return 0.0

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders."""
        if not self._client:
            return []
        try:
            if symbol:
                return self._client.get_open_orders(symbol=symbol)
            return self._client.get_open_orders()
        except Exception as e:
            logger.warning(f"Open orders fetch failed: {e}")
            return []

    # ----------------------------------------------------------- utils

    def get_api_stats(self) -> Dict[str, Any]:
        return {
            "client_type": "binance",
            "testnet": self.testnet,
            "available": self._client is not None,
            "requests_last_minute": len(self._request_timestamps),
        }


# Module-level singleton
_crypto_client: Optional[EnhancedCryptoClient] = None


def get_crypto_client() -> EnhancedCryptoClient:
    global _crypto_client
    if _crypto_client is None:
        _crypto_client = EnhancedCryptoClient()
    return _crypto_client
