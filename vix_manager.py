#!/usr/bin/env python3
"""
VIX Manager — current VIX, VIX percentile rank, and SPY trend filter.

Used by mean-reversion strategies (entry only when VIX rank < 50) and as a
component of the macro-regime filter. Caches values once per market day to
avoid hammering yfinance.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import yfinance as yf  # type: ignore
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed — VIX manager will return neutral defaults")


CACHE_TTL_HOURS = 12  # refresh twice a day; sufficient for daily-bar strategies


class VIXManager:
    """Fetches and caches VIX + SPY trend data via yfinance."""

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self._cached_vix: Optional[float] = None
        self._cached_rank: Optional[float] = None
        self._cached_history: List[float] = []
        self._cached_at: Optional[datetime] = None
        self._cached_spy_trend_up: Optional[bool] = None

    # ----------------------------------------------------------- public

    def get_vix(self) -> Optional[float]:
        """Return the latest VIX close. None if unavailable."""
        self._refresh_if_stale()
        return self._cached_vix

    def get_vix_rank(self, lookback: Optional[int] = None) -> float:
        """Return percentile rank of current VIX over last N trading days (0-100).

        Defaults to 50.0 (neutral) when data is unavailable so downstream
        filters never crash.
        """
        self._refresh_if_stale()
        if self._cached_rank is not None:
            return self._cached_rank
        return 50.0

    def is_low_vix(self, threshold: float = 50.0) -> bool:
        """True if VIX rank is below threshold (favorable for mean reversion)."""
        return self.get_vix_rank() < threshold

    def is_spy_uptrend(self) -> bool:
        """True if SPY closed above its 200-day moving average."""
        self._refresh_if_stale()
        return bool(self._cached_spy_trend_up) if self._cached_spy_trend_up is not None else True

    def snapshot(self) -> dict:
        """Compact dict for logging / Telegram."""
        return {
            "vix": self.get_vix(),
            "vix_rank": self.get_vix_rank(),
            "spy_uptrend": self.is_spy_uptrend(),
            "fetched_at": self._cached_at.isoformat() if self._cached_at else None,
        }

    # ----------------------------------------------------------- internals

    def _refresh_if_stale(self) -> None:
        if not self._is_stale():
            return
        if not YFINANCE_AVAILABLE:
            return
        try:
            self._fetch_vix()
            self._fetch_spy_trend()
            self._cached_at = datetime.utcnow()
        except Exception as e:
            logger.warning(f"VIX refresh failed: {e}")

    def _is_stale(self) -> bool:
        if self._cached_at is None:
            return True
        return datetime.utcnow() - self._cached_at > timedelta(hours=CACHE_TTL_HOURS)

    def _fetch_vix(self) -> None:
        end = datetime.utcnow()
        start = end - timedelta(days=int(self.lookback_days * 1.6) + 30)
        hist = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return
        closes = hist["Close"].dropna().tolist()
        if not closes:
            return
        self._cached_vix = float(closes[-1])
        self._cached_history = closes[-self.lookback_days:]
        # Percentile rank of current value within lookback window
        if len(self._cached_history) > 1:
            current = self._cached_vix
            below = sum(1 for v in self._cached_history if v <= current)
            self._cached_rank = (below / len(self._cached_history)) * 100.0
        else:
            self._cached_rank = 50.0

    def _fetch_spy_trend(self) -> None:
        end = datetime.utcnow()
        start = end - timedelta(days=300)
        hist = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return
        closes = hist["Close"].dropna()
        if len(closes) < 200:
            return
        ma200 = closes.tail(200).mean()
        # ma200/last_close may come back as Series in some pandas versions; coerce
        try:
            last = float(closes.iloc[-1])
            ma = float(ma200)
        except Exception:
            last = float(closes.iloc[-1].iloc[0]) if hasattr(closes.iloc[-1], "iloc") else float(closes.iloc[-1])
            ma = float(ma200.iloc[0]) if hasattr(ma200, "iloc") else float(ma200)
        self._cached_spy_trend_up = last > ma


_vix_manager: Optional[VIXManager] = None


def get_vix_manager() -> VIXManager:
    global _vix_manager
    if _vix_manager is None:
        _vix_manager = VIXManager()
    return _vix_manager
