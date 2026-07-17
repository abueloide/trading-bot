#!/usr/bin/env python3
"""
Alternative Data Manager — congress, insider, institutional flow, dark pool.

Provides a single ``AlternativeDataManager`` with caching and graceful
degradation: any failed external call returns 0.5 (neutral) so signal
evaluation never crashes. The smart-money composite score uses the weights
from config.SMART_MONEY_WEIGHTS:

    insider_cluster      0.40   (strongest academic backing)
    congress_direction   0.30
    institutional_flow   0.20
    dark_pool_activity   0.10

Backtesting note: callers must respect filing lags
(``ALTERNATIVE_DATA_CONFIG['congress_filing_lag_days']`` = 45,
``insider_filing_lag_days`` = 2). The backtest engine threads these via
the ``extra_data`` parameter so signals can only consume data that was
public at the trade date.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    httpx = None
    logger.warning("httpx not installed — alternative data will run in degraded mode")

try:
    from config import ALTERNATIVE_DATA_CONFIG, SMART_MONEY_WEIGHTS
except ImportError:
    ALTERNATIVE_DATA_CONFIG = {
        "enabled": False,
        "quiver_token": "",
        "congress_lookback_days": 90,
        "insider_window_days": 30,
        "congress_filing_lag_days": 45,
        "insider_filing_lag_days": 2,
        "cache_ttl_seconds": 86_400,
        "fallback_score": 0.5,
    }
    SMART_MONEY_WEIGHTS = {
        "insider_cluster": 0.40,
        "congress_direction": 0.30,
        "institutional_flow": 0.20,
        "dark_pool": 0.10,
    }


QUIVER_BASE = "https://api.quiverquant.com/beta"
SECURITIESDB_BASE = "https://www.sec.gov"  # filings index, free


@dataclass
class SmartMoneyScore:
    symbol: str
    timestamp: datetime
    insider_score: float = 0.5
    congress_score: float = 0.5
    institutional_score: float = 0.5
    dark_pool_score: float = 0.5
    composite: float = 0.5
    details: Dict[str, Any] = field(default_factory=dict)


class _Cache:
    """In-memory TTL cache."""
    def __init__(self, ttl_seconds: int = 86_400):
        self._store: Dict[str, tuple] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        ts, value = item
        if time.time() - ts > self._ttl:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)


class AlternativeDataManager:
    """Smart-money signals with caching and graceful degradation."""

    def __init__(
        self,
        quiver_token: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
    ):
        self.quiver_token = quiver_token or ALTERNATIVE_DATA_CONFIG.get("quiver_token", "") or os.getenv("QUIVER_API_TOKEN", "")
        self._cache = _Cache(cache_ttl_seconds or ALTERNATIVE_DATA_CONFIG.get("cache_ttl_seconds", 86_400))
        self._fallback = ALTERNATIVE_DATA_CONFIG.get("fallback_score", 0.5)
        self._weights = dict(SMART_MONEY_WEIGHTS)

    # ----------------------------------------------------------- public API

    def get_smart_money_score(self, symbol: str) -> SmartMoneyScore:
        """Combined 0-1 smart-money score. Always returns a value (no crashes)."""
        cached = self._cache.get(f"smart_money:{symbol}")
        if cached:
            return cached

        congress = self.get_congress_signal(symbol)
        insider = self.detect_insider_cluster(symbol)
        institutional = self.get_institutional_flow(symbol)
        dark_pool = self.get_dark_pool_activity(symbol)

        composite = (
            insider * self._weights["insider_cluster"]
            + congress * self._weights["congress_direction"]
            + institutional * self._weights["institutional_flow"]
            + dark_pool * self._weights["dark_pool"]
        )

        score = SmartMoneyScore(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            insider_score=insider,
            congress_score=congress,
            institutional_score=institutional,
            dark_pool_score=dark_pool,
            composite=max(0.0, min(1.0, composite)),
        )
        self._cache.set(f"smart_money:{symbol}", score)
        return score

    def get_congress_signal(
        self, symbol: str, lookback: Optional[int] = None
    ) -> float:
        """0-1 score for net congressional buying.  >0.5 = bullish."""
        lookback = lookback or ALTERNATIVE_DATA_CONFIG.get("congress_lookback_days", 90)
        cached = self._cache.get(f"congress:{symbol}:{lookback}")
        if cached is not None:
            return cached

        if not self.quiver_token or httpx is None:
            return self._fallback

        try:
            data = self._quiver_get(f"/historical/congresstrading/{symbol}")
            if not data:
                return self._fallback
            cutoff = datetime.utcnow() - timedelta(days=lookback)
            recent = [d for d in data if self._parse_date(d.get("TransactionDate")) >= cutoff]
            if not recent:
                return self._fallback
            buys = sum(1 for d in recent if str(d.get("Transaction", "")).lower().startswith("purchase"))
            sells = sum(1 for d in recent if str(d.get("Transaction", "")).lower().startswith("sale"))
            total = buys + sells
            score = (buys / total) if total > 0 else self._fallback
            self._cache.set(f"congress:{symbol}:{lookback}", score)
            return score
        except Exception as e:
            logger.warning(f"Congress signal failed for {symbol}: {e}")
            return self._fallback

    def detect_insider_cluster(
        self, symbol: str, window: Optional[int] = None
    ) -> float:
        """0-1 score for insider buying clusters within ``window`` days."""
        window = window or ALTERNATIVE_DATA_CONFIG.get("insider_window_days", 30)
        cached = self._cache.get(f"insider:{symbol}:{window}")
        if cached is not None:
            return cached

        if not self.quiver_token or httpx is None:
            return self._fallback

        try:
            data = self._quiver_get(f"/historical/insiders/{symbol}")
            if not data:
                return self._fallback
            cutoff = datetime.utcnow() - timedelta(days=window)
            recent = [d for d in data if self._parse_date(d.get("Date")) >= cutoff]
            if not recent:
                return self._fallback
            buys = [d for d in recent if str(d.get("AcquiredDisposed", "")).upper() == "A"]
            sells = [d for d in recent if str(d.get("AcquiredDisposed", "")).upper() == "D"]
            buy_dollars = sum(float(d.get("Shares", 0)) * float(d.get("PricePerShare", 0) or 0) for d in buys)
            sell_dollars = sum(float(d.get("Shares", 0)) * float(d.get("PricePerShare", 0) or 0) for d in sells)
            total = buy_dollars + sell_dollars
            base = (buy_dollars / total) if total > 0 else self._fallback
            # Cluster boost: 3+ insider buyers in window -> tilt toward 1.
            unique_buyers = len({d.get("InsiderName") for d in buys if d.get("InsiderName")})
            cluster_boost = 0.15 if unique_buyers >= 3 else 0.0
            score = max(0.0, min(1.0, base + cluster_boost))
            self._cache.set(f"insider:{symbol}:{window}", score)
            return score
        except Exception as e:
            logger.warning(f"Insider cluster failed for {symbol}: {e}")
            return self._fallback

    def get_institutional_flow(self, symbol: str) -> float:
        """0-1 score for institutional accumulation via 13F filings."""
        cached = self._cache.get(f"13f:{symbol}")
        if cached is not None:
            return cached

        if not self.quiver_token or httpx is None:
            return self._fallback

        try:
            data = self._quiver_get(f"/historical/13f/{symbol}")
            if not data or len(data) < 2:
                return self._fallback
            # Compare most recent two quarters
            recent = sorted(data, key=lambda d: d.get("Date", ""), reverse=True)[:2]
            if len(recent) < 2:
                return self._fallback
            curr = float(recent[0].get("Shares", 0) or 0)
            prev = float(recent[1].get("Shares", 0) or 0)
            if prev <= 0:
                return self._fallback
            change = (curr - prev) / prev
            # Map [-50%, +50%] change to [0, 1]
            score = max(0.0, min(1.0, 0.5 + change))
            self._cache.set(f"13f:{symbol}", score)
            return score
        except Exception as e:
            logger.warning(f"13F flow failed for {symbol}: {e}")
            return self._fallback

    def get_dark_pool_activity(self, symbol: str) -> float:
        """0-1 score for unusual off-exchange volume."""
        cached = self._cache.get(f"darkpool:{symbol}")
        if cached is not None:
            return cached

        if not self.quiver_token or httpx is None:
            return self._fallback

        try:
            data = self._quiver_get(f"/historical/offexchange/{symbol}")
            if not data or len(data) < 30:
                return self._fallback
            sorted_data = sorted(data, key=lambda d: d.get("Date", ""), reverse=True)
            recent_5 = [float(d.get("OffExchangeVolume", 0) or 0) for d in sorted_data[:5]]
            baseline = [float(d.get("OffExchangeVolume", 0) or 0) for d in sorted_data[5:35]]
            if not baseline or sum(baseline) <= 0:
                return self._fallback
            avg_recent = sum(recent_5) / len(recent_5)
            avg_baseline = sum(baseline) / len(baseline)
            ratio = avg_recent / avg_baseline if avg_baseline > 0 else 1.0
            # 1x = neutral 0.5; 2x+ = 1.0; 0.5x or less = 0.
            score = max(0.0, min(1.0, 0.5 + (ratio - 1.0) * 0.5))
            self._cache.set(f"darkpool:{symbol}", score)
            return score
        except Exception as e:
            logger.warning(f"Dark pool query failed for {symbol}: {e}")
            return self._fallback

    # ----------------------------------------------------------- internals

    def _quiver_get(self, path: str) -> Optional[List[Dict]]:
        if httpx is None or not self.quiver_token:
            return None
        url = f"{QUIVER_BASE}{path}"
        headers = {"Authorization": f"Bearer {self.quiver_token}", "Accept": "application/json"}
        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning(f"Quiver request failed: {url} → {e}")
            return None

    @staticmethod
    def _parse_date(s: Any) -> datetime:
        if not s:
            return datetime.min
        try:
            return datetime.fromisoformat(str(s).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            try:
                return datetime.strptime(str(s)[:10], "%Y-%m-%d")
            except Exception:
                return datetime.min


_alt_data_manager: Optional[AlternativeDataManager] = None


def get_alternative_data_manager() -> AlternativeDataManager:
    global _alt_data_manager
    if _alt_data_manager is None:
        _alt_data_manager = AlternativeDataManager()
    return _alt_data_manager
