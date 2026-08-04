#!/usr/bin/env python3
"""
Stock Universe — S&P 500 constituents with sector data, screening, and earnings.

Replaces crypto_universe. Provides:
    default_universe          : preset symbol lists for backwards compatibility
    TRADING_PRESETS           : preset name -> dict(symbols=[...]) for compatibility
    StockUniverse             : screening, sector lookup, earnings calendar
"""

from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
UNIVERSE_CACHE_FILE = DATA_DIR / "sp500_universe.csv"

# A representative slice of the S&P 500 across all 11 GICS sectors. This is the
# bootstrap universe used until the live screen runs and replaces it.  Maintained
# in-source so the bot has a working universe with no internet on first launch.
SP500_SEED: List[Dict[str, str]] = [
    # Technology
    {"symbol": "AAPL",  "sector": "Technology"},
    {"symbol": "MSFT",  "sector": "Technology"},
    {"symbol": "NVDA",  "sector": "Technology"},
    {"symbol": "AVGO",  "sector": "Technology"},
    {"symbol": "ORCL",  "sector": "Technology"},
    {"symbol": "CRM",   "sector": "Technology"},
    {"symbol": "ADBE",  "sector": "Technology"},
    {"symbol": "AMD",   "sector": "Technology"},
    {"symbol": "INTC",  "sector": "Technology"},
    {"symbol": "CSCO",  "sector": "Technology"},
    {"symbol": "QCOM",  "sector": "Technology"},
    {"symbol": "TXN",   "sector": "Technology"},
    {"symbol": "IBM",   "sector": "Technology"},
    # Communication Services
    {"symbol": "GOOGL", "sector": "Communication Services"},
    {"symbol": "META",  "sector": "Communication Services"},
    {"symbol": "NFLX",  "sector": "Communication Services"},
    {"symbol": "DIS",   "sector": "Communication Services"},
    {"symbol": "T",     "sector": "Communication Services"},
    {"symbol": "VZ",    "sector": "Communication Services"},
    # Consumer Discretionary
    {"symbol": "AMZN",  "sector": "Consumer Discretionary"},
    {"symbol": "TSLA",  "sector": "Consumer Discretionary"},
    {"symbol": "HD",    "sector": "Consumer Discretionary"},
    {"symbol": "MCD",   "sector": "Consumer Discretionary"},
    {"symbol": "NKE",   "sector": "Consumer Discretionary"},
    {"symbol": "SBUX",  "sector": "Consumer Discretionary"},
    {"symbol": "LOW",   "sector": "Consumer Discretionary"},
    {"symbol": "TJX",   "sector": "Consumer Discretionary"},
    # Consumer Staples
    {"symbol": "WMT",   "sector": "Consumer Staples"},
    {"symbol": "PG",    "sector": "Consumer Staples"},
    {"symbol": "KO",    "sector": "Consumer Staples"},
    {"symbol": "PEP",   "sector": "Consumer Staples"},
    {"symbol": "COST",  "sector": "Consumer Staples"},
    {"symbol": "MDLZ",  "sector": "Consumer Staples"},
    {"symbol": "MO",    "sector": "Consumer Staples"},
    {"symbol": "PM",    "sector": "Consumer Staples"},
    # Financials
    {"symbol": "JPM",   "sector": "Financials"},
    {"symbol": "BAC",   "sector": "Financials"},
    {"symbol": "WFC",   "sector": "Financials"},
    {"symbol": "GS",    "sector": "Financials"},
    {"symbol": "MS",    "sector": "Financials"},
    {"symbol": "BLK",   "sector": "Financials"},
    {"symbol": "C",     "sector": "Financials"},
    {"symbol": "AXP",   "sector": "Financials"},
    {"symbol": "V",     "sector": "Financials"},
    {"symbol": "MA",    "sector": "Financials"},
    {"symbol": "BRK.B", "sector": "Financials"},
    # Health Care
    {"symbol": "UNH",   "sector": "Health Care"},
    {"symbol": "JNJ",   "sector": "Health Care"},
    {"symbol": "LLY",   "sector": "Health Care"},
    {"symbol": "PFE",   "sector": "Health Care"},
    {"symbol": "MRK",   "sector": "Health Care"},
    {"symbol": "ABBV",  "sector": "Health Care"},
    {"symbol": "TMO",   "sector": "Health Care"},
    {"symbol": "ABT",   "sector": "Health Care"},
    {"symbol": "DHR",   "sector": "Health Care"},
    {"symbol": "BMY",   "sector": "Health Care"},
    # Industrials
    {"symbol": "CAT",   "sector": "Industrials"},
    {"symbol": "BA",    "sector": "Industrials"},
    {"symbol": "HON",   "sector": "Industrials"},
    {"symbol": "GE",    "sector": "Industrials"},
    {"symbol": "RTX",   "sector": "Industrials"},
    {"symbol": "UPS",   "sector": "Industrials"},
    {"symbol": "LMT",   "sector": "Industrials"},
    {"symbol": "DE",    "sector": "Industrials"},
    {"symbol": "UNP",   "sector": "Industrials"},
    # Energy
    {"symbol": "XOM",   "sector": "Energy"},
    {"symbol": "CVX",   "sector": "Energy"},
    {"symbol": "COP",   "sector": "Energy"},
    {"symbol": "SLB",   "sector": "Energy"},
    {"symbol": "EOG",   "sector": "Energy"},
    # Materials
    {"symbol": "LIN",   "sector": "Materials"},
    {"symbol": "APD",   "sector": "Materials"},
    {"symbol": "SHW",   "sector": "Materials"},
    {"symbol": "FCX",   "sector": "Materials"},
    # Real Estate
    {"symbol": "PLD",   "sector": "Real Estate"},
    {"symbol": "AMT",   "sector": "Real Estate"},
    {"symbol": "EQIX",  "sector": "Real Estate"},
    {"symbol": "SPG",   "sector": "Real Estate"},
    # Utilities
    {"symbol": "NEE",   "sector": "Utilities"},
    {"symbol": "DUK",   "sector": "Utilities"},
    {"symbol": "SO",    "sector": "Utilities"},
    {"symbol": "AEP",   "sector": "Utilities"},
]


# Symbols in the static S&P 500 snapshot that Alpaca will NOT trade. When a horse
# selects one of these, the BUY fails graceful ("asset not found", exit 0) and the
# horse leaks that slot to cash, biasing its return vs the others. Dropping them at
# universe-build time means each strategy picks its next-best TRADEABLE name instead.
# ponytail: pinned exclusion set, not a live Alpaca asset query. Observed offender
# is SATS (EchoStar) — yfinance prices it, Alpaca rejects it. If this set grows past
# a handful, swap to a startup Alpaca get_all_assets() intersection.
UNTRADEABLE_AT_BROKER = frozenset({"SATS"})


def sp500_constituents() -> List[Dict[str, str]]:
    """Full static S&P 500 constituent list (symbol+sector), committed in-source.

    Reads the build-time snapshot in ``sp500_constituents.py``. Falls back to the
    smaller ``SP500_SEED`` if that module is missing, so the bot still runs with a
    valid (if narrower) universe. NEVER hits the network at runtime — refreshing
    the snapshot is a deliberate manual rebuild. Symbols in
    ``UNTRADEABLE_AT_BROKER`` are dropped so no strategy wastes a slot on a name
    Alpaca rejects.
    """
    try:
        from sp500_constituents import SP500_CONSTITUENTS  # type: ignore
        if SP500_CONSTITUENTS:
            rows = list(SP500_CONSTITUENTS)
        else:
            rows = list(SP500_SEED)
    except Exception as e:  # pragma: no cover - defensive import guard
        logger.warning("sp500_constituents snapshot unavailable, using seed: %s", e)
        rows = list(SP500_SEED)
    return [row for row in rows if row["symbol"] not in UNTRADEABLE_AT_BROKER]


def sp500_symbols() -> List[str]:
    """yfinance-ready symbol list for the full static S&P 500 universe."""
    return [row["symbol"] for row in sp500_constituents()]


def sp500_sector_map() -> Dict[str, str]:
    """symbol -> GICS sector for the full static universe."""
    return {row["symbol"]: row["sector"] for row in sp500_constituents()}


# Backwards-compatible presets (replaces crypto_universe.TRADING_PRESETS).
TRADING_PRESETS: Dict[str, Dict[str, Any]] = {
    "CONSERVATIVE": {
        "description": "Mega-cap, low-volatility names — paper-trading default",
        "symbols": ["AAPL", "MSFT", "JNJ", "JPM", "WMT", "PG", "KO", "UNH"],
    },
    "BALANCED": {
        "description": "Diversified across sectors",
        "symbols": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
            "JPM", "JNJ", "UNH", "WMT", "PG", "XOM", "CAT", "HD",
        ],
    },
    "AGGRESSIVE": {
        "description": "High-beta growth & momentum names",
        "symbols": [
            "TSLA", "NVDA", "AMD", "META", "NFLX", "CRM",
            "AMZN", "GOOGL", "AVGO", "ADBE",
        ],
    },
    "FULL_UNIVERSE": {
        "description": "Full S&P 500 seed list",
        "symbols": [row["symbol"] for row in SP500_SEED],
    },
}


@dataclass
class StockMeta:
    symbol: str
    sector: str = "Unknown"
    last_price: Optional[float] = None
    avg_volume_30d: Optional[float] = None
    next_earnings_date: Optional[date] = None
    last_screened_at: Optional[datetime] = None


class StockUniverse:
    """S&P 500 universe with screening, sector lookup, and earnings calendar."""

    PRICE_MIN = 5.0
    PRICE_MAX = 500.0
    MIN_AVG_VOLUME = 500_000
    SCREEN_INTERVAL_DAYS = 7

    def __init__(self, symbols: Optional[Iterable[str]] = None):
        self._meta: Dict[str, StockMeta] = {}
        for row in SP500_SEED:
            self._meta[row["symbol"]] = StockMeta(symbol=row["symbol"], sector=row["sector"])
        if symbols:
            for s in symbols:
                self._meta.setdefault(s, StockMeta(symbol=s))
        self._load_cache()

    # -------------------------------------------------- public lookups

    def all_symbols(self) -> List[str]:
        return list(self._meta.keys())

    def sector_for(self, symbol: str) -> str:
        meta = self._meta.get(symbol)
        return meta.sector if meta else "Unknown"

    def by_sector(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for m in self._meta.values():
            out.setdefault(m.sector, []).append(m.symbol)
        return out

    def is_in_universe(self, symbol: str) -> bool:
        return symbol in self._meta

    # -------------------------------------------------- screening

    def screen(
        self,
        price_history: Dict[str, List[float]],
        volume_history: Dict[str, List[float]],
    ) -> List[str]:
        """Return symbols that pass price/volume filters.

        price_history / volume_history are caller-provided so this method has
        no Alpaca dependency (testable, deterministic).
        """
        passed: List[str] = []
        now = datetime.utcnow()
        for symbol, meta in self._meta.items():
            prices = price_history.get(symbol)
            volumes = volume_history.get(symbol)
            if not prices or not volumes:
                continue
            last_price = prices[-1]
            avg_vol = sum(volumes[-30:]) / max(len(volumes[-30:]), 1)
            meta.last_price = last_price
            meta.avg_volume_30d = avg_vol
            meta.last_screened_at = now
            if self.PRICE_MIN <= last_price <= self.PRICE_MAX and avg_vol >= self.MIN_AVG_VOLUME:
                passed.append(symbol)
        self._save_cache()
        return passed

    # -------------------------------------------------- earnings

    def fetch_next_earnings_date(self, symbol: str) -> Optional[date]:
        """Best-effort earnings lookup via yfinance. Returns None on failure."""
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            logger.debug("yfinance not installed; earnings calendar disabled")
            return None
        try:
            cal = yf.Ticker(symbol).calendar
            # yfinance returns either DataFrame or dict depending on version
            earnings_date: Any = None
            if isinstance(cal, dict):
                earnings_date = cal.get("Earnings Date")
            elif cal is not None and hasattr(cal, "loc"):
                try:
                    earnings_date = cal.loc["Earnings Date"].iloc[0]
                except Exception:
                    earnings_date = None
            if isinstance(earnings_date, list) and earnings_date:
                earnings_date = earnings_date[0]
            if hasattr(earnings_date, "date"):
                next_date = earnings_date.date()
            elif isinstance(earnings_date, date):
                next_date = earnings_date
            else:
                return None
            meta = self._meta.setdefault(symbol, StockMeta(symbol=symbol))
            meta.next_earnings_date = next_date
            return next_date
        except Exception as e:
            logger.debug(f"Earnings lookup failed for {symbol}: {e}")
            return None

    def in_earnings_blackout(self, symbol: str, days_window: int = 3) -> bool:
        meta = self._meta.get(symbol)
        if not meta or not meta.next_earnings_date:
            return False
        delta = (meta.next_earnings_date - date.today()).days
        return 0 <= delta <= days_window

    # -------------------------------------------------- persistence

    def _save_cache(self) -> None:
        try:
            with open(UNIVERSE_CACHE_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["symbol", "sector", "last_price", "avg_volume_30d", "next_earnings", "last_screened"])
                for m in self._meta.values():
                    writer.writerow([
                        m.symbol,
                        m.sector,
                        m.last_price if m.last_price is not None else "",
                        m.avg_volume_30d if m.avg_volume_30d is not None else "",
                        m.next_earnings_date.isoformat() if m.next_earnings_date else "",
                        m.last_screened_at.isoformat() if m.last_screened_at else "",
                    ])
        except Exception as e:
            logger.debug(f"Could not write universe cache: {e}")

    def _load_cache(self) -> None:
        if not UNIVERSE_CACHE_FILE.exists():
            return
        try:
            with open(UNIVERSE_CACHE_FILE) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sym = row["symbol"]
                    meta = self._meta.setdefault(sym, StockMeta(symbol=sym, sector=row.get("sector", "Unknown")))
                    meta.sector = row.get("sector", meta.sector)
                    if row.get("last_price"):
                        meta.last_price = float(row["last_price"])
                    if row.get("avg_volume_30d"):
                        meta.avg_volume_30d = float(row["avg_volume_30d"])
                    if row.get("next_earnings"):
                        meta.next_earnings_date = date.fromisoformat(row["next_earnings"])
                    if row.get("last_screened"):
                        meta.last_screened_at = datetime.fromisoformat(row["last_screened"])
        except Exception as e:
            logger.debug(f"Could not read universe cache: {e}")


default_universe = StockUniverse()
