#!/usr/bin/env python3
"""
Mexico Market Universe — MX exposure via US-listed ETFs and ADRs.

Strategy: trade Mexico exposure through Alpaca (no new broker needed).
    - EWW (iShares MSCI Mexico ETF) — broad MX market exposure
    - Mexican ADRs on US exchanges (AMX, CX, KOF, BSMX, etc.)
    - Nearshoring beneficiaries (US companies with heavy MX operations)

Data:
    - Alpaca for execution (same API, same account)
    - yfinance for BMV reference data (.MX tickers) and IPC index (^MXX)
    - USD/MXN exchange rate for peso strength signal

Future: plug in Interactive Brokers or GBM+ API for direct BMV access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ============================================================================
# MX exposure via US-listed instruments (tradeable on Alpaca)
# ============================================================================

@dataclass
class MXInstrument:
    symbol: str
    name: str
    category: str           # etf | adr | nearshoring
    mx_sector: str
    bmv_equivalent: str = ""  # corresponding BMV ticker if any
    weight: float = 1.0       # relative weight within category


# Core MX ETF
MX_ETF = [
    MXInstrument("EWW", "iShares MSCI Mexico ETF", "etf", "Broad Market",
                 bmv_equivalent="NAFTRACISHRS.MX", weight=3.0),
]

# Mexican ADRs listed on US exchanges
MX_ADRS = [
    MXInstrument("AMX", "América Móvil", "adr", "Telecom",
                 bmv_equivalent="AMXB.MX", weight=2.0),
    MXInstrument("CX", "CEMEX", "adr", "Materials",
                 bmv_equivalent="CEMEXCPO.MX", weight=1.5),
    MXInstrument("KOF", "Coca-Cola FEMSA", "adr", "Consumer Staples",
                 bmv_equivalent="KOFUBL.MX", weight=1.5),
    MXInstrument("BSMX", "Banco Santander México", "adr", "Financials",
                 bmv_equivalent="BSMXB.MX", weight=1.0),
    MXInstrument("OMAB", "OMA Aeropuertos", "adr", "Industrials",
                 bmv_equivalent="OMAB.MX", weight=1.0),
    MXInstrument("PAC", "Grupo Aeroportuario del Pacífico", "adr", "Industrials",
                 bmv_equivalent="GAPB.MX", weight=1.0),
    MXInstrument("ASR", "Grupo Aeroportuario del Sureste", "adr", "Industrials",
                 bmv_equivalent="ASURB.MX", weight=1.0),
    MXInstrument("TV", "Grupo Televisa (TelevisaUnivision)", "adr", "Communication",
                 bmv_equivalent="TLEVISACPO.MX", weight=0.8),
    MXInstrument("VLRS", "Volaris", "adr", "Industrials",
                 bmv_equivalent="VOLARA.MX", weight=0.8),
    MXInstrument("WMMVY", "Walmart de México (OTC)", "adr", "Consumer Staples",
                 bmv_equivalent="WALMEX.MX", weight=0.5),
]

# US companies that benefit heavily from nearshoring/Mexico operations
MX_NEARSHORING = [
    MXInstrument("PCAR", "PACCAR (trucks, MX manufacturing)", "nearshoring",
                 "Industrials", weight=0.8),
    MXInstrument("GE", "GE Aerospace (Queretaro plant)", "nearshoring",
                 "Industrials", weight=0.6),
    MXInstrument("GM", "General Motors (MX is #2 production)", "nearshoring",
                 "Consumer Discretionary", weight=0.8),
    MXInstrument("F", "Ford (Hermosillo, Cuautitlan)", "nearshoring",
                 "Consumer Discretionary", weight=0.7),
    MXInstrument("DE", "Deere & Co (Monterrey operations)", "nearshoring",
                 "Industrials", weight=0.6),
    MXInstrument("SWK", "Stanley Black & Decker (MX plants)", "nearshoring",
                 "Industrials", weight=0.5),
]

ALL_MX_INSTRUMENTS = MX_ETF + MX_ADRS + MX_NEARSHORING


# ============================================================================
# BMV reference tickers (for monitoring only, not tradeable via Alpaca)
# ============================================================================

BMV_REFERENCE = {
    "^MXX": "IPC Mexico Index",
    "MXNUSD=X": "MXN/USD Exchange Rate",
    "NAFTRACISHRS.MX": "NAFTRAC (MX S&P equivalent)",
    "CEMEXCPO.MX": "CEMEX",
    "AMXB.MX": "América Móvil",
    "FEMSAUBD.MX": "FEMSA",
    "GMEXICOB.MX": "GMéxico",
    "WALMEX.MX": "Walmart de México",
    "GFNORTEO.MX": "Banorte",
    "BIMBOA.MX": "Bimbo",
    "TLEVISACPO.MX": "Televisa",
}


# ============================================================================
# MX Market Data Manager
# ============================================================================

class MXMarketData:
    """Fetch MX market reference data via yfinance (free)."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_at: Dict[str, float] = {}
        self._cache_ttl = 3600  # 1 hour

    def get_ipc_level(self) -> Optional[float]:
        """Current IPC (^MXX) index level."""
        return self._get_last_close("^MXX")

    def get_usdmxn(self) -> Optional[float]:
        """Current USD/MXN exchange rate."""
        return self._get_last_close("MXNUSD=X")

    def get_peso_strength(self) -> float:
        """
        Peso strength signal: 0..1.
        1.0 = peso very strong (low USD/MXN), good for MX assets.
        0.0 = peso very weak (high USD/MXN), reduce MX exposure.

        Uses 200-day percentile rank of USD/MXN (inverted).
        """
        if not YFINANCE_AVAILABLE:
            return 0.5
        try:
            hist = yf.download("MXNUSD=X", period="1y", progress=False, auto_adjust=True)
            if hist is None or hist.empty or "Close" not in hist.columns:
                return 0.5
            closes = hist["Close"].dropna().tolist()
            if len(closes) < 50:
                return 0.5
            current = closes[-1]
            # Handle potential Series return
            if hasattr(current, "iloc"):
                current = float(current.iloc[0])
            else:
                current = float(current)
            # Lower USD/MXN = stronger peso = higher score
            below = sum(1 for v in closes if float(v.iloc[0]) if hasattr(v, "iloc") else float(v) >= current)
            rank = below / len(closes)  # % of days peso was weaker
            return max(0.0, min(1.0, rank))
        except Exception as e:
            logger.warning(f"Peso strength calculation failed: {e}")
            return 0.5

    def get_ipc_trend(self) -> bool:
        """True if IPC is above its 50-day moving average (uptrend)."""
        if not YFINANCE_AVAILABLE:
            return True
        try:
            hist = yf.download("^MXX", period="6mo", progress=False, auto_adjust=True)
            if hist is None or hist.empty or "Close" not in hist.columns:
                return True
            closes = hist["Close"].dropna()
            if len(closes) < 50:
                return True
            ma50 = closes.tail(50).mean()
            last = closes.iloc[-1]
            # Coerce to float
            if hasattr(last, "iloc"):
                last = float(last.iloc[0])
            if hasattr(ma50, "iloc"):
                ma50 = float(ma50.iloc[0])
            return float(last) > float(ma50)
        except Exception as e:
            logger.warning(f"IPC trend check failed: {e}")
            return True

    def snapshot(self) -> Dict[str, Any]:
        """Compact dict for logging / Telegram."""
        return {
            "ipc": self.get_ipc_level(),
            "usdmxn": self.get_usdmxn(),
            "peso_strength": round(self.get_peso_strength(), 2),
            "ipc_uptrend": self.get_ipc_trend(),
        }

    def _get_last_close(self, ticker: str) -> Optional[float]:
        import time as _time
        cached = self._cache.get(ticker)
        cached_at = self._cache_at.get(ticker, 0)
        if cached is not None and _time.time() - cached_at < self._cache_ttl:
            return cached
        if not YFINANCE_AVAILABLE:
            return None
        try:
            data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if data is None or data.empty or "Close" not in data.columns:
                return None
            val = data["Close"].dropna().iloc[-1]
            if hasattr(val, "iloc"):
                val = float(val.iloc[0])
            else:
                val = float(val)
            self._cache[ticker] = val
            self._cache_at[ticker] = _time.time()
            return val
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
            return None


# ============================================================================
# MX Universe (analogous to StockUniverse but for MX exposure)
# ============================================================================

class MXUniverse:
    """MX market universe: instruments tradeable via Alpaca + BMV reference."""

    def __init__(self):
        self.instruments = {i.symbol: i for i in ALL_MX_INSTRUMENTS}
        self.market_data = MXMarketData()

    def tradeable_symbols(self) -> List[str]:
        """All MX-exposure symbols tradeable on Alpaca."""
        return [i.symbol for i in ALL_MX_INSTRUMENTS]

    def etf_symbols(self) -> List[str]:
        return [i.symbol for i in MX_ETF]

    def adr_symbols(self) -> List[str]:
        return [i.symbol for i in MX_ADRS]

    def nearshoring_symbols(self) -> List[str]:
        return [i.symbol for i in MX_NEARSHORING]

    def by_category(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for i in ALL_MX_INSTRUMENTS:
            out.setdefault(i.category, []).append(i.symbol)
        return out

    def weighted_symbols(self, category: Optional[str] = None) -> List[str]:
        """Return symbols sorted by weight (highest first)."""
        items = ALL_MX_INSTRUMENTS
        if category:
            items = [i for i in items if i.category == category]
        return [i.symbol for i in sorted(items, key=lambda x: x.weight, reverse=True)]

    def filter_by_regime(self, regime: str) -> List[str]:
        """
        Adjust MX instrument selection based on macro regime.

        RISK_ON:  all instruments (ETF + ADRs + nearshoring)
        CAUTIOUS: ETF + top ADRs only
        RISK_OFF: ETF only (most liquid)
        CRISIS:   nothing — don't trade MX
        """
        regime = regime.upper()
        if regime == "CRISIS":
            return []
        if regime == "RISK_OFF":
            return self.etf_symbols()
        if regime == "CAUTIOUS":
            top_adrs = [i.symbol for i in MX_ADRS if i.weight >= 1.5]
            return self.etf_symbols() + top_adrs
        # RISK_ON — everything
        return self.tradeable_symbols()

    def get_telegram_summary(self) -> str:
        """Format MX market state for Telegram."""
        snap = self.market_data.snapshot()
        ipc = snap.get("ipc")
        usdmxn = snap.get("usdmxn")
        peso = snap.get("peso_strength", 0.5)
        uptrend = snap.get("ipc_uptrend", True)

        peso_emoji = "💪" if peso > 0.6 else "😐" if peso > 0.4 else "📉"

        lines = [
            "🇲🇽 Mexico Market",
            f"IPC: {ipc:,.0f}" if ipc else "IPC: unavailable",
            f"USD/MXN: {usdmxn:.2f}" if usdmxn else "USD/MXN: unavailable",
            f"Peso strength: {peso:.0%} {peso_emoji}",
            f"IPC > 50d MA: {'yes ✅' if uptrend else 'no ❌'}",
            "",
            f"Tradeable via Alpaca: {len(ALL_MX_INSTRUMENTS)} instruments",
            f"  ETF: {', '.join(self.etf_symbols())}",
            f"  ADRs: {len(MX_ADRS)} ({', '.join(self.adr_symbols()[:5])}...)",
            f"  Nearshoring: {len(MX_NEARSHORING)} names",
        ]
        return "\n".join(lines)


# ============================================================================
# Module-level singleton
# ============================================================================

_mx_universe: Optional[MXUniverse] = None


def get_mx_universe() -> MXUniverse:
    global _mx_universe
    if _mx_universe is None:
        _mx_universe = MXUniverse()
    return _mx_universe
