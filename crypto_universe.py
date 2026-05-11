#!/usr/bin/env python3
"""
Crypto Universe — tradeable crypto assets with regime-aware filtering.

Replaces the v1 crypto_universe that was removed in PR #1.
Designed to work alongside stock_universe (US) and mx_universe (MX)
in the multi-market portfolio.

Categories:
    - Major:  BTC, ETH (always tradeable)
    - Large:  SOL, BNB, ADA, etc (RISK_ON / CAUTIOUS)
    - DeFi:   AAVE, UNI, etc (RISK_ON only)
    - Stables: USDT, USDC (always available for parking)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CryptoAsset:
    symbol: str           # Binance pair (e.g., BTCUSDT)
    base: str             # Base asset (e.g., BTC)
    name: str
    category: str         # major | large | mid | defi | stables
    weight: float = 1.0   # relative weight for allocation


# ============================================================================
# Universe definition
# ============================================================================

CRYPTO_MAJOR = [
    CryptoAsset("BTCUSDT", "BTC", "Bitcoin", "major", weight=5.0),
    CryptoAsset("ETHUSDT", "ETH", "Ethereum", "major", weight=3.0),
]

CRYPTO_LARGE = [
    CryptoAsset("SOLUSDT", "SOL", "Solana", "large", weight=2.0),
    CryptoAsset("BNBUSDT", "BNB", "BNB", "large", weight=1.5),
    CryptoAsset("ADAUSDT", "ADA", "Cardano", "large", weight=1.0),
    CryptoAsset("XRPUSDT", "XRP", "XRP", "large", weight=1.0),
    CryptoAsset("AVAXUSDT", "AVAX", "Avalanche", "large", weight=1.0),
    CryptoAsset("DOTUSDT", "DOT", "Polkadot", "large", weight=0.8),
    CryptoAsset("LINKUSDT", "LINK", "Chainlink", "large", weight=0.8),
]

CRYPTO_DEFI = [
    CryptoAsset("AAVEUSDT", "AAVE", "Aave", "defi", weight=0.8),
    CryptoAsset("UNIUSDT", "UNI", "Uniswap", "defi", weight=0.8),
    CryptoAsset("MKRUSDT", "MKR", "Maker", "defi", weight=0.6),
]

CRYPTO_STABLES = [
    CryptoAsset("USDTUSDT", "USDT", "Tether", "stables", weight=0.0),
    CryptoAsset("USDCUSDT", "USDC", "USD Coin", "stables", weight=0.0),
]

ALL_CRYPTO = CRYPTO_MAJOR + CRYPTO_LARGE + CRYPTO_DEFI
ALL_CRYPTO_WITH_STABLES = ALL_CRYPTO + CRYPTO_STABLES


# ============================================================================
# Crypto Universe class
# ============================================================================

class CryptoUniverse:
    """Crypto asset universe with regime-aware filtering."""

    def __init__(self):
        self.assets = {a.symbol: a for a in ALL_CRYPTO_WITH_STABLES}

    def tradeable_symbols(self) -> List[str]:
        """All non-stable crypto pairs."""
        return [a.symbol for a in ALL_CRYPTO]

    def major_symbols(self) -> List[str]:
        return [a.symbol for a in CRYPTO_MAJOR]

    def large_symbols(self) -> List[str]:
        return [a.symbol for a in CRYPTO_LARGE]

    def defi_symbols(self) -> List[str]:
        return [a.symbol for a in CRYPTO_DEFI]

    def stable_symbols(self) -> List[str]:
        return [a.symbol for a in CRYPTO_STABLES]

    def by_category(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for a in ALL_CRYPTO_WITH_STABLES:
            out.setdefault(a.category, []).append(a.symbol)
        return out

    def weighted_symbols(self, category: Optional[str] = None) -> List[str]:
        """Return symbols sorted by weight (highest first)."""
        items = ALL_CRYPTO
        if category:
            items = [a for a in items if a.category == category]
        return [a.symbol for a in sorted(items, key=lambda x: x.weight, reverse=True)]

    def filter_by_regime(self, regime: str) -> List[str]:
        """
        Adjust crypto selection based on macro regime.

        RISK_ON:  all crypto (majors + large + defi)
        CAUTIOUS: majors + top large caps only
        RISK_OFF: BTC only (digital gold thesis)
        CRISIS:   BTC only (small allocation, rest in stables)
        """
        regime = regime.upper()
        if regime == "CRISIS":
            return ["BTCUSDT"]  # BTC as digital gold, minimal
        if regime == "RISK_OFF":
            return ["BTCUSDT", "ETHUSDT"]
        if regime == "CAUTIOUS":
            top_large = [a.symbol for a in CRYPTO_LARGE if a.weight >= 1.5]
            return self.major_symbols() + top_large
        # RISK_ON — everything
        return self.tradeable_symbols()

    def get_allocation_weights(self, regime: str) -> Dict[str, float]:
        """
        Get relative allocation weights for filtered symbols.
        Weights sum to 1.0 within the filtered set.
        """
        symbols = self.filter_by_regime(regime)
        if not symbols:
            return {}
        weights = {}
        for sym in symbols:
            asset = self.assets.get(sym)
            weights[sym] = asset.weight if asset else 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 3) for k, v in weights.items()}
        return weights

    def get_telegram_summary(self) -> str:
        """Format crypto universe for Telegram."""
        lines = [
            "₿ Crypto Universe",
            f"Total assets: {len(ALL_CRYPTO)}",
            f"  Majors: {', '.join(a.base for a in CRYPTO_MAJOR)}",
            f"  Large:  {', '.join(a.base for a in CRYPTO_LARGE)}",
            f"  DeFi:   {', '.join(a.base for a in CRYPTO_DEFI)}",
            "",
            "Regime filtering:",
            f"  RISK_ON:  {len(self.filter_by_regime('RISK_ON'))} assets",
            f"  CAUTIOUS: {len(self.filter_by_regime('CAUTIOUS'))} assets",
            f"  RISK_OFF: {len(self.filter_by_regime('RISK_OFF'))} assets",
            f"  CRISIS:   {len(self.filter_by_regime('CRISIS'))} assets (BTC only)",
        ]
        return "\n".join(lines)


# ============================================================================
# Module-level singleton
# ============================================================================

_crypto_universe: Optional[CryptoUniverse] = None


def get_crypto_universe() -> CryptoUniverse:
    global _crypto_universe
    if _crypto_universe is None:
        _crypto_universe = CryptoUniverse()
    return _crypto_universe
