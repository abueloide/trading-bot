#!/usr/bin/env python3
"""
Crypto Universe Configuration - 30 Primary Trading Pairs
Selection based on market cap, liquidity, and trading volume
"""

from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class CryptoPair:
    """Configuration for a crypto trading pair"""
    symbol: str
    name: str
    market_cap_rank: int
    category: str
    min_volume_24h: float
    volatility_tier: str
    liquidity_score: float
    enabled: bool = True

# =============================================================================
# 30 PRIMARY CRYPTO TRADING PAIRS
# =============================================================================

CRYPTO_UNIVERSE: List[CryptoPair] = [
    # TIER 1: BLUE CHIP (Top 10 by market cap)
    CryptoPair("BTCUSDT", "Bitcoin", 1, "Store of Value", 50000000000, "LOW", 1.0),
    CryptoPair("ETHUSDT", "Ethereum", 2, "Smart Contract Platform", 20000000000, "MEDIUM", 0.95),
    CryptoPair("BNBUSDT", "BNB", 3, "Exchange Token", 2000000000, "MEDIUM", 0.90),
    CryptoPair("XRPUSDT", "Ripple", 4, "Payments", 3000000000, "HIGH", 0.85),
    CryptoPair("SOLUSDT", "Solana", 5, "Smart Contract Platform", 1500000000, "HIGH", 0.88),
    CryptoPair("ADAUSDT", "Cardano", 6, "Smart Contract Platform", 800000000, "MEDIUM", 0.82),
    CryptoPair("DOGEUSDT", "Dogecoin", 7, "Meme/Payments", 2000000000, "HIGH", 0.80),
    CryptoPair("AVAXUSDT", "Avalanche", 8, "Smart Contract Platform", 500000000, "HIGH", 0.85),
    CryptoPair("TRXUSDT", "TRON", 9, "Smart Contract Platform", 400000000, "MEDIUM", 0.78),
    CryptoPair("LINKUSDT", "Chainlink", 10, "Oracle Network", 300000000, "MEDIUM", 0.82),

    # TIER 2: LARGE CAP (Top 11-20)
    CryptoPair("DOTUSDT", "Polkadot", 11, "Interoperability", 200000000, "HIGH", 0.80),
    CryptoPair("MATICUSDT", "Polygon", 12, "Layer 2 Scaling", 250000000, "HIGH", 0.83),
    CryptoPair("LTCUSDT", "Litecoin", 13, "Digital Silver", 800000000, "MEDIUM", 0.85),
    CryptoPair("SHIBUSDT", "Shiba Inu", 14, "Meme Token", 500000000, "VERY_HIGH", 0.75),
    CryptoPair("UNIUSDT", "Uniswap", 15, "DEX Protocol", 150000000, "HIGH", 0.78),
    CryptoPair("ATOMUSDT", "Cosmos", 16, "Interoperability", 100000000, "HIGH", 0.76),
    CryptoPair("ETCUSDT", "Ethereum Classic", 17, "Smart Contract Platform", 200000000, "HIGH", 0.70),
    CryptoPair("XLMUSDT", "Stellar", 18, "Payments", 150000000, "MEDIUM", 0.72),
    CryptoPair("FILUSDT", "Filecoin", 19, "Decentralized Storage", 80000000, "HIGH", 0.68),
    CryptoPair("HBARUSDT", "Hedera", 20, "Enterprise Blockchain", 60000000, "MEDIUM", 0.65),

    # TIER 3: MID CAP GROWTH (Top 21-30)
    CryptoPair("NEARUSDT", "NEAR Protocol", 21, "Smart Contract Platform", 50000000, "HIGH", 0.70),
    CryptoPair("VETUSDT", "VeChain", 22, "Supply Chain", 40000000, "MEDIUM", 0.68),
    CryptoPair("ALGOUSDT", "Algorand", 23, "Smart Contract Platform", 45000000, "MEDIUM", 0.70),
    CryptoPair("FTMUSDT", "Fantom", 24, "Smart Contract Platform", 35000000, "HIGH", 0.65),
    CryptoPair("SANDUSDT", "The Sandbox", 25, "Gaming/Metaverse", 30000000, "VERY_HIGH", 0.62),
    CryptoPair("MANAUSDT", "Decentraland", 26, "Gaming/Metaverse", 25000000, "VERY_HIGH", 0.60),
    CryptoPair("AAVEUSDT", "Aave", 27, "DeFi Lending", 40000000, "HIGH", 0.72),
    CryptoPair("AXSUSDT", "Axie Infinity", 28, "Gaming", 20000000, "VERY_HIGH", 0.58),
    CryptoPair("FLOWUSDT", "Flow", 29, "Gaming/NFT Platform", 15000000, "HIGH", 0.55),
    CryptoPair("COMPUSDT", "Compound", 30, "DeFi Lending", 30000000, "HIGH", 0.68),
]

# =============================================================================
# UNIVERSE CATEGORIZATION
# =============================================================================

def get_universe_by_tier(tier: str) -> List[str]:
    """Get symbols by market cap tier"""
    tier_ranges = {
        "TIER1": (1, 10),
        "TIER2": (11, 20), 
        "TIER3": (21, 30)
    }
    
    start, end = tier_ranges.get(tier, (1, 30))
    return [pair.symbol for pair in CRYPTO_UNIVERSE 
            if start <= pair.market_cap_rank <= end and pair.enabled]

def get_universe_by_category(category: str) -> List[str]:
    """Get symbols by category"""
    return [pair.symbol for pair in CRYPTO_UNIVERSE 
            if pair.category == category and pair.enabled]

def get_universe_by_volatility(volatility_tier: str) -> List[str]:
    """Get symbols by volatility tier"""
    return [pair.symbol for pair in CRYPTO_UNIVERSE 
            if pair.volatility_tier == volatility_tier and pair.enabled]

def get_high_liquidity_universe(min_liquidity: float = 0.75) -> List[str]:
    """Get symbols with high liquidity scores"""
    return [pair.symbol for pair in CRYPTO_UNIVERSE 
            if pair.liquidity_score >= min_liquidity and pair.enabled]

def get_full_universe() -> List[str]:
    """Get all enabled symbols"""
    return [pair.symbol for pair in CRYPTO_UNIVERSE if pair.enabled]

def get_conservative_universe() -> List[str]:
    """Get conservative trading universe (low-medium volatility, high liquidity)"""
    return [pair.symbol for pair in CRYPTO_UNIVERSE 
            if pair.volatility_tier in ["LOW", "MEDIUM"] 
            and pair.liquidity_score >= 0.80 
            and pair.enabled]

def get_aggressive_universe() -> List[str]:
    """Get aggressive trading universe (high volatility, decent liquidity)"""
    return [pair.symbol for pair in CRYPTO_UNIVERSE 
            if pair.volatility_tier in ["HIGH", "VERY_HIGH"] 
            and pair.liquidity_score >= 0.60 
            and pair.enabled]

# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

TRADING_PRESETS = {
    "CONSERVATIVE": {
        "symbols": get_conservative_universe(),
        "max_positions": 5,
        "position_size_pct": 0.15,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.06,
        "description": "Conservative trading with blue chips and stable coins"
    },
    
    "BALANCED": {
        "symbols": get_universe_by_tier("TIER1") + get_universe_by_tier("TIER2")[:5],
        "max_positions": 8,
        "position_size_pct": 0.12,
        "stop_loss_pct": 0.04,
        "take_profit_pct": 0.08,
        "description": "Balanced approach with top 15 cryptocurrencies"
    },
    
    "AGGRESSIVE": {
        "symbols": get_aggressive_universe(),
        "max_positions": 12,
        "position_size_pct": 0.08,
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.12,
        "description": "Aggressive trading with high volatility assets"
    },
    
    "FULL_UNIVERSE": {
        "symbols": get_full_universe(),
        "max_positions": 15,
        "position_size_pct": 0.06,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.10,
        "description": "Full 30-coin universe with diversified exposure"
    },
    
    "DEFI_FOCUSED": {
        "symbols": get_universe_by_category("DeFi Lending") + ["UNIUSDT", "AAVEUSDT", "COMPUSDT"],
        "max_positions": 6,
        "position_size_pct": 0.15,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.10,
        "description": "DeFi-focused trading strategy"
    },
    
    "SMART_CONTRACT_PLATFORMS": {
        "symbols": get_universe_by_category("Smart Contract Platform"),
        "max_positions": 10,
        "position_size_pct": 0.10,
        "stop_loss_pct": 0.04,
        "take_profit_pct": 0.08,
        "description": "Smart contract platform focused trading"
    }
}

# =============================================================================
# DYNAMIC UNIVERSE MANAGEMENT
# =============================================================================

class UniverseManager:
    """Manages dynamic crypto universe selection"""
    
    def __init__(self, preset: str = "BALANCED"):
        self.current_preset = preset
        self.active_symbols = TRADING_PRESETS[preset]["symbols"]
        self.custom_filters = {}
    
    def set_preset(self, preset: str):
        """Change trading preset"""
        if preset in TRADING_PRESETS:
            self.current_preset = preset
            self.active_symbols = TRADING_PRESETS[preset]["symbols"]
            return True
        return False
    
    def add_custom_filter(self, name: str, filter_func):
        """Add custom filter function"""
        self.custom_filters[name] = filter_func
    
    def apply_liquidity_filter(self, min_liquidity: float = 0.70):
        """Filter by minimum liquidity score"""
        filtered_symbols = []
        for symbol in self.active_symbols:
            pair = next((p for p in CRYPTO_UNIVERSE if p.symbol == symbol), None)
            if pair and pair.liquidity_score >= min_liquidity:
                filtered_symbols.append(symbol)
        self.active_symbols = filtered_symbols
    
    def apply_volatility_filter(self, allowed_tiers: List[str]):
        """Filter by volatility tiers"""
        filtered_symbols = []
        for symbol in self.active_symbols:
            pair = next((p for p in CRYPTO_UNIVERSE if p.symbol == symbol), None)
            if pair and pair.volatility_tier in allowed_tiers:
                filtered_symbols.append(symbol)
        self.active_symbols = filtered_symbols
    
    def apply_market_cap_filter(self, max_rank: int = 20):
        """Filter by maximum market cap rank"""
        filtered_symbols = []
        for symbol in self.active_symbols:
            pair = next((p for p in CRYPTO_UNIVERSE if p.symbol == symbol), None)
            if pair and pair.market_cap_rank <= max_rank:
                filtered_symbols.append(symbol)
        self.active_symbols = filtered_symbols
    
    def get_active_symbols(self) -> List[str]:
        """Get currently active symbols"""
        return self.active_symbols
    
    def get_universe_stats(self) -> Dict[str, Any]:
        """Get statistics about current universe"""
        active_pairs = [p for p in CRYPTO_UNIVERSE if p.symbol in self.active_symbols]
        
        if not active_pairs:
            return {}
        
        return {
            "total_symbols": len(active_pairs),
            "avg_liquidity_score": sum(p.liquidity_score for p in active_pairs) / len(active_pairs),
            "volatility_distribution": {
                tier: len([p for p in active_pairs if p.volatility_tier == tier])
                for tier in ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
            },
            "category_distribution": {
                cat: len([p for p in active_pairs if p.category == cat])
                for cat in set(p.category for p in active_pairs)
            },
            "tier_distribution": {
                f"TIER{i}": len([p for p in active_pairs if 1 + (i-1)*10 <= p.market_cap_rank <= i*10])
                for i in range(1, 4)
            }
        }

# =============================================================================
# UTILITIES
# =============================================================================

def get_symbol_info(symbol: str) -> Dict[str, Any]:
    """Get detailed information about a symbol"""
    pair = next((p for p in CRYPTO_UNIVERSE if p.symbol == symbol), None)
    if not pair:
        return {}
    
    return {
        "symbol": pair.symbol,
        "name": pair.name,
        "market_cap_rank": pair.market_cap_rank,
        "category": pair.category,
        "volatility_tier": pair.volatility_tier,
        "liquidity_score": pair.liquidity_score,
        "enabled": pair.enabled,
        "tier": f"TIER{(pair.market_cap_rank - 1) // 10 + 1}"
    }

def print_universe_summary():
    """Print a summary of the crypto universe"""
    print("🪙 Crypto Universe Summary (30 Pairs)")
    print("=" * 50)
    
    for tier in ["TIER1", "TIER2", "TIER3"]:
        symbols = get_universe_by_tier(tier)
        print(f"\n{tier} ({len(symbols)} pairs):")
        for symbol in symbols:
            pair_info = get_symbol_info(symbol)
            print(f"  • {symbol:<12} {pair_info['name']:<20} ({pair_info['volatility_tier']})")
    
    print(f"\n📊 Universe Statistics:")
    print(f"  • Total Pairs: {len(get_full_universe())}")
    print(f"  • Conservative: {len(get_conservative_universe())} pairs")
    print(f"  • Aggressive: {len(get_aggressive_universe())} pairs")
    print(f"  • High Liquidity: {len(get_high_liquidity_universe())} pairs")

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Default symbols for backward compatibility
TARGET_SYMBOLS = get_universe_by_tier("TIER1")[:10]  # Top 10 by default

# Create default universe manager
default_universe = UniverseManager("BALANCED")

if __name__ == "__main__":
    print_universe_summary()
