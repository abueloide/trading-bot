#!/usr/bin/env python3
"""
Configuration File — US-stocks Trading System v2.

Single source of truth for:
- Alpaca API credentials and trading config
- SIGNAL_WEIGHTS (definitive composite signal weights)
- RISK_CONFIG (definitive risk parameters)
- Crowding / anti-herding (HERD-001) preserved from v1
- SQLite database path (replaces PostgreSQL from v1)
- Feature flags
"""

import os
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CORE TRADING CONFIGURATION (US Stocks via Alpaca)
# =============================================================================

# Default universe — paper-trading-safe defaults; overridden by stock_universe.
TARGET_SYMBOLS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "JPM", "JNJ", "UNH", "WMT", "PG", "XOM", "CAT", "HD",
]
TRADING_PAIRS = TARGET_SYMBOLS  # alias for legacy code paths
PRIMARY_PAIR = TARGET_SYMBOLS[0]

# Capital and sizing
TOTAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "5000"))
RISK_PER_TRADE = 0.02
MAX_POSITION_SIZE = 0.25  # max 25% of portfolio per position
MAX_POSITION_SIZE_USD = TOTAL_CAPITAL * MAX_POSITION_SIZE

# Trade plumbing
SCAN_INTERVAL = 300       # 5 minutes between scans
MAX_DAILY_TRADES = 20
MAX_TOTAL_POSITIONS = 4
MAX_DAILY_LOSS = 0.05
STOP_LOSS_PERCENTAGE = 0.02
TAKE_PROFIT_PERCENTAGE = 0.06
MAX_SLIPPAGE = 0.001

# Market hours (NYSE) — strategies/executor enforce these.
TRADING_HOURS_START = 9.5   # 9:30 ET
TRADING_HOURS_END = 16.0    # 16:00 ET
TRADING_DAYS = [0, 1, 2, 3, 4]  # Mon-Fri


# =============================================================================
# ALPACA API CONFIGURATION
# =============================================================================

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_PAPER = ALPACA_BASE_URL.startswith("https://paper-api.")
ALPACA_RATE_LIMIT_PER_MIN = 200

ALPACA_CONFIG: Dict[str, Any] = {
    "api_key": ALPACA_API_KEY,
    "secret_key": ALPACA_SECRET_KEY,
    "base_url": ALPACA_BASE_URL,
    "paper": ALPACA_PAPER,
    "feed": os.getenv("ALPACA_FEED", "iex"),  # iex (free) or sip (paid)
    "rate_limit_per_min": ALPACA_RATE_LIMIT_PER_MIN,
}


def get_api_credentials() -> Dict[str, str]:
    """Compatibility shim: legacy code calls this for Binance creds."""
    return {
        "ALPACA_API_KEY": ALPACA_API_KEY,
        "ALPACA_SECRET_KEY": ALPACA_SECRET_KEY,
        # Aliases retained for any v1 code path not yet migrated.
        "BINANCE_API": ALPACA_API_KEY,
        "BINANCE_SECRET": ALPACA_SECRET_KEY,
    }


# =============================================================================
# DATA COLLECTION CONFIGURATION
# =============================================================================

DATA_COLLECTION_INTERVAL = 60       # seconds between adaptive samples
PRICE_HISTORY_LENGTH = 252          # ~1 year of daily bars
VOLUME_HISTORY_LENGTH = 252


# =============================================================================
# TECHNICAL ANALYSIS CONFIGURATION
# =============================================================================

TECHNICAL_CONFIG = {
    "RSI_PERIOD": 14,
    "RSI_OVERBOUGHT": 70,
    "RSI_OVERSOLD": 30,
    "RSI_FAST_PERIOD": 2,        # for Strategy A/B (mean reversion)
    "RSI_FAST_BUY": 10,
    "RSI_FAST_SELL": 70,
}
RSI_PERIOD = TECHNICAL_CONFIG["RSI_PERIOD"]
RSI_OVERBOUGHT = TECHNICAL_CONFIG["RSI_OVERBOUGHT"]
RSI_OVERSOLD = TECHNICAL_CONFIG["RSI_OVERSOLD"]

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
EMA_SHORT = 10
EMA_LONG = 50

DECISION_THRESHOLDS = {
    "BUY": 6.5,
    "SELL": 3.5,
    "HOLD_LOWER": 3.5,
    "HOLD_UPPER": 6.5,
}
MIN_SIGNAL_STRENGTH = 0.6
MIN_CONFIDENCE_LEVEL = 0.65
SIGNAL_TIMEOUT_MINUTES = 60


# =============================================================================
# DEFINITIVE COMPOSITE SIGNAL WEIGHTS  (Section 4 of the v2 spec)
# These REPLACE the legacy 35/25/20/15/5 scheme. Total must sum to 1.00.
# =============================================================================

SIGNAL_WEIGHTS: Dict[str, float] = {
    "technical_score":     0.25,
    "volume_confirmation": 0.15,
    "regime_alignment":    0.10,
    "crowding_safety":     0.10,
    "risk_reward":         0.05,
    "smart_money_signal":  0.15,
    "news_sentiment":      0.10,
    "macro_regime":        0.10,
}
assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-6, "SIGNAL_WEIGHTS must sum to 1.0"


# =============================================================================
# RISK MANAGEMENT (DEFINITIVE — Section 6 of the v2 spec)
# =============================================================================

RISK_CONFIG: Dict[str, Any] = {
    # Position-level
    "max_position_pct": 0.25,
    "max_risk_per_trade": 0.02,
    "default_stop_loss_pct": 0.02,           # NOT applied to mean reversion
    "default_take_profit_pct": 0.06,
    "mean_reversion_max_hold_days": 10,
    "momentum_rebalance_days": 21,

    # Portfolio-level
    "max_open_positions": 4,
    "max_sector_exposure_pct": 0.40,
    "min_cash_reserve_pct": 0.20,
    "max_cash_reserve_pct": 0.50,

    # Circuit breakers
    "max_daily_loss_pct": 0.05,
    "max_weekly_loss_pct": 0.08,

    # PDT compliance
    "max_day_trades_per_week": 3,
    "pdt_equity_threshold": 25_000,

    # Filters
    "vix_rank_threshold": 50,
    "market_trend_filter": True,
    "earnings_blackout_days": 3,

    # Psychology of Money — risk inversely proportional to portfolio size
    "risk_scaling": {
        "up_to_5k": 0.02,
        "up_to_10k": 0.015,
        "above_10k": 0.01,
    },
}


# =============================================================================
# HERD-001 MARKET CROWDING DETECTION — preserved from v1
# =============================================================================

ENABLE_CROWDING_DETECTION = os.getenv("ENABLE_CROWDING_DETECTION", "true").lower() == "true"
ENABLE_ANTI_HERDING = os.getenv("ENABLE_ANTI_HERDING", "true").lower() == "true"

CROWDING_CONFIG: Dict[str, Any] = {
    "correlation_analysis": {
        "enabled": True,
        "correlation_window": 100,
        "correlation_threshold": 0.7,
        "cross_asset_correlation_weight": 0.4,
    },
    "sentiment_analysis": {
        "enabled": True,
        "sentiment_threshold": 0.7,
        "extreme_rsi_threshold": 80,
        "fear_greed_weight": 0.3,
        "momentum_alignment_weight": 0.4,
    },
    "volume_analysis": {
        "enabled": True,
        "volume_spike_threshold": 2.0,
        "volume_trend_window": 20,
        "burst_activity_threshold": 3.0,
    },
    "trade_crowding": {
        "enabled": True,
        "order_book_clustering_threshold": 0.7,
        "directional_bias_threshold": 0.8,
        "size_concentration_threshold": 0.6,
        "timing_correlation_window": 50,
    },
    "responsibility_scoring": {
        "market_weight": 0.6,
        "trade_weight": 0.4,
        "confidence_adjustment": True,
        "regime_adjustment": True,
    },
}

CROWDING_THRESHOLDS = {
    "extreme_crowding": 0.85,    # block
    "high_crowding": 0.6,        # reduce size
    "moderate_crowding": 0.4,    # delay
    "low_crowding": 0.2,
}

# NOTE: TIMING_DECORRELATION removed in v2 — irrelevant for daily stock scans.
ANTI_HERDING_CONFIG: Dict[str, Any] = {
    "size_adjustments": {
        "min_size_factor": 0.4,
        "max_size_factor": 1.0,
        "adjustment_curve": "linear",
    },
    "trade_blocking": {
        "enable_blocking": True,
        "block_threshold": 0.85,
        "emergency_block_threshold": 0.95,
    },
    "regime_adjustments": {
        "extreme_herding_multiplier": 1.3,
        "high_volatility_multiplier": 1.2,
        "low_volatility_multiplier": 1.1,
        "normal_market_multiplier": 0.8,
    },
}

CROWDING_PERFORMANCE_CONFIG = {
    "analysis_cache_ttl": 300,
    "max_cache_size": 100,
    "analysis_timeout_seconds": 5,
    "fallback_on_timeout": True,
    "fallback_responsibility_score": 0.5,
}


# =============================================================================
# DATABASE (SQLite — replaces PostgreSQL from v1)
# =============================================================================

ENABLE_DATABASE = os.getenv("ENABLE_DATABASE", "true").lower() == "true"
DATABASE_LOGGING = os.getenv("DATABASE_LOGGING", "true").lower() == "true"

DB_TYPE = os.getenv("DB_TYPE", "sqlite")
DB_PATH = os.getenv("DB_PATH", "data/trading.db")

DATABASE_CONFIG: Dict[str, Any] = {
    "type": DB_TYPE,
    "path": DB_PATH,
}

DB_LOGGING_CONFIG = {
    "log_market_data": True,
    "log_trades": True,
    "log_signals": True,
    "log_crowding_analysis": True,
    "log_regime_changes": True,
    "log_system_health": True,
    "batch_size": 100,
    "flush_interval_seconds": 60,
}


# =============================================================================
# REGIME / MICROSTRUCTURE / CRISIS — preserved feature flags
# =============================================================================

ENABLE_REGIME_DETECTION = os.getenv("ENABLE_REGIME_DETECTION", "true").lower() == "true"
ENABLE_MICROSTRUCTURE = os.getenv("ENABLE_MICROSTRUCTURE", "true").lower() == "true"
ENABLE_CRISIS_DETECTION = os.getenv("ENABLE_CRISIS_DETECTION", "true").lower() == "true"

REGIME_CONFIG = {
    "enabled": ENABLE_REGIME_DETECTION,
    "volatility_windows": [20, 50, 100],
    "trend_detection_period": 50,
    "regime_classification": {
        "bull_market_threshold": 0.15,
        "bear_market_threshold": -0.15,
        "sideways_volatility_threshold": 0.02,
        "high_volatility_threshold": 0.05,
    },
    "regime_confidence_threshold": 0.7,
    "regime_persistence_periods": 5,
    "update_interval": 3600,
}

CRISIS_CONFIG = {
    "enabled": ENABLE_CRISIS_DETECTION,
    "check_interval": 300,
    "auto_shutdown": True,
    "flash_crash_detection": {
        "price_drop_threshold": 0.10,
        "time_window_minutes": 15,
        "volume_spike_threshold": 5.0,
    },
    "liquidity_crisis_detection": {
        "bid_ask_spread_threshold": 0.005,
        "market_impact_threshold": 0.01,
    },
    "volatility_crisis_detection": {
        "volatility_spike_threshold": 3.0,
        "volatility_persistence_periods": 3,
    },
}

MICROSTRUCTURE_CONFIG = {
    # Alpaca free tier is L1 only — depth-based metrics use fallbacks.
    "order_book_depth_levels": 1,
    "trade_flow_analysis_window": 300,
    "market_impact_estimation": True,
    "liquidity_metrics": {
        "bid_ask_spread": True,
        "effective_spread": True,
        "price_impact": True,
        "order_book_imbalance": True,
    },
}

MANIPULATION_CONFIG = {
    "enabled": True,
    "sensitivity": "MEDIUM",
    "block_trades": True,
}


# =============================================================================
# ALTERNATIVE DATA & NEWS — Phase 4/5 config
# =============================================================================

ENABLE_ALTERNATIVE_DATA = os.getenv("ENABLE_ALTERNATIVE_DATA", "false").lower() == "true"
ENABLE_NEWS_INTELLIGENCE = os.getenv("ENABLE_NEWS_INTELLIGENCE", "false").lower() == "true"

ALTERNATIVE_DATA_CONFIG: Dict[str, Any] = {
    "enabled": ENABLE_ALTERNATIVE_DATA,
    "quiver_token": os.getenv("QUIVER_API_TOKEN", ""),
    "congress_lookback_days": 90,
    "insider_window_days": 30,
    # Backtest delays — congress filings have a 45-day lag, insider 2 days.
    "congress_filing_lag_days": 45,
    "insider_filing_lag_days": 2,
    "cache_ttl_seconds": 86_400,  # daily/weekly data, cache aggressively
    "fallback_score": 0.5,
}

SMART_MONEY_WEIGHTS: Dict[str, float] = {
    "insider_cluster": 0.40,
    "congress_direction": 0.30,
    "institutional_flow": 0.20,
    "dark_pool": 0.10,
}
assert abs(sum(SMART_MONEY_WEIGHTS.values()) - 1.0) < 1e-6

NEWS_CONFIG: Dict[str, Any] = {
    "enabled": ENABLE_NEWS_INTELLIGENCE,
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    "alphavantage_api_key": os.getenv("ALPHAVANTAGE_API_KEY", ""),
    "claude_max_calls_per_hour": 30,
    "finbert_weight": 0.6,
    "claude_weight": 0.4,
    "min_impact_magnitude": 0.6,
    "min_confidence": 0.7,
    "narrative_window_days": 30,
}


# =============================================================================
# BINANCE (CRYPTO) CONFIGURATION
# =============================================================================

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
ENABLE_CRYPTO = os.getenv("ENABLE_CRYPTO", "false").lower() == "true"

CRYPTO_CONFIG: Dict[str, Any] = {
    "enabled": ENABLE_CRYPTO,
    "api_key": BINANCE_API_KEY,
    "secret_key": BINANCE_SECRET_KEY,
    "testnet": BINANCE_TESTNET,
    "max_position_pct": 0.20,           # max 20% of crypto allocation per coin
    "default_quote_asset": "USDT",
    "scan_interval": 300,               # 5 min (crypto is 24/7)
}


# =============================================================================
# GEOPOLITICAL ENGINE — Phase B
# =============================================================================

ENABLE_GEOPOLITICAL = os.getenv("ENABLE_GEOPOLITICAL", "false").lower() == "true"

GEOPOLITICAL_CONFIG: Dict[str, Any] = {
    "enabled": ENABLE_GEOPOLITICAL,
    "claude_max_calls_per_hour": 20,
    "event_lookback_hours": 48,
    "regime_hold_hours": 24,
    "rebalance_cooldown_hours": 24,
    "min_events_for_regime_change": 2,
    "tail_event_override": True,
    "fmp_api_key": os.getenv("FMP_API_KEY", "demo"),
    "banxico_token": os.getenv("BANXICO_TOKEN", ""),
}


# =============================================================================
# BACKTESTING
# =============================================================================

BACKTESTING_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "data_source": "alpaca",       # alpaca | yfinance
    "fallback_data_source": "yfinance",
    "in_sample_months": 24,
    "out_of_sample_months": 6,
    "transaction_cost_pct": 0.0005,  # 0.05% slippage per side, ALWAYS
    "results_dir": "backtesting/results",
    "baseline_symbol": "SPY",
}


# =============================================================================
# TELEGRAM
# =============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


# =============================================================================
# LOGGING / MONITORING
# =============================================================================

LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_logging": True,
    "log_file": "trading_bot.log",
    "max_file_size": 10 * 1024 * 1024,
    "backup_count": 5,
    "console_logging": True,
}

SYSTEM_MONITORING_CONFIG = {
    "health_check_interval": 60,
    "performance_logging_interval": 300,
    "error_alert_threshold": 5,
    "memory_usage_alert_threshold": 0.8,
    "cpu_usage_alert_threshold": 0.9,
}

ALERT_CONFIG = {
    "enable_alerts": os.getenv("ENABLE_ALERTS", "false").lower() == "true",
    "alert_thresholds": {
        "large_loss": 0.05,
        "system_error": True,
        "database_connection_loss": True,
        "exchange_connection_loss": True,
    },
}


# =============================================================================
# FEATURE FLAGS
# =============================================================================

FEATURE_FLAGS: Dict[str, bool] = {
    "enable_trading": os.getenv("ENABLE_TRADING", "true").lower() == "true",
    "enable_stop_losses": True,
    "enable_take_profits": True,
    "enable_herd001": ENABLE_CROWDING_DETECTION,
    "enable_market_herding_analysis": True,
    "enable_trade_crowding_analysis": True,
    "enable_position_sizing_adjustment": True,
    "enable_trade_blocking": True,
    "enable_database_logging": DATABASE_LOGGING,
    "enable_analytics": True,
    "enable_performance_tracking": True,
    "enable_regime_detection": ENABLE_REGIME_DETECTION,
    "enable_crisis_detection": ENABLE_CRISIS_DETECTION,
    "enable_microstructure_analysis": ENABLE_MICROSTRUCTURE,
    "enable_alternative_data": ENABLE_ALTERNATIVE_DATA,
    "enable_news_intelligence": ENABLE_NEWS_INTELLIGENCE,
    "enable_geopolitical": ENABLE_GEOPOLITICAL,
    "enable_crypto": ENABLE_CRYPTO,
    "enable_debug_mode": os.getenv("DEBUG_MODE", "false").lower() == "true",
    "enable_verbose_logging": os.getenv("VERBOSE_LOGGING", "false").lower() == "true",
}


# =============================================================================
# CONFIG ACCESSORS / VALIDATION
# =============================================================================

def validate_configuration() -> List[str]:
    errors: List[str] = []
    if not ALPACA_PAPER and not ALPACA_API_KEY:
        errors.append("ALPACA_API_KEY missing for live trading")
    if TOTAL_CAPITAL <= 0:
        errors.append("INITIAL_CAPITAL must be positive")
    if not TRADING_PAIRS:
        errors.append("At least one symbol must be configured")
    return errors


def get_trading_config() -> Dict[str, Any]:
    return {
        "api_key": ALPACA_API_KEY,
        "api_secret": ALPACA_SECRET_KEY,
        "base_url": ALPACA_BASE_URL,
        "paper": ALPACA_PAPER,
        "trading_pairs": TRADING_PAIRS,
        "total_capital": TOTAL_CAPITAL,
        "max_position_size": MAX_POSITION_SIZE,
        "stop_loss_percentage": STOP_LOSS_PERCENTAGE,
        "take_profit_percentage": TAKE_PROFIT_PERCENTAGE,
    }


def get_herd001_config() -> Dict[str, Any]:
    return {
        "enabled": ENABLE_CROWDING_DETECTION,
        "crowding_config": CROWDING_CONFIG,
        "thresholds": CROWDING_THRESHOLDS,
        "anti_herding_config": ANTI_HERDING_CONFIG,
        "performance_config": CROWDING_PERFORMANCE_CONFIG,
    }


def get_database_config() -> Dict[str, Any]:
    return {
        "enabled": ENABLE_DATABASE,
        "type": DB_TYPE,
        "path": DB_PATH,
    }


def get_signal_weights() -> Dict[str, float]:
    return dict(SIGNAL_WEIGHTS)


def get_risk_config() -> Dict[str, Any]:
    return dict(RISK_CONFIG)


# Run a soft validation on import; do not raise unless the user opts in.
_config_errors = validate_configuration()
if _config_errors and os.getenv("STRICT_CONFIG", "false").lower() == "true":
    raise ValueError(f"Configuration validation failed: {_config_errors}")
