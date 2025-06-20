#!/usr/bin/env python3
"""
Configuration File - Complete Trading Bot Configuration
Includes HERD-001 Market Crowding Detection and Database Integration
"""

import os
from typing import Dict, List, Any
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# CORE TRADING CONFIGURATION
# =============================================================================

# =============================================================================
# EXISTING CONFIGURATION (FROM YOUR BASE)
# =============================================================================

# Import your existing configuration
try:
    from config import (
        get_api_credentials, TARGET_SYMBOLS, DECISION_THRESHOLDS, 
        TOTAL_CAPITAL, RISK_PER_TRADE, ORDER_CONFIG
    )
    
    # Use existing symbols
    TRADING_PAIRS = TARGET_SYMBOLS
    PRIMARY_PAIR = TARGET_SYMBOLS[0] if TARGET_SYMBOLS else 'BTCUSDT'
    
    # Use existing capital settings
    MAX_POSITION_SIZE = RISK_PER_TRADE
    
    # Get API credentials using your existing function
    credentials = get_api_credentials()
    API_KEY = credentials.get('BINANCE_API', '')
    API_SECRET = credentials.get('BINANCE_SECRET', '')
    
except ImportError:
    # Fallback configuration if imports fail
    TARGET_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
    TRADING_PAIRS = TARGET_SYMBOLS
    PRIMARY_PAIR = 'BTCUSDT'
    TOTAL_CAPITAL = 1000.0
    RISK_PER_TRADE = 0.02
    MAX_POSITION_SIZE = 0.02
    API_KEY = ''
    API_SECRET = ''

# Additional settings
TESTNET = os.getenv('USE_TESTNET', 'true').lower() == 'true'
BASE_URL = 'https://testnet.binance.vision' if TESTNET else 'https://api.binance.com'
MAX_DAILY_LOSS = 0.05  # 5% daily loss limit
MAX_TOTAL_POSITIONS = 3

# Risk management
STOP_LOSS_PERCENTAGE = 0.02  # 2% stop loss
TAKE_PROFIT_PERCENTAGE = 0.04  # 4% take profit
MAX_SLIPPAGE = 0.001  # 0.1% maximum slippage

# Trading schedule
TRADING_HOURS_START = 0  # 24/7 for crypto
TRADING_HOURS_END = 24
MAX_DAILY_TRADES = 20

# Data collection intervals
DATA_COLLECTION_INTERVAL = 60  # seconds
PRICE_HISTORY_LENGTH = 200
VOLUME_HISTORY_LENGTH = 200

# =============================================================================
# TECHNICAL ANALYSIS CONFIGURATION (COMPATIBLE WITH YOUR BASE)
# =============================================================================

# Use your existing technical config if available
try:
    from config import TECHNICAL_CONFIG
    RSI_PERIOD = TECHNICAL_CONFIG.get('RSI_PERIOD', 14)
    RSI_OVERBOUGHT = TECHNICAL_CONFIG.get('RSI_OVERBOUGHT', 70)
    RSI_OVERSOLD = TECHNICAL_CONFIG.get('RSI_OVERSOLD', 30)
except ImportError:
    # Fallback values
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

EMA_SHORT = 10
EMA_LONG = 50

# Signal thresholds - use your existing decision thresholds
try:
    MIN_SIGNAL_STRENGTH = DECISION_THRESHOLDS.get('BUY', 6.5) / 10  # Convert to 0-1 scale
    MIN_CONFIDENCE_LEVEL = 0.65
except:
    MIN_SIGNAL_STRENGTH = 0.6
    MIN_CONFIDENCE_LEVEL = 0.65

SIGNAL_TIMEOUT_MINUTES = 30

# =============================================================================
# HERD-001 MARKET CROWDING DETECTION CONFIGURATION
# =============================================================================

# Enable/disable HERD-001 features
ENABLE_CROWDING_DETECTION = os.getenv('ENABLE_CROWDING_DETECTION', 'true').lower() == 'true'
ENABLE_ANTI_HERDING = os.getenv('ENABLE_ANTI_HERDING', 'true').lower() == 'true'
ENABLE_TIMING_DECORRELATION = os.getenv('ENABLE_TIMING_DECORRELATION', 'true').lower() == 'true'

# HERD-001 Core Configuration
CROWDING_CONFIG = {
    # Market herding analysis
    'correlation_analysis': {
        'enabled': True,
        'correlation_window': 100,
        'correlation_threshold': 0.7,
        'cross_asset_correlation_weight': 0.4
    },
    
    # Sentiment herding analysis
    'sentiment_analysis': {
        'enabled': True,
        'sentiment_threshold': 0.7,
        'extreme_rsi_threshold': 80,
        'fear_greed_weight': 0.3,
        'momentum_alignment_weight': 0.4
    },
    
    # Volume herding analysis
    'volume_analysis': {
        'enabled': True,
        'volume_spike_threshold': 2.0,
        'volume_trend_window': 20,
        'burst_activity_threshold': 3.0
    },
    
    # Trade crowding analysis
    'trade_crowding': {
        'enabled': True,
        'order_book_clustering_threshold': 0.7,
        'directional_bias_threshold': 0.8,
        'size_concentration_threshold': 0.6,
        'timing_correlation_window': 50
    },
    
    # Responsibility scoring
    'responsibility_scoring': {
        'market_weight': 0.6,
        'trade_weight': 0.4,
        'confidence_adjustment': True,
        'regime_adjustment': True
    }
}

# HERD-001 Decision Thresholds
CROWDING_THRESHOLDS = {
    'extreme_crowding': 0.8,    # Block trades
    'high_crowding': 0.6,       # Reduce size significantly
    'moderate_crowding': 0.4,   # Apply timing delays
    'low_crowding': 0.2         # Normal execution
}

# HERD-001 Response Configuration
ANTI_HERDING_CONFIG = {
    # Timing decorrelation
    'timing_delays': {
        'max_delay_seconds': 180,
        'min_delay_seconds': 0,
        'delay_randomization': True,
        'delay_distribution': 'uniform'  # 'uniform', 'exponential', 'normal'
    },
    
    # Position size adjustments
    'size_adjustments': {
        'min_size_factor': 0.4,     # Minimum 40% of original size
        'max_size_factor': 1.0,     # Maximum 100% of original size
        'adjustment_curve': 'linear' # 'linear', 'exponential', 'logarithmic'
    },
    
    # Trade blocking
    'trade_blocking': {
        'enable_blocking': True,
        'block_threshold': 0.85,
        'block_duration_minutes': 15,
        'emergency_block_threshold': 0.95
    },
    
    # Market regime adjustments
    'regime_adjustments': {
        'extreme_herding_multiplier': 1.3,
        'high_volatility_multiplier': 1.2,
        'low_volatility_multiplier': 1.1,
        'normal_market_multiplier': 0.8
    }
}

# HERD-001 Performance Configuration
CROWDING_PERFORMANCE_CONFIG = {
    'analysis_cache_ttl': 300,      # 5 minutes
    'max_cache_size': 100,
    'analysis_timeout_seconds': 5,
    'fallback_on_timeout': True,
    'fallback_responsibility_score': 0.5
}

# =============================================================================
# DATABASE INTEGRATION CONFIGURATION
# =============================================================================

# Enable/disable database features
ENABLE_DATABASE = os.getenv('ENABLE_DATABASE', 'true').lower() == 'true'
DATABASE_LOGGING = os.getenv('DATABASE_LOGGING', 'true').lower() == 'true'

# Database connection configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'crypto_trading_db'),
    'user': os.getenv('DB_USER', 'trading_bot_app'),
    'password': os.getenv('DB_PASSWORD', 'TradingBot2025'),
    'sslmode': os.getenv('DB_SSLMODE', 'prefer'),
    'connect_timeout': 10,
    'application_name': 'crypto_trading_bot'
}

# Database connection pool configuration
DB_POOL_CONFIG = {
    'minconn': 2,
    'maxconn': 10,
    'connection_retry_attempts': 3,
    'connection_retry_delay': 5,
    'health_check_interval': 300  # 5 minutes
}

# Database logging configuration
DB_LOGGING_CONFIG = {
    'log_market_data': True,
    'log_trades': True,
    'log_signals': True,
    'log_crowding_analysis': True,
    'log_regime_changes': True,
    'log_system_health': True,
    'batch_size': 100,
    'flush_interval_seconds': 60
}

# =============================================================================
# MARKET REGIME DETECTION CONFIGURATION
# =============================================================================

# Enable market regime detection
ENABLE_REGIME_DETECTION = os.getenv('ENABLE_REGIME_DETECTION', 'true').lower() == 'true'

REGIME_CONFIG = {
    'volatility_windows': [20, 50, 100],
    'trend_detection_period': 50,
    'regime_classification': {
        'bull_market_threshold': 0.15,
        'bear_market_threshold': -0.15,
        'sideways_volatility_threshold': 0.02,
        'high_volatility_threshold': 0.05
    },
    'regime_confidence_threshold': 0.7,
    'regime_persistence_periods': 5
}

# =============================================================================
# CRISIS DETECTION CONFIGURATION
# =============================================================================

# Enable crisis detection
ENABLE_CRISIS_DETECTION = os.getenv('ENABLE_CRISIS_DETECTION', 'true').lower() == 'true'

CRISIS_CONFIG = {
    'flash_crash_detection': {
        'price_drop_threshold': 0.10,      # 10% drop
        'time_window_minutes': 15,
        'volume_spike_threshold': 5.0
    },
    'liquidity_crisis_detection': {
        'bid_ask_spread_threshold': 0.005, # 0.5%
        'order_book_depth_threshold': 0.3,
        'market_impact_threshold': 0.01
    },
    'volatility_crisis_detection': {
        'volatility_spike_threshold': 3.0,
        'volatility_persistence_periods': 3
    }
}

# =============================================================================
# MICROSTRUCTURE DATA CONFIGURATION
# =============================================================================

# Enable microstructure data collection
ENABLE_MICROSTRUCTURE = os.getenv('ENABLE_MICROSTRUCTURE', 'true').lower() == 'true'

MICROSTRUCTURE_CONFIG = {
    'order_book_depth_levels': 20,
    'order_book_update_interval': 1,    # seconds
    'trade_flow_analysis_window': 300,  # 5 minutes
    'market_impact_estimation': True,
    'liquidity_metrics': {
        'bid_ask_spread': True,
        'effective_spread': True,
        'price_impact': True,
        'order_book_imbalance': True
    }
}

# =============================================================================
# SYSTEM MONITORING CONFIGURATION
# =============================================================================

# System health monitoring
SYSTEM_MONITORING_CONFIG = {
    'health_check_interval': 60,        # seconds
    'performance_logging_interval': 300, # 5 minutes
    'error_alert_threshold': 5,
    'memory_usage_alert_threshold': 0.8, # 80%
    'cpu_usage_alert_threshold': 0.9,    # 90%
}

# Logging configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_logging': True,
    'log_file': 'trading_bot.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
    'console_logging': True
}

# Alert configuration
ALERT_CONFIG = {
    'enable_alerts': os.getenv('ENABLE_ALERTS', 'false').lower() == 'true',
    'email_alerts': os.getenv('ENABLE_EMAIL_ALERTS', 'false').lower() == 'true',
    'webhook_alerts': os.getenv('ENABLE_WEBHOOK_ALERTS', 'false').lower() == 'true',
    'alert_thresholds': {
        'large_loss': 0.05,              # 5% loss
        'system_error': True,
        'database_connection_loss': True,
        'exchange_connection_loss': True
    }
}

# =============================================================================
# BACKTESTING AND SIMULATION CONFIGURATION
# =============================================================================

BACKTESTING_CONFIG = {
    'enable_backtesting': False,
    'historical_data_start': '2023-01-01',
    'historical_data_end': '2024-01-01',
    'initial_capital': 10000,
    'commission_rate': 0.001,  # 0.1% commission
    'slippage_model': 'linear'
}

# Paper trading configuration
PAPER_TRADING_CONFIG = {
    'enable_paper_trading': os.getenv('PAPER_TRADING', 'false').lower() == 'true',
    'simulated_capital': 10000,
    'realistic_execution_delays': True,
    'simulated_slippage': True,
    'order_fill_simulation': 'market_impact'
}

# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Feature toggle system
FEATURE_FLAGS = {
    # Core features
    'enable_trading': os.getenv('ENABLE_TRADING', 'true').lower() == 'true',
    'enable_stop_losses': True,
    'enable_take_profits': True,
    
    # HERD-001 features
    'enable_herd001': ENABLE_CROWDING_DETECTION,
    'enable_market_herding_analysis': True,
    'enable_trade_crowding_analysis': True,
    'enable_timing_decorrelation': True,
    'enable_position_sizing_adjustment': True,
    'enable_trade_blocking': True,
    
    # Database features
    'enable_database_logging': DATABASE_LOGGING,
    'enable_analytics': True,
    'enable_performance_tracking': True,
    
    # Advanced features
    'enable_regime_detection': ENABLE_REGIME_DETECTION,
    'enable_crisis_detection': ENABLE_CRISIS_DETECTION,
    'enable_microstructure_analysis': ENABLE_MICROSTRUCTURE,
    
    # Development features
    'enable_debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
    'enable_verbose_logging': os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true',
    'enable_performance_profiling': os.getenv('ENABLE_PROFILING', 'false').lower() == 'true'
}

# =============================================================================
# VALIDATION AND SAFETY CHECKS
# =============================================================================

def validate_configuration():
    """Validate configuration settings - compatible with existing base"""
    errors = []
    
    # Test if we can get API credentials
    try:
        creds = get_api_credentials() if 'get_api_credentials' in globals() else {}
        if not creds.get('BINANCE_API') and not TESTNET:
            errors.append("Binance API credentials missing for live trading")
    except Exception as e:
        errors.append(f"Failed to get API credentials: {e}")
    
    # Validate basic settings
    try:
        if TOTAL_CAPITAL <= 0:
            errors.append("Total capital must be positive")
        
        if MAX_POSITION_SIZE <= 0 or MAX_POSITION_SIZE > 1:
            errors.append("Max position size must be between 0 and 1")
        
        if not TRADING_PAIRS:
            errors.append("At least one trading pair must be configured")
        
        if PRIMARY_PAIR not in TRADING_PAIRS:
            errors.append("Primary pair must be in trading pairs list")
            
    except Exception as e:
        errors.append(f"Configuration validation error: {e}")
    
    return errors

# =============================================================================
# CONFIGURATION EXPORT
# =============================================================================

def get_trading_config() -> Dict[str, Any]:
    """Get core trading configuration - compatible with existing base"""
    try:
        creds = get_api_credentials() if 'get_api_credentials' in globals() else {}
        return {
            'api_key': creds.get('BINANCE_API', API_KEY),
            'api_secret': creds.get('BINANCE_SECRET', API_SECRET),
            'base_url': BASE_URL,
            'trading_pairs': TRADING_PAIRS,
            'total_capital': TOTAL_CAPITAL,
            'max_position_size': MAX_POSITION_SIZE,
            'stop_loss_percentage': STOP_LOSS_PERCENTAGE,
            'take_profit_percentage': TAKE_PROFIT_PERCENTAGE
        }
    except Exception as e:
        logger.error(f"Error getting trading config: {e}")
        return {
            'api_key': API_KEY,
            'api_secret': API_SECRET,
            'base_url': BASE_URL,
            'trading_pairs': TRADING_PAIRS,
            'total_capital': TOTAL_CAPITAL,
            'max_position_size': MAX_POSITION_SIZE,
            'stop_loss_percentage': STOP_LOSS_PERCENTAGE,
            'take_profit_percentage': TAKE_PROFIT_PERCENTAGE
        }

def get_herd001_config() -> Dict[str, Any]:
    """Get HERD-001 configuration"""
    return {
        'enabled': ENABLE_CROWDING_DETECTION,
        'crowding_config': CROWDING_CONFIG,
        'thresholds': CROWDING_THRESHOLDS,
        'anti_herding_config': ANTI_HERDING_CONFIG,
        'performance_config': CROWDING_PERFORMANCE_CONFIG
    }

def get_database_config() -> Dict[str, Any]:
    """Get database configuration"""
    return {
        'enabled': ENABLE_DATABASE,
        'connection': DATABASE_CONFIG,
        'pool': DB_POOL_CONFIG,
        'logging': DB_LOGGING_CONFIG
    }

# =============================================================================
# RUNTIME CONFIGURATION VALIDATION
# =============================================================================

# Validate configuration on import
_config_errors = validate_configuration()
if _config_errors:
    import logging
    logger = logging.getLogger(__name__)
    logger.error("Configuration validation errors:")
    for error in _config_errors:
        logger.error(f"  - {error}")
    
    if not os.getenv('IGNORE_CONFIG_ERRORS', 'false').lower() == 'true':
        raise ValueError(f"Configuration validation failed: {_config_errors}")

# =============================================================================
# END OF CONFIGURATION
# =============================================================================
# =============================================================================
# MISSING CRITICAL PARAMETERS - ADDED BY FIX
# =============================================================================

# Position and Risk Management
MAX_POSITION_SIZE_USD = 50.0  # Safe initial amount for testing
SCAN_INTERVAL = 300  # 5 minutes between scans

# API Base URLs (if missing)
if 'BASE_URL' not in globals():
    BASE_URL = "https://api.binance.com"

# Ensure API credentials are properly set
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

if not BINANCE_API_KEY:
    print("⚠️ Warning: BINANCE_API_KEY not set - using demo mode")
    BINANCE_API_KEY = "demo_key_for_testing"

if not BINANCE_SECRET_KEY:
    print("⚠️ Warning: BINANCE_SECRET_KEY not set - using demo mode")
    BINANCE_SECRET_KEY = "demo_secret_for_testing"

# Safety validation function
def validate_config():
    """Validate configuration parameters"""
    issues = []
    
    if not MAX_POSITION_SIZE_USD or MAX_POSITION_SIZE_USD <= 0:
        issues.append("MAX_POSITION_SIZE_USD must be > 0")
    
    if not SCAN_INTERVAL or SCAN_INTERVAL <= 0:
        issues.append("SCAN_INTERVAL must be > 0")
        
    if not TARGET_SYMBOLS:
        issues.append("TARGET_SYMBOLS cannot be empty")
    
    return len(issues) == 0, issues

# Regime configuration
REGIME_CONFIG = {
    'enabled': True,
    'update_interval': 3600,  # 1 hour
    'confidence_threshold': 0.6
}

# Crisis configuration  
CRISIS_CONFIG = {
    'enabled': True,
    'check_interval': 300,  # 5 minutes
    'auto_shutdown': True
}

# Manipulation configuration
MANIPULATION_CONFIG = {
    'enabled': True,
    'sensitivity': 'MEDIUM',
    'block_trades': True
}

print("✅ Configuration parameters added successfully")

# Override API keys directly (hardcoded)
os.environ['BINANCE_API_KEY'] = 'cJh2vz60bb5O6RR2BqcZTuCokMjVW1HIWA0pikVLf8xiPbNnoq2SSvGozpbyZoPj'
os.environ['BINANCE_SECRET_KEY'] = 'bEnTaarhiD0QHqBpwferv0a4SeKrW5XT3b91NyQeIEBYqnTQkf3BRDtSRNSE2NeJ'

# =============================================================================
# TELEGRAM CONFIGURATION
# =============================================================================
import os
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7839884699:AAEowznxtXpRLuTEqZk9rIcT5K3Gv5QCXlE')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '7266808827')
ENABLE_TELEGRAM = True
