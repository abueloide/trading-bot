#!/usr/bin/env python3
"""
Database Manager - Complete Database Integration for Trading Bot
Handles all database operations with enterprise-grade error handling and connection pooling
"""

import psycopg2
import psycopg2.pool
import psycopg2.extras
import logging
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import time

logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'crypto_trading_db'),
    'user': os.getenv('DB_USER', 'trading_bot_app'),
    'password': os.getenv('DB_PASSWORD', 'TradingBot2025'),
    'sslmode': os.getenv('DB_SSLMODE', 'prefer'),
    'connect_timeout': 10,
    'application_name': 'crypto_trading_bot'
}

CONNECTION_POOL_CONFIG = {
    'minconn': 2,
    'maxconn': 10,
    'host': DATABASE_CONFIG['host'],
    'port': DATABASE_CONFIG['port'],
    'database': DATABASE_CONFIG['database'],
    'user': DATABASE_CONFIG['user'],
    'password': DATABASE_CONFIG['password'],
    'sslmode': DATABASE_CONFIG['sslmode'],
    'connect_timeout': DATABASE_CONFIG['connect_timeout'],
    'application_name': DATABASE_CONFIG['application_name']
}

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class TradeExecution:
    """Trade execution record"""
    symbol: str
    side: str
    quantity: float
    price: float
    total_value: float
    fee: float
    timestamp: datetime
    signal_id: str
    strategy_name: str
    execution_time_ms: float
    slippage: float
    market_impact: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class MarketDataRecord:
    """Market data record"""
    symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_volume: float
    trade_count: int
    data_quality: float

@dataclass
class SystemHealthMetric:
    """System health metric record"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    available_capital: float
    active_positions: int
    daily_trades: int
    daily_pnl: float
    system_status: str
    error_count: int
    last_error: Optional[str] = None

# =============================================================================
# DATABASE MANAGER CLASS
# =============================================================================

class DatabaseManager:
    """Enterprise-grade database manager for trading bot"""
    
    def __init__(self):
        self.connection_pool = None
        self.connection_lock = threading.Lock()
        self.last_health_check = None
        self.health_check_interval = 300  # 5 minutes
        self.batch_buffer = []
        self.batch_lock = threading.Lock()
        self.max_batch_size = 100
        
        # Initialize connection pool
        self._initialize_connection_pool()

    def _initialize_connection_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(**CONNECTION_POOL_CONFIG)
            logger.info("Database connection pool initialized successfully")
            
            # Test connection
            if self._test_connection():
                logger.info("Database connection test successful")
            else:
                logger.error("Database connection test failed")
                
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            self.connection_pool = None

    def _test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    return result[0] == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        connection = None
        try:
            if self.connection_pool:
                connection = self.connection_pool.getconn()
                yield connection
            else:
                raise Exception("Connection pool not available")
        except Exception as e:
            logger.error(f"Failed to get database connection: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection and self.connection_pool:
                self.connection_pool.putconn(connection)

    def execute_query(self, query: str, params: Optional[Tuple] = None, 
                     fetch: bool = False, fetch_one: bool = False) -> Optional[Any]:
        """Execute database query with error handling"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    
                    if fetch_one:
                        return cursor.fetchone()
                    elif fetch:
                        return cursor.fetchall()
                    else:
                        conn.commit()
                        return cursor.rowcount
                        
        except Exception as e:
            logger.error(f"Database query failed: {query[:100]}... Error: {e}")
            raise

    def log_trade_execution(self, trade_data: Dict):
        """Log trade execution to database"""
        try:
            query = """
                INSERT INTO trade_executions (
                    symbol, side, quantity, price, total_value, fee, timestamp,
                    signal_id, strategy_name, execution_time_ms, slippage,
                    market_impact, success, error_message
                ) VALUES (
                    %(symbol)s, %(side)s, %(quantity)s, %(price)s, %(total_value)s,
                    %(fee)s, %(timestamp)s, %(signal_id)s, %(strategy_name)s,
                    %(execution_time_ms)s, %(slippage)s, %(market_impact)s,
                    %(success)s, %(error_message)s
                )
            """
            
            params = {
                'symbol': trade_data.get('symbol'),
                'side': trade_data.get('side'),
                'quantity': trade_data.get('quantity', 0),
                'price': trade_data.get('price', 0),
                'total_value': trade_data.get('total_value', 0),
                'fee': trade_data.get('fee', 0),
                'timestamp': trade_data.get('timestamp', datetime.now()),
                'signal_id': trade_data.get('signal_id', ''),
                'strategy_name': trade_data.get('strategy_name', 'default'),
                'execution_time_ms': trade_data.get('execution_time_ms', 0),
                'slippage': trade_data.get('slippage', 0),
                'market_impact': trade_data.get('market_impact', 0),
                'success': trade_data.get('success', True),
                'error_message': trade_data.get('error_message')
            }
            
            self.execute_query(query, params)
            logger.debug(f"Trade execution logged: {trade_data.get('symbol')} {trade_data.get('side')}")
            
        except Exception as e:
            logger.error(f"Failed to log trade execution: {e}")

    def log_market_data(self, market_data: Dict):
        """Log market data to database"""
        try:
            query = """
                INSERT INTO market_data_temporal (
                    symbol, timeframe, open_time, close_time, open_price,
                    high_price, low_price, close_price, volume, quote_volume,
                    trade_count, data_quality, created_at
                ) VALUES (
                    %(symbol)s, %(timeframe)s, %(open_time)s, %(close_time)s,
                    %(open_price)s, %(high_price)s, %(low_price)s, %(close_price)s,
                    %(volume)s, %(quote_volume)s, %(trade_count)s, %(data_quality)s,
                    NOW()
                ) ON CONFLICT (symbol, timeframe, open_time) DO UPDATE SET
                    close_price = EXCLUDED.close_price,
                    high_price = GREATEST(market_data_temporal.high_price, EXCLUDED.high_price),
                    low_price = LEAST(market_data_temporal.low_price, EXCLUDED.low_price),
                    volume = EXCLUDED.volume,
                    quote_volume = EXCLUDED.quote_volume,
                    data_quality = EXCLUDED.data_quality,
                    created_at = NOW()
            """
            
            params = {
                'symbol': market_data.get('symbol'),
                'timeframe': market_data.get('timeframe', '1h'),
                'open_time': market_data.get('open_time', datetime.now() - timedelta(hours=1)),
                'close_time': market_data.get('close_time', datetime.now()),
                'open_price': market_data.get('open_price', 0),
                'high_price': market_data.get('high_price', 0),
                'low_price': market_data.get('low_price', 0),
                'close_price': market_data.get('close_price', 0),
                'volume': market_data.get('volume', 0),
                'quote_volume': market_data.get('quote_volume', 0),
                'trade_count': market_data.get('trade_count', 0),
                'data_quality': market_data.get('data_quality', 1.0)
            }
            
            self.execute_query(query, params)
            logger.debug(f"Market data logged: {market_data.get('symbol')}")
            
        except Exception as e:
            logger.error(f"Failed to log market data: {e}")

    def log_signal_analysis(self, signal_data: Dict):
        """Log signal analysis to database"""
        try:
            query = """
                INSERT INTO signal_analysis_history (
                    signal_id, symbol, signal_type, strength, confidence,
                    technical_indicators, crowding_analysis, timestamp,
                    strategy_name, market_regime, recommendation
                ) VALUES (
                    %(signal_id)s, %(symbol)s, %(signal_type)s, %(strength)s,
                    %(confidence)s, %(technical_indicators)s, %(crowding_analysis)s,
                    %(timestamp)s, %(strategy_name)s, %(market_regime)s, %(recommendation)s
                )
            """
            
            # Extract crowding analysis if available
            crowding_analysis = signal_data.get('crowding_analysis')
            if crowding_analysis:
                crowding_json = json.dumps(crowding_analysis)
            else:
                crowding_json = None
            
            # Extract technical indicators
            technical_indicators = signal_data.get('technical_indicators', {})
            technical_json = json.dumps(technical_indicators) if technical_indicators else None
            
            params = {
                'signal_id': signal_data.get('signal_id', f"signal_{int(time.time())}"),
                'symbol': signal_data.get('symbol'),
                'signal_type': signal_data.get('action', 'UNKNOWN'),
                'strength': signal_data.get('signal_strength', 0),
                'confidence': signal_data.get('confidence', 0),
                'technical_indicators': technical_json,
                'crowding_analysis': crowding_json,
                'timestamp': signal_data.get('timestamp', datetime.now()),
                'strategy_name': signal_data.get('strategy_name', 'default'),
                'market_regime': crowding_analysis.get('market_herding_analysis', {}).get('market_regime', 'UNKNOWN') if crowding_analysis else 'UNKNOWN',
                'recommendation': signal_data.get('action', 'HOLD')
            }
            
            self.execute_query(query, params)
            logger.debug(f"Signal analysis logged: {signal_data.get('symbol')} {signal_data.get('action')}")
            
        except Exception as e:
            logger.error(f"Failed to log signal analysis: {e}")

    def log_crowding_analysis(self, crowding_data: Dict):
        """Log HERD-001 crowding analysis to database"""
        try:
            query = """
                INSERT INTO crowding_analysis_history (
                    analysis_id, symbol, direction, size_usd, market_herding_score,
                    trade_crowding_score, final_action, timing_delay_seconds,
                    size_adjustment_factor, responsibility_score, analysis_details,
                    timestamp, market_regime
                ) VALUES (
                    %(analysis_id)s, %(symbol)s, %(direction)s, %(size_usd)s,
                    %(market_herding_score)s, %(trade_crowding_score)s, %(final_action)s,
                    %(timing_delay_seconds)s, %(size_adjustment_factor)s, %(responsibility_score)s,
                    %(analysis_details)s, %(timestamp)s, %(market_regime)s
                )
            """
            
            # Extract analysis components
            market_herding = crowding_data.get('market_herding_analysis', {})
            trade_crowding = crowding_data.get('trade_crowding_analysis', {})
            final_decision = crowding_data.get('final_decision', {})
            metadata = crowding_data.get('analysis_metadata', {})
            
            params = {
                'analysis_id': f"herd_{int(time.time())}_{hash(str(crowding_data)) % 10000}",
                'symbol': metadata.get('symbol'),
                'direction': metadata.get('direction'),
                'size_usd': metadata.get('original_size_usd', 0),
                'market_herding_score': market_herding.get('overall_herding_score', 0),
                'trade_crowding_score': trade_crowding.get('trade_crowding_score', 0),
                'final_action': final_decision.get('action', 'UNKNOWN'),
                'timing_delay_seconds': final_decision.get('timing_delay_seconds', 0),
                'size_adjustment_factor': final_decision.get('size_adjustment_factor', 1.0),
                'responsibility_score': final_decision.get('responsibility_score', 0),
                'analysis_details': json.dumps(crowding_data),
                'timestamp': metadata.get('analysis_timestamp', datetime.now()),
                'market_regime': market_herding.get('market_regime', 'UNKNOWN')
            }
            
            self.execute_query(query, params)
            logger.debug(f"Crowding analysis logged: {metadata.get('symbol')} {final_decision.get('action')}")
            
        except Exception as e:
            logger.error(f"Failed to log crowding analysis: {e}")

    def log_system_health(self, health_data: Dict):
        """Log system health metrics to database"""
        try:
            query = """
                INSERT INTO system_health_metrics (
                    timestamp, cpu_usage, memory_usage, available_capital,
                    active_positions, daily_trades, daily_pnl, system_status,
                    error_count, last_error
                ) VALUES (
                    %(timestamp)s, %(cpu_usage)s, %(memory_usage)s, %(available_capital)s,
                    %(active_positions)s, %(daily_trades)s, %(daily_pnl)s, %(system_status)s,
                    %(error_count)s, %(last_error)s
                )
            """
            
            params = {
                'timestamp': health_data.get('timestamp', datetime.now()),
                'cpu_usage': health_data.get('cpu_usage', 0),
                'memory_usage': health_data.get('memory_usage', 0),
                'available_capital': health_data.get('available_capital', 0),
                'active_positions': health_data.get('active_positions', 0),
                'daily_trades': health_data.get('daily_trades', 0),
                'daily_pnl': health_data.get('daily_pnl', 0),
                'system_status': health_data.get('system_status', 'UNKNOWN'),
                'error_count': health_data.get('error_count', 0),
                'last_error': health_data.get('last_error')
            }
            
            self.execute_query(query, params)
            logger.debug(f"System health logged: {health_data.get('system_status')}")
            
        except Exception as e:
            logger.error(f"Failed to log system health: {e}")

    def log_regime_classification(self, regime_data: Dict):
        """Log market regime classification to database"""
        try:
            query = """
                INSERT INTO regime_classification_history (
                    classification_id, symbol, regime_type, confidence_score,
                    volatility_estimate, trend_strength, regime_features,
                    timestamp, duration_estimate, transition_probability
                ) VALUES (
                    %(classification_id)s, %(symbol)s, %(regime_type)s, %(confidence_score)s,
                    %(volatility_estimate)s, %(trend_strength)s, %(regime_features)s,
                    %(timestamp)s, %(duration_estimate)s, %(transition_probability)s
                )
            """
            
            params = {
                'classification_id': f"regime_{int(time.time())}",
                'symbol': regime_data.get('symbol'),
                'regime_type': regime_data.get('regime_type', 'UNKNOWN'),
                'confidence_score': regime_data.get('confidence_score', 0),
                'volatility_estimate': regime_data.get('volatility_estimate', 0),
                'trend_strength': regime_data.get('trend_strength', 0),
                'regime_features': json.dumps(regime_data.get('features', {})),
                'timestamp': regime_data.get('timestamp', datetime.now()),
                'duration_estimate': regime_data.get('duration_estimate', 0),
                'transition_probability': regime_data.get('transition_probability', 0)
            }
            
            self.execute_query(query, params)
            logger.debug(f"Regime classification logged: {regime_data.get('symbol')} {regime_data.get('regime_type')}")
            
        except Exception as e:
            logger.error(f"Failed to log regime classification: {e}")

    def get_recent_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get recent trade executions"""
        try:
            if symbol:
                query = """
                    SELECT * FROM trade_executions 
                    WHERE symbol = %s 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """
                params = (symbol, limit)
            else:
                query = """
                    SELECT * FROM trade_executions 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """
                params = (limit,)
            
            results = self.execute_query(query, params, fetch=True)
            return [dict(row) for row in results] if results else []

    async def log_download_session(self, session_id: str, symbol: str, start_date, 
                                   end_date, timeframe: str, status: str,
        
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []

    async def log_download_session(self, session_id: str, symbol: str, start_date, 
                                   end_date, timeframe: str, status: str,
                                   records_processed: int, total_records_expected: int,
                                   progress_percentage: float, current_timestamp=None,
                                   error_count: int = 0, last_error: str = None) -> bool:
        """Log download session for historical_data_downloader"""
        try:
            query = """
                INSERT INTO data_download_sessions 
                (session_id, symbol, start_date, end_date, timeframe, status,
                 records_processed, total_records_expected, progress_percentage,
                 current_timestamp, error_count, last_error, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (session_id) 
                DO UPDATE SET status = EXCLUDED.status, updated_at = NOW()
            """
            
            self.execute_query(query, (
                session_id, symbol, start_date, end_date, timeframe, status,
                records_processed, total_records_expected, progress_percentage,
                current_timestamp, error_count, last_error
            ))
            return True
            
        except Exception as e:
            logger.error(f"Failed to log download session: {e}")
            return False
