#!/usr/bin/env python3
"""
Database Manager - Complete Database Integration for Trading Bot
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

class DatabaseManager:
    def __init__(self):
        self.connection_pool = None
        self.connection_lock = threading.Lock()
        self._initialize_connection_pool()

    def _initialize_connection_pool(self):
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(**CONNECTION_POOL_CONFIG)
            logger.info("Database connection pool initialized successfully")
            if self._test_connection():
                logger.info("Database connection test successful")
            else:
                logger.error("Database connection test failed")
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            self.connection_pool = None

    def _test_connection(self) -> bool:
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

    def execute_query(self, query: str, params: Optional[Tuple] = None, fetch: bool = False, fetch_one: bool = False) -> Optional[Any]:
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

    async def log_download_session(self, session_id: str, symbol: str, start_date, end_date, timeframe: str, status: str, records_processed: int, total_records_expected: int, progress_percentage: float, current_timestamp=None, error_count: int = 0, last_error: str = None, download_rate_per_second: float = 0.0, estimated_completion=None, completed_at=None) -> bool:
        try:
            query = "INSERT INTO data_download_sessions (session_id, symbol, start_date, end_date, timeframe, status, records_processed, total_records_expected, progress_percentage, error_count, last_error, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()) ON CONFLICT (session_id) DO UPDATE SET status = EXCLUDED.status, updated_at = NOW()"
            self.execute_query(query, (session_id, symbol, start_date, end_date, timeframe, status, records_processed, total_records_expected, progress_percentage, error_count, last_error))
            return True
        except Exception as e:
            logger.error(f"Failed to log download session: {e}")
            return False

    async def log_historical_market_data(self, symbol: str, timeframe: str, open_time, close_time, open_price: float, high_price: float, low_price: float, close_price: float, volume: float, **kwargs) -> bool:
        try:
            query = "INSERT INTO market_data (symbol, timeframe, open_time, close_time, open_price, high_price, low_price, close_price, volume, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()) ON CONFLICT (symbol, open_time) DO NOTHING"
            self.execute_query(query, (symbol, timeframe, open_time, close_time, open_price, high_price, low_price, close_price, volume))
            return True
        except Exception as e:
            logger.error(f"Failed to log historical market data: {e}")
            return False

_database_manager = None

def get_database_manager():
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager
