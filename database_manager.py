#!/usr/bin/env python3
"""
Database Manager — SQLite backend for the v2 stocks trading system.

Replaces the v1 PostgreSQL backend. Same `get_database_manager()` accessor and
`execute_query()` shape so existing call sites keep working.

Tables:
    market_data            : OHLCV bars by symbol/timeframe
    trading_signals        : composite-score signals emitted by the evaluator
    trades                 : opened/closed positions with strategy attribution
    trade_journal          : one row per executed action (JSONL also written)
    crowding_analysis      : HERD-001 outputs over time
    regime_changes         : market-regime transitions
    system_health          : heartbeat/error log
    download_sessions      : historical-data download progress (compat)
    news_events            : news ingestion + sentiment scores
    smart_money_signals    : alternative-data outputs
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from config import DB_PATH
except ImportError:
    DB_PATH = os.getenv("DB_PATH", "data/trading.db")


@dataclass
class MarketDataRecord:
    symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_volume: float = 0.0
    trade_count: int = 0
    data_quality: float = 1.0


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS market_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        open_time TIMESTAMP NOT NULL,
        close_time TIMESTAMP NOT NULL,
        open_price REAL NOT NULL,
        high_price REAL NOT NULL,
        low_price REAL NOT NULL,
        close_price REAL NOT NULL,
        volume REAL NOT NULL,
        quote_volume REAL DEFAULT 0,
        trade_count INTEGER DEFAULT 0,
        data_quality REAL DEFAULT 1.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(symbol, timeframe, open_time)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trading_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        action TEXT NOT NULL,
        strategy TEXT,
        strategy_type TEXT,
        confidence REAL,
        signal_strength REAL,
        composite_score REAL,
        position_size REAL,
        entry_price REAL,
        stop_loss REAL,
        take_profit REAL,
        reasoning TEXT,
        signals_snapshot TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        strategy TEXT,
        strategy_type TEXT,
        side TEXT NOT NULL,
        qty REAL NOT NULL,
        entry_price REAL NOT NULL,
        entry_time TIMESTAMP NOT NULL,
        exit_price REAL,
        exit_time TIMESTAMP,
        stop_loss REAL,
        take_profit REAL,
        max_hold_days INTEGER,
        pnl REAL,
        pnl_pct REAL,
        status TEXT DEFAULT 'open',
        order_id TEXT,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trade_journal (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP NOT NULL,
        symbol TEXT NOT NULL,
        action TEXT NOT NULL,
        strategy TEXT,
        strategy_type TEXT,
        signals_snapshot TEXT,
        reasoning TEXT,
        entry_price REAL,
        exit_price REAL,
        pnl REAL,
        hold_days INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS crowding_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        herding_strength REAL,
        crowding_score REAL,
        contrarian_opportunity REAL,
        responsibility_score REAL,
        analysis_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS regime_changes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        regime TEXT NOT NULL,
        prev_regime TEXT,
        confidence REAL,
        timestamp TIMESTAMP NOT NULL,
        details_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS system_health (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP NOT NULL,
        component TEXT NOT NULL,
        status TEXT NOT NULL,
        message TEXT,
        metrics_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS download_sessions (
        session_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        start_date TIMESTAMP,
        end_date TIMESTAMP,
        timeframe TEXT,
        status TEXT,
        records_processed INTEGER DEFAULT 0,
        total_records_expected INTEGER DEFAULT 0,
        progress_percentage REAL DEFAULT 0,
        error_count INTEGER DEFAULT 0,
        last_error TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS news_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        headline TEXT NOT NULL,
        url TEXT,
        published_at TIMESTAMP NOT NULL,
        source TEXT,
        finbert_score REAL,
        claude_score REAL,
        combined_score REAL,
        impact_magnitude REAL,
        category TEXT,
        details_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS smart_money_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        congress_score REAL,
        insider_score REAL,
        institutional_score REAL,
        dark_pool_score REAL,
        composite_score REAL,
        details_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_market_data_sym_time ON market_data(symbol, open_time)",
    "CREATE INDEX IF NOT EXISTS idx_signals_sym_time ON trading_signals(symbol, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status, symbol)",
    "CREATE INDEX IF NOT EXISTS idx_journal_time ON trade_journal(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_news_sym_time ON news_events(symbol, published_at)",
]


class DatabaseManager:
    """SQLite-backed manager with the v1 ``execute_query`` interface."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize_schema()

    # --------------------------------------------------------------- internals

    def _initialize_schema(self) -> None:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for stmt in SCHEMA:
                    cursor.execute(stmt)
                conn.commit()
            logger.info(f"SQLite schema ready at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite schema: {e}")

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _convert_query(query: str) -> str:
        """Convert psycopg2-style %s placeholders to sqlite ? placeholders."""
        # Naive but sufficient: count standalone %s tokens.
        return query.replace("%s", "?")

    # ------------------------------------------------------------------ API

    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch: bool = False,
        fetch_one: bool = False,
    ) -> Optional[Any]:
        sql = self._convert_query(query)
        try:
            with self._lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params or ())
                if fetch_one:
                    row = cursor.fetchone()
                    return dict(row) if row else None
                if fetch:
                    return [dict(r) for r in cursor.fetchall()]
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Query failed ({sql[:80]}...): {e}")
            raise

    def executemany(self, query: str, rows: List[Tuple]) -> int:
        sql = self._convert_query(query)
        try:
            with self._lock, self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(sql, rows)
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Bulk query failed ({sql[:80]}...): {e}")
            raise

    # ------------------------------------------------------------------ helpers

    async def log_market_data(self, record: Dict[str, Any]) -> bool:
        try:
            self.execute_query(
                """
                INSERT OR IGNORE INTO market_data
                  (symbol, timeframe, open_time, close_time, open_price, high_price,
                   low_price, close_price, volume, quote_volume, trade_count, data_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["symbol"],
                    record.get("timeframe", "1d"),
                    record["open_time"],
                    record["close_time"],
                    record["open_price"],
                    record["high_price"],
                    record["low_price"],
                    record["close_price"],
                    record["volume"],
                    record.get("quote_volume", 0.0),
                    record.get("trade_count", 0),
                    record.get("data_quality", 1.0),
                ),
            )
            return True
        except Exception as e:
            logger.error(f"log_market_data failed: {e}")
            return False

    async def log_historical_market_data(
        self,
        symbol: str,
        timeframe: str,
        open_time,
        close_time,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
        **kwargs: Any,
    ) -> bool:
        return await self.log_market_data(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "open_time": open_time,
                "close_time": close_time,
                "open_price": open_price,
                "high_price": high_price,
                "low_price": low_price,
                "close_price": close_price,
                "volume": volume,
                "quote_volume": kwargs.get("quote_volume", 0.0),
                "trade_count": kwargs.get("trade_count", 0),
            }
        )

    async def log_download_session(
        self,
        session_id: str,
        symbol: str,
        start_date,
        end_date,
        timeframe: str,
        status: str,
        records_processed: int,
        total_records_expected: int,
        progress_percentage: float,
        current_timestamp=None,
        error_count: int = 0,
        last_error: Optional[str] = None,
        download_rate_per_second: float = 0.0,
        estimated_completion=None,
        completed_at=None,
    ) -> bool:
        try:
            self.execute_query(
                """
                INSERT INTO download_sessions
                  (session_id, symbol, start_date, end_date, timeframe, status,
                   records_processed, total_records_expected, progress_percentage,
                   error_count, last_error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                  status=excluded.status,
                  records_processed=excluded.records_processed,
                  progress_percentage=excluded.progress_percentage,
                  error_count=excluded.error_count,
                  last_error=excluded.last_error,
                  updated_at=CURRENT_TIMESTAMP
                """,
                (
                    session_id, symbol, start_date, end_date, timeframe, status,
                    records_processed, total_records_expected, progress_percentage,
                    error_count, last_error,
                ),
            )
            return True
        except Exception as e:
            logger.error(f"log_download_session failed: {e}")
            return False

    def log_trade_journal(self, entry: Dict[str, Any]) -> bool:
        try:
            self.execute_query(
                """
                INSERT INTO trade_journal
                  (timestamp, symbol, action, strategy, strategy_type, signals_snapshot,
                   reasoning, entry_price, exit_price, pnl, hold_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.get("timestamp", datetime.utcnow()),
                    entry["symbol"],
                    entry["action"],
                    entry.get("strategy"),
                    entry.get("strategy_type"),
                    json.dumps(entry.get("signals_snapshot", {}), default=str),
                    entry.get("reasoning"),
                    entry.get("entry_price"),
                    entry.get("exit_price"),
                    entry.get("pnl"),
                    entry.get("hold_days"),
                ),
            )
            return True
        except Exception as e:
            logger.error(f"log_trade_journal failed: {e}")
            return False


_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager
