#!/usr/bin/env python3
"""
Enhanced Trading Bot - Main System
"""

import time
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from database_manager import get_database_manager
from signal_evaluator import SignalEvaluator
from database_manager import get_database_manager  # ← AGREGAR AQUÍ

# Import existing managers
try:
    from config import get_trading_config, TRADING_PAIRS
    from data_manager import DataManager
    from signal_evaluator import SignalEvaluator
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, paper_trading=True):
        self.paper_trading = paper_trading
        self.db_manager = get_database_manager()
        self.is_running = False
        self.active_positions = {}

        # Initialize managers
        try:
            self.data_manager = DataManager()
            self.signal_evaluator = SignalEvaluator()
            logger.info("✅ Trading bot initialized")
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")

        try:
            self.db_manager = get_database_manager()
            logger.info("✅ Database manager initialized")
        except Exception as e:
            logger.warning(f"Database manager not available: {e}")
            self.db_manager = None
    
    def start(self):
        """Start the trading bot"""
        self.is_running = True
        logger.info("🚀 Trading bot started")
        
        # Simple trading loop
        while self.is_running:
            try:
                self.trading_cycle()
                time.sleep(60)  # Wait 1 minute
            except KeyboardInterrupt:
                logger.info("�� Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Trading cycle error: {e}")
                time.sleep(30)
    
    def trading_cycle(self):
        """Execute one trading cycle"""
        logger.info("📊 Running trading cycle...")
        
        # Get market data for primary symbol
        symbol = TRADING_PAIRS[0] if TRADING_PAIRS else 'BTCUSDT'
        
        try:
            # Get signal
            market_data = self.data_manager.get_market_data(symbol)
            signal = self.signal_evaluator.evaluate_signal(symbol, market_data)
            if hasattr(self, 'db_manager') and self.db_manager and signal:
            
                try:
                    self.db_manager.execute_query(
                        "INSERT INTO trading_signals_enhanced (symbol, action, confidence, signal_strength, position_size, entry_price) VALUES (%s, %s, %s, %s, %s, %s)",
                        (signal.get('symbol'), signal.get('action'), float(signal.get('confidence', 0)), float(signal.get('signal_strength', 0)), float(signal.get('position_size', 0)), float(signal.get('entry_price', 0)))
                    )
                    logger.info(f"💾 Signal saved to database for {symbol}")
                except Exception as e:
                     logger.error(f"Failed to save signal: {e}")
            logger.info(f"📈 Signal for {symbol}: {signal.get('action', 'HOLD')}")
            
            return {"status": "completed", "symbol": symbol, "signal": signal}
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}")
            return {"status": "error", "error": str(e)}
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        logger.info("🛑 Trading bot stopped")

# Test function

    @property
    def symbol(self):
        """Default trading symbol"""
        return getattr(self, '_symbol', 'BTCUSDT')
    
    @symbol.setter  
    def symbol(self, value):
        """Set trading symbol"""
        self._symbol = value

    def run_trading_cycle(self):
        """Run a single trading cycle (wrapper for trading_cycle)"""
        return self.trading_cycle()

def test_bot():
    """Test bot functionality"""
    try:
        bot = TradingBot(paper_trading=True)
        result = bot.trading_cycle()
        print(f"✅ Bot test successful: {result}")
        return True
    except Exception as e:
        print(f"❌ Bot test failed: {e}")
        return False

if __name__ == "__main__":
    test_bot()


if __name__ == "__main__":
    test_bot()
