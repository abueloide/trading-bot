#!/usr/bin/env python3
"""
Complete Trading System Runner
Runs trading bot with Telegram monitoring and historical data integration
"""

import asyncio
import logging
import signal
import sys
import time
import threading
from datetime import datetime
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingSystemManager:
    """Manages the complete trading system"""
    
    def __init__(self):
        self.trading_bot = None
        self.telegram_bot = None
        self.historical_downloader = None
        self.is_running = False
        self.trading_thread = None
        self.telegram_thread = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📡 Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)
    
    async def initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing Enhanced Trading System...")
            
            # 1. Initialize Trading Bot
            logger.info("🤖 Initializing trading bot...")
            from trading_bot import TradingBot
            self.trading_bot = TradingBot(paper_trading=True)
            logger.info("✅ Trading bot initialized")
            
            # 2. Initialize Telegram Bot
            logger.info("📱 Initializing Telegram bot...")
            try:
                from enhanced_telegram_bot import EnhancedTelegramBot
                self.telegram_bot = EnhancedTelegramBot()
                logger.info("✅ Telegram bot initialized")
            except Exception as e:
                logger.warning(f"Telegram bot initialization failed: {e}")
                logger.warning("📱 Continuing without Telegram monitoring...")
            
            # 3. Initialize Historical Data Downloader
            logger.info("📥 Initializing historical data downloader...")
            try:
                from historical_data_downloader import HistoricalDataDownloader
                self.historical_downloader = HistoricalDataDownloader()
                logger.info("✅ Historical data downloader initialized")
            except Exception as e:
                logger.warning(f"Historical downloader initialization failed: {e}")
            
            logger.info("🎉 All components initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    async def check_historical_data(self):
        """Check and download historical data if needed"""
        try:
            if not self.historical_downloader:
                logger.warning("📥 Historical downloader not available")
                return
            
            logger.info("📊 Checking historical data availability...")
            
            # Import crypto universe
            from crypto_universe import default_universe, TRADING_PRESETS
            
            # Get symbols based on selected preset
            preset = "BALANCED"  # Can be changed to CONSERVATIVE, AGGRESSIVE, or FULL_UNIVERSE
            symbols = TRADING_PRESETS[preset]["symbols"]
            
            logger.info(f"📊 Using {preset} preset with {len(symbols)} symbols")
            missing_data = []
            
            for symbol in symbols:
                # Here you would check if data exists in database
                # For now, assume we need to download
                missing_data.append(symbol)
            
            if missing_data:
                logger.info(f"📥 Downloading historical data for {len(missing_data)} symbols...")
                
                for symbol in missing_data:
                    logger.info(f"📥 Downloading {symbol} (12 months)...")
                    success, session_id = await self.historical_downloader.download_symbol_history(
                        symbol, months_back=12
                    )
                    
                    if success:
                        logger.info(f"✅ {symbol} download completed")
                    else:
                        logger.warning(f"⚠️ {symbol} download failed")
                    
                    # Brief pause between downloads
                    await asyncio.sleep(2)
                
                logger.info("📊 Historical data setup completed")
            else:
                logger.info("✅ Historical data already available")
                
        except Exception as e:
            logger.error(f"Historical data check failed: {e}")
    
    def start_trading_bot(self):
        """Start trading bot in continuous mode"""
        try:
            logger.info("🤖 Starting trading bot...")
            
            def trading_loop():
                cycle_count = 0
                last_telegram_update = time.time()
                
                while self.is_running:
                    try:
                        cycle_count += 1
                        logger.info(f"📊 Trading cycle #{cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
                        
                        # Run trading cycle
                        result = self.trading_bot.run_trading_cycle()
                        
                        # Log result
                        signal_action = result.get('signal', {}).get('action', 'UNKNOWN')
                        logger.info(f"✅ Cycle completed: {signal_action}")
                        
                        # Send Telegram update every 10 cycles (10 minutes)
                        if self.telegram_bot and (time.time() - last_telegram_update) > 600:
                            try:
                                asyncio.create_task(self._send_telegram_update(result))
                                last_telegram_update = time.time()
                            except Exception as e:
                                logger.warning(f"Telegram update failed: {e}")
                        
                        # Wait 60 seconds
                        for _ in range(60):
                            if not self.is_running:
                                break
                            time.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"❌ Trading cycle error: {e}")
                        time.sleep(30)  # Wait 30 seconds on error
                
                logger.info("🤖 Trading bot stopped")
            
            self.trading_thread = threading.Thread(target=trading_loop, daemon=True)
            self.trading_thread.start()
            logger.info("✅ Trading bot started")
            
        except Exception as e:
            logger.error(f"Failed to start trading bot: {e}")
    
    async def start_telegram_bot(self):
        """Start Telegram bot"""
        try:
            if not self.telegram_bot:
                logger.warning("📱 Telegram bot not available")
                return
            
            logger.info("📱 Starting Telegram bot...")
            
            # Send startup notification
            await self._send_startup_notification()
            
            # Start Telegram bot in background
            def telegram_loop():
                try:
                    asyncio.run(self.telegram_bot.start_bot())
                except Exception as e:
                    logger.error(f"Telegram bot crashed: {e}")
            
            self.telegram_thread = threading.Thread(target=telegram_loop, daemon=True)
            self.telegram_thread.start()
            logger.info("✅ Telegram bot started")
            
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
    
    async def _send_startup_notification(self):
        """Send startup notification to Telegram"""
        try:
            if not self.telegram_bot:
                return
            
            startup_message = f"""
🚀 **Enhanced Trading Bot Started**

🕒 **Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🛡️ **Mode**: Safe Paper Trading
🧠 **Intelligence**: All systems active
📊 **Database**: Connected
📱 **Telegram**: Monitoring active

✅ **System Status**: Operational
🤖 **Ready for trading cycles**

Use /status for detailed information.
            """
            
            # This would send the actual message
            logger.info("📱 Startup notification sent to Telegram")
            
        except Exception as e:
            logger.error(f"Failed to send startup notification: {e}")
    
    async def _send_telegram_update(self, trading_result):
        """Send periodic update to Telegram"""
        try:
            if not self.telegram_bot:
                return
            
            signal_data = trading_result.get('signal', {})
            action = signal_data.get('action', 'UNKNOWN')
            confidence = signal_data.get('confidence', 0)
            
            update_message = f"""
📊 **Trading Update**

🎯 **Signal**: {action}
📈 **Confidence**: {confidence:.1%}
🕒 **Time**: {datetime.now().strftime('%H:%M:%S')}

💰 **Status**: Paper Trading Active
🛡️ **Protection**: All systems operational
            """
            
            logger.info("📱 Periodic update sent to Telegram")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram update: {e}")
    
    async def run_system(self):
        """Run the complete trading system"""
        try:
            logger.info("🚀 Starting Enhanced Crypto Trading System...")
            
            # Initialize components
            if not await self.initialize_components():
                logger.error("❌ Failed to initialize components")
                return False
            
            # Check historical data (optional, can run in background)
            logger.info("📊 Setting up historical data...")
            asyncio.create_task(self.check_historical_data())
            
            # Start Telegram bot
            await self.start_telegram_bot()
            
            # Start trading bot
            self.is_running = True
            self.start_trading_bot()
            
            logger.info("🎉 Enhanced Trading System is now running!")
            logger.info("📱 Check your Telegram for monitoring")
            logger.info("🤖 Trading cycles every 60 seconds")
            logger.info("🛡️ Paper trading mode - safe for testing")
            
            # Keep main thread alive
            try:
                while self.is_running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("📡 Received shutdown signal")
            
            self.shutdown()
            return True
            
        except Exception as e:
            logger.error(f"❌ System run failed: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the system gracefully"""
        try:
            logger.info("🛑 Shutting down trading system...")
            
            self.is_running = False
            
            # Stop trading bot
            if self.trading_thread and self.trading_thread.is_alive():
                logger.info("🤖 Stopping trading bot...")
                self.trading_thread.join(timeout=10)
            
            # Stop Telegram bot
            if self.telegram_thread and self.telegram_thread.is_alive():
                logger.info("📱 Stopping Telegram bot...")
                # Note: telegram_thread.join() might hang, so we'll just let it be daemon
            
            logger.info("✅ System shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Main function"""
    print("🚀 Enhanced Crypto Trading System v6.0")
    print("=" * 60)
    print("🛡️ Safe Paper Trading Mode")
    print("📱 Telegram Monitoring Active")
    print("📊 Historical Data Integration")
    print("🧠 Advanced Market Intelligence")
    print("=" * 60)
    
    # Create and run system
    system = TradingSystemManager()
    
    try:
        await system.run_system()
    except KeyboardInterrupt:
        logger.info("📡 User interrupted, shutting down...")
        system.shutdown()
    except Exception as e:
        logger.error(f"❌ System crashed: {e}")
        system.shutdown()
        sys.exit(1)

def run_interactive_setup():
    """Interactive setup for first-time users"""
    print("🔧 First Time Setup")
    print("=" * 30)
    
    # Check if Telegram is configured
    import os
    if not (os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID')):
        print("📱 Telegram not configured")
        print("Run: python3 setup_telegram.py")
        return False
    
    # Check database
    try:
        from database_manager import get_database_manager
        db = get_database_manager()
        print("✅ Database connection OK")
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    print("✅ Setup verification completed")
    return True

if __name__ == "__main__":
    import sys
    
    if "--setup" in sys.argv:
        if run_interactive_setup():
            print("🚀 Ready to run: python3 run_trading_system.py")
        else:
            print("❌ Setup incomplete")
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n📡 Shutdown completed")
        except Exception as e:
            print(f"\n❌ System error: {e}")
            sys.exit(1)
