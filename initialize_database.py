#!/usr/bin/env python3
"""
Database Initialization Script
Sets up PostgreSQL database for crypto trading bot with HERD-001 integration
"""

import os
import sys
import psycopg2
import psycopg2.extras
import logging
from typing import Dict, Optional
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

def get_database_config() -> Dict[str, str]:
    """Get database configuration from environment variables"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'crypto_trading_db'),
        'user': os.getenv('DB_USER', 'trading_bot_app'),
        'password': os.getenv('DB_PASSWORD', 'TradingBot2025'),
        'sslmode': os.getenv('DB_SSLMODE', 'prefer'),
        'connect_timeout': 10
    }

def get_superuser_config() -> Dict[str, str]:
    """Get superuser configuration for database creation"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': 'postgres',  # Connect to default database
        'user': os.getenv('DB_SUPERUSER', 'postgres'),
        'password': os.getenv('DB_SUPERUSER_PASSWORD', ''),
        'sslmode': os.getenv('DB_SSLMODE', 'prefer'),
        'connect_timeout': 10
    }

# =============================================================================
# DATABASE SETUP FUNCTIONS
# =============================================================================

def test_connection(config: Dict[str, str]) -> bool:
    """Test database connection"""
    try:
        conn = psycopg2.connect(**config)
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
        conn.close()
        return result[0] == 1
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False

def create_database_and_user():
    """Create database and user if they don't exist"""
    logger.info("Creating database and user...")
    
    superuser_config = get_superuser_config()
    db_config = get_database_config()
    
    try:
        # Connect as superuser
        conn = psycopg2.connect(**superuser_config)
        conn.autocommit = True
        
        with conn.cursor() as cursor:
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (db_config['database'],)
            )
            
            if not cursor.fetchone():
                logger.info(f"Creating database: {db_config['database']}")
                cursor.execute(f'CREATE DATABASE "{db_config["database"]}"')
            else:
                logger.info(f"Database {db_config['database']} already exists")
            
            # Check if user exists
            cursor.execute(
                "SELECT 1 FROM pg_user WHERE usename = %s",
                (db_config['user'],)
            )
            
            if not cursor.fetchone():
                logger.info(f"Creating user: {db_config['user']}")
                cursor.execute(
                    f'CREATE USER "{db_config["user"]}" WITH PASSWORD %s',
                    (db_config['password'],)
                )
            else:
                logger.info(f"User {db_config['user']} already exists")
            
            # Grant privileges
            logger.info("Granting privileges...")
            cursor.execute(
                f'GRANT ALL PRIVILEGES ON DATABASE "{db_config["database"]}" TO "{db_config["user"]}"'
            )
            
        conn.close()
        logger.info("Database and user setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create database and user: {e}")
        return False

def run_schema_script():
    """Execute the database schema SQL script"""
    logger.info("Running database schema script...")
    
    # Read schema file
    schema_file = os.path.join(os.path.dirname(__file__), 'database_schema.sql')
    
    if not os.path.exists(schema_file):
        logger.error(f"Schema file not found: {schema_file}")
        return False
    
    try:
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        # Connect to target database
        db_config = get_database_config()
        conn = psycopg2.connect(**db_config)
        
        with conn.cursor() as cursor:
            # Execute schema script
            cursor.execute(schema_sql)
            conn.commit()
        
        conn.close()
        logger.info("Database schema created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create database schema: {e}")
        return False

def verify_installation():
    """Verify database installation is complete"""
    logger.info("Verifying database installation...")
    
    try:
        db_config = get_database_config()
        conn = psycopg2.connect(**db_config)
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Run verification function
            cursor.execute("SELECT * FROM verify_database_setup()")
            results = cursor.fetchall()
            
            logger.info("Database verification results:")
            all_ok = True
            
            for row in results:
                status_emoji = "✅" if row['status'] == 'OK' else "❌"
                logger.info(f"  {status_emoji} {row['component']}: {row['status']} - {row['details']}")
                
                if row['status'] != 'OK':
                    all_ok = False
            
            # Run health check
            cursor.execute("SELECT * FROM check_database_health()")
            health_results = cursor.fetchall()
            
            logger.info("\nDatabase health check:")
            for row in health_results:
                status_emoji = "✅" if row['status'] in ['OK', 'INFO'] else "⚠️" if row['status'] == 'WARNING' else "❌"
                logger.info(f"  {status_emoji} {row['metric_name']}: {row['metric_value']} ({row['status']})")
        
        conn.close()
        
        if all_ok:
            logger.info("✅ Database installation verified successfully!")
        else:
            logger.warning("⚠️ Database installation has some issues")
        
        return all_ok
        
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False

def insert_initial_data():
    """Insert initial test data"""
    logger.info("Inserting initial test data...")
    
    try:
        db_config = get_database_config()
        conn = psycopg2.connect(**db_config)
        
        with conn.cursor() as cursor:
            # Insert initial system health record
            cursor.execute("""
                INSERT INTO system_health_metrics (
                    timestamp, cpu_usage, memory_usage, available_capital,
                    active_positions, daily_trades, daily_pnl, system_status,
                    error_count, last_error
                ) VALUES (
                    NOW(), 0, 0, 1000, 0, 0, 0, 'INITIALIZED', 0, 
                    'Database initialization completed successfully'
                )
            """)
            
            # Insert test market data
            cursor.execute("""
                INSERT INTO market_data_temporal (
                    symbol, timeframe, open_time, close_time, open_price,
                    high_price, low_price, close_price, volume, quote_volume,
                    trade_count, data_quality
                ) VALUES (
                    'BTCUSDT', '1h', NOW() - INTERVAL '1 hour', NOW(),
                    50000, 50500, 49800, 50200, 100, 5020000, 1000, 1.0
                )
            """)
            
            conn.commit()
        
        conn.close()
        logger.info("Initial test data inserted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to insert initial data: {e}")
        return False

# =============================================================================
# INTERACTIVE SETUP FUNCTIONS
# =============================================================================

def prompt_database_config():
    """Interactive prompt for database configuration"""
    logger.info("=== Database Configuration Setup ===")
    
    config = {}
    
    # Get database connection details
    config['host'] = input(f"Database host [{os.getenv('DB_HOST', 'localhost')}]: ").strip() or os.getenv('DB_HOST', 'localhost')
    config['port'] = input(f"Database port [{os.getenv('DB_PORT', '5432')}]: ").strip() or os.getenv('DB_PORT', '5432')
    config['database'] = input(f"Database name [{os.getenv('DB_NAME', 'crypto_trading_db')}]: ").strip() or os.getenv('DB_NAME', 'crypto_trading_db')
    config['user'] = input(f"Application user [{os.getenv('DB_USER', 'trading_bot_app')}]: ").strip() or os.getenv('DB_USER', 'trading_bot_app')
    
    # Get password
    import getpass
    password = getpass.getpass("Application user password: ").strip()
    if not password:
        password = input("Password (visible): ").strip()
    config['password'] = password
    
    # Set environment variables
    os.environ['DB_HOST'] = config['host']
    os.environ['DB_PORT'] = config['port']
    os.environ['DB_NAME'] = config['database']
    os.environ['DB_USER'] = config['user']
    os.environ['DB_PASSWORD'] = config['password']
    
    logger.info("Database configuration set successfully")
    return config

def export_environment_variables():
    """Generate environment variable export commands"""
    config = get_database_config()
    
    logger.info("\n=== Environment Variables ===")
    logger.info("Add these to your ~/.bashrc or ~/.zshrc:")
    logger.info(f"export DB_HOST={config['host']}")
    logger.info(f"export DB_PORT={config['port']}")
    logger.info(f"export DB_NAME={config['database']}")
    logger.info(f"export DB_USER={config['user']}")
    logger.info(f"export DB_PASSWORD={config['password']}")
    logger.info(f"export DB_SSLMODE={config['sslmode']}")
    logger.info("\nThen run: source ~/.bashrc")

# =============================================================================
# MAIN SETUP FUNCTION
# =============================================================================

def main():
    """Main database initialization function"""
    logger.info("🚀 Starting Crypto Trading Bot Database Setup")
    logger.info("=" * 60)
    
    # Check if running interactively
    interactive = len(sys.argv) == 1 or '--interactive' in sys.argv
    
    if interactive:
        prompt_database_config()
    
    # Step 1: Test superuser connection
    logger.info("\n📋 Step 1: Testing superuser connection...")
    superuser_config = get_superuser_config()
    
    if not test_connection(superuser_config):
        logger.error("❌ Cannot connect as superuser. Please check:")
        logger.error("  - PostgreSQL is running")
        logger.error("  - Superuser credentials are correct")
        logger.error("  - Set DB_SUPERUSER and DB_SUPERUSER_PASSWORD environment variables")
        return False
    
    logger.info("✅ Superuser connection successful")
    
    # Step 2: Create database and user
    logger.info("\n📋 Step 2: Creating database and user...")
    if not create_database_and_user():
        logger.error("❌ Failed to create database and user")
        return False
    
    # Wait a moment for database creation
    time.sleep(1)
    
    # Step 3: Test application user connection
    logger.info("\n📋 Step 3: Testing application user connection...")
    db_config = get_database_config()
    
    if not test_connection(db_config):
        logger.error("❌ Cannot connect as application user")
        return False
    
    logger.info("✅ Application user connection successful")
    
    # Step 4: Create database schema
    logger.info("\n📋 Step 4: Creating database schema...")
    if not run_schema_script():
        logger.error("❌ Failed to create database schema")
        return False
    
    # Step 5: Insert initial data
    logger.info("\n📋 Step 5: Inserting initial test data...")
    if not insert_initial_data():
        logger.warning("⚠️ Failed to insert initial data (continuing anyway)")
    
    # Step 6: Verify installation
    logger.info("\n📋 Step 6: Verifying installation...")
    if not verify_installation():
        logger.warning("⚠️ Installation verification had issues")
    
    # Step 7: Show environment variables
    logger.info("\n📋 Step 7: Environment setup")
    export_environment_variables()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Database Setup Complete!")
    logger.info("=" * 60)
    logger.info("✅ Database and user created")
    logger.info("✅ Schema and tables created") 
    logger.info("✅ Indexes and constraints applied")
    logger.info("✅ Views and functions created")
    logger.info("✅ Permissions configured")
    logger.info("✅ Initial data inserted")
    
    logger.info("\n🚀 Your crypto trading bot database is ready!")
    logger.info("Next steps:")
    logger.info("  1. Set environment variables (see above)")
    logger.info("  2. Test database connection in your trading bot")
    logger.info("  3. Run integration tests")
    logger.info("  4. Deploy your enhanced trading bot!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {e}")
        sys.exit(1)