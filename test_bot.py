#!/usr/bin/env python3
import sys
import os

# Load environment variables
from pathlib import Path
env_file = Path('.env')
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Test imports
try:
    import config
    print("✅ config.py loaded")
except Exception as e:
    print(f"❌ config.py error: {e}")

try:
    import data_manager
    print("✅ data_manager.py loaded")
except Exception as e:
    print(f"❌ data_manager.py error: {e}")

try:
    import trading_bot
    print("✅ trading_bot.py loaded")
except Exception as e:
    print(f"❌ trading_bot.py error: {e}")

print("🎯 Basic test complete")
