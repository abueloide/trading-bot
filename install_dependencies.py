#!/usr/bin/env python3
"""
Dependency Installer for Enhanced Trading Bot
Installs and validates all required dependencies
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a Python package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Install all required dependencies"""
    print("📦 Installing Enhanced Trading Bot Dependencies")
    print("=" * 50)
    
    # Required packages
    packages = [
        "python-telegram-bot==20.3",
        "python-binance",
        "pandas",
        "numpy",
        "matplotlib",
        "requests",
        "psycopg2-binary",
        "python-dotenv",
        "asyncio",
        "aiohttp"
    ]
    
    # Install packages
    failed_packages = []
    
    for package in packages:
        print(f"📦 Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed")
        else:
            print(f"❌ {package} failed")
            failed_packages.append(package)
    
    # Results
    print("\n" + "=" * 50)
    if failed_packages:
        print(f"❌ Installation completed with {len(failed_packages)} failures:")
        for package in failed_packages:
            print(f"  • {package}")
        print("\nTry installing manually:")
        for package in failed_packages:
            print(f"pip install {package}")
    else:
        print("✅ All dependencies installed successfully!")
        print("\n🚀 Next steps:")
        print("1. python3 setup_telegram.py")
        print("2. python3 run_trading_system.py")

if __name__ == "__main__":
    main()
