#!/usr/bin/env python3
"""
Quick test to verify everything works
Basado en especificación del project knowledge
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """Run a quick test of all components"""
    print("🧪 Quick Test - Enhanced Crypto Trading Bot")
    print("="*50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Import config
    total_tests += 1
    try:
        import config
        print("✅ config.py imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ config.py import failed: {e}")
    
    # Test 2: Import data manager
    total_tests += 1
    try:
        from data_manager import DataManager
        print("✅ data_manager.py imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ data_manager.py import failed: {e}")
    
    # Test 3: Import signal evaluator
    total_tests += 1
    try:
        from signal_evaluator import SignalEvaluator
        print("✅ signal_evaluator.py imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ signal_evaluator.py import failed: {e}")
    
    # Test 4: Import trading bot
    total_tests += 1
    try:
        from trading_bot import TradingBot
        print("✅ trading_bot.py imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ trading_bot.py import failed: {e}")
    
    # Test 5: Import intelligence modules
    total_tests += 1
    try:
        from regime_classifier import MarketRegimeClassifier
        from manipulation_detector import ManipulationDetector
        from crisis_detector import CrisisDetector
        from regime_risk_manager import RegimeRiskCalculator
        print("✅ All intelligence modules imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Intelligence modules import failed: {e}")
    
    # Test 6: Configuration validation
    total_tests += 1
    try:
        # Test if config has required attributes
        if hasattr(config, 'TOTAL_CAPITAL') and hasattr(config, 'TARGET_SYMBOLS'):
            print("✅ Configuration validation passed")
            tests_passed += 1
        else:
            print("❌ Configuration validation failed: Missing required attributes")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
    
    # Test 7: Basic API connectivity (if APIs are configured)
    total_tests += 1
    try:
        # Try to create DataManager and test basic functionality
        dm = DataManager()
        # Don't actually call APIs in quick test, just verify object creation
        if hasattr(dm, 'get_market_data'):
            print("✅ DataManager creation successful")
            tests_passed += 1
        else:
            print("❌ DataManager missing required methods")
    except Exception as e:
        print(f"❌ DataManager creation failed: {e}")
    
    # Test 8: Basic bot instantiation
    total_tests += 1
    try:
        # Try to create TradingBot in paper mode
        bot = TradingBot(paper_trading=True)
        if hasattr(bot, 'run_trading_cycle'):
            print("✅ TradingBot instantiation successful")
            tests_passed += 1
        else:
            print("❌ TradingBot missing required methods")
    except Exception as e:
        print(f"❌ TradingBot instantiation failed: {e}")
    
    # Results
    success_rate = (tests_passed / total_tests) * 100
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 QUICK TEST PASSED - System ready!")
        print("✅ You can proceed to the next phase")
        return True
    elif success_rate >= 50:
        print("⚠️ PARTIAL SUCCESS - Some issues need fixing")
        print("🔧 Fix the failed tests above before proceeding")
        return False
    else:
        print("❌ QUICK TEST FAILED - Major issues detected")
        print("🚨 Critical problems must be resolved before proceeding")
        return False

def test_file_existence():
    """Check if required files exist"""
    print("\n🔍 File Existence Check:")
    required_files = [
        'config.py',
        'data_manager.py', 
        'signal_evaluator.py',
        'trading_bot.py',
        'regime_classifier.py',
        'manipulation_detector.py',
        'crisis_detector.py',
        'regime_risk_manager.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - FILE MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n🚨 MISSING FILES: {', '.join(missing_files)}")
        print("Please ensure all required files are present before running tests")
        return False
    else:
        print("\n✅ All required files present")
        return True

if __name__ == "__main__":
    print("🚀 CRYPTO TRADING BOT - QUICK VALIDATION")
    print("="*60)
    
    # First check if files exist
    if not test_file_existence():
        print("\n❌ Cannot proceed - missing required files")
        sys.exit(1)
    
    # Run the actual tests
    success = quick_test()
    
    print("\n" + "="*60)
    if success:
        print("🎯 VALIDATION COMPLETE - PROCEED TO NEXT STEP")
    else:
        print("🛑 VALIDATION INCOMPLETE - RESOLVE ISSUES FIRST")
    
    sys.exit(0 if success else 1)