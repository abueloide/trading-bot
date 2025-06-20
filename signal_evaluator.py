#!/usr/bin/env python3
"""
Signal Evaluator - Complete Enhanced Technical Analysis with HERD-001 Integration
Enhanced version with manipulation protection, regime awareness, and complete testing compatibility
Version: 2.0 - Complete Integration Ready
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import time

# Import crowding detection
try:
    from crowding_detector import CrowdingDetector
    CROWDING_AVAILABLE = True
except ImportError:
    CROWDING_AVAILABLE = False
    print("Warning: Crowding detection not available")

# Import manipulation analysis
try:
    from manipulation_detector import analyze_manipulation_risk
    MANIPULATION_AVAILABLE = True
except ImportError:
    MANIPULATION_AVAILABLE = False
    print("Warning: Manipulation detection not available")

# Import config for crowding
try:
    from config import ENABLE_CROWDING_DETECTION, CROWDING_CONFIG, FEATURE_FLAGS
    from config import RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD
    from config import MACD_FAST, MACD_SLOW, MACD_SIGNAL
    from config import BOLLINGER_PERIOD, BOLLINGER_STD
    from config import EMA_SHORT, EMA_LONG
    from config import MIN_SIGNAL_STRENGTH, MIN_CONFIDENCE_LEVEL
    from config import STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE
    from config import MAX_POSITION_SIZE, TOTAL_CAPITAL
except ImportError:
    ENABLE_CROWDING_DETECTION = False
    CROWDING_CONFIG = {}
    FEATURE_FLAGS = {}
    # Default values
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
    MIN_SIGNAL_STRENGTH = 0.6
    MIN_CONFIDENCE_LEVEL = 0.65
    STOP_LOSS_PERCENTAGE = 0.02
    TAKE_PROFIT_PERCENTAGE = 0.04
    MAX_POSITION_SIZE = 0.1
    TOTAL_CAPITAL = 1000

logger = logging.getLogger(__name__)

# =============================================================================
# TECHNICAL INDICATORS CALCULATION
# =============================================================================

class TechnicalIndicators:
    """Technical indicators calculation with optimized performance"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = RSI_PERIOD) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI
            
            prices_array = np.array(prices)
            deltas = np.diff(prices_array)
            
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
            
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return 50.0

    @staticmethod
    def calculate_macd(prices: List[float], fast: int = MACD_FAST, 
                      slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if len(prices) < slow + signal:
                return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
            
            prices_array = np.array(prices)
            
            # Calculate EMAs
            ema_fast = TechnicalIndicators._calculate_ema(prices_array, fast)
            ema_slow = TechnicalIndicators._calculate_ema(prices_array, slow)
            
            # MACD line
            macd_line = ema_fast[-1] - ema_slow[-1]
            
            # Signal line (EMA of MACD)
            if len(prices) >= slow + signal:
                macd_history = ema_fast[slow-1:] - ema_slow[slow-1:]
                signal_line = TechnicalIndicators._calculate_ema(macd_history, signal)[-1]
            else:
                signal_line = 0.0
            
            # Histogram
            histogram = macd_line - signal_line
            
            return {
                'macd': float(macd_line),
                'signal': float(signal_line),
                'histogram': float(histogram)
            }
            
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}

    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = BOLLINGER_PERIOD, 
                                 std_dev: float = BOLLINGER_STD) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                current_price = prices[-1] if prices else 0
                return {
                    'upper': current_price * 1.02,
                    'middle': current_price,
                    'lower': current_price * 0.98,
                    'position': 0.5
                }
            
            prices_array = np.array(prices[-period:])
            sma = np.mean(prices_array)
            std = np.std(prices_array)
            
            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)
            
            current_price = prices[-1]
            
            # Calculate position within bands (0 = lower band, 1 = upper band)
            if upper_band != lower_band:
                position = (current_price - lower_band) / (upper_band - lower_band)
                position = max(0, min(1, position))
            else:
                position = 0.5
            
            return {
                'upper': float(upper_band),
                'middle': float(sma),
                'lower': float(lower_band),
                'position': float(position)
            }
            
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            current_price = prices[-1] if prices else 0
            return {
                'upper': current_price * 1.02,
                'middle': current_price,
                'lower': current_price * 0.98,
                'position': 0.5
            }

    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < period:
                return np.mean(prices) if prices else 0.0
            
            prices_array = np.array(prices)
            ema_values = TechnicalIndicators._calculate_ema(prices_array, period)
            return float(ema_values[-1])
            
        except Exception as e:
            logger.error(f"EMA calculation failed: {e}")
            return np.mean(prices) if prices else 0.0

    @staticmethod
    def _calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Internal EMA calculation"""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema

    @staticmethod
    def calculate_volatility(prices: List[float], period: int = 20) -> float:
        """Calculate price volatility"""
        try:
            if len(prices) < 2:
                return 0.02  # Default volatility
            
            prices_array = np.array(prices[-period:])
            returns = np.diff(prices_array) / prices_array[:-1]
            volatility = np.std(returns)
            
            return float(volatility)
            
        except Exception as e:
            logger.error(f"Volatility calculation failed: {e}")
            return 0.02

# =============================================================================
# MARKET REGIME DETECTOR
# =============================================================================

class MarketRegimeDetector:
    """Detect market regimes for enhanced signal evaluation"""
    
    def __init__(self):
        self.volatility_threshold_low = 0.02
        self.volatility_threshold_high = 0.05
        self.trend_threshold = 0.15
    
    def detect_regime(self, market_data: Dict) -> Dict[str, Any]:
        """Detect current market regime"""
        try:
            price_history = market_data.get('price_history', [])
            volume_history = market_data.get('volume_history', [])
            
            if len(price_history) < 50:
                return {
                    'regime': 'UNKNOWN',
                    'confidence': 0.5,
                    'volatility': 0.02,
                    'trend_strength': 0.0,
                    'volume_trend': 'NEUTRAL'
                }
            
            # Calculate volatility
            volatility = TechnicalIndicators.calculate_volatility(price_history)
            
            # Calculate trend strength
            prices = np.array(price_history[-50:])
            trend_slope = np.polyfit(range(len(prices)), prices, 1)[0]
            trend_strength = abs(trend_slope) / np.mean(prices)
            
            # Determine regime
            regime = self._classify_regime(volatility, trend_strength, trend_slope)
            
            # Calculate confidence
            confidence = self._calculate_regime_confidence(volatility, trend_strength)
            
            # Volume trend
            volume_trend = self._analyze_volume_trend(volume_history)
            
            return {
                'regime': regime,
                'confidence': confidence,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'volume_trend': volume_trend,
                'trend_direction': 'UP' if trend_slope > 0 else 'DOWN'
            }
            
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return {
                'regime': 'UNKNOWN',
                'confidence': 0.5,
                'volatility': 0.02,
                'trend_strength': 0.0,
                'volume_trend': 'NEUTRAL'
            }
    
    def _classify_regime(self, volatility: float, trend_strength: float, trend_slope: float) -> str:
        """Classify market regime"""
        if volatility > self.volatility_threshold_high:
            if trend_strength > self.trend_threshold:
                return 'HIGH_VOLATILITY_TRENDING'
            else:
                return 'HIGH_VOLATILITY_RANGING'
        elif volatility < self.volatility_threshold_low:
            if trend_strength > self.trend_threshold:
                return 'LOW_VOLATILITY_TRENDING'
            else:
                return 'LOW_VOLATILITY_RANGING'
        else:
            if trend_strength > self.trend_threshold:
                return 'NORMAL_VOLATILITY_TRENDING'
            else:
                return 'NORMAL_VOLATILITY_RANGING'
    
    def _calculate_regime_confidence(self, volatility: float, trend_strength: float) -> float:
        """Calculate confidence in regime classification"""
        # Higher confidence for more extreme values
        volatility_confidence = min(abs(volatility - 0.035) / 0.035, 1.0)
        trend_confidence = min(trend_strength / 0.1, 1.0)
        
        return (volatility_confidence + trend_confidence) / 2
    
    def _analyze_volume_trend(self, volume_history: List[float]) -> str:
        """Analyze volume trend"""
        try:
            if len(volume_history) < 20:
                return 'NEUTRAL'
            
            recent_volume = np.mean(volume_history[-10:])
            historical_volume = np.mean(volume_history[-30:-10])
            
            if recent_volume > historical_volume * 1.2:
                return 'INCREASING'
            elif recent_volume < historical_volume * 0.8:
                return 'DECREASING'
            else:
                return 'NEUTRAL'
                
        except Exception:
            return 'NEUTRAL'

# =============================================================================
# ENHANCED SIGNAL EVALUATOR
# =============================================================================

class SignalEvaluator:
    """Enhanced signal evaluation engine with HERD-001 integration and protection layers"""
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
        self.regime_detector = MarketRegimeDetector()
        
        # Initialize crowding detector if available
        self.crowding_detector = None
        if ENABLE_CROWDING_DETECTION and CROWDING_AVAILABLE:
            try:
                self.crowding_detector = CrowdingDetector()
                logger.info("HERD-001 crowding detector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize crowding detector: {e}")
        
        # Enhanced configuration
        self.regime_thresholds = {
            'BULL': {
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'confidence_minimum': 0.8,
                'volume_confirmation': 2.0,
                'momentum_threshold': 0.7
            },
            'BEAR': {
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'confidence_minimum': 0.8,
                'volume_confirmation': 2.0,
                'momentum_threshold': 0.7
            },
            'SIDEWAYS': {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'confidence_minimum': 0.6,
                'volume_confirmation': 1.3,
                'momentum_threshold': 0.4
            },
            'UNKNOWN': {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'confidence_minimum': 0.7,
                'volume_confirmation': 1.5,
                'momentum_threshold': 0.5
            }
        }
        
        # Signal quality weights
        self.signal_weights = {
            'technical_score': 0.35,
            'volume_confirmation': 0.25,
            'regime_alignment': 0.20,
            'manipulation_safety': 0.15,
            'risk_reward': 0.05
        }
        
        # Signal history for performance tracking
        self.signal_history = []
        self.performance_cache = {}
        
        self.signal_cache = {}
        self.cache_ttl = 60  # 1 minute cache
        
        logger.info("📈 Enhanced SignalEvaluator initialized")

    def evaluate_signal(self, symbol: str, market_data: Dict, 
                       technical_indicators: Optional[Dict] = None) -> Dict[str, Any]:
        """Main signal evaluation function with integrated crowding analysis"""
        try:
            start_time = time.time()
            
            # Extract market data
            price_history = market_data.get('price_history', [])
            volume_history = market_data.get('volume_history', [])
            
            if not price_history or len(price_history) < 20:
                return self._create_neutral_signal(symbol, "Insufficient price data")
            
            # Calculate technical indicators
            tech_indicators = self._calculate_all_indicators(price_history, volume_history)
            
            # Detect market regime
            regime_info = self.regime_detector.detect_regime(market_data)
            
            # Generate base trading signal
            base_signal = self._generate_base_signal(symbol, tech_indicators, regime_info)
            
            # Evaluate signal strength and confidence
            signal_strength = self._calculate_signal_strength(tech_indicators, regime_info)
            confidence = self._calculate_confidence(tech_indicators, regime_info, market_data)
            
            # Determine position sizing
            position_size = self._calculate_position_size(signal_strength, confidence)
            
            # Calculate entry, stop loss, and take profit
            current_price = price_history[-1]
            entry_price, stop_loss, take_profit = self._calculate_price_levels(
                current_price, base_signal['action'], tech_indicators
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(tech_indicators, regime_info, base_signal)
            
            # Create base signal
            signal = {
                'symbol': symbol,
                'action': base_signal['action'],
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'reasoning': reasoning,
                'timestamp': datetime.now(),
                'signal_strength': signal_strength,
                'technical_indicators': tech_indicators,
                'market_regime': regime_info,
                'analysis_time_ms': round((time.time() - start_time) * 1000, 2)
            }
            
            # Add crowding analysis if enabled
            if ENABLE_CROWDING_DETECTION and CROWDING_AVAILABLE and self.crowding_detector:
                try:
                    crowding_analysis = self.crowding_detector.analyze_trade_crowding(
                        symbol=symbol,
                        direction=base_signal['action'],
                        size_usd=position_size * entry_price,
                        market_data=market_data
                    )
                    signal['crowding_analysis'] = {
                        'trade_crowding_score': crowding_analysis.trade_crowding_score,
                        'order_book_clustering': crowding_analysis.order_book_clustering,
                        'directional_bias': crowding_analysis.directional_bias,
                        'size_concentration': crowding_analysis.size_concentration,
                        'timing_correlation': crowding_analysis.timing_correlation,
                        'components': crowding_analysis.components,
                        'timestamp': crowding_analysis.timestamp
                    }
                    
                    # Log crowding analysis
                    logger.info(f"Crowding analysis for {symbol}: {crowding_analysis.trade_crowding_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Crowding analysis failed for {symbol}: {e}")
                    signal['crowding_analysis'] = None
            else:
                signal['crowding_analysis'] = None
            
            # Final signal validation
            if signal_strength < MIN_SIGNAL_STRENGTH or confidence < MIN_CONFIDENCE_LEVEL:
                signal['action'] = 'HOLD'
                signal['reasoning'] += f" Signal strength ({signal_strength:.2f}) or confidence ({confidence:.2f}) below threshold."
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal evaluation failed for {symbol}: {e}")
            return self._create_neutral_signal(symbol, f"Evaluation error: {str(e)}")

    def evaluate_signal_with_protection(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Enhanced signal evaluation with manipulation protection and regime awareness"""
        try:
            # 1. Generate base signal using original method
            base_signal = self.evaluate_signal(symbol, market_data)
            if not base_signal or base_signal['action'] == 'HOLD':
                return None
            
            # 2. Analyze manipulation risk
            manipulation_analysis = self._get_manipulation_analysis(market_data, symbol)
            
            # 3. Apply manipulation filtering
            filtered_signal = self._apply_manipulation_filter(base_signal, manipulation_analysis)
            if not filtered_signal:
                logger.warning(f"🛡️ Signal filtered due to manipulation risk: {symbol}")
                return None
            
            # 4. Apply regime-based adjustments
            regime_adjusted_signal = self._apply_regime_adjustments(filtered_signal, market_data)
            
            # 5. Enhanced signal scoring
            final_signal = self._calculate_enhanced_signal_score(regime_adjusted_signal, market_data)
            
            # 6. Add protection metadata
            final_signal = self._add_protection_metadata(final_signal, manipulation_analysis, market_data)
            
            # 7. Final validation
            if self._validate_enhanced_signal(final_signal):
                logger.info(f"📈 Enhanced signal generated for {symbol}: {final_signal['action']} (confidence: {final_signal['confidence']:.2f})")
                self._update_signal_history(final_signal)
                return final_signal
            else:
                logger.info(f"📈 Signal rejected after final validation: {symbol}")
                return None
            
        except Exception as e:
            logger.error(f"Error in enhanced signal evaluation for {symbol}: {e}")
            # Fallback to basic signal evaluation
            return self.evaluate_signal(symbol, market_data)

    def analyze_signal_quality(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """Analyze signal quality with multiple factors"""
        try:
            # Generate signal for analysis
            signal = self.evaluate_signal(symbol, market_data)
            if not signal:
                return {
                    'overall_assessment': {'recommendation': 'NO_SIGNAL'},
                    'quality_score': 0.0,
                    'risk_factors': ['No signal generated'],
                    'timestamp': datetime.now()
                }
            
            # Analyze quality factors
            quality_factors = {
                'technical_strength': signal.get('signal_strength', 0),
                'confidence_level': signal.get('confidence', 0),
                'regime_alignment': self._assess_regime_alignment(signal),
                'volume_confirmation': self._assess_volume_confirmation(signal),
                'risk_reward_ratio': self._calculate_risk_reward_ratio(signal)
            }
            
            # Calculate overall quality score
            quality_score = sum(quality_factors.values()) / len(quality_factors)
            
            # Generate assessment
            if quality_score > 0.8:
                recommendation = 'STRONG_SIGNAL'
            elif quality_score > 0.6:
                recommendation = 'MODERATE_SIGNAL'
            elif quality_score > 0.4:
                recommendation = 'WEAK_SIGNAL'
            else:
                recommendation = 'POOR_SIGNAL'
            
            # Identify risk factors
            risk_factors = []
            if quality_factors['technical_strength'] < 0.6:
                risk_factors.append('Low technical strength')
            if quality_factors['confidence_level'] < 0.7:
                risk_factors.append('Low confidence')
            if quality_factors['volume_confirmation'] < 0.5:
                risk_factors.append('Poor volume confirmation')
            
            return {
                'overall_assessment': {
                    'recommendation': recommendation,
                    'quality_score': quality_score
                },
                'quality_factors': quality_factors,
                'risk_factors': risk_factors,
                'signal_details': signal,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Signal quality analysis failed for {symbol}: {e}")
            return {
                'overall_assessment': {'recommendation': 'ERROR'},
                'quality_score': 0.0,
                'risk_factors': [f'Analysis error: {str(e)}'],
                'timestamp': datetime.now()
            }

    def get_signal_performance_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get signal performance metrics for recent period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_signals = [s for s in self.signal_history if s.get('timestamp', datetime.min) > cutoff_date]
            
            if not recent_signals:
                return {
                    'status': 'no_data',
                    'period_days': days,
                    'total_signals': 0,
                    'timestamp': datetime.now()
                }
            
            # Calculate metrics
            total_signals = len(recent_signals)
            buy_signals = len([s for s in recent_signals if s.get('action') == 'BUY'])
            sell_signals = len([s for s in recent_signals if s.get('action') == 'SELL'])
            hold_signals = len([s for s in recent_signals if s.get('action') == 'HOLD'])
            
            avg_confidence = np.mean([s.get('confidence', 0) for s in recent_signals])
            avg_strength = np.mean([s.get('signal_strength', 0) for s in recent_signals])
            
            # Performance by action
            action_distribution = {
                'BUY': buy_signals / total_signals,
                'SELL': sell_signals / total_signals,
                'HOLD': hold_signals / total_signals
            }
            
            return {
                'status': 'success',
                'period_days': days,
                'total_signals': total_signals,
                'action_distribution': action_distribution,
                'avg_confidence': avg_confidence,
                'avg_signal_strength': avg_strength,
                'quality_trend': self._calculate_quality_trend(recent_signals),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now()
            }

    def _calculate_all_indicators(self, price_history: List[float], 
                                 volume_history: List[float]) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        try:
            current_price = price_history[-1]
            
            # Core indicators
            rsi = self.technical_indicators.calculate_rsi(price_history)
            macd_data = self.technical_indicators.calculate_macd(price_history)
            bb_data = self.technical_indicators.calculate_bollinger_bands(price_history)
            
            # EMAs
            ema_short = self.technical_indicators.calculate_ema(price_history, EMA_SHORT)
            ema_long = self.technical_indicators.calculate_ema(price_history, EMA_LONG)
            
            # Volume analysis
            volume_ratio = 1.0
            if len(volume_history) >= 20:
                recent_volume = np.mean(volume_history[-5:])
                avg_volume = np.mean(volume_history[-20:])
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volatility
            volatility = self.technical_indicators.calculate_volatility(price_history)
            
            return {
                'current_price': current_price,
                'rsi': rsi,
                'macd': macd_data['macd'],
                'signal': macd_data['signal'],
                'histogram': macd_data['histogram'],
                'bb_upper': bb_data['upper'],
                'bb_middle': bb_data['middle'],
                'bb_lower': bb_data['lower'],
                'bb_position': bb_data['position'],
                'ema_short': ema_short,
                'ema_long': ema_long,
                'volume_ratio': volume_ratio,
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"Technical indicators calculation failed: {e}")
            return {
                'current_price': price_history[-1] if price_history else 0,
                'rsi': 50.0,
                'macd': 0.0,
                'signal': 0.0,
                'histogram': 0.0,
                'bb_upper': 0.0,
                'bb_middle': 0.0,
                'bb_lower': 0.0,
                'bb_position': 0.5,
                'ema_short': 0.0,
                'ema_long': 0.0,
                'volume_ratio': 1.0,
                'volatility': 0.02
            }

    def _get_manipulation_analysis(self, market_data: Dict, symbol: str) -> Dict:
        """Get manipulation analysis from market data or calculate if missing"""
        manipulation_data = market_data.get('manipulation')
        
        if manipulation_data:
            return manipulation_data
        
        # Fallback: calculate manipulation analysis if module available
        if MANIPULATION_AVAILABLE:
            try:
                manipulation_analysis = analyze_manipulation_risk(symbol, market_data)
                return {
                    'risk_level': manipulation_analysis.risk_level,
                    'manipulation_risk': manipulation_analysis.manipulation_risk,
                    'confidence': manipulation_analysis.confidence,
                    'recommendation': manipulation_analysis.recommendation,
                    'manipulation_type': manipulation_analysis.manipulation_type
                }
            except Exception as e:
                logger.warning(f"Manipulation analysis failed: {e}")
        
        # Default safe values
        return {
            'risk_level': 'UNKNOWN',
            'manipulation_risk': 0.5,
            'confidence': 0.5,
            'recommendation': 'PROCEED_WITH_CAUTION',
            'manipulation_type': 'NONE'
        }

    def _apply_manipulation_filter(self, signal: Dict, manipulation_analysis: Dict) -> Optional[Dict]:
        """Apply manipulation filtering to signal"""
        try:
            risk_level = manipulation_analysis.get('risk_level', 'UNKNOWN')
            manipulation_risk = manipulation_analysis.get('manipulation_risk', 0.5)
            
            # Filter out high-risk signals
            if risk_level == 'HIGH' or manipulation_risk > 0.8:
                logger.warning(f"Signal filtered: HIGH manipulation risk ({manipulation_risk:.2f})")
                return None
            
            # Adjust confidence for medium risk
            if risk_level == 'MEDIUM' or manipulation_risk > 0.6:
                signal['confidence'] *= 0.8
                signal['reasoning'] += " (Adjusted for manipulation risk)"
            
            return signal
            
        except Exception as e:
            logger.error(f"Manipulation filtering failed: {e}")
            return signal

    def _apply_regime_adjustments(self, signal: Dict, market_data: Dict) -> Dict:
        """Apply regime-based adjustments to signal"""
        try:
            regime_info = signal.get('market_regime', {})
            regime_type = regime_info.get('regime', 'UNKNOWN')
            
            # Get regime-specific thresholds
            regime_config = self.regime_thresholds.get(regime_type, self.regime_thresholds['UNKNOWN'])
            
            # Adjust confidence based on regime
            regime_confidence = regime_info.get('confidence', 0.5)
            if regime_confidence > 0.8:
                signal['confidence'] *= 1.1  # Boost for high regime confidence
            elif regime_confidence < 0.5:
                signal['confidence'] *= 0.9  # Reduce for low regime confidence
            
            # Adjust position size based on volatility regime
            volatility = regime_info.get('volatility', 0.02)
            if volatility > 0.05:  # High volatility
                signal['position_size'] *= 0.8
            elif volatility < 0.01:  # Low volatility
                signal['position_size'] *= 1.2
            
            # Add regime context
            signal['regime_context'] = {
                'regime_type': regime_type,
                'regime_confidence': regime_confidence,
                'volatility_adjustment': volatility,
                'applied_adjustments': ['confidence', 'position_size']
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Regime adjustment failed: {e}")
            return signal

    def _calculate_enhanced_signal_score(self, signal: Dict, market_data: Dict) -> Dict:
        """Calculate enhanced signal score with multiple factors"""
        try:
            # Base scores
            technical_score = signal.get('signal_strength', 0)
            confidence_score = signal.get('confidence', 0)
            
            # Volume confirmation score
            volume_ratio = signal.get('technical_indicators', {}).get('volume_ratio', 1.0)
            volume_score = min(volume_ratio / 2.0, 1.0)
            
            # Regime alignment score
            regime_score = self._calculate_regime_alignment_score(signal)
            
            # Manipulation safety score
            crowding_analysis = signal.get('crowding_analysis')
            if crowding_analysis:
                # Lower crowding = higher safety score
                crowding_score = crowding_analysis.get('trade_crowding_score', 0.5)
                safety_score = 1.0 - crowding_score
            else:
                safety_score = 0.7  # Default when no crowding analysis
            
            # Risk-reward score
            risk_reward_score = self._calculate_risk_reward_score(signal)
            
            # Calculate weighted score
            weighted_score = (
                technical_score * self.signal_weights['technical_score'] +
                volume_score * self.signal_weights['volume_confirmation'] +
                regime_score * self.signal_weights['regime_alignment'] +
                safety_score * self.signal_weights['manipulation_safety'] +
                risk_reward_score * self.signal_weights['risk_reward']
            )
            
            # Update signal with enhanced scoring
            signal['enhanced_score'] = weighted_score
            signal['score_components'] = {
                'technical_score': technical_score,
                'volume_score': volume_score,
                'regime_score': regime_score,
                'safety_score': safety_score,
                'risk_reward_score': risk_reward_score
            }
            
            # Adjust final confidence based on enhanced score
            signal['confidence'] = min(weighted_score, 1.0)
            
            return signal
            
        except Exception as e:
            logger.error(f"Enhanced signal scoring failed: {e}")
            return signal

    def _add_protection_metadata(self, signal: Dict, manipulation_analysis: Dict, market_data: Dict) -> Dict:
        """Add protection metadata to signal"""
        try:
            signal['protection_metadata'] = {
                'manipulation_analysis': manipulation_analysis,
                'regime_protection': signal.get('regime_context', {}),
                'crowding_protection': signal.get('crowding_analysis', {}),
                'quality_assessment': {
                    'enhanced_score': signal.get('enhanced_score', 0),
                    'score_components': signal.get('score_components', {}),
                    'protection_level': self._determine_protection_level(signal)
                },
                'timestamp': datetime.now()
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Protection metadata addition failed: {e}")
            return signal

    def _validate_enhanced_signal(self, signal: Dict) -> bool:
        """Validate enhanced signal before returning"""
        try:
            # Basic validation
            if not self.validate_signal(signal):
                return False
            
            # Enhanced validation
            enhanced_score = signal.get('enhanced_score', 0)
            if enhanced_score < 0.5:  # Minimum enhanced score threshold
                logger.info(f"Signal rejected: Low enhanced score ({enhanced_score:.2f})")
                return False
            
            # Check protection metadata
            protection_metadata = signal.get('protection_metadata', {})
            if not protection_metadata:
                logger.warning("Signal missing protection metadata")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced signal validation failed: {e}")
            return False

    def _update_signal_history(self, signal: Dict) -> None:
        """Update signal history for performance tracking"""
        try:
            # Add to history
            self.signal_history.append(signal.copy())
            
            # Limit history size (keep last 1000 signals)
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
        except Exception as e:
            logger.error(f"Signal history update failed: {e}")

    def _generate_base_signal(self, symbol: str, indicators: Dict[str, Any], 
                             regime_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate base trading signal"""
        try:
            signals = []
            
            # RSI signals
            if indicators['rsi'] < RSI_OVERSOLD:
                signals.append(('BUY', 0.7, 'RSI oversold'))
            elif indicators['rsi'] > RSI_OVERBOUGHT:
                signals.append(('SELL', 0.7, 'RSI overbought'))
            
            # MACD signals
            if indicators['macd'] > indicators['signal'] and indicators['histogram'] > 0:
                signals.append(('BUY', 0.6, 'MACD bullish crossover'))
            elif indicators['macd'] < indicators['signal'] and indicators['histogram'] < 0:
                signals.append(('SELL', 0.6, 'MACD bearish crossover'))
            
            # EMA signals
            if indicators['ema_short'] > indicators['ema_long']:
                signals.append(('BUY', 0.5, 'Short EMA above long EMA'))
            else:
                signals.append(('SELL', 0.5, 'Short EMA below long EMA'))
            
            # Bollinger Bands signals
            if indicators['bb_position'] < 0.1:
                signals.append(('BUY', 0.6, 'Price near lower Bollinger Band'))
            elif indicators['bb_position'] > 0.9:
                signals.append(('SELL', 0.6, 'Price near upper Bollinger Band'))
            
            # Volume confirmation
            if indicators['volume_ratio'] > 1.5:
                # High volume confirms signals
                signals = [(action, strength * 1.1, reason + ' with high volume') 
                          for action, strength, reason in signals]
            
            # Aggregate signals
            if not signals:
                return {'action': 'HOLD', 'reason': 'No clear signals'}
            
            # Calculate weighted signal
            buy_weight = sum(strength for action, strength, _ in signals if action == 'BUY')
            sell_weight = sum(strength for action, strength, _ in signals if action == 'SELL')
            
            if buy_weight > sell_weight * 1.1:
                action = 'BUY'
                primary_reasons = [reason for act, _, reason in signals if act == 'BUY']
            elif sell_weight > buy_weight * 1.1:
                action = 'SELL'
                primary_reasons = [reason for act, _, reason in signals if act == 'SELL']
            else:
                action = 'HOLD'
                primary_reasons = ['Mixed signals']
            
            return {
                'action': action,
                'reason': '; '.join(primary_reasons[:3])  # Top 3 reasons
            }
            
        except Exception as e:
            logger.error(f"Base signal generation failed: {e}")
            return {'action': 'HOLD', 'reason': 'Signal generation error'}

    def _calculate_signal_strength(self, indicators: Dict[str, Any], 
                                  regime_info: Dict[str, Any]) -> float:
        """Calculate overall signal strength"""
        try:
            strength_factors = []
            
            # RSI strength
            rsi = indicators['rsi']
            if rsi < 20 or rsi > 80:
                strength_factors.append(0.9)  # Strong signal
            elif rsi < 30 or rsi > 70:
                strength_factors.append(0.7)  # Moderate signal
            else:
                strength_factors.append(0.3)  # Weak signal
            
            # MACD strength
            macd_strength = min(abs(indicators['histogram']) / 10, 0.8)
            strength_factors.append(macd_strength)
            
            # Volatility adjustment
            volatility = regime_info['volatility']
            if 0.01 < volatility < 0.04:  # Optimal volatility range
                vol_factor = 0.8
            elif volatility > 0.06:  # High volatility reduces reliability
                vol_factor = 0.5
            else:  # Low volatility
                vol_factor = 0.6
            
            strength_factors.append(vol_factor)
            
            # Volume confirmation
            volume_factor = min(indicators['volume_ratio'] / 2, 0.8)
            strength_factors.append(volume_factor)
            
            # Market regime adjustment
            regime_factor = regime_info['confidence']
            strength_factors.append(regime_factor)
            
            return np.mean(strength_factors)
            
        except Exception as e:
            logger.error(f"Signal strength calculation failed: {e}")
            return 0.5

    def _calculate_confidence(self, indicators: Dict[str, Any], 
                             regime_info: Dict[str, Any], market_data: Dict) -> float:
        """Calculate signal confidence"""
        try:
            confidence_factors = []
            
            # Data quality
            data_quality = market_data.get('data_quality', 0.8)
            confidence_factors.append(data_quality)
            
            # Indicator alignment
            alignment_score = self._calculate_indicator_alignment(indicators)
            confidence_factors.append(alignment_score)
            
            # Market regime confidence
            confidence_factors.append(regime_info['confidence'])
            
            # Historical price stability
            price_history = market_data.get('price_history', [])
            if len(price_history) > 50:
                recent_volatility = np.std(price_history[-20:]) / np.mean(price_history[-20:])
                historical_volatility = np.std(price_history[-50:-20]) / np.mean(price_history[-50:-20])
                
                if historical_volatility > 0:
                    volatility_ratio = recent_volatility / historical_volatility
                    # Stable volatility increases confidence
                    stability_factor = max(0.3, 1.0 - abs(volatility_ratio - 1.0))
                    confidence_factors.append(stability_factor)
            
            return np.mean(confidence_factors)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    def _calculate_indicator_alignment(self, indicators: Dict[str, Any]) -> float:
        """Calculate how well indicators align"""
        try:
            signals = []
            
            # RSI signal direction
            if indicators['rsi'] > 50:
                signals.append(1)
            else:
                signals.append(-1)
            
            # MACD signal direction
            if indicators['macd'] > indicators['signal']:
                signals.append(1)
            else:
                signals.append(-1)
            
            # EMA signal direction
            if indicators['ema_short'] > indicators['ema_long']:
                signals.append(1)
            else:
                signals.append(-1)
            
            # Price vs Bollinger middle
            if indicators['current_price'] > indicators['bb_middle']:
                signals.append(1)
            else:
                signals.append(-1)
            
            # Calculate alignment (how many agree)
            positive_signals = sum(1 for s in signals if s > 0)
            total_signals = len(signals)
            
            alignment = max(positive_signals, total_signals - positive_signals) / total_signals
            return alignment
            
        except Exception as e:
            logger.error(f"Indicator alignment calculation failed: {e}")
            return 0.5

    def _calculate_position_size(self, signal_strength: float, confidence: float) -> float:
        """Calculate position size based on signal quality"""
        try:
            # Base position size
            base_size = MAX_POSITION_SIZE * TOTAL_CAPITAL
            
            # Adjust based on signal strength and confidence
            quality_factor = (signal_strength + confidence) / 2
            
            # Size scaling
            if quality_factor > 0.8:
                size_factor = 1.0
            elif quality_factor > 0.6:
                size_factor = 0.8
            elif quality_factor > 0.4:
                size_factor = 0.6
            else:
                size_factor = 0.4
            
            return base_size * size_factor
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 1000.0  # Default position size

    def _calculate_price_levels(self, current_price: float, action: str, 
                               indicators: Dict[str, Any]) -> Tuple[float, float, float]:
        """Calculate entry, stop loss, and take profit levels"""
        try:
            entry_price = current_price
            
            if action == 'BUY':
                stop_loss = current_price * (1 - STOP_LOSS_PERCENTAGE)
                take_profit = current_price * (1 + TAKE_PROFIT_PERCENTAGE)
                
                # Adjust based on Bollinger Bands
                if 'bb_lower' in indicators:
                    # Use Bollinger lower band as more conservative stop
                    bb_stop = indicators['bb_lower']
                    if bb_stop < stop_loss and bb_stop > current_price * 0.95:
                        stop_loss = bb_stop
                
            elif action == 'SELL':
                stop_loss = current_price * (1 + STOP_LOSS_PERCENTAGE)
                take_profit = current_price * (1 - TAKE_PROFIT_PERCENTAGE)
                
                # Adjust based on Bollinger Bands
                if 'bb_upper' in indicators:
                    # Use Bollinger upper band as more conservative stop
                    bb_stop = indicators['bb_upper']
                    if bb_stop > stop_loss and bb_stop < current_price * 1.05:
                        stop_loss = bb_stop
            
            else:  # HOLD
                stop_loss = current_price
                take_profit = current_price
            
            return entry_price, stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Price levels calculation failed: {e}")
            return current_price, current_price * 0.98, current_price * 1.02

    def _generate_reasoning(self, indicators: Dict[str, Any], 
                           regime_info: Dict[str, Any], 
                           base_signal: Dict[str, str]) -> str:
        """Generate human-readable reasoning for the signal"""
        try:
            reasoning_parts = []
            
            # Market regime context
            regime = regime_info['regime']
            reasoning_parts.append(f"Market regime: {regime}")
            
            # Primary signal reason
            reasoning_parts.append(base_signal['reason'])
            
            # Key indicator values
            rsi = indicators['rsi']
            if rsi < 30:
                reasoning_parts.append(f"RSI oversold at {rsi:.1f}")
            elif rsi > 70:
                reasoning_parts.append(f"RSI overbought at {rsi:.1f}")
            
            # MACD status
            if indicators['histogram'] > 5:
                reasoning_parts.append("Strong bullish MACD momentum")
            elif indicators['histogram'] < -5:
                reasoning_parts.append("Strong bearish MACD momentum")
            
            # Volume context
            volume_ratio = indicators.get('volume_ratio', 1.0)
            if volume_ratio > 2.0:
                reasoning_parts.append("High volume confirmation")
            elif volume_ratio < 0.5:
                reasoning_parts.append("Low volume warning")
            
            # Volatility context
            volatility = regime_info['volatility']
            if volatility > 0.05:
                reasoning_parts.append("High volatility environment")
            elif volatility < 0.01:
                reasoning_parts.append("Low volatility environment")
            
            return ". ".join(reasoning_parts[:5])  # Limit to 5 key points
            
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            return "Signal generated based on technical analysis"

    def _assess_regime_alignment(self, signal: Dict) -> float:
        """Assess how well signal aligns with market regime"""
        try:
            regime_info = signal.get('market_regime', {})
            regime_type = regime_info.get('regime', 'UNKNOWN')
            action = signal.get('action', 'HOLD')
            
            # Alignment scoring
            if 'TRENDING' in regime_type:
                if action in ['BUY', 'SELL']:
                    return 0.8  # Good alignment with trending market
                else:
                    return 0.4  # Poor alignment
            elif 'RANGING' in regime_type:
                if action == 'HOLD':
                    return 0.7  # Good for ranging markets
                else:
                    return 0.6  # Moderate alignment
            else:
                return 0.5  # Unknown regime
                
        except Exception:
            return 0.5

    def _assess_volume_confirmation(self, signal: Dict) -> float:
        """Assess volume confirmation for signal"""
        try:
            indicators = signal.get('technical_indicators', {})
            volume_ratio = indicators.get('volume_ratio', 1.0)
            
            # Volume confirmation scoring
            if volume_ratio > 2.0:
                return 1.0  # Excellent volume confirmation
            elif volume_ratio > 1.5:
                return 0.8  # Good confirmation
            elif volume_ratio > 1.0:
                return 0.6  # Moderate confirmation
            else:
                return 0.3  # Poor confirmation
                
        except Exception:
            return 0.5

    def _calculate_risk_reward_ratio(self, signal: Dict) -> float:
        """Calculate risk-reward ratio for signal"""
        try:
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)
            
            if entry_price == 0 or stop_loss == 0 or take_profit == 0:
                return 0.5
            
            # Calculate risk and reward
            if signal.get('action') == 'BUY':
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
            elif signal.get('action') == 'SELL':
                risk = abs(stop_loss - entry_price)
                reward = abs(entry_price - take_profit)
            else:
                return 0.5
            
            # Risk-reward ratio
            if risk > 0:
                ratio = reward / risk
                # Convert to 0-1 score (target ratio of 2:1 or better)
                return min(ratio / 2.0, 1.0)
            else:
                return 0.5
                
        except Exception:
            return 0.5

    def _calculate_regime_alignment_score(self, signal: Dict) -> float:
        """Calculate regime alignment score"""
        try:
            regime_context = signal.get('regime_context', {})
            regime_confidence = regime_context.get('regime_confidence', 0.5)
            
            # Base alignment score from regime confidence
            base_score = regime_confidence
            
            # Bonus for appropriate signal in regime
            regime_type = regime_context.get('regime_type', 'UNKNOWN')
            action = signal.get('action', 'HOLD')
            
            if 'TRENDING' in regime_type and action != 'HOLD':
                base_score *= 1.2  # Boost for trend-following
            elif 'RANGING' in regime_type and action == 'HOLD':
                base_score *= 1.1  # Boost for range-appropriate action
            
            return min(base_score, 1.0)
            
        except Exception:
            return 0.5

    def _calculate_risk_reward_score(self, signal: Dict) -> float:
        """Calculate risk-reward score"""
        return self._calculate_risk_reward_ratio(signal)

    def _determine_protection_level(self, signal: Dict) -> str:
        """Determine protection level for signal"""
        try:
            enhanced_score = signal.get('enhanced_score', 0)
            
            if enhanced_score > 0.8:
                return 'HIGH_PROTECTION'
            elif enhanced_score > 0.6:
                return 'MEDIUM_PROTECTION'
            elif enhanced_score > 0.4:
                return 'LOW_PROTECTION'
            else:
                return 'MINIMAL_PROTECTION'
                
        except Exception:
            return 'UNKNOWN_PROTECTION'

    def _calculate_quality_trend(self, recent_signals: List[Dict]) -> str:
        """Calculate quality trend from recent signals"""
        try:
            if len(recent_signals) < 5:
                return 'INSUFFICIENT_DATA'
            
            # Get quality scores over time
            scores = [s.get('enhanced_score', s.get('signal_strength', 0.5)) for s in recent_signals]
            
            # Calculate trend
            recent_avg = np.mean(scores[-5:])
            earlier_avg = np.mean(scores[:-5]) if len(scores) > 5 else np.mean(scores)
            
            if recent_avg > earlier_avg * 1.1:
                return 'IMPROVING'
            elif recent_avg < earlier_avg * 0.9:
                return 'DECLINING'
            else:
                return 'STABLE'
                
        except Exception:
            return 'UNKNOWN'

    def _create_neutral_signal(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Create neutral/hold signal"""
        return {
            'symbol': symbol,
            'action': 'HOLD',
            'confidence': 0.5,
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'position_size': 0,
            'reasoning': reason,
            'timestamp': datetime.now(),
            'signal_strength': 0.0,
            'technical_indicators': {},
            'market_regime': {'regime': 'UNKNOWN'},
            'crowding_analysis': None,
            'analysis_time_ms': 0
        }

    def get_signal_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get recent signal history for a symbol"""
        try:
            # Filter by symbol and sort by timestamp
            symbol_signals = [s for s in self.signal_history if s.get('symbol') == symbol]
            symbol_signals.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
            return symbol_signals[:limit]
        except Exception as e:
            logger.error(f"Failed to get signal history for {symbol}: {e}")
            return []

    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate signal structure and values"""
        try:
            required_fields = [
                'symbol', 'action', 'confidence', 'entry_price',
                'stop_loss', 'take_profit', 'position_size', 'reasoning'
            ]
            
            # Check required fields
            for field in required_fields:
                if field not in signal:
                    logger.error(f"Signal missing required field: {field}")
                    return False
            
            # Validate action
            if signal['action'] not in ['BUY', 'SELL', 'HOLD']:
                logger.error(f"Invalid signal action: {signal['action']}")
                return False
            
            # Validate confidence range
            if not 0 <= signal['confidence'] <= 1:
                logger.error(f"Invalid confidence value: {signal['confidence']}")
                return False
            
            # Validate price values
            if signal['action'] != 'HOLD':
                if signal['entry_price'] <= 0:
                    logger.error(f"Invalid entry price: {signal['entry_price']}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Signal validation failed: {e}")
            return False

    def get_market_summary(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """Get market summary for dashboard/monitoring"""
        try:
            price_history = market_data.get('price_history', [])
            volume_history = market_data.get('volume_history', [])
            
            if not price_history:
                return {'error': 'No price data available'}
            
            current_price = price_history[-1]
            
            # Calculate basic metrics
            price_change_24h = 0
            if len(price_history) >= 24:
                price_change_24h = (current_price / price_history[-24] - 1) * 100
            
            # Technical indicators summary
            indicators = self._calculate_all_indicators(price_history, volume_history)
            regime_info = self.regime_detector.detect_regime(market_data)
            
            # HERD-001 status
            herd_status = 'UNKNOWN'
            crowding_score = 0.0
            
            if ENABLE_CROWDING_DETECTION and CROWDING_AVAILABLE and self.crowding_detector:
                try:
                    herding_signal = self.crowding_detector.analyze_market_herding(market_data)
                    herd_status = herding_signal.market_regime
                    crowding_score = herding_signal.overall_herding_score
                except Exception as e:
                    logger.warning(f"HERD-001 analysis failed: {e}")
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_change_24h': price_change_24h,
                'volume_ratio': indicators.get('volume_ratio', 1.0),
                'rsi': indicators['rsi'],
                'macd_signal': 'BULLISH' if indicators['macd'] > indicators['signal'] else 'BEARISH',
                'bb_position': indicators['bb_position'],
                'market_regime': regime_info['regime'],
                'volatility': regime_info['volatility'],
                'herd_status': herd_status,
                'crowding_score': crowding_score,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Market summary generation failed: {e}")
            return {'error': str(e)}

# =============================================================================
# SIGNAL EVALUATION UTILITIES
# =============================================================================

def create_signal_evaluator() -> SignalEvaluator:
    """Factory function to create signal evaluator"""
    return SignalEvaluator()

def evaluate_multiple_symbols(symbols: List[str], market_data_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    """Evaluate signals for multiple symbols"""
    evaluator = create_signal_evaluator()
    results = {}
    
    for symbol in symbols:
        if symbol in market_data_dict:
            try:
                signal = evaluator.evaluate_signal(symbol, market_data_dict[symbol])
                results[symbol] = signal
            except Exception as e:
                logger.error(f"Failed to evaluate signal for {symbol}: {e}")
                results[symbol] = evaluator._create_neutral_signal(symbol, f"Evaluation failed: {e}")
        else:
            logger.warning(f"No market data available for {symbol}")
            results[symbol] = evaluator._create_neutral_signal(symbol, "No market data")
    
    return results

def get_signal_quality_score(signal: Dict[str, Any]) -> float:
    """Calculate overall signal quality score"""
    try:
        factors = []
        
        # Signal strength
        factors.append(signal.get('signal_strength', 0))
        
        # Confidence
        factors.append(signal.get('confidence', 0))
        
        # Market regime confidence
        regime_info = signal.get('market_regime', {})
        factors.append(regime_info.get('confidence', 0.5))
        
        # HERD-001 consideration (lower crowding = higher quality)
        crowding_analysis = signal.get('crowding_analysis')
        if crowding_analysis:
            crowding_score = crowding_analysis.get('trade_crowding_score', 0.5)
            # Invert crowding score (low crowding = high quality)
            factors.append(1.0 - crowding_score)
        else:
            factors.append(0.7)  # Default quality when no crowding analysis
        
        return np.mean(factors)
        
    except Exception as e:
        logger.error(f"Signal quality calculation failed: {e}")
        return 0.5

def get_best_trade(symbols: List[str]) -> Optional[Dict]:
    """Find best trade opportunity from list of symbols"""
    try:
        from data_manager import data_manager
        
        evaluator = create_signal_evaluator()
        candidates = []
        
        for symbol in symbols:
            try:
                market_data = data_manager.get_market_data(symbol, limit=100)
                if market_data:
                    signal = evaluator.evaluate_signal_with_protection(symbol, market_data)
                    if signal and signal.get('action') != 'HOLD':
                        candidates.append(signal)
            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}")
                continue
        
        if not candidates:
            logger.info("No valid trade candidates found")
            return None
        
        # Sort by enhanced score if available, otherwise by confidence
        def signal_score(signal):
            enhanced_score = signal.get('enhanced_score')
            if enhanced_score is not None:
                return enhanced_score
            else:
                confidence = signal.get('confidence', 0)
                regime_bonus = 0.1 if signal.get('regime_context', {}).get('regime_type') in ['BULL', 'SIDEWAYS'] else 0
                return confidence + regime_bonus
        
        candidates.sort(key=signal_score, reverse=True)
        best_signal = candidates[0]
        
        logger.info(f"📈 Best trade selected: {best_signal['symbol']} {best_signal['action']} (score: {signal_score(best_signal):.2f})")
        
        return best_signal
        
    except Exception as e:
        logger.error(f"Error finding best trade: {e}")
        return None

# =============================================================================
# COMPATIBILITY FUNCTIONS FOR EXISTING SYSTEM
# =============================================================================

# Create global instance for backward compatibility
signal_evaluator = SignalEvaluator()

# Export main functions for compatibility
def evaluate_signal(symbol: str, market_data: Dict) -> Dict[str, Any]:
    """Backward compatibility function"""
    return signal_evaluator.evaluate_signal(symbol, market_data)

def create_enhanced_signal_evaluator() -> SignalEvaluator:
    """Create enhanced signal evaluator with all features"""
    return SignalEvaluator()

# =============================================================================
# END OF ENHANCED SIGNAL EVALUATOR
# =============================================================================