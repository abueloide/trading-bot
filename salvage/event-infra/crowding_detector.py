#!/usr/bin/env python3
"""
HERD-001: Market Crowding Detection System
Complete implementation of anti-herding behavior for crypto trading bot
"""

import numpy as np
import pandas as pd
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import threading

logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES FOR HERD-001 ANALYSIS
# =============================================================================

@dataclass
class HerdingSignal:
    """Market-wide herding signal"""
    overall_herding_score: float
    correlation_herding: float
    sentiment_herding: float
    volume_herding: float
    confidence_level: float
    market_regime: str
    components: Dict
    timestamp: datetime

@dataclass
class TradeCrowdingSignal:
    """Trade-specific crowding signal"""
    trade_crowding_score: float
    order_book_clustering: float
    directional_bias: float
    size_concentration: float
    timing_correlation: float
    components: Dict
    timestamp: datetime

@dataclass
class ResponsibilityDecision:
    """Final trading decision with responsibility analysis"""
    action: str  # 'EXECUTE', 'DELAY', 'REDUCE_SIZE', 'BLOCK'
    timing_delay_seconds: int
    size_adjustment_factor: float
    original_size: float
    adjusted_size: float
    reasoning: str
    responsibility_score: float
    crowding_analysis: Dict
    timestamp: datetime

# =============================================================================
# HERDING ANALYZER - MARKET-WIDE BEHAVIOR DETECTION
# =============================================================================

class HerdingAnalyzer:
    """Analyzes market-wide herding behavior"""
    
    def __init__(self):
        self.correlation_window = 100
        self.sentiment_threshold = 0.7
        self.volume_spike_threshold = 2.0
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def analyze_market_herding(self, market_data: Dict) -> HerdingSignal:
        """Analyze overall market herding behavior"""
        try:
            start_time = time.time()
            
            # Extract market data components
            price_history = market_data.get('price_history', [])
            volume_history = market_data.get('volume_history', [])
            technical_indicators = market_data.get('technical', {})
            
            # Calculate herding components
            correlation_score = self._calculate_correlation_herding(market_data)
            sentiment_score = self._calculate_sentiment_herding(technical_indicators)
            volume_score = self._calculate_volume_herding(volume_history)
            
            # Combine scores with weights
            weights = {'correlation': 0.4, 'sentiment': 0.3, 'volume': 0.3}
            overall_score = (
                correlation_score * weights['correlation'] +
                sentiment_score * weights['sentiment'] +
                volume_score * weights['volume']
            )
            
            # Determine market regime
            market_regime = self._classify_market_regime(overall_score, market_data)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(market_data)
            
            analysis_time = time.time() - start_time
            
            components = {
                'correlation_analysis': {
                    'score': correlation_score,
                    'price_correlation': self._get_price_correlation(price_history),
                    'cross_asset_correlation': 0.65  # Placeholder for multi-asset
                },
                'sentiment_analysis': {
                    'score': sentiment_score,
                    'fear_greed_proxy': technical_indicators.get('rsi', 50) / 100,
                    'momentum_alignment': self._get_momentum_alignment(technical_indicators)
                },
                'volume_analysis': {
                    'score': volume_score,
                    'volume_spike_ratio': self._get_volume_spike_ratio(volume_history),
                    'volume_trend': self._get_volume_trend(volume_history)
                },
                'performance_metrics': {
                    'analysis_time_ms': round(analysis_time * 1000, 2),
                    'data_quality_score': confidence,
                    'cache_hit': False
                }
            }
            
            return HerdingSignal(
                overall_herding_score=overall_score,
                correlation_herding=correlation_score,
                sentiment_herding=sentiment_score,
                volume_herding=volume_score,
                confidence_level=confidence,
                market_regime=market_regime,
                components=components,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Herding analysis failed: {e}")
            return self._create_fallback_herding_signal()

    def _calculate_correlation_herding(self, market_data: Dict) -> float:
        """Calculate correlation-based herding score"""
        try:
            # Get price histories for correlation analysis
            price_history = market_data.get('price_history', [])
            
            if len(price_history) < 20:
                return 0.5  # Neutral score for insufficient data
            
            # Calculate price returns
            returns = np.diff(price_history) / np.array(price_history[:-1])
            
            # Calculate rolling correlations (proxy for market-wide correlation)
            if len(returns) > 50:
                window_size = min(20, len(returns) // 3)
                correlations = []
                
                for i in range(window_size, len(returns)):
                    window_returns = returns[i-window_size:i]
                    # Correlation with lagged returns (persistence indicator)
                    if len(window_returns) > 10:
                        corr = np.corrcoef(window_returns[:-1], window_returns[1:])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                
                if correlations:
                    avg_correlation = np.mean(correlations)
                    # High correlation indicates herding
                    return min(avg_correlation * 1.5, 1.0)
            
            # Fallback: Use price momentum clustering
            recent_returns = returns[-10:] if len(returns) >= 10 else returns
            if len(recent_returns) > 0:
                momentum_consistency = np.std(recent_returns)
                # Low volatility in returns can indicate herding
                return max(0.0, 1.0 - momentum_consistency * 10)
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Correlation herding calculation failed: {e}")
            return 0.5

    def _calculate_sentiment_herding(self, technical_indicators: Dict) -> float:
        """Calculate sentiment-based herding score"""
        try:
            # Use RSI as fear/greed proxy
            rsi = technical_indicators.get('rsi', 50)
            
            # Extreme RSI values indicate potential herding
            if rsi > 80:  # Extreme greed
                sentiment_score = min((rsi - 80) / 20, 1.0)
            elif rsi < 20:  # Extreme fear
                sentiment_score = min((20 - rsi) / 20, 1.0)
            else:
                sentiment_score = 0.0
            
            # Additional sentiment indicators
            macd = technical_indicators.get('macd', 0)
            bollinger_position = technical_indicators.get('bb_position', 0.5)
            
            # MACD divergence can indicate herding
            macd_herding = min(abs(macd) / 10, 0.3) if macd else 0
            
            # Bollinger band position (extreme positions indicate herding)
            bb_herding = max(0, abs(bollinger_position - 0.5) - 0.3) * 2
            
            # Combine sentiment components
            total_sentiment = sentiment_score + macd_herding + bb_herding
            return min(total_sentiment, 1.0)
            
        except Exception as e:
            logger.error(f"Sentiment herding calculation failed: {e}")
            return 0.5

    def _calculate_volume_herding(self, volume_history: List[float]) -> float:
        """Calculate volume-based herding score"""
        try:
            if len(volume_history) < 10:
                return 0.5
            
            volumes = np.array(volume_history)
            
            # Calculate volume spike ratio
            recent_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
            historical_volume = np.mean(volumes[:-5]) if len(volumes) > 5 else recent_volume
            
            if historical_volume > 0:
                spike_ratio = recent_volume / historical_volume
                # Volume spikes indicate potential herding
                volume_spike_score = min(max(spike_ratio - 1.0, 0) / 2.0, 0.7)
            else:
                volume_spike_score = 0.0
            
            # Volume trend consistency
            if len(volumes) > 20:
                recent_trend = np.polyfit(range(10), volumes[-10:], 1)[0]
                historical_trend = np.polyfit(range(10), volumes[-20:-10], 1)[0]
                
                # Trend acceleration indicates herding
                if historical_trend != 0:
                    trend_acceleration = abs(recent_trend / historical_trend - 1)
                    trend_score = min(trend_acceleration / 3, 0.3)
                else:
                    trend_score = 0.0
            else:
                trend_score = 0.0
            
            return min(volume_spike_score + trend_score, 1.0)
            
        except Exception as e:
            logger.error(f"Volume herding calculation failed: {e}")
            return 0.5

    def _classify_market_regime(self, herding_score: float, market_data: Dict) -> str:
        """Classify current market regime"""
        try:
            volatility = self._estimate_volatility(market_data.get('price_history', []))
            
            if herding_score > 0.8:
                return 'EXTREME_HERDING'
            elif herding_score > 0.6:
                if volatility > 0.05:
                    return 'HIGH_HERDING_HIGH_VOL'
                else:
                    return 'HIGH_HERDING_LOW_VOL'
            elif herding_score > 0.4:
                return 'MODERATE_HERDING'
            elif herding_score > 0.2:
                return 'LOW_HERDING'
            else:
                return 'NORMAL_MARKET'
                
        except Exception:
            return 'UNKNOWN'

    def _calculate_confidence(self, market_data: Dict) -> float:
        """Calculate confidence in herding analysis"""
        try:
            data_quality_factors = []
            
            # Price history quality
            price_history = market_data.get('price_history', [])
            if len(price_history) > 50:
                data_quality_factors.append(0.9)
            elif len(price_history) > 20:
                data_quality_factors.append(0.7)
            else:
                data_quality_factors.append(0.3)
            
            # Volume history quality
            volume_history = market_data.get('volume_history', [])
            if len(volume_history) > 50:
                data_quality_factors.append(0.9)
            elif len(volume_history) > 20:
                data_quality_factors.append(0.7)
            else:
                data_quality_factors.append(0.3)
            
            # Technical indicators availability
            technical = market_data.get('technical', {})
            if len(technical) > 3:
                data_quality_factors.append(0.9)
            elif len(technical) > 1:
                data_quality_factors.append(0.7)
            else:
                data_quality_factors.append(0.4)
            
            return np.mean(data_quality_factors)
            
        except Exception:
            return 0.5

    def _create_fallback_herding_signal(self) -> HerdingSignal:
        """Create fallback signal when analysis fails"""
        return HerdingSignal(
            overall_herding_score=0.5,
            correlation_herding=0.5,
            sentiment_herding=0.5,
            volume_herding=0.5,
            confidence_level=0.1,
            market_regime='UNKNOWN',
            components={'error': 'analysis_failed'},
            timestamp=datetime.now()
        )

    def _get_price_correlation(self, price_history: List[float]) -> float:
        """Calculate price correlation metric"""
        try:
            if len(price_history) < 20:
                return 0.5
            
            returns = np.diff(price_history) / np.array(price_history[:-1])
            if len(returns) > 10:
                # Autocorrelation in returns
                return abs(np.corrcoef(returns[:-1], returns[1:])[0, 1]) if len(returns) > 1 else 0.5
            return 0.5
        except Exception:
            return 0.5

    def _get_momentum_alignment(self, technical: Dict) -> float:
        """Calculate momentum alignment score"""
        try:
            rsi = technical.get('rsi', 50)
            macd = technical.get('macd', 0)
            
            # Strong momentum in same direction indicates alignment
            if rsi > 70 and macd > 0:
                return 0.8
            elif rsi < 30 and macd < 0:
                return 0.8
            elif 40 <= rsi <= 60:
                return 0.2
            else:
                return 0.5
        except Exception:
            return 0.5

    def _get_volume_spike_ratio(self, volume_history: List[float]) -> float:
        """Calculate volume spike ratio"""
        try:
            if len(volume_history) < 10:
                return 1.0
            
            recent = np.mean(volume_history[-5:])
            historical = np.mean(volume_history[:-5])
            
            return recent / historical if historical > 0 else 1.0
        except Exception:
            return 1.0

    def _get_volume_trend(self, volume_history: List[float]) -> float:
        """Calculate volume trend"""
        try:
            if len(volume_history) < 10:
                return 0.0
            
            x = np.arange(len(volume_history))
            slope = np.polyfit(x, volume_history, 1)[0]
            return slope / np.mean(volume_history) if np.mean(volume_history) > 0 else 0.0
        except Exception:
            return 0.0

    def _estimate_volatility(self, price_history: List[float]) -> float:
        """Estimate price volatility"""
        try:
            if len(price_history) < 10:
                return 0.02
            
            returns = np.diff(price_history) / np.array(price_history[:-1])
            return np.std(returns) if len(returns) > 0 else 0.02
        except Exception:
            return 0.02

# =============================================================================
# TRADE CROWDING ANALYZER - TRADE-SPECIFIC BEHAVIOR DETECTION
# =============================================================================

class TradeCrowdingAnalyzer:
    """Analyzes trade-specific crowding behavior"""
    
    def __init__(self):
        self.order_book_clustering_threshold = 0.7
        self.directional_bias_threshold = 0.8
        self.size_concentration_threshold = 0.6
        self.timing_correlation_window = 50
    
    def analyze_trade_crowding(self, symbol: str, direction: str, size_usd: float, market_data: Dict) -> TradeCrowdingSignal:
        """Analyze crowding for specific trade"""
        try:
            start_time = time.time()
            
            # Analyze different crowding dimensions
            order_book_clustering = self._analyze_order_book_clustering(market_data)
            directional_bias = self._analyze_directional_bias(direction, market_data)
            size_concentration = self._analyze_size_concentration(size_usd, market_data)
            timing_correlation = self._analyze_timing_correlation(market_data)
            
            # Calculate composite crowding score
            weights = {
                'order_book': 0.3,
                'directional': 0.3,
                'size': 0.2,
                'timing': 0.2
            }
            
            crowding_score = (
                order_book_clustering * weights['order_book'] +
                directional_bias * weights['directional'] +
                size_concentration * weights['size'] +
                timing_correlation * weights['timing']
            )
            
            analysis_time = time.time() - start_time
            
            components = {
                'order_book_analysis': {
                    'clustering_score': order_book_clustering,
                    'bid_ask_imbalance': self._get_bid_ask_imbalance(market_data),
                    'depth_concentration': self._get_depth_concentration(market_data)
                },
                'directional_analysis': {
                    'bias_score': directional_bias,
                    'recent_trade_direction': self._get_recent_trade_direction(market_data),
                    'momentum_alignment': self._get_directional_momentum(direction, market_data)
                },
                'size_analysis': {
                    'concentration_score': size_concentration,
                    'size_percentile': self._get_size_percentile(size_usd, market_data),
                    'whale_activity': self._detect_whale_activity(market_data)
                },
                'timing_analysis': {
                    'correlation_score': timing_correlation,
                    'burst_activity': self._detect_burst_activity(market_data),
                    'market_hours_factor': self._get_market_hours_factor()
                },
                'performance_metrics': {
                    'analysis_time_ms': round(analysis_time * 1000, 2),
                    'symbol': symbol,
                    'direction': direction,
                    'size_usd': size_usd
                }
            }
            
            return TradeCrowdingSignal(
                trade_crowding_score=crowding_score,
                order_book_clustering=order_book_clustering,
                directional_bias=directional_bias,
                size_concentration=size_concentration,
                timing_correlation=timing_correlation,
                components=components,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Trade crowding analysis failed: {e}")
            return self._create_fallback_crowding_signal()

    def _analyze_order_book_clustering(self, market_data: Dict) -> float:
        """Analyze order book clustering"""
        try:
            microstructure = market_data.get('microstructure', {})
            
            # Order book depth concentration
            order_book_depth = microstructure.get('order_book_depth', 0.5)
            bid_ask_ratio = microstructure.get('bid_ask_ratio', 1.0)
            
            # Higher concentration indicates more clustering
            depth_clustering = 1.0 - order_book_depth  # Invert: low depth = high clustering
            
            # Bid/ask imbalance indicates directional clustering
            imbalance_clustering = abs(bid_ask_ratio - 1.0) / max(bid_ask_ratio, 1.0)
            
            # Combine clustering signals
            clustering_score = (depth_clustering * 0.6 + imbalance_clustering * 0.4)
            return min(clustering_score, 1.0)
            
        except Exception as e:
            logger.error(f"Order book clustering analysis failed: {e}")
            return 0.5

    def _analyze_directional_bias(self, direction: str, market_data: Dict) -> float:
        """Analyze directional bias in market"""
        try:
            # Recent price movements
            price_history = market_data.get('price_history', [])
            if len(price_history) < 10:
                return 0.5
            
            # Calculate recent directional bias
            recent_prices = price_history[-10:]
            price_changes = np.diff(recent_prices)
            
            if len(price_changes) == 0:
                return 0.5
            
            # Calculate directional consistency
            positive_moves = sum(1 for change in price_changes if change > 0)
            negative_moves = sum(1 for change in price_changes if change < 0)
            total_moves = len(price_changes)
            
            if total_moves == 0:
                return 0.5
            
            directional_ratio = max(positive_moves, negative_moves) / total_moves
            
            # Check if our direction aligns with bias
            market_direction = 'BUY' if positive_moves > negative_moves else 'SELL'
            
            if direction == market_direction:
                # We're following the crowd
                return directional_ratio
            else:
                # We're contrarian
                return 1.0 - directional_ratio
                
        except Exception as e:
            logger.error(f"Directional bias analysis failed: {e}")
            return 0.5

    def _analyze_size_concentration(self, size_usd: float, market_data: Dict) -> float:
        """Analyze size concentration relative to market"""
        try:
            # Get recent volume data
            volume_history = market_data.get('volume_history', [])
            if not volume_history:
                return 0.5
            
            # Estimate recent average trade size
            recent_volume = np.mean(volume_history[-10:]) if len(volume_history) >= 10 else volume_history[-1]
            price_history = market_data.get('price_history', [])
            
            if price_history:
                recent_price = price_history[-1]
                estimated_trades_per_period = 100  # Rough estimate
                avg_trade_size = (recent_volume * recent_price) / estimated_trades_per_period
                
                # Calculate size percentile
                size_ratio = size_usd / avg_trade_size if avg_trade_size > 0 else 1.0
                
                # Larger trades indicate more concentration risk
                if size_ratio > 10:  # Very large trade
                    return 0.9
                elif size_ratio > 5:  # Large trade
                    return 0.7
                elif size_ratio > 2:  # Above average
                    return 0.5
                else:  # Small trade
                    return 0.2
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Size concentration analysis failed: {e}")
            return 0.5

    def _analyze_timing_correlation(self, market_data: Dict) -> float:
        """Analyze timing correlation with market activity"""
        try:
            # Check for burst activity patterns
            volume_history = market_data.get('volume_history', [])
            if len(volume_history) < 20:
                return 0.5
            
            # Detect volume bursts (indicator of coordinated activity)
            recent_volume = np.mean(volume_history[-5:])
            historical_volume = np.mean(volume_history[-20:-5])
            
            if historical_volume > 0:
                volume_spike_ratio = recent_volume / historical_volume
                
                # High volume spikes indicate timing correlation
                if volume_spike_ratio > 3.0:
                    return 0.9
                elif volume_spike_ratio > 2.0:
                    return 0.7
                elif volume_spike_ratio > 1.5:
                    return 0.5
                else:
                    return 0.2
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Timing correlation analysis failed: {e}")
            return 0.5

    def _create_fallback_crowding_signal(self) -> TradeCrowdingSignal:
        """Create fallback signal when analysis fails"""
        return TradeCrowdingSignal(
            trade_crowding_score=0.5,
            order_book_clustering=0.5,
            directional_bias=0.5,
            size_concentration=0.5,
            timing_correlation=0.5,
            components={'error': 'analysis_failed'},
            timestamp=datetime.now()
        )

    def _get_bid_ask_imbalance(self, market_data: Dict) -> float:
        """Calculate bid/ask imbalance"""
        try:
            microstructure = market_data.get('microstructure', {})
            return abs(microstructure.get('bid_ask_ratio', 1.0) - 1.0)
        except Exception:
            return 0.0

    def _get_depth_concentration(self, market_data: Dict) -> float:
        """Calculate order book depth concentration"""
        try:
            microstructure = market_data.get('microstructure', {})
            return 1.0 - microstructure.get('order_book_depth', 0.5)
        except Exception:
            return 0.5

    def _get_recent_trade_direction(self, market_data: Dict) -> str:
        """Get recent trade direction bias"""
        try:
            price_history = market_data.get('price_history', [])
            if len(price_history) >= 2:
                return 'BUY' if price_history[-1] > price_history[-2] else 'SELL'
            return 'NEUTRAL'
        except Exception:
            return 'NEUTRAL'

    def _get_directional_momentum(self, direction: str, market_data: Dict) -> float:
        """Calculate directional momentum alignment"""
        try:
            technical = market_data.get('technical', {})
            rsi = technical.get('rsi', 50)
            
            if direction == 'BUY':
                return (rsi - 50) / 50 if rsi > 50 else 0.0
            else:
                return (50 - rsi) / 50 if rsi < 50 else 0.0
        except Exception:
            return 0.0

    def _get_size_percentile(self, size_usd: float, market_data: Dict) -> float:
        """Calculate size percentile"""
        try:
            # Rough size classification
            if size_usd > 100000:
                return 0.95
            elif size_usd > 50000:
                return 0.85
            elif size_usd > 10000:
                return 0.70
            elif size_usd > 1000:
                return 0.50
            else:
                return 0.20
        except Exception:
            return 0.50

    def _detect_whale_activity(self, market_data: Dict) -> bool:
        """Detect whale activity in market"""
        try:
            volume_history = market_data.get('volume_history', [])
            if len(volume_history) < 10:
                return False
            
            recent_volume = max(volume_history[-5:])
            avg_volume = np.mean(volume_history[-20:])
            
            return recent_volume > avg_volume * 5
        except Exception:
            return False

    def _detect_burst_activity(self, market_data: Dict) -> bool:
        """Detect burst activity patterns"""
        try:
            volume_history = market_data.get('volume_history', [])
            if len(volume_history) < 10:
                return False
            
            recent_std = np.std(volume_history[-5:])
            historical_std = np.std(volume_history[-20:-5])
            
            return recent_std > historical_std * 2 if historical_std > 0 else False
        except Exception:
            return False

    def _get_market_hours_factor(self) -> float:
        """Get market hours factor for timing analysis"""
        try:
            current_hour = datetime.now().hour
            # Crypto markets are 24/7, but activity varies
            if 8 <= current_hour <= 16:  # Business hours
                return 0.8
            elif 16 <= current_hour <= 22:  # Evening trading
                return 0.6
            else:  # Night/early morning
                return 0.3
        except Exception:
            return 0.5

# =============================================================================
# RESPONSIBILITY ENGINE - FINAL DECISION MAKER
# =============================================================================

class ResponsibilityEngine:
    """Makes final trading decisions based on crowding analysis"""
    
    def __init__(self):
        self.extreme_crowding_threshold = 0.8
        self.high_crowding_threshold = 0.6
        self.moderate_crowding_threshold = 0.4
        self.max_delay_seconds = 180
        self.min_size_factor = 0.4
    
    def make_responsibility_decision(self, herding_signal: HerdingSignal, 
                                  crowding_signal: TradeCrowdingSignal,
                                  original_size: float, symbol: str, direction: str) -> ResponsibilityDecision:
        """Make final trading decision based on responsibility analysis"""
        try:
            # Calculate composite responsibility score
            responsibility_score = self._calculate_responsibility_score(herding_signal, crowding_signal)
            
            # Determine action based on responsibility score
            action, timing_delay, size_factor, reasoning = self._determine_action(
                responsibility_score, herding_signal, crowding_signal
            )
            
            adjusted_size = original_size * size_factor
            
            crowding_analysis = {
                'market_herding': {
                    'overall_score': herding_signal.overall_herding_score,
                    'regime': herding_signal.market_regime,
                    'confidence': herding_signal.confidence_level
                },
                'trade_crowding': {
                    'overall_score': crowding_signal.trade_crowding_score,
                    'order_book_clustering': crowding_signal.order_book_clustering,
                    'directional_bias': crowding_signal.directional_bias,
                    'size_concentration': crowding_signal.size_concentration,
                    'timing_correlation': crowding_signal.timing_correlation
                },
                'responsibility_metrics': {
                    'composite_score': responsibility_score,
                    'systemic_risk_level': self._assess_systemic_risk(responsibility_score),
                    'market_impact_estimate': self._estimate_market_impact(adjusted_size, crowding_signal),
                    'decorrelation_effectiveness': self._estimate_decorrelation_effectiveness(timing_delay)
                }
            }
            
            return ResponsibilityDecision(
                action=action,
                timing_delay_seconds=timing_delay,
                size_adjustment_factor=size_factor,
                original_size=original_size,
                adjusted_size=adjusted_size,
                reasoning=reasoning,
                responsibility_score=responsibility_score,
                crowding_analysis=crowding_analysis,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Responsibility decision failed: {e}")
            return self._create_fallback_decision(original_size)

    def _calculate_responsibility_score(self, herding_signal: HerdingSignal, 
                                      crowding_signal: TradeCrowdingSignal) -> float:
        """Calculate composite responsibility score"""
        try:
            # Weight market-wide vs trade-specific factors
            market_weight = 0.6
            trade_weight = 0.4
            
            market_component = herding_signal.overall_herding_score * market_weight
            trade_component = crowding_signal.trade_crowding_score * trade_weight
            
            base_score = market_component + trade_component
            
            # Adjust for confidence and market regime
            confidence_adjustment = herding_signal.confidence_level
            regime_adjustment = self._get_regime_adjustment(herding_signal.market_regime)
            
            final_score = base_score * confidence_adjustment * regime_adjustment
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Responsibility score calculation failed: {e}")
            return 0.5

    def _determine_action(self, responsibility_score: float, herding_signal: HerdingSignal,
                         crowding_signal: TradeCrowdingSignal) -> Tuple[str, int, float, str]:
        """Determine action based on responsibility score"""
        try:
            if responsibility_score >= self.extreme_crowding_threshold:
                # Extreme crowding - block trade
                if herding_signal.market_regime == 'EXTREME_HERDING':
                    return ('BLOCK', 0, 0.0, 
                           f'Extreme market herding detected (score: {responsibility_score:.3f}). '
                           'Blocking trade to prevent contributing to systemic risk.')
                else:
                    # Severe reduction and delay
                    delay = self.max_delay_seconds
                    size_factor = self.min_size_factor
                    return ('REDUCE_SIZE', delay, size_factor,
                           f'High crowding detected (score: {responsibility_score:.3f}). '
                           f'Reducing size to {size_factor*100:.0f}% and delaying {delay}s.')
                           
            elif responsibility_score >= self.high_crowding_threshold:
                # High crowding - significant adjustments
                delay = int(self.max_delay_seconds * 0.7)
                size_factor = 0.6
                return ('REDUCE_SIZE', delay, size_factor,
                       f'Moderate-high crowding detected (score: {responsibility_score:.3f}). '
                       f'Reducing size to {size_factor*100:.0f}% and delaying {delay}s.')
                       
            elif responsibility_score >= self.moderate_crowding_threshold:
                # Moderate crowding - timing delay only
                delay = int(self.max_delay_seconds * 0.4)
                size_factor = 0.8
                return ('DELAY', delay, size_factor,
                       f'Moderate crowding detected (score: {responsibility_score:.3f}). '
                       f'Applying {delay}s delay and small size reduction.')
                       
            else:
                # Low crowding - execute normally
                return ('EXECUTE', 0, 1.0,
                       f'Low crowding detected (score: {responsibility_score:.3f}). '
                       'Executing trade normally.')
                       
        except Exception as e:
            logger.error(f"Action determination failed: {e}")
            return ('EXECUTE', 0, 1.0, 'Fallback: executing normally due to analysis error.')

    def _get_regime_adjustment(self, market_regime: str) -> float:
        """Get adjustment factor based on market regime"""
        regime_adjustments = {
            'EXTREME_HERDING': 1.3,
            'HIGH_HERDING_HIGH_VOL': 1.2,
            'HIGH_HERDING_LOW_VOL': 1.1,
            'MODERATE_HERDING': 1.0,
            'LOW_HERDING': 0.9,
            'NORMAL_MARKET': 0.8,
            'UNKNOWN': 1.0
        }
        return regime_adjustments.get(market_regime, 1.0)

    def _assess_systemic_risk(self, responsibility_score: float) -> str:
        """Assess systemic risk level"""
        if responsibility_score >= 0.8:
            return 'HIGH'
        elif responsibility_score >= 0.6:
            return 'MODERATE'
        elif responsibility_score >= 0.4:
            return 'LOW'
        else:
            return 'MINIMAL'

    def _estimate_market_impact(self, trade_size: float, crowding_signal: TradeCrowdingSignal) -> str:
        """Estimate market impact of trade"""
        impact_factors = [
            crowding_signal.order_book_clustering,
            crowding_signal.size_concentration,
            crowding_signal.timing_correlation
        ]
        
        avg_impact = np.mean(impact_factors)
        
        if avg_impact >= 0.7:
            return 'HIGH'
        elif avg_impact >= 0.5:
            return 'MODERATE'
        else:
            return 'LOW'

    def _estimate_decorrelation_effectiveness(self, delay_seconds: int) -> str:
        """Estimate effectiveness of timing decorrelation"""
        if delay_seconds >= 120:
            return 'HIGH'
        elif delay_seconds >= 60:
            return 'MODERATE'
        elif delay_seconds > 0:
            return 'LOW'
        else:
            return 'NONE'

    def _create_fallback_decision(self, original_size: float) -> ResponsibilityDecision:
        """Create fallback decision when analysis fails"""
        return ResponsibilityDecision(
            action='EXECUTE',
            timing_delay_seconds=0,
            size_adjustment_factor=1.0,
            original_size=original_size,
            adjusted_size=original_size,
            reasoning='Fallback decision due to analysis error',
            responsibility_score=0.5,
            crowding_analysis={'error': 'analysis_failed'},
            timestamp=datetime.now()
        )

# =============================================================================
# MAIN CROWDING DETECTOR - ORCHESTRATES ALL COMPONENTS
# =============================================================================

class CrowdingDetector:
    """Main HERD-001 crowding detection system"""
    
    def __init__(self):
        self.herding_analyzer = HerdingAnalyzer()
        self.crowding_analyzer = TradeCrowdingAnalyzer()
        self.responsibility_engine = ResponsibilityEngine()
        self.analysis_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.performance_stats = {
            'total_analyses': 0,
            'total_time_ms': 0,
            'cache_hits': 0,
            'errors': 0
        }
    
    def analyze_complete_responsibility(self, symbol: str, direction: str, size_usd: float, 
                                     market_data: Dict) -> Dict[str, Any]:
        """Complete HERD-001 analysis for trading decision"""
        try:
            start_time = time.time()
            self.performance_stats['total_analyses'] += 1
            
            # Check cache first
            cache_key = self._generate_cache_key(symbol, direction, size_usd, market_data)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                self.performance_stats['cache_hits'] += 1
                return cached_result
            
            # Perform market herding analysis
            herding_signal = self.herding_analyzer.analyze_market_herding(market_data)
            
            # Perform trade crowding analysis
            crowding_signal = self.crowding_analyzer.analyze_trade_crowding(
                symbol, direction, size_usd, market_data
            )
            
            # Make final responsibility decision
            final_decision = self.responsibility_engine.make_responsibility_decision(
                herding_signal, crowding_signal, size_usd, symbol, direction
            )
            
            # Compile complete analysis result
            analysis_result = {
                'market_herding_analysis': asdict(herding_signal),
                'trade_crowding_analysis': asdict(crowding_signal),
                'final_decision': asdict(final_decision),
                'analysis_metadata': {
                    'total_analysis_time_ms': round((time.time() - start_time) * 1000, 2),
                    'analysis_timestamp': datetime.now(),
                    'symbol': symbol,
                    'direction': direction,
                    'original_size_usd': size_usd,
                    'herd001_version': '1.0.0',
                    'cache_used': False
                }
            }
            
            # Update performance stats
            self.performance_stats['total_time_ms'] += analysis_result['analysis_metadata']['total_analysis_time_ms']
            
            # Cache result
            self._cache_result(cache_key, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Complete responsibility analysis failed: {e}")
            self.performance_stats['errors'] += 1
            return self._create_fallback_analysis(symbol, direction, size_usd)

    def analyze_market_herding(self, market_data: Dict) -> HerdingSignal:
        """Analyze market-wide herding behavior only"""
        return self.herding_analyzer.analyze_market_herding(market_data)

    def analyze_trade_crowding(self, symbol: str, direction: str, size_usd: float, market_data: Dict) -> TradeCrowdingSignal:
        """Analyze trade-specific crowding only"""
        return self.crowding_analyzer.analyze_trade_crowding(symbol, direction, size_usd, market_data)

    def _generate_cache_key(self, symbol: str, direction: str, size_usd: float, market_data: Dict) -> str:
        """Generate cache key for analysis"""
        try:
            # Create deterministic key based on inputs
            key_data = {
                'symbol': symbol,
                'direction': direction,
                'size_bucket': int(size_usd / 1000) * 1000,  # Round to nearest $1000
                'price_hash': hash(tuple(market_data.get('price_history', [])[-10:])),
                'volume_hash': hash(tuple(market_data.get('volume_history', [])[-10:])),
                'timestamp_bucket': int(time.time() / 60) * 60  # Round to nearest minute
            }
            
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()[:16]
            
        except Exception:
            return f"{symbol}_{direction}_{int(time.time()/300)*300}"  # Fallback key

    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached analysis result if valid"""
        try:
            if cache_key in self.analysis_cache:
                cached_data, timestamp = self.analysis_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    cached_data['analysis_metadata']['cache_used'] = True
                    return cached_data
                else:
                    # Remove expired cache entry
                    del self.analysis_cache[cache_key]
            return None
        except Exception:
            return None

    def _cache_result(self, cache_key: str, result: Dict):
        """Cache analysis result"""
        try:
            self.analysis_cache[cache_key] = (result, time.time())
            
            # Clean old cache entries periodically
            if len(self.analysis_cache) > 100:
                self._clean_cache()
        except Exception:
            pass  # Caching is non-critical

    def _clean_cache(self):
        """Clean expired cache entries"""
        try:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self.analysis_cache.items()
                if current_time - timestamp > self.cache_ttl
            ]
            
            for key in expired_keys:
                del self.analysis_cache[key]
        except Exception:
            pass

    def _create_fallback_analysis(self, symbol: str, direction: str, size_usd: float) -> Dict:
        """Create fallback analysis when complete analysis fails"""
        return {
            'market_herding_analysis': {
                'overall_herding_score': 0.5,
                'market_regime': 'UNKNOWN',
                'confidence_level': 0.1
            },
            'trade_crowding_analysis': {
                'trade_crowding_score': 0.5,
                'order_book_clustering': 0.5,
                'directional_bias': 0.5,
                'size_concentration': 0.5,
                'timing_correlation': 0.5
            },
            'final_decision': {
                'action': 'EXECUTE',
                'timing_delay_seconds': 0,
                'size_adjustment_factor': 1.0,
                'original_size': size_usd,
                'adjusted_size': size_usd,
                'reasoning': 'Fallback decision due to analysis error',
                'responsibility_score': 0.5
            },
            'analysis_metadata': {
                'total_analysis_time_ms': 0,
                'analysis_timestamp': datetime.now(),
                'symbol': symbol,
                'direction': direction,
                'original_size_usd': size_usd,
                'herd001_version': '1.0.0',
                'error': 'fallback_analysis_used'
            }
        }

    def get_system_status(self) -> Dict:
        """Get HERD-001 system status"""
        return {
            'system': 'HERD-001 Market Crowding Detection',
            'version': '1.0.0',
            'status': 'operational',
            'components': {
                'herding_analyzer': 'active',
                'trade_cluster_analyzer': 'active',
                'timing_decorrelator': 'active',
                'position_adjuster': 'active'
            },
            'cache_size': len(self.analysis_cache),
            'last_analysis': max(
                [result['analysis_metadata']['analysis_timestamp'] 
                 for result in self.analysis_cache.values()],
                default=None
            ),
            'uptime': datetime.now()
        }

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        
        if stats['total_analyses'] > 0:
            stats['avg_analysis_time_ms'] = stats['total_time_ms'] / stats['total_analyses']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_analyses']
            stats['error_rate'] = stats['errors'] / stats['total_analyses']
        else:
            stats['avg_analysis_time_ms'] = 0
            stats['cache_hit_rate'] = 0
            stats['error_rate'] = 0
        
        return stats

# =============================================================================
# END OF CROWDING DETECTOR
# =============================================================================