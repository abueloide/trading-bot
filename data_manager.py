#!/usr/bin/env python3
"""
Enhanced Data Manager - Complete Implementation
Provides market data collection, microstructure analysis, anti-herding intelligence,
and complete database integration for crypto trading bot.
Compatible with existing infrastructure while adding advanced capabilities.
"""

import requests
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import threading
from collections import defaultdict, deque
import json
import uuid
from dataclasses import dataclass, asdict

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION AND IMPORTS
# =============================================================================

# Import configuration with enhanced fallbacks
try:
    from config import (
        get_api_credentials, ENABLE_DATABASE, ENABLE_MICROSTRUCTURE, 
        DATABASE_LOGGING, DB_LOGGING_CONFIG
    )
except ImportError:
    logger.warning("Config not available, using defaults")
    ENABLE_DATABASE = False
    ENABLE_MICROSTRUCTURE = False
    DATABASE_LOGGING = False
    DB_LOGGING_CONFIG = {}
    
    def get_api_credentials():
        return {}

# Database integration with safety
try:
    from database_manager import get_database_manager, MarketDataRecord
    DATABASE_AVAILABLE = True
    logger.info("✅ Database manager available")
except ImportError:
    DATABASE_AVAILABLE = False
    logger.warning("Database manager not available, running without persistence")

# Constants
BASE_URL = "https://api.binance.com"
PRICE_HISTORY_LENGTH = 200
VOLUME_HISTORY_LENGTH = 200
DATA_COLLECTION_INTERVAL = 60
CACHE_TTL = 30
MICROSTRUCTURE_UPDATE_INTERVAL = 10
ANTI_HERDING_ANALYSIS_INTERVAL = 300  # 5 minutes

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MarketDataPoint:
    """Enhanced market data point with microstructure"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    liquidity_score: Optional[float] = None

@dataclass
class MicrostructureMetrics:
    """Comprehensive microstructure analysis"""
    symbol: str
    timestamp: datetime
    bid_ask_spread: float
    order_book_depth: float
    bid_ask_ratio: float
    price_impact: float
    liquidity_score: float
    market_efficiency: float
    volatility_clustering: float
    
@dataclass
class AntiHerdingSignals:
    """Anti-herding analysis results"""
    symbol: str
    timestamp: datetime
    herding_strength: float
    crowding_score: float
    contrarian_opportunity: float
    market_sentiment_divergence: float
    volume_concentration: float
    price_momentum_dispersion: float
    systemic_risk_indicator: float

# =============================================================================
# ENHANCED BINANCE API CLIENT
# =============================================================================

class EnhancedBinanceClient:
    """Enhanced Binance API client with advanced features"""
    
    def __init__(self):
        self.base_url = BASE_URL
        self.session = requests.Session()
        self.rate_limit_delay = 0.1
        self.last_request_time = 0
        self.request_history = deque(maxlen=100)
        
        # Enhanced session configuration
        self.session.headers.update({
            'User-Agent': 'CryptoTradingBot/2.0',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Performance tracking
        self.api_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'last_error': None
        }

    def _enhanced_rate_limit(self):
        """Enhanced rate limiting with adaptive delays"""
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        cutoff_time = current_time - 60
        while self.request_history and self.request_history[0] < cutoff_time:
            self.request_history.popleft()
        
        # Adaptive rate limiting based on recent request frequency
        if len(self.request_history) > 50:  # More than 50 requests in last minute
            self.rate_limit_delay = 0.2  # Slow down
        elif len(self.request_history) < 10:  # Less than 10 requests in last minute
            self.rate_limit_delay = 0.05  # Speed up
        else:
            self.rate_limit_delay = 0.1  # Default
        
        # Apply rate limit
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
        self.request_history.append(self.last_request_time)

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Enhanced request method with error handling and stats"""
        try:
            self._enhanced_rate_limit()
            start_time = time.time()
            
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params or {}, timeout=15)
            
            # Update timing stats
            response_time = time.time() - start_time
            self._update_api_stats(response_time, True)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self._update_api_stats(0, False, str(e))
            logger.error(f"API request failed for {endpoint}: {e}")
            return None
        except Exception as e:
            self._update_api_stats(0, False, str(e))
            logger.error(f"Unexpected error in API request: {e}")
            return None

    def _update_api_stats(self, response_time: float, success: bool, error: str = None):
        """Update API performance statistics"""
        self.api_stats['total_requests'] += 1
        
        if success:
            self.api_stats['successful_requests'] += 1
            # Update average response time
            total_successful = self.api_stats['successful_requests']
            current_avg = self.api_stats['avg_response_time']
            self.api_stats['avg_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
        else:
            self.api_stats['failed_requests'] += 1
            self.api_stats['last_error'] = error

    def get_enhanced_ticker(self, symbol: str) -> Optional[Dict]:
        """Get enhanced ticker data with additional metrics"""
        data = self._make_request("/api/v3/ticker/24hr", {'symbol': symbol})
        if not data:
            return None
            
        try:
            return {
                'symbol': data['symbol'],
                'current_price': float(data['lastPrice']),
                'price_change': float(data['priceChange']),
                'price_change_percent': float(data['priceChangePercent']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice']),
                'volume_24h': float(data['volume']),
                'quote_volume_24h': float(data['quoteVolume']),
                'bid_price': float(data['bidPrice']),
                'ask_price': float(data['askPrice']),
                'bid_qty': float(data['bidQty']),
                'ask_qty': float(data['askQty']),
                'trade_count': int(data['count']),
                'timestamp': datetime.now()
            }
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing ticker data: {e}")
            return None

    def get_enhanced_klines(self, symbol: str, interval: str = '1h', limit: int = 200) -> Optional[List[Dict]]:
        """Get enhanced historical kline data"""
        data = self._make_request("/api/v3/klines", {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000)
        })
        
        if not data:
            return None
            
        try:
            klines = []
            for kline in data:
                klines.append({
                    'open_time': datetime.fromtimestamp(kline[0] / 1000),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': datetime.fromtimestamp(kline[6] / 1000),
                    'quote_volume': float(kline[7]),
                    'trade_count': int(kline[8]),
                    'taker_buy_volume': float(kline[9]),
                    'taker_buy_quote': float(kline[10])
                })
            return klines
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Error parsing kline data: {e}")
            return None

    def get_order_book_enhanced(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """Get enhanced order book data for microstructure analysis"""
        if not ENABLE_MICROSTRUCTURE:
            return None
            
        data = self._make_request("/api/v3/depth", {
            'symbol': symbol,
            'limit': min(limit, 5000)
        })
        
        if not data:
            return None
            
        try:
            bids = [(float(price), float(qty)) for price, qty in data['bids']]
            asks = [(float(price), float(qty)) for price, qty in data['asks']]
            
            return {
                'bids': bids,
                'asks': asks,
                'last_update_id': data['lastUpdateId'],
                'timestamp': datetime.now()
            }
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing order book data: {e}")
            return None

    def get_api_stats(self) -> Dict[str, Any]:
        """Get API performance statistics"""
        stats = self.api_stats.copy()
        stats['success_rate'] = (
            stats['successful_requests'] / max(stats['total_requests'], 1) * 100
        )
        stats['requests_per_minute'] = len(self.request_history)
        return stats

# =============================================================================
# ADVANCED MICROSTRUCTURE ANALYZER
# =============================================================================

class AdvancedMicrostructureAnalyzer:
    """Advanced microstructure analysis for anti-herding intelligence"""
    
    def __init__(self):
        self.historical_spreads = defaultdict(deque)
        self.historical_depths = defaultdict(deque)
        self.market_efficiency_cache = {}
        self.volatility_clustering_cache = {}
        
    def analyze_comprehensive_microstructure(self, symbol: str, order_book: Dict, 
                                           current_price: float, 
                                           historical_data: List[Dict]) -> MicrostructureMetrics:
        """Comprehensive microstructure analysis"""
        try:
            if not order_book or not order_book.get('bids') or not order_book.get('asks'):
                return self._get_fallback_microstructure(symbol)
            
            bids = order_book['bids']
            asks = order_book['asks']
            
            # Basic microstructure metrics
            best_bid = bids[0][0] if bids else current_price * 0.999
            best_ask = asks[0][0] if asks else current_price * 1.001
            
            # Bid-ask spread analysis
            spread_absolute = best_ask - best_bid
            spread_relative = spread_absolute / current_price if current_price > 0 else 0.001
            
            # Order book depth analysis
            depth_metrics = self._analyze_order_book_depth(bids, asks, current_price)
            
            # Bid-ask ratio (market imbalance)
            bid_ask_ratio = self._calculate_bid_ask_ratio(bids, asks, current_price)
            
            # Price impact estimation
            price_impact = self._estimate_enhanced_price_impact(bids, asks, current_price)
            
            # Market efficiency score
            market_efficiency = self._calculate_market_efficiency(symbol, historical_data)
            
            # Volatility clustering
            volatility_clustering = self._calculate_volatility_clustering(symbol, historical_data)
            
            # Composite liquidity score
            liquidity_score = self._calculate_enhanced_liquidity_score(
                spread_relative, depth_metrics['total_depth'], price_impact, market_efficiency
            )
            
            # Update historical data for future analysis
            self.historical_spreads[symbol].append(spread_relative)
            self.historical_depths[symbol].append(depth_metrics['total_depth'])
            
            # Limit historical data size
            if len(self.historical_spreads[symbol]) > 1000:
                self.historical_spreads[symbol].popleft()
            if len(self.historical_depths[symbol]) > 1000:
                self.historical_depths[symbol].popleft()
            
            return MicrostructureMetrics(
                symbol=symbol,
                timestamp=datetime.now(),
                bid_ask_spread=spread_relative,
                order_book_depth=depth_metrics['normalized_depth'],
                bid_ask_ratio=bid_ask_ratio,
                price_impact=price_impact,
                liquidity_score=liquidity_score,
                market_efficiency=market_efficiency,
                volatility_clustering=volatility_clustering
            )
            
        except Exception as e:
            logger.error(f"Microstructure analysis failed for {symbol}: {e}")
            return self._get_fallback_microstructure(symbol)

    def _analyze_order_book_depth(self, bids: List[Tuple], asks: List[Tuple], 
                                 current_price: float) -> Dict[str, float]:
        """Analyze order book depth with multiple metrics"""
        try:
            # Define depth ranges
            ranges = [0.001, 0.005, 0.01, 0.02, 0.05]  # 0.1%, 0.5%, 1%, 2%, 5%
            depth_analysis = {}
            
            for range_pct in ranges:
                range_value = current_price * range_pct
                
                # Calculate bid depth within range
                bid_depth = sum(qty for price, qty in bids 
                              if price >= current_price - range_value)
                
                # Calculate ask depth within range
                ask_depth = sum(qty for price, qty in asks 
                              if price <= current_price + range_value)
                
                depth_analysis[f'depth_{range_pct*100:.1f}pct'] = bid_depth + ask_depth
            
            # Total depth (1% range is most important)
            total_depth = depth_analysis.get('depth_1.0pct', 0)
            
            # Normalized depth score (0-1 scale)
            normalized_depth = min(total_depth / 10000, 1.0)  # Normalize to reasonable range
            
            return {
                'total_depth': total_depth,
                'normalized_depth': normalized_depth,
                'depth_analysis': depth_analysis
            }
            
        except Exception as e:
            logger.error(f"Order book depth analysis failed: {e}")
            return {'total_depth': 100, 'normalized_depth': 0.5, 'depth_analysis': {}}

    def _calculate_bid_ask_ratio(self, bids: List[Tuple], asks: List[Tuple], 
                                current_price: float) -> float:
        """Calculate bid-ask ratio for market imbalance detection"""
        try:
            # Focus on orders within 1% of current price
            price_range = current_price * 0.01
            
            bid_volume = sum(qty for price, qty in bids 
                           if price >= current_price - price_range)
            ask_volume = sum(qty for price, qty in asks 
                           if price <= current_price + price_range)
            
            if ask_volume > 0:
                ratio = bid_volume / ask_volume
            else:
                ratio = 5.0  # Heavy bid side
                
            # Normalize ratio (cap extreme values)
            return min(max(ratio, 0.1), 10.0)
            
        except Exception as e:
            logger.error(f"Bid-ask ratio calculation failed: {e}")
            return 1.0

    def _estimate_enhanced_price_impact(self, bids: List[Tuple], asks: List[Tuple], 
                                       current_price: float) -> float:
        """Enhanced price impact estimation for multiple trade sizes"""
        try:
            trade_sizes_usd = [100, 500, 1000, 5000]  # Different trade sizes
            price_impacts = []
            
            for trade_size in trade_sizes_usd:
                quantity_needed = trade_size / current_price
                
                # Simulate market buy (walk through asks)
                remaining_qty = quantity_needed
                total_cost = 0
                
                for price, qty in asks:
                    if remaining_qty <= 0:
                        break
                    
                    take_qty = min(remaining_qty, qty)
                    total_cost += take_qty * price
                    remaining_qty -= take_qty
                
                if quantity_needed > remaining_qty:
                    executed_qty = quantity_needed - remaining_qty
                    avg_price = total_cost / executed_qty if executed_qty > 0 else current_price
                    impact = abs(avg_price - current_price) / current_price
                    price_impacts.append(impact)
            
            # Return weighted average impact (favor smaller trade sizes)
            if price_impacts:
                weights = [0.4, 0.3, 0.2, 0.1]  # Weight smaller trades more
                weighted_impact = sum(impact * weight for impact, weight in zip(price_impacts, weights))
                return min(weighted_impact, 0.1)  # Cap at 10%
            else:
                return 0.005  # Default 0.5%
                
        except Exception as e:
            logger.error(f"Price impact estimation failed: {e}")
            return 0.005

    def _calculate_market_efficiency(self, symbol: str, historical_data: List[Dict]) -> float:
        """Calculate market efficiency score based on price discovery"""
        try:
            cache_key = f"{symbol}_efficiency_{datetime.now().hour}"
            if cache_key in self.market_efficiency_cache:
                return self.market_efficiency_cache[cache_key]
            
            if not historical_data or len(historical_data) < 20:
                return 0.5
            
            # Extract price data
            prices = [float(data.get('close', data.get('current_price', 0))) 
                     for data in historical_data[-20:]]
            
            if not prices or len(prices) < 10:
                return 0.5
            
            # Calculate price changes
            price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] 
                           for i in range(1, len(prices)) if prices[i-1] > 0]
            
            if not price_changes:
                return 0.5
            
            # Market efficiency indicators
            # 1. Price change variance (lower is more efficient)
            price_variance = np.var(price_changes) if len(price_changes) > 1 else 0
            
            # 2. Autocorrelation in returns (lower is more efficient)
            if len(price_changes) > 5:
                returns = np.array(price_changes)
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
                autocorr = abs(autocorr) if not np.isnan(autocorr) else 0
            else:
                autocorr = 0
            
            # 3. Price momentum persistence (lower is more efficient)
            momentum_changes = sum(1 for i in range(1, len(price_changes)) 
                                 if (price_changes[i] > 0) == (price_changes[i-1] > 0))
            momentum_persistence = momentum_changes / max(len(price_changes) - 1, 1)
            
            # Composite efficiency score (higher is more efficient)
            efficiency_score = 1.0 - min(price_variance * 1000, 0.5) - min(autocorr, 0.3) - min(momentum_persistence * 0.5, 0.2)
            efficiency_score = max(0.0, min(1.0, efficiency_score))
            
            # Cache result
            self.market_efficiency_cache[cache_key] = efficiency_score
            
            return efficiency_score
            
        except Exception as e:
            logger.error(f"Market efficiency calculation failed: {e}")
            return 0.5

    def _calculate_volatility_clustering(self, symbol: str, historical_data: List[Dict]) -> float:
        """Calculate volatility clustering metric"""
        try:
            cache_key = f"{symbol}_volatility_{datetime.now().hour}"
            if cache_key in self.volatility_clustering_cache:
                return self.volatility_clustering_cache[cache_key]
            
            if not historical_data or len(historical_data) < 30:
                return 0.5
            
            # Extract price data
            prices = [float(data.get('close', data.get('current_price', 0))) 
                     for data in historical_data[-30:]]
            
            if len(prices) < 20:
                return 0.5
            
            # Calculate returns
            returns = [abs(prices[i] - prices[i-1]) / prices[i-1] 
                      for i in range(1, len(prices)) if prices[i-1] > 0]
            
            if len(returns) < 10:
                return 0.5
            
            # Calculate volatility clustering using GARCH-like approach
            # High volatility tends to be followed by high volatility
            volatility_persistence = 0
            high_vol_threshold = np.percentile(returns, 75)
            
            consecutive_high_vol = 0
            max_consecutive = 0
            
            for ret in returns:
                if ret > high_vol_threshold:
                    consecutive_high_vol += 1
                    max_consecutive = max(max_consecutive, consecutive_high_vol)
                else:
                    consecutive_high_vol = 0
            
            # Normalize clustering score
            clustering_score = min(max_consecutive / 5.0, 1.0)  # Max 5 consecutive periods
            
            # Cache result
            self.volatility_clustering_cache[cache_key] = clustering_score
            
            return clustering_score
            
        except Exception as e:
            logger.error(f"Volatility clustering calculation failed: {e}")
            return 0.5

    def _calculate_enhanced_liquidity_score(self, spread: float, depth: float, 
                                          price_impact: float, efficiency: float) -> float:
        """Calculate enhanced composite liquidity score"""
        try:
            # Component scores (normalized to 0-1)
            spread_score = max(0, 1 - spread * 1000)  # Lower spread = higher score
            depth_score = min(depth, 1.0)  # Higher depth = higher score
            impact_score = max(0, 1 - price_impact * 100)  # Lower impact = higher score
            efficiency_score = efficiency  # Higher efficiency = higher score
            
            # Weighted composite score
            liquidity_score = (
                spread_score * 0.25 +
                depth_score * 0.30 +
                impact_score * 0.25 +
                efficiency_score * 0.20
            )
            
            return max(0.0, min(1.0, liquidity_score))
            
        except Exception as e:
            logger.error(f"Liquidity score calculation failed: {e}")
            return 0.5

    def _get_fallback_microstructure(self, symbol: str) -> MicrostructureMetrics:
        """Get fallback microstructure data when analysis fails"""
        return MicrostructureMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_ask_spread=0.001,
            order_book_depth=0.5,
            bid_ask_ratio=1.0,
            price_impact=0.005,
            liquidity_score=0.5,
            market_efficiency=0.5,
            volatility_clustering=0.5
        )

# =============================================================================
# ANTI-HERDING INTELLIGENCE ENGINE
# =============================================================================

class AntiHerdingIntelligence:
    """Advanced anti-herding analysis and signal generation"""
    
    def __init__(self):
        self.historical_signals = defaultdict(deque)
        self.market_sentiment_cache = {}
        self.crowding_indicators = defaultdict(list)
        self.contrarian_opportunities = defaultdict(list)
        
    def analyze_herding_behavior(self, symbol: str, market_data: Dict, 
                                microstructure: MicrostructureMetrics) -> AntiHerdingSignals:
        """Comprehensive anti-herding analysis"""
        try:
            # Extract relevant data
            price_history = market_data.get('price_history', [])
            volume_history = market_data.get('volume_history', [])
            current_price = market_data.get('current_price', 0)
            
            if not price_history or len(price_history) < 20:
                return self._get_fallback_herding_signals(symbol)
            
            # 1. Herding strength detection
            herding_strength = self._detect_herding_strength(price_history, volume_history)
            
            # 2. Market crowding analysis
            crowding_score = self._calculate_crowding_score(
                symbol, price_history, volume_history, microstructure
            )
            
            # 3. Contrarian opportunity identification
            contrarian_opportunity = self._identify_contrarian_opportunities(
                symbol, price_history, volume_history, microstructure
            )
            
            # 4. Market sentiment divergence
            sentiment_divergence = self._calculate_sentiment_divergence(
                symbol, price_history, volume_history
            )
            
            # 5. Volume concentration analysis
            volume_concentration = self._analyze_volume_concentration(volume_history)
            
            # 6. Price momentum dispersion
            momentum_dispersion = self._calculate_momentum_dispersion(price_history)
            
            # 7. Systemic risk indicator
            systemic_risk = self._calculate_systemic_risk_indicator(
                symbol, market_data, microstructure
            )
            
            # Create anti-herding signals
            signals = AntiHerdingSignals(
                symbol=symbol,
                timestamp=datetime.now(),
                herding_strength=herding_strength,
                crowding_score=crowding_score,
                contrarian_opportunity=contrarian_opportunity,
                market_sentiment_divergence=sentiment_divergence,
                volume_concentration=volume_concentration,
                price_momentum_dispersion=momentum_dispersion,
                systemic_risk_indicator=systemic_risk
            )
            
            # Store historical signals
            self.historical_signals[symbol].append(signals)
            if len(self.historical_signals[symbol]) > 100:
                self.historical_signals[symbol].popleft()
            
            return signals
            
        except Exception as e:
            logger.error(f"Anti-herding analysis failed for {symbol}: {e}")
            return self._get_fallback_herding_signals(symbol)

    def _detect_herding_strength(self, price_history: List[float], 
                                volume_history: List[float]) -> float:
        """Detect herding behavior strength in market movements"""
        try:
            if len(price_history) < 10 or len(volume_history) < 10:
                return 0.5
            
            # Calculate price momentum
            recent_prices = price_history[-10:]
            price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # Calculate volume surge
            recent_volumes = volume_history[-10:]
            avg_volume = np.mean(volume_history[-30:]) if len(volume_history) >= 30 else np.mean(volume_history)
            current_volume = recent_volumes[-1]
            volume_surge = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price volatility acceleration
            if len(recent_prices) >= 5:
                recent_volatility = np.std(recent_prices[-5:])
                historical_volatility = np.std(price_history[-20:]) if len(price_history) >= 20 else recent_volatility
                volatility_acceleration = recent_volatility / historical_volatility if historical_volatility > 0 else 1.0
            else:
                volatility_acceleration = 1.0
            
            # Herding strength indicators
            momentum_factor = min(abs(price_momentum) * 10, 1.0)  # Strong momentum indicates herding
            volume_factor = min((volume_surge - 1.0) * 0.5, 1.0) if volume_surge > 1.0 else 0.0
            volatility_factor = min((volatility_acceleration - 1.0) * 0.3, 1.0) if volatility_acceleration > 1.0 else 0.0
            
            # Composite herding strength
            herding_strength = (momentum_factor * 0.4 + volume_factor * 0.35 + volatility_factor * 0.25)
            
            return max(0.0, min(1.0, herding_strength))
            
        except Exception as e:
            logger.error(f"Herding strength detection failed: {e}")
            return 0.5

    def _calculate_crowding_score(self, symbol: str, price_history: List[float], 
                                 volume_history: List[float], 
                                 microstructure: MicrostructureMetrics) -> float:
        """Calculate market crowding score"""
        try:
            crowding_indicators = []
            
            # 1. Liquidity deterioration (lower liquidity = more crowding)
            liquidity_factor = 1.0 - microstructure.liquidity_score
            crowding_indicators.append(liquidity_factor * 0.3)
            
            # 2. Bid-ask spread expansion
            spread_factor = min(microstructure.bid_ask_spread * 1000, 1.0)
            crowding_indicators.append(spread_factor * 0.2)
            
            # 3. Price impact increase
            impact_factor = min(microstructure.price_impact * 100, 1.0)
            crowding_indicators.append(impact_factor * 0.2)
            
            # 4. Volume concentration
            if len(volume_history) >= 10:
                recent_volumes = volume_history[-10:]
                volume_std = np.std(recent_volumes)
                volume_mean = np.mean(recent_volumes)
                concentration_factor = min(volume_std / volume_mean, 1.0) if volume_mean > 0 else 0
                crowding_indicators.append(concentration_factor * 0.15)
            
            # 5. Price trend uniformity
            if len(price_history) >= 20:
                recent_changes = [price_history[i] - price_history[i-1] 
                                for i in range(-10, 0) if i < len(price_history)]
                positive_changes = sum(1 for change in recent_changes if change > 0)
                uniformity = abs(positive_changes - 5) / 5.0  # Deviation from 50/50
                crowding_indicators.append(uniformity * 0.15)
            
            # Composite crowding score
            crowding_score = sum(crowding_indicators)
            
            return max(0.0, min(1.0, crowding_score))
            
        except Exception as e:
            logger.error(f"Crowding score calculation failed: {e}")
            return 0.5

    def _identify_contrarian_opportunities(self, symbol: str, price_history: List[float], 
                                         volume_history: List[float], 
                                         microstructure: MicrostructureMetrics) -> float:
        """Identify contrarian trading opportunities"""
        try:
            contrarian_factors = []
            
            # 1. Extreme price movements (potential reversals)
            if len(price_history) >= 20:
                recent_change = (price_history[-1] - price_history[-10]) / price_history[-10]
                historical_volatility = np.std([
                    (price_history[i] - price_history[i-10]) / price_history[i-10]
                    for i in range(10, len(price_history))
                    if price_history[i-10] > 0
                ])
                
                if historical_volatility > 0:
                    extreme_factor = min(abs(recent_change) / (2 * historical_volatility), 1.0)
                    contrarian_factors.append(extreme_factor * 0.3)
            
            # 2. Volume-price divergence
            if len(price_history) >= 10 and len(volume_history) >= 10:
                price_trend = (price_history[-1] - price_history[-5]) / price_history[-5]
                volume_trend = (volume_history[-1] - np.mean(volume_history[-10:-5])) / np.mean(volume_history[-10:-5])
                
                # Strong price move with weak volume = contrarian opportunity
                if abs(price_trend) > 0.02 and abs(volume_trend) < 0.1:
                    divergence_factor = min(abs(price_trend) * 10, 1.0)
                    contrarian_factors.append(divergence_factor * 0.25)
            
            # 3. Liquidity recovery after stress
            efficiency_recovery = microstructure.market_efficiency
            if efficiency_recovery > 0.7:  # Market returning to efficiency
                contrarian_factors.append(efficiency_recovery * 0.2)
            
            # 4. Mean reversion signals
            if len(price_history) >= 30:
                current_price = price_history[-1]
                long_term_avg = np.mean(price_history[-30:])
                deviation = abs(current_price - long_term_avg) / long_term_avg
                
                if deviation > 0.05:  # More than 5% deviation
                    reversion_factor = min(deviation * 5, 1.0)
                    contrarian_factors.append(reversion_factor * 0.25)
            
            # Composite contrarian opportunity score
            contrarian_opportunity = sum(contrarian_factors)
            
            return max(0.0, min(1.0, contrarian_opportunity))
            
        except Exception as e:
            logger.error(f"Contrarian opportunity identification failed: {e}")
            return 0.5

    def _calculate_sentiment_divergence(self, symbol: str, price_history: List[float], 
                                       volume_history: List[float]) -> float:
        """Calculate market sentiment divergence"""
        try:
            if len(price_history) < 20 or len(volume_history) < 20:
                return 0.5
            
            # Price-based sentiment
            recent_price_change = (price_history[-1] - price_history[-10]) / price_history[-10]
            price_sentiment = 1.0 if recent_price_change > 0 else 0.0
            
            # Volume-based sentiment
            recent_avg_volume = np.mean(volume_history[-5:])
            historical_avg_volume = np.mean(volume_history[-20:-5])
            volume_sentiment = 1.0 if recent_avg_volume > historical_avg_volume else 0.0
            
            # Momentum-based sentiment
            short_ma = np.mean(price_history[-5:])
            long_ma = np.mean(price_history[-20:])
            momentum_sentiment = 1.0 if short_ma > long_ma else 0.0
            
            # Calculate divergence
            sentiments = [price_sentiment, volume_sentiment, momentum_sentiment]
            sentiment_variance = np.var(sentiments)
            
            # Higher variance = more divergence
            divergence_score = min(sentiment_variance * 4, 1.0)
            
            return divergence_score
            
        except Exception as e:
            logger.error(f"Sentiment divergence calculation failed: {e}")
            return 0.5

    def _analyze_volume_concentration(self, volume_history: List[float]) -> float:
        """Analyze volume concentration patterns"""
        try:
            if len(volume_history) < 20:
                return 0.5
            
            # Calculate volume distribution
            recent_volumes = volume_history[-10:]
            historical_volumes = volume_history[-20:-10]
            
            # Volume concentration indicators
            recent_avg = np.mean(recent_volumes)
            historical_avg = np.mean(historical_volumes)
            
            # Volume spike concentration
            volume_spikes = sum(1 for vol in recent_volumes if vol > historical_avg * 1.5)
            spike_concentration = volume_spikes / len(recent_volumes)
            
            # Volume variance
            volume_cv = np.std(recent_volumes) / recent_avg if recent_avg > 0 else 0
            
            # Concentration score (higher = more concentrated/unusual)
            concentration_score = min(spike_concentration + volume_cv * 0.5, 1.0)
            
            return concentration_score
            
        except Exception as e:
            logger.error(f"Volume concentration analysis failed: {e}")
            return 0.5

    def _calculate_momentum_dispersion(self, price_history: List[float]) -> float:
        """Calculate price momentum dispersion"""
        try:
            if len(price_history) < 30:
                return 0.5
            
            # Calculate momentum across different timeframes
            momentums = []
            timeframes = [5, 10, 15, 20]
            
            for tf in timeframes:
                if len(price_history) >= tf:
                    momentum = (price_history[-1] - price_history[-tf]) / price_history[-tf]
                    momentums.append(momentum)
            
            if not momentums:
                return 0.5
            
            # Calculate dispersion (variance) of momentums
            momentum_dispersion = np.var(momentums) if len(momentums) > 1 else 0
            
            # Normalize dispersion score
            dispersion_score = min(momentum_dispersion * 1000, 1.0)
            
            return dispersion_score
            
        except Exception as e:
            logger.error(f"Momentum dispersion calculation failed: {e}")
            return 0.5

    def _calculate_systemic_risk_indicator(self, symbol: str, market_data: Dict, 
                                         microstructure: MicrostructureMetrics) -> float:
        """Calculate systemic risk indicator"""
        try:
            risk_factors = []
            
            # 1. Liquidity risk
            liquidity_risk = 1.0 - microstructure.liquidity_score
            risk_factors.append(liquidity_risk * 0.3)
            
            # 2. Volatility clustering risk
            clustering_risk = microstructure.volatility_clustering
            risk_factors.append(clustering_risk * 0.2)
            
            # 3. Market efficiency deterioration
            efficiency_risk = 1.0 - microstructure.market_efficiency
            risk_factors.append(efficiency_risk * 0.2)
            
            # 4. Price impact risk
            impact_risk = min(microstructure.price_impact * 100, 1.0)
            risk_factors.append(impact_risk * 0.15)
            
            # 5. Bid-ask spread expansion risk
            spread_risk = min(microstructure.bid_ask_spread * 1000, 1.0)
            risk_factors.append(spread_risk * 0.15)
            
            # Composite systemic risk
            systemic_risk = sum(risk_factors)
            
            return max(0.0, min(1.0, systemic_risk))
            
        except Exception as e:
            logger.error(f"Systemic risk calculation failed: {e}")
            return 0.5

    def _get_fallback_herding_signals(self, symbol: str) -> AntiHerdingSignals:
        """Get fallback anti-herding signals when analysis fails"""
        return AntiHerdingSignals(
            symbol=symbol,
            timestamp=datetime.now(),
            herding_strength=0.5,
            crowding_score=0.5,
            contrarian_opportunity=0.5,
            market_sentiment_divergence=0.5,
            volume_concentration=0.5,
            price_momentum_dispersion=0.5,
            systemic_risk_indicator=0.5
        )

    def get_contrarian_signals(self, symbol: str) -> List[Dict]:
        """Get actionable contrarian trading signals"""
        try:
            if symbol not in self.historical_signals or not self.historical_signals[symbol]:
                return []
            
            latest_signals = self.historical_signals[symbol][-1]
            signals = []
            
            # Strong contrarian buy signal
            if (latest_signals.contrarian_opportunity > 0.7 and 
                latest_signals.crowding_score > 0.6 and
                latest_signals.systemic_risk_indicator < 0.4):
                
                signals.append({
                    'signal_type': 'contrarian_buy',
                    'strength': latest_signals.contrarian_opportunity,
                    'confidence': (latest_signals.contrarian_opportunity + 
                                 (1.0 - latest_signals.systemic_risk_indicator)) / 2,
                    'reasoning': 'High contrarian opportunity with manageable systemic risk'
                })
            
            # Herding avoidance signal
            if (latest_signals.herding_strength > 0.8 and 
                latest_signals.crowding_score > 0.7):
                
                signals.append({
                    'signal_type': 'avoid_herding',
                    'strength': latest_signals.herding_strength,
                    'confidence': latest_signals.crowding_score,
                    'reasoning': 'Strong herding behavior detected - avoid following crowd'
                })
            
            # Liquidity warning signal
            if latest_signals.systemic_risk_indicator > 0.8:
                signals.append({
                    'signal_type': 'liquidity_warning',
                    'strength': latest_signals.systemic_risk_indicator,
                    'confidence': 0.9,
                    'reasoning': 'High systemic risk - reduce position sizes'
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating contrarian signals: {e}")
            return []

# =============================================================================
# ENHANCED DATA MANAGER - MAIN CLASS
# =============================================================================

class DataManager:
    """Enhanced Data Manager with complete database integration and anti-herding intelligence"""
    
    def __init__(self):
        logger.info("🚀 Initializing Enhanced DataManager...")
        
        # Core components
        self.api_client = EnhancedBinanceClient()
        self.microstructure_analyzer = AdvancedMicrostructureAnalyzer()
        self.anti_herding_engine = AntiHerdingIntelligence()
        
        # Data storage
        self.price_data = {}
        self.volume_data = {}
        self.microstructure_data = {}
        self.anti_herding_data = {}
        self.ticker_cache = {}
        
        # Configuration
        self.cache_ttl = CACHE_TTL
        self.symbols_to_collect = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
        
        # Database integration
        self.db_manager = None
        self.db_enabled = False
        if ENABLE_DATABASE and DATABASE_AVAILABLE:
            try:
                self.db_manager = get_database_manager()
                self.db_enabled = True
                logger.info("✅ Database integration enabled in DataManager")
            except Exception as e:
                logger.error(f"Database initialization failed: {e}")
                self.db_enabled = False
        else:
            logger.info("📊 Database integration disabled or unavailable")
        
        # Data collection control
        self.is_collecting = False
        self.collection_thread = None
        self.collection_lock = threading.Lock()
        
        # Performance tracking
        self.performance_stats = {
            'data_requests': 0,
            'successful_requests': 0,
            'cache_hits': 0,
            'database_operations': 0,
            'avg_response_time': 0.0
        }
        
        # Initialize data structures
        self._initialize_data_structures()
        
        logger.info("✅ Enhanced DataManager initialized successfully")

    def _initialize_data_structures(self):
        """Initialize data storage structures"""
        for symbol in self.symbols_to_collect:
            self.price_data[symbol] = deque(maxlen=PRICE_HISTORY_LENGTH)
            self.volume_data[symbol] = deque(maxlen=VOLUME_HISTORY_LENGTH)
            self.microstructure_data[symbol] = None
            self.anti_herding_data[symbol] = None

    # =========================================================================
    # CORE DATA RETRIEVAL METHODS (Enhanced)
    # =========================================================================

    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Dict[str, Any]:
        """Enhanced market data retrieval with comprehensive analysis"""
        try:
            start_time = time.time()
            self.performance_stats['data_requests'] += 1
            
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{limit}"
            if self._is_cache_valid(cache_key):
                self.performance_stats['cache_hits'] += 1
                return self.ticker_cache[cache_key]['data']
            
            # Get current price and ticker data
            ticker_data = self.api_client.get_enhanced_ticker(symbol)
            if not ticker_data:
                return self._get_fallback_market_data(symbol)
            
            # Get historical kline data
            historical_data = self.api_client.get_enhanced_klines(symbol, timeframe, limit)
            if not historical_data:
                return self._get_fallback_market_data(symbol)
            
            # Extract price and volume history
            price_history = [kline['close'] for kline in historical_data]
            volume_history = [kline['volume'] for kline in historical_data]
            timestamps = [kline['close_time'] for kline in historical_data]
            
            # Update internal storage
            with self.collection_lock:
                self.price_data[symbol] = deque(price_history, maxlen=PRICE_HISTORY_LENGTH)
                self.volume_data[symbol] = deque(volume_history, maxlen=VOLUME_HISTORY_LENGTH)
            
            # Calculate enhanced technical indicators
            technical_data = self._calculate_enhanced_technical_indicators(
                price_history, volume_history, historical_data
            )
            
            # Create comprehensive market data
            market_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': ticker_data['current_price'],
                'price_history': price_history,
                'volume_history': volume_history,
                'timestamps': timestamps,
                'ticker_data': ticker_data,
                'technical_indicators': technical_data,
                'data_quality': self._calculate_data_quality(symbol, historical_data),
                'data_points': len(price_history),
                'source': 'enhanced_api',
                'timestamp': datetime.now(),
                'response_time': time.time() - start_time
            }
            
            # Cache the result
            self.ticker_cache[cache_key] = {
                'data': market_data,
                'timestamp': datetime.now()
            }
            
            # Log to database if enabled
            if self.db_enabled:
                asyncio.create_task(self._async_log_market_data(symbol, market_data))
            
            # Update performance stats
            self.performance_stats['successful_requests'] += 1
            self._update_avg_response_time(time.time() - start_time)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Enhanced market data retrieval failed for {symbol}: {e}")
            return self._get_fallback_market_data(symbol)

    def get_market_data_with_microstructure(self, symbol: str) -> Dict[str, Any]:
        """Get market data enhanced with comprehensive microstructure analysis"""
        try:
            # Get base market data
            market_data = self.get_market_data(symbol)
            if not market_data:
                return None
            
            # Add microstructure analysis if enabled
            if ENABLE_MICROSTRUCTURE:
                microstructure = self._get_enhanced_microstructure_data(symbol, market_data)
                market_data['microstructure'] = asdict(microstructure) if microstructure else {}
                market_data['microstructure_enabled'] = True
                
                # Store for future reference
                self.microstructure_data[symbol] = microstructure
            else:
                market_data['microstructure_enabled'] = False
                market_data['microstructure'] = {}
            
            # Add anti-herding analysis
            anti_herding = self._get_anti_herding_analysis(symbol, market_data)
            market_data['anti_herding'] = asdict(anti_herding) if anti_herding else {}
            
            # Store anti-herding data
            self.anti_herding_data[symbol] = anti_herding
            
            return market_data
            
        except Exception as e:
            logger.error(f"Enhanced market data with microstructure failed for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with enhanced caching and error handling"""
        try:
            # Check cache first
            cache_key = f"price_{symbol}"
            if self._is_cache_valid(cache_key):
                return self.ticker_cache[cache_key]['data']
            
            # Get fresh ticker data
            ticker_data = self.api_client.get_enhanced_ticker(symbol)
            if ticker_data:
                price = ticker_data['current_price']
                
                # Cache the result
                self.ticker_cache[cache_key] = {
                    'data': price,
                    'timestamp': datetime.now()
                }
                
                return price
            
            # Fallback to stored data
            if symbol in self.price_data and self.price_data[symbol]:
                return self.price_data[symbol][-1]
            
            return None
            
        except Exception as e:
            logger.error(f"Enhanced current price retrieval failed for {symbol}: {e}")
            return None

    # =========================================================================
    # DATABASE INTEGRATION METHODS
    # =========================================================================

    async def _async_log_market_data(self, symbol: str, market_data: Dict):
        """Asynchronously log market data to database"""
        if not self.db_enabled:
            return
        
        try:
            # Extract latest data point
            price_history = market_data.get('price_history', [])
            volume_history = market_data.get('volume_history', [])
            timestamps = market_data.get('timestamps', [])
            
            if not price_history or not volume_history:
                return
            
            # Create market data record
            latest_timestamp = timestamps[-1] if timestamps else datetime.now()
            
            market_record = {
                'symbol': symbol,
                'timeframe': market_data.get('timeframe', '1h'),
                'open_time': latest_timestamp - timedelta(hours=1),
                'close_time': latest_timestamp,
                'open_price': price_history[-1],
                'high_price': price_history[-1],
                'low_price': price_history[-1],
                'close_price': price_history[-1],
                'volume': volume_history[-1],
                'quote_volume': volume_history[-1] * price_history[-1],
                'trade_count': 100,  # Estimated
                'data_quality': market_data.get('data_quality', 0.9)
            }
            
            # Log to database
            success = await self.db_manager.log_market_data(market_record)
            if success:
                self.performance_stats['database_operations'] += 1
            
        except Exception as e:
            logger.error(f"Database logging failed for {symbol}: {e}")

    async def log_market_data_to_db(self, symbol: str, data: Dict) -> bool:
        """Public method to log market data to database"""
        try:
            await self._async_log_market_data(symbol, data)
            return True
        except Exception as e:
            logger.error(f"Failed to log market data to database: {e}")
            return False

    async def get_historical_data_from_db(self, symbol: str, start_date: datetime, 
                                        end_date: datetime) -> List[Dict]:
        """Get historical data from database"""
        if not self.db_enabled:
            return []
        
        try:
            # This would need to be implemented in database_manager
            # For now, return empty list
            logger.warning("Historical data retrieval from DB not yet implemented")
            return []
        except Exception as e:
            logger.error(f"Failed to get historical data from DB: {e}")
            return []

    async def cleanup_old_data(self, retention_days: int = 30) -> int:
        """Clean up old data from database"""
        if not self.db_enabled:
            return 0
        
        try:
            # This would need to be implemented in database_manager
            logger.warning("Database cleanup not yet implemented")
            return 0
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")
            return 0

    def get_database_status(self) -> Dict[str, Any]:
        """Get database connection and performance status"""
        return {
            'database_enabled': self.db_enabled,
            'database_available': DATABASE_AVAILABLE,
            'database_operations': self.performance_stats['database_operations'],
            'last_operation': datetime.now() if self.db_enabled else None
        }

    # =========================================================================
    # MICROSTRUCTURE AND ANTI-HERDING METHODS
    # =========================================================================

    def _get_enhanced_microstructure_data(self, symbol: str, market_data: Dict) -> Optional[MicrostructureMetrics]:
        """Get enhanced microstructure analysis"""
        try:
            # Get order book data
            order_book = self.api_client.get_order_book_enhanced(symbol)
            if not order_book:
                return None
            
            # Perform comprehensive analysis
            current_price = market_data.get('current_price', 0)
            historical_data = [market_data]  # Could be enhanced with more history
            
            microstructure = self.microstructure_analyzer.analyze_comprehensive_microstructure(
                symbol, order_book, current_price, historical_data
            )
            
            return microstructure
            
        except Exception as e:
            logger.error(f"Enhanced microstructure data failed for {symbol}: {e}")
            return None

    def _get_anti_herding_analysis(self, symbol: str, market_data: Dict) -> Optional[AntiHerdingSignals]:
        """Get anti-herding analysis"""
        try:
            # Get microstructure data (or fallback)
            microstructure = self.microstructure_data.get(symbol)
            if not microstructure:
                microstructure = self.microstructure_analyzer._get_fallback_microstructure(symbol)
            
            # Perform anti-herding analysis
            anti_herding = self.anti_herding_engine.analyze_herding_behavior(
                symbol, market_data, microstructure
            )
            
            return anti_herding
            
        except Exception as e:
            logger.error(f"Anti-herding analysis failed for {symbol}: {e}")
            return None

    def analyze_order_book_depth(self, symbol: str) -> Dict[str, float]:
        """Analyze order book depth for liquidity assessment"""
        try:
            order_book = self.api_client.get_order_book_enhanced(symbol, limit=100)
            if not order_book:
                return {'error': 'No order book data available'}
            
            current_price = self.get_current_price(symbol)
            if not current_price:
                return {'error': 'No current price available'}
            
            depth_analysis = self.microstructure_analyzer._analyze_order_book_depth(
                order_book['bids'], order_book['asks'], current_price
            )
            
            return depth_analysis
            
        except Exception as e:
            logger.error(f"Order book depth analysis failed for {symbol}: {e}")
            return {'error': str(e)}

    def calculate_liquidity_metrics(self, symbol: str) -> Dict[str, float]:
        """Calculate comprehensive liquidity metrics"""
        try:
            microstructure = self.microstructure_data.get(symbol)
            if not microstructure:
                return {'error': 'No microstructure data available'}
            
            return {
                'liquidity_score': microstructure.liquidity_score,
                'bid_ask_spread': microstructure.bid_ask_spread,
                'order_book_depth': microstructure.order_book_depth,
                'price_impact': microstructure.price_impact,
                'market_efficiency': microstructure.market_efficiency
            }
            
        except Exception as e:
            logger.error(f"Liquidity metrics calculation failed for {symbol}: {e}")
            return {'error': str(e)}

    def detect_market_crowding(self, symbol: str) -> Dict[str, Any]:
        """Detect market crowding conditions"""
        try:
            anti_herding = self.anti_herding_data.get(symbol)
            if not anti_herding:
                return {'error': 'No anti-herding data available'}
            
            return {
                'crowding_score': anti_herding.crowding_score,
                'herding_strength': anti_herding.herding_strength,
                'volume_concentration': anti_herding.volume_concentration,
                'systemic_risk_indicator': anti_herding.systemic_risk_indicator,
                'crowding_detected': anti_herding.crowding_score > 0.7,
                'risk_level': 'HIGH' if anti_herding.systemic_risk_indicator > 0.7 else 
                           'MEDIUM' if anti_herding.systemic_risk_indicator > 0.4 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Market crowding detection failed for {symbol}: {e}")
            return {'error': str(e)}

    def get_microstructure_signals(self, symbol: str) -> List[Dict]:
        """Get actionable microstructure-based trading signals"""
        try:
            signals = []
            microstructure = self.microstructure_data.get(symbol)
            
            if not microstructure:
                return signals
            
            # Liquidity warning signal
            if microstructure.liquidity_score < 0.3:
                signals.append({
                    'signal_type': 'liquidity_warning',
                    'strength': 1.0 - microstructure.liquidity_score,
                    'message': 'Low liquidity detected - consider reducing position size'
                })
            
            # Spread expansion signal
            if microstructure.bid_ask_spread > 0.01:  # 1% spread
                signals.append({
                    'signal_type': 'spread_expansion',
                    'strength': min(microstructure.bid_ask_spread * 100, 1.0),
                    'message': 'High bid-ask spread indicates stressed market conditions'
                })
            
            # Market efficiency signal
            if microstructure.market_efficiency > 0.8:
                signals.append({
                    'signal_type': 'high_efficiency',
                    'strength': microstructure.market_efficiency,
                    'message': 'High market efficiency - good conditions for systematic trading'
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Microstructure signals generation failed for {symbol}: {e}")
            return []

    def detect_herding_behavior(self, symbol: str) -> Dict[str, Any]:
        """Detect herding behavior in market"""
        try:
            anti_herding = self.anti_herding_data.get(symbol)
            if not anti_herding:
                return {'error': 'No anti-herding data available'}
            
            return {
                'herding_detected': anti_herding.herding_strength > 0.7,
                'herding_strength': anti_herding.herding_strength,
                'sentiment_divergence': anti_herding.market_sentiment_divergence,
                'momentum_dispersion': anti_herding.price_momentum_dispersion,
                'recommendation': 'AVOID' if anti_herding.herding_strength > 0.8 else
                               'CAUTION' if anti_herding.herding_strength > 0.6 else 'NORMAL'
            }
            
        except Exception as e:
            logger.error(f"Herding behavior detection failed for {symbol}: {e}")
            return {'error': str(e)}

    def calculate_crowding_score(self, symbol: str) -> float:
        """Calculate market crowding score"""
        try:
            anti_herding = self.anti_herding_data.get(symbol)
            return anti_herding.crowding_score if anti_herding else 0.5
        except Exception as e:
            logger.error(f"Crowding score calculation failed for {symbol}: {e}")
            return 0.5

    def get_contrarian_signals(self, symbol: str) -> List[Dict]:
        """Get contrarian trading signals"""
        try:
            return self.anti_herding_engine.get_contrarian_signals(symbol)
        except Exception as e:
            logger.error(f"Contrarian signals generation failed for {symbol}: {e}")
            return []

    def assess_systemic_risk(self, portfolio: List[str]) -> Dict[str, float]:
        """Assess systemic risk across portfolio"""
        try:
            risk_scores = {}
            total_risk = 0.0
            
            for symbol in portfolio:
                anti_herding = self.anti_herding_data.get(symbol)
                if anti_herding:
                    risk_score = anti_herding.systemic_risk_indicator
                    risk_scores[symbol] = risk_score
                    total_risk += risk_score
                else:
                    risk_scores[symbol] = 0.5  # Default moderate risk
                    total_risk += 0.5
            
            # Calculate portfolio-wide metrics
            avg_risk = total_risk / len(portfolio) if portfolio else 0.0
            max_risk = max(risk_scores.values()) if risk_scores else 0.0
            risk_concentration = max_risk - avg_risk  # How concentrated is the risk
            
            return {
                'portfolio_average_risk': avg_risk,
                'maximum_individual_risk': max_risk,
                'risk_concentration': risk_concentration,
                'total_portfolio_risk': min(total_risk * 1.2, 1.0),  # 20% correlation adjustment
                'individual_risks': risk_scores,
                'risk_level': 'HIGH' if avg_risk > 0.7 else 'MEDIUM' if avg_risk > 0.4 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Systemic risk assessment failed: {e}")
            return {'error': str(e)}

    # =========================================================================
    # INTELLIGENT DATA COLLECTION METHODS
    # =========================================================================

    def start_adaptive_collection(self, symbols: List[str] = None) -> bool:
        """Start adaptive data collection with intelligent frequency adjustment"""
        try:
            if self.is_collecting:
                logger.warning("Data collection already running")
                return False
            
            if symbols:
                self.symbols_to_collect = symbols
            
            self.is_collecting = True
            self.collection_thread = threading.Thread(
                target=self._adaptive_collection_loop,
                daemon=True
            )
            self.collection_thread.start()
            
            logger.info(f"✅ Adaptive data collection started for {len(self.symbols_to_collect)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start adaptive collection: {e}")
            self.is_collecting = False
            return False

    def stop_data_collection(self):
        """Stop data collection gracefully"""
        try:
            self.is_collecting = False
            
            if self.collection_thread and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=10)
            
            logger.info("✅ Data collection stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop data collection: {e}")

    def _adaptive_collection_loop(self):
        """Adaptive data collection loop with intelligent frequency adjustment"""
        logger.info("🔄 Adaptive data collection loop started")
        
        collection_intervals = {symbol: DATA_COLLECTION_INTERVAL for symbol in self.symbols_to_collect}
        last_collection_times = {symbol: 0 for symbol in self.symbols_to_collect}
        
        while self.is_collecting:
            try:
                current_time = time.time()
                
                for symbol in self.symbols_to_collect:
                    # Check if it's time to collect data for this symbol
                    time_since_last = current_time - last_collection_times[symbol]
                    
                    if time_since_last >= collection_intervals[symbol]:
                        # Collect data for this symbol
                        self._collect_symbol_data_enhanced(symbol)
                        last_collection_times[symbol] = current_time
                        
                        # Adjust collection frequency based on market conditions
                        new_interval = self._calculate_adaptive_interval(symbol)
                        collection_intervals[symbol] = new_interval
                
                # Sleep for a short time before next check
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in adaptive collection loop: {e}")
                time.sleep(30)  # Wait longer on error

    def _collect_symbol_data_enhanced(self, symbol: str):
        """Enhanced data collection for a specific symbol"""
        try:
            start_time = time.time()
            
            # Get comprehensive market data
            market_data = self.get_market_data_with_microstructure(symbol)
            if not market_data:
                return
            
            # Update internal caches
            current_price = market_data.get('current_price')
            if current_price:
                self.ticker_cache[f"price_{symbol}"] = {
                    'data': current_price,
                    'timestamp': datetime.now()
                }
            
            # Log collection performance
            collection_time = time.time() - start_time
            logger.debug(f"📊 Collected data for {symbol} in {collection_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Enhanced data collection failed for {symbol}: {e}")

    def _calculate_adaptive_interval(self, symbol: str) -> int:
        """Calculate adaptive collection interval based on market conditions"""
        try:
            # Get current anti-herding data
            anti_herding = self.anti_herding_data.get(symbol)
            microstructure = self.microstructure_data.get(symbol)
            
            base_interval = DATA_COLLECTION_INTERVAL
            
            # Adjust based on market volatility and herding
            if anti_herding and microstructure:
                # More frequent collection during high volatility or herding
                volatility_factor = anti_herding.price_momentum_dispersion
                herding_factor = anti_herding.herding_strength
                liquidity_factor = 1.0 - microstructure.liquidity_score
                
                # Calculate adjustment multiplier
                urgency_score = (volatility_factor + herding_factor + liquidity_factor) / 3.0
                
                # More urgent conditions = shorter intervals
                if urgency_score > 0.8:
                    return max(base_interval // 4, 15)  # 4x faster, min 15 seconds
                elif urgency_score > 0.6:
                    return max(base_interval // 2, 30)  # 2x faster, min 30 seconds
                elif urgency_score < 0.3:
                    return min(base_interval * 2, 300)  # 2x slower, max 5 minutes
            
            return base_interval
            
        except Exception as e:
            logger.error(f"Adaptive interval calculation failed for {symbol}: {e}")
            return DATA_COLLECTION_INTERVAL

    def adjust_collection_frequency(self, symbol: str, market_volatility: float) -> None:
        """Manually adjust collection frequency for a symbol"""
        try:
            # This would integrate with the adaptive collection system
            logger.info(f"Manual frequency adjustment for {symbol}: volatility={market_volatility}")
            
            # Store the adjustment for the adaptive system to use
            if not hasattr(self, 'manual_adjustments'):
                self.manual_adjustments = {}
            
            self.manual_adjustments[symbol] = {
                'volatility': market_volatility,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Manual frequency adjustment failed for {symbol}: {e}")

    def prioritize_data_sources(self, symbol: str) -> List[str]:
        """Prioritize data sources based on quality and reliability"""
        try:
            # Get API performance stats
            api_stats = self.api_client.get_api_stats()
            
            # Priority list based on performance and reliability
            sources = []
            
            # Primary source: Binance API (if performing well)
            if api_stats.get('success_rate', 0) > 90:
                sources.append('binance_api_primary')
            
            # Secondary source: Cached data (if fresh)
            cache_key = f"price_{symbol}"
            if self._is_cache_valid(cache_key):
                sources.append('internal_cache')
            
            # Tertiary source: Database (if available)
            if self.db_enabled:
                sources.append('database_historical')
            
            # Fallback source: Mock data generator
            sources.append('fallback_generator')
            
            return sources
            
        except Exception as e:
            logger.error(f"Data source prioritization failed for {symbol}: {e}")
            return ['binance_api_primary', 'fallback_generator']

    def validate_data_quality(self, symbol: str, data: Dict) -> float:
        """Validate and score data quality"""
        try:
            quality_factors = []
            
            # 1. Data completeness
            required_fields = ['current_price', 'price_history', 'volume_history']
            completeness = sum(1 for field in required_fields if data.get(field)) / len(required_fields)
            quality_factors.append(completeness * 0.3)
            
            # 2. Data freshness
            timestamp = data.get('timestamp')
            if timestamp:
                age_minutes = (datetime.now() - timestamp).total_seconds() / 60
                freshness = max(0, 1 - age_minutes / 60)  # Decay over 1 hour
                quality_factors.append(freshness * 0.25)
            
            # 3. Data consistency
            price_history = data.get('price_history', [])
            if len(price_history) > 1:
                # Check for unrealistic price jumps
                price_changes = [abs(price_history[i] - price_history[i-1]) / price_history[i-1] 
                               for i in range(1, len(price_history)) if price_history[i-1] > 0]
                
                if price_changes:
                    max_change = max(price_changes)
                    consistency = 1.0 if max_change < 0.1 else max(0, 1 - max_change)  # Penalize >10% jumps
                    quality_factors.append(consistency * 0.25)
            
            # 4. Source reliability
            source = data.get('source', 'unknown')
            source_reliability = {
                'enhanced_api': 1.0,
                'binance_api': 0.9,
                'internal_cache': 0.7,
                'historical_fetch': 0.8,
                'fallback_mock': 0.3,
                'emergency_fallback': 0.1
            }
            reliability = source_reliability.get(source, 0.5)
            quality_factors.append(reliability * 0.2)
            
            # Calculate composite quality score
            quality_score = sum(quality_factors)
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Data quality validation failed for {symbol}: {e}")
            return 0.5

    # =========================================================================
    # LEGACY COMPATIBILITY METHODS
    # =========================================================================

    def get_fear_greed_index(self) -> Dict[str, Any]:
        """Get Fear & Greed Index (legacy compatibility)"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    return {
                        "value": int(data["data"][0]["value"]),
                        "classification": data["data"][0]["value_classification"],
                        "timestamp": datetime.now()
                    }
            
            # Fallback to neutral values
            return {"value": 50, "classification": "Neutral", "timestamp": datetime.now()}
            
        except Exception as e:
            logger.error(f"Error getting fear & greed index: {e}")
            return {"value": 50, "classification": "Neutral", "timestamp": datetime.now()}

    def get_fred_data(self, series_id: str) -> Dict[str, Any]:
        """Get FRED economic data (placeholder for legacy compatibility)"""
        return {
            "series_id": series_id,
            "value": None,
            "status": "not_implemented",
            "note": "FRED API requires API key configuration",
            "timestamp": datetime.now()
        }

    def get_market_data_with_regime(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get market data with regime analysis (enhanced legacy compatibility)"""
        try:
            # Get enhanced market data
            market_data = self.get_market_data_with_microstructure(symbol)
            if not market_data:
                return self._get_fallback_regime_data(symbol)
            
            # Enhanced regime classification using anti-herding data
            anti_herding = market_data.get('anti_herding', {})
            microstructure = market_data.get('microstructure', {})
            
            # Determine regime based on multiple factors
            regime = self._classify_market_regime(anti_herding, microstructure)
            crisis = self._detect_crisis_conditions(anti_herding, microstructure)
            manipulation = self._detect_manipulation_patterns(anti_herding, microstructure)
            
            return {
                "market_data": market_data,
                "regime": regime,
                "crisis": crisis,
                "manipulation": manipulation,
                "symbol": symbol,
                "timestamp": datetime.now(),
                "enhanced_analysis": True
            }
            
        except Exception as e:
            logger.error(f"Enhanced regime analysis failed for {symbol}: {e}")
            return self._get_fallback_regime_data(symbol)

    def _classify_market_regime(self, anti_herding: Dict, microstructure: Dict) -> Dict[str, Any]:
        """Classify market regime using enhanced indicators"""
        try:
            herding_strength = anti_herding.get('herding_strength', 0.5)
            liquidity_score = microstructure.get('liquidity_score', 0.5)
            volatility_clustering = microstructure.get('volatility_clustering', 0.5)
            market_efficiency = microstructure.get('market_efficiency', 0.5)
            
            # Regime classification logic
            if herding_strength > 0.8 and liquidity_score < 0.3:
                regime_type = "CRISIS_HERDING"
                confidence = 0.9
            elif market_efficiency > 0.8 and liquidity_score > 0.7:
                regime_type = "EFFICIENT_NORMAL"
                confidence = 0.8
            elif volatility_clustering > 0.7 and herding_strength > 0.6:
                regime_type = "VOLATILE_TRENDING"
                confidence = 0.7
            elif liquidity_score < 0.4:
                regime_type = "ILLIQUID_STRESSED"
                confidence = 0.6
            else:
                regime_type = "MIXED_UNCERTAIN"
                confidence = 0.4
            
            return {
                "regime_type": regime_type,
                "confidence": confidence,
                "indicators": {
                    "herding_strength": herding_strength,
                    "liquidity_score": liquidity_score,
                    "market_efficiency": market_efficiency,
                    "volatility_clustering": volatility_clustering
                }
            }
            
        except Exception as e:
            logger.error(f"Regime classification failed: {e}")
            return {"regime_type": "ERROR", "confidence": 0.0}

    def _detect_crisis_conditions(self, anti_herding: Dict, microstructure: Dict) -> Dict[str, Any]:
        """Detect crisis conditions using enhanced indicators"""
        try:
            systemic_risk = anti_herding.get('systemic_risk_indicator', 0.5)
            liquidity_score = microstructure.get('liquidity_score', 0.5)
            herding_strength = anti_herding.get('herding_strength', 0.5)
            
            # Crisis detection logic
            crisis_indicators = []
            
            if systemic_risk > 0.8:
                crisis_indicators.append("HIGH_SYSTEMIC_RISK")
            
            if liquidity_score < 0.2:
                crisis_indicators.append("LIQUIDITY_CRISIS")
            
            if herding_strength > 0.9:
                crisis_indicators.append("EXTREME_HERDING")
            
            # Determine crisis level
            if len(crisis_indicators) >= 2:
                crisis_level = "SEVERE"
            elif len(crisis_indicators) == 1:
                crisis_level = "MODERATE"
            else:
                crisis_level = "NORMAL"
            
            return {
                "crisis_level": crisis_level,
                "indicators": crisis_indicators,
                "systemic_risk_score": systemic_risk,
                "confidence": min(systemic_risk + (1.0 - liquidity_score), 1.0)
            }
            
        except Exception as e:
            logger.error(f"Crisis detection failed: {e}")
            return {"crisis_level": "ERROR", "indicators": []}

    def _detect_manipulation_patterns(self, anti_herding: Dict, microstructure: Dict) -> Dict[str, Any]:
        """Detect potential market manipulation patterns"""
        try:
            volume_concentration = anti_herding.get('volume_concentration', 0.5)
            price_impact = microstructure.get('price_impact', 0.005)
            bid_ask_spread = microstructure.get('bid_ask_spread', 0.001)
            
            # Manipulation indicators
            manipulation_score = 0.0
            indicators = []
            
            # Unusual volume concentration
            if volume_concentration > 0.8:
                manipulation_score += 0.3
                indicators.append("VOLUME_CONCENTRATION")
            
            # Excessive price impact
            if price_impact > 0.01:  # 1% price impact
                manipulation_score += 0.25
                indicators.append("EXCESSIVE_PRICE_IMPACT")
            
            # Abnormal spread patterns
            if bid_ask_spread > 0.005:  # 0.5% spread
                manipulation_score += 0.2
                indicators.append("ABNORMAL_SPREADS")
            
            # Detection threshold
            manipulation_detected = manipulation_score > 0.5
            
            return {
                "manipulation_detected": manipulation_detected,
                "confidence": manipulation_score,
                "indicators": indicators,
                "manipulation_score": manipulation_score
            }
            
        except Exception as e:
            logger.error(f"Manipulation detection failed: {e}")
            return {"manipulation_detected": False, "confidence": 0.0}

    def _get_fallback_regime_data(self, symbol: str) -> Dict[str, Any]:
        """Get fallback regime data when enhanced analysis fails"""
        fallback_market_data = self._get_fallback_market_data(symbol)
        
        return {
            "market_data": fallback_market_data,
            "regime": {"regime_type": "UNKNOWN", "confidence": 0.0},
            "crisis": {"crisis_level": "UNKNOWN", "indicators": []},
            "manipulation": {"manipulation_detected": False, "confidence": 0.0},
            "symbol": symbol,
            "timestamp": datetime.now(),
            "enhanced_analysis": False
        }

    # =========================================================================
    # UTILITY AND HELPER METHODS
    # =========================================================================

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        try:
            if cache_key not in self.ticker_cache:
                return False
            
            cache_entry = self.ticker_cache[cache_key]
            age = (datetime.now() - cache_entry['timestamp']).total_seconds()
            
            return age < self.cache_ttl
            
        except Exception:
            return False

    def _calculate_enhanced_technical_indicators(self, price_history: List[float], 
                                               volume_history: List[float], 
                                               historical_data: List[Dict]) -> Dict[str, Any]:
        """Calculate enhanced technical indicators"""
        try:
            if not price_history or len(price_history) < 20:
                return {}
            
            prices = np.array(price_history)
            volumes = np.array(volume_history) if volume_history else np.zeros(len(prices))
            
            # Basic price metrics
            current_price = prices[-1]
            price_change_24h = (current_price / prices[-24] - 1) if len(prices) >= 24 else 0
            
            # Enhanced volatility measures
            returns = np.diff(prices) / prices[:-1]
            volatility_1d = np.std(returns[-24:]) if len(returns) >= 24 else np.std(returns)
            volatility_7d = np.std(returns[-168:]) if len(returns) >= 168 else volatility_1d
            
            # Moving averages
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else current_price
            sma_200 = np.mean(prices[-200:]) if len(prices) >= 200 else current_price
            
            # Enhanced RSI
            rsi = self._calculate_enhanced_rsi(prices)
            
            # MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(prices)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices)
            
            # Volume indicators
            volume_sma = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1] if len(volumes) > 0 else 0
            volume_ratio = volumes[-1] / volume_sma if volume_sma > 0 else 1.0
            
            # On-Balance Volume
            obv = self._calculate_obv(prices, volumes)
            
            return {
                'current_price': float(current_price),
                'price_change_24h': float(price_change_24h),
                'volatility_1d': float(volatility_1d),
                'volatility_7d': float(volatility_7d),
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'sma_200': float(sma_200),
                'rsi': float(rsi),
                'macd_line': float(macd_line),
                'macd_signal': float(macd_signal),
                'macd_histogram': float(macd_histogram),
                'bb_upper': float(bb_upper),
                'bb_middle': float(bb_middle),
                'bb_lower': float(bb_lower),
                'volume_ratio': float(volume_ratio),
                'obv': float(obv),
                'trend_strength': abs(float(sma_20 - sma_50) / sma_50) if sma_50 > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Enhanced technical indicators calculation failed: {e}")
            return {}

    def _calculate_enhanced_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate enhanced RSI with smoothing"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Use Wilder's smoothing
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception:
            return 50.0

    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD indicators"""
        try:
            if len(prices) < slow:
                return 0.0, 0.0, 0.0
            
            # Calculate EMAs
            ema_fast = self._calculate_ema(prices, fast)
            ema_slow = self._calculate_ema(prices, slow)
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line (EMA of MACD line)
            # For simplicity, use SMA approximation
            macd_values = [ema_fast - ema_slow for _ in range(len(prices))]
            macd_signal = np.mean(macd_values[-signal:]) if len(macd_values) >= signal else macd_line
            
            # Histogram
            macd_histogram = macd_line - macd_signal
            
            return macd_line, macd_signal, macd_histogram
            
        except Exception:
            return 0.0, 0.0, 0.0

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) == 0:
                return 0.0
            
            alpha = 2.0 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            
            return ema
            
        except Exception:
            return prices[-1] if len(prices) > 0 else 0.0

    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                current_price = prices[-1] if len(prices) > 0 else 0
                return current_price, current_price, current_price
            
            # Middle band (SMA)
            middle = np.mean(prices[-period:])
            
            # Standard deviation
            std = np.std(prices[-period:])
            
            # Upper and lower bands
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            
            return upper, middle, lower
            
        except Exception:
            current_price = prices[-1] if len(prices) > 0 else 0
            return current_price, current_price, current_price

    def _calculate_obv(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate On-Balance Volume"""
        try:
            if len(prices) != len(volumes) or len(prices) < 2:
                return 0.0
            
            obv = 0.0
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    obv += volumes[i]
                elif prices[i] < prices[i-1]:
                    obv -= volumes[i]
            
            return obv
            
        except Exception:
            return 0.0

    def _calculate_data_quality(self, symbol: str, historical_data: List[Dict]) -> float:
        """Calculate comprehensive data quality score"""
        try:
            quality_factors = []
            
            # Data completeness
            if historical_data and len(historical_data) >= 50:
                completeness = min(len(historical_data) / 100, 1.0)
                quality_factors.append(completeness * 0.3)
            
            # Data consistency (check for gaps or anomalies)
            if len(historical_data) > 1:
                # Check timestamp consistency
                timestamps = [data.get('close_time') for data in historical_data if data.get('close_time')]
                if len(timestamps) > 1:
                    time_gaps = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                               for i in range(1, len(timestamps))]
                    avg_gap = np.mean(time_gaps)
                    gap_consistency = 1.0 - min(np.std(time_gaps) / avg_gap, 1.0) if avg_gap > 0 else 0.5
                    quality_factors.append(gap_consistency * 0.25)
            
            # Price data validity
            prices = [data.get('close', 0) for data in historical_data if data.get('close')]
            if prices:
                # Check for zero or negative prices
                valid_prices = sum(1 for price in prices if price > 0)
                price_validity = valid_prices / len(prices)
                quality_factors.append(price_validity * 0.25)
                
                # Check for extreme price movements
                if len(prices) > 1:
                    price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] 
                                   for i in range(1, len(prices)) if prices[i-1] > 0]
                    if price_changes:
                        extreme_moves = sum(1 for change in price_changes if change > 0.5)  # >50% moves
                        stability = 1.0 - (extreme_moves / len(price_changes))
                        quality_factors.append(stability * 0.2)
            
            # Calculate composite score
            quality_score = sum(quality_factors) if quality_factors else 0.5
            return max(0.1, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Data quality calculation failed: {e}")
            return 0.5

    def _get_fallback_market_data(self, symbol: str) -> Dict[str, Any]:
        """Enhanced fallback market data generation"""
        try:
            # Generate more realistic mock data based on symbol
            if 'BTC' in symbol:
                base_price = 50000 + np.random.normal(0, 2000)
            elif 'ETH' in symbol:
                base_price = 3000 + np.random.normal(0, 200)
            elif 'ADA' in symbol:
                base_price = 0.5 + np.random.normal(0, 0.05)
            elif 'SOL' in symbol:
                base_price = 100 + np.random.normal(0, 10)
            else:
                base_price = 100 + np.random.normal(0, 10)
            
            # Generate realistic price history with trends
            price_history = []
            current_price = base_price
            
            for i in range(100):
                # Add realistic price movement
                daily_volatility = 0.02  # 2% daily volatility
                price_change = np.random.normal(0, daily_volatility)
                current_price *= (1 + price_change)
                price_history.append(max(current_price, 0.01))  # Prevent negative prices
            
            # Generate corresponding volume
            base_volume = 1000 + np.random.uniform(500, 2000)
            volume_history = [base_volume * np.random.uniform(0.5, 2.0) for _ in range(100)]
            
            # Calculate basic technical data
            technical_data = self._calculate_enhanced_technical_indicators(
                price_history, volume_history, []
            )
            
            return {
                'symbol': symbol,
                'timeframe': '1h',
                'current_price': price_history[-1],
                'price_history': price_history,
                'volume_history': volume_history,
                'timestamps': [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)],
                'technical_indicators': technical_data,
                'data_quality': 0.3,  # Lower quality for fallback
                'data_points': len(price_history),
                'source': 'enhanced_fallback',
                'timestamp': datetime.now(),
                'response_time': 0.1
            }
            
        except Exception as e:
            logger.error(f"Enhanced fallback data generation failed: {e}")
            # Emergency fallback
            return {
                'symbol': symbol,
                'current_price': 100.0,
                'price_history': [100.0],
                'volume_history': [1000.0],
                'data_quality': 0.1,
                'source': 'emergency_fallback',
                'timestamp': datetime.now()
            }

    def _update_avg_response_time(self, response_time: float):
        """Update average response time statistics"""
        try:
            current_avg = self.performance_stats['avg_response_time']
            total_requests = self.performance_stats['successful_requests']
            
            if total_requests > 0:
                self.performance_stats['avg_response_time'] = (
                    (current_avg * (total_requests - 1) + response_time) / total_requests
                )
            else:
                self.performance_stats['avg_response_time'] = response_time
                
        except Exception as e:
            logger.error(f"Response time update failed: {e}")

    # =========================================================================
    # STATUS AND MONITORING METHODS
    # =========================================================================

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # API performance
            api_stats = self.api_client.get_api_stats()
            
            # Data collection status
            collection_status = {
                'is_collecting': self.is_collecting,
                'symbols_tracked': len(self.symbols_to_collect),
                'symbols_with_data': sum(1 for symbol in self.symbols_to_collect 
                                       if symbol in self.price_data and len(self.price_data[symbol]) > 0)
            }
            
            # Cache status
            cache_status = {
                'cache_entries': len(self.ticker_cache),
                'cache_hit_rate': (self.performance_stats['cache_hits'] / 
                                 max(self.performance_stats['data_requests'], 1)) * 100
            }
            
            # Database status
            db_status = self.get_database_status()
            
            # Microstructure status
            microstructure_status = {
                'enabled': ENABLE_MICROSTRUCTURE,
                'symbols_with_microstructure': sum(1 for data in self.microstructure_data.values() if data)
            }
            
            # Anti-herding status
            anti_herding_status = {
                'symbols_with_analysis': sum(1 for data in self.anti_herding_data.values() if data),
                'active_signals': sum(len(self.anti_herding_engine.get_contrarian_signals(symbol)) 
                                    for symbol in self.symbols_to_collect)
            }
            
            return {
                'system_status': 'OPERATIONAL',
                'timestamp': datetime.now(),
                'api_performance': api_stats,
                'data_collection': collection_status,
                'cache_performance': cache_status,
                'database': db_status,
                'microstructure': microstructure_status,
                'anti_herding': anti_herding_status,
                'performance_stats': self.performance_stats
            }
            
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {
                'system_status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now()
            }

    def get_symbol_detailed_status(self, symbol: str) -> Dict[str, Any]:
        """Get detailed status for a specific symbol"""
        try:
            # Basic data status
            price_count = len(self.price_data.get(symbol, []))
            volume_count = len(self.volume_data.get(symbol, []))
            
            # Current data
            current_price = self.get_current_price(symbol)
            
            # Microstructure status
            microstructure = self.microstructure_data.get(symbol)
            microstructure_summary = {}
            if microstructure:
                microstructure_summary = {
                    'liquidity_score': microstructure.liquidity_score,
                    'market_efficiency': microstructure.market_efficiency,
                    'bid_ask_spread': microstructure.bid_ask_spread
                }
            
            # Anti-herding status
            anti_herding = self.anti_herding_data.get(symbol)
            anti_herding_summary = {}
            if anti_herding:
                anti_herding_summary = {
                    'herding_strength': anti_herding.herding_strength,
                    'crowding_score': anti_herding.crowding_score,
                    'contrarian_opportunity': anti_herding.contrarian_opportunity
                }
            
            # Signal status
            contrarian_signals = self.get_contrarian_signals(symbol)
            microstructure_signals = self.get_microstructure_signals(symbol)
            
            return {
                'symbol': symbol,
                'data_points': {
                    'price_history': price_count,
                    'volume_history': volume_count
                },
                'current_price': current_price,
                'microstructure': microstructure_summary,
                'anti_herding': anti_herding_summary,
                'active_signals': {
                    'contrarian': len(contrarian_signals),
                    'microstructure': len(microstructure_signals)
                },
                'data_quality': self.validate_data_quality(symbol, self.get_market_data(symbol)) if price_count > 0 else 0.0,
                'last_update': datetime.now(),
                'status': 'ACTIVE' if price_count > 0 else 'INACTIVE'
            }
            
        except Exception as e:
            logger.error(f"Detailed symbol status failed for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def clear_all_caches(self):
        """Clear all cached data and reset system"""
        try:
            with self.collection_lock:
                # Clear data storage
                self.price_data.clear()
                self.volume_data.clear()
                self.microstructure_data.clear()
                self.anti_herding_data.clear()
                self.ticker_cache.clear()
                
                # Reset performance stats
                self.performance_stats = {
                    'data_requests': 0,
                    'successful_requests': 0,
                    'cache_hits': 0,
                    'database_operations': 0,
                    'avg_response_time': 0.0
                }
                
                # Reinitialize data structures
                self._initialize_data_structures()
            
            logger.info("✅ All caches cleared and system reset")
            
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")

# =============================================================================
# FACTORY FUNCTIONS AND GLOBAL INSTANCE
# =============================================================================

def create_enhanced_data_manager() -> DataManager:
    """Factory function to create enhanced data manager"""
    return DataManager()

def test_enhanced_data_manager():
    """Test enhanced data manager functionality"""
    try:
        logger.info("🧪 Testing Enhanced DataManager...")
        
        # Create instance
        dm = DataManager()
        
        # Test basic functionality
        market_data = dm.get_market_data('BTCUSDT')
        assert market_data is not None, "Market data retrieval failed"
        
        # Test enhanced functionality
        enhanced_data = dm.get_market_data_with_microstructure('BTCUSDT')
        assert enhanced_data is not None, "Enhanced market data failed"
        
        # Test anti-herding analysis
        herding_data = dm.detect_herding_behavior('BTCUSDT')
        assert 'herding_detected' in herding_data, "Herding detection failed"
        
        # Test contrarian signals
        signals = dm.get_contrarian_signals('BTCUSDT')
        assert isinstance(signals, list), "Contrarian signals failed"
        
        # Test system status
        status = dm.get_comprehensive_status()
        assert status['system_status'] == 'OPERATIONAL', "System status check failed"
        
        logger.info("✅ Enhanced DataManager tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced DataManager test failed: {e}")
        return False

# Global instance for backward compatibility
data_manager = DataManager()

# Test on import (optional)
if __name__ == "__main__":
    test_enhanced_data_manager()

# =============================================================================
# END OF ENHANCED DATA MANAGER
# =============================================================================