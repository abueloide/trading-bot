#!/usr/bin/env python3
"""
Geopolitical Intelligence Engine — macro-political regime detection and
capital allocation for multi-market trading.

This module is the core differentiator: it reads geopolitical signals
(trade policy, central bank decisions, elections, conflicts) and outputs
a capital allocation recommendation across US stocks, crypto, and MX.

Components:
    GeopoliticalEvent       — structured event with impact scoring
    EventSourceManager      — pulls events from multiple free sources
    GeopoliticalAnalyzer    — Claude-powered deep analysis of events
    RegimeEngine            — determines current macro regime
    AllocationEngine        — maps regime → capital allocation

Trade rules:
    - Regime changes trigger rebalance (max 1x/day)
    - Tail events (war, default, pandemic) → CRISIS mode immediately
    - High uncertainty → increase stables/cash allocation
    - Claude API capped at 20 calls/hour for cost control (~$3/month)

All external dependencies degrade gracefully.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    httpx = None

try:
    from config import NEWS_CONFIG
except ImportError:
    NEWS_CONFIG = {"anthropic_api_key": "", "claude_max_calls_per_hour": 30}


# ============================================================================
# Configuration
# ============================================================================

GEOPOLITICAL_CONFIG = {
    "claude_max_calls_per_hour": 20,
    "event_lookback_hours": 48,
    "regime_hold_hours": 24,          # min time before regime can flip again
    "rebalance_cooldown_hours": 24,   # max 1 rebalance per day
    "min_events_for_regime_change": 2, # need 2+ confirming events
    "tail_event_override": True,       # tail events skip cooldown
}


# ============================================================================
# Data classes
# ============================================================================

class MacroRegime(str, Enum):
    RISK_ON = "RISK_ON"
    CAUTIOUS = "CAUTIOUS"
    RISK_OFF = "RISK_OFF"
    CRISIS = "CRISIS"


class EventCategory(str, Enum):
    TRADE_POLICY = "trade_policy"       # tariffs, sanctions, trade deals
    CENTRAL_BANK = "central_bank"       # Fed, Banxico, ECB rate decisions
    ELECTION = "election"               # elections, political transitions
    CONFLICT = "conflict"               # wars, military actions
    REGULATION = "regulation"           # crypto regulation, SEC, financial law
    FISCAL_POLICY = "fiscal_policy"     # stimulus, tax reform, spending
    COMMODITY = "commodity"             # oil shocks, supply disruptions
    PANDEMIC = "pandemic"               # health crises
    DEFAULT_RISK = "default_risk"       # sovereign/corporate defaults


@dataclass
class GeopoliticalEvent:
    headline: str
    category: EventCategory
    severity: float = 0.5              # 0..1
    direction: float = 0.5            # 0=bearish, 0.5=neutral, 1=bullish
    is_tail_event: bool = False
    affected_markets: List[str] = field(default_factory=lambda: ["US", "CRYPTO", "MX"])
    confidence: float = 0.5
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    rationale: str = ""
    # Market-specific impact overrides (-1 to 1, 0 = neutral)
    us_impact: float = 0.0
    crypto_impact: float = 0.0
    mx_impact: float = 0.0


@dataclass
class RegimeState:
    regime: MacroRegime = MacroRegime.CAUTIOUS
    confidence: float = 0.5
    since: datetime = field(default_factory=datetime.utcnow)
    trigger_events: List[str] = field(default_factory=list)
    allocation: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Event source manager — free sources only
# ============================================================================

class EventSourceManager:
    """Pull geopolitical events from free public sources."""

    def __init__(self):
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl = 1800  # 30 min

    def fetch_all_events(self, hours: int = 48) -> List[GeopoliticalEvent]:
        """Aggregate events from all sources."""
        events: List[GeopoliticalEvent] = []
        events.extend(self._fetch_economic_calendar())
        events.extend(self._fetch_news_rss())
        events.extend(self._fetch_fed_calendar())
        events.extend(self._fetch_banxico())
        return events

    def _cached_get(self, key: str, url: str, headers: Optional[Dict] = None) -> Optional[Any]:
        if httpx is None:
            return None
        cached = self._cache.get(key)
        if cached and time.time() - cached[0] < self._cache_ttl:
            return cached[1]
        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(url, headers=headers or {})
                resp.raise_for_status()
                data = resp.json() if "json" in resp.headers.get("content-type", "") else resp.text
                self._cache[key] = (time.time(), data)
                return data
        except Exception as e:
            logger.warning(f"Event source fetch failed [{key}]: {e}")
            return None

    # --- Economic calendar (Trading Economics RSS proxy via free API) ---
    def _fetch_economic_calendar(self) -> List[GeopoliticalEvent]:
        """Fetch upcoming high-impact economic events from FMP free tier."""
        data = self._cached_get(
            "fmp_calendar",
            f"https://financialmodelingprep.com/api/v3/economic_calendar"
            f"?apikey={os.getenv('FMP_API_KEY', 'demo')}"
        )
        if not data or not isinstance(data, list):
            return []

        events = []
        for item in data[:30]:
            impact = str(item.get("impact", "")).lower()
            if impact not in ("high", "medium"):
                continue

            headline = f"{item.get('event', '')} ({item.get('country', '')})"
            category = self._classify_economic_event(item.get("event", ""))

            actual = item.get("actual")
            estimate = item.get("estimate")
            direction = 0.5
            if actual is not None and estimate is not None:
                try:
                    diff = float(actual) - float(estimate)
                    direction = 0.5 + min(max(diff / 10.0, -0.5), 0.5)
                except (ValueError, TypeError):
                    pass

            severity = 0.8 if impact == "high" else 0.5
            country = str(item.get("country", "")).upper()
            affected = self._country_to_markets(country)

            events.append(GeopoliticalEvent(
                headline=headline,
                category=category,
                severity=severity,
                direction=direction,
                affected_markets=affected,
                confidence=0.7 if actual is not None else 0.5,
                source="fmp_calendar",
                us_impact=0.0,
                crypto_impact=0.0,
                mx_impact=0.0,
            ))
        return events

    def _fetch_news_rss(self) -> List[GeopoliticalEvent]:
        """Fetch geopolitical headlines from free news APIs."""
        api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
        if not api_key:
            return []
        data = self._cached_get(
            "av_news",
            f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
            f"&topics=economy_macro,financial_markets&apikey={api_key}"
        )
        if not data or not isinstance(data, dict):
            return []

        events = []
        for item in data.get("feed", [])[:20]:
            headline = item.get("title", "")
            sentiment = float(item.get("overall_sentiment_score", 0))
            # Map -1..1 to 0..1
            direction = 0.5 + sentiment / 2.0

            category = self._classify_headline(headline)
            if category is None:
                continue  # not geopolitically relevant

            events.append(GeopoliticalEvent(
                headline=headline,
                category=category,
                severity=0.6,
                direction=direction,
                confidence=0.6,
                source="alphavantage_news",
            ))
        return events

    def _fetch_fed_calendar(self) -> List[GeopoliticalEvent]:
        """Check for upcoming FOMC decisions."""
        data = self._cached_get(
            "fmp_fomc",
            f"https://financialmodelingprep.com/api/v3/economic_calendar"
            f"?apikey={os.getenv('FMP_API_KEY', 'demo')}"
        )
        if not data or not isinstance(data, list):
            return []

        events = []
        for item in data:
            event_name = str(item.get("event", "")).lower()
            if "federal" not in event_name and "fomc" not in event_name:
                continue
            events.append(GeopoliticalEvent(
                headline=f"FOMC: {item.get('event', '')}",
                category=EventCategory.CENTRAL_BANK,
                severity=0.9,
                direction=0.5,
                affected_markets=["US", "CRYPTO", "MX"],
                confidence=0.9,
                source="fmp_fomc",
            ))
        return events

    def _fetch_banxico(self) -> List[GeopoliticalEvent]:
        """Fetch Banxico rate decisions from their public API."""
        data = self._cached_get(
            "banxico_rate",
            "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43783/datos/oportuno"
            "?token=" + os.getenv("BANXICO_TOKEN", ""),
        )
        if not data:
            return []
        try:
            series = data.get("bmx", {}).get("series", [{}])[0]
            datos = series.get("datos", [])
            if not datos:
                return []
            latest = datos[-1]
            rate = float(latest.get("dato", "0").replace(",", ""))
            return [GeopoliticalEvent(
                headline=f"Banxico tasa objetivo: {rate}%",
                category=EventCategory.CENTRAL_BANK,
                severity=0.8,
                direction=0.5,
                affected_markets=["MX"],
                confidence=0.9,
                source="banxico_api",
                mx_impact=0.0,
            )]
        except Exception as e:
            logger.warning(f"Banxico parse failed: {e}")
            return []

    # --- Classification helpers ---

    TRADE_KEYWORDS = ("tariff", "trade war", "trade deal", "sanctions", "embargo",
                      "nearshoring", "t-mec", "usmca", "import duty", "export ban",
                      "arancel", "trade agreement", "trade deficit")
    CENTRAL_BANK_KEYWORDS = ("fed ", "federal reserve", "fomc", "rate hike", "rate cut",
                             "interest rate", "banxico", "ecb", "boj", "quantitative",
                             "tightening", "easing", "inflation target")
    ELECTION_KEYWORDS = ("election", "vote", "ballot", "inauguration", "impeachment",
                         "political transition", "reform bill", "legislature")
    CONFLICT_KEYWORDS = ("war", "military", "invasion", "missile", "airstrike",
                         "ceasefire", "conflict", "troops", "nuclear")
    REGULATION_KEYWORDS = ("sec ", "regulation", "crypto ban", "stablecoin law",
                           "financial regulation", "antitrust", "ftc")
    FISCAL_KEYWORDS = ("stimulus", "tax reform", "spending bill", "deficit",
                       "infrastructure bill", "budget", "debt ceiling")
    CRISIS_KEYWORDS = ("default", "bankruptcy", "crash", "panic", "circuit breaker",
                       "bank run", "liquidity crisis", "contagion", "collapse",
                       "pandemic", "outbreak", "lockdown")

    def _classify_headline(self, headline: str) -> Optional[EventCategory]:
        text = headline.lower()
        if any(k in text for k in self.CRISIS_KEYWORDS):
            return EventCategory.DEFAULT_RISK
        if any(k in text for k in self.TRADE_KEYWORDS):
            return EventCategory.TRADE_POLICY
        if any(k in text for k in self.CENTRAL_BANK_KEYWORDS):
            return EventCategory.CENTRAL_BANK
        if any(k in text for k in self.CONFLICT_KEYWORDS):
            return EventCategory.CONFLICT
        if any(k in text for k in self.ELECTION_KEYWORDS):
            return EventCategory.ELECTION
        if any(k in text for k in self.REGULATION_KEYWORDS):
            return EventCategory.REGULATION
        if any(k in text for k in self.FISCAL_KEYWORDS):
            return EventCategory.FISCAL_POLICY
        return None

    def _classify_economic_event(self, event_name: str) -> EventCategory:
        text = event_name.lower()
        if any(k in text for k in ("rate", "fomc", "fed", "banxico", "ecb")):
            return EventCategory.CENTRAL_BANK
        if any(k in text for k in ("cpi", "inflation", "ppi", "gdp", "employment", "nonfarm")):
            return EventCategory.FISCAL_POLICY
        if any(k in text for k in ("trade", "tariff", "export", "import")):
            return EventCategory.TRADE_POLICY
        return EventCategory.FISCAL_POLICY

    @staticmethod
    def _country_to_markets(country: str) -> List[str]:
        country = country.upper()
        if country in ("US", "USA", "UNITED STATES"):
            return ["US", "CRYPTO"]
        if country in ("MX", "MEX", "MEXICO"):
            return ["MX"]
        if country in ("CN", "CHINA"):
            return ["US", "MX"]  # China trade impacts both
        return ["US", "CRYPTO", "MX"]


# ============================================================================
# Geopolitical analyzer — Claude-powered deep analysis
# ============================================================================

class GeopoliticalAnalyzer:
    """Use Claude to deeply analyze geopolitical events for trading impact."""

    def __init__(self):
        self._client = None
        self._calls: List[float] = []
        self._max_per_hour = GEOPOLITICAL_CONFIG["claude_max_calls_per_hour"]
        self._init_claude()

    def _init_claude(self) -> None:
        api_key = NEWS_CONFIG.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return
        try:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)
        except ImportError:
            logger.info("anthropic not installed — geopolitical deep analysis disabled")
        except Exception as e:
            logger.warning(f"Anthropic init failed: {e}")

    def _rate_limit(self) -> bool:
        now = time.time()
        self._calls = [t for t in self._calls if t > now - 3600]
        if len(self._calls) >= self._max_per_hour:
            return False
        self._calls.append(now)
        return True

    def analyze_events(self, events: List[GeopoliticalEvent]) -> List[GeopoliticalEvent]:
        """Enrich events with Claude analysis for per-market impact."""
        if not self._client or not events:
            return events

        # Only analyze high-severity events to save API calls
        high_events = [e for e in events if e.severity >= 0.6]
        if not high_events:
            return events

        # Batch up to 5 events per call
        batch = high_events[:5]
        if not self._rate_limit():
            logger.info("Claude rate limit hit — skipping geopolitical analysis")
            return events

        headlines = "\n".join(
            f"- [{e.category.value}] {e.headline}" for e in batch
        )

        prompt = f"""You are a macro-political trading strategist. Analyze these geopolitical events
and determine their impact on three markets: US stocks, cryptocurrency, and Mexican stocks.

Events:
{headlines}

For EACH event, return a JSON array with objects containing:
- "headline": the event headline (abbreviated)
- "regime_signal": one of "RISK_ON", "CAUTIOUS", "RISK_OFF", "CRISIS"
- "is_tail_event": true/false
- "us_impact": number from -1 (very bearish) to 1 (very bullish)
- "crypto_impact": number from -1 to 1
- "mx_impact": number from -1 to 1
- "rationale": one sentence explaining the second-order effects

Think about:
- How tariffs/trade policy affect nearshoring to Mexico
- How rate decisions affect crypto (inverse correlation with real rates)
- How geopolitical conflicts drive flight to Bitcoin vs USD
- Mexico-specific factors: peso strength, Banxico, energy policy, T-MEC

Return ONLY the JSON array, no markdown."""

        try:
            msg = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(getattr(b, "text", "") for b in msg.content) if msg.content else ""
            parsed = self._parse_response(text)

            for i, analysis in enumerate(parsed):
                if i >= len(batch):
                    break
                event = batch[i]
                event.us_impact = float(analysis.get("us_impact", 0))
                event.crypto_impact = float(analysis.get("crypto_impact", 0))
                event.mx_impact = float(analysis.get("mx_impact", 0))
                event.rationale = str(analysis.get("rationale", ""))[:200]
                event.is_tail_event = bool(analysis.get("is_tail_event", False))

                regime_str = str(analysis.get("regime_signal", "CAUTIOUS")).upper()
                if regime_str in MacroRegime.__members__:
                    event.direction = {
                        "RISK_ON": 0.8, "CAUTIOUS": 0.5,
                        "RISK_OFF": 0.3, "CRISIS": 0.1,
                    }.get(regime_str, 0.5)

        except Exception as e:
            logger.warning(f"Claude geopolitical analysis failed: {e}")

        return events

    @staticmethod
    def _parse_response(text: str) -> List[Dict]:
        match = re.search(r"\[.*\]", text, re.S)
        if not match:
            return []
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return []


# ============================================================================
# Regime engine — determines current macro regime from events
# ============================================================================

class RegimeEngine:
    """Determine macro regime from aggregated geopolitical events."""

    def __init__(self):
        self._current = RegimeState()
        self._history: List[RegimeState] = []

    @property
    def current_regime(self) -> RegimeState:
        return self._current

    def update(self, events: List[GeopoliticalEvent]) -> RegimeState:
        """Update regime based on latest events."""
        if not events:
            return self._current

        # Check tail events first — immediate override
        tail_events = [e for e in events if e.is_tail_event]
        if tail_events and GEOPOLITICAL_CONFIG["tail_event_override"]:
            return self._set_regime(
                MacroRegime.CRISIS,
                confidence=0.95,
                triggers=[e.headline for e in tail_events],
            )

        # Aggregate event signals
        regime_votes: Dict[MacroRegime, float] = {r: 0 for r in MacroRegime}
        for event in events:
            if event.severity < 0.4:
                continue

            weight = event.severity * event.confidence
            if event.direction > 0.7:
                regime_votes[MacroRegime.RISK_ON] += weight
            elif event.direction > 0.55:
                regime_votes[MacroRegime.CAUTIOUS] += weight * 0.5
                regime_votes[MacroRegime.RISK_ON] += weight * 0.5
            elif event.direction > 0.4:
                regime_votes[MacroRegime.CAUTIOUS] += weight
            elif event.direction > 0.25:
                regime_votes[MacroRegime.RISK_OFF] += weight
            else:
                regime_votes[MacroRegime.CRISIS] += weight * 0.5
                regime_votes[MacroRegime.RISK_OFF] += weight * 0.5

        # Pick winning regime
        if not any(v > 0 for v in regime_votes.values()):
            return self._current

        winner = max(regime_votes, key=regime_votes.get)
        total_weight = sum(regime_votes.values())
        confidence = regime_votes[winner] / total_weight if total_weight > 0 else 0.5

        # Cooldown check — don't flip-flop
        hours_since_change = (datetime.utcnow() - self._current.since).total_seconds() / 3600
        min_hold = GEOPOLITICAL_CONFIG["regime_hold_hours"]
        if winner != self._current.regime and hours_since_change < min_hold:
            # Only override cooldown for escalation (RISK_ON→CRISIS ok, CRISIS→RISK_ON blocked)
            severity_order = [MacroRegime.RISK_ON, MacroRegime.CAUTIOUS,
                              MacroRegime.RISK_OFF, MacroRegime.CRISIS]
            if severity_order.index(winner) <= severity_order.index(self._current.regime):
                logger.info(f"Regime change blocked by cooldown ({hours_since_change:.1f}h < {min_hold}h)")
                return self._current

        triggers = [e.headline for e in sorted(events, key=lambda e: e.severity, reverse=True)[:3]]
        return self._set_regime(winner, confidence, triggers)

    def _set_regime(self, regime: MacroRegime, confidence: float,
                    triggers: List[str]) -> RegimeState:
        if regime != self._current.regime:
            logger.info(f"REGIME CHANGE: {self._current.regime.value} → {regime.value} "
                       f"(confidence: {confidence:.2f})")
            self._history.append(self._current)

        self._current = RegimeState(
            regime=regime,
            confidence=confidence,
            since=datetime.utcnow(),
            trigger_events=triggers,
            allocation=AllocationEngine.get_allocation(regime, confidence),
        )
        return self._current


# ============================================================================
# Allocation engine — maps regime to capital allocation
# ============================================================================

class AllocationEngine:
    """Map macro regime to target capital allocation across markets."""

    # Base allocations per regime
    ALLOCATIONS: Dict[MacroRegime, Dict[str, float]] = {
        MacroRegime.RISK_ON: {
            "us_stocks": 0.40,
            "crypto": 0.30,
            "mx_stocks": 0.20,
            "stables_cash": 0.10,
        },
        MacroRegime.CAUTIOUS: {
            "us_stocks": 0.30,
            "crypto": 0.15,
            "mx_stocks": 0.15,
            "stables_cash": 0.40,
        },
        MacroRegime.RISK_OFF: {
            "us_stocks": 0.15,    # defensive sectors only
            "crypto": 0.10,       # BTC only, no alts
            "mx_stocks": 0.05,
            "stables_cash": 0.70,
        },
        MacroRegime.CRISIS: {
            "us_stocks": 0.00,
            "crypto": 0.10,       # BTC as digital gold
            "mx_stocks": 0.00,
            "stables_cash": 0.90,
        },
    }

    @classmethod
    def get_allocation(cls, regime: MacroRegime, confidence: float) -> Dict[str, float]:
        """Get target allocation, blended with CAUTIOUS when confidence is low."""
        base = dict(cls.ALLOCATIONS[regime])

        # Low confidence → blend toward CAUTIOUS
        if confidence < 0.7:
            cautious = cls.ALLOCATIONS[MacroRegime.CAUTIOUS]
            blend = confidence  # 0.5 confidence = 50/50 blend
            for key in base:
                base[key] = base[key] * blend + cautious[key] * (1 - blend)

        # Normalize to 1.0
        total = sum(base.values())
        if total > 0:
            base = {k: round(v / total, 3) for k, v in base.items()}

        return base

    @classmethod
    def adjust_for_events(
        cls,
        allocation: Dict[str, float],
        events: List[GeopoliticalEvent],
    ) -> Dict[str, float]:
        """Fine-tune allocation based on per-market event impacts."""
        if not events:
            return allocation

        adj = dict(allocation)

        # Average per-market impacts from analyzed events
        us_impacts = [e.us_impact for e in events if e.us_impact != 0]
        crypto_impacts = [e.crypto_impact for e in events if e.crypto_impact != 0]
        mx_impacts = [e.mx_impact for e in events if e.mx_impact != 0]

        def avg(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        # Apply shifts (max ±10% per market)
        shift_cap = 0.10
        us_shift = max(-shift_cap, min(shift_cap, avg(us_impacts) * 0.15))
        crypto_shift = max(-shift_cap, min(shift_cap, avg(crypto_impacts) * 0.15))
        mx_shift = max(-shift_cap, min(shift_cap, avg(mx_impacts) * 0.15))

        adj["us_stocks"] = max(0, adj.get("us_stocks", 0) + us_shift)
        adj["crypto"] = max(0, adj.get("crypto", 0) + crypto_shift)
        adj["mx_stocks"] = max(0, adj.get("mx_stocks", 0) + mx_shift)

        # Absorb excess into stables/cash
        non_cash = adj["us_stocks"] + adj["crypto"] + adj["mx_stocks"]
        adj["stables_cash"] = max(0.10, 1.0 - non_cash)

        # Renormalize
        total = sum(adj.values())
        if total > 0:
            adj = {k: round(v / total, 3) for k, v in adj.items()}

        return adj


# ============================================================================
# Top-level facade
# ============================================================================

class GeopoliticalIntelligence:
    """High-level facade: sources → analysis → regime → allocation."""

    def __init__(self):
        self.sources = EventSourceManager()
        self.analyzer = GeopoliticalAnalyzer()
        self.regime_engine = RegimeEngine()
        self._last_events: List[GeopoliticalEvent] = []
        self._last_update: Optional[datetime] = None

    def update(self) -> RegimeState:
        """Full pipeline: fetch events → analyze → update regime."""
        events = self.sources.fetch_all_events(
            hours=GEOPOLITICAL_CONFIG["event_lookback_hours"]
        )
        logger.info(f"Fetched {len(events)} geopolitical events")

        # Enrich with Claude analysis
        events = self.analyzer.analyze_events(events)

        # Update regime
        state = self.regime_engine.update(events)

        # Fine-tune allocation with per-market impacts
        state.allocation = AllocationEngine.adjust_for_events(
            state.allocation, events
        )

        self._last_events = events
        self._last_update = datetime.utcnow()

        logger.info(
            f"Regime: {state.regime.value} | Confidence: {state.confidence:.2f} | "
            f"Allocation: {state.allocation}"
        )
        return state

    @property
    def current_state(self) -> RegimeState:
        return self.regime_engine.current_regime

    @property
    def last_events(self) -> List[GeopoliticalEvent]:
        return self._last_events

    def get_telegram_summary(self) -> str:
        """Format current state for Telegram bot."""
        state = self.current_state
        alloc = state.allocation

        regime_emoji = {
            MacroRegime.RISK_ON: "🟢",
            MacroRegime.CAUTIOUS: "🟡",
            MacroRegime.RISK_OFF: "🟠",
            MacroRegime.CRISIS: "🔴",
        }

        lines = [
            f"{regime_emoji.get(state.regime, '⚪')} Macro Regime: {state.regime.value}",
            f"Confidence: {state.confidence:.0%}",
            f"Since: {state.since.strftime('%Y-%m-%d %H:%M')} UTC",
            "",
            "Target Allocation:",
            f"  US Stocks: {alloc.get('us_stocks', 0):.0%}",
            f"  Crypto:    {alloc.get('crypto', 0):.0%}",
            f"  MX Stocks: {alloc.get('mx_stocks', 0):.0%}",
            f"  Cash/Stbl: {alloc.get('stables_cash', 0):.0%}",
        ]

        if state.trigger_events:
            lines.append("")
            lines.append("Triggers:")
            for t in state.trigger_events[:3]:
                lines.append(f"  • {t[:80]}")

        return "\n".join(lines)


# ============================================================================
# Module-level singleton
# ============================================================================

_geo_intel: Optional[GeopoliticalIntelligence] = None


def get_geopolitical_intelligence() -> GeopoliticalIntelligence:
    global _geo_intel
    if _geo_intel is None:
        _geo_intel = GeopoliticalIntelligence()
    return _geo_intel
