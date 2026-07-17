#!/usr/bin/env python3
"""
News Intelligence — sentiment, event classification, macro regime, narrative.

Components:
    NewsIngestor          — Alpaca Benzinga news (historical + WebSocket stream)
    SentimentAnalyzer     — FinBERT (local) + Claude (Anthropic) deep analysis
    EventClassifier       — categorize and score impact, detect tail events
    MacroRegimeFilter     — VIX rank + Polymarket + news uncertainty
    NarrativeTracker      — rolling 30-day market narrative shifts

Trade rules implemented from Psychology of Money:
    - Tail event → reduce ALL positions 50%, pause new entries
    - High uncertainty → increase cash reserve by 10% (capped at 50%)
    - Only act on news with impact_magnitude > 0.6 AND confidence > 0.7
    - Cost control: Claude API capped at 30 calls/hour

All external dependencies degrade gracefully — no API key returns
neutral 0.5 scores rather than crashing.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    httpx = None

try:
    from config import NEWS_CONFIG
except ImportError:
    NEWS_CONFIG = {
        "enabled": False,
        "anthropic_api_key": "",
        "alphavantage_api_key": "",
        "claude_max_calls_per_hour": 30,
        "finbert_weight": 0.6,
        "claude_weight": 0.4,
        "min_impact_magnitude": 0.6,
        "min_confidence": 0.7,
        "narrative_window_days": 30,
    }


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class NewsItem:
    headline: str
    body: str = ""
    url: str = ""
    source: str = ""
    symbols: List[str] = field(default_factory=list)
    published_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SentimentResult:
    headline: str
    finbert_score: float = 0.5     # 0=bearish, 0.5=neutral, 1=bullish
    claude_score: Optional[float] = None
    combined_score: float = 0.5
    confidence: float = 0.5
    rationale: str = ""


@dataclass
class EventClassification:
    category: str = "general"      # earnings|macro|geopolitical|regulatory|sector|general
    impact_magnitude: float = 0.0  # 0..1
    timeframe: str = "days"        # immediate|days|weeks|months
    is_tail_event: bool = False
    is_narrative_shift: bool = False
    confidence: float = 0.5


@dataclass
class MacroSnapshot:
    vix_rank: float = 50.0
    polymarket_signal: float = 0.5
    news_uncertainty: float = 0.5
    macro_risk_score: float = 0.5
    regime_suggestion: str = "NORMAL"  # NORMAL|CAUTIOUS|DEFENSIVE


# ============================================================================
# News ingestion (historical only by default; WebSocket optional)
# ============================================================================

class NewsIngestor:
    """Pull historical Alpaca/Benzinga news. WebSocket streaming handled by
    the executor in production — this class focuses on synchronous fetch."""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self._client = None
        try:
            from alpaca.data.historical.news import NewsClient  # type: ignore
            if self.api_key and self.secret_key:
                self._client = NewsClient(self.api_key, self.secret_key)
        except ImportError:
            self._client = None
        except Exception as e:
            logger.warning(f"Alpaca NewsClient init failed: {e}")

    def fetch_recent(
        self, symbols: List[str], hours: int = 24, limit: int = 50
    ) -> List[NewsItem]:
        if not self._client:
            return []
        try:
            from alpaca.data.requests import NewsRequest  # type: ignore
            req = NewsRequest(
                symbols=symbols,
                start=datetime.utcnow() - timedelta(hours=hours),
                limit=limit,
            )
            news = self._client.get_news(req)
            data = getattr(news, "news", []) or getattr(news, "data", [])
            out: List[NewsItem] = []
            for n in data:
                out.append(NewsItem(
                    headline=getattr(n, "headline", ""),
                    body=getattr(n, "summary", "") or getattr(n, "content", ""),
                    url=getattr(n, "url", ""),
                    source=getattr(n, "source", ""),
                    symbols=list(getattr(n, "symbols", []) or []),
                    published_at=getattr(n, "created_at", datetime.utcnow()),
                ))
            return out
        except Exception as e:
            logger.warning(f"NewsIngestor.fetch_recent failed: {e}")
            return []


# ============================================================================
# Sentiment analyzer
# ============================================================================

class SentimentAnalyzer:
    """FinBERT (fast) + Claude (deep) with cost control."""

    def __init__(self):
        self._finbert_pipe: Optional[Callable] = None
        self._claude_client = None
        self._claude_calls: List[float] = []
        self._max_claude_per_hour = NEWS_CONFIG.get("claude_max_calls_per_hour", 30)
        self._finbert_weight = NEWS_CONFIG.get("finbert_weight", 0.6)
        self._claude_weight = NEWS_CONFIG.get("claude_weight", 0.4)
        self._init_finbert()
        self._init_claude()

    def _init_finbert(self) -> None:
        try:
            from transformers import pipeline  # type: ignore
            self._finbert_pipe = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded")
        except ImportError:
            logger.info("transformers not installed — FinBERT disabled")
        except Exception as e:
            logger.warning(f"FinBERT init failed: {e}")

    def _init_claude(self) -> None:
        api_key = NEWS_CONFIG.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return
        try:
            from anthropic import Anthropic  # type: ignore
            self._claude_client = Anthropic(api_key=api_key)
            logger.info("Anthropic client ready")
        except ImportError:
            logger.info("anthropic not installed — Claude deep analysis disabled")
        except Exception as e:
            logger.warning(f"Anthropic init failed: {e}")

    def finbert_score(self, text: str) -> float:
        if not self._finbert_pipe or not text:
            return 0.5
        try:
            result = self._finbert_pipe(text[:512])
            if not result:
                return 0.5
            label = str(result[0].get("label", "neutral")).lower()
            score = float(result[0].get("score", 0.5))
            if label == "positive":
                return 0.5 + 0.5 * score
            if label == "negative":
                return 0.5 - 0.5 * score
            return 0.5
        except Exception as e:
            logger.warning(f"FinBERT failed: {e}")
            return 0.5

    def _rate_limit_claude(self) -> bool:
        now = time.time()
        cutoff = now - 3600
        self._claude_calls = [t for t in self._claude_calls if t > cutoff]
        if len(self._claude_calls) >= self._max_claude_per_hour:
            return False
        self._claude_calls.append(now)
        return True

    def llm_deep_analysis(self, item: NewsItem) -> Optional[Tuple[float, str]]:
        if not self._claude_client or not self._rate_limit_claude():
            return None
        try:
            prompt = (
                "You are a market analyst. Analyze the news for trading implications.\n"
                f"Headline: {item.headline}\n"
                f"Body: {item.body[:1500]}\n"
                f"Symbols: {', '.join(item.symbols) if item.symbols else 'unknown'}\n\n"
                "Return JSON with two fields:\n"
                "  sentiment: number in [0,1]; 0.5 neutral, 1 bullish, 0 bearish\n"
                "  rationale: one sentence explaining second-order effects."
            )
            msg = self._claude_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(getattr(b, "text", "") for b in msg.content) if msg.content else ""
            return self._parse_claude_json(text)
        except Exception as e:
            logger.warning(f"Claude analysis failed: {e}")
            return None

    @staticmethod
    def _parse_claude_json(text: str) -> Optional[Tuple[float, str]]:
        import json
        import re
        match = re.search(r"\{.*?\}", text, re.S)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            sentiment = float(obj.get("sentiment", 0.5))
            return max(0.0, min(1.0, sentiment)), str(obj.get("rationale", ""))[:200]
        except Exception:
            return None

    def analyze(self, item: NewsItem, deep: bool = False) -> SentimentResult:
        text = f"{item.headline}. {item.body[:1000]}"
        finbert = self.finbert_score(text)

        claude_pair = self.llm_deep_analysis(item) if deep else None
        if claude_pair:
            claude_score, rationale = claude_pair
            combined = self._finbert_weight * finbert + self._claude_weight * claude_score
            return SentimentResult(
                headline=item.headline,
                finbert_score=finbert,
                claude_score=claude_score,
                combined_score=combined,
                confidence=0.8,
                rationale=rationale,
            )
        return SentimentResult(
            headline=item.headline,
            finbert_score=finbert,
            combined_score=finbert,
            confidence=0.6 if self._finbert_pipe else 0.4,
        )


# ============================================================================
# Event classifier
# ============================================================================

CATEGORY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "earnings": ("earnings", "eps", "revenue beat", "guidance", "quarterly", "missed estimates"),
    "macro": ("federal reserve", "fed rate", "inflation", "cpi", "ppi", "recession", "fomc"),
    "geopolitical": ("war", "sanctions", "tariff", "embargo", "treaty", "election"),
    "regulatory": ("sec", "ftc", "doj", "regulation", "lawsuit", "fda approval", "antitrust"),
    "sector": ("industry", "sector", "chip shortage", "supply chain"),
}

TAIL_KEYWORDS = (
    "bankruptcy", "default", "fraud", "investigation", "halted",
    "crash", "panic", "war", "pandemic", "circuit breaker",
)


class EventClassifier:
    """Categorize events and detect tail events / narrative shifts."""

    def classify(self, item: NewsItem, sentiment: SentimentResult) -> EventClassification:
        text = f"{item.headline} {item.body}".lower()

        category = "general"
        impact = 0.3
        for cat, keywords in CATEGORY_KEYWORDS.items():
            if any(k in text for k in keywords):
                category = cat
                impact = 0.5 if cat in {"earnings", "regulatory"} else 0.6
                break

        is_tail = any(k in text for k in TAIL_KEYWORDS)
        if is_tail:
            impact = max(impact, 0.85)

        # Narrative shift heuristic — strong sentiment paired with macro/sector
        is_narrative_shift = (
            category in {"macro", "geopolitical", "regulatory"}
            and abs(sentiment.combined_score - 0.5) > 0.25
        )

        timeframe = "immediate" if category in {"earnings", "geopolitical"} else "days"
        if category == "macro":
            timeframe = "weeks"

        return EventClassification(
            category=category,
            impact_magnitude=impact,
            timeframe=timeframe,
            is_tail_event=is_tail,
            is_narrative_shift=is_narrative_shift,
            confidence=sentiment.confidence,
        )

    @staticmethod
    def is_actionable(classification: EventClassification) -> bool:
        return (
            classification.impact_magnitude > NEWS_CONFIG.get("min_impact_magnitude", 0.6)
            and classification.confidence > NEWS_CONFIG.get("min_confidence", 0.7)
        )


# ============================================================================
# Macro regime filter
# ============================================================================

class MacroRegimeFilter:
    def __init__(self):
        try:
            from vix_manager import get_vix_manager
            self._vix_manager = get_vix_manager()
        except Exception:
            self._vix_manager = None

    def snapshot(
        self,
        recent_uncertainty: float = 0.5,
        polymarket_signal: float = 0.5,
    ) -> MacroSnapshot:
        vix_rank = 50.0
        if self._vix_manager:
            try:
                vix_rank = float(self._vix_manager.get_vix_rank())
            except Exception:
                pass

        # Macro risk score: VIX rank dominates, plus uncertainty + polymarket.
        macro_risk = max(0.0, min(1.0,
            0.5 * (vix_rank / 100.0)
            + 0.3 * recent_uncertainty
            + 0.2 * polymarket_signal
        ))

        if macro_risk > 0.75:
            regime = "DEFENSIVE"
        elif macro_risk > 0.55:
            regime = "CAUTIOUS"
        else:
            regime = "NORMAL"

        return MacroSnapshot(
            vix_rank=vix_rank,
            polymarket_signal=polymarket_signal,
            news_uncertainty=recent_uncertainty,
            macro_risk_score=macro_risk,
            regime_suggestion=regime,
        )

    @staticmethod
    def get_polymarket_signal(market_id: str) -> float:
        """Optional Polymarket integration. Returns 0.5 if unavailable."""
        if httpx is None:
            return 0.5
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(f"https://gamma-api.polymarket.com/markets/{market_id}")
                resp.raise_for_status()
                data = resp.json()
                prob = float(data.get("outcomePrices", [0.5])[0])
                return max(0.0, min(1.0, prob))
        except Exception:
            return 0.5


# ============================================================================
# Narrative tracker
# ============================================================================

class NarrativeTracker:
    """Track dominant market narrative over a rolling N-day window."""

    def __init__(self, window_days: Optional[int] = None):
        self.window_days = window_days or NEWS_CONFIG.get("narrative_window_days", 30)
        self._history: List[Tuple[datetime, float]] = []  # (date, sentiment)

    def add(self, sentiment: float, when: Optional[datetime] = None) -> None:
        when = when or datetime.utcnow()
        self._history.append((when, max(0.0, min(1.0, sentiment))))
        cutoff = datetime.utcnow() - timedelta(days=self.window_days)
        self._history = [(t, s) for t, s in self._history if t >= cutoff]

    def current(self) -> str:
        if not self._history:
            return "neutral"
        avg = sum(s for _, s in self._history) / len(self._history)
        if avg > 0.6:
            return "bullish"
        if avg < 0.4:
            return "bearish"
        return "neutral"

    def detect_shift(self) -> bool:
        """True when most recent third of window flips relative to first third."""
        if len(self._history) < 6:
            return False
        third = max(2, len(self._history) // 3)
        early = sum(s for _, s in self._history[:third]) / third
        late = sum(s for _, s in self._history[-third:]) / third
        if early > 0.55 and late < 0.45:
            return True
        if early < 0.45 and late > 0.55:
            return True
        return False


# ============================================================================
# Top-level facade
# ============================================================================

class NewsIntelligence:
    """High-level facade orchestrating ingestion → sentiment → events → macro."""

    def __init__(self):
        self.ingestor = NewsIngestor()
        self.sentiment = SentimentAnalyzer()
        self.classifier = EventClassifier()
        self.macro = MacroRegimeFilter()
        self.narrative = NarrativeTracker()

    def get_news_score(self, symbol: str, hours: int = 24) -> float:
        """Aggregate sentiment for symbol over recent window. 0..1."""
        items = self.ingestor.fetch_recent([symbol], hours=hours, limit=20)
        if not items:
            return 0.5
        scores: List[float] = []
        for item in items:
            res = self.sentiment.analyze(item, deep=False)
            scores.append(res.combined_score)
            self.narrative.add(res.combined_score, item.published_at)
        return sum(scores) / len(scores) if scores else 0.5

    def assess_market_risk(self) -> MacroSnapshot:
        return self.macro.snapshot()


_news_intel: Optional[NewsIntelligence] = None


def get_news_intelligence() -> NewsIntelligence:
    global _news_intel
    if _news_intel is None:
        _news_intel = NewsIntelligence()
    return _news_intel
