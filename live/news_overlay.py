"""News-sentiment overlay for the momentum horse (AlphaVantage NEWS_SENTIMENT).

The fourth horse is plain momentum with one extra gate: drop ranked names that
carry materially negative recent news, then take the top-n survivors. The point
is differentiation — the other three horses are pure price; this one also reads
the tape.

Two concerns, deliberately split:

  fetch_sentiment(tickers)        -> {ticker: score}   (I/O, fails soft)
      One batched AlphaVantage call covers many tickers. Free tier is 25
      requests/day, so the live cycle (one call for ~15 names) fits easily.
      Any failure (no key, rate limit, network) returns {} and the overlay
      degrades to pure momentum — it must never crash a trading cycle.

  apply_news_overlay(ranked, ...) -> [ticker, ...]     (PURE, offline-testable)
      A name with no coverage is treated as neutral (kept), so we only ever
      veto on actual negative evidence, never on silence.

AlphaVantage returns, per article, a `ticker_sentiment` list with a
`ticker_sentiment_score` in [-1, 1] and a `relevance_score` in [0, 1]. The
per-ticker score is the relevance-weighted mean of those article scores.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Mapping, Sequence

logger = logging.getLogger(__name__)

# A name scoring below this (relevance-weighted) is vetoed from the basket.
# Slightly below zero: tolerate genuinely neutral coverage, cut clear negatives.
NEG_SENTIMENT_FLOOR: float = -0.05

_AV_ENDPOINT = "https://www.alphavantage.co/query"
_HTTP_TIMEOUT = 15  # seconds
# AlphaVantage caps the comma-joined ticker list; stay well under it per call.
_MAX_TICKERS_PER_CALL = 50

JsonGetter = Callable[[str], object]


def apply_news_overlay(
    ranked: Sequence[str],
    sentiment: Mapping[str, float],
    n: int,
    floor: float = NEG_SENTIMENT_FLOOR,
) -> List[str]:
    """Filter a momentum-ranked list by news sentiment, keeping the top-n.

    `ranked` is in priority order (strongest momentum first). Names absent from
    `sentiment` are neutral and survive; only scores below `floor` are dropped.
    """
    survivors = [sym for sym in ranked if sentiment.get(sym, 0.0) >= floor]
    return survivors[:n]


def _default_get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "trading-bot/1.0"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def fetch_sentiment(
    tickers: Sequence[str],
    *,
    api_key: str | None = None,
    limit: int = 200,
    get_json: JsonGetter | None = None,
) -> Dict[str, float]:
    """Relevance-weighted recent-news sentiment per ticker, or {} on any failure.

    `get_json` is injectable so the parsing logic is unit-testable without the
    network; in production it defaults to a plain urllib GET.
    """
    key = api_key if api_key is not None else os.getenv("ALPHAVANTAGE_API_KEY", "")
    symbols = [t for t in tickers if t]
    if not key or not symbols:
        return {}

    getter = get_json or _default_get_json
    result: Dict[str, float] = {}
    for batch in _chunks(symbols, _MAX_TICKERS_PER_CALL):
        params = urllib.parse.urlencode(
            {
                "function": "NEWS_SENTIMENT",
                "tickers": ",".join(batch),
                "limit": limit,
                "apikey": key,
            }
        )
        url = f"{_AV_ENDPOINT}?{params}"
        try:
            payload = getter(url)
        except Exception as exc:  # network, decode, timeout — degrade to momentum
            logger.warning("news sentiment fetch failed (%s); skipping overlay", exc)
            return {}
        parsed = _parse_feed(payload, set(batch))
        if parsed is None:  # rate-limit / error envelope — give up cleanly
            return {}
        result.update(parsed)
    return result


def _parse_feed(payload: object, wanted: set[str]) -> Dict[str, float] | None:
    """Aggregate an AlphaVantage NEWS_SENTIMENT payload into {ticker: score}.

    Returns None when the payload is a rate-limit/error envelope (no `feed`),
    so the caller can distinguish "no signal" from "data present, all neutral".
    """
    if not isinstance(payload, dict):
        return None
    if "feed" not in payload:
        # AlphaVantage signals throttling/errors via Information/Note/Error Message.
        logger.warning("news sentiment envelope without feed: %s", _envelope(payload))
        return None

    weighted_sum: Dict[str, float] = {}
    weight_total: Dict[str, float] = {}
    for article in payload.get("feed", []):
        for ts in article.get("ticker_sentiment", []):
            sym = ts.get("ticker")
            if sym not in wanted:
                continue
            try:
                score = float(ts.get("ticker_sentiment_score"))
                relevance = float(ts.get("relevance_score"))
            except (TypeError, ValueError):
                continue
            weighted_sum[sym] = weighted_sum.get(sym, 0.0) + score * relevance
            weight_total[sym] = weight_total.get(sym, 0.0) + relevance

    out: Dict[str, float] = {}
    for sym, total in weight_total.items():
        if total > 0:
            out[sym] = weighted_sum[sym] / total
    return out


def _envelope(payload: Mapping[str, object]) -> str:
    keys = ("Information", "Note", "Error Message")
    return "; ".join(str(payload[k]) for k in keys if k in payload) or "unknown"


def _chunks(items: Sequence[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]
