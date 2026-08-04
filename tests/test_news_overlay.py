"""Tests for the news-sentiment overlay (AlphaVantage).

Two units under test:
  - apply_news_overlay: PURE ranking filter (no I/O).
  - fetch_sentiment: AlphaVantage parsing, with the HTTP getter injected so the
    suite never touches the network.
"""
from __future__ import annotations

import pytest

from live.news_overlay import (
    NEG_SENTIMENT_FLOOR,
    apply_news_overlay,
    fetch_sentiment,
)


# --------------------------------------------------------------------------- #
# apply_news_overlay — pure
# --------------------------------------------------------------------------- #

def test_overlay_keeps_top_n_when_all_neutral():
    ranked = ["AAA", "BBB", "CCC", "DDD"]
    # no sentiment data at all → nobody penalized, momentum order preserved
    assert apply_news_overlay(ranked, {}, n=2) == ["AAA", "BBB"]


def test_overlay_vetoes_negative_and_refills_from_next_ranked():
    ranked = ["AAA", "BBB", "CCC", "DDD"]
    sentiment = {"AAA": -0.30, "BBB": 0.10, "CCC": 0.05}
    # AAA is below the floor → dropped; survivors keep momentum order
    assert apply_news_overlay(ranked, sentiment, n=2) == ["BBB", "CCC"]


def test_overlay_missing_ticker_is_neutral_not_penalized():
    ranked = ["AAA", "BBB"]
    sentiment = {"AAA": 0.20}  # BBB has no coverage
    assert apply_news_overlay(ranked, sentiment, n=2) == ["AAA", "BBB"]


def test_overlay_exactly_at_floor_survives():
    ranked = ["AAA", "BBB"]
    sentiment = {"AAA": NEG_SENTIMENT_FLOOR}
    assert apply_news_overlay(ranked, sentiment, n=2) == ["AAA", "BBB"]


def test_overlay_can_return_fewer_than_n_when_everything_vetoed():
    ranked = ["AAA", "BBB"]
    sentiment = {"AAA": -0.9, "BBB": -0.9}
    assert apply_news_overlay(ranked, sentiment, n=2) == []


def test_overlay_empty_ranked():
    assert apply_news_overlay([], {"AAA": 0.5}, n=3) == []


# --------------------------------------------------------------------------- #
# fetch_sentiment — parsing with injected getter
# --------------------------------------------------------------------------- #

def _feed(*items):
    return {"items": str(len(items)), "feed": list(items)}


def _article(ticker_scores):
    """ticker_scores: {ticker: (sentiment_score, relevance_score)}"""
    return {
        "title": "x",
        "ticker_sentiment": [
            {
                "ticker": t,
                "ticker_sentiment_score": str(s),
                "relevance_score": str(r),
            }
            for t, (s, r) in ticker_scores.items()
        ],
    }


def test_fetch_sentiment_relevance_weighted_mean():
    payload = _feed(
        _article({"AAA": (0.4, 1.0)}),
        _article({"AAA": (0.0, 1.0)}),
    )
    out = fetch_sentiment(["AAA"], api_key="k", get_json=lambda url: payload)
    assert out["AAA"] == pytest.approx(0.2)


def test_fetch_sentiment_weights_by_relevance():
    # high-relevance positive should dominate low-relevance negative
    payload = _feed(
        _article({"AAA": (1.0, 1.0)}),
        _article({"AAA": (-1.0, 0.0)}),
    )
    out = fetch_sentiment(["AAA"], api_key="k", get_json=lambda url: payload)
    assert out["AAA"] == pytest.approx(1.0)


def test_fetch_sentiment_splits_multiple_tickers():
    payload = _feed(_article({"AAA": (0.5, 1.0), "BBB": (-0.5, 1.0)}))
    out = fetch_sentiment(["AAA", "BBB"], api_key="k", get_json=lambda url: payload)
    assert out["AAA"] == pytest.approx(0.5)
    assert out["BBB"] == pytest.approx(-0.5)


def test_fetch_sentiment_rate_limited_returns_empty():
    note = {"Information": "rate limit, 25 requests/day"}
    out = fetch_sentiment(["AAA"], api_key="k", get_json=lambda url: note)
    assert out == {}


def test_fetch_sentiment_no_key_returns_empty():
    called = []
    out = fetch_sentiment(["AAA"], api_key="", get_json=lambda url: called.append(url))
    assert out == {}
    assert called == []  # never hit the network without a key


def test_fetch_sentiment_empty_tickers_returns_empty():
    out = fetch_sentiment([], api_key="k", get_json=lambda url: pytest.fail("no call"))
    assert out == {}


def test_fetch_sentiment_getter_error_fails_soft():
    def boom(url):
        raise RuntimeError("network down")

    out = fetch_sentiment(["AAA"], api_key="k", get_json=boom)
    assert out == {}  # overlay degrades to pure momentum, never crashes the cycle
