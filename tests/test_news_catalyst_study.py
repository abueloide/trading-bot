"""Tests del harness R2 — auditoría backtest del news reactor."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from events.news_backfill import PAGE, walk_news
from events.news_catalyst_study import (catalyst_days, conditioned_placebo,
                                        long_returns, percentile_of)


# ------------------------------------------------------------ paginación

class _FakeNewsClient:
    """Emula la API real: ignora page_token, filtra por `end`, tope de 50."""

    def __init__(self, articles):
        self.articles = articles
        self.calls = 0

    def get_news(self, req):
        self.calls += 1
        sel = [a for a in self.articles
               if datetime.fromisoformat(a["created_at"].replace("Z", "+00:00")) < req.end]
        return {"news": sel[:PAGE], "next_page_token": None}


def _articles(n):
    return [{"id": i, "created_at": f"2020-01-0{1 + i // 60}T{23 - i % 24:02d}:{59 - i % 60:02d}:00Z",
             "headline": f"h{i}", "symbols": ["X"]} for i in range(n)]


def _walk(monkeypatch, arts):
    fake = _FakeNewsClient(sorted(arts, key=lambda a: a["created_at"], reverse=True))
    monkeypatch.setattr("events.news_backfill._client", lambda: fake)
    got = list(walk_news("X", datetime(2019, 1, 1, tzinfo=timezone.utc),
                         datetime(2021, 1, 1, tzinfo=timezone.utc)))
    return got, fake


def test_pagination_walks_past_the_first_page(monkeypatch):
    """El bug que dejó al reactor sin backtest: quedarse en la página 1."""
    got, fake = _walk(monkeypatch, _articles(130))
    assert len(got) == 130
    assert fake.calls > 1, "no paginó"


def test_pagination_yields_no_duplicates(monkeypatch):
    got, _ = _walk(monkeypatch, _articles(130))
    assert len({g["id"] for g in got}) == len(got)


def test_pagination_terminates_when_cursor_does_not_advance(monkeypatch):
    """Artículos con el mismo timestamp no deben colgar el walk."""
    same = [{"id": i, "created_at": "2020-01-01T12:00:00Z", "headline": "h",
             "symbols": ["X"]} for i in range(200)]
    got, fake = _walk(monkeypatch, same)
    assert fake.calls < 10, "giró en falso"
    assert len(got) == PAGE


# ------------------------------------------------------------ eventos

def test_multiple_headlines_same_day_are_one_event():
    """Episodios, no titulares (lección F1)."""
    arts = [
        {"created_at": "2020-05-01T12:00:00Z", "headline": "Acme beats estimates", "symbols": ["ACME"]},
        {"created_at": "2020-05-01T18:00:00Z", "headline": "Acme raises guidance", "symbols": ["ACME"]},
    ]
    bull, _ = catalyst_days("ACME", arts)
    assert list(bull) == ["2020-05-01"]
    assert bull["2020-05-01"] == "earnings_beat"


def test_multi_ticker_roundup_is_not_an_event():
    arts = [{"created_at": "2020-05-01T12:00:00Z", "headline": "Acme beats estimates",
             "symbols": ["ACME", "BCME"]}]
    bull, ctrl = catalyst_days("ACME", arts)
    assert bull == {}
    assert ctrl == ["2020-05-01"], "el día sigue siendo día-con-noticia (piscina de control)"


def test_bearish_headline_is_control_not_event():
    arts = [{"created_at": "2020-05-01T12:00:00Z", "headline": "Acme misses estimates",
             "symbols": ["ACME"]}]
    bull, ctrl = catalyst_days("ACME", arts)
    assert bull == {} and ctrl == ["2020-05-01"]


def test_control_pool_excludes_event_days():
    arts = [
        {"created_at": "2020-05-01T12:00:00Z", "headline": "Acme beats estimates", "symbols": ["ACME"]},
        {"created_at": "2020-05-02T12:00:00Z", "headline": "Acme names new CFO", "symbols": ["ACME"]},
    ]
    bull, ctrl = catalyst_days("ACME", arts)
    assert "2020-05-01" not in ctrl and ctrl == ["2020-05-02"]


# ------------------------------------------------------------ retornos

@pytest.fixture
def prices():
    idx = pd.to_datetime(["2020-05-01", "2020-05-04", "2020-05-05", "2020-05-06", "2020-05-07"])
    return pd.DataFrame({"close": [100.0, 110.0, 121.0, 121.0, 100.0]}, index=idx)


def test_long_return_is_close_to_close(prices):
    (_, sess, r), = long_returns(prices, ["2020-05-01"], 1)
    assert sess == "2020-05-01" and r == pytest.approx(0.10)


def test_weekend_headline_enters_next_session(prices):
    """Sin lookahead: un titular del sábado entra el lunes, no el viernes."""
    (day, sess, r), = long_returns(prices, ["2020-05-02"], 1)
    assert day == "2020-05-02" and sess == "2020-05-04" and r == pytest.approx(0.10)


def test_event_without_full_window_is_dropped(prices):
    assert long_returns(prices, ["2020-05-07"], 1) == []
    assert long_returns(prices, ["2020-05-05"], 3) == []


def test_stale_headline_beyond_four_days_is_dropped(prices):
    assert long_returns(prices, ["2020-04-20"], 1) == []


# ------------------------------------------------------------ placebo

def test_conditioned_placebo_needs_enough_control_days(prices):
    out = conditioned_placebo(prices, ["2020-05-01"], 1, n=5)
    assert out["exp_dist"] == [] and out["pool"] == 1


def test_conditioned_placebo_is_deterministic(prices):
    ctrl = ["2020-05-01", "2020-05-04", "2020-05-05"]
    a = conditioned_placebo(prices, ctrl, 1, n=2, draws=20)
    b = conditioned_placebo(prices, ctrl, 1, n=2, draws=20)
    assert a["exp_dist"] == b["exp_dist"], "misma semilla debe dar la misma distribución"


def test_percentile_of():
    assert percentile_of(5.0, [1.0, 2.0, 3.0, 4.0]) == 100.0
    assert percentile_of(0.0, [1.0, 2.0, 3.0, 4.0]) == 0.0
    assert percentile_of(2.5, [1.0, 2.0, 3.0, 4.0]) == 50.0
