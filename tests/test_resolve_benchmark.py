"""Guard the prod benchmark path against silent-null regressions.

The alpha column went dark for 10 days because ``compute_benchmark`` returned
None every run and *nothing* in the prod path complained — the equity curve just
quietly carried ``alpha_pct: null`` until the 2-week checkpoint would have read a
useless window. ``resolve_benchmark`` closes that gap: when the benchmark can't
be computed it must shout into the log (WARNING), not fail silently.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

import run_trading_system as rts


def _spy(dates, closes) -> pd.DataFrame:
    return pd.DataFrame({"close": closes}, index=pd.to_datetime(dates))


def test_resolve_benchmark_returns_pct_for_valid_snapshot():
    snapshot = {rts.BENCHMARK_SYMBOL: _spy(
        ["2026-06-05", "2026-06-12"], [600.0, 660.0],
    )}
    # anchor the inception to the snapshot so this stays independent of the
    # module-level RACE_INCEPTION constant.
    bm, bm_pct = rts.resolve_benchmark(snapshot, inception=date(2026, 6, 5))
    assert bm is not None
    assert bm_pct is not None
    assert bm_pct == bm["return_pct"]


def test_resolve_benchmark_warns_and_returns_none_when_symbol_missing(caplog):
    with caplog.at_level(logging.WARNING, logger="horse_race"):
        bm, bm_pct = rts.resolve_benchmark({}, inception=date(2026, 6, 5))
    assert bm is None
    assert bm_pct is None
    assert any(r.levelno == logging.WARNING for r in caplog.records), \
        "a missing benchmark must log a WARNING, not fail silently"


def test_resolve_benchmark_warns_when_all_bars_predate_inception(caplog):
    # SPY is present but every bar is before inception -> compute_benchmark None.
    # This is the subtle case the in_snapshot=True log line would NOT catch.
    snapshot = {rts.BENCHMARK_SYMBOL: _spy(["2026-06-04"], [500.0])}
    with caplog.at_level(logging.WARNING, logger="horse_race"):
        bm, bm_pct = rts.resolve_benchmark(snapshot, inception=date(2026, 6, 5))
    assert bm is None
    assert bm_pct is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_resolve_benchmark_defaults_to_race_inception():
    # Called without an explicit inception, it uses the module constant.
    snapshot = {rts.BENCHMARK_SYMBOL: _spy(
        ["2026-06-05", "2026-06-12"], [600.0, 660.0],
    )}
    bm, _ = rts.resolve_benchmark(snapshot)
    assert bm is not None
