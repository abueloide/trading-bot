from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from live.benchmark import compute_benchmark


def _spy(dates, closes) -> pd.DataFrame:
    return pd.DataFrame({"close": closes}, index=pd.to_datetime(dates))


def test_benchmark_return_from_inception_to_last_bar():
    df = _spy(
        ["2026-06-04", "2026-06-05", "2026-06-08", "2026-06-12"],
        [500.0, 600.0, 610.0, 660.0],
    )
    bm = compute_benchmark({"SPY": df}, "SPY", date(2026, 6, 5))
    # anchored at the 2026-06-05 close (600), ends at 660 -> +10%
    assert bm is not None
    assert bm["return_pct"] == pytest.approx(10.0)
    assert bm["start_close"] == pytest.approx(600.0)
    assert bm["end_close"] == pytest.approx(660.0)
    assert bm["start_date"] == "2026-06-05"


def test_benchmark_anchors_to_first_bar_on_or_after_inception():
    # inception falls on a weekend/holiday; anchor to the next session.
    df = _spy(["2026-06-05", "2026-06-08", "2026-06-09"], [600.0, 600.0, 630.0])
    bm = compute_benchmark({"SPY": df}, "SPY", date(2026, 6, 6))
    assert bm["start_date"] == "2026-06-08"
    assert bm["return_pct"] == pytest.approx(5.0)


def test_benchmark_missing_symbol_returns_none():
    assert compute_benchmark({}, "SPY", date(2026, 6, 5)) is None


def test_benchmark_all_bars_before_inception_returns_none():
    df = _spy(["2026-06-04"], [500.0])
    assert compute_benchmark({"SPY": df}, "SPY", date(2026, 6, 5)) is None
