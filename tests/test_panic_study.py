"""F1 panic-study: selección de evento condicional-al-precio, forward returns, placebo."""
import pandas as pd

from events.panic_study import (
    forward_returns,
    p90,
    panic_dates,
    percentile_of,
    placebo,
)


def _series():
    prices = [100.0] * 10 + [95.0, 97.0, 99.0, 100.0] + [100.0] * 10
    idx = pd.bdate_range("2020-01-01", periods=len(prices))
    return pd.DataFrame({"close": prices}, index=idx), idx


def test_panic_day_detected_at_threshold():
    df, idx = _series()
    assert panic_dates(df, -3.0) == [idx[10]]


def test_harder_threshold_finds_nothing():
    df, _ = _series()
    assert panic_dates(df, -10.0) == []


def test_forward_return_is_long_only():
    df, idx = _series()
    r = forward_returns(df, [idx[10]], 1)
    assert len(r) == 1 and abs(r[0] - (97.0 / 95.0 - 1)) < 1e-9


def test_window_past_end_is_dropped_not_crashed():
    df, idx = _series()
    assert forward_returns(df, [idx[-1]], 5) == []


def test_p90_reads_right_tail():
    assert p90([0.01, 0.02, 0.03, 0.10]) == 10.0
    assert p90([]) == 0.0


def test_placebo_is_deterministic_and_excludes_events():
    df, idx = _series()
    a = placebo(df, [idx[10]], 1, 5, draws=30, seed=7)
    b = placebo(df, [idx[10]], 1, 5, draws=30, seed=7)
    assert a["exp_dist"] == b["exp_dist"] and len(a["exp_dist"]) == 30


def test_percentile_of_bounds():
    assert percentile_of(99.0, [0.0, 1.0, 2.0]) == 100.0
    assert percentile_of(-1.0, [0.0, 1.0, 2.0]) == 0.0
    assert percentile_of(1.0, []) == 0.0
