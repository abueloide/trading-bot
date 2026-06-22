from __future__ import annotations

import pytest

from live.risk_metrics import compute_risk_metrics, format_risk_table


def _snap(date_str, strategy, equity):
    return {"date": date_str, "strategy": strategy, "equity": equity}


def test_empty_snapshots_returns_empty():
    assert compute_risk_metrics([]) == {}


def test_max_drawdown_peak_to_trough():
    # One horse rising to 110 then falling to 99: drawdown = 99/110 - 1 = -10%.
    snaps = [
        _snap("2026-06-15", "rsi_mr", 100.0),
        _snap("2026-06-16", "rsi_mr", 110.0),
        _snap("2026-06-17", "rsi_mr", 99.0),
        _snap("2026-06-18", "rsi_mr", 104.5),
    ]
    metrics = compute_risk_metrics(snaps)
    assert metrics["rsi_mr"]["max_drawdown_pct"] == pytest.approx(-10.0)
    assert metrics["rsi_mr"]["n_days"] == 4


def test_monotonic_rise_has_zero_drawdown():
    snaps = [
        _snap("2026-06-15", "up", 100.0),
        _snap("2026-06-16", "up", 105.0),
        _snap("2026-06-17", "up", 112.0),
    ]
    metrics = compute_risk_metrics(snaps)
    assert metrics["up"]["max_drawdown_pct"] == pytest.approx(0.0)


def test_volatility_is_stdev_of_daily_returns():
    # Returns: +10%, then -10% (99/110). Sample stdev of [0.10, -0.10] ~= 0.1414.
    snaps = [
        _snap("2026-06-15", "rsi_mr", 100.0),
        _snap("2026-06-16", "rsi_mr", 110.0),
        _snap("2026-06-17", "rsi_mr", 99.0),
    ]
    vol = compute_risk_metrics(snaps)["rsi_mr"]["volatility_pct"]
    assert vol == pytest.approx(14.1421, abs=1e-3)


def test_single_day_has_no_drawdown_or_vol():
    metrics = compute_risk_metrics([_snap("2026-06-15", "solo", 100.0)])
    assert metrics["solo"]["max_drawdown_pct"] == pytest.approx(0.0)
    assert metrics["solo"]["volatility_pct"] is None
    assert metrics["solo"]["n_days"] == 1


def test_records_sorted_by_date_regardless_of_input_order():
    # Out-of-order input must still compute the curve chronologically.
    snaps = [
        _snap("2026-06-17", "x", 99.0),
        _snap("2026-06-15", "x", 100.0),
        _snap("2026-06-16", "x", 110.0),
    ]
    metrics = compute_risk_metrics(snaps)
    assert metrics["x"]["max_drawdown_pct"] == pytest.approx(-10.0)


def test_zero_or_missing_equity_is_skipped():
    # A bad row (equity 0 or absent) must not crash or poison the series.
    snaps = [
        _snap("2026-06-15", "x", 100.0),
        {"date": "2026-06-16", "strategy": "x"},  # missing equity
        _snap("2026-06-17", "x", 0.0),  # zero equity
        _snap("2026-06-18", "x", 90.0),
    ]
    metrics = compute_risk_metrics(snaps)
    assert metrics["x"]["n_days"] == 2
    assert metrics["x"]["max_drawdown_pct"] == pytest.approx(-10.0)


def test_format_table_lists_each_strategy():
    metrics = {
        "rsi_mr": {"max_drawdown_pct": -10.0, "volatility_pct": 14.14, "n_days": 3},
        "up": {"max_drawdown_pct": 0.0, "volatility_pct": None, "n_days": 3},
    }
    out = format_risk_table(metrics)
    assert "rsi_mr" in out
    assert "up" in out
    assert "-10.00" in out
    # None volatility renders as a placeholder, not a crash.
    assert "n/a" in out


def test_format_table_empty_metrics():
    assert "no equity curve" in format_risk_table({}).lower()


def test_contiguous_weekday_curve_has_no_gap():
    # Mon→Tue→Wed: consecutive trading days, no missing weekday slots.
    snaps = [
        _snap("2026-06-15", "x", 100.0),  # Mon
        _snap("2026-06-16", "x", 101.0),  # Tue
        _snap("2026-06-17", "x", 102.0),  # Wed
    ]
    assert compute_risk_metrics(snaps)["x"]["gap_days"] == 0


def test_weekend_is_not_counted_as_a_gap():
    # Fri→Mon skips Sat/Sun only: the bot runs L–V, so this is contiguous.
    snaps = [
        _snap("2026-06-19", "x", 100.0),  # Fri
        _snap("2026-06-22", "x", 101.0),  # Mon
    ]
    assert compute_risk_metrics(snaps)["x"]["gap_days"] == 0


def test_skipped_weekday_is_counted_as_a_gap():
    # Mon→Wed skips Tue: one missing trading day → a real gap.
    snaps = [
        _snap("2026-06-15", "x", 100.0),  # Mon
        _snap("2026-06-17", "x", 102.0),  # Wed (Tue missing)
    ]
    assert compute_risk_metrics(snaps)["x"]["gap_days"] == 1


def test_multiple_skipped_weekdays_accumulate():
    # Mon→Fri with Tue/Wed/Thu all missing → 3 missing trading days.
    # Holiday-free week (06-19 is Juneteenth, so this week avoids it).
    snaps = [
        _snap("2026-06-22", "x", 100.0),  # Mon
        _snap("2026-06-26", "x", 102.0),  # Fri
    ]
    assert compute_risk_metrics(snaps)["x"]["gap_days"] == 3


def test_single_day_has_no_gap():
    assert compute_risk_metrics([_snap("2026-06-15", "x", 100.0)])["x"]["gap_days"] == 0


def test_market_holiday_is_not_a_gap():
    # Thu 06-18 → Mon 06-22 spans Juneteenth (Fri 06-19, NYSE closed) + weekend.
    # The curve is contiguous in *trading* days, so no gap. This is the exact
    # false positive prod showed on 2026-06-22 before holiday awareness landed.
    snaps = [
        _snap("2026-06-18", "x", 100.0),  # Thu
        _snap("2026-06-22", "x", 101.0),  # Mon (06-19 Juneteenth, no trading)
    ]
    assert compute_risk_metrics(snaps)["x"]["gap_days"] == 0
