from __future__ import annotations

import pytest

from live.checkpoint_report import (
    EDGE_MIN_DAYS,
    build_checkpoint,
    classify_edge,
    format_checkpoint,
)


def _snap(date_str, strategy, equity, return_pct, alpha_pct, benchmark_pct=0.0):
    return {
        "date": date_str,
        "strategy": strategy,
        "equity": equity,
        "return_pct": return_pct,
        "alpha_pct": alpha_pct,
        "benchmark_pct": benchmark_pct,
    }


# ---- classify_edge: descriptive verdicts only, never an automated decision ----

def test_classify_inconclusive_when_too_few_days():
    v = classify_edge(alpha_pct=5.0, max_drawdown_pct=-2.0, n_days=2, min_days=5)
    assert "inconclusive" in v.lower()


def test_classify_no_benchmark_when_alpha_missing():
    v = classify_edge(alpha_pct=None, max_drawdown_pct=-2.0, n_days=10, min_days=5)
    assert "no benchmark" in v.lower()


def test_classify_no_edge_when_alpha_not_positive():
    v = classify_edge(alpha_pct=-0.5, max_drawdown_pct=-2.0, n_days=10, min_days=5)
    assert "no edge" in v.lower()


def test_classify_edge_candidate_when_alpha_positive_and_shallow_dd():
    v = classify_edge(alpha_pct=3.0, max_drawdown_pct=-4.0, n_days=10, min_days=5)
    assert "candidate" in v.lower()


def test_classify_flags_risk_when_alpha_positive_but_deep_dd():
    v = classify_edge(alpha_pct=3.0, max_drawdown_pct=-25.0, n_days=10, min_days=5)
    assert "risk" in v.lower()


# ---- build_checkpoint: latest cut per horse + merged risk + verdict ----

def test_empty_snapshots_returns_empty():
    assert build_checkpoint([]) == []


def test_uses_latest_record_per_strategy():
    snaps = [
        _snap("2026-06-15", "rsi_mr", 100.0, 0.0, 0.0),
        _snap("2026-06-16", "rsi_mr", 110.0, 10.0, 7.0),
    ]
    rows = build_checkpoint(snaps)
    assert len(rows) == 1
    assert rows[0]["strategy"] == "rsi_mr"
    assert rows[0]["equity"] == pytest.approx(110.0)
    assert rows[0]["return_pct"] == pytest.approx(10.0)
    assert rows[0]["alpha_pct"] == pytest.approx(7.0)


def test_merges_risk_metrics_from_curve():
    # Rise to 110 then back to 99 → max drawdown -10%, 3 curve days.
    snaps = [
        _snap("2026-06-15", "x", 100.0, 0.0, 0.0),
        _snap("2026-06-16", "x", 110.0, 10.0, 5.0),
        _snap("2026-06-17", "x", 99.0, -1.0, -3.0),
    ]
    row = build_checkpoint(snaps)[0]
    assert row["max_drawdown_pct"] == pytest.approx(-10.0)
    assert row["n_days"] == 3
    assert row["volatility_pct"] is not None


def test_sorted_by_alpha_descending():
    snaps = [
        _snap("2026-06-16", "low", 102.0, 2.0, 1.0),
        _snap("2026-06-16", "high", 108.0, 8.0, 6.0),
    ]
    rows = build_checkpoint(snaps)
    assert [r["strategy"] for r in rows] == ["high", "low"]


def test_each_row_carries_a_verdict():
    snaps = [_snap("2026-06-16", "x", 110.0, 10.0, 7.0)]
    row = build_checkpoint(snaps, min_days=1)[0]
    assert "verdict" in row
    assert isinstance(row["verdict"], str) and row["verdict"]


# ---- format_checkpoint: readable single table + honest footer ----

def test_format_lists_strategies_and_verdict():
    snaps = [
        _snap("2026-06-16", "rsi_mr", 110.0, 10.0, 7.0),
        _snap("2026-06-16", "confirmed_mr", 98.0, -2.0, -5.0),
    ]
    out = format_checkpoint(build_checkpoint(snaps, min_days=1))
    assert "rsi_mr" in out
    assert "confirmed_mr" in out
    # Footer must make clear the decision is the operator's, not automated.
    assert "operator" in out.lower() or "decisión" in out.lower()


def test_format_empty_is_friendly():
    out = format_checkpoint([])
    assert "no equity curve" in out.lower()


def test_edge_min_days_is_a_sane_default():
    assert isinstance(EDGE_MIN_DAYS, int) and EDGE_MIN_DAYS >= 2


# ---- gap detection: a curve with skipped trading days must not pass silently ----

def test_row_carries_gap_days():
    # Mon→Wed skips Tue → one missing trading day surfaced on the row.
    snaps = [
        _snap("2026-06-15", "x", 100.0, 0.0, 0.0),  # Mon
        _snap("2026-06-17", "x", 102.0, 2.0, 1.0),  # Wed (Tue missing)
    ]
    row = build_checkpoint(snaps)[0]
    assert row["gap_days"] == 1


def test_contiguous_curve_has_zero_gap_days():
    snaps = [
        _snap("2026-06-15", "x", 100.0, 0.0, 0.0),  # Mon
        _snap("2026-06-16", "x", 101.0, 1.0, 0.5),  # Tue
    ]
    assert build_checkpoint(snaps)[0]["gap_days"] == 0


def test_format_warns_when_curve_has_gaps():
    snaps = [
        _snap("2026-06-15", "x", 100.0, 0.0, 0.0),  # Mon
        _snap("2026-06-17", "x", 102.0, 2.0, 1.0),  # Wed (Tue missing)
    ]
    out = format_checkpoint(build_checkpoint(snaps, min_days=1))
    assert "gap" in out.lower()


def test_format_no_gap_warning_on_contiguous_curve():
    snaps = [
        _snap("2026-06-15", "x", 100.0, 0.0, 0.0),  # Mon
        _snap("2026-06-16", "x", 101.0, 1.0, 0.5),  # Tue
    ]
    out = format_checkpoint(build_checkpoint(snaps, min_days=1))
    assert "missing trading day" not in out.lower()
