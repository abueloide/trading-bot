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
# The verdict reads the *alpha series* (chronological, may carry null entries
# from before the benchmark was wired) — not a single latest value — so it can
# gate on real alpha observations and judge stability, both of which feed the
# operator's irreversible real-money call.

def test_classify_inconclusive_when_too_few_alpha_days():
    v = classify_edge(alpha_series=[5.0, 5.0], max_drawdown_pct=-2.0, min_days=5)
    assert "inconclusive" in v.lower()


def test_classify_no_benchmark_when_no_real_alpha():
    v = classify_edge(alpha_series=[None, None, None], max_drawdown_pct=-2.0, min_days=5)
    assert "no benchmark" in v.lower()


def test_classify_counts_real_alpha_days_ignoring_null_prefix():
    # Equity curve ran longer, but alpha only became real for 3 days → still
    # inconclusive at min 5. This is the exact prod shape after the SPY-null bug:
    # null alpha 06-05..06-15, real alpha from 06-16. Gating on equity-days would
    # overstate confidence; gating on real alpha-days does not.
    v = classify_edge(
        alpha_series=[None, None, None, 1.0, 2.0, 3.0],
        max_drawdown_pct=-2.0,
        min_days=5,
    )
    assert "inconclusive" in v.lower()


def test_classify_no_edge_when_latest_alpha_not_positive():
    v = classify_edge(
        alpha_series=[1.0, 2.0, 1.0, 0.5, -0.5], max_drawdown_pct=-2.0, min_days=5
    )
    assert "no edge" in v.lower()


def test_classify_edge_candidate_when_alpha_positive_stable_shallow_dd():
    v = classify_edge(
        alpha_series=[2.0, 2.5, 3.0, 2.8, 3.2], max_drawdown_pct=-4.0, min_days=5
    )
    assert "candidate" in v.lower()


def test_classify_flags_unstable_when_alpha_dipped_nonpositive():
    # Latest alpha is positive, but it went ≤0 within the window → not the
    # "positivo y estable" the PLAN requires. A single last-day bounce is not edge.
    v = classify_edge(
        alpha_series=[1.0, -0.5, 0.2, 1.5, 2.0], max_drawdown_pct=-3.0, min_days=5
    )
    assert "unstable" in v.lower()


def test_classify_flags_risk_when_alpha_positive_but_deep_dd():
    v = classify_edge(
        alpha_series=[2.0, 2.5, 3.0, 2.8, 3.2], max_drawdown_pct=-25.0, min_days=5
    )
    assert "risk" in v.lower()


def test_classify_flags_within_noise_when_alpha_positive_but_jumpy():
    # Every alpha observation is > 0 (so not "unstable") and the drawdown is
    # shallow, yet the alpha swings so wildly that its mean is swamped by its own
    # day-to-day dispersion. Over 3 horses × ~10 days, a horse can clear SPY by
    # luck; an alpha whose signal is smaller than its noise is exactly that case.
    # It must NOT read as a clean "edge candidate" — discipline over return.
    v = classify_edge(
        alpha_series=[0.1, 4.0, 0.2, 3.5, 0.15], max_drawdown_pct=-3.0, min_days=5
    )
    assert "noise" in v.lower()
    assert "candidate" not in v.lower()


def test_classify_candidate_survives_when_signal_beats_noise():
    # Consistently positive AND tight: the mean alpha dwarfs its dispersion, so
    # the within-noise gate must not fire — this is the genuine candidate shape.
    v = classify_edge(
        alpha_series=[3.0, 3.2, 2.9, 3.1, 3.05], max_drawdown_pct=-3.0, min_days=5
    )
    assert "candidate" in v.lower()
    assert "noise" not in v.lower()


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


def test_verdict_gates_on_real_alpha_days_not_equity_days():
    # 5 equity days but alpha real only on the last 2 → inconclusive at min 5.
    # The curve length must not buy confidence the alpha history can't back.
    snaps = [
        _snap("2026-06-15", "x", 100.0, 0.0, None),
        _snap("2026-06-16", "x", 101.0, 1.0, None),
        _snap("2026-06-17", "x", 102.0, 2.0, None),
        _snap("2026-06-18", "x", 103.0, 3.0, 1.0),
        _snap("2026-06-19", "x", 104.0, 4.0, 2.0),
    ]
    row = build_checkpoint(snaps, min_days=5)[0]
    assert row["n_days"] == 5  # equity-curve length unchanged
    assert row["alpha_days"] == 2  # only two real alpha observations
    assert "inconclusive" in row["verdict"].lower()


def test_verdict_flags_unstable_alpha_through_build():
    snaps = [
        _snap("2026-06-15", "x", 100.0, 0.0, 1.0),
        _snap("2026-06-16", "x", 99.0, -1.0, -0.5),  # dipped ≤0 mid-window
        _snap("2026-06-17", "x", 101.0, 1.0, 0.4),
        _snap("2026-06-18", "x", 103.0, 3.0, 1.6),
        _snap("2026-06-19", "x", 105.0, 5.0, 2.2),  # latest positive
    ]
    row = build_checkpoint(snaps, min_days=5)[0]
    assert "unstable" in row["verdict"].lower()


def test_format_table_shows_alpha_days_distinct_from_curve_days():
    # The exact prod situation on 2026-06-16: 2 equity-days but alpha real for
    # only 1 (it was null before the benchmark was wired). The verdict gates on
    # alpha-days, so the table MUST surface that count — otherwise a reader sees
    # "days: 2" next to "need ≥5 alpha days" and can't tell we have just 1.
    # The alpha-lag note stays suppressed here (curve < min_days), so the column
    # is the only honest signal.
    snaps = [
        _snap("2026-06-15", "x", 100.0, 0.0, None),  # alpha null (pre-benchmark)
        _snap("2026-06-16", "x", 109.2, 9.2, 9.2),   # first real alpha day
    ]
    rows = build_checkpoint(snaps, min_days=5)
    assert rows[0]["n_days"] == 2
    assert rows[0]["alpha_days"] == 1
    out = format_checkpoint(rows, min_days=5)
    # The header advertises an alpha-days column distinct from curve days.
    header = out.splitlines()[0]
    assert "αdays" in header
    # The data row carries both counts: 2 curve days and 1 alpha day.
    data_row = [ln for ln in out.splitlines() if ln.startswith("x")][0]
    fields = data_row.split()
    assert "2" in fields and "1" in fields


def test_format_notes_alpha_lag_when_curve_longer_than_alpha():
    # Curve has 3 days, alpha real for only 1 → the readout must say *why* the
    # verdict can still be inconclusive even though the days column looks long.
    snaps = [
        _snap("2026-06-15", "x", 100.0, 0.0, None),
        _snap("2026-06-16", "x", 101.0, 1.0, None),
        _snap("2026-06-17", "x", 102.0, 2.0, 1.0),
    ]
    out = format_checkpoint(build_checkpoint(snaps, min_days=2))
    assert "alpha" in out.lower()


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


# ---- risk window disclosure: max_dd%/vol% cover the curve, return% covers the
# epoch since inception. The curve started accumulating ~10 days after inception,
# so a drawdown before the first snapshot is invisible to max_dd — the readout
# must say so, or max_dd reads as "worst since inception" when it is not.

def test_row_carries_curve_window_bounds():
    snaps = [
        _snap("2026-06-15", "x", 100.0, 0.0, None),
        _snap("2026-06-16", "x", 101.0, 1.0, 0.5),
    ]
    row = build_checkpoint(snaps)[0]
    assert row["curve_start"] == "2026-06-15"
    assert row["curve_end"] == "2026-06-16"


def test_format_discloses_risk_window_covers_curve_only():
    # return%/equity are since inception; max_dd%/vol% only cover the curve.
    snaps = [
        _snap("2026-06-15", "x", 100.0, 0.0, None),
        _snap("2026-06-16", "x", 101.0, 1.0, 0.5),
    ]
    out = format_checkpoint(build_checkpoint(snaps, min_days=1))
    assert "risk window" in out.lower()
    # Names the curve start so max_dd is never read as inception-to-date.
    assert "2026-06-15" in out


def test_risk_window_note_absent_when_no_curve():
    out = format_checkpoint([])
    assert "risk window" not in out.lower()


# ---- selection-bias note: reading the best of N horses inflates the edge ----
# The PLAN's central statistical risk: with several horses over a short window,
# the single best one beating SPY is partly a selection effect. classify_edge
# judges each horse against its own noise but can't see we'll pick the winner.

def test_format_warns_selection_bias_when_candidate_among_several():
    # winner: flat positive alpha over 3 days → "edge candidate".
    winner = [
        _snap("2026-06-15", "winner", 110.0, 10.0, 5.0),
        _snap("2026-06-16", "winner", 110.0, 10.0, 5.0),
        _snap("2026-06-17", "winner", 110.0, 10.0, 5.0),
    ]
    # loser: latest alpha ≤ 0 → judged but "no edge", so a real comparison set.
    loser = [
        _snap("2026-06-15", "loser", 90.0, -10.0, -5.0),
        _snap("2026-06-16", "loser", 90.0, -10.0, -5.0),
        _snap("2026-06-17", "loser", 90.0, -10.0, -5.0),
    ]
    out = format_checkpoint(build_checkpoint(winner + loser, min_days=3), min_days=3)
    assert "selection bias" in out.lower()
    assert "best of" in out.lower()
    # Names the candidate and stays descriptive (promising, not proven).
    assert "winner" in out
    assert "not proven" in out.lower()


def test_no_selection_bias_note_for_a_single_candidate_horse():
    # One horse alone: no best-of-N effect to warn about.
    only = [
        _snap("2026-06-15", "solo", 110.0, 10.0, 5.0),
        _snap("2026-06-16", "solo", 110.0, 10.0, 5.0),
        _snap("2026-06-17", "solo", 110.0, 10.0, 5.0),
    ]
    out = format_checkpoint(build_checkpoint(only, min_days=3), min_days=3)
    assert "selection bias" not in out.lower()


def test_no_selection_bias_note_when_no_candidate():
    # Several horses but none reads as a candidate → nothing to over-credit.
    a = [_snap("2026-06-16", "a", 90.0, -10.0, -5.0), _snap("2026-06-17", "a", 90.0, -10.0, -5.0)]
    b = [_snap("2026-06-16", "b", 95.0, -5.0, -2.0), _snap("2026-06-17", "b", 95.0, -5.0, -2.0)]
    out = format_checkpoint(build_checkpoint(a + b, min_days=2), min_days=2)
    assert "selection bias" not in out.lower()
