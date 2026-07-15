from __future__ import annotations

from datetime import date

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
    # Cumulative alpha climbing steadily (+0.7, +0.5, +0.7, +0.5 per day): a
    # consistently positive daily active return, shallow drawdown → clean candidate.
    # sample_min_days=5 isolates the gate logic from the months-scale sample bar
    # (tested separately) so this asserts the quality gates promote to candidate.
    v = classify_edge(
        alpha_series=[1.0, 1.7, 2.2, 2.9, 3.4],
        max_drawdown_pct=-4.0,
        min_days=5,
        sample_min_days=5,
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
    # alpha_pct is CUMULATIVE (return − benchmark since inception). A cumulative
    # lead that whipsaws 0.1→4.0→0.2→3.5→0.15 means the *daily* active return
    # swings violently (+3.9, −3.8, +3.3, −3.35): the day-to-day edge is pure
    # noise around zero. Over 3 horses × ~10 days that is exactly the by-luck
    # shape; it must NOT read as a clean "edge candidate" — discipline over return.
    v = classify_edge(
        alpha_series=[0.1, 4.0, 0.2, 3.5, 0.15], max_drawdown_pct=-3.0, min_days=5
    )
    assert "noise" in v.lower()
    assert "candidate" not in v.lower()


def test_classify_flags_within_noise_when_alpha_is_one_lucky_day_then_coast():
    # THE dominant luck mode: a horse jumps to a big cumulative lead on a single
    # day (0.2→7.0) and then merely tracks SPY (7.0, 7.1, 6.9, 7.05). Every level
    # is positive and the lead looks stable, so a metric reading cumulative
    # *levels* (mean 5.65 / stdev 3.06 → ratio 1.85) waves it through as a
    # candidate. But the daily active return is one +6.8 outlier buried in noise:
    # the strategy showed no repeatable edge after day one. The information ratio
    # must be computed on daily increments so this reads as within-noise.
    v = classify_edge(
        alpha_series=[0.2, 7.0, 7.1, 6.9, 7.05], max_drawdown_pct=-3.0, min_days=5
    )
    assert "noise" in v.lower()
    assert "candidate" not in v.lower()


def test_classify_candidate_survives_when_signal_beats_noise():
    # A cumulative alpha that climbs steadily day after day (+0.6, +0.4, +0.7,
    # +0.4) means a consistently positive *daily* active return whose mean dwarfs
    # its dispersion — genuine, repeated outperformance. The within-noise gate
    # must not fire here.
    v = classify_edge(
        alpha_series=[1.0, 1.6, 2.0, 2.7, 3.1],
        max_drawdown_pct=-3.0,
        min_days=5,
        sample_min_days=5,
    )
    assert "candidate" in v.lower()
    assert "noise" not in v.lower()


# ---- sample-size bar: the ROADMAP money contract needs ~3–6 months, not weeks ----
# GATE #1 of docs/ROADMAP-real-money.md is explicit: edge evidence needs ~3–6
# MONTHS of live paper, and "2 semanas ganando = suerte". So a horse that clears
# every statistical *quality* gate over a 1–2 week window is plumbing-grade, not
# edge-grade — it must NOT read as "edge candidate" (the one verdict that signals
# a real go-look for the irreversible real-money call). This is the gap between the
# code's verdict semantics and the signed contract; the gate is strictly
# conservative (it can only downgrade a verdict, never promote).

def test_classify_promising_not_candidate_when_sample_too_short():
    # Passes every quality gate (positive, stable, signal beats noise, shallow DD)
    # but over only 5 alpha-days — far below the months-scale contract bar.
    v = classify_edge(
        alpha_series=[1.0, 1.7, 2.2, 2.9, 3.4],
        max_drawdown_pct=-4.0,
        min_days=5,
        sample_min_days=63,
    )
    assert "candidate" not in v.lower()
    assert "promising" in v.lower()


def test_classify_earns_candidate_only_at_months_scale_sample():
    # Same clean, steadily-climbing shape, but now over a months-scale sample
    # (≥ sample_min_days real alpha-days). Only here does it earn "edge candidate".
    series = [round(1.0 + 0.5 * i, 4) for i in range(63)]  # +0.5/day, no dispersion
    v = classify_edge(
        alpha_series=series,
        max_drawdown_pct=-4.0,
        min_days=5,
        sample_min_days=63,
    )
    assert "candidate" in v.lower()
    assert "promising" not in v.lower()


def test_edge_sample_min_days_is_months_scale():
    # The contract floor is ~3 months of paper ≈ 63 trading days. Guard against a
    # silent drift back toward a weeks-scale bar that would re-open the gap.
    from live.checkpoint_report import EDGE_SAMPLE_MIN_DAYS

    assert EDGE_SAMPLE_MIN_DAYS >= 60


def test_build_uses_months_scale_sample_bar_by_default():
    # Default build_checkpoint (no sample_min_days override) must apply the contract
    # bar: a clean 3-day winner reads promising, never "edge candidate".
    winner = [
        _snap("2026-06-15", "winner", 110.0, 10.0, 5.0),
        _snap("2026-06-16", "winner", 110.0, 10.0, 5.0),
        _snap("2026-06-17", "winner", 110.0, 10.0, 5.0),
    ]
    rows = build_checkpoint(winner, min_days=3)
    assert rows[0]["verdict"] != "edge candidate"
    assert "promising" in rows[0]["verdict"].lower()


def test_format_explains_sample_bar_when_horse_is_promising():
    # A clean short-sample horse: the footer must explain WHY it reads promising,
    # tying it to the 3–6 month contract bar so the operator can't mistake a clean
    # 1–2 week reading for the gate being near.
    winner = [
        _snap("2026-06-15", "winner", 110.0, 10.0, 5.0),
        _snap("2026-06-16", "winner", 110.0, 10.0, 5.0),
        _snap("2026-06-17", "winner", 110.0, 10.0, 5.0),
    ]
    out = format_checkpoint(build_checkpoint(winner, min_days=3), min_days=3)
    assert "sample bar" in out.lower()
    assert "3–6 mo" in out or "3-6 mo" in out


def test_format_no_sample_bar_note_without_promising_horse():
    # No horse in the promising tier (all inconclusive) → no sample-bar note.
    weak = [
        _snap("2026-06-16", "h", 90.0, -10.0, -5.0),
        _snap("2026-06-17", "h", 90.0, -10.0, -5.0),
    ]
    out = format_checkpoint(build_checkpoint(weak, min_days=2), min_days=2)
    assert "sample bar" not in out.lower()


def test_short_sample_promising_does_not_trip_selection_bias():
    # Two clean short-sample winners: both read promising, NOT "edge candidate".
    # The selection-bias note keys off the candidate verdict, so it must stay
    # silent — there's no over-credited candidate to warn about yet.
    a = [
        _snap("2026-06-15", "a", 110.0, 10.0, 5.0),
        _snap("2026-06-16", "a", 110.0, 10.0, 5.0),
        _snap("2026-06-17", "a", 110.0, 10.0, 5.0),
    ]
    b = [
        _snap("2026-06-15", "b", 108.0, 8.0, 4.0),
        _snap("2026-06-16", "b", 108.0, 8.0, 4.0),
        _snap("2026-06-17", "b", 108.0, 8.0, 4.0),
    ]
    out = format_checkpoint(build_checkpoint(a + b, min_days=3), min_days=3)
    assert "selection bias" not in out.lower()
    assert "sample bar" in out.lower()


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


def test_verdict_ignores_intraday_today_bar():
    # 4 settled alpha-days + today's intraday provisional bar (dated as_of, a
    # trading day). The ⚠ INTRADAY note warns the human that today's row is a
    # mid-session partial that "typically shifts by the close" (06-22's +19.19%
    # alpha fell to +11.60% by the next close). The verdict — which feeds the
    # irreversible money call — must NOT consume it: counting it would cross the
    # ≥5 alpha-day gate on a bar that hasn't settled.
    settled = [
        _snap("2026-06-15", "x", 100.0, 0.0, 1.0),
        _snap("2026-06-16", "x", 101.0, 1.0, 1.2),
        _snap("2026-06-17", "x", 102.0, 2.0, 1.4),
        _snap("2026-06-18", "x", 103.0, 3.0, 1.6),
    ]
    intraday = _snap("2026-06-23", "x", 120.0, 20.0, 19.0)  # mid-session spike
    row = build_checkpoint(
        settled + [intraday], min_days=5, as_of=date(2026, 6, 23)
    )[0]
    assert row["alpha_days"] == 4  # intraday bar excluded from the verdict count
    assert "inconclusive" in row["verdict"].lower()  # gate not crossed on a partial


def test_verdict_counts_latest_when_not_intraday():
    # Healthy cadence: the latest snapshot lags ≥1 trading day behind as_of
    # (today's bar isn't closed at the 13:00 run), so the last row is a settled
    # close and MUST count toward the verdict.
    settled = [
        _snap("2026-06-15", "x", 100.0, 0.0, 1.0),
        _snap("2026-06-16", "x", 101.0, 1.0, 1.2),
        _snap("2026-06-17", "x", 102.0, 2.0, 1.4),
        _snap("2026-06-18", "x", 103.0, 3.0, 1.6),
        _snap("2026-06-22", "x", 104.0, 4.0, 1.8),
    ]
    row = build_checkpoint(settled, min_days=5, as_of=date(2026, 6, 23))[0]
    assert row["alpha_days"] == 5  # every settled close counted


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


def test_gap_note_names_the_missing_date():
    # The GAP warning must surface *which* trading day is missing so a reader can
    # tell a known permanent hole from a fresh missed run without opening cron.log.
    snaps = [
        _snap("2026-06-15", "x", 100.0, 0.0, 0.0),  # Mon
        _snap("2026-06-17", "x", 102.0, 2.0, 1.0),  # Wed (Tue 06-16 missing)
    ]
    rows = build_checkpoint(snaps, min_days=1)
    assert rows[0]["gap_dates"] == ["2026-06-16"]
    out = format_checkpoint(rows, min_days=1)
    assert "2026-06-16" in out


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


# ---- benchmark anchor: alpha% = return − SPY, so the readout must show the SPY
# return it is measured against. A +14% alpha in a flat market is a different
# signal than the same alpha in a crash; without the anchor the operator can't
# tell which regime the horses beat.

def test_format_shows_benchmark_anchor():
    snaps = [
        _snap("2026-06-16", "x", 110.0, 10.0, 8.5, benchmark_pct=1.5),
        _snap("2026-06-17", "x", 112.0, 12.0, 9.0, benchmark_pct=3.0),
    ]
    out = format_checkpoint(build_checkpoint(snaps, min_days=1))
    assert "benchmark spy" in out.lower()
    # Latest benchmark return, not an earlier one.
    assert "3.00" in out


def test_benchmark_anchor_absent_when_no_benchmark():
    # Pre-benchmark snapshots carry null benchmark_pct; no anchor to show.
    snaps = [
        _snap("2026-06-15", "x", 100.0, 0.0, None, benchmark_pct=None),
        _snap("2026-06-16", "x", 101.0, 1.0, None, benchmark_pct=None),
    ]
    out = format_checkpoint(build_checkpoint(snaps, min_days=1))
    assert "benchmark spy" not in out.lower()


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
    out = format_checkpoint(
        build_checkpoint(winner + loser, min_days=3, sample_min_days=3),
        min_days=3,
        sample_min_days=3,
    )
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


# ---- staleness: a dead cron must not read as a fresh checkpoint ----
# checkpoint_report reads only the persisted curve, with no notion of "today".
# If the L–V cron stops (Mac asleep, LaunchAgent broken), the curve freezes but
# the readout would print an old snapshot as if current — and the operator could
# make the irreversible real-money call on stale data. The staleness note closes
# that gap. `as_of` is injectable so the clock is testable.

def test_format_warns_when_curve_is_stale():
    from datetime import date

    # Curve frozen on Tue 06-23; reading the following Fri 06-26 means Wed/Thu/Fri
    # ran with no snapshot → 3 trading days behind → the cron is genuinely dead.
    # Holiday-free window (avoids Juneteenth 06-19) so the count is unambiguous.
    snaps = [
        _snap("2026-06-22", "h", 110.0, 10.0, 5.0),
        _snap("2026-06-23", "h", 110.0, 10.0, 5.0),
    ]
    out = format_checkpoint(
        build_checkpoint(snaps, min_days=2), min_days=2, as_of=date(2026, 6, 26)
    )
    assert "stale" in out.lower()
    assert "2026-06-23" in out  # names the last snapshot we actually have


def test_format_no_stale_warning_when_curve_is_current():
    from datetime import date

    snaps = [
        _snap("2026-06-17", "h", 110.0, 10.0, 5.0),
        _snap("2026-06-18", "h", 110.0, 10.0, 5.0),
    ]
    out = format_checkpoint(
        build_checkpoint(snaps, min_days=2), min_days=2, as_of=date(2026, 6, 18)
    )
    assert "stale" not in out.lower()


def test_format_no_stale_warning_within_one_trading_day():
    from datetime import date

    # Reading Thu 06-18 right after that day's 13:00 run: the run stamps the
    # snapshot with the latest *completed* bar (Wed 06-17, since today's bar isn't
    # closed at 13:00), so the freshest curve sits one trading day back. Not stale.
    snaps = [
        _snap("2026-06-16", "h", 110.0, 10.0, 5.0),
        _snap("2026-06-17", "h", 110.0, 10.0, 5.0),
    ]
    out = format_checkpoint(
        build_checkpoint(snaps, min_days=2), min_days=2, as_of=date(2026, 6, 18)
    )
    assert "stale" not in out.lower()


def test_format_warns_when_latest_snapshot_is_dated_today():
    from datetime import date

    # The 13:00 cron stamps df.index[-1]. A healthy curve NEVER stamps today (the
    # day's bar isn't closed at 13:00, so it lags ≥1 trading day). A snapshot dated
    # today is the intraday-partial-bar taint (06-22/06-23): an unsettled price that
    # typically moves by the close. Read on that same trading day, the latest row
    # must be flagged provisional before any go/no-go.
    snaps = [
        _snap("2026-06-22", "h", 110.0, 10.0, 5.0),
        _snap("2026-06-23", "h", 120.0, 20.0, 5.0),  # dated as_of below
    ]
    out = format_checkpoint(
        build_checkpoint(snaps, min_days=2), min_days=2, as_of=date(2026, 6, 23)
    )
    assert "intraday" in out.lower()
    assert "2026-06-23" in out


def test_format_no_intraday_warning_on_healthy_lagged_curve():
    from datetime import date

    # Healthy cadence: last snapshot lags the run by ≥1 trading day, so it is never
    # dated today. Reading Tue 06-23 with the freshest snapshot at Mon 06-22 → no
    # intraday warning (firing here would cry wolf on every settled curve).
    snaps = [
        _snap("2026-06-18", "h", 110.0, 10.0, 5.0),
        _snap("2026-06-22", "h", 110.0, 10.0, 5.0),
    ]
    out = format_checkpoint(
        build_checkpoint(snaps, min_days=2), min_days=2, as_of=date(2026, 6, 23)
    )
    assert "intraday" not in out.lower()


def test_format_no_stale_warning_on_healthy_weekday_morning():
    from datetime import date

    # Production cadence: each run stamps the snapshot with the prior trading day's
    # close (the latest *completed* daily bar — today's isn't closed at 13:00). So
    # the freshest a healthy curve can be, read on a weekday MORNING before today's
    # run, is TWO trading days back: yesterday's run produced a snapshot dated the
    # day before yesterday. Reading Thu 06-18 morning, the last run (Wed 06-17)
    # stamped Tue 06-16. That's a healthy cron with today's run pending — warning
    # here would cry wolf every weekday morning and train the operator to ignore
    # the one guard protecting the irreversible real-money call.
    snaps = [
        _snap("2026-06-15", "h", 110.0, 10.0, 5.0),
        _snap("2026-06-16", "h", 110.0, 10.0, 5.0),
    ]
    out = format_checkpoint(
        build_checkpoint(snaps, min_days=2), min_days=2, as_of=date(2026, 6, 18)
    )
    assert "stale" not in out.lower()


def test_stale_warning_ignores_weekend_gap():
    from datetime import date

    # Curve ends Fri 06-19; reading Mon 06-22. Only Monday is a trading day in
    # between → 1 trading day → today's run pending over the weekend, not stale.
    snaps = [
        _snap("2026-06-18", "h", 110.0, 10.0, 5.0),
        _snap("2026-06-19", "h", 110.0, 10.0, 5.0),
    ]
    out = format_checkpoint(
        build_checkpoint(snaps, min_days=2), min_days=2, as_of=date(2026, 6, 22)
    )
    assert "stale" not in out.lower()


def test_stale_note_absent_when_no_curve():
    out = format_checkpoint([])
    assert "stale" not in out.lower()


# ---- gap-aware verdict: the noise gate differences consecutive alpha values as
# if one trading day apart. A hole in the *alpha window* (a missed cron day, or a
# mid-series null from a transient benchmark-fetch failure) collapses a multi-day
# move into one "daily" increment — masking exactly the single-day luck the gate
# exists to catch. When the alpha window isn't contiguous the gate is unreliable,
# so the verdict must not promote to candidate. Same test-verde/prod-roto class,
# in the component that feeds the irreversible real-money call.

from live.checkpoint_report import _alpha_window_has_gap


def test_alpha_window_has_gap_detects_missing_weekday():
    # Mon, Tue, [skip Wed], Thu — real alpha on each present day.
    dated = [
        ("2026-06-15", 1.0),
        ("2026-06-16", 2.0),
        ("2026-06-18", 3.0),
    ]
    assert _alpha_window_has_gap(dated) is True


def test_alpha_window_has_gap_false_on_contiguous():
    dated = [
        ("2026-06-15", 1.0),
        ("2026-06-16", 2.0),
        ("2026-06-17", 3.0),
    ]
    assert _alpha_window_has_gap(dated) is False


def test_alpha_window_has_gap_ignores_null_prefix():
    # Nulls before the benchmark was wired are not part of the alpha window; a
    # contiguous real-alpha suffix has no gap even though the curve is longer.
    dated = [
        ("2026-06-12", None),
        ("2026-06-15", 1.0),
        ("2026-06-16", 2.0),
    ]
    assert _alpha_window_has_gap(dated) is False


def test_alpha_window_has_gap_ignores_weekend():
    # Fri then Mon is contiguous trading — the weekend is not a gap.
    dated = [
        ("2026-06-19", 1.0),  # Friday
        ("2026-06-22", 2.0),  # Monday
    ]
    assert _alpha_window_has_gap(dated) is False


def test_alpha_window_has_gap_false_under_two_observations():
    assert _alpha_window_has_gap([("2026-06-15", 1.0)]) is False
    assert _alpha_window_has_gap([]) is False


def test_alpha_window_has_gap_detects_midseries_null():
    # A transient benchmark-fetch failure writes a null for one trading day,
    # leaving real alpha on Mon and Wed but not Tue → the increment Mon→Wed spans
    # two days, so the gate's "daily" reading is distorted. Treated as a gap.
    dated = [
        ("2026-06-15", 1.0),
        ("2026-06-16", None),
        ("2026-06-17", 3.0),
    ]
    assert _alpha_window_has_gap(dated) is True


def test_classify_flags_gap_when_alpha_window_not_contiguous():
    # An otherwise-clean candidate series, but its alpha window has a hole → the
    # noise gate can't be trusted, so it must NOT read as a candidate.
    v = classify_edge(
        alpha_series=[1.0, 1.6, 2.0, 2.7, 3.1],
        max_drawdown_pct=-3.0,
        min_days=5,
        alpha_window_has_gap=True,
    )
    assert "gap" in v.lower()
    assert "candidate" not in v.lower()


def test_classify_no_gap_flag_by_default_keeps_candidate():
    # Back-compat: without the gap flag the same series still reads as a candidate.
    v = classify_edge(
        alpha_series=[1.0, 1.6, 2.0, 2.7, 3.1],
        max_drawdown_pct=-3.0,
        min_days=5,
        sample_min_days=5,
    )
    assert "candidate" in v.lower()
    assert "gap" not in v.lower()


def test_classify_gap_does_not_override_more_specific_verdicts():
    # A gap only blocks promotion to candidate; an already-cautious verdict
    # (too few alpha days) is not made noisier by it.
    v = classify_edge(
        alpha_series=[1.0, 2.0],
        max_drawdown_pct=-2.0,
        min_days=5,
        alpha_window_has_gap=True,
    )
    assert "inconclusive" in v.lower()


def test_build_flags_gap_in_alpha_window_through_verdict():
    # Real alpha on 5 days but Wed 06-24 is missing → the winning horse's verdict
    # reads as gapped, not "edge candidate", agreeing with the ⚠ GAP note.
    # Holiday-free window (avoids Juneteenth 06-19) so the hole is the only gap.
    dates = ["2026-06-22", "2026-06-23", "2026-06-25", "2026-06-26", "2026-06-29"]
    alphas = [1.0, 1.6, 2.0, 2.7, 3.1]
    snaps = [
        _snap(d, "momentum_rotation", 25000 + a * 100, a, a)
        for d, a in zip(dates, alphas)
    ]
    # Read the window as_of a settled day after it closes, so the intraday guard
    # (which drops a snapshot dated today) leaves all five alpha-days intact.
    rows = build_checkpoint(snaps, min_days=5, as_of=date(2026, 6, 30))
    assert "gap" in rows[0]["verdict"].lower()
    assert "candidate" not in rows[0]["verdict"].lower()
