"""Consolidated 2-week checkpoint readout from the persisted equity curve.

The cron run prints return, alpha and risk in *separate* tables, and only while
a live run is happening. The go/no-go decision (PLAN item 3) needs all of it in
one place, on demand, with no network and no Alpaca call. This reads
``data/ledgers/equity_curve.jsonl`` and puts each horse's latest return + alpha
next to its max-drawdown and volatility in ONE table, with a descriptive verdict
per horse.

It automates NO decision. The real-money go/no-go is the operator's; the verdict
column is a plain-language reading of the numbers, nothing more. "Edge candidate"
means "worth Luis's attention", never "ship it".
"""
from __future__ import annotations

import statistics
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from live.market_calendar import is_trading_day

from live.equity_snapshot import load_snapshots
from live.risk_metrics import compute_risk_metrics

# Default curve location, written by live.equity_snapshot during each cron run.
DEFAULT_CURVE_PATH = Path("data/ledgers/equity_curve.jsonl")

# Below this many curve days, any verdict is noise — the clean epoch only starts
# accumulating mid-June, so a handful of days can't separate edge from luck. This
# is the "stop saying inconclusive" bar (enough alpha-days to read the gates), NOT
# the bar for proven edge — see EDGE_SAMPLE_MIN_DAYS.
EDGE_MIN_DAYS = 5

# The sample a horse must clear before its verdict may read as a real go-look
# signal ("edge candidate"). The signed money contract (docs/ROADMAP-real-money.md,
# GATE #1) is explicit: edge evidence needs ~3–6 MONTHS of live paper, and "2
# semanas ganando = suerte". So a clean reading over a 1–2 week sample is
# plumbing-grade, NOT edge-grade — passing the statistical gates over a handful of
# alpha-days proves the wiring works, not that an edge exists. 63 trading days ≈ 3
# months is the conservative floor of the contract's range. Below it the best a
# horse can read is "promising"; only a months-scale sample earns "edge candidate".
EDGE_SAMPLE_MIN_DAYS = 63

# A positive-alpha horse that got there through a drawdown deeper than this is
# flagged as risky rather than a clean candidate (descriptive threshold only).
DEEP_DRAWDOWN_PCT = -15.0

# Minimum information ratio (mean ÷ stdev of the *daily* active return) for a
# horse to read as a clean candidate. If the average day-over-day alpha gain
# doesn't clear its day-to-day standard deviation, the "edge" is smaller than its
# noise — exactly the false positive a field of N horses × ~10 days will manufacture by luck.
MIN_ALPHA_SIGNAL_RATIO = 1.0

# The one verdict string that reads as a real go-look signal for the operator.
# Kept as a constant so the selection-bias note and classify_edge can't drift.
EDGE_CANDIDATE_VERDICT = "edge candidate"

# A horse that passes every quality gate but over a sample shorter than
# EDGE_SAMPLE_MIN_DAYS. It reads as encouraging but deliberately does NOT use the
# "edge candidate" words: per the contract a short clean window is plumbing-grade,
# not proof. Kept distinct so the selection-bias note (which keys off the candidate
# verdict) never over-credits a months-away sample.
PROMISING_VERDICT = "promising — plumbing-grade (need ~3–6 mo for edge)"

# A curve whose latest snapshot is this many *trading* days behind "today" reads
# as stale: the L–V cron likely stopped. The bar is ≥3 because the snapshot date
# lags the run by one trading day — each run stamps the latest *completed* daily
# bar, and at 13:00 today's bar isn't closed, so a Friday run produces a Thursday
# snapshot. That structural lag means a perfectly healthy curve, read on a weekday
# MORNING before today's run, already sits two trading days back (yesterday's run
# stamped the day before yesterday). A bar of 2 would cry wolf every morning and
# train the operator to ignore the one guard on the irreversible real-money call;
# ≥3 means a run was genuinely missed (detected the second missed-run day).
STALE_TRADING_DAYS = 3


def _alpha_signal_ratio(real_alpha: List[float]) -> Optional[float]:
    """Information ratio of the *daily* active return — mean ÷ stdev of the
    day-over-day change in alpha.

    ``alpha_pct`` is cumulative (return − benchmark since inception), so the raw
    series is a path of cumulative *levels*, not daily observations. Reading the
    ratio off the levels is order-insensitive and fooled by the dominant luck
    mode: a horse that jumps to a big lead on one day and then merely tracks SPY
    has a high mean-over-stdev of levels (the lead looks stable) yet showed no
    repeatable edge after day one. Differencing first turns the path into daily
    active returns, so a single outlier day is correctly swamped by its own noise.

    Returns ``None`` when it can't be computed or is meaningless: fewer than two
    daily increments (stdev undefined) or zero dispersion (a perfectly steady
    daily gain is *more* convincing, not less — never flag it as noise).
    """
    deltas = [b - a for a, b in zip(real_alpha, real_alpha[1:])]
    if len(deltas) < 2:
        return None
    dispersion = statistics.stdev(deltas)
    if dispersion == 0:
        return None
    return statistics.fmean(deltas) / dispersion


def _alpha_window_has_gap(dated_alpha: List[tuple]) -> bool:
    """True if the *real* (non-null) alpha observations skip a trading day.

    The noise gate (`_alpha_signal_ratio`) differences consecutive alpha values
    as if each pair were one trading day apart. If the alpha window has a hole — a
    missed cron day, or a mid-series null from a transient benchmark-fetch
    failure — two observations that are really several days apart collapse into a
    single "daily" increment, masking exactly the single-day luck the gate exists
    to catch. So when the window isn't contiguous the gate is unreliable and the
    verdict must not promote to candidate.

    Counts only NYSE trading days, so a Friday→Monday pair is contiguous, and a
    weekday holiday (e.g. Juneteenth) is not mistaken for a gap. Returns ``False``
    when it can't be judged: fewer than two real
    observations, or any unparseable date (don't manufacture a gap from bad data).
    """
    real_dates: List[date] = []
    for day, alpha in dated_alpha:
        if alpha is None:
            continue
        try:
            real_dates.append(date.fromisoformat(day))
        except (TypeError, ValueError):
            return False
    if len(real_dates) < 2:
        return False
    real_dates.sort()
    expected = 0
    cur = real_dates[0]
    while cur <= real_dates[-1]:
        if is_trading_day(cur):  # weekday and not a NYSE holiday
            expected += 1
        cur += timedelta(days=1)
    return expected > len({d.isoformat() for d in real_dates})


def classify_edge(
    alpha_series: List[Optional[float]],
    max_drawdown_pct: Optional[float],
    min_days: int = EDGE_MIN_DAYS,
    alpha_window_has_gap: bool = False,
    sample_min_days: int = EDGE_SAMPLE_MIN_DAYS,
) -> str:
    """Plain-language reading of one horse's alpha *history*. Never a decision.

    Reads the chronological alpha series rather than a single latest value, for
    two reasons that both feed the operator's irreversible real-money call:

    - **Real alpha-days, not equity-days.** Alpha is null until the benchmark is
      wired; a long equity curve with a short alpha history must not buy
      confidence the alpha data can't back. We gate on non-null observations.
    - **Stability, not a last-day bounce.** The PLAN requires alpha "positivo y
      estable". A horse whose alpha dipped ≤0 within the window and only just
      turned positive is flagged as unstable, not waved through as a candidate.
    - **Signal over noise.** Even an all-positive alpha can be a fluke if it
      swings more than it averages. We require the mean alpha to clear its own
      day-to-day dispersion before calling it a clean candidate.
    - **Contiguity.** The noise gate reads consecutive alphas as daily
      increments; if the alpha window has a hole (`alpha_window_has_gap`) that
      reading is distorted, so we refuse to promote to candidate.
    - **Sample size.** Passing every gate over a 1–2 week window proves the
      plumbing works, not that an edge exists. The money contract (ROADMAP GATE
      #1) needs ~3–6 months of paper and calls 2 winning weeks luck, so below
      ``sample_min_days`` the verdict reads "promising", never "edge candidate".
    """
    real = [a for a in alpha_series if a is not None]
    if not real:
        return "no benchmark"
    if len(real) < min_days:
        return f"inconclusive (need ≥{min_days} alpha days)"
    if real[-1] <= 0:
        return "no edge (alpha ≤ 0)"
    if min(real) <= 0:
        return "edge? but unstable (alpha dipped ≤0)"
    if max_drawdown_pct is not None and max_drawdown_pct < DEEP_DRAWDOWN_PCT:
        return "edge? but deep-drawdown risk"
    if alpha_window_has_gap:
        return "edge? but alpha curve has gaps"
    ratio = _alpha_signal_ratio(real)
    if ratio is not None and ratio < MIN_ALPHA_SIGNAL_RATIO:
        return "edge? but within noise (alpha < its own swing)"
    if len(real) < sample_min_days:
        return PROMISING_VERDICT
    return EDGE_CANDIDATE_VERDICT


def _dated_alpha_by_strategy(snapshots: List[dict]) -> Dict[str, List[tuple]]:
    """Chronological (date, alpha) pairs per strategy (null entries preserved).

    Order is by ISO date so ``classify_edge`` can read the latest value and the
    window's history, and so the gap check sees the real observation dates. Nulls
    are kept so the count of *real* alpha observations stays honest — they are
    filtered inside the verdict and the gap check, not here.
    """
    dated: Dict[str, List[tuple]] = {}
    for rec in snapshots:
        strategy = rec.get("strategy")
        if strategy is None:
            continue
        dated.setdefault(strategy, []).append((rec.get("date", ""), rec.get("alpha_pct")))
    return {
        strategy: sorted(rows, key=lambda t: t[0])
        for strategy, rows in dated.items()
    }


def _curve_window_by_strategy(
    snapshots: List[dict],
) -> Dict[str, tuple[Optional[str], Optional[str]]]:
    """Earliest and latest snapshot date per strategy (the curve's real span).

    ``max_dd``/``vol`` only cover this window, while ``return%``/``equity`` are
    cumulative since inception — which predates the curve, because snapshots only
    started accumulating ~10 days into the race. Surfacing the window keeps a
    drawdown that happened *before* the first snapshot from hiding behind a
    reassuring ``max_dd``.
    """
    dates: Dict[str, List[str]] = {}
    for rec in snapshots:
        strategy = rec.get("strategy")
        if strategy is None:
            continue
        day = rec.get("date")
        if day:
            dates.setdefault(strategy, []).append(day)
    return {
        strategy: (min(days), max(days))
        for strategy, days in dates.items()
        if days
    }


def _latest_by_strategy(snapshots: List[dict]) -> Dict[str, dict]:
    """Most recent snapshot row per strategy (by ISO date string)."""
    latest: Dict[str, dict] = {}
    for rec in snapshots:
        strategy = rec.get("strategy")
        if strategy is None:
            continue
        cur = latest.get(strategy)
        if cur is None or rec.get("date", "") >= cur.get("date", ""):
            latest[strategy] = rec
    return latest


def _settled_alpha(dated: List[tuple], as_of: date) -> List[tuple]:
    """Drop a trailing snapshot dated *today* — it's an unsettled intraday bar.

    The 13:00 cron stamps the snapshot with the latest bar, which on a trading
    day is today's *mid-session* partial, not a settled close (06-22's +19.19%
    alpha fell to +11.60% by the next close). ``_intraday_snapshot_note`` warns
    the human; the verdict must likewise refuse to count a provisional alpha, or
    it could cross the alpha-day gate on a value that shifts at the close.
    Healthy cadence leaves the latest snapshot ≥1 trading day back, so this only
    fires on the same-day intraday signature. Reader-side guard, independent of
    the source-side fix in PR #5.
    """
    if not is_trading_day(as_of):
        return dated
    today = as_of.isoformat()
    return [(d, a) for (d, a) in dated if d != today]


def build_checkpoint(
    snapshots: List[dict],
    min_days: int = EDGE_MIN_DAYS,
    sample_min_days: int = EDGE_SAMPLE_MIN_DAYS,
    as_of: Optional[date] = None,
) -> List[dict]:
    """One row per horse: latest return/alpha + curve risk + a verdict.

    Sorted by alpha descending (horses with no alpha sink to the bottom), then
    by equity, so the most market-beating horse reads first.

    ``as_of`` defaults to today and is injectable so the verdict can exclude an
    unsettled intraday bar (a snapshot dated today). The display row still shows
    that latest partial alongside the ⚠ INTRADAY note; only the verdict and the
    alpha-day count read settled closes.
    """
    if as_of is None:
        as_of = date.today()
    latest = _latest_by_strategy(snapshots)
    risk = compute_risk_metrics(snapshots)
    dated_alpha = _dated_alpha_by_strategy(snapshots)
    windows = _curve_window_by_strategy(snapshots)

    rows: List[dict] = []
    for strategy, rec in latest.items():
        m = risk.get(strategy, {})
        max_dd = m.get("max_drawdown_pct")
        n_days = m.get("n_days", 0)
        dated = _settled_alpha(dated_alpha.get(strategy, []), as_of)
        series = [alpha for _, alpha in dated]
        alpha_days = sum(1 for a in series if a is not None)
        has_gap = _alpha_window_has_gap(dated)
        start, end = windows.get(strategy, (None, None))
        rows.append(
            {
                "strategy": strategy,
                "date": rec.get("date"),
                "equity": rec.get("equity"),
                "return_pct": rec.get("return_pct"),
                "alpha_pct": rec.get("alpha_pct"),
                "benchmark_pct": rec.get("benchmark_pct"),
                "max_drawdown_pct": max_dd,
                "volatility_pct": m.get("volatility_pct"),
                "n_days": n_days,
                "alpha_days": alpha_days,
                "gap_days": m.get("gap_days", 0),
                "curve_start": start,
                "curve_end": end,
                "verdict": classify_edge(
                    series,
                    max_dd,
                    min_days,
                    alpha_window_has_gap=has_gap,
                    sample_min_days=sample_min_days,
                ),
            }
        )

    rows.sort(
        key=lambda r: (
            r["alpha_pct"] if r["alpha_pct"] is not None else float("-inf"),
            r["equity"] if r["equity"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    return rows


def _selection_bias_note(rows: List[dict]) -> Optional[str]:
    """Warn that reading the *best* of N horses inflates the apparent edge.

    The PLAN names this as the central statistical risk: with a field of N
    horses over ~10 days (currently 4), the single best one beating SPY is
    partly a selection effect, not
    proof of skill. ``classify_edge`` gates each horse against its *own* noise,
    but it can't see that the operator will look at the winner — and the chance
    that the best of N independent horses clears the bar by luck scales roughly
    with N. So a lone "edge candidate" among several compared horses deserves a
    higher bar than the same verdict from a single horse.

    Returns ``None`` (no note) unless at least two horses carry a real verdict
    *and* at least one reads as a candidate. Descriptive only.
    """
    judged = [r for r in rows if r.get("verdict") not in (None, "no benchmark")]
    candidates = [r for r in judged if r.get("verdict") == EDGE_CANDIDATE_VERDICT]
    if len(judged) < 2 or not candidates:
        return None
    names = ", ".join(r["strategy"] for r in candidates)
    return (
        f"⚠ selection bias: {len(candidates)} of {len(judged)} compared horses "
        f"read as '{EDGE_CANDIDATE_VERDICT}' ({names}). You're picking the best of "
        f"{len(judged)} — the odds that *some* horse beats SPY by luck scale with "
        "the number of horses, so the winner's edge is biased upward. Treat a "
        "candidate here as promising, NOT proven: require it to persist (and ideally "
        "repeat out-of-sample) before it counts toward the real-money call."
    )


def _benchmark_anchor_note(rows: List[dict]) -> Optional[str]:
    """Show the SPY return that ``alpha%`` is measured against.

    ``alpha% = return − SPY buy&hold``, so the alpha column is only interpretable
    next to the benchmark it nets out. The same +14% alpha means something very
    different in a flat market than in a crash — the operator's go/no-go needs to
    see which regime the horses beat, not just the spread. The cron output prints
    this line, but the on-demand checkpoint (the actual decision tool) dropped it.

    Uses the latest snapshot's ``benchmark_pct`` (cumulative since inception, like
    ``return%``). Returns ``None`` when no horse carries a real benchmark — alpha
    was null before the benchmark was wired, so there's no anchor to show.
    """
    dated_bench = [
        (r.get("curve_end") or "", r["benchmark_pct"])
        for r in rows
        if r.get("benchmark_pct") is not None
    ]
    if not dated_bench:
        return None
    _, latest_bench = max(dated_bench, key=lambda t: t[0])
    return (
        f"benchmark SPY buy&hold since inception: {latest_bench:+.2f}%  "
        "(alpha% = each horse's return − this)"
    )


def _trading_days_after(start: date, end: date) -> int:
    """Count NYSE trading days strictly after ``start`` up to and including ``end``.

    Weekends and market holidays never count, so a checkpoint read on Monday over
    a Friday curve isn't penalised for the weekend — only genuine missed trading
    days show up.
    """
    if end <= start:
        return 0
    count = 0
    cur = start + timedelta(days=1)
    while cur <= end:
        if is_trading_day(cur):  # weekday and not a NYSE holiday
            count += 1
        cur += timedelta(days=1)
    return count


def _staleness_note(rows: List[dict], as_of: date) -> Optional[str]:
    """Warn when the curve's latest snapshot is too many trading days behind today.

    ``checkpoint_report`` reads only the persisted curve and has no clock, so a
    frozen curve (the L–V cron stopped: Mac asleep, LaunchAgent broken) would
    print an old snapshot as if current — and the operator could make the
    irreversible real-money call on stale numbers. The snapshot date lags the run
    by one trading day (it stamps the latest *completed* bar), so a healthy curve
    read on a weekday morning is already two trading days back; only
    ``STALE_TRADING_DAYS`` (≥3) or more means a run was genuinely missed.

    Returns ``None`` (no note) when the curve is current or its dates are
    unparseable. Descriptive only.
    """
    ends = [r["curve_end"] for r in rows if r.get("curve_end")]
    if not ends:
        return None
    try:
        last = date.fromisoformat(max(ends))
    except (TypeError, ValueError):
        return None
    elapsed = _trading_days_after(last, as_of)
    if elapsed < STALE_TRADING_DAYS:
        return None
    return (
        f"⚠ STALE: latest snapshot is {last.isoformat()}, {elapsed} trading days "
        f"behind today ({as_of.isoformat()}) — the L–V cron may have stopped. The "
        "numbers below are NOT current; check data/cron.log and "
        "`launchctl list | grep horserace` before trusting this for any decision."
    )


def _intraday_snapshot_note(rows: List[dict], as_of: date) -> Optional[str]:
    """Warn when the latest snapshot is dated today — recorded on an unsettled bar.

    The 13:00 CST cron stamps each snapshot with ``df.index[-1]`` (the latest bar
    yfinance returns). Mid-session that bar has NOT closed, so a snapshot dated
    *today* reflects an intraday price, not a settled close — its return/alpha/vol
    are provisional and usually move by the close (06-22's +19.19% momentum alpha
    collapsed to +11.60% the next day, once a real close landed). A healthy curve
    never stamps today: it lags ≥1 trading day behind the run (today's bar isn't
    closed at 13:00), so ``last == today`` is the intraday-partial-bar signature.
    The source fix lives in PR #5 (drops in-progress bars before they're recorded);
    this is the independent reader-side warning so the operator never weighs an
    unsettled row in the irreversible go/no-go. Descriptive only.
    """
    ends = [r["curve_end"] for r in rows if r.get("curve_end")]
    if not ends:
        return None
    try:
        last = date.fromisoformat(max(ends))
    except (TypeError, ValueError):
        return None
    if last != as_of or not is_trading_day(as_of):
        return None
    return (
        f"⚠ INTRADAY: latest snapshot ({last.isoformat()}) is dated today and was "
        "recorded by the 13:00 cron on an unsettled intraday bar, NOT a settled "
        "close — its return%/alpha%/vol% are provisional and typically shift by the "
        "close (06-22's +19.19% alpha fell to +11.60% the next day). Don't weigh the "
        "latest row in any decision until a settled close supersedes it. Source fix "
        "pending in PR #5."
    )


def _sample_bar_note(rows: List[dict], sample_min_days: int) -> Optional[str]:
    """Explain why a clean horse reads "promising" instead of "edge candidate".

    The verdict column already carries the short string, but the operator needs
    the *why* tied to the signed contract: edge evidence needs ~3–6 months of live
    paper (ROADMAP GATE #1), and 2 winning weeks is luck. Without this, a reader
    who waited out the original "2-week window" could mistake a clean short-sample
    reading for the gate being near. Fires only when a horse is in the promising
    tier; descriptive only.
    """
    promising = [r for r in rows if r.get("verdict") == PROMISING_VERDICT]
    if not promising:
        return None
    best_alpha_days = max(r.get("alpha_days", 0) for r in promising)
    return (
        f"ℹ sample bar: '{PROMISING_VERDICT}' means every quality gate passed but "
        f"over too short a sample (best is {best_alpha_days} alpha-days; edge needs "
        f"≥{sample_min_days}, ~3–6 mo of live paper per ROADMAP GATE #1). A clean "
        "1–2 week reading proves the plumbing, NOT an edge — 2 winning weeks is luck. "
        "It only earns 'edge candidate' after a months-scale sample."
    )


def format_checkpoint(
    rows: List[dict],
    min_days: int = EDGE_MIN_DAYS,
    as_of: Optional[date] = None,
    sample_min_days: int = EDGE_SAMPLE_MIN_DAYS,
) -> str:
    """Render the consolidated checkpoint table with an honest footer.

    ``as_of`` defaults to today and is injectable so the staleness check is
    testable without a real clock.
    """
    if as_of is None:
        as_of = date.today()
    if not rows:
        return (
            "no equity curve yet — the checkpoint readout needs at least one "
            "snapshot day in data/ledgers/equity_curve.jsonl (fills on the next "
            "L–V cron run)."
        )

    # Two distinct day counts: `days` is the equity-curve length (drives
    # max_dd%/vol%), `αdays` is the count of real alpha observations (drives the
    # verdict). They diverge while alpha is null pre-benchmark, so showing only
    # one would let a reader misjudge how close a horse is to a verdict.
    header = (
        f"{'strategy':<20}{'equity':>12}{'return%':>10}{'alpha%':>9}"
        f"{'max_dd%':>10}{'vol%':>9}{'days':>6}{'αdays':>7}  verdict"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        alpha = r["alpha_pct"]
        alpha_s = f"{alpha:>9.2f}" if alpha is not None else f"{'n/a':>9}"
        dd = r["max_drawdown_pct"]
        dd_s = f"{dd:>10.2f}" if dd is not None else f"{'n/a':>10}"
        vol = r["volatility_pct"]
        vol_s = f"{vol:>9.2f}" if vol is not None else f"{'n/a':>9}"
        equity = r["equity"] if r["equity"] is not None else 0.0
        ret = r["return_pct"] if r["return_pct"] is not None else 0.0
        lines.append(
            f"{r['strategy']:<20}{equity:>12.2f}{ret:>10.2f}{alpha_s}"
            f"{dd_s}{vol_s}{r['n_days']:>6}{r.get('alpha_days', 0):>7}  {r['verdict']}"
        )

    lines.append("-" * len(header))
    anchor = _benchmark_anchor_note(rows)
    if anchor:
        lines.append(anchor)
    stale_note = _staleness_note(rows, as_of)
    if stale_note:
        lines.append(stale_note)
    intraday_note = _intraday_snapshot_note(rows, as_of)
    if intraday_note:
        lines.append(intraday_note)
    lagging = [
        r for r in rows
        if r.get("n_days", 0) >= min_days and r.get("alpha_days", r.get("n_days", 0)) < min_days
    ]
    if lagging:
        detail = ", ".join(
            f"{r['strategy']} ({r.get('alpha_days', 0)}/{r['n_days']})" for r in lagging
        )
        lines.append(
            f"ℹ alpha lag: curve is ≥{min_days} days but alpha is real for fewer "
            f"(strategy: alpha-days/curve-days → {detail}). Alpha was null before the "
            "benchmark was wired, so the verdict gates on real alpha-days, not curve "
            "length — that's why 'days' can look long while the verdict stays inconclusive."
        )
    starts = [r["curve_start"] for r in rows if r.get("curve_start")]
    ends = [r["curve_end"] for r in rows if r.get("curve_end")]
    if starts and ends:
        lines.append(
            f"ℹ risk window: max_dd%/vol% cover the equity curve only "
            f"({min(starts)} → {max(ends)}); return%/equity are cumulative since "
            "inception, which predates the curve — any drawdown before the curve "
            "started is NOT in max_dd. Read max_dd as worst-since-curve-start, not "
            "worst-since-inception."
        )
    gappy = [r for r in rows if r.get("gap_days", 0) > 0]
    if gappy:
        detail = ", ".join(f"{r['strategy']} ({r['gap_days']})" for r in gappy)
        lines.append(
            f"⚠ GAP: equity curve is missing trading day(s) — {detail}. A skipped "
            "cron run leaves holes; vol/max_dd treat a multi-day jump as one day, "
            "so read those numbers with caution and check data/cron.log."
        )
    sample_note = _sample_bar_note(rows, sample_min_days)
    if sample_note:
        lines.append(sample_note)
    selection_note = _selection_bias_note(rows)
    if selection_note:
        lines.append(selection_note)
    lines.append(
        "Descriptive only — alpha% = return − SPY buy&hold; verdict is a reading "
        "of the numbers, NOT a decision."
    )
    lines.append(
        "Real-money go/no-go is the operator's call (see PLAN item 3 + "
        "docs/ROADMAP-real-money.md). NO real money without proven, stable edge."
    )
    return "\n".join(lines)


def main(path=DEFAULT_CURVE_PATH) -> None:
    snapshots = load_snapshots(path)
    print(f"=== Paper horse-race checkpoint readout ({path}) ===")
    print(format_checkpoint(build_checkpoint(snapshots)))


if __name__ == "__main__":
    main()
