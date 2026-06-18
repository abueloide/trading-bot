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

from live.equity_snapshot import load_snapshots
from live.risk_metrics import compute_risk_metrics

# Default curve location, written by live.equity_snapshot during each cron run.
DEFAULT_CURVE_PATH = Path("data/ledgers/equity_curve.jsonl")

# Below this many curve days, any verdict is noise — the clean epoch only starts
# accumulating mid-June, so a handful of days can't separate edge from luck.
EDGE_MIN_DAYS = 5

# A positive-alpha horse that got there through a drawdown deeper than this is
# flagged as risky rather than a clean candidate (descriptive threshold only).
DEEP_DRAWDOWN_PCT = -15.0

# Minimum information ratio (mean ÷ stdev of the *daily* active return) for a
# horse to read as a clean candidate. If the average day-over-day alpha gain
# doesn't clear its day-to-day standard deviation, the "edge" is smaller than its
# noise — exactly the false positive 3 horses × ~10 days will manufacture by luck.
MIN_ALPHA_SIGNAL_RATIO = 1.0

# The one verdict string that reads as a real go-look signal for the operator.
# Kept as a constant so the selection-bias note and classify_edge can't drift.
EDGE_CANDIDATE_VERDICT = "edge candidate"

# A curve whose latest snapshot is this many *trading* days behind "today" reads
# as stale: the L–V cron likely stopped. One trading day of lag is just today's
# 13:00 run pending (or a weekend), so the bar is ≥2 to avoid crying wolf daily.
STALE_TRADING_DAYS = 2


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


def classify_edge(
    alpha_series: List[Optional[float]],
    max_drawdown_pct: Optional[float],
    min_days: int = EDGE_MIN_DAYS,
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
    ratio = _alpha_signal_ratio(real)
    if ratio is not None and ratio < MIN_ALPHA_SIGNAL_RATIO:
        return "edge? but within noise (alpha < its own swing)"
    return EDGE_CANDIDATE_VERDICT


def _alpha_series_by_strategy(snapshots: List[dict]) -> Dict[str, List[Optional[float]]]:
    """Chronological alpha values per strategy (null entries preserved).

    Order is by ISO date so ``classify_edge`` can read the latest value and the
    window's history. Nulls are kept so the count of *real* alpha observations
    stays honest — they are filtered inside the verdict, not here.
    """
    dated: Dict[str, List[tuple]] = {}
    for rec in snapshots:
        strategy = rec.get("strategy")
        if strategy is None:
            continue
        dated.setdefault(strategy, []).append((rec.get("date", ""), rec.get("alpha_pct")))
    return {
        strategy: [alpha for _, alpha in sorted(rows, key=lambda t: t[0])]
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


def build_checkpoint(
    snapshots: List[dict],
    min_days: int = EDGE_MIN_DAYS,
) -> List[dict]:
    """One row per horse: latest return/alpha + curve risk + a verdict.

    Sorted by alpha descending (horses with no alpha sink to the bottom), then
    by equity, so the most market-beating horse reads first.
    """
    latest = _latest_by_strategy(snapshots)
    risk = compute_risk_metrics(snapshots)
    alpha_series = _alpha_series_by_strategy(snapshots)
    windows = _curve_window_by_strategy(snapshots)

    rows: List[dict] = []
    for strategy, rec in latest.items():
        m = risk.get(strategy, {})
        max_dd = m.get("max_drawdown_pct")
        n_days = m.get("n_days", 0)
        series = alpha_series.get(strategy, [])
        alpha_days = sum(1 for a in series if a is not None)
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
                "verdict": classify_edge(series, max_dd, min_days),
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

    The PLAN names this as the central statistical risk: with 3 horses over
    ~10 days, the single best one beating SPY is partly a selection effect, not
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


def _trading_days_after(start: date, end: date) -> int:
    """Count Mon–Fri days strictly after ``start`` up to and including ``end``.

    Weekends never count, so a checkpoint read on Monday over a Friday curve
    isn't penalised for the weekend — only genuine missed trading days show up.
    """
    if end <= start:
        return 0
    count = 0
    cur = start + timedelta(days=1)
    while cur <= end:
        if cur.weekday() < 5:  # Mon=0 … Fri=4
            count += 1
        cur += timedelta(days=1)
    return count


def _staleness_note(rows: List[dict], as_of: date) -> Optional[str]:
    """Warn when the curve's latest snapshot is too many trading days behind today.

    ``checkpoint_report`` reads only the persisted curve and has no clock, so a
    frozen curve (the L–V cron stopped: Mac asleep, LaunchAgent broken) would
    print an old snapshot as if current — and the operator could make the
    irreversible real-money call on stale numbers. One trading day of lag is just
    today's pending run; ``STALE_TRADING_DAYS`` or more means the cron is behind.

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


def format_checkpoint(
    rows: List[dict],
    min_days: int = EDGE_MIN_DAYS,
    as_of: Optional[date] = None,
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
    stale_note = _staleness_note(rows, as_of)
    if stale_note:
        lines.append(stale_note)
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
