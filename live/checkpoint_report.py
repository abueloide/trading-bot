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
    return "edge candidate"


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


def format_checkpoint(rows: List[dict], min_days: int = EDGE_MIN_DAYS) -> str:
    """Render the consolidated checkpoint table with an honest footer."""
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
