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
    alpha_pct: Optional[float],
    max_drawdown_pct: Optional[float],
    n_days: int,
    min_days: int = EDGE_MIN_DAYS,
) -> str:
    """Plain-language reading of one horse's numbers. Never a decision."""
    if n_days < min_days:
        return f"inconclusive (need ≥{min_days} days)"
    if alpha_pct is None:
        return "no benchmark"
    if alpha_pct <= 0:
        return "no edge (alpha ≤ 0)"
    if max_drawdown_pct is not None and max_drawdown_pct < DEEP_DRAWDOWN_PCT:
        return "edge? but deep-drawdown risk"
    return "edge candidate"


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

    rows: List[dict] = []
    for strategy, rec in latest.items():
        m = risk.get(strategy, {})
        max_dd = m.get("max_drawdown_pct")
        n_days = m.get("n_days", 0)
        alpha = rec.get("alpha_pct")
        rows.append(
            {
                "strategy": strategy,
                "date": rec.get("date"),
                "equity": rec.get("equity"),
                "return_pct": rec.get("return_pct"),
                "alpha_pct": alpha,
                "benchmark_pct": rec.get("benchmark_pct"),
                "max_drawdown_pct": max_dd,
                "volatility_pct": m.get("volatility_pct"),
                "n_days": n_days,
                "gap_days": m.get("gap_days", 0),
                "verdict": classify_edge(alpha, max_dd, n_days, min_days),
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

    header = (
        f"{'strategy':<20}{'equity':>12}{'return%':>10}{'alpha%':>9}"
        f"{'max_dd%':>10}{'vol%':>9}{'days':>6}  verdict"
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
            f"{dd_s}{vol_s}{r['n_days']:>6}  {r['verdict']}"
        )

    lines.append("-" * len(header))
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
