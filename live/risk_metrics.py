"""Risk metrics over the equity curve — return alone can't pick a horse.

A horse that wins the window by riding a brutal drawdown is not an edge; it is
luck plus leverage of timing. Reading the per-strategy equity *curve*
(``equity_snapshot``'s JSONL) we can put two risk numbers next to the return:

- ``max_drawdown_pct``: worst peak-to-trough decline along the curve (≤ 0).
- ``volatility_pct``: sample stdev of day-over-day returns (dispersion of the
  ride). ``None`` until there are at least two daily returns to disperse.

Both are descriptive only — no decision is automated here. They feed the
2-week checkpoint so the human reads risk-adjusted, not just raw, return.
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Optional


def _equity_series(records: List[dict]) -> List[float]:
    """Chronological list of usable equity marks for one strategy.

    Sorted by date; rows without a positive ``equity`` are dropped so a bad
    snapshot can't poison drawdown or inject a spurious return.
    """
    usable = [r for r in records if isinstance(r.get("equity"), (int, float)) and r["equity"] > 0]
    usable.sort(key=lambda r: r.get("date", ""))
    return [float(r["equity"]) for r in usable]


def _max_drawdown_pct(series: List[float]) -> float:
    """Most negative peak-to-trough decline along the series, as a percent."""
    peak = series[0]
    worst = 0.0
    for equity in series:
        if equity > peak:
            peak = equity
        drawdown = (equity / peak - 1.0) * 100.0
        if drawdown < worst:
            worst = drawdown
    return worst


def _volatility_pct(series: List[float]) -> Optional[float]:
    """Sample stdev of day-over-day returns, as a percent. None if <2 returns."""
    returns = [series[i] / series[i - 1] - 1.0 for i in range(1, len(series))]
    if len(returns) < 2:
        return None
    return statistics.stdev(returns) * 100.0


def compute_risk_metrics(snapshots: List[dict]) -> Dict[str, dict]:
    """Map strategy → {max_drawdown_pct, volatility_pct, n_days} from snapshots."""
    by_strategy: Dict[str, List[dict]] = {}
    for rec in snapshots:
        strategy = rec.get("strategy")
        if strategy is not None:
            by_strategy.setdefault(strategy, []).append(rec)

    metrics: Dict[str, dict] = {}
    for strategy, records in by_strategy.items():
        series = _equity_series(records)
        if not series:
            continue
        metrics[strategy] = {
            "max_drawdown_pct": _max_drawdown_pct(series),
            "volatility_pct": _volatility_pct(series),
            "n_days": len(series),
        }
    return metrics


def format_risk_table(metrics: Dict[str, dict]) -> str:
    """Render the risk table, sorted by shallowest drawdown (least painful first)."""
    if not metrics:
        return "risk: no equity curve yet (needs ≥2 snapshot days to mean anything)"

    header = f"{'strategy':<20}{'max_dd%':>10}{'vol%':>9}{'days':>6}"
    lines = [header, "-" * len(header)]
    ordered = sorted(metrics.items(), key=lambda kv: kv[1]["max_drawdown_pct"], reverse=True)
    for strategy, m in ordered:
        vol = m["volatility_pct"]
        vol_str = f"{vol:>9.2f}" if vol is not None else f"{'n/a':>9}"
        lines.append(f"{strategy:<20}{m['max_drawdown_pct']:>10.2f}{vol_str}{m['n_days']:>6}")
    return "\n".join(lines)
