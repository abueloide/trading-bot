"""Horse-race report — side-by-side equity/return per strategy ledger.

When a benchmark return is supplied, each row also carries ``alpha_pct``
(strategy return minus the market) so the table answers the only question that
matters for a real-money decision: did the horse beat buy-and-hold, or just the
other horses?
"""
from __future__ import annotations

from typing import Dict, List, Optional

from live.virtual_portfolio import VirtualPortfolio


def build_report(
    portfolios: List[VirtualPortfolio],
    marks: Dict[str, float],
    benchmark_pct: Optional[float] = None,
) -> List[dict]:
    rows = []
    for vp in portfolios:
        state = vp.to_portfolio_state(marks)
        ret = ((state.equity / vp.starting_cash) - 1.0) * 100.0 if vp.starting_cash else 0.0
        row = {
            "strategy": vp.strategy,
            "equity": state.equity,
            "cash": vp.cash,
            "realized_pnl": vp.realized_pnl,
            "return_pct": ret,
            "n_positions": len(state.positions),
        }
        if benchmark_pct is not None:
            row["alpha_pct"] = ret - benchmark_pct
        rows.append(row)
    rows.sort(key=lambda r: r["equity"], reverse=True)
    return rows


def format_table(rows: List[dict], benchmark: Optional[dict] = None) -> str:
    has_alpha = bool(rows) and "alpha_pct" in rows[0]
    header = f"{'strategy':<20}{'equity':>12}{'return%':>10}{'realized':>12}{'pos':>5}"
    if has_alpha:
        header += f"{'alpha%':>9}"
    lines = [header, "-" * len(header)]
    for r in rows:
        line = (
            f"{r['strategy']:<20}{r['equity']:>12.2f}{r['return_pct']:>10.2f}"
            f"{r['realized_pnl']:>12.2f}{r['n_positions']:>5}"
        )
        if has_alpha:
            line += f"{r['alpha_pct']:>9.2f}"
        lines.append(line)
    if benchmark is not None:
        lines.append("-" * len(header))
        lines.append(
            f"benchmark {benchmark['symbol']} buy&hold since {benchmark['start_date']}: "
            f"{benchmark['return_pct']:+.2f}%  (alpha% = strategy return − this)"
        )
    return "\n".join(lines)
