"""Horse-race report — side-by-side equity/return per strategy ledger."""
from __future__ import annotations

from typing import Dict, List

from live.virtual_portfolio import VirtualPortfolio


def build_report(portfolios: List[VirtualPortfolio], marks: Dict[str, float]) -> List[dict]:
    rows = []
    for vp in portfolios:
        state = vp.to_portfolio_state(marks)
        ret = ((state.equity / vp.starting_cash) - 1.0) * 100.0 if vp.starting_cash else 0.0
        rows.append({
            "strategy": vp.strategy,
            "equity": state.equity,
            "cash": vp.cash,
            "realized_pnl": vp.realized_pnl,
            "return_pct": ret,
            "n_positions": len(state.positions),
        })
    rows.sort(key=lambda r: r["equity"], reverse=True)
    return rows


def format_table(rows: List[dict]) -> str:
    header = f"{'strategy':<20}{'equity':>12}{'return%':>10}{'realized':>12}{'pos':>5}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['strategy']:<20}{r['equity']:>12.2f}{r['return_pct']:>10.2f}"
            f"{r['realized_pnl']:>12.2f}{r['n_positions']:>5}"
        )
    return "\n".join(lines)
