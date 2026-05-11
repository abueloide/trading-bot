#!/usr/bin/env python3
"""
CLI entry point for backtests.

Usage:
    python backtesting/run_backtest.py --strategy rsi_mr --start 2022-01-01 --end 2024-12-31
    python backtesting/run_backtest.py --strategy ema_crossover --symbols AAPL,MSFT,SPY
    python backtesting/run_backtest.py --strategy momentum_rotation --no-walk-forward

Outputs go to backtesting/results/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List

# Allow `python backtesting/run_backtest.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.engine import run_backtest  # noqa: E402
from backtesting.strategies import STRATEGY_REGISTRY  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _default_symbols() -> List[str]:
    try:
        from stock_universe import TRADING_PRESETS
        return TRADING_PRESETS["BALANCED"]["symbols"]
    except Exception:
        return ["AAPL", "MSFT", "SPY"]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward backtest runner")
    parser.add_argument("--strategy", required=True, choices=list(STRATEGY_REGISTRY.keys()))
    parser.add_argument("--start", type=_parse_date, default=date(2022, 1, 1))
    parser.add_argument("--end", type=_parse_date, default=date.today())
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbol list")
    parser.add_argument("--initial-capital", type=float, default=10_000)
    parser.add_argument("--position-size-pct", type=float, default=0.25)
    parser.add_argument("--in-sample-months", type=int, default=24)
    parser.add_argument("--oos-months", type=int, default=6)
    parser.add_argument("--no-walk-forward", action="store_true")
    parser.add_argument("--data-source", choices=["alpaca", "yfinance"], default="alpaca")
    args = parser.parse_args(argv)

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",")]
        if args.symbols
        else _default_symbols()
    )
    spec = STRATEGY_REGISTRY[args.strategy]

    print(f"Running backtest: {args.strategy} ({spec['description']})")
    print(f"  Symbols ({len(symbols)}): {', '.join(symbols)}")
    print(f"  Window: {args.start} → {args.end}")
    print(f"  Walk-forward: {not args.no_walk_forward}")
    print(f"  Transaction cost: 0.05% slippage per side (always)")

    results = run_backtest(
        strategy_name=args.strategy,
        strategy_fn=spec["fn"],
        strategy_type=spec["type"],
        max_hold_days=spec.get("max_hold_days"),
        symbols=symbols,
        start=args.start,
        end=args.end,
        initial_capital=args.initial_capital,
        position_size_pct=args.position_size_pct,
        walk_forward=not args.no_walk_forward,
        in_sample_months=args.in_sample_months,
        oos_months=args.oos_months,
        data_source=args.data_source,
    )

    if not results:
        print("\nNo results — check data availability and credentials.")
        return 1

    print(f"\n{'Symbol':<8} {'Return%':>10} {'CAGR%':>10} {'Sharpe':>8} {'MaxDD%':>10} {'WinRt%':>8} {'PF':>6} {'N':>5} {'vs SPY%':>10}")
    print("-" * 95)
    for sym, r in results.items():
        print(
            f"{sym:<8} {r.total_return_pct:>10.2f} {r.cagr_pct:>10.2f} "
            f"{r.sharpe:>8.2f} {r.max_drawdown_pct:>10.2f} {r.win_rate_pct:>8.1f} "
            f"{r.profit_factor:>6.2f} {r.n_trades:>5d} {r.excess_return_pct:>10.2f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
