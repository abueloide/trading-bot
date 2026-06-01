#!/usr/bin/env python3
"""Run the paper horse race: 3 backtested strategies, one paper account."""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from executor import Executor
from live.live_executor_adapter import LiveExecutorAdapter
from live.orchestrator import Orchestrator, StrategyConfig
from live.yfinance_bars import YFinanceBars
from live.horse_race_report import build_report, format_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("horse_race")

UNIVERSE = ["SPY", "AAPL", "MSFT", "QQQ", "NVDA"]
SLICE = 33_000.0  # virtual cash per strategy (paper)

STRATEGIES = [
    StrategyConfig("momentum_rotation", UNIVERSE, SLICE),
    StrategyConfig("confirmed_mr", UNIVERSE, SLICE),
    StrategyConfig("rsi_mr", UNIVERSE, SLICE),
]


def main() -> int:
    load_dotenv()
    base_url = os.getenv("ALPACA_BASE_URL", "")
    if not base_url.startswith("https://paper-api.alpaca.markets"):
        raise SystemExit(
            "Refusing to run: ALPACA_BASE_URL is not a paper endpoint "
            f"(got {base_url!r}). This bot only runs on paper."
        )
    executor = Executor()
    bars = YFinanceBars()
    orch = Orchestrator(STRATEGIES, bars, LiveExecutorAdapter(executor))
    orch.run_cycle()
    # NOTE: executor.check_time_exits() is intentionally NOT called here in v1.
    # Time-exit reconciliation requires per-strategy ledger lookup to know which
    # strategy's shares are being closed (same CORE invariant as SELL). This is a
    # follow-up item; close_position() has the same whole-position bug as the old sell.
    marks = {}
    for sym in UNIVERSE:
        df = bars.get_bars(sym, 2)
        if df is not None and len(df):
            marks[sym] = float(df["close"].iloc[-1])
    rows = build_report([orch.portfolio(s.strategy) for s in STRATEGIES], marks=marks)
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
