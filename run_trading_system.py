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
    if "paper" not in base_url:
        raise SystemExit(
            "Refusing to run: ALPACA_BASE_URL is not a paper endpoint "
            f"(got {base_url!r}). This bot only runs on paper."
        )
    executor = Executor()
    orch = Orchestrator(STRATEGIES, YFinanceBars(), LiveExecutorAdapter(executor))
    orch.run_cycle()
    rows = build_report([orch.portfolio(s.strategy) for s in STRATEGIES], marks={})
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
