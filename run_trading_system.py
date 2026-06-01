#!/usr/bin/env python3
"""Run the paper horse race: 3 backtested strategies, one paper account."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from executor import Executor
from live.ledger_store import load_ledgers, save_ledgers
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
STATE_PATH = Path("data/ledgers/state.json")

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
    initial_states = load_ledgers(STATE_PATH)
    executor = Executor()
    bars = YFinanceBars()
    orch = Orchestrator(STRATEGIES, bars, LiveExecutorAdapter(executor),
                        initial_states=initial_states)
    orch.run_cycle()
    # NOTE: executor.check_time_exits() is intentionally NOT called here in v1.
    # Time-exit reconciliation requires per-strategy ledger lookup to know which
    # strategy's shares are being closed (same CORE invariant as SELL). This is a
    # follow-up item; close_position() has the same whole-position bug as the old sell.
    portfolios = [orch.portfolio(s.strategy) for s in STRATEGIES]
    save_ledgers(portfolios, STATE_PATH)
    logger.info("ledger state saved to %s", STATE_PATH)
    marks = {}
    for sym in UNIVERSE:
        df = bars.get_bars(sym, 2)
        if df is not None and len(df):
            marks[sym] = float(df["close"].iloc[-1])
    rows = build_report(portfolios, marks=marks)
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
