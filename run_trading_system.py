#!/usr/bin/env python3
"""Run the paper horse race: 3 backtested strategies, one paper account.

Universe = the full static S&P 500 snapshot (no runtime network for membership;
see sp500_constituents.py). Bars are downloaded ONCE in a batch and reused for
the cycle, the monthly-rebalance check, and end-of-run marks.
"""
from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from dotenv import load_dotenv

from executor import Executor
from live.benchmark import compute_benchmark
from live.equity_snapshot import append_snapshot, load_snapshots
from live.ledger_store import load_ledgers, save_ledgers
from live.risk_metrics import compute_risk_metrics, format_risk_table
from live.live_executor_adapter import LiveExecutorAdapter
from live.orchestrator import LOOKBACK_BARS, Orchestrator, StrategyConfig
from live.yfinance_bars import CachedBars, YFinanceBars, drop_in_progress_bars
from live.horse_race_report import build_report, format_table
from stock_universe import sp500_symbols

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("horse_race")

SLICE = 25_000.0  # virtual cash per strategy (paper) — 4 horses × $25k = $100k
STATE_PATH = Path("data/ledgers/state.json")
EQUITY_CURVE_PATH = Path("data/ledgers/equity_curve.jsonl")
REBALANCE_REFERENCE = "SPY"  # market-calendar anchor for the monthly rebalance
BENCHMARK_SYMBOL = "SPY"  # buy-and-hold yardstick for the alpha column
# Inception of the clean $25k×N epoch: the ledgers were reset to $25k and the
# Alpaca paper account flattened when the 4th horse (donchian_breakout) joined,
# so all four read 0.00% on 2026-06-25 (see scripts/reset_race.py + cron.log).
# Alpha is measured from here so the benchmark covers the same window as the
# live ledgers.
RACE_INCEPTION = date(2026, 6, 25)

# Risk overlay tuned for a diversified equal-weight horse race: many small
# equal-weight slots (momentum 15, MR 10), near-full deployment, no sector cap
# (this is a pure strategy comparison; a sector overlay is a separate concern).
HORSE_RISK_CONFIG = {
    "max_open_positions": 50,
    "min_cash_reserve_pct": 0.0,
    "max_cash_reserve_pct": 0.05,
    "max_position_pct": 0.15,
    "max_sector_exposure_pct": 1.0,
}

# The full S&P 500 universe feeds every strategy; each builds its own basket.
UNIVERSE = sp500_symbols()

# Symbols fetched in the batch = the tradeable universe PLUS the benchmark /
# rebalance yardstick (SPY). SPY is NOT an S&P 500 constituent, so without this
# it was never in the snapshot and compute_benchmark() silently returned None on
# every run (alpha_pct: null in the equity curve, alpha column dropped from the
# report). The orchestrator only ever requests its own strategy symbols from the
# cache, so SPY's bars feed the benchmark and the rebalance calendar without ever
# becoming a tradeable position.
_EXTRA_SYMBOLS = list(
    dict.fromkeys(s for s in (BENCHMARK_SYMBOL, REBALANCE_REFERENCE) if s not in UNIVERSE)
)
STRATEGIES = [
    StrategyConfig("momentum_rotation", UNIVERSE, SLICE, max_positions=15),
    StrategyConfig("confirmed_mr", UNIVERSE, SLICE, max_positions=10),
    StrategyConfig("rsi_mr", UNIVERSE, SLICE, max_positions=10),
    # 4th horse (2026-06-25): Donchian 20/10 breakout — a trend-following style
    # distinct from momentum (6m winners) and mean-reversion (oversold dips).
    StrategyConfig("donchian_breakout", UNIVERSE, SLICE, max_positions=10),
    # C2 — OpEx 1d-drift (long-only). Universo fijo SPY/QQQ: es una anomalía de
    # microestructura de índices, no un screen sobre el S&P 500. IVV (no SPY):
    # SPY es el benchmark y la invariante dice que la vara no se opera; IVV sigue
    # el mismo indice y replica el edge medido (+0.06%/+0.56% por evento).
    # max_positions=6 (no 2) es SIZING, no un limite de nombres: el peso es
    # slice/slots, y con 2 slots cada nombre pesaria 25% > cap duro de 20% y el
    # risk gate VETA la orden (verificado 2026-07-26). Con 6 -> 16.7%, pasa.
    StrategyConfig("opex_drift", ["IVV", "QQQ"], SLICE, max_positions=6),
    # NOTE: a 4th horse (momentum_news) was retired 2026-06-13. It paired the
    # momentum engine with an AlphaVantage NEWS_SENTIMENT veto, but the free tier
    # cannot serve it: NEWS_SENTIMENT returns 0 articles for a multi-ticker basket
    # (and "Invalid inputs" past ~15 tickers), so fetch_sentiment always came back
    # empty and the overlay was a permanent no-op — momentum_news was byte-for-byte
    # momentum_rotation. The overlay code (live/news_overlay.py) stays for a future
    # revival with a per-ticker fetch + a paid/alternate news source.
]

# Fetch = TODO símbolo que alguna estrategia pueda operar + los extras (benchmark,
# calendario). Se computa DESPUÉS de STRATEGIES: una estrategia con universo propio
# (p.ej. opex_drift → IVV/QQQ) quedaría sin barras si esto se derivara solo de
# UNIVERSE — la misma clase de bug que dejó el alpha en null en junio 2026.
FETCH_SYMBOLS = list(
    dict.fromkeys(
        list(UNIVERSE)
        + [sym for cfg in STRATEGIES for sym in cfg.symbols]
        + _EXTRA_SYMBOLS
    )
)


def _is_first_trading_day_of_month(snapshot: Dict[str, pd.DataFrame]) -> bool:
    """True when the latest bar is the first trading day of its month.

    Uses the SPY calendar (falls back to any available symbol) so the monthly
    momentum rebalance lands on a real session, not a weekend/holiday guess.
    """
    df = snapshot.get(REBALANCE_REFERENCE)
    if df is None or len(df) == 0:
        for candidate in snapshot.values():
            if candidate is not None and len(candidate):
                df = candidate
                break
    if df is None or len(df) == 0:
        return False
    idx = df.index
    last = idx[-1]
    earlier_same_month = any(
        d.year == last.year and d.month == last.month and d < last for d in idx
    )
    return not earlier_same_month


def resolve_benchmark(
    snapshot: Dict[str, pd.DataFrame],
    inception: date = RACE_INCEPTION,
) -> Tuple[Optional[dict], Optional[float]]:
    """Compute the SPY buy&hold benchmark, loudly flagging the silent-null case.

    The alpha column went dark for 10 days because ``compute_benchmark`` returned
    None every run and nothing complained — every equity-curve snapshot quietly
    carried ``alpha_pct: null``. If the benchmark can't be computed, alpha is null
    for the whole window, so shout it into ``cron.log`` here instead of letting the
    2-week checkpoint discover a useless curve. ``in_snapshot`` distinguishes the
    "SPY never fetched" bug from the subtler "fetched but no usable bar at/after
    inception" one.
    """
    benchmark = compute_benchmark(snapshot, BENCHMARK_SYMBOL, inception)
    if benchmark is None:
        logger.warning(
            "benchmark %s could not be computed (in_snapshot=%s) -> alpha will be "
            "null this run; check the SPY fetch/inception wiring",
            BENCHMARK_SYMBOL, BENCHMARK_SYMBOL in snapshot,
        )
        return None, None
    return benchmark, benchmark["return_pct"]


def main() -> int:
    load_dotenv()
    base_url = os.getenv("ALPACA_BASE_URL", "")
    if not base_url.startswith("https://paper-api.alpaca.markets"):
        raise SystemExit(
            "Refusing to run: ALPACA_BASE_URL is not a paper endpoint "
            f"(got {base_url!r}). This bot only runs on paper."
        )

    logger.info("fetching bars for %d symbols (batch)...", len(FETCH_SYMBOLS))
    snapshot = YFinanceBars().get_bars_batch(FETCH_SYMBOLS, LOOKBACK_BARS)
    # The 13:00 CST cron runs mid-session, so yfinance's trailing bar is today's
    # in-progress (partial) print. Trade and snapshot on settled closes only.
    snapshot = drop_in_progress_bars(snapshot, date.today())
    if not snapshot:
        raise SystemExit("No bars returned for the universe; aborting (no trades).")
    cached = CachedBars(snapshot)
    is_rebalance = _is_first_trading_day_of_month(snapshot)
    logger.info("universe bars: %d/%d usable; benchmark(%s)=%s; rebalance_day=%s",
                len(snapshot), len(FETCH_SYMBOLS), BENCHMARK_SYMBOL,
                BENCHMARK_SYMBOL in snapshot, is_rebalance)

    initial_states = load_ledgers(STATE_PATH)
    executor = Executor()
    orch = Orchestrator(STRATEGIES, cached, LiveExecutorAdapter(executor),
                        risk_config=HORSE_RISK_CONFIG, initial_states=initial_states)
    orch.run_cycle(is_rebalance_day=is_rebalance)
    # NOTE: executor.check_time_exits() is intentionally NOT called here in v1.
    # Time-exit reconciliation requires per-strategy ledger lookup to know which
    # strategy's shares are being closed (same CORE invariant as SELL). This is a
    # follow-up item; close_position() has the same whole-position bug as the old sell.

    portfolios = [orch.portfolio(s.strategy) for s in STRATEGIES]
    save_ledgers(portfolios, STATE_PATH)
    logger.info("ledger state saved to %s", STATE_PATH)

    marks = {sym: float(df["close"].iloc[-1]) for sym, df in snapshot.items() if len(df)}
    # No silent cost-basis fallback: VirtualPortfolio.to_portfolio_state marks a
    # held name with no fresh price at its entry cost (avg_entry), which fabricates
    # that position's value — equity then looks calmer than reality on exactly the
    # days data is flaky, with zero trace. Coverage is ~full today (503-504/504),
    # but shout the day it isn't so a tainted equity/return is never silent.
    held_unmarked = sorted(
        {sym for vp in portfolios for sym in vp.to_dict()["lots"] if sym not in marks}
    )
    if held_unmarked:
        logger.warning(
            "%d HELD symbols missing a fresh mark -> valued at entry cost (equity "
            "understated/tainted this run): %s",
            len(held_unmarked), ", ".join(held_unmarked),
        )
    benchmark, benchmark_pct = resolve_benchmark(snapshot)
    rows = build_report(portfolios, marks=marks, benchmark_pct=benchmark_pct)
    print(format_table(rows, benchmark=benchmark))

    # Persist a daily equity/alpha snapshot per horse so the 2-week checkpoint
    # can read an equity *curve* (drawdown, alpha stability), not just the last
    # cut. Stamp it with the latest bar's date (matches the benchmark window and
    # stays idempotent if the cron re-fires the same session).
    snapshot_day = max(
        (df.index[-1].date() for df in snapshot.values() if len(df)),
        default=date.today(),
    )
    append_snapshot(rows, benchmark_pct, snapshot_day, EQUITY_CURVE_PATH)
    logger.info("equity snapshot appended for %s to %s", snapshot_day, EQUITY_CURVE_PATH)

    # Risk side of the ledger: a horse can lead on return while riding a brutal
    # drawdown. Read the curve we just extended and print max-drawdown / vol so
    # the 2-week checkpoint reads risk-adjusted, not raw, return.
    risk = compute_risk_metrics(load_snapshots(EQUITY_CURVE_PATH))
    print(format_risk_table(risk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
