"""Orchestrator — one cycle of the paper horse race.

Each strategy now builds a *portfolio*, not a stream of per-symbol signals:

  * momentum  → rank the whole universe by trailing momentum, hold the top-N
                equal-weight, and only rotate on a monthly rebalance day.
  * mean-reversion → scan the universe, close recovered names, then fill open
                slots with the most-oversold candidates (RSI(2) ascending),
                each at an equal weight.

BUYs are sized to a fixed slice weight and funded from CASH through the
RiskManager (never over-fund the slice → no ledger/reality drift). SELLs only
ever dispose of shares THIS strategy currently holds, by exact quantity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from live.news_overlay import apply_news_overlay, fetch_sentiment
from live.portfolio_targets import (
    exit_signals,
    momentum_top,
    oversold_candidates,
)
from live.ports import ExecutorPort
from live.strategy_runner import StrategyRunner
from live.virtual_portfolio import VirtualPortfolio
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

LOOKBACK_BARS = 260  # ~1y of daily bars; enough for 200d / momentum filters.

# Default basket sizes per strategy type (number of equal-weight slots).
DEFAULT_SLOTS = {"momentum": 15, "mean_reversion": 10}
_FALLBACK_SLOTS = 5

# When a momentum horse runs a news overlay, rank a deeper pool so vetoed names
# can be replaced by the next-best survivors instead of shrinking the basket.
NEWS_POOL_FACTOR = 3


@dataclass
class StrategyConfig:
    strategy: str
    symbols: List[str]
    starting_cash: float
    max_positions: Optional[int] = None  # override basket size; None → type default
    news_overlay: bool = False  # momentum horses only: veto negative-news names


def _last_price(df: Optional[pd.DataFrame]) -> Optional[float]:
    if df is None or len(df) == 0 or "close" not in df.columns:
        return None
    price = float(df["close"].iloc[-1])
    return price if price > 0 else None


class Orchestrator:
    def __init__(
        self,
        configs: List[StrategyConfig],
        bars,
        executor: ExecutorPort,
        risk_config: Optional[dict] = None,
        initial_states: Optional[Dict[str, dict]] = None,
        news_fetcher: Callable[[Sequence[str]], Dict[str, float]] = fetch_sentiment,
    ) -> None:
        self._bars = bars
        self._executor = executor
        self._risk = RiskManager(risk_config)
        self._news_fetcher = news_fetcher
        self._runners: Dict[str, StrategyRunner] = {}
        self._portfolios: Dict[str, VirtualPortfolio] = {}
        self._symbols: Dict[str, List[str]] = {}
        self._slots: Dict[str, int] = {}
        self._news_overlay: Dict[str, bool] = {}
        for c in configs:
            runner = StrategyRunner(c.strategy)
            self._runners[c.strategy] = runner
            if initial_states and c.strategy in initial_states:
                self._portfolios[c.strategy] = VirtualPortfolio.from_dict(initial_states[c.strategy])
            else:
                self._portfolios[c.strategy] = VirtualPortfolio(c.strategy, c.starting_cash)
            self._symbols[c.strategy] = list(c.symbols)
            self._slots[c.strategy] = c.max_positions or DEFAULT_SLOTS.get(
                runner.strategy_type, _FALLBACK_SLOTS
            )
            self._news_overlay[c.strategy] = c.news_overlay

    def portfolio(self, strategy: str) -> VirtualPortfolio:
        return self._portfolios[strategy]

    # ------------------------------------------------------------ cycle

    def run_cycle(self, is_rebalance_day: bool = False) -> None:
        bars_map = self._fetch_bars()
        for strategy, runner in self._runners.items():
            local = {s: bars_map.get(s) for s in self._symbols[strategy]}
            vp = self._portfolios[strategy]
            if runner.strategy_type == "momentum":
                self._run_momentum(runner, vp, local, is_rebalance_day)
            else:
                self._run_mean_reversion(runner, vp, local)

    def _fetch_bars(self) -> Dict[str, Optional[pd.DataFrame]]:
        symbols = sorted({s for syms in self._symbols.values() for s in syms})
        batch = getattr(self._bars, "get_bars_batch", None)
        if callable(batch):
            try:
                return dict(batch(symbols, LOOKBACK_BARS))
            except Exception as e:
                logger.warning("batch fetch failed, falling back per-symbol: %s", e)
        out: Dict[str, Optional[pd.DataFrame]] = {}
        for sym in symbols:
            try:
                out[sym] = self._bars.get_bars(sym, LOOKBACK_BARS)
            except Exception as e:
                logger.warning("bars fetch failed %s: %s", sym, e)
                out[sym] = None
        return out

    # ------------------------------------------------------------ momentum

    def _momentum_basket(self, name: str, local, n: int) -> List[str]:
        """Top-n momentum names, optionally filtered by news sentiment.

        With the overlay on, rank a deeper pool and drop names carrying
        materially negative news; survivors keep momentum order. If the news
        feed returns nothing (no key, rate limit, error), the overlay is a
        no-op and this degrades to plain momentum top-n.
        """
        if not self._news_overlay.get(name):
            return momentum_top(local, n)
        pool = momentum_top(local, n * NEWS_POOL_FACTOR)
        sentiment = self._news_fetcher(pool)
        basket = apply_news_overlay(pool, sentiment, n)
        vetoed = len(pool[:n]) - len(set(basket) & set(pool[:n]))
        logger.info(
            "news overlay %s: pool=%d sentiment=%d basket=%d (vetoed≈%d)",
            name, len(pool), len(sentiment), len(basket), max(vetoed, 0),
        )
        return basket

    def _run_momentum(self, runner, vp, local, is_rebalance_day: bool) -> None:
        held = {s for s in self._symbols[runner.name] if vp.qty(s) > 0}
        # Between monthly rebalances a populated book just rides; rotate only on
        # the rebalance day (or the very first deployment, when the book is empty).
        if held and not is_rebalance_day:
            return
        n = self._slots[runner.name]
        top = self._momentum_basket(runner.name, local, n)
        top_set = set(top)
        # Drop names that fell out of the top-N.
        for sym in sorted(held - top_set):
            price = _last_price(local.get(sym))
            if price is not None:
                self._do_sell(vp, sym, price)
        # Add new entrants, equal-weight.
        weight_dollars = vp.starting_cash / n
        for sym in top:
            if vp.qty(sym) > 0:
                continue
            price = _last_price(local.get(sym))
            if price is not None:
                self._do_buy(runner, vp, sym, price, weight_dollars)

    # -------------------------------------------------------- mean reversion

    def _run_mean_reversion(self, runner, vp, local) -> None:
        held = {s for s in self._symbols[runner.name] if vp.qty(s) > 0}
        # 1) Close recovered holdings first (frees both slots and cash).
        for sym in exit_signals(runner.name, local, held):
            price = _last_price(local.get(sym))
            if price is not None:
                self._do_sell(vp, sym, price)
        held = {s for s in self._symbols[runner.name] if vp.qty(s) > 0}
        # 2) Fill open slots with the most-oversold fresh candidates.
        slots = self._slots[runner.name] - len(held)
        if slots <= 0:
            return
        weight_dollars = vp.starting_cash / self._slots[runner.name]
        candidates = oversold_candidates(runner.name, local, exclude=held)
        for sym, _rsi2, price in candidates[:slots]:
            self._do_buy(runner, vp, sym, price, weight_dollars)

    # ------------------------------------------------------------ execution

    def _do_buy(self, runner, vp, symbol: str, price: float, proposed_dollars: float) -> None:
        if vp.qty(symbol) > 0:
            return  # no pyramiding in v1
        state = vp.to_portfolio_state(marks={symbol: price})
        # Never propose more than the slice's free cash can fund: with open
        # positions, equity > cash, and an equity-based size could pass risk but
        # fail record_buy after the broker already filled (ledger/reality drift).
        proposed = min(proposed_dollars, state.cash)
        if proposed <= 0:
            return
        decision = self._risk.evaluate_entry(
            symbol=symbol,
            proposed_size_usd=proposed,
            proposed_price=price,
            sector=None,
            portfolio=state,
            is_day_trade=False,
            pdt_can_day_trade=True,
        )
        if not decision.allowed or decision.adjusted_qty <= 0:
            logger.info("entry denied %s/%s: %s", runner.name, symbol, decision.reason)
            return
        qty = decision.adjusted_qty
        if self._executor.buy(
            symbol=symbol, qty=qty, price=price, strategy=runner.name,
            strategy_type=runner.strategy_type, max_hold_days=runner.max_hold_days,
        ):
            try:
                vp.record_buy(symbol, qty, price)
            except ValueError as e:
                logger.critical(
                    "LEDGER DRIFT %s/%s: broker filled but ledger rejected the buy: %s",
                    runner.name, symbol, e,
                )

    def _do_sell(self, vp, symbol: str, price: float) -> None:
        held = vp.qty(symbol)
        if held <= 0:
            return
        if self._executor.sell(symbol=symbol, qty=held, price=price, strategy=vp.strategy):
            vp.record_sell(symbol, held, price)
