"""Orchestrator — one cycle of the paper horse race.

For each strategy and its symbols: fetch trailing bars, get a live signal,
size BUYs through the RiskManager against that strategy's virtual ledger,
and route BUY/SELL to the ExecutorPort with a strategy tag. SELLs only fire
for symbols the strategy currently holds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from live.ports import BarProvider, ExecutorPort
from live.strategy_runner import Signal, StrategyRunner
from live.virtual_portfolio import VirtualPortfolio
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

LOOKBACK_BARS = 260  # ~1y of daily bars; enough for 200d / momentum filters.


@dataclass
class StrategyConfig:
    strategy: str
    symbols: List[str]
    starting_cash: float


class Orchestrator:
    def __init__(
        self,
        configs: List[StrategyConfig],
        bars: BarProvider,
        executor: ExecutorPort,
        risk_config: Optional[dict] = None,
    ) -> None:
        self._bars = bars
        self._executor = executor
        self._risk = RiskManager(risk_config)
        self._runners: Dict[str, StrategyRunner] = {}
        self._portfolios: Dict[str, VirtualPortfolio] = {}
        self._symbols: Dict[str, List[str]] = {}
        for c in configs:
            self._runners[c.strategy] = StrategyRunner(c.strategy)
            self._portfolios[c.strategy] = VirtualPortfolio(c.strategy, c.starting_cash)
            self._symbols[c.strategy] = list(c.symbols)

    def portfolio(self, strategy: str) -> VirtualPortfolio:
        return self._portfolios[strategy]

    def run_cycle(self) -> None:
        for strategy, runner in self._runners.items():
            vp = self._portfolios[strategy]
            for symbol in self._symbols[strategy]:
                try:
                    df = self._bars.get_bars(symbol, LOOKBACK_BARS)
                except Exception as e:
                    logger.warning("bars fetch failed %s/%s: %s", strategy, symbol, e)
                    continue
                if df is None or len(df) == 0:
                    continue
                sig = runner.run(symbol, df)
                if sig.action == "BUY":
                    self._handle_buy(runner, vp, sig)
                elif sig.action == "SELL":
                    self._handle_sell(vp, sig)

    def _handle_buy(self, runner: StrategyRunner, vp: VirtualPortfolio, sig: Signal) -> None:
        if vp.qty(sig.symbol) > 0:
            return  # already holding for this strategy; no pyramiding in v1
        state = vp.to_portfolio_state(marks={sig.symbol: sig.price})
        # Size against deployable CASH, not equity: with open positions held by
        # this strategy, equity > cash, and the risk manager could otherwise
        # approve more than the slice can fund, making record_buy raise after
        # the broker already filled (ledger/reality drift).
        proposed = state.cash
        decision = self._risk.evaluate_entry(
            symbol=sig.symbol,
            proposed_size_usd=proposed,
            proposed_price=sig.price,
            sector=None,
            portfolio=state,
            is_day_trade=False,
            pdt_can_day_trade=True,
        )
        if not decision.allowed or decision.adjusted_qty <= 0:
            logger.info("entry denied %s/%s: %s", runner.name, sig.symbol, decision.reason)
            return
        qty = decision.adjusted_qty
        if self._executor.buy(
            symbol=sig.symbol, qty=qty, price=sig.price, strategy=runner.name,
            strategy_type=runner.strategy_type, max_hold_days=runner.max_hold_days,
        ):
            try:
                vp.record_buy(sig.symbol, qty, sig.price)
            except ValueError as e:
                logger.critical(
                    "LEDGER DRIFT %s/%s: broker filled but ledger rejected the buy: %s",
                    runner.name, sig.symbol, e,
                )

    def _handle_sell(self, vp: VirtualPortfolio, sig: Signal) -> None:
        held = vp.qty(sig.symbol)
        if held <= 0:
            return
        if self._executor.sell(
            symbol=sig.symbol, qty=held, price=sig.price, strategy=vp.strategy
        ):
            vp.record_sell(sig.symbol, held, sig.price)
