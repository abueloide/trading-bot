from __future__ import annotations

import pandas as pd

from live.orchestrator import Orchestrator, StrategyConfig
from live.virtual_portfolio import VirtualPortfolio


class FakeBars:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_bars(self, symbol: str, lookback: int) -> pd.DataFrame:
        return self._df


class RecordingExecutor:
    def __init__(self):
        self.buys = []
        self.sells = []

    def buy(self, *, symbol, qty, price, strategy, strategy_type, max_hold_days):
        self.buys.append((symbol, qty, strategy))
        return True

    def sell(self, *, symbol, qty, price, strategy):
        self.sells.append((symbol, qty, strategy))
        return True


def test_buy_signal_places_tagged_order_and_updates_ledger(oversold_then_bars):
    execu = RecordingExecutor()
    cfg = [StrategyConfig(strategy="confirmed_mr", symbols=["AAPL"], starting_cash=1000.0)]
    orch = Orchestrator(cfg, FakeBars(oversold_then_bars), execu,
                        risk_config={"max_position_pct": 1.0, "min_cash_reserve_pct": 0.0})
    orch.run_cycle()
    assert len(execu.buys) == 1
    assert execu.buys[0][0] == "AAPL"
    assert execu.buys[0][2] == "confirmed_mr"
    assert orch.portfolio("confirmed_mr").qty("AAPL") > 0


def test_hold_signal_places_no_order(rising_bars):
    execu = RecordingExecutor()
    cfg = [StrategyConfig(strategy="confirmed_mr", symbols=["AAPL"], starting_cash=1000.0)]
    orch = Orchestrator(cfg, FakeBars(rising_bars), execu)
    orch.run_cycle()
    assert execu.buys == []


def test_sell_signal_only_when_holding(oversold_then_bars):
    execu = RecordingExecutor()
    cfg = [StrategyConfig(strategy="confirmed_mr", symbols=["AAPL"], starting_cash=1000.0)]
    # Build an exit bar: append strong up bars so RSI(2) > 65 on the last row.
    bars = oversold_then_bars.copy()
    last_idx = pd.date_range(bars.index[-1], periods=6, freq="B")[1:]
    up = pd.DataFrame({c: [bars["close"].iloc[-1] * (1.05 ** (i + 1)) for i in range(5)]
                       for c in ["open", "high", "low", "close"]}, index=last_idx)
    up["volume"] = 1_000_000.0
    bars2 = pd.concat([bars, up])
    orch = Orchestrator(cfg, FakeBars(bars2), execu,
                        risk_config={"max_position_pct": 1.0, "min_cash_reserve_pct": 0.0})
    orch.portfolio("confirmed_mr").record_buy("AAPL", qty=1.0, price=50.0)
    orch.run_cycle()
    assert any(s[0] == "AAPL" and s[2] == "confirmed_mr" for s in execu.sells)
