from __future__ import annotations

import pandas as pd
import pytest

from live.orchestrator import Orchestrator, StrategyConfig
from live.virtual_portfolio import VirtualPortfolio


class FakeBars:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_bars(self, symbol: str, lookback: int) -> pd.DataFrame:
        return self._df


class KeyedBars:
    """Returns a different bar frame per symbol."""
    def __init__(self, by_symbol: dict):
        self._by = by_symbol

    def get_bars(self, symbol: str, lookback: int):
        return self._by.get(symbol)


class KeyedBatchBars:
    """Per-symbol frames, exposing the batch path the real provider uses."""
    def __init__(self, by_symbol: dict):
        self._by = by_symbol
        self.batch_calls = 0

    def get_bars_batch(self, symbols, lookback):
        self.batch_calls += 1
        return {s: self._by[s] for s in symbols if self._by.get(s) is not None}


def _ramp(slope: float, n: int = 200, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = pd.Series([start + slope * i for i in range(n)], index=idx, dtype="float64")
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": pd.Series(1_000_000, index=idx, dtype="float64"),
    })


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


_HORSE_RISK = {"max_position_pct": 1.0, "min_cash_reserve_pct": 0.0,
               "max_open_positions": 50, "max_sector_exposure_pct": 1.0}


def test_momentum_buys_top_n_by_rank_not_alphabetical():
    # WEAK ranks last by slope but first alphabetically — the old per-symbol loop
    # would have bought it; cross-sectional ranking must NOT.
    execu = RecordingExecutor()
    bars = KeyedBatchBars({
        "AAA_WEAK": _ramp(0.05),
        "MMM_MID": _ramp(0.5),
        "ZZZ_STRONG": _ramp(1.5),
    })
    cfg = [StrategyConfig("momentum_rotation", list(bars._by), 30_000.0, max_positions=2)]
    orch = Orchestrator(cfg, bars, execu, risk_config=_HORSE_RISK)
    orch.run_cycle()
    bought = {b[0] for b in execu.buys}
    assert bought == {"ZZZ_STRONG", "MMM_MID"}
    assert "AAA_WEAK" not in bought
    assert bars.batch_calls == 1  # used the batch path


def test_momentum_holds_between_rebalances():
    execu = RecordingExecutor()
    bars = KeyedBatchBars({"A": _ramp(1.5), "B": _ramp(1.0), "C": _ramp(0.5)})
    cfg = [StrategyConfig("momentum_rotation", list(bars._by), 30_000.0, max_positions=2)]
    orch = Orchestrator(cfg, bars, execu, risk_config=_HORSE_RISK)
    orch.run_cycle(is_rebalance_day=True)   # initial deployment: A, B
    execu.buys.clear()
    orch.run_cycle(is_rebalance_day=False)  # non-rebalance day → ride the book
    assert execu.buys == [] and execu.sells == []


def test_momentum_rotates_on_rebalance_day():
    execu = RecordingExecutor()
    # Book holds C (now weakest); D is strongest and should rotate in.
    bars = KeyedBatchBars({"A": _ramp(1.5), "B": _ramp(1.2), "C": _ramp(0.2), "D": _ramp(2.0)})
    cfg = [StrategyConfig("momentum_rotation", list(bars._by), 30_000.0, max_positions=2)]
    initial = {"momentum_rotation": {
        "strategy": "momentum_rotation", "starting_cash": 30_000.0,
        "cash": 10_000.0, "realized_pnl": 0.0,
        "lots": {"A": {"qty": 50.0, "avg_entry": 200.0},
                 "C": {"qty": 50.0, "avg_entry": 100.0}},
    }}
    orch = Orchestrator(cfg, bars, execu, risk_config=_HORSE_RISK, initial_states=initial)
    orch.run_cycle(is_rebalance_day=True)
    sold = {s[0] for s in execu.sells}
    bought = {b[0] for b in execu.buys}
    assert "C" in sold        # weakest dropped
    assert "D" in bought      # strongest rotated in
    assert "A" not in sold    # still top-2 → held


def test_news_overlay_vetoes_negative_name_and_refills_from_rank():
    # STRONG has the best momentum but carries negative news → it must be vetoed
    # and replaced by the next-best survivor.
    execu = RecordingExecutor()
    bars = KeyedBatchBars({
        "STRONG": _ramp(2.0),
        "MID": _ramp(1.0),
        "WEAK": _ramp(0.5),
    })
    cfg = [StrategyConfig("momentum_news", list(bars._by), 30_000.0,
                          max_positions=2, news_overlay=True)]
    sentiment = {"STRONG": -0.40, "MID": 0.10, "WEAK": 0.05}
    orch = Orchestrator(cfg, bars, execu, risk_config=_HORSE_RISK,
                        news_fetcher=lambda tickers: sentiment)
    orch.run_cycle()
    bought = {b[0] for b in execu.buys}
    assert bought == {"MID", "WEAK"}      # STRONG vetoed, refilled by WEAK
    assert "STRONG" not in bought


def test_news_overlay_degrades_to_pure_momentum_when_feed_empty():
    # Feed returns nothing (rate-limited/no key) → overlay is a no-op.
    execu = RecordingExecutor()
    bars = KeyedBatchBars({"STRONG": _ramp(2.0), "MID": _ramp(1.0), "WEAK": _ramp(0.5)})
    cfg = [StrategyConfig("momentum_news", list(bars._by), 30_000.0,
                          max_positions=2, news_overlay=True)]
    orch = Orchestrator(cfg, bars, execu, risk_config=_HORSE_RISK,
                        news_fetcher=lambda tickers: {})
    orch.run_cycle()
    assert {b[0] for b in execu.buys} == {"STRONG", "MID"}  # plain top-2


def test_news_fetcher_not_called_for_plain_momentum():
    calls = []
    execu = RecordingExecutor()
    bars = KeyedBatchBars({"STRONG": _ramp(2.0), "MID": _ramp(1.0)})
    cfg = [StrategyConfig("momentum_rotation", list(bars._by), 30_000.0, max_positions=2)]
    orch = Orchestrator(cfg, bars, execu, risk_config=_HORSE_RISK,
                        news_fetcher=lambda tickers: calls.append(tickers) or {})
    orch.run_cycle()
    assert calls == []  # overlay off → no API spend on the price-only horses


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


def test_risk_denied_entry_places_no_order(oversold_then_bars):
    execu = RecordingExecutor()
    cfg = [StrategyConfig(strategy="confirmed_mr", symbols=["AAPL"], starting_cash=1000.0)]
    # max_position_pct=0 → risk manager trims size to 0 → entry denied.
    orch = Orchestrator(cfg, FakeBars(oversold_then_bars), execu,
                        risk_config={"max_position_pct": 0.0})
    orch.run_cycle()
    assert execu.buys == []
    assert orch.portfolio("confirmed_mr").qty("AAPL") == 0


def test_sell_suppressed_when_not_holding(oversold_then_bars):
    execu = RecordingExecutor()
    cfg = [StrategyConfig(strategy="confirmed_mr", symbols=["AAPL"], starting_cash=1000.0)]
    bars = oversold_then_bars.copy()
    last_idx = pd.date_range(bars.index[-1], periods=6, freq="B")[1:]
    up = pd.DataFrame({c: [bars["close"].iloc[-1] * (1.05 ** (i + 1)) for i in range(5)]
                       for c in ["open", "high", "low", "close"]}, index=last_idx)
    up["volume"] = 1_000_000.0
    bars2 = pd.concat([bars, up])
    orch = Orchestrator(cfg, FakeBars(bars2), execu,
                        risk_config={"max_position_pct": 1.0, "min_cash_reserve_pct": 0.0})
    # No position seeded → SELL signal must NOT call the executor.
    orch.run_cycle()
    assert execu.sells == []


def test_multi_strategy_isolation_in_one_cycle(oversold_then_bars, rising_bars):
    execu = RecordingExecutor()
    cfg = [
        StrategyConfig(strategy="confirmed_mr", symbols=["AAPL"], starting_cash=1000.0),
        StrategyConfig(strategy="rsi_mr", symbols=["MSFT"], starting_cash=1000.0),
    ]
    bars = KeyedBars({"AAPL": oversold_then_bars, "MSFT": rising_bars})
    orch = Orchestrator(cfg, bars, execu,
                        risk_config={"max_position_pct": 1.0, "min_cash_reserve_pct": 0.0})
    orch.run_cycle()
    # confirmed_mr bought AAPL; rsi_mr saw a flat uptrend on MSFT → HOLD.
    assert orch.portfolio("confirmed_mr").qty("AAPL") > 0
    assert orch.portfolio("rsi_mr").qty("MSFT") == 0
    assert orch.portfolio("rsi_mr").cash == pytest.approx(1000.0)  # B's slice untouched
    # confirmed_mr's buy did not leak into rsi_mr's ledger
    assert orch.portfolio("rsi_mr").qty("AAPL") == 0


def test_hydrated_holding_does_not_rebuy(oversold_then_bars):
    # A prior run left confirmed_mr already holding AAPL. A fresh BUY signal
    # must NOT re-buy (no-pyramiding guard works across runs via persistence).
    execu = RecordingExecutor()
    cfg = [StrategyConfig(strategy="confirmed_mr", symbols=["AAPL"], starting_cash=1000.0)]
    initial = {
        "confirmed_mr": {
            "strategy": "confirmed_mr", "starting_cash": 1000.0,
            "cash": 800.0, "realized_pnl": 0.0,
            "lots": {"AAPL": {"qty": 2.0, "avg_entry": 100.0}},
        }
    }
    orch = Orchestrator(cfg, FakeBars(oversold_then_bars), execu,
                        risk_config={"max_position_pct": 1.0, "min_cash_reserve_pct": 0.0},
                        initial_states=initial)
    assert orch.portfolio("confirmed_mr").qty("AAPL") == 2.0  # loaded
    orch.run_cycle()
    assert execu.buys == []  # already holding → no re-buy


def test_no_initial_state_starts_fresh(rising_bars):
    execu = RecordingExecutor()
    cfg = [StrategyConfig(strategy="confirmed_mr", symbols=["AAPL"], starting_cash=1000.0)]
    orch = Orchestrator(cfg, FakeBars(rising_bars), execu, initial_states=None)
    assert orch.portfolio("confirmed_mr").cash == 1000.0
