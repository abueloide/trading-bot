from __future__ import annotations

import pytest

from live.strategy_runner import Signal, StrategyRunner


def test_unknown_strategy_raises():
    with pytest.raises(KeyError):
        StrategyRunner("does_not_exist")


def test_hold_when_no_entry_or_exit(rising_bars):
    # confirmed_mr needs an oversold candle to enter; a pure uptrend → HOLD.
    runner = StrategyRunner("confirmed_mr")
    sig = runner.run("AAPL", rising_bars)
    assert isinstance(sig, Signal)
    assert sig.action == "HOLD"
    assert sig.symbol == "AAPL"
    assert sig.price == pytest.approx(float(rising_bars["close"].iloc[-1]))


def test_buy_on_oversold_last_bar(oversold_then_bars):
    # confirmed_mr: RSI(2) < 15 on the deep-drop last bar → BUY (SPY filter optional).
    runner = StrategyRunner("confirmed_mr")
    sig = runner.run("AAPL", oversold_then_bars)
    assert sig.action == "BUY"


def test_too_few_bars_is_hold():
    import pandas as pd
    df = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2],
                       "close": [1.0, 2.0], "volume": [1, 1]})
    runner = StrategyRunner("rsi_mr")
    assert runner.run("AAPL", df).action == "HOLD"
