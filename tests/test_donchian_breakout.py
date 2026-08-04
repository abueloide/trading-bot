"""Donchian breakout strategy + its candidate ranking."""
from __future__ import annotations

import pandas as pd

from backtesting.strategies import STRATEGY_REGISTRY, strategy_donchian_breakout
from live.portfolio_targets import breakout_candidates


def _bars(highs, lows, closes) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": 1},
        index=idx,
    )


def test_registry_entry_is_breakout_type():
    spec = STRATEGY_REGISTRY["donchian_breakout"]
    assert spec["type"] == "breakout"
    assert spec["fn"] is strategy_donchian_breakout


def test_entry_fires_on_new_20day_high():
    # 25 flat bars at 100, then a close at 110 that clears the prior 20d high.
    closes = [100.0] * 25 + [110.0]
    highs = [100.0] * 25 + [110.0]
    lows = [99.0] * 26
    sig = strategy_donchian_breakout(_bars(highs, lows, closes))
    assert bool(sig["entry"].iloc[-1]) is True
    assert bool(sig["entry"].iloc[-2]) is False  # flat bar before the break


def test_exit_fires_on_new_10day_low():
    # Ride up, then a close below the prior 10-day low → exit.
    closes = [100.0] * 25 + [120.0] * 10 + [90.0]
    highs = [c + 1 for c in closes]
    lows = [100.0] * 25 + [119.0] * 10 + [90.0]
    sig = strategy_donchian_breakout(_bars(highs, lows, closes))
    assert bool(sig["exit"].iloc[-1]) is True


def test_too_few_bars_is_all_false():
    df = _bars([1, 2], [1, 2], [1.0, 2.0])
    sig = strategy_donchian_breakout(df)
    assert not sig["entry"].any()
    assert not sig["exit"].any()


def test_candidates_ranked_by_breakout_strength_strongest_first():
    # WEAK clears the prior high by ~1%, STRONG by ~10%. STRONG must rank first.
    weak = _bars([100.0] * 25 + [101.0], [99.0] * 26, [100.0] * 25 + [101.0])
    strong = _bars([100.0] * 25 + [110.0], [99.0] * 26, [100.0] * 25 + [110.0])
    out = breakout_candidates("donchian_breakout", {"WEAK": weak, "STRONG": strong})
    syms = [sym for sym, _strength, _price in out]
    assert syms == ["STRONG", "WEAK"]


def test_candidates_exclude_held_names():
    strong = _bars([100.0] * 25 + [110.0], [99.0] * 26, [100.0] * 25 + [110.0])
    out = breakout_candidates(
        "donchian_breakout", {"STRONG": strong}, exclude={"STRONG"}
    )
    assert out == []
