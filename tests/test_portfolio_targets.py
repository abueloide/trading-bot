"""Tests for cross-sectional target construction (momentum rank + MR oversold)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from live.portfolio_targets import (
    exit_signals,
    momentum_scores,
    momentum_top,
    oversold_candidates,
)


def _bars(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = pd.Series(closes, index=idx, dtype="float64")
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": pd.Series(1_000_000, index=idx, dtype="float64"),
        }
    )


def _steady(slope: float, n: int = 200, start: float = 100.0) -> pd.DataFrame:
    return _bars([start + slope * i for i in range(n)])


def _oversold_reversal(n: int = 240) -> pd.DataFrame:
    base = [100.0 + i * 0.5 for i in range(n - 5)]
    drop = [base[-1] * f for f in (0.96, 0.92, 0.88, 0.85, 0.82)]
    df = _bars(base + drop)
    last_close = float(df["close"].iloc[-1])
    df.iloc[-1, df.columns.get_loc("open")] = last_close * 0.99
    df.iloc[-1, df.columns.get_loc("low")] = last_close * 0.985
    return df


def _downtrend_oversold(n: int = 240) -> pd.DataFrame:
    # Long decline (50d MA below 200d MA) ending in an oversold dip — the kind of
    # falling knife confirmed_mr must now reject via its per-name uptrend filter.
    base = [200.0 - i * 0.5 for i in range(n - 5)]
    drop = [base[-1] * f for f in (0.96, 0.92, 0.88, 0.85, 0.82)]
    df = _bars(base + drop)
    last_close = float(df["close"].iloc[-1])
    df.iloc[-1, df.columns.get_loc("open")] = last_close * 0.99  # bullish candle
    return df


def test_confirmed_mr_rejects_oversold_downtrend():
    # Oversold but in a downtrend → uptrend filter blocks the entry.
    bars = {"FALLING": _downtrend_oversold()}
    assert oversold_candidates("confirmed_mr", bars) == []


def test_momentum_scores_rank_strong_above_weak():
    bars = {"FAST": _steady(1.0), "SLOW": _steady(0.1)}
    scores = momentum_scores(bars)
    assert scores["FAST"] > scores["SLOW"] > 0


def test_momentum_top_takes_n_strongest_positive():
    bars = {
        "A": _steady(1.0),
        "B": _steady(0.5),
        "C": _steady(0.2),
        "D": _bars([200.0 - i for i in range(200)]),  # downtrend -> negative score
    }
    top = momentum_top(bars, n=2)
    assert top == ["A", "B"]
    assert "D" not in momentum_top(bars, n=10)  # negative score excluded


def test_momentum_top_caps_per_sector():
    # 4 strong tech names + 1 weaker energy name. Without a cap the basket is
    # all-tech; with max_per_sector=2 it must leave room for the energy name.
    bars = {
        "T1": _steady(1.0), "T2": _steady(0.9), "T3": _steady(0.8), "T4": _steady(0.7),
        "E1": _steady(0.3),
    }
    sectors = {"T1": "Tech", "T2": "Tech", "T3": "Tech", "T4": "Tech", "E1": "Energy"}
    top = momentum_top(bars, n=3, sector_of=sectors.get, max_per_sector=2)
    assert top == ["T1", "T2", "E1"]  # T3/T4 skipped: Tech cap hit


def test_momentum_top_cap_noop_without_both_args():
    bars = {"A": _steady(1.0), "B": _steady(0.5)}
    # sector_of without max_per_sector → unchanged behaviour
    assert momentum_top(bars, n=2, sector_of=lambda s: "X") == ["A", "B"]


def test_momentum_ignores_too_short_series():
    bars = {"SHORT": _bars([100.0, 101.0, 102.0])}
    assert momentum_scores(bars) == {}


def test_oversold_candidates_ranked_by_rsi_ascending():
    bars = {
        "DEEP": _oversold_reversal(),
        "RISING": _steady(1.0),  # no entry signal
    }
    cands = oversold_candidates("confirmed_mr", bars)
    syms = [c[0] for c in cands]
    assert "DEEP" in syms
    assert "RISING" not in syms
    # tuple shape: (symbol, rsi2, price)
    assert cands[0][1] < 20.0 and cands[0][2] > 0


def test_oversold_candidates_excludes_held():
    bars = {"DEEP": _oversold_reversal()}
    assert oversold_candidates("confirmed_mr", bars, exclude={"DEEP"}) == []


def test_exit_signals_fires_for_recovered_holding():
    bars = {"DEEP": _oversold_reversal()}
    # Append a sharp recovery so RSI(2) climbs above the exit threshold.
    base = bars["DEEP"]
    idx = pd.date_range(base.index[-1], periods=6, freq="B")[1:]
    up = pd.DataFrame(
        {c: [base["close"].iloc[-1] * (1.05 ** (i + 1)) for i in range(5)]
         for c in ["open", "high", "low", "close"]},
        index=idx,
    )
    up["volume"] = 1_000_000.0
    recovered = {"DEEP": pd.concat([base, up])}
    assert "DEEP" in exit_signals("confirmed_mr", recovered, held={"DEEP"})
