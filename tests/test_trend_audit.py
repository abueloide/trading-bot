"""Invariantes del harness R4 (`events/trend_audit.py`).

Lo que se prueba es la MECÁNICA, no el veredicto: que el rebalanceo caiga en el
primer día hábil del mes, que el trade de Donchian salga donde la regla dice, que
el placebo condicionado excluya las rupturas reales, y que el ranking del tranche
sea el que el vivo opera.
"""
from __future__ import annotations

import pandas as pd
import pytest

from events.trend_audit import (
    donchian_trades,
    momentum_bottom,
    near_breakout_days,
    rebalance_positions,
)


def _frame(closes, highs=None, lows=None, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    highs = highs if highs is not None else closes
    lows = lows if lows is not None else closes
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes,
         "volume": [1_000] * len(closes)},
        index=idx,
    )


# ---------------------------------------------------------------- rebalanceo

def test_rebalance_hits_first_session_of_each_month():
    idx = pd.bdate_range("2021-01-01", "2021-03-31")
    marks = rebalance_positions(idx, (2021, 2022))
    assert [idx[i].strftime("%Y-%m-%d") for i in marks] == [
        "2021-01-01", "2021-02-01", "2021-03-01",
    ]


def test_rebalance_skips_years_outside_the_regime():
    idx = pd.bdate_range("2021-11-01", "2022-02-28")
    marks = rebalance_positions(idx, (2022, 2023))
    assert all(idx[i].year == 2022 for i in marks)
    assert len(marks) == 2  # enero y febrero


def test_rebalance_uses_first_available_session_not_the_1st():
    # 2021-08-01 cae domingo: la primera sesión del mes es el lunes 2.
    idx = pd.bdate_range("2021-07-01", "2021-08-31")
    marks = rebalance_positions(idx, (2021, 2022))
    august = [idx[i] for i in marks if idx[i].month == 8]
    assert august and august[0].strftime("%Y-%m-%d") == "2021-08-02"


# ---------------------------------------------------------------- donchian

def test_donchian_trade_exits_on_ten_day_low():
    # 25 barras planas en 100, una ruptura a 120, luego colapso bajo el mínimo de 10d.
    closes = [100.0] * 25 + [120.0] + [119.0] * 3 + [80.0] + [81.0] * 5
    df = _frame(closes)
    rets, durs, strength, by_year = donchian_trades(df, (2020, 2021))
    assert len(rets) == 1
    # Entra a 120 (la ruptura) y sale a 80 (primer cierre bajo el mínimo de 10d).
    assert rets[0] == pytest.approx(80.0 / 120.0 - 1.0)
    assert durs[0] == 4
    assert strength[0] == pytest.approx(0.20)  # 120 sobre un máximo previo de 100
    assert by_year["2020"] == rets


def test_donchian_does_not_reenter_while_in_position():
    # Ruptura, y siguientes cierres siguen haciendo máximos: una sola entrada.
    closes = [100.0] * 25 + [110.0, 111.0, 112.0, 113.0] + [70.0]
    df = _frame(closes)
    rets, _, _, _ = donchian_trades(df, (2020, 2021))
    assert len(rets) == 1


def test_donchian_respects_the_regime_window():
    closes = [100.0] * 25 + [120.0] * 10
    df = _frame(closes, start="2020-01-01")
    assert donchian_trades(df, (2019, 2020))[0] == []


# ---------------------------------------------------------------- placebo

def test_near_breakout_excludes_actual_breakouts():
    closes = [100.0] * 25 + [120.0] + [119.0] * 5
    df = _frame(closes)
    entry_positions = {
        i for i, e in enumerate(
            __import__("backtesting.strategies", fromlist=["x"])
            .strategy_donchian_breakout(df)["entry"].tolist()
        ) if e
    }
    placebo = set(near_breakout_days(df, (2020, 2021)))
    assert entry_positions  # el fixture sí rompe
    assert placebo.isdisjoint(entry_positions), "el control no puede incluir la ruptura"


def test_near_breakout_keeps_days_high_in_the_range():
    # Sube y luego se estanca JUSTO debajo del máximo: alto en el rango, sin romperlo.
    # (Una rampa monótona no sirve de fixture: es ruptura permanente, no casi-ruptura.)
    closes = [100.0 + i for i in range(30)] + [129.0, 128.5, 129.0, 128.8] * 5
    df = _frame(closes)
    days = near_breakout_days(df, (2020, 2021))
    assert days, "días cerca del máximo sin romperlo deben calificar"
    assert all(closes[i] <= 129.0 for i in days)


# ---------------------------------------------------------------- ranking

def test_momentum_bottom_is_the_mirror_of_top():
    from live.portfolio_targets import momentum_top

    n = 200
    bars = {
        "STRONG": _frame([100.0 + i for i in range(n)]),
        "FLAT": _frame([100.0] * n),
        "WEAK": _frame([200.0 - i * 0.5 for i in range(n)]),
    }
    assert momentum_top(bars, 1) == ["STRONG"]
    assert momentum_bottom(bars, 1) == ["WEAK"]
