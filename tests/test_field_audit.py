"""Tests del harness R3 — auditoría del field daily-bar en vivo."""
from __future__ import annotations

import pandas as pd
import pytest

from backtesting.strategies import STRATEGY_REGISTRY, rsi
from events.field_audit import (
    RSI_BUY, _hold_return, placebo_days, placebo_mask, rule_returns, signals,
)


def _bars(closes, opens=None):
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    opens = opens if opens is not None else [c * 0.99 for c in closes]
    return pd.DataFrame(
        {"close": closes, "open": opens, "high": closes, "low": closes},
        index=idx,
    )


# ------------------------------------------------------------ holding

def test_hold_sale_en_la_primera_senal_de_salida():
    close = [100.0, 101.0, 103.0, 99.0, 98.0]
    ret, j = _hold_return(close, [False, False, True, False, False], 0, 4)
    assert j == 2
    assert ret == pytest.approx(0.03)


def test_hold_sin_senal_sale_a_max_hold():
    close = [100.0, 101.0, 103.0, 99.0, 98.0]
    ret, j = _hold_return(close, [False] * 5, 0, 2)
    assert j == 2
    assert ret == pytest.approx(0.03)


def test_hold_no_se_pasa_del_final_de_la_serie():
    close = [100.0, 101.0, 103.0, 99.0, 98.0]
    _, j = _hold_return(close, [False] * 5, 3, 10)
    assert j == 4


# ------------------------------------------------------------ regla

def _patched(strategy, sig, hold):
    reg = STRATEGY_REGISTRY[strategy]
    saved = (reg["fn"], reg["max_hold_days"])
    reg["fn"] = lambda d, **kw: sig
    reg["max_hold_days"] = hold
    return saved


def _restore(strategy, saved):
    STRATEGY_REGISTRY[strategy]["fn"], STRATEGY_REGISTRY[strategy]["max_hold_days"] = saved


def test_no_solapa_posiciones():
    """El vivo excluye los nombres ya held → una entrada dentro del holding no cuenta."""
    df = _bars([100.0, 101, 102, 103, 104, 105])
    sig = pd.DataFrame({"entry": [True, True, False, False, False, False],
                        "exit": [False] * 6}, index=df.index)
    saved = _patched("confirmed_mr", sig, 3)
    try:
        rets, durs, rsi2s, by_year = rule_returns("confirmed_mr", df, None, (2020, 2021))
    finally:
        _restore("confirmed_mr", saved)
    assert len(rets) == 1
    assert durs == [3]
    assert len(rsi2s) == 1
    assert by_year["2020"] == rets


def test_respeta_la_ventana_de_anios():
    df = _bars([100.0 + k for k in range(600)])
    sig = pd.DataFrame({"entry": True, "exit": False}, index=df.index)
    saved = _patched("confirmed_mr", sig, 3)
    try:
        rets, _, _, by_year = rule_returns("confirmed_mr", df, None, (2021, 2022))
    finally:
        _restore("confirmed_mr", saved)
    assert set(by_year) == {"2021"}
    assert rets


# ------------------------------------------------------------ semántica live vs backtest

def test_semantica_live_llama_la_fn_pelona():
    """`signals(..., None)` debe reproducir la llamada del vivo: sin spy_close."""
    df = _bars([100.0] * 5)
    seen = {}

    def fake(d, **kw):
        seen.update(kw)
        return pd.DataFrame({"entry": False, "exit": False}, index=d.index)

    saved = _patched("confirmed_mr", None, 7)
    STRATEGY_REGISTRY["confirmed_mr"]["fn"] = fake
    try:
        signals("confirmed_mr", df, None)
        assert seen == {}, "el vivo llama fn(df) pelón — sin filtros de régimen"
        signals("confirmed_mr", df, df["close"])
        assert "spy_close" in seen
    finally:
        _restore("confirmed_mr", saved)


# ------------------------------------------------------------ placebo

def test_placebo_no_contiene_dias_sobrevendidos():
    """El control lleva todas las condiciones MENOS el gatillo (regla de R1)."""
    prices = [100.0 * (1.01 ** k) for k in range(260)]
    df = _bars(prices)
    mask = placebo_mask("confirmed_mr", df, None)
    r2 = rsi(df["close"], period=2)
    assert not bool((mask & (r2 < RSI_BUY["confirmed_mr"])).any())


def test_placebo_devuelve_indices_no_retornos():
    """El holding del control lo fija el trade real (duration-matched), no el control."""
    prices = [100.0 * (1.01 ** k) for k in range(260)]
    df = _bars(prices)
    days, close = placebo_days("confirmed_mr", df, None, (2019, 2022))
    assert days, "debe haber días control en una serie alcista"
    assert all(isinstance(i, int) for i in days)
    assert max(days) < len(close) - 1
    assert len(close) == len(df)


def test_placebo_condicionado_al_filtro_spy_cuando_la_regla_lo_lleva():
    prices = [100.0 * (1.01 ** k) for k in range(300)]
    df = _bars(prices)
    bear_spy = pd.Series([100.0 * (0.995 ** k) for k in range(300)], index=df.index)
    con_spy = placebo_days("confirmed_mr", df, bear_spy, (2019, 2022))[0]
    sin_spy = placebo_days("confirmed_mr", df, None, (2019, 2022))[0]
    assert not con_spy, "con SPY bajo su 200dMA el control no puede tener días"
    assert sin_spy
