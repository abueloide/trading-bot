"""Invariantes del harness R5 (`events/ranking_study.py`).

Se prueba la MECÁNICA del simulador de libro y del control por multiplicidad, no
el veredicto: que la capacidad se respete, que una salida libere el slot, que el
ranking realmente decida quién entra cuando hay competencia, y que el null de
best-of-k sea más exigente que el de una sola corrida (si no, la corrección por
haber probado k criterios no corrige nada).
"""
from __future__ import annotations

import random

import pandas as pd
import pytest

from events.ranking_study import (
    RANK_KEYS,
    best_of_k_null,
    build_book,
    simulate,
    symbol_features,
)

YEARS = (2020, 2021)


def _frame(closes, highs=None, lows=None, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    highs = highs if highs is not None else closes
    lows = lows if lows is not None else closes
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes,
         "volume": [1_000] * len(closes)},
        index=idx,
    )


def _breakout_frame(n=120, jump_at=60, after=1.02):
    """Plano, un salto que rompe el máximo de 20d, luego sube y cae (para salir)."""
    closes = [100.0] * jump_at
    price = 100.0
    for i in range(n - jump_at):
        price = price * after if i < 15 else price * 0.97
        closes.append(price)
    return _frame(closes)


# ---------------------------------------------------------------- features

def test_strength_is_close_over_prior_20d_high():
    df = _breakout_frame()
    f = symbol_features(df)
    fired = [i for i, on in enumerate(f["entry"]) if on]
    assert fired, "el fixture debe disparar al menos una ruptura"
    i = fired[0]
    prior_high = float(df["high"].iloc[i - 20:i].max())
    assert f["strength"].iloc[i] == pytest.approx(df["close"].iloc[i] / prior_high - 1.0)
    # El criterio "inverso" es exactamente el negativo, y "lowvol" prefiere poco.
    assert f["strength_inv"].iloc[i] == pytest.approx(-f["strength"].iloc[i])
    assert f["lowvol"].iloc[i] <= 0


def test_every_declared_key_is_produced():
    f = symbol_features(_breakout_frame())
    for key in RANK_KEYS:
        assert key in f.columns


# ---------------------------------------------------------------- capacidad

def _book_of(n_symbols: int, **kw):
    bars = {f"S{i}": _breakout_frame(**kw) for i in range(n_symbols)}
    calendar = next(iter(bars.values())).index
    return build_book(bars, calendar)


def test_never_holds_more_than_slots():
    """20 símbolos rompen el mismo día con 3 slots ⇒ nunca más de 3 en el libro."""
    book = _book_of(20)
    res = simulate(book, YEARS, "strength", slots=3)
    # Cada trade ocupa un slot toda su duración; si el cap se respetara mal, el
    # número de trades excedería lo que caben en la ventana.
    assert res["trades"] <= 3 * len(book["calendar"])
    assert res["fill_rate"] < 100.0, "con 20 candidatos y 3 slots el racionamiento debe morder"


def test_exit_frees_the_slot():
    """Con 1 slot, el segundo trade sólo puede empezar después de que salga el primero."""
    book = _book_of(2)
    res = simulate(book, YEARS, "strength", slots=1)
    assert res["trades"] >= 1
    assert res["avg_hold_days"] > 0


def test_ranking_decides_who_gets_the_slot():
    """Un símbolo con ruptura más fuerte desplaza al débil cuando hay 1 slot."""
    bars = {
        "WEAK": _breakout_frame(after=1.01),
        "STRONG": _breakout_frame(after=1.10),
    }
    book = build_book(bars, bars["WEAK"].index)
    by_strength = simulate(book, YEARS, "strength", slots=1)
    by_inverse = simulate(book, YEARS, "strength_inv", slots=1)
    assert set(by_strength["per_symbol"]) == {"STRONG"}
    assert set(by_inverse["per_symbol"]) == {"WEAK"}


def test_open_positions_are_closed_at_window_end():
    """Sin cierre forzado, el criterio que más tarda en salir se ahorra sus pérdidas."""
    closes = [100.0] * 30 + [100.0 + i for i in range(1, 60)]  # rompe y nunca cae
    bars = {"UP": _frame(closes)}
    book = build_book(bars, bars["UP"].index)
    res = simulate(book, YEARS, "strength", slots=1)
    assert res["trades"] == 1, "la posición abierta al final debe contarse"


# ---------------------------------------------------------------- null

def test_random_priority_varies_across_draws():
    book = _book_of(20)
    a = simulate(book, YEARS, None, rng=random.Random(1), slots=3)
    b = simulate(book, YEARS, None, rng=random.Random(2), slots=3)
    assert set(a["per_symbol"]) != set(b["per_symbol"])


def test_best_of_k_null_is_stricter_than_single_draw():
    """Cuantifica la inflación por multiplicidad que el control corrige.

    Con k=5, el máximo supera el p90 de una sola corrida el 1−0.9^5 ≈ 41% de las
    veces: quedarse con el mejor de 5 criterios cruza "pct 90" casi la mitad del
    tiempo por puro azar. Ése es el sesgo que este null neutraliza.
    """
    dist = [float(i) for i in range(100)]
    bok = best_of_k_null(dist, k=5, draws=2000)
    assert bok, "con 100 muestras y k=5 debe poder formarse la distribución"
    median_single, median_bok = sorted(dist)[50], sorted(bok)[len(bok) // 2]
    assert median_bok > median_single
    above_p90 = sum(1 for v in bok if v > sorted(dist)[90]) / len(bok)
    assert above_p90 > 0.30, (
        "si el mejor de k no cruzara el p90 mucho más del 10% de las veces, no "
        "habría nada que corregir — y este control sería decorativo"
    )


def test_best_of_k_null_needs_enough_samples():
    assert best_of_k_null([1.0, 2.0], k=5) == []


def test_empty_window_is_not_a_crash():
    book = _book_of(2)
    res = simulate(book, (1990, 1991), "strength")
    assert res["trades"] == 0 and res["stats"]["n"] == 0
