"""Invariantes del harness R6 (`events/opex_multiplicity.py`).

Se prueba la MECÁNICA de la corrección por multiplicidad, no el veredicto de C2:
que el null best-of-k sea más exigente que el de una sola celda, que un
calendario-placebo que NO produce candidata no aporte su máximo al null, y que el
max-t estudentizado sea invariante a escala (el defecto que el máximo crudo tiene
cuando la familia mezcla ventanas de 1d y 5d).
"""
from __future__ import annotations

import random
from datetime import date

import pandas as pd
import pytest

from events.opex_multiplicity import (
    NO_CANDIDATE,
    Series,
    best_of_family,
    cell_passes_both,
    cell_returns,
    cell_score,
    long_only_returns,
    max_t_null,
    placebo_pool,
    real_calendar,
    regime_of,
    studentize,
    sweep,
)
from events.panic_study import percentile_of


def _series(n=800, seed=3, start="2015-01-01"):
    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.01)))
    idx = pd.bdate_range(start, periods=n)
    return Series(pd.DataFrame({"close": closes}, index=idx))


# ------------------------------------------------------------------ mecánica

def test_locate_cae_a_la_siguiente_sesion_y_no_inventa():
    s = _series()
    assert s.locate(s.dates[42]) == 42
    friday = next(i for i, d in enumerate(s.dates) if d.weekday() == 4)
    saturday = date.fromordinal(s.dates[friday].toordinal() + 1)
    assert s.locate(saturday) == friday + 1
    assert s.locate(date.fromordinal(s.dates[-1].toordinal() + 30)) is None


def test_drift_y_fade_son_espejo_exacto():
    s = _series()
    pos = list(range(5, 300, 3))
    d = cell_returns(s, pos, 1, "drift")
    f = cell_returns(s, pos, 1, "fade")
    assert len(d) == len(f) > 50
    assert all(abs(x + y) < 1e-12 for x, y in zip(d, f))


def test_long_only_solo_toma_dias_verdes_sin_voltear_signo():
    s = _series()
    pos = list(range(5, 300, 3))
    lo = long_only_returns(s, pos, 1)
    esperados = [s.fwd[1][i] for i in pos if s.move[i] > 0 and s.fwd[1][i] is not None]
    assert lo == esperados, "el long-only no aplica sign(); es el forward crudo"
    assert len(lo) < len(pos), "el filtro de día verde debe descartar algo"


def test_placebo_pool_excluye_el_evento_y_su_vecindad():
    """El control no puede caer en la resaca del propio vencimiento (±3 sesiones)."""
    s = _series(n=1400)
    real = real_calendar(s)
    pool = placebo_pool(s, real)
    ocupadas = {d for v in real.values() for d in v}
    for lbl, dias in pool.items():
        assert not (set(dias) & ocupadas), "un día OpEx no puede ser su propio placebo"
        for d in dias:
            cerca = min(abs((d - e).days) for e in ocupadas)
            assert cerca > 3, f"{d} está a {cerca} días de un vencimiento"
            assert regime_of(d) == lbl


# ------------------------------------------------------------------ el null

def _cells(scores):
    """Celdas sintéticas: {(sym,w,mode): {regimen: stats}} con expectativa dada."""
    out = {}
    for key, (oos, is_) in scores.items():
        out[key] = {
            "OOS 2015-21": {"n": 40, "expectancy_pct": oos, "tail_ratio": 1.5},
            "IS 2022-24": {"n": 20, "expectancy_pct": is_, "tail_ratio": 1.5},
        }
    return out


def test_cell_score_es_el_regimen_mas_flojo():
    c = _cells({("A", 1, "drift"): (0.30, 0.10)})
    assert cell_score(c[("A", 1, "drift")]) == 0.10


def test_sin_celda_que_pase_el_gate_el_procedimiento_no_produce_candidata():
    """Un sorteo que no produce candidata NO debe aportar su máximo al null.

    Si aportara, el null incluiría 'la mejor de un mundo donde el investigador
    habría declarado la familia muerta' — y eso infla el null, no la corrección.
    """
    c = _cells({("A", 1, "drift"): (0.30, 0.10), ("B", 1, "drift"): (0.20, 0.05)})
    for cell in c.values():                    # n bajo el piso → nadie pasa el gate
        cell["OOS 2015-21"]["n"] = 3
        cell["IS 2022-24"]["n"] = 3
    assert not any(cell_passes_both(v) for v in c.values())
    _, score, n_pass = best_of_family(c, require_gate=True)
    assert score == NO_CANDIDATE and n_pass == 0
    # sin exigir gate, el mismo sorteo sí produce su máximo
    _, raw, _ = best_of_family(c, require_gate=False)
    assert raw == 0.10


def test_best_of_family_elige_entre_las_que_pasan_no_la_de_mayor_score():
    """El procedimiento real gatea PRIMERO: una celda con mejor score pero que no
    pasa el gate no es la que el investigador habría reportado."""
    c = _cells({("ALTA", 1, "drift"): (0.90, 0.90), ("GATED", 1, "drift"): (0.20, 0.20)})
    c[("ALTA", 1, "drift")]["IS 2022-24"]["tail_ratio"] = 0.5   # falla tail → no pasa
    best, score, n_pass = best_of_family(c, require_gate=True)
    assert best == ("GATED", 1, "drift") and score == 0.20 and n_pass == 1
    assert best_of_family(c, require_gate=False)[0] == ("ALTA", 1, "drift")


def test_null_best_of_k_es_mas_exigente_que_el_de_una_celda():
    """La razón de ser de R6: el mismo valor NUNCA puede salir mejor contra el máximo."""
    per_cell = {
        ("A", 1, "drift"): [0.0, 0.1, 0.2, 0.3, 0.4],
        ("B", 1, "drift"): [0.5, 0.1, 0.05, 0.35, 0.0],
    }
    maxes = [max(v[i] for v in per_cell.values()) for i in range(5)]
    for valor in (0.05, 0.25, 0.45):
        assert percentile_of(valor, maxes) <= percentile_of(valor, per_cell[("A", 1, "drift")])


def test_max_t_corrige_el_sesgo_de_escala_del_maximo_crudo():
    """Familia con una celda de 5× la escala: el máximo crudo casi siempre lo gana
    ella, así que una celda chica queda mal medida aunque sea igual de rara. El
    max-t estudentizado las pone en la misma vara."""
    # Réplica del defecto real: una ventana de 5d tiene más media Y más desvío que
    # una de 1d por pura exposición al mercado, no por ser más rara.
    rng = random.Random(11)
    chica = [rng.gauss(0.08, 0.1) for _ in range(400)]
    grande = [rng.gauss(0.40, 0.5) for _ in range(400)]
    per_cell = {("CHICA", 1, "drift"): chica, ("GRANDE", 5, "drift"): grande}

    crudo = [max(chica[i], grande[i]) for i in range(400)]
    gana_grande = sum(1 for i in range(400) if grande[i] > chica[i])
    assert gana_grande > 0.7 * 400, "control del test: la celda de mayor escala domina el crudo"

    moments = studentize(per_cell)
    maxt = max_t_null(per_cell, moments)

    real_chica = 0.38                     # +3 desvíos para la celda chica
    z = (real_chica - moments[("CHICA", 1, "drift")][0]) / moments[("CHICA", 1, "drift")][1]
    assert percentile_of(z, maxt) > percentile_of(real_chica, crudo) + 20, (
        "estudentizar debe rescatar a la celda chica que el máximo crudo aplasta"
    )


def test_max_t_sigue_penalizando_al_crecer_k():
    """Invariante que la corrección debe conservar: más hipótesis, vara más alta."""
    rng = random.Random(5)
    dists = {(f"S{i}", 1, "drift"): [rng.gauss(0, 1) for _ in range(500)] for i in range(8)}
    chico = {k: dists[k] for k in list(dists)[:2]}
    z = 2.0
    p_chico = percentile_of(z, max_t_null(chico, studentize(chico)))
    p_grande = percentile_of(z, max_t_null(dists, studentize(dists)))
    assert p_grande < p_chico, "k=8 debe exigir más que k=2"


# ------------------------------------------------------------------ barrido

def test_sweep_cubre_la_familia_completa_y_respeta_windows():
    s = {"AAA": _series(seed=1), "BBB": _series(seed=2)}
    fechas = {"OOS 2015-21": [s["AAA"].dates[i] for i in range(10, 400, 20)],
              "IS 2022-24": []}
    cells = sweep(s, fechas, long_only=False, windows=(1, 3))
    assert len(cells) == 2 * 2 * 2                       # sym × w × modo
    solo1 = sweep(s, fechas, long_only=True, windows=(1,))
    assert set(solo1) == {("AAA", 1, "long_only"), ("BBB", 1, "long_only")}


def test_sweep_es_determinista():
    s = {"AAA": _series(seed=1)}
    fechas = {"OOS 2015-21": [s["AAA"].dates[i] for i in range(10, 400, 20)],
              "IS 2022-24": []}
    a = sweep(s, fechas, windows=(1,))
    b = sweep(s, fechas, windows=(1,))
    assert a == b


def test_regime_of_parte_por_ano_sin_solaparse():
    assert regime_of(date(2015, 1, 2)) == "OOS 2015-21"
    assert regime_of(date(2021, 12, 31)) == "OOS 2015-21"
    assert regime_of(date(2022, 1, 3)) == "IS 2022-24"
    assert regime_of(date(2024, 12, 31)) == "IS 2022-24"
    assert regime_of(date(2025, 1, 2)) is None
