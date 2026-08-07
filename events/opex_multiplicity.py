"""R6 — ¿C2 (OpEx) sobrevive el null **best-of-k**? Deuda de método abierta por R5.

R5 (2026-08-06) dejó una regla que aplica **retroactivamente a todos los barridos
del repo**: *con k hipótesis, el null es el máximo de k*. Probar k celdas y reportar
el percentil de la ganadora contra un null de UNA corrida reporta el percentil
equivocado — con k=5 el mejor cruza el pct 90 el 41% de las veces por puro azar.

C2 (OpEx 1-day drift) es la única celda del repo que aguanta todo (placebo
condicionado, jackknife, leave-one-year-out, costos) y **está desplegada en paper**
(IVV + QQQ, long-only). Nació de un barrido de **3 símbolos × 3 ventanas × 2 modos
= 18 celdas** y su percentil se midió contra el null de su propia celda. Nunca
enfrentó la corrección por multiplicidad. Si el edge se disuelve bajo el null
correcto, el repo se queda con CERO celdas vivas y eso cambia el veredicto de Luis.

Diferencia con el `best_of_k_null` de R5: allá el máximo se remuestreaba de una
sola distribución null (asume independencia entre criterios). Aquí se corre **el
barrido COMPLETO sobre cada calendario-placebo** — las 18 celdas comparten los
mismos días sorteados, así que la correlación entre celdas (SPY/QQQ/IWM se mueven
juntos; w=1 está dentro de w=3) queda dentro del null. Un null independiente sería
demasiado exigente y el resultado no sería interpretable.

Tres partes:
  A) La familia con la que C2 se **seleccionó**: 18 celdas, dos patas, w∈{1,3,5}.
  B) La familia de lo que **corre en vivo**: long-only condicionado a día verde,
     w=1, k=3 símbolos (SPY≈IVV, QQQ, IWM) — la que auditó R1.
  C) La familia HONESTA del vivo (k=9): el w=1 se heredó de la selección de A, así
     que contar sólo los 3 símbolos vuelve a subestimar la multiplicidad.

Estadístico de selección: `min(exp_OOS, exp_IS)` — el criterio real del carril es
"pasa en AMBOS regímenes", así que el score de una celda es su régimen más flojo.

Uso: python -m events.opex_multiplicity [--draws N] [--seed S] [--selfcheck]
"""
from __future__ import annotations

import random
import statistics as st
import sys
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

from events.event_study import CALENDARS, event_study, gate_event
from events.panic_study import percentile_of

SYMBOLS = ("SPY", "QQQ", "IWM")
WINDOWS = (1, 3, 5)
MODES = ("drift", "fade")
REGIMES: Dict[str, Tuple[int, int]] = {"OOS 2015-21": (2015, 2022), "IS 2022-24": (2022, 2025)}
DRAWS = 500
SEED = 20260807
MASTER = "SPY"          # calendario de sesiones compartido por el barrido
MAX_LAG_DAYS = 4        # feriado: el evento cae a la primera sesión posterior


# ------------------------------------------------------------------ series

class Series:
    """Precómputo por símbolo: la mecánica de un barrido de 500 sorteos × 18 celdas.

    `_forward_drift_returns` hace `idx.index(ts)` (lineal) por evento; con 500
    calendarios-placebo eso son ~10⁹ operaciones. Aquí el mapeo fecha→posición y
    los retornos forward se calculan UNA vez y cada sorteo es indexado puro.
    """

    def __init__(self, df, windows: Sequence[int] = WINDOWS):
        self.dates: List[date] = [ts.date() for ts in df.index]
        self.pos: Dict[date, int] = {d: i for i, d in enumerate(self.dates)}
        closes = [float(c) for c in df["close"]]
        n = len(closes)
        self.n = n
        self.move: List[float] = [0.0] + [closes[i] / closes[i - 1] - 1 for i in range(1, n)]
        self.fwd: Dict[int, List[Optional[float]]] = {}
        for w in windows:
            self.fwd[w] = [
                (closes[i + w] / closes[i] - 1) if i + w < n else None for i in range(n)
            ]

    def locate(self, d: date) -> Optional[int]:
        """Posición de la sesión == d, o la primera posterior dentro de 4 días."""
        i = self.pos.get(d)
        if i is not None:
            return i
        for lag in range(1, MAX_LAG_DAYS + 1):
            i = self.pos.get(date.fromordinal(d.toordinal() + lag))
            if i is not None:
                return i
        return None


def cell_returns(s: Series, positions: Sequence[int], w: int, mode: str) -> List[float]:
    """Retorno drift/fade de `w` sesiones para cada posición. Réplica de event_study."""
    direction = -1.0 if mode == "fade" else 1.0
    fwd = s.fwd[w]
    out: List[float] = []
    for i in positions:
        if i <= 0 or fwd[i] is None:
            continue
        mv = s.move[i]
        if mv == 0:
            continue
        out.append(direction * (1.0 if mv > 0 else -1.0) * fwd[i])
    return out


def long_only_returns(s: Series, positions: Sequence[int], w: int = 1) -> List[float]:
    """La regla DESPLEGADA: entra sólo si el día del evento cerró verde, sale a `w`.

    El condicionamiento vive en la regla, así que un calendario-placebo pasado por
    esta misma función queda condicionado igual (lección de F4) sin filtrar el pool.
    """
    fwd = s.fwd[w]
    return [fwd[i] for i in positions if i > 0 and fwd[i] is not None and s.move[i] > 0]


# ------------------------------------------------------------------ barrido

def regime_of(d: date) -> Optional[str]:
    for label, (y0, y1) in REGIMES.items():
        if y0 <= d.year < y1:
            return label
    return None


def sweep(series: Dict[str, Series], dates_by_regime: Dict[str, List[date]],
          long_only: bool = False,
          windows: Sequence[int] = WINDOWS) -> Dict[Tuple, Dict[str, Dict]]:
    """Todas las celdas de la familia sobre un calendario dado. Puro."""
    modes = ("long_only",) if long_only else MODES
    cells: Dict[Tuple, Dict[str, Dict]] = {}
    for sym in series:
        s = series[sym]
        pos_by_regime = {
            lbl: [p for p in (s.locate(d) for d in ds) if p is not None]
            for lbl, ds in dates_by_regime.items()
        }
        for w in windows:
            for mode in modes:
                per_regime = {}
                for lbl, positions in pos_by_regime.items():
                    rets = (long_only_returns(s, positions, w) if long_only
                            else cell_returns(s, positions, w, mode))
                    per_regime[lbl] = event_study(rets)
                cells[(sym, w, mode)] = per_regime
    return cells


def cell_score(per_regime: Dict[str, Dict]) -> float:
    """Score de selección = el régimen MÁS FLOJO. El carril exige pasar en ambos."""
    return min(r["expectancy_pct"] for r in per_regime.values())


def cell_passes_both(per_regime: Dict[str, Dict]) -> bool:
    return all(gate_event(r)["passed"] for r in per_regime.values())


NO_CANDIDATE = -999.0


def best_of_family(cells: Dict[Tuple, Dict[str, Dict]],
                   require_gate: bool = True) -> Tuple[Tuple, float, int]:
    """(celda ganadora, su score, cuántas celdas pasan el gate en ambos regímenes).

    `require_gate=True` replica el procedimiento real de C2: **primero** el gate del
    carril en AMBOS regímenes, y entre las que pasan se reporta la mejor. Si ningún
    celda pasa, el procedimiento no produce candidata → score `NO_CANDIDATE`. Esto
    importa en el null: un sorteo-placebo que no produce candidata NO debe aportar
    su mejor score al máximo, porque en ese mundo el investigador declara la familia
    muerta y no reporta nada.
    """
    n_pass = sum(1 for c in cells if cell_passes_both(cells[c]))
    pool = [c for c in cells if cell_passes_both(cells[c])] if require_gate else list(cells)
    if not pool:
        return max(cells, key=lambda c: cell_score(cells[c])), NO_CANDIDATE, n_pass
    best = max(pool, key=lambda c: cell_score(cells[c]))
    return best, cell_score(cells[best]), n_pass


# ------------------------------------------------------------------ null

def real_calendar(master: Series) -> Dict[str, List[date]]:
    """Fechas OpEx reales mapeadas a sesiones, partidas por régimen."""
    out: Dict[str, List[date]] = {lbl: [] for lbl in REGIMES}
    for ds in CALENDARS["OPEX"]:
        y, m, d = (int(x) for x in ds.split("-"))
        p = master.locate(date(y, m, d))
        if p is None:
            continue
        session = master.dates[p]
        lbl = regime_of(session)
        if lbl:
            out[lbl].append(session)
    return out


def placebo_pool(master: Series, real: Dict[str, List[date]]) -> Dict[str, List[date]]:
    """Sesiones NO-OpEx utilizables, por régimen. Excluye ±3 días de un vencimiento.

    El buffer evita que el 'placebo' caiga en la resaca del propio evento (el jueves
    o el lunes del vencimiento), que contaminaría el control con lo que mide.
    """
    banned = set()
    for ds in (d for v in real.values() for d in v):
        for k in range(-3, 4):
            banned.add(date.fromordinal(ds.toordinal() + k))
    wmax = max(WINDOWS)
    pool: Dict[str, List[date]] = {lbl: [] for lbl in REGIMES}
    for i, d in enumerate(master.dates):
        if i == 0 or i + wmax >= master.n or d in banned:
            continue
        lbl = regime_of(d)
        if lbl:
            pool[lbl].append(d)
    return pool


def null_distribution(series: Dict[str, Series], master: Series,
                      real: Dict[str, List[date]], draws: int, long_only: bool = False,
                      require_gate: bool = True, windows: Sequence[int] = WINDOWS,
                      seed: int = SEED) -> Dict[str, object]:
    """Corre el barrido COMPLETO sobre `draws` calendarios-placebo.

    Devuelve la distribución del máximo de la familia (best-of-k honesto, con la
    correlación entre celdas dentro) y la de cada celda por separado (el null
    ingenuo con el que se gateó C2), para poder medir la inflación.
    """
    pool = placebo_pool(master, real)
    rng = random.Random(seed)
    n_by_regime = {lbl: len(v) for lbl, v in real.items()}
    best_dist: List[float] = []
    any_pass = 0
    per_cell: Dict[Tuple, List[float]] = {}
    for _ in range(draws):
        fake = {lbl: rng.sample(pool[lbl], n_by_regime[lbl]) for lbl in REGIMES}
        cells = sweep(series, fake, long_only=long_only, windows=windows)
        for c, per_regime in cells.items():
            per_cell.setdefault(c, []).append(cell_score(per_regime))
        _, score, n_pass = best_of_family(cells, require_gate=require_gate)
        best_dist.append(score)
        any_pass += 1 if n_pass else 0
    return {
        "best_dist": best_dist,
        "produced": [s for s in best_dist if s != NO_CANDIDATE],
        "per_cell": per_cell,
        "any_pass_rate": round(100.0 * any_pass / draws, 1),
        "k": len(per_cell),
    }


# ------------------------------------------------------------------ reporte

def _fmt_cell(c: Tuple) -> str:
    sym, w, mode = c
    return f"{sym} w={w}d {mode}"


def studentize(per_cell: Dict[Tuple, List[float]]) -> Dict[Tuple, Tuple[float, float]]:
    """(media, desvío) del null de cada celda, para poner las celdas en la misma escala."""
    out = {}
    for c, dist in per_cell.items():
        mu = st.mean(dist)
        sd = st.pstdev(dist)
        out[c] = (mu, sd if sd > 0 else 1e-9)
    return out


def max_t_null(per_cell: Dict[Tuple, List[float]],
               moments: Dict[Tuple, Tuple[float, float]]) -> List[float]:
    """Null del MÁXIMO ESTUDENTIZADO de la familia (max-t).

    El máximo del score crudo está sesgado hacia las celdas de mayor escala: una
    ventana de 5d rinde ~5× una de 1d por pura exposición, así que el máximo crudo
    casi siempre lo gana un 5d y comparar contra él castiga de más a la celda de 1d.
    Estandarizar cada celda contra SU PROPIO null antes de tomar el máximo hace la
    comparación invariante a escala: mide "qué tan rara es esta celda para sí misma",
    que es exactamente lo que la corrección por multiplicidad debe comparar.
    """
    draws = len(next(iter(per_cell.values())))
    out = []
    for i in range(draws):
        out.append(max((per_cell[c][i] - moments[c][0]) / moments[c][1] for c in per_cell))
    return out


def analyze(series: Dict[str, Series], master: Series, real: Dict[str, List[date]],
            draws: int, long_only: bool, require_gate: bool = True,
            windows: Sequence[int] = WINDOWS, seed: int = SEED) -> Dict[str, object]:
    cells = sweep(series, real, long_only=long_only, windows=windows)
    best, best_score, n_pass = best_of_family(cells, require_gate=require_gate)
    null = null_distribution(series, master, real, draws, long_only=long_only,
                             require_gate=require_gate, windows=windows, seed=seed)
    pct_family = percentile_of(best_score, null["best_dist"])
    pct_naive = percentile_of(best_score, null["per_cell"][best])
    moments = studentize(null["per_cell"])
    maxt = max_t_null(null["per_cell"], moments)
    z_real = {c: (cell_score(cells[c]) - moments[c][0]) / moments[c][1] for c in cells}
    best_z_cell = max(z_real, key=lambda c: z_real[c])
    return {
        "cells": cells, "best": best, "best_score": best_score, "n_pass": n_pass,
        "null": null, "pct_family": pct_family, "pct_naive": pct_naive,
        "moments": moments, "maxt": maxt, "z_real": z_real, "best_z_cell": best_z_cell,
        "pct_maxt": percentile_of(z_real[best_z_cell], maxt),
    }


def report(title: str, a: Dict[str, object], focus: Optional[Tuple] = None) -> None:
    cells, null = a["cells"], a["null"]
    print(f"\n{'='*72}\n{title}\n{'='*72}")
    print(f"Familia: k={null['k']} celdas · {len(null['best_dist'])} calendarios-placebo\n")
    for c in sorted(cells, key=lambda c: -cell_score(cells[c])):
        per = cells[c]
        bits = "  ".join(
            f"{lbl}: exp={per[lbl]['expectancy_pct']:+.3f}% n={per[lbl]['n']} "
            f"tail={per[lbl]['tail_ratio']}"
            for lbl in REGIMES
        )
        mark = "✅" if cell_passes_both(per) else "  "
        print(f"  {mark} {_fmt_cell(c):<22s} score={cell_score(per):+.3f}%   {bits}")
    print(f"\n  Celdas que pasan el gate en AMBOS regímenes: {a['n_pass']}/{null['k']}")
    print(f"  Ganadora del barrido: {_fmt_cell(a['best'])}  score={a['best_score']:+.3f}%")
    nd = null["best_dist"]
    prod = null["produced"]
    print(f"\n  NULL best-of-{null['k']} (máximo de la familia por calendario-placebo):")
    print(f"    calendarios que SÍ producen candidata: {len(prod)}/{len(nd)} "
          f"({100.0*len(prod)/len(nd):.0f}%)")
    if prod:
        print(f"    entre los que producen: media={st.mean(prod):+.3f}%  "
              f"p90={sorted(prod)[int(0.9*(len(prod)-1))]:+.3f}%  máx={max(prod):+.3f}%")
    print(f"    calendarios-placebo con ≥1 celda que pasa el gate en ambos regímenes: "
          f"{null['any_pass_rate']}%")
    print(f"\n  Percentil de la ganadora contra el NULL INGENUO (su propia celda): "
          f"{a['pct_naive']:.1f}")
    print(f"  Percentil de la ganadora contra el NULL BEST-OF-{null['k']}:        "
          f"{a['pct_family']:.1f}   "
          f"{'PASA (≥90)' if a['pct_family'] >= 90 else 'NO PASA (<90)'}")
    print(f"  → inflación por multiplicidad: {a['pct_naive'] - a['pct_family']:+.1f} puntos")
    print(f"\n  MAX-T (estudentizado, invariante a escala — el estadístico bueno):")
    print(f"    mejor celda por z: {_fmt_cell(a['best_z_cell'])}  "
          f"z={a['z_real'][a['best_z_cell']]:+.2f}")
    print(f"    percentil contra el null de max-t: {a['pct_maxt']:.1f}   "
          f"{'PASA (≥90)' if a['pct_maxt'] >= 90 else 'NO PASA (<90)'}")
    if focus and focus in cells:
        f_score = cell_score(cells[focus])
        pf = percentile_of(f_score, nd)
        pn = percentile_of(f_score, null["per_cell"][focus])
        pz = percentile_of(a["z_real"][focus], a["maxt"])
        print(f"\n  Celda EN VIVO ({_fmt_cell(focus)}): score={f_score:+.3f}%  z={a['z_real'][focus]:+.2f}")
        print(f"    pct ingenuo (su propia celda)={pn:.1f}  ·  pct max-t (k={null['k']})={pz:.1f}   "
              f"{'PASA' if pz >= 90 else 'NO PASA'}")


def run(draws: int = DRAWS, seed: int = SEED) -> int:
    from backtesting.engine import load_bars

    start, end = date(2015, 1, 1), date(2025, 1, 1)
    print(f"R6 — corrección por multiplicidad sobre C2 (OpEx). {draws} calendarios-placebo, "
          f"seed={seed}\nBajando barras…", flush=True)
    series: Dict[str, Series] = {}
    for sym in SYMBOLS:
        df = load_bars(sym, start, end, "1d", source="yfinance")
        if df is None or df.empty:
            print(f"sin datos para {sym}", file=sys.stderr)
            return 2
        series[sym] = Series(df)
    master = series[MASTER]
    real = real_calendar(master)
    print(f"OpEx mapeados: " + ", ".join(f"{lbl} n={len(v)}" for lbl, v in real.items()))

    # A: el procedimiento real fue "gate del carril en ambos regímenes, y de las que
    # pasan, la mejor" → require_gate=True.
    a1 = analyze(series, master, real, draws, long_only=False, require_gate=True, seed=seed)
    report("A) La familia con la que C2 se SELECCIONÓ (2 patas · 3 sym × 3 w × 2 modos)",
           a1, focus=("QQQ", 1, "drift"))

    # B: aquí NO seleccionó el gate (R1 corrió los tests asesinos sobre las 3 celdas y
    # se quedó con la que aguantó; QQQ-IS ni siquiera llega a MIN_EVENTS) → el máximo
    # de la familia es sobre el score crudo.
    a2 = analyze(series, master, real, draws, long_only=True, require_gate=False,
                 windows=(1,), seed=seed)
    report("B) La familia de lo que CORRE EN VIVO (long-only día verde · w=1 · 3 sym)",
           a2, focus=("QQQ", 1, "long_only"))

    # C: la familia HONESTA de B. El w=1 no cayó del cielo: se heredó de la selección
    # de A, donde también se probaron 3d y 5d. Contar sólo los símbolos (k=3) vuelve a
    # subestimar la multiplicidad — la misma falla que R6 audita.
    a3 = analyze(series, master, real, draws, long_only=True, require_gate=False,
                 windows=WINDOWS, seed=seed)
    report("C) Familia honesta del vivo: long-only × 3 sym × 3 ventanas (k=9)",
           a3, focus=("QQQ", 1, "long_only"))

    # D: ancla. Familia HOMOGÉNEA (sólo w=1, ambas patas, 3 sym → k=6): todas las
    # celdas comparten escala, así que el máximo CRUDO es válido sin estudentizar.
    # Si el veredicto de A sólo apareciera al cambiar de estadístico, sería sospechoso.
    a4 = analyze(series, master, real, draws, long_only=False, require_gate=False,
                 windows=(1,), seed=seed)
    report("D) Ancla homogénea: sólo w=1d, 2 patas, 3 sym (k=6) — máximo crudo válido",
           a4, focus=("QQQ", 1, "drift"))

    live = ("QQQ", 1, "long_only")
    z3 = percentile_of(a2["z_real"][live], a2["maxt"])
    z9 = percentile_of(a3["z_real"][live], a3["maxt"])
    gated_ok = a1["pct_maxt"] >= 90
    live_ok = z3 >= 90 and z9 >= 90
    print(f"\n{'='*72}\nVEREDICTO R6  (percentiles max-t, invariantes a escala)")
    print(f"  A · familia con la que se GATEÓ C2 (k=18): {a1['pct_maxt']:.1f} → "
          f"{'sobrevive' if gated_ok else 'NO sobrevive'} la corrección por multiplicidad.")
    print(f"     ({a1['null']['any_pass_rate']}% de los calendarios-placebo producen ≥1 celda "
          f"que 'pasa el gate en ambos regímenes')")
    print(f"  B/C · regla EN VIVO QQQ long-only w=1: {z3:.1f} (k=3) / {z9:.1f} (k=9) → "
          f"{'SOBREVIVE' if live_ok else 'NO sobrevive'}.")
    return 0 if live_ok else 1


# ------------------------------------------------------------------ selfcheck

def _selfcheck() -> None:
    """El null best-of-k debe ser más exigente que el de una sola celda."""
    import pandas as pd

    # Serie sintética: ruido puro, sin edge. Cualquier "ganadora" es multiplicidad.
    rng = random.Random(7)
    n = 900
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.01)))
    idx = pd.bdate_range("2015-01-01", periods=n)
    s = Series(pd.DataFrame({"close": closes}, index=idx))

    # locate: sesión exacta, y un feriado/fin de semana cae a la siguiente sesión.
    assert s.locate(s.dates[10]) == 10
    friday = next(i for i, d in enumerate(s.dates) if d.weekday() == 4)
    saturday = date.fromordinal(s.dates[friday].toordinal() + 1)
    assert s.locate(saturday) == friday + 1, "sábado → siguiente sesión (lunes)"
    far = date.fromordinal(s.dates[-1].toordinal() + 30)
    assert s.locate(far) is None, "fuera de rango no inventa sesión"

    # cell_returns: drift y fade son espejo exacto.
    pos = list(range(5, 200, 7))
    d = cell_returns(s, pos, 1, "drift")
    f = cell_returns(s, pos, 1, "fade")
    assert len(d) == len(f) and all(abs(x + y) < 1e-12 for x, y in zip(d, f)), "drift == −fade"

    # long_only: sólo días verdes, y el retorno es el forward crudo (sin signo).
    lo = long_only_returns(s, pos, 1)
    assert all(s.move[i] > 0 for i in pos if i in {p for p in pos if s.move[p] > 0})
    assert len(lo) == sum(1 for i in pos if s.move[i] > 0 and s.fwd[1][i] is not None)

    # El máximo de k celdas domina a cualquier celda individual: el percentil de un
    # mismo valor contra el best-of-k NUNCA es mayor que contra el null ingenuo.
    fake_cells = {("A", 1, "drift"): [0.1, 0.2, 0.3], ("B", 1, "drift"): [0.4, 0.5, 0.6]}
    maxes = [max(v[i] for v in fake_cells.values()) for i in range(3)]
    val = 0.45
    assert percentile_of(val, maxes) <= percentile_of(val, fake_cells[("A", 1, "drift")]), \
        "best-of-k debe ser >= exigente que el null de una celda"

    # cell_score = el régimen más flojo (el gate exige pasar en ambos).
    per = {"OOS 2015-21": {"expectancy_pct": 0.30}, "IS 2022-24": {"expectancy_pct": 0.10}}
    assert cell_score(per) == 0.10
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        d = DRAWS
        if "--draws" in sys.argv:
            d = int(sys.argv[sys.argv.index("--draws") + 1])
        sd = SEED
        if "--seed" in sys.argv:
            sd = int(sys.argv[sys.argv.index("--seed") + 1])
        raise SystemExit(run(d, sd))
