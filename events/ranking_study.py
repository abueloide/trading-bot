"""R5 — ¿el RANKING de `donchian_breakout` contiene información?

Único item que R4 (2026-08-05) dejó abierto y que el loop puede drenar sin decisión
de Luis: *"no matar donchian pero medir rankings alternativos en un estudio aparte
con su propio OOS"*.

El hallazgo que lo motiva: con ~248 rupturas/año, hold ≈28d y 10 slots, el libro
está **permanentemente lleno** ⇒ lo que el caballo entrega no es la señal, es
`señal + ranking + capacidad`. Y el ranking que el vivo usa (fuerza de ruptura DESC,
`live/portfolio_targets.breakout_candidates`) rinde en IS **−0.534%** en su quintil
más fuerte contra **+0.659%** en el más débil (R4).

**La pregunta NO es "cuál ranking gana"** — elegir el ganador de un barrido sobre el
mismo dato que produjo el hallazgo es exactamente el error de F4 (girar la perilla
contra el dato que la eligió). La pregunta es:

    ¿algún criterio de racionamiento bate a **repartir los slots al azar entre las
    señales del día**, en AMBOS regímenes, después de corregir por haber probado k?

Si la respuesta es no, la conclusión desplegable no es "usa el ranking X" sino
**"el ranking no informa ⇒ el racionamiento debe ser neutral"**, que no es
curve-fitting porque no elige nada del dato.

Método (barra vigente):
  · **Se simula el libro que corre**, no trades pooled: 10 slots, cierre por señal
    de salida (mínimo de 10d), sin re-entrada estando dentro, y cuando hay más
    señales que slots libres el ranking decide. Réplica de
    `Orchestrator._run_slot_filler` (cierra salidas → llena slots libres con el
    top del ranking).
  · **NULL = prioridad AL AZAR entre los candidatos del mismo día.** Es el control
    honesto de R4 (regla 1): para un long-only el placebo es otra canasta del mismo
    universo. Aquí es más estricto todavía — mismo universo, mismas señales, misma
    capacidad, mismos días: **lo único que cambia es a quién le toca el slot**.
  · **Corrección por multiplicidad:** se prueban k=5 criterios ⇒ el percentil del
    mejor se compara contra la distribución del **máximo de k rankings al azar**,
    no contra la de uno solo. (Conservador: k rankings random son menos
    correlacionados entre sí que nuestros k criterios, así que su máximo es más
    exigente que el null verdadero.)
  · Split OOS 2015-21 / IS 2022-24; un criterio sólo cuenta si bate al null en LOS
    DOS. Jackknife por símbolo + LOYO sobre el mejor.
  · Dos métricas, y tienen que coincidir: **expectativa por trade** (comparable con
    el resto del repo) y **retorno anual del libro** (`Σ retornos / slots / años`),
    que es lo que el racionamiento realmente entrega — un criterio puede subir la
    expectativa por trade simplemente tomando menos trades.

Uso: python -m events.ranking_study [n_symbols]
"""
from __future__ import annotations

import random
import statistics as st
import sys
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from backtesting.strategies import STRATEGY_REGISTRY
from events.event_study import (
    MIN_EVENTS, event_study, jackknife_by_group, leave_one_year_out,
)
from events.field_audit import REGIMES, ROUND_TRIP_COST_PCT, sample_symbols
from events.panic_study import percentile_of
from events.trend_audit import DONCHIAN_ENTRY_LOOKBACK, DONCHIAN_SLOTS

STRATEGY = "donchian_breakout"
SLOTS = DONCHIAN_SLOTS       # run_trading_system.py:79 · DEFAULT_SLOTS["breakout"]
NULL_DRAWS = 300
NULL_SEED = 11
BEST_OF_K_DRAWS = 2000
JACKKNIFE_K = 3

# Los k criterios se declaran ANTES de mirar el resultado (pre-registro). Todos se
# ordenan DESC sobre el valor guardado; los que "prefieren poco" se guardan negados.
RANK_KEYS: Tuple[str, ...] = (
    "strength",       # el del VIVO: cuánto superó el máximo de 20d
    "strength_inv",   # el opuesto (R4 lo vio mejor en IS — hay que poder falsarlo)
    "atr_strength",   # fuerza normalizada por ATR14 (comparable entre volatilidades)
    "lowvol",         # menor vol realizada 20d primero
    "trend",          # más arriba de su SMA200 primero
)


# ---------------------------------------------------------------- features

def symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Señales del VIVO + el valor de cada criterio de ranking, por barra.

    `entry`/`exit` salen de `fn(df)` tal cual lo llama el orquestador (disciplina
    de R2: se mide la regla que corre). Todo criterio usa sólo datos hasta la barra
    de entrada — las ventanas son `rolling`/`shift(1)`, nunca miran adelante.
    """
    fn = STRATEGY_REGISTRY[STRATEGY]["fn"]
    sig = fn(df)
    prior_high = df["high"].rolling(DONCHIAN_ENTRY_LOOKBACK).max().shift(1)
    prev_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    vol20 = df["close"].pct_change().rolling(20).std()
    sma200 = df["close"].rolling(200).mean()
    strength = df["close"] / prior_high.where(prior_high > 0) - 1.0
    return pd.DataFrame({
        "entry": sig["entry"].astype(bool),
        "exit": sig["exit"].astype(bool),
        "close": df["close"],
        "strength": strength,
        "strength_inv": -strength,
        "atr_strength": (df["close"] - prior_high) / atr.where(atr > 0),
        "lowvol": -vol20,
        "trend": df["close"] / sma200.where(sma200 > 0) - 1.0,
    })


def build_book(bars: Dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> Dict[str, object]:
    """Aplana las features a listas alineadas al calendario común.

    El simulador corre 300+ veces; en pandas por-día sería inviable. Las señales de
    un símbolo sin barra ese día quedan en False (no cotiza ⇒ no compite por slot).
    """
    close: Dict[str, List[float]] = {}
    exits: Dict[str, List[bool]] = {}
    ranks: Dict[str, Dict[str, List[float]]] = {k: {} for k in RANK_KEYS}
    entries_by_day: List[List[str]] = [[] for _ in range(len(calendar))]

    for sym, df in bars.items():
        f = symbol_features(df).reindex(calendar)
        close[sym] = f["close"].tolist()
        exits[sym] = f["exit"].fillna(False).astype(bool).tolist()
        for k in RANK_KEYS:
            ranks[k][sym] = f[k].tolist()
        for i, on in enumerate(f["entry"].fillna(False).astype(bool).tolist()):
            if on:
                entries_by_day[i].append(sym)

    return {"close": close, "exits": exits, "ranks": ranks,
            "entries_by_day": entries_by_day, "calendar": calendar}


# ---------------------------------------------------------------- simulador

def _valid(x: Optional[float]) -> bool:
    return x is not None and x == x and x > 0  # NaN != NaN


def simulate(
    book: Dict[str, object],
    years: Tuple[int, int],
    key: Optional[str],
    *,
    rng: Optional[random.Random] = None,
    slots: int = SLOTS,
) -> Dict[str, object]:
    """Corre el libro desplegado durante `years` racionando por `key`.

    `key=None` + `rng` ⇒ NULL: la prioridad entre los candidatos del día se sortea.
    Réplica de `Orchestrator._run_slot_filler`: primero cierra las salidas (libera
    slot y cash), después llena los slots libres con el top del ranking. Las
    posiciones abiertas al final de la ventana se cierran al último cierre (si no,
    el criterio que más tarda en salir se ahorra sus pérdidas).
    """
    calendar = book["calendar"]
    close, exits, entries_by_day = book["close"], book["exits"], book["entries_by_day"]
    rank = book["ranks"][key] if key is not None else None

    held: Dict[str, int] = {}
    rets: List[float] = []
    per_symbol: Dict[str, List[float]] = {}
    by_year: Dict[str, List[float]] = {}
    durs: List[int] = []
    competed = filled = 0

    def _close(sym: str, t: int) -> None:
        i = held.pop(sym)
        r = close[sym][t] / close[sym][i] - 1.0
        rets.append(r)
        durs.append(t - i)
        per_symbol.setdefault(sym, []).append(r)
        by_year.setdefault(str(calendar[i].year), []).append(r)

    in_window = [i for i, ts in enumerate(calendar) if years[0] <= ts.year < years[1]]
    if not in_window:
        return {"stats": event_study([]), "book_pct_per_year": 0.0, "trades": 0,
                "avg_hold_days": 0.0, "fill_rate": 0.0, "per_symbol": {}, "by_year": {}}

    for t in in_window:
        for sym in [s for s in list(held) if exits[s][t] and _valid(close[s][t])]:
            _close(sym, t)
        free = slots - len(held)
        cands = [s for s in entries_by_day[t] if s not in held and _valid(close[s][t])]
        if not cands:
            continue
        competed += len(cands)
        if free <= 0:
            continue
        if rng is not None:
            order = cands[:]
            rng.shuffle(order)
        else:
            # NaN al final (un candidato sin criterio computable no se cuela arriba),
            # valor DESC, y el símbolo como desempate determinista.
            def _priority(s: str, _t: int = t):
                v = rank[s][_t]
                return (1, 0.0, s) if v != v else (0, -v, s)
            order = sorted(cands, key=_priority)
        for sym in order[:free]:
            held[sym] = t
            filled += 1

    last = in_window[-1]
    for sym in list(held):
        if _valid(close[sym][last]):
            _close(sym, last)
        else:
            held.pop(sym)

    span_years = max((calendar[in_window[-1]] - calendar[in_window[0]]).days / 365.25, 1e-9)
    return {
        "stats": event_study(rets),
        # Retorno anual del libro: cada slot es 1/slots del capital, así que la suma
        # de retornos dividida entre slots y años es lo que el caballo entrega.
        "book_pct_per_year": round(sum(rets) / slots / span_years * 100, 3),
        "trades": len(rets),
        "avg_hold_days": round(st.mean(durs), 1) if durs else 0.0,
        "fill_rate": round(filled / competed * 100, 1) if competed else 0.0,
        "per_symbol": per_symbol,
        "by_year": by_year,
    }


def null_distribution(
    book: Dict[str, object], years: Tuple[int, int], draws: int = NULL_DRAWS,
) -> Tuple[List[float], List[float]]:
    """`draws` corridas del MISMO libro con prioridad al azar. (expectativas, libros)."""
    rng = random.Random(NULL_SEED)
    exps, books = [], []
    for _ in range(draws):
        r = simulate(book, years, None, rng=rng)
        if r["trades"]:
            exps.append(r["stats"]["expectancy_pct"])
            books.append(r["book_pct_per_year"])
    return exps, books


def best_of_k_null(dist: Sequence[float], k: int, draws: int = BEST_OF_K_DRAWS) -> List[float]:
    """Distribución del MÁXIMO de k rankings al azar — corrección por multiplicidad.

    Probar k criterios y quedarse con el mejor infla el percentil: contra un null
    de una sola corrida, el mejor de 5 cruza 90 por construcción ~40% de las veces.
    El control correcto es el máximo de k sorteos.
    """
    rng = random.Random(NULL_SEED + 1)
    return [max(rng.sample(list(dist), k)) for _ in range(draws)] if len(dist) >= k else []


# ---------------------------------------------------------------- reporte

def evaluate(book: Dict[str, object], years: Tuple[int, int]) -> Dict[str, object]:
    exp_null, book_null = null_distribution(book, years)
    cells: Dict[str, Dict[str, object]] = {}
    for key in RANK_KEYS:
        r = simulate(book, years, key)
        r["exp_pct_vs_null"] = percentile_of(r["stats"]["expectancy_pct"], exp_null) if exp_null else None
        r["book_pct_vs_null"] = percentile_of(r["book_pct_per_year"], book_null) if book_null else None
        cells[key] = r
    best = max(RANK_KEYS, key=lambda k: cells[k]["book_pct_per_year"])
    bok = best_of_k_null(book_null, len(RANK_KEYS))
    return {
        "cells": cells,
        "best": best,
        "best_vs_best_of_k_pct": percentile_of(cells[best]["book_pct_per_year"], bok) if bok else None,
        "null_exp_mean": round(st.mean(exp_null), 3) if exp_null else None,
        "null_book_mean": round(st.mean(book_null), 3) if book_null else None,
        "null_book_p90": round(sorted(book_null)[int(0.9 * len(book_null))], 3) if book_null else None,
    }


def fmt_cell(key: str, c: Dict[str, object]) -> str:
    s = c["stats"]
    net = s["expectancy_pct"] - ROUND_TRIP_COST_PCT
    flag = "" if s["n"] >= MIN_EVENTS else "  ⚠ n<MIN"
    return (
        f"    {key:<14s} exp={s['expectancy_pct']:+.3f}% (neto {net:+.3f}%)  "
        f"hit={s['hit_rate']:.0%}  tail={s['tail_ratio']}  n={s['n']}{flag}\n"
        f"    {'':<14s} libro={c['book_pct_per_year']:+.2f}%/año  hold≈{c['avg_hold_days']}d  "
        f"fill={c['fill_rate']}%  ·  pct vs null: exp {c['exp_pct_vs_null']}  libro {c['book_pct_vs_null']}"
    )


def run(n_symbols: int = 60) -> None:
    from backtesting.engine import load_bars

    symbols = sample_symbols(n_symbols)
    start, end = date(2014, 1, 1), date(2025, 1, 1)  # warm-up para SMA200
    print(f"R5 — ¿el ranking de donchian informa? · {len(symbols)} símbolos S&P · {SLOTS} slots")
    print("Bajando barras…", flush=True)
    bars: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_bars(sym, start, end, "1d", source="yfinance")
        if df is not None and len(df) > 250:
            bars[sym] = df
    print(f"  {len(bars)}/{len(symbols)} símbolos con datos")

    spy = load_bars("SPY", start, end, "1d", source="yfinance")
    calendar = spy.index if spy is not None and len(spy) else next(iter(bars.values())).index
    book = build_book(bars, calendar)

    for regime, years in REGIMES.items():
        res = evaluate(book, years)
        print(f"\n=== [{regime}] ===")
        print(f"  NULL (prioridad al azar entre las señales del día, {NULL_DRAWS} corridas): "
              f"exp {res['null_exp_mean']:+.3f}%  libro {res['null_book_mean']:+.2f}%/año  "
              f"(p90 libro {res['null_book_p90']:+.2f}%)")
        for key in RANK_KEYS:
            print(fmt_cell(key, res["cells"][key]))
        best = res["cells"][res["best"]]
        print(f"  mejor criterio = {res['best']}  →  percentil contra el MÁXIMO de "
              f"{len(RANK_KEYS)} rankings al azar: {res['best_vs_best_of_k_pct']}")
        if best["per_symbol"]:
            jk = jackknife_by_group(best["per_symbol"], k=JACKKNIFE_K)
            print(f"    jackknife −{JACKKNIFE_K}: {jk['jackknifed']['expectancy_pct']:+.3f}%  "
                  f"(quita {','.join(jk['dropped'])})  ·  {jk['groups_positive']}/{jk['groups']} símbolos +")
        if best["by_year"]:
            loyo = leave_one_year_out(best["by_year"])
            print(f"    LOYO: peor año {loyo['worst_drop_year']} → {loyo['worst_drop_exp_pct']:+.3f}%  ·  "
                  f"{loyo['years_positive']}/{loyo['years']} años +")


if __name__ == "__main__":
    # Los invariantes del harness viven en tests/test_ranking_study.py (pytest).
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
