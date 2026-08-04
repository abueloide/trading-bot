"""R3 — Auditoría del field daily-bar EN VIVO contra la barra vigente.

Tercera aplicación del patrón R1/R2: **auditar la regla que CORRE**, no la que se
validó. Los dos caballos de mean-reversion (`confirmed_mr`, `rsi_mr`) llevan en
paper desde el arranque de la carrera y nunca enfrentaron el método actual
(placebo condicionado de R1 + jackknife de F3/F4). Y hay una divergencia abierta
desde la auditoría del 2026-07-02 (defecto #3, HIGH, GATED):

    `live/portfolio_targets.py:oversold_candidates` llama `fn(df)` PELÓN.
    `backtesting/engine.py:413` inyecta `spy_close` → filtro SPY>200dMA.

⇒ el caballo que corre en paper NO tiene el filtro de régimen con el que se
backtesteó. Esto mide las dos semánticas sobre el mismo dato:

    LIVE     = fn(df)                    (sin filtro SPY — lo que opera hoy)
    BACKTEST = fn(df, spy_close=SPY)     (con filtro SPY — lo que se validó)

Hallazgo de plomería encontrado leyendo el código: el filtro VIX de `rsi_mr` está
MUERTO EN AMBOS CAMINOS — `extra_data["vix_rank"]` (engine.py:415) no lo llena
nadie en todo el repo (`grep vix_rank` → solo la firma y el config). O sea, la
única divergencia real live↔backtest es `spy_close`.

Método (barra vigente):
  · Evento = día en que dispara `entry` de la regla. Holding = igual que el vivo:
    sale al primer día con `exit` (RSI2 > umbral) o a `max_hold_days`, lo que
    ocurra primero; sin re-entrada mientras está dentro (el vivo excluye held).
  · Split OOS 2015-21 / IS 2022-24.
  · PLACEBO CONDICIONADO (regla de R1) y DURATION-MATCHED: el control lleva TODAS
    las condiciones de entrada MENOS el gatillo de sobreventa, y se mantiene los
    MISMOS días que el trade real con el que se compara. Sin lo segundo el control
    saldría casi de inmediato (un día no-sobrevendido ya suele estar sobre el
    umbral de salida) y el percentil mediría tiempo en el mercado, no sobreventa.
  · JACKKNIFE por símbolo (F3) y leave-one-year-out.
  · CONTROL SIN SUPERVIVENCIA: la misma regla sobre ETFs de índice. El universo es
    el snapshot ESTÁTICO del S&P 500 (sobrevivientes de HOY) y comprar caídas es
    justo el trade que ese sesgo más adorna — toda caída de un sobreviviente se
    recuperó. Sin esta celda el número del universo estático no es interpretable.
  · TRANCHE OPERADO: el vivo no toma todas las señales; ordena por RSI(2) asc. y
    llena 10 slots. Se reporta el quintil más sobrevendido vs el menos.

Uso: python -m events.field_audit [n_symbols]
"""
from __future__ import annotations

import random
import statistics as st
import sys
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from backtesting.strategies import STRATEGY_REGISTRY, rsi, sma
from events.event_study import (
    MIN_EVENTS, event_study, jackknife_by_group, leave_one_year_out,
)
from events.panic_study import percentile_of
from sp500_constituents import SP500_CONSTITUENTS

STRATS = ("confirmed_mr", "rsi_mr")
# Control SIN sesgo de supervivencia: ETFs de índice que existieron todo el periodo.
# El universo S&P estático son los sobrevivientes de HOY, y comprar caídas es
# justo el trade que ese sesgo más adorna (toda caída de un sobreviviente se
# recuperó). Si el edge vive aquí también, el sesgo no lo explica todo.
ETF_CONTROL = ("QQQ", "IWM", "DIA", "IVV")
REGIMES = {"OOS 2015-21": (2015, 2022), "IS 2022-24": (2022, 2025)}
SAMPLE_N = 60
SAMPLE_SEED = 7
PLACEBO_DRAWS = 200
PLACEBO_SEED = 42
JACKKNIFE_K = 3
ROUND_TRIP_COST_PCT = 0.10  # 0.05% por lado (BACKTESTING_CONFIG.transaction_cost_pct)

# Umbral de sobreventa por caballo (el gatillo que el placebo NO lleva).
RSI_BUY = {"confirmed_mr": 15, "rsi_mr": 10}


def sample_symbols(n: int = SAMPLE_N) -> List[str]:
    syms = sorted({d["symbol"] for d in SP500_CONSTITUENTS})
    return sorted(random.Random(SAMPLE_SEED).sample(syms, min(n, len(syms))))


# ---------------------------------------------------------------- señales

def signals(strategy: str, df: pd.DataFrame, spy_close: Optional[pd.Series]):
    """Señales de la regla, en la semántica pedida.

    `spy_close=None` reproduce EXACTAMENTE la llamada del vivo (`fn(df)`).
    """
    fn = STRATEGY_REGISTRY[strategy]["fn"]
    return fn(df) if spy_close is None else fn(df, spy_close=spy_close)


def placebo_mask(strategy: str, df: pd.DataFrame, spy_close: Optional[pd.Series]) -> pd.Series:
    """Días que cumplen todo MENOS el gatillo de sobreventa.

    Réplica de las condiciones no-RSI de cada estrategia (ver
    `backtesting/strategies.py`). Si el placebo no las lleva, el percentil mide el
    filtro de tendencia y no el evento (la lección de F4/R1).
    """
    r = rsi(df["close"], period=2)
    uptrend = sma(df["close"], 50) > sma(df["close"], 200)
    ok = uptrend & (r >= RSI_BUY[strategy])
    if strategy == "confirmed_mr":
        ok = ok & (df["close"] > df["open"])
    if spy_close is not None:
        spy_aligned = spy_close.reindex(df.index).ffill()
        ok = ok & (spy_aligned > sma(spy_aligned, 200))
    return ok.fillna(False)


# ---------------------------------------------------------------- holding

def _hold_return(close: Sequence[float], exits: Sequence[bool], i: int, max_hold: int) -> Tuple[float, int]:
    """Retorno de entrar al cierre de `i` y salir como sale el vivo.

    Sale al cierre del primer día con señal de salida, o a `max_hold` sesiones.
    Devuelve (retorno, índice de salida).
    """
    last = min(i + max_hold, len(close) - 1)
    j = last
    for k in range(i + 1, last + 1):
        if exits[k]:
            j = k
            break
    return close[j] / close[i] - 1.0, j


def rule_returns(
    strategy: str, df: pd.DataFrame, spy_close: Optional[pd.Series], years: Tuple[int, int],
) -> Tuple[List[float], List[int], List[float], Dict[str, List[float]]]:
    """Retornos de la regla (sin solapar), duraciones, RSI(2) de entrada, y por año.

    El RSI(2) sale porque el vivo NO toma todas las señales: `oversold_candidates`
    las ordena por RSI(2) ascendente y el orchestrator llena solo sus 10 slots →
    lo que se opera es la cola MÁS sobrevendida, no la señal promedio.
    """
    sig = signals(strategy, df, spy_close)
    max_hold = int(STRATEGY_REGISTRY[strategy]["max_hold_days"])
    close = df["close"].tolist()
    entries = sig["entry"].tolist()
    exits = sig["exit"].tolist()
    r2 = rsi(df["close"], period=2).tolist()
    idx = df.index
    out: List[float] = []
    durations: List[int] = []
    rsi2s: List[float] = []
    by_year: Dict[str, List[float]] = {}
    i, busy_until = 0, -1
    while i < len(close) - 1:
        y = idx[i].year
        if entries[i] and i > busy_until and years[0] <= y < years[1]:
            ret, j = _hold_return(close, exits, i, max_hold)
            out.append(ret)
            durations.append(j - i)
            rsi2s.append(r2[i])
            by_year.setdefault(str(y), []).append(ret)
            busy_until = j
        i += 1
    return out, durations, rsi2s, by_year


def placebo_days(
    strategy: str, df: pd.DataFrame, spy_close: Optional[pd.Series], years: Tuple[int, int],
) -> Tuple[List[int], List[float]]:
    """Índices control (todo menos el gatillo RSI) + la serie de cierres.

    Devuelve índices, NO retornos: el control se mantiene **exactamente los mismos
    días** que el trade real con el que se compara (duration-matched). Sin eso, el
    control saldría casi de inmediato — un día no-sobrevendido ya suele estar sobre
    el umbral de salida — y el placebo compararía 1 día contra ~5 en un mercado que
    sube: mediría tiempo en el mercado, no la sobreventa.
    """
    mask = placebo_mask(strategy, df, spy_close).tolist()
    close = df["close"].tolist()
    idx = df.index
    days = [i for i in range(len(close) - 1)
            if mask[i] and years[0] <= idx[i].year < years[1]]
    return days, close


# ---------------------------------------------------------------- estudio

def audit_cell(
    strategy: str,
    bars: Dict[str, pd.DataFrame],
    spy_close: Optional[pd.Series],
    years: Tuple[int, int],
) -> Dict[str, object]:
    per_symbol: Dict[str, List[float]] = {}
    per_symbol_dur: Dict[str, List[int]] = {}
    by_year: Dict[str, List[float]] = {}
    pools: Dict[str, Tuple[List[int], List[float]]] = {}
    ranked: List[Tuple[float, float]] = []  # (rsi2 de entrada, retorno)
    for sym, df in bars.items():
        rets, durs, rsi2s, years_map = rule_returns(strategy, df, spy_close, years)
        if rets:
            per_symbol[sym] = rets
            per_symbol_dur[sym] = durs
            ranked.extend(zip(rsi2s, rets))
            for y, v in years_map.items():
                by_year.setdefault(y, []).extend(v)
        days, close = placebo_days(strategy, df, spy_close, years)
        if days:
            pools[sym] = (days, close)

    all_r = [r for v in per_symbol.values() for r in v]
    stats = event_study(all_r)
    all_dur = [d for v in per_symbol_dur.values() for d in v]

    # Placebo condicionado y DURATION-MATCHED: por cada trade real (símbolo, duración d)
    # se sortea un día control del mismo símbolo y se mantiene exactamente d sesiones.
    rng = random.Random(PLACEBO_SEED)
    draws: List[float] = []
    for _ in range(PLACEBO_DRAWS):
        sample: List[float] = []
        for sym, durs in per_symbol_dur.items():
            pool = pools.get(sym)
            if not pool:
                continue
            days, close = pool
            for d in durs:
                i = rng.choice(days)
                j = min(i + d, len(close) - 1)
                sample.append(close[j] / close[i] - 1.0)
        if sample:
            draws.append(st.mean(sample) * 100)

    # Tranche que el vivo REALMENTE opera: las señales más sobrevendidas primero.
    ranked.sort(key=lambda t: t[0])
    q = max(1, len(ranked) // 5)
    return {
        "stats": stats,
        "deepest_q": event_study([r for _, r in ranked[:q]]),
        "shallowest_q": event_study([r for _, r in ranked[-q:]]),
        "avg_hold_days": round(st.mean(all_dur), 1) if all_dur else 0.0,
        "placebo_pct": percentile_of(stats["expectancy_pct"], draws) if draws else None,
        "placebo_mean_pct": round(st.mean(draws), 3) if draws else None,
        "jk": jackknife_by_group(per_symbol, k=JACKKNIFE_K) if per_symbol else None,
        "loyo": leave_one_year_out(by_year) if by_year else None,
        "symbols": len(per_symbol),
    }


def _fmt(cell: Dict[str, object]) -> str:
    s = cell["stats"]
    jk = cell["jk"]
    loyo = cell["loyo"]
    net = s["expectancy_pct"] - ROUND_TRIP_COST_PCT
    lines = [
        f"    n={s['n']:<5d} exp={s['expectancy_pct']:+.3f}%  (neto {net:+.3f}%)  "
        f"hit={s['hit_rate']:.0%}  tail={s['tail_ratio']}  maxL={s['max_loss_pct']:.1f}%  "
        f"symbols={cell['symbols']}  hold≈{cell['avg_hold_days']}d",
        f"    placebo cond.+duration-matched: pct {cell['placebo_pct']}  "
        f"(control exp {cell['placebo_mean_pct']:+.3f}%)",
    ]
    if jk:
        lines.append(
            f"    jackknife −{JACKKNIFE_K} símbolos: {jk['jackknifed']['expectancy_pct']:+.3f}%  "
            f"(quita {','.join(jk['dropped'])})  ·  {jk['groups_positive']}/{jk['groups']} símbolos +"
        )
    if loyo:
        lines.append(
            f"    leave-one-year-out: peor año {loyo['worst_drop_year']} → "
            f"{loyo['worst_drop_exp_pct']:+.3f}%  ·  {loyo['years_positive']}/{loyo['years']} años +"
        )
    dq, sq = cell["deepest_q"], cell["shallowest_q"]
    lines.append(
        f"    tranche que el vivo opera (RSI2 más bajo, quintil 1): exp {dq['expectancy_pct']:+.3f}%  "
        f"hit={dq['hit_rate']:.0%}  n={dq['n']}   |   quintil 5 (menos sobrevendido): "
        f"exp {sq['expectancy_pct']:+.3f}%  hit={sq['hit_rate']:.0%}"
    )
    return "\n".join(lines)


def run(n_symbols: int = SAMPLE_N) -> None:
    from backtesting.engine import load_bars

    symbols = sample_symbols(n_symbols)
    start, end = date(2014, 1, 1), date(2025, 1, 1)  # 1 año extra: las SMA200 necesitan warm-up
    print(f"R3 — auditoría del field EN VIVO · muestra {len(symbols)} símbolos S&P (seed {SAMPLE_SEED})")
    print("Bajando barras…", flush=True)
    spy = load_bars("SPY", start, end, "1d", source="yfinance")
    if spy is None or spy.empty:
        print("sin SPY — abortando", file=sys.stderr)
        raise SystemExit(2)
    spy_close = spy["close"]

    bars: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_bars(sym, start, end, "1d", source="yfinance")
        if df is not None and len(df) > 250:
            bars[sym] = df
    print(f"  {len(bars)}/{len(symbols)} símbolos con datos\n")

    etf: Dict[str, pd.DataFrame] = {}
    for sym in ETF_CONTROL:
        df = load_bars(sym, start, end, "1d", source="yfinance")
        if df is not None and len(df) > 250:
            etf[sym] = df

    for strategy in STRATS:
        hold = STRATEGY_REGISTRY[strategy]["max_hold_days"]
        print(f"=== {strategy}  (max_hold={hold}d, rsi_buy={RSI_BUY[strategy]}) ===")
        for regime, years in REGIMES.items():
            for label, sc in (("LIVE     (fn(df), sin filtro SPY)", None),
                              ("BACKTEST (con filtro SPY>200dMA)", spy_close)):
                cell = audit_cell(strategy, bars, sc, years)
                s = cell["stats"]
                verdict = "n<MIN" if s["n"] < MIN_EVENTS else ("exp+" if s["expectancy_pct"] > 0 else "exp−")
                print(f"  [{regime}] {label}  → {verdict}")
                print(_fmt(cell))
            if etf:
                cell = audit_cell(strategy, etf, None, years)
                print(f"  [{regime}] CONTROL ETFs sin supervivencia ({','.join(etf)}), semántica LIVE")
                print(_fmt(cell))
            print()


if __name__ == "__main__":
    # Los invariantes del harness viven en tests/test_field_audit.py (pytest).
    run(int(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_N)
