"""R4 — Auditoría de los dos caballos NO-MR en vivo contra la barra vigente.

Cuarta aplicación del patrón R1/R2/R3: **auditar la regla que CORRE**, no la que
se validó. `momentum_rotation` (15 slots) y `donchian_breakout` (10 slots) llevan
en paper desde el arranque / 2026-06-25 y nunca enfrentaron el método actual
(placebo condicionado de R1, duration-matched de R3, jackknife de F3/F4, control
sin supervivencia de R3).

Divergencia live↔backtest encontrada leyendo el código — MAYOR que la de R3:

    momentum_rotation
      REGISTRADO: `strategy_momentum_rotation(df)` → señal POR SÍMBOLO
                  (`entry = momentum_score > 0`). Eso es lo que un gate
                  per-symbol mediría.
      VIVO:       `orchestrator._run_momentum` → `momentum_top(bars, 15,
                  sector_of=_sector_of, max_per_sector=3)`, rebalanceo el
                  primer día hábil del mes, equal-weight.
      ⇒ el vivo es una regla CROSS-SECTIONAL (ranking + capacidad + cap
        sectorial). "score > 0" no selecciona nada: en un bull la mayoría del
        universo lo cumple. Lo que decide el P&L es el RANKING, y el ranking
        nunca se gateó. El cap sectorial (MAX_PER_SECTOR=3) sólo existe en el
        vivo — no hay backtest de él en ninguna parte del repo.

    donchian_breakout
      Misma señal en ambos caminos (`fn(df)` sin inyecciones), pero el vivo
      ordena por FUERZA de ruptura (`breakout_candidates`) y llena 10 slots
      ⇒ opera el tranche más extremo, no la señal promedio (lección 4 de R3).
      Además la familia se declaró agotada DESPUÉS de desplegarla: H2 (commodities,
      2026-07-14) y H5 (variante ATR, 2026-07-16) FAIL — pero el caballo que corre
      es Donchian 20/10 sobre LARGE-CAPS, celda que nunca se gateó.

Método (barra vigente):
  · Split OOS 2015-21 / IS 2022-24.
  · PLACEBO CONDICIONADO Y DURATION-MATCHED (R1 + R3):
      - momentum: 15 nombres AL AZAR del mismo pool elegible, mantenidos los
        MISMOS periodos. Es el control honesto: "¿el ranking aporta algo sobre
        estar largo en 15 nombres cualesquiera del universo?". Un placebo
        random-day sería ciego — el edge aparente de un long-only en un bull es
        estar invertido, no el momentum.
      - donchian: días "casi-ruptura" (cierre en el decil alto del rango previo
        de 20d pero SIN superar el máximo). Aísla LA RUPTURA de ESTAR CERCA DEL
        MÁXIMO. Duration-matched.
  · CONTROL SIN SUPERVIVENCIA (R3): la misma regla sobre ETFs de índice. El
    universo es el snapshot ESTÁTICO del S&P de HOY; momentum sobre sobrevivientes
    compra justo a los que sabemos que siguieron subiendo.
  · JACKKNIFE por símbolo (fracción de grupos positivos, R3 regla 1) + LOYO.
  · TRANCHE OPERADO: momentum top-15 vs bottom-15; donchian quintil de ruptura
    más fuerte (lo que el vivo elige) vs el más débil.
  · PLOMERÍA: momentum se corre CON y SIN cap sectorial para medir cuánto vale
    una perilla que sólo existe en el vivo.

Uso: python -m events.trend_audit [n_symbols]
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
from events.field_audit import ETF_CONTROL, REGIMES, ROUND_TRIP_COST_PCT, sample_symbols
from events.panic_study import percentile_of
# Se importa del módulo VIVO a propósito (disciplina de R2): se mide la regla que
# corre, no una reimplementación que podría diferir.
from live.orchestrator import MAX_PER_SECTOR, _sector_of
from live.portfolio_targets import momentum_top

MOM_SLOTS = 15          # run_trading_system.py:74
DONCHIAN_SLOTS = 10     # run_trading_system.py:79
DONCHIAN_ENTRY_LOOKBACK = 20
DONCHIAN_EXIT_LOOKBACK = 10
NEAR_BREAKOUT_DECILE = 0.90  # placebo donchian: cierre en el decil alto del rango 20d
PLACEBO_DRAWS = 200
PLACEBO_SEED = 42
JACKKNIFE_K = 3


# ---------------------------------------------------------------- momentum

def rebalance_positions(index: pd.DatetimeIndex, years: Tuple[int, int]) -> List[int]:
    """Posiciones del primer día hábil de cada mes dentro del rango de años.

    Réplica de `run_trading_system._is_first_trading_day_of_month`, que ancla el
    rebalanceo mensual del vivo al calendario de SPY.
    """
    out: List[int] = []
    seen: set = set()
    for i, ts in enumerate(index):
        key = (ts.year, ts.month)
        if key in seen:
            continue
        seen.add(key)
        if years[0] <= ts.year < years[1]:
            out.append(i)
    return out


def _basket_return(
    basket: Sequence[str], bars: Dict[str, pd.DataFrame], t0: pd.Timestamp, t1: pd.Timestamp,
) -> Tuple[Optional[float], Dict[str, float]]:
    """Retorno equal-weight de mantener `basket` de t0 a t1, y el de cada nombre."""
    per_name: Dict[str, float] = {}
    for sym in basket:
        df = bars.get(sym)
        if df is None:
            continue
        window = df.loc[t0:t1, "close"]
        if len(window) < 2 or window.iloc[0] <= 0:
            continue
        per_name[sym] = float(window.iloc[-1] / window.iloc[0] - 1.0)
    if not per_name:
        return None, {}
    return st.mean(per_name.values()), per_name


def momentum_cell(
    bars: Dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    years: Tuple[int, int],
    *,
    slots: int = MOM_SLOTS,
    sector_cap: Optional[int] = MAX_PER_SECTOR,
) -> Dict[str, object]:
    """Un régimen de la regla de momentum tal como corre en vivo.

    Evento = un periodo de tenencia entre rebalanceos mensuales. El retorno del
    periodo es el equal-weight de la canasta, que es el P&L que el caballo genera.
    """
    marks = rebalance_positions(calendar, years)
    basket_rets: List[float] = []
    bottom_rets: List[float] = []
    per_symbol: Dict[str, List[float]] = {}
    by_year: Dict[str, List[float]] = {}
    pools: List[Tuple[pd.Timestamp, pd.Timestamp, List[str]]] = []
    turnover: List[float] = []
    prev_basket: set = set()

    for a, b in zip(marks, marks[1:]):
        t0, t1 = calendar[a], calendar[b]
        # Vista truncada: sólo barras hasta t0 (no mirar el futuro al rankear).
        local = {s: df.loc[:t0] for s, df in bars.items() if len(df.loc[:t0]) > 0}
        eligible = [s for s, df in local.items() if len(df) > 150]
        if len(eligible) < 2:
            continue
        top = (momentum_top(local, slots, sector_of=_sector_of, max_per_sector=sector_cap)
               if sector_cap is not None else momentum_top(local, slots))
        if not top:
            continue
        ret, per_name = _basket_return(top, bars, t0, t1)
        if ret is None:
            continue
        basket_rets.append(ret)
        by_year.setdefault(str(t0.year), []).append(ret)
        for sym, r in per_name.items():
            per_symbol.setdefault(sym, []).append(r)
        pools.append((t0, t1, eligible))
        turnover.append(1.0 - len(prev_basket & set(top)) / max(len(top), 1))
        prev_basket = set(top)

        # Tranche opuesto: los 15 PEORES por score, mismo periodo. Si el ranking
        # informa, top y bottom no pueden rendir igual.
        worst = momentum_bottom(local, slots)
        if worst:
            wret, _ = _basket_return(worst, bars, t0, t1)
            if wret is not None:
                bottom_rets.append(wret)

    # Placebo: 15 nombres AL AZAR del pool elegible, mismos periodos.
    # Degenerado (y por tanto omitido) cuando el pool no es mayor que la canasta:
    # ahí "elegir al azar" devuelve el pool entero y el control sería la regla misma.
    # Es el caso del control ETF, donde la celda ES el índice equal-weight.
    rng = random.Random(PLACEBO_SEED)
    draws: List[float] = []
    if all(len(e) > slots for _, _, e in pools):
        for _ in range(PLACEBO_DRAWS):
            sample: List[float] = []
            for t0, t1, eligible in pools:
                pick = rng.sample(eligible, slots)
                r, _ = _basket_return(pick, bars, t0, t1)
                if r is not None:
                    sample.append(r)
            if sample:
                draws.append(st.mean(sample) * 100)

    stats = event_study(basket_rets)
    return {
        "stats": stats,
        "bottom": event_study(bottom_rets),
        "placebo_pct": percentile_of(stats["expectancy_pct"], draws) if draws else None,
        "placebo_mean_pct": round(st.mean(draws), 3) if draws else None,
        "jk": jackknife_by_group(per_symbol, k=JACKKNIFE_K) if per_symbol else None,
        "loyo": leave_one_year_out(by_year) if by_year else None,
        "symbols": len(per_symbol),
        "turnover": round(st.mean(turnover) * 100, 1) if turnover else 0.0,
        "periods": len(basket_rets),
    }


def momentum_bottom(bars_by_symbol: Dict[str, pd.DataFrame], n: int) -> List[str]:
    """Los `n` símbolos con PEOR score de momentum (espejo de `momentum_top`)."""
    from live.portfolio_targets import momentum_scores
    scores = momentum_scores(bars_by_symbol)
    return [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1])][:n]


# ---------------------------------------------------------------- donchian

def donchian_trades(
    df: pd.DataFrame, years: Tuple[int, int],
) -> Tuple[List[float], List[int], List[float], Dict[str, List[float]]]:
    """Trades de la regla que CORRE: entra en ruptura de 20d, sale en mínimo de 10d.

    Sin cap de tiempo (`max_hold_days=None` en el registry) y sin re-entrada
    mientras está dentro (el vivo excluye los nombres ya en cartera).
    Devuelve (retornos, duraciones, fuerza de ruptura, retornos por año).
    """
    fn = STRATEGY_REGISTRY["donchian_breakout"]["fn"]
    sig = fn(df)
    close = df["close"].tolist()
    entries = sig["entry"].tolist()
    exits = sig["exit"].tolist()
    prior_high = df["high"].rolling(DONCHIAN_ENTRY_LOOKBACK).max().shift(1).tolist()
    idx = df.index
    rets: List[float] = []
    durs: List[int] = []
    strength: List[float] = []
    by_year: Dict[str, List[float]] = {}
    i, busy_until = 0, -1
    while i < len(close) - 1:
        if entries[i] and i > busy_until and years[0] <= idx[i].year < years[1]:
            j = len(close) - 1
            for k in range(i + 1, len(close)):
                if exits[k]:
                    j = k
                    break
            r = close[j] / close[i] - 1.0
            rets.append(r)
            durs.append(j - i)
            ph = prior_high[i]
            strength.append(close[i] / ph - 1.0 if ph and ph > 0 else 0.0)
            by_year.setdefault(str(idx[i].year), []).append(r)
            busy_until = j
        i += 1
    return rets, durs, strength, by_year


def near_breakout_days(df: pd.DataFrame, years: Tuple[int, int]) -> List[int]:
    """Días CASI-ruptura: cierre en el decil alto del rango previo 20d, sin romperlo.

    Es el control condicionado que exige la regla de R1: si el placebo fueran días
    al azar, el percentil mediría "estar en tendencia", no la ruptura.
    """
    high = df["high"].rolling(DONCHIAN_ENTRY_LOOKBACK).max().shift(1)
    low = df["low"].rolling(DONCHIAN_ENTRY_LOOKBACK).min().shift(1)
    rng = high - low
    pos = (df["close"] - low) / rng.where(rng > 0)
    mask = (pos >= NEAR_BREAKOUT_DECILE) & (df["close"] <= high)
    mask = mask.fillna(False).tolist()
    idx = df.index
    return [i for i in range(len(mask) - 1)
            if mask[i] and years[0] <= idx[i].year < years[1]]


def donchian_cell(bars: Dict[str, pd.DataFrame], years: Tuple[int, int]) -> Dict[str, object]:
    per_symbol: Dict[str, List[float]] = {}
    per_symbol_dur: Dict[str, List[int]] = {}
    by_year: Dict[str, List[float]] = {}
    pools: Dict[str, Tuple[List[int], List[float]]] = {}
    ranked: List[Tuple[float, float]] = []

    for sym, df in bars.items():
        rets, durs, strength, years_map = donchian_trades(df, years)
        if rets:
            per_symbol[sym] = rets
            per_symbol_dur[sym] = durs
            ranked.extend(zip(strength, rets))
            for y, v in years_map.items():
                by_year.setdefault(y, []).extend(v)
        days = near_breakout_days(df, years)
        if days:
            pools[sym] = (days, df["close"].tolist())

    all_r = [r for v in per_symbol.values() for r in v]
    stats = event_study(all_r)
    all_dur = [d for v in per_symbol_dur.values() for d in v]

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

    # Tranche que el vivo opera: `breakout_candidates` ordena por fuerza DESC y
    # llena 10 slots ⇒ se queda con las rupturas más extremas.
    ranked.sort(key=lambda t: t[0], reverse=True)
    q = max(1, len(ranked) // 5)
    return {
        "stats": stats,
        "strongest_q": event_study([r for _, r in ranked[:q]]),
        "weakest_q": event_study([r for _, r in ranked[-q:]]),
        "avg_hold_days": round(st.mean(all_dur), 1) if all_dur else 0.0,
        "placebo_pct": percentile_of(stats["expectancy_pct"], draws) if draws else None,
        "placebo_mean_pct": round(st.mean(draws), 3) if draws else None,
        "jk": jackknife_by_group(per_symbol, k=JACKKNIFE_K) if per_symbol else None,
        "loyo": leave_one_year_out(by_year) if by_year else None,
        "symbols": len(per_symbol),
    }


# ---------------------------------------------------------------- reporte

def _common_lines(cell: Dict[str, object], cost: float) -> List[str]:
    s = cell["stats"]
    placebo = (
        f"pct {cell['placebo_pct']}  (control exp {cell['placebo_mean_pct']:+.3f}%)"
        if cell["placebo_pct"] is not None else "n/a (pool ≤ canasta: el control sería la regla)"
    )
    lines = [
        f"    n={s['n']:<5d} exp={s['expectancy_pct']:+.3f}%  (neto {s['expectancy_pct'] - cost:+.3f}%)  "
        f"hit={s['hit_rate']:.0%}  tail={s['tail_ratio']}  maxL={s['max_loss_pct']:.1f}%  "
        f"symbols={cell['symbols']}",
        f"    placebo cond.+duration-matched: {placebo}",
    ]
    jk, loyo = cell["jk"], cell["loyo"]
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
    return lines


def fmt_momentum(cell: Dict[str, object]) -> str:
    # Coste: sólo la fracción de la canasta que rota cada mes paga round-trip.
    cost = ROUND_TRIP_COST_PCT * cell["turnover"] / 100.0
    lines = _common_lines(cell, cost)
    b = cell["bottom"]
    lines.insert(1, f"    periodos={cell['periods']}  turnover mensual≈{cell['turnover']}%")
    lines.append(
        f"    tranche opuesto (bottom-{MOM_SLOTS} por score): exp {b['expectancy_pct']:+.3f}%  "
        f"hit={b['hit_rate']:.0%}  n={b['n']}"
    )
    return "\n".join(lines)


def fmt_donchian(cell: Dict[str, object]) -> str:
    lines = _common_lines(cell, ROUND_TRIP_COST_PCT)
    lines.insert(1, f"    hold≈{cell['avg_hold_days']}d (sin cap de tiempo: sale en mínimo de 10d)")
    sq, wq = cell["strongest_q"], cell["weakest_q"]
    lines.append(
        f"    tranche que el vivo opera (ruptura más fuerte, quintil 1): exp {sq['expectancy_pct']:+.3f}%  "
        f"hit={sq['hit_rate']:.0%}  n={sq['n']}   |   quintil 5 (más débil): "
        f"exp {wq['expectancy_pct']:+.3f}%  hit={wq['hit_rate']:.0%}"
    )
    return "\n".join(lines)


def _verdict(cell: Dict[str, object]) -> str:
    s = cell["stats"]
    if s["n"] < MIN_EVENTS:
        return "n<MIN"
    return "exp+" if s["expectancy_pct"] > 0 else "exp−"


def run(n_symbols: int = 60) -> None:
    from backtesting.engine import load_bars

    symbols = sample_symbols(n_symbols)
    start, end = date(2014, 1, 1), date(2025, 1, 1)  # warm-up para el lookback 126+21
    print(f"R4 — auditoría de los caballos NO-MR EN VIVO · muestra {len(symbols)} símbolos S&P")
    print("Bajando barras…", flush=True)
    bars: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_bars(sym, start, end, "1d", source="yfinance")
        if df is not None and len(df) > 250:
            bars[sym] = df
    print(f"  {len(bars)}/{len(symbols)} símbolos con datos")

    etf: Dict[str, pd.DataFrame] = {}
    for sym in ETF_CONTROL:
        df = load_bars(sym, start, end, "1d", source="yfinance")
        if df is not None and len(df) > 250:
            etf[sym] = df
    print(f"  {len(etf)} ETFs de control\n")

    spy = load_bars("SPY", start, end, "1d", source="yfinance")
    calendar = spy.index if spy is not None and len(spy) else next(iter(bars.values())).index

    print(f"=== momentum_rotation  (VIVO: top-{MOM_SLOTS} mensual, cap sectorial {MAX_PER_SECTOR}) ===")
    for regime, years in REGIMES.items():
        for label, cap in ((f"LIVE     (cap sectorial {MAX_PER_SECTOR})", MAX_PER_SECTOR),
                           ("SIN CAP  (nunca backtesteado tampoco)", None)):
            cell = momentum_cell(bars, calendar, years, sector_cap=cap)
            print(f"  [{regime}] {label}  → {_verdict(cell)}")
            print(fmt_momentum(cell))
        if etf:
            cell = momentum_cell(etf, calendar, years, slots=min(MOM_SLOTS, len(etf)), sector_cap=None)
            print(f"  [{regime}] CONTROL ETFs sin supervivencia ({','.join(etf)})")
            print(fmt_momentum(cell))
        print()

    print(f"=== donchian_breakout  (VIVO: {DONCHIAN_SLOTS} slots, ranking por fuerza de ruptura) ===")
    for regime, years in REGIMES.items():
        cell = donchian_cell(bars, years)
        print(f"  [{regime}] LIVE = BACKTEST (misma llamada fn(df))  → {_verdict(cell)}")
        print(fmt_donchian(cell))
        if etf:
            cell = donchian_cell(etf, years)
            print(f"  [{regime}] CONTROL ETFs sin supervivencia ({','.join(etf)})")
            print(fmt_donchian(cell))
        print()


if __name__ == "__main__":
    # Los invariantes del harness viven en tests/test_trend_audit.py (pytest).
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
