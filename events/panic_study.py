#!/usr/bin/env python3
"""
F1 — Rebote post-pánico (carril COLA GORDA, docs/RESEARCH-BACKLOG.md).

Tesis: tras una caída extrema de 1 día en el mercado (SPY ≤ −3% / −4%), la
liquidación forzada deja sobre-venta y el rebote a 1-20d tiene **cola derecha
gorda**. Eventos raros por diseño (N chico) — es la forma de payoff del carril:
tail_ratio ≥ 3 pesa más que hit rate.

Diferencia con `event_study.py`: ahí el evento es una FECHA de calendario y la
señal es el signo del día; aquí el evento es **condicional al precio** (pánico) y
la señal es fija: LARGO al cierre del día de pánico, salida a w sesiones.

Uso:
    python events/panic_study.py SPY,QQQ,IWM,ARKK -3 2015:2022
    python events/panic_study.py --selfcheck
"""
from __future__ import annotations

import random
import statistics as st
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from events.event_study import event_study, gate_event  # noqa: E402

TRIGGER_SYMBOL = "SPY"       # el pánico es de MERCADO, no del símbolo medido
WINDOWS = (1, 5, 10, 20)
PLACEBO_DRAWS = 500          # muestras random para el killer test
PLACEBO_SEED = 20260727      # determinista: el veredicto debe ser reproducible
MIN_TAIL_COLA_GORDA = 3.0    # umbral del carril (informativo; el gate sigue en 1.2)


def panic_dates(df, threshold_pct: float) -> List:
    """Fechas (timestamps del índice) cuyo retorno diario ≤ threshold_pct. Puro."""
    rets = df["close"].pct_change()
    thr = threshold_pct / 100.0
    return [ts for ts, r in rets.items() if r == r and r <= thr]


def forward_returns(df, event_ts: Sequence, window: int) -> List[float]:
    """Retorno LARGO de `window` sesiones desde el cierre de cada evento. Puro.

    Long-only y sin signo: la tesis es rebote, no continuación.
    """
    idx = list(df.index)
    pos = {ts: i for i, ts in enumerate(idx)}
    close = df["close"]
    out: List[float] = []
    for ts in event_ts:
        i = pos.get(ts)
        if i is None or i + window >= len(idx):
            continue
        out.append(float(close.iloc[i + window] / close.iloc[i] - 1))
    return out


def p90(returns: Sequence[float]) -> float:
    """Percentil 90 (la cola derecha que este carril compra). Puro."""
    if not returns:
        return 0.0
    s = sorted(returns)
    k = int(round(0.9 * (len(s) - 1)))
    return round(s[k] * 100, 3)


def placebo(df, exclude_ts: Sequence, window: int, n: int,
            draws: int = PLACEBO_DRAWS, seed: int = PLACEBO_SEED) -> Dict[str, float]:
    """Killer test: mismas N muestras pero en días NO-pánico, `draws` veces.

    Devuelve la media de expectativa/tail del placebo y el percentil en que cae
    el resultado real. Si el real no está arriba del ~90 del placebo, el "edge"
    es solo el drift genérico del mercado y la hipótesis muere (mató a C1).
    """
    idx = list(df.index)
    excl = set(exclude_ts)
    pool = [ts for ts in idx[:-window] if ts not in excl]
    if n == 0 or len(pool) < n:
        return {"exp_mean": 0.0, "tail_mean": 0.0, "p90_mean": 0.0,
                "exp_dist": [], "tail_dist": [], "p90_dist": []}
    rng = random.Random(seed)
    exps, tails, p90s = [], [], []
    for _ in range(draws):
        sample = rng.sample(pool, n)
        rets = forward_returns(df, sample, window)
        if not rets:
            continue
        s = event_study(rets)
        exps.append(s["expectancy_pct"])
        tails.append(s["tail_ratio"])
        p90s.append(p90(rets))
    return {
        "exp_mean": round(st.mean(exps), 3) if exps else 0.0,
        "tail_mean": round(st.mean(tails), 2) if tails else 0.0,
        "p90_mean": round(st.mean(p90s), 3) if p90s else 0.0,
        "exp_dist": exps, "tail_dist": tails, "p90_dist": p90s,
    }


def percentile_of(value: float, dist: Sequence[float]) -> float:
    """% de la distribución placebo que el valor real supera. Puro."""
    if not dist:
        return 0.0
    return round(100.0 * sum(1 for d in dist if value > d) / len(dist), 1)


def run(symbols: List[str], threshold_pct: float, years=(2015, 2022),
        windows=Sequence[int]) -> None:
    from backtesting.engine import load_bars
    windows = windows if isinstance(windows, tuple) else WINDOWS
    start, end = date(years[0], 1, 1), date(years[1], 1, 1)
    trig = load_bars(TRIGGER_SYMBOL, start, end, "1d", source="yfinance")
    if trig is None or trig.empty:
        print(f"sin datos de {TRIGGER_SYMBOL}", file=sys.stderr)
        raise SystemExit(2)
    panics = panic_dates(trig, threshold_pct)
    print(f"F1 rebote post-pánico · trigger {TRIGGER_SYMBOL} ≤ {threshold_pct}% "
          f"· {years[0]}-{years[1] - 1} · {len(panics)} días de pánico\n")
    if not panics:
        print("  (cero eventos en la ventana)"); return

    for sym in symbols:
        df = load_bars(sym, start, end, "1d", source="yfinance")
        if df is None or df.empty:
            print(f"  {sym}: sin datos\n"); continue
        # los timestamps de pánico vienen de SPY: mapear a sesiones existentes del símbolo
        own = set(df.index)
        evs = [ts for ts in panics if ts in own]
        for w in windows:
            rets = forward_returns(df, evs, w)
            s = event_study(rets)
            g = gate_event(s)
            pl = placebo(df, evs, w, len(rets))
            pct_exp = percentile_of(s["expectancy_pct"], pl["exp_dist"])
            pct_tail = percentile_of(s["tail_ratio"], pl["tail_dist"])
            verdict = "PASS ✅" if g["passed"] else "FAIL ❌"
            cola = "COLA ✅" if s["tail_ratio"] >= MIN_TAIL_COLA_GORDA else "cola-flaca"
            print(f"  {sym:5s} w={w:2d}d  {verdict} {cola}  exp={s['expectancy_pct']:+.2f}%  "
                  f"hit={s['hit_rate']:.0%}  tail={s['tail_ratio']}  p90={p90(rets):+.2f}%  "
                  f"maxL={s['max_loss_pct']:.1f}%  n={s['n']}")
            print(f"          placebo: exp={pl['exp_mean']:+.2f}% tail={pl['tail_mean']} "
                  f"p90={pl['p90_mean']:+.2f}%  →  real supera {pct_exp}% (exp) / "
                  f"{pct_tail}% (tail) del placebo")
        print()


def _selfcheck() -> None:
    import pandas as pd
    # Serie construida: un solo día de −5% seguido de rebote.
    prices = [100.0] * 10 + [95.0, 97.0, 99.0, 100.0] + [100.0] * 10
    idx = pd.bdate_range("2020-01-01", periods=len(prices))
    df = pd.DataFrame({"close": prices}, index=idx)

    pans = panic_dates(df, -3.0)
    assert len(pans) == 1 and pans[0] == idx[10], f"un solo pánico, no {pans}"
    assert panic_dates(df, -10.0) == [], "umbral más duro no debe disparar"

    r1 = forward_returns(df, pans, 1)
    assert len(r1) == 1 and abs(r1[0] - (97.0 / 95.0 - 1)) < 1e-9, r1
    # ventana que se sale del rango → evento descartado, no crash
    assert forward_returns(df, [idx[-1]], 5) == []

    assert p90([]) == 0.0
    assert p90([0.01, 0.02, 0.03, 0.10]) == 10.0, "p90 toma la cola derecha"

    # placebo: en una serie plana la expectativa random es ~0 y el real la supera
    pl = placebo(df, pans, 1, 5, draws=50, seed=1)
    assert len(pl["exp_dist"]) == 50
    assert percentile_of(99.0, [0.0, 1.0, 2.0]) == 100.0
    assert percentile_of(-1.0, [0.0, 1.0, 2.0]) == 0.0
    print("selfcheck ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selfcheck"]:
        _selfcheck()
    elif len(args) in (2, 3):
        yrs = tuple(int(y) for y in args[2].split(":")) if len(args) == 3 else (2015, 2022)
        run([s.strip().upper() for s in args[0].split(",")], float(args[1]), years=yrs)
    else:
        print(__doc__)
        raise SystemExit(2)
