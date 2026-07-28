#!/usr/bin/env python3
"""
F2 — Explosión de volatilidad (carril COLA GORDA, docs/RESEARCH-BACKLOG.md).

Tesis: cuando el VIX salta >X% en un día, el movimiento subsecuente del índice
tiene varianza brutal. Buscar la **pata con cola derecha** (largo o corto), no el
promedio. Criterio del carril: tail_ratio ≥ 3 pesa más que hit rate.

Diferencias vs `panic_study.py` (F1):
  - Trigger = salto del ^VIX (no caída de precio de SPY) → captura expansiones de
    vol que no son días de −3%.
  - Se miden AMBAS patas (largo y corto) — F1 solo probó rebote largo.
  - **Episodios, no días** (learning de F1): los días de vol se agrupan por
    construcción; el piso de muestra se aplica sobre episodios independientes.

Uso:
    python events/vix_study.py SPY,QQQ,IWM 20 2015:2022
    python events/vix_study.py --selfcheck
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from events.event_study import event_study, gate_event  # noqa: E402
from events.panic_study import p90, percentile_of, placebo  # noqa: E402

VIX_SYMBOL = "^VIX"
WINDOWS = (1, 5, 10, 20)
EPISODE_GAP = 5          # sesiones: saltos separados por <5 sesiones = mismo episodio
MIN_EPISODES = 8         # piso de independencia (F1 murió con ~5-6 episodios efectivos)
MIN_TAIL_COLA_GORDA = 3.0
LEGS = {"largo": 1, "corto": -1}


def vix_spike_dates(vix_df, jump_pct: float) -> List:
    """Timestamps donde el ^VIX sube ≥ jump_pct% en un día. Puro."""
    rets = vix_df["close"].pct_change()
    thr = jump_pct / 100.0
    return [ts for ts, r in rets.items() if r == r and r >= thr]


def episodes(event_ts: Sequence, index: Sequence, gap: int = EPISODE_GAP) -> List:
    """Colapsa eventos separados por < `gap` sesiones en uno solo (el primero).

    Learning de F1: en eventos condicionales-al-precio la vol se autocorrelaciona,
    así que N crudo sobre-cuenta. Contar episodios independientes.
    """
    pos = {ts: i for i, ts in enumerate(index)}
    ordered = sorted(ts for ts in event_ts if ts in pos)
    out: List = []
    last = None
    for ts in ordered:
        if last is None or pos[ts] - pos[last] >= gap:
            out.append(ts)
            last = ts
    return out


def forward_returns(df, event_ts: Sequence, window: int, sign: int = 1) -> List[float]:
    """Retorno de `window` sesiones desde el cierre del evento, con signo de pata."""
    idx = list(df.index)
    pos = {ts: i for i, ts in enumerate(idx)}
    close = df["close"]
    out: List[float] = []
    for ts in event_ts:
        i = pos.get(ts)
        if i is None or i + window >= len(idx):
            continue
        out.append(sign * float(close.iloc[i + window] / close.iloc[i] - 1))
    return out


def abs_move(returns: Sequence[float]) -> float:
    """Movimiento absoluto medio (%) — la 'varianza brutal' de la tesis. Puro."""
    if not returns:
        return 0.0
    return round(100.0 * sum(abs(r) for r in returns) / len(returns), 3)


def run(symbols: List[str], jump_pct: float, years=(2015, 2022),
        windows: Sequence[int] = WINDOWS) -> None:
    from backtesting.engine import load_bars
    start, end = date(years[0], 1, 1), date(years[1], 1, 1)
    vix = load_bars(VIX_SYMBOL, start, end, "1d", source="yfinance")
    if vix is None or vix.empty:
        print(f"sin datos de {VIX_SYMBOL}", file=sys.stderr)
        raise SystemExit(2)
    spikes = vix_spike_dates(vix, jump_pct)
    print(f"F2 explosión de vol · {VIX_SYMBOL} +{jump_pct}%/día · {years[0]}-{years[1] - 1} "
          f"· {len(spikes)} días de salto\n")
    if not spikes:
        print("  (cero eventos en la ventana)")
        return

    for sym in symbols:
        df = load_bars(sym, start, end, "1d", source="yfinance")
        if df is None or df.empty:
            print(f"  {sym}: sin datos\n")
            continue
        idx = list(df.index)
        raw = [ts for ts in spikes if ts in set(idx)]
        evs = episodes(raw, idx)
        ep_ok = "ep✅" if len(evs) >= MIN_EPISODES else "ep❌"
        print(f"  {sym}  días={len(raw)} → episodios={len(evs)} {ep_ok}")
        for w in windows:
            base = forward_returns(df, evs, w, 1)
            print(f"    w={w:2d}d  |mov|={abs_move(base):.2f}%  (episodios n={len(base)})")
            for leg, sign in LEGS.items():
                rets = forward_returns(df, evs, w, sign)
                s = event_study(rets)
                g = gate_event(s)
                pl = placebo(df, raw, w, len(rets))
                if sign < 0:  # el placebo es largo; para la pata corta se invierte
                    pl = {**pl, "exp_mean": -pl["exp_mean"],
                          "exp_dist": [-e for e in pl["exp_dist"]]}
                pct_exp = percentile_of(s["expectancy_pct"], pl["exp_dist"])
                pct_tail = percentile_of(s["tail_ratio"], pl["tail_dist"])
                gate_pass = g["passed"] and len(evs) >= MIN_EPISODES
                verdict = "PASS ✅" if gate_pass else "FAIL ❌"
                cola = "COLA ✅" if s["tail_ratio"] >= MIN_TAIL_COLA_GORDA else "cola-flaca"
                print(f"      {leg:5s} {verdict} {cola}  exp={s['expectancy_pct']:+.2f}%  "
                      f"hit={s['hit_rate']:.0%}  tail={s['tail_ratio']}  p90={p90(rets):+.2f}%  "
                      f"maxL={s['max_loss_pct']:.1f}%  | placebo exp={pl['exp_mean']:+.2f}% "
                      f"tail={pl['tail_mean']} → supera {pct_exp}%/{pct_tail}%")
        print()


def _selfcheck() -> None:
    import pandas as pd
    idx = pd.bdate_range("2020-01-01", periods=30)
    # VIX plano con dos saltos: uno en i=5 y otro en i=6 (mismo episodio) y uno en i=20.
    vix = [20.0] * 30
    vix[5] = 26.0   # +30%
    vix[6] = 34.0   # +30% respecto al día previo → contiguo
    vix[20] = 30.0  # +50% respecto a 20.0
    vdf = pd.DataFrame({"close": vix}, index=idx)

    sp = vix_spike_dates(vdf, 20.0)
    assert sp == [idx[5], idx[6], idx[20]], sp
    assert vix_spike_dates(vdf, 60.0) == [], "umbral alto no debe disparar"

    eps = episodes(sp, list(idx))
    assert eps == [idx[5], idx[20]], f"i=6 debe colapsar en el episodio de i=5: {eps}"

    prices = [100.0 + i for i in range(30)]
    pdf = pd.DataFrame({"close": prices}, index=idx)
    long1 = forward_returns(pdf, [idx[5]], 1, 1)
    short1 = forward_returns(pdf, [idx[5]], 1, -1)
    assert abs(long1[0] - (106.0 / 105.0 - 1)) < 1e-9, long1
    assert abs(short1[0] + long1[0]) < 1e-12, "la pata corta es el negativo exacto"
    assert forward_returns(pdf, [idx[-1]], 5, 1) == [], "ventana fuera de rango se descarta"

    assert abs_move([]) == 0.0
    assert abs_move([0.01, -0.03]) == 2.0, "promedio de |ret| en %"
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
