#!/usr/bin/env python3
"""
R2 — Auditoría del NEWS REACTOR contra la barra vigente (backtest que "no existía").

El reactor (`live/news_reactor.py`) corre EN VIVO en paper desde 2026-07-30 con la
justificación explícita de que **no se podía backtestear**: "el histórico de noticias
del tier actual topa en 1 página (medido 2026-07-21)". Eso era un **bug de paginación**
(ver `events/news_backfill.py`): el histórico llega a 2016 con filtro por símbolo.
Con el histórico desbloqueado, la regla desplegada SÍ se puede medir hacia atrás.

Qué se mide: **la regla que CORRE**, importando `classify` y `pick_symbol` del módulo
vivo (lección R1: auditar lo desplegado, no una reimplementación que se le parezca).
Evento = titular con catalizador ALCISTA + exactamente 1 ticker. Señal = LARGO fijo,
entrada al cierre de la sesión del titular, salida `w` sesiones después (w=1 es el
`HOLD_DAYS` desplegado).

Killer tests vigentes (todos, no un subconjunto):
- Split OOS 2016-2021 vs IS 2022-2024 (regímenes distintos).
- **Placebo CONDICIONADO** (lección R1/F4): el control son días en que el símbolo SÍ
  tuvo noticias pero SIN catalizador alcista. Un control de días random mide "¿el
  símbolo sube?", no "¿el catalizador predice?".
- Jackknife por símbolo (F3) y por evento (F4) + leave-one-year-out.
- Episodios, no titulares (F1): varios titulares del mismo símbolo el mismo día = 1.

Sesgos declarados: universo de 31 supervivientes líquidos (el reactor opera cualquier
ticker de Benzinga, incluidas small caps y adquiridas) → el sesgo de supervivencia
**infla** el resultado. Un FAIL aquí es más fuerte que el número; un PASS necesitaría
re-medirse en el universo real antes de creerle.

Uso:
    python events/news_catalyst_study.py            # universo default
    python events/news_catalyst_study.py AAPL,MSFT
"""
from __future__ import annotations

import random
import statistics as st
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from events.event_study import (event_study, gate_event, jackknife_by_event,
                                jackknife_by_group, leave_one_year_out)
from events.news_backfill import load_cached
from live.news_reactor import classify, pick_symbol

UNIVERSE = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "BAC",
            "GS", "XOM", "CVX", "JNJ", "PFE", "MRK", "ABBV", "WMT", "HD", "MCD",
            "KO", "PG", "DIS", "NKE", "BA", "CAT", "GE", "T", "VZ", "INTC", "AMD", "MU"]
WINDOWS = (1, 3, 5)
SPLITS = {"OOS 2016-21": (2016, 2022), "IS 2022-24": (2022, 2025)}
PLACEBO_DRAWS = 500
PLACEBO_SEED = 20260803
YEARS = (2016, 2025)


# ------------------------------------------------------------------ eventos

def catalyst_days(symbol: str, articles: Sequence[dict]) -> Tuple[Dict[str, str], List[str]]:
    """Aplica la regla DESPLEGADA a los titulares cacheados.

    Devuelve `(dias_con_catalizador_alcista -> tipo, dias_con_noticia_sin_catalizador)`.
    El segundo es la piscina del placebo condicionado. Puro.
    """
    bullish: Dict[str, str] = {}
    newsy: set = set()
    for a in articles:
        ts = str(a.get("created_at") or "")[:10]
        if not ts:
            continue
        newsy.add(ts)
        cat = classify(a.get("headline", ""))
        if cat is None or cat.direction != "bullish":
            continue
        if pick_symbol(a.get("symbols") or []) != symbol:
            continue  # roundup multi-ticker: el reactor tampoco opera
        bullish.setdefault(ts, cat.kind)   # 1er catalizador del día gana (como el reactor)
    control = sorted(newsy - set(bullish))
    return bullish, control


def long_returns(df, day_strings: Sequence[str], window: int) -> List[Tuple[str, str, float]]:
    """Retorno LARGO de `window` sesiones desde el cierre del día del evento.

    Devuelve `(dia_evento, fecha_sesion, retorno)`. Titular fuera de sesión (noche o
    fin de semana) entra en la siguiente sesión — sin lookahead, es lo que el reactor
    puede ejecutar.
    """
    idx = list(df.index)
    dates = [ts.date() for ts in idx]
    out: List[Tuple[str, str, float]] = []
    close = df["close"]
    for ds in day_strings:
        target = date.fromisoformat(ds)
        i = next((k for k, d in enumerate(dates) if d >= target), None)
        if i is None or (dates[i] - target).days > 4 or i + window >= len(idx):
            continue
        out.append((ds, dates[i].isoformat(), float(close.iloc[i + window] / close.iloc[i] - 1)))
    return out


def percentile_of(value: float, dist: Sequence[float]) -> float:
    if not dist:
        return 0.0
    return round(100.0 * sum(1 for d in dist if value > d) / len(dist), 1)


def conditioned_placebo(df, control_days: Sequence[str], window: int, n: int,
                        draws: int = PLACEBO_DRAWS, seed: int = PLACEBO_SEED) -> Dict:
    """Mismas N muestras, pero en días CON noticia y SIN catalizador alcista.

    Aísla el catalizador de "el símbolo está en las noticias" (lección R1). Puro
    salvo por el df de precios.
    """
    if n == 0 or len(control_days) < n:
        return {"exp_dist": [], "tail_dist": [], "exp_mean": 0.0, "tail_mean": 0.0, "pool": len(control_days)}
    rng = random.Random(seed)
    exps, tails = [], []
    for _ in range(draws):
        rets = [r for _, _, r in long_returns(df, rng.sample(list(control_days), n), window)]
        if not rets:
            continue
        s = event_study(rets)
        exps.append(s["expectancy_pct"])
        tails.append(s["tail_ratio"])
    return {"exp_dist": exps, "tail_dist": tails,
            "exp_mean": round(st.mean(exps), 3) if exps else 0.0,
            "tail_mean": round(st.mean(tails), 2) if tails else 0.0,
            "pool": len(control_days)}


# ------------------------------------------------------------------ corrida

def run(symbols: List[str] = None) -> Dict:
    from backtesting.engine import load_bars

    symbols = symbols or UNIVERSE
    print(f"R2 · News reactor backtest — regla DESPLEGADA sobre {len(symbols)} símbolos, "
          f"{YEARS[0]}-{YEARS[1]-1}\n")

    # por split -> por ventana -> por símbolo -> retornos ; + placebo agregado
    pooled: Dict[str, Dict[int, Dict[str, List[float]]]] = {
        s: {w: defaultdict(list) for w in WINDOWS} for s in SPLITS}
    by_year: Dict[int, Dict[str, List[float]]] = {w: defaultdict(list) for w in WINDOWS}
    by_catalyst: Dict[int, Dict[str, List[float]]] = {w: defaultdict(list) for w in WINDOWS}
    placebo_pool: Dict[str, Dict[int, List[float]]] = {s: {w: [] for w in WINDOWS} for s in SPLITS}
    n_articles = n_events = 0

    for sym in symbols:
        arts = load_cached(sym, *YEARS)
        if not arts:
            print(f"  {sym:6s} sin caché — corre events/news_backfill.py")
            continue
        n_articles += len(arts)
        bullish, control = catalyst_days(sym, arts)
        n_events += len(bullish)
        df = load_bars(sym, date(YEARS[0], 1, 1), date(YEARS[1], 1, 1), "1d", source="yfinance")
        if df is None or df.empty:
            print(f"  {sym:6s} sin precios"); continue

        for split, (y0, y1) in SPLITS.items():
            days = [d for d in bullish if y0 <= int(d[:4]) < y1]
            ctrl = [d for d in control if y0 <= int(d[:4]) < y1]
            for w in WINDOWS:
                rets = long_returns(df, sorted(days), w)
                pooled[split][w][sym].extend(r for _, _, r in rets)
                if rets:
                    pb = conditioned_placebo(df, ctrl, w, len(rets))
                    placebo_pool[split][w].extend(pb["exp_dist"])
        for w in WINDOWS:
            for day, sess, r in long_returns(df, sorted(bullish), w):
                by_year[w][sess[:4]].append(r)
                by_catalyst[w][bullish[day]].append(r)
        print(f"  {sym:6s} {len(arts):6d} titulares → {len(bullish):4d} días-catalizador "
              f"({len(control)} días-control)", flush=True)

    print(f"\nTotal: {n_articles} titulares, {n_events} días-evento alcistas\n")
    _report(pooled, placebo_pool, by_year, by_catalyst)
    return {"pooled": pooled, "n_events": n_events}


def _report(pooled, placebo_pool, by_year, by_catalyst) -> None:
    for split in SPLITS:
        print(f"── {split} " + "─" * 50)
        for w in WINDOWS:
            per_sym = {s: v for s, v in pooled[split][w].items() if v}
            allr = [r for v in per_sym.values() for r in v]
            if not allr:
                print(f"  w={w}d  sin muestra"); continue
            s = event_study(allr)
            g = gate_event(s)
            pct = percentile_of(s["expectancy_pct"], placebo_pool[split][w])
            jk = jackknife_by_group(per_sym, k=3)
            jke = jackknife_by_event(allr, k=3)
            print(f"  w={w}d  {'PASS ✅' if g['passed'] else 'FAIL ❌'}  "
                  f"exp={s['expectancy_pct']:+.3f}%  hit={s['hit_rate']:.0%}  "
                  f"tail={s['tail_ratio']}  n={s['n']}  ({g['reason']})")
            print(f"        placebo-condicionado: media {round(st.mean(placebo_pool[split][w]),3) if placebo_pool[split][w] else 0}%  "
                  f"→ percentil {pct}")
            print(f"        jackknife símbolos (sin {', '.join(jk['dropped'])}): "
                  f"{jk['jackknifed']['expectancy_pct']:+.3f}%  · "
                  f"{jk['groups_positive']}/{jk['groups']} símbolos positivos")
            print(f"        jackknife eventos (sin top-3): {jke['jackknifed']['expectancy_pct']:+.3f}%")
        print()

    print("── Leave-one-year-out (full sample) " + "─" * 20)
    for w in WINDOWS:
        if not by_year[w]:
            continue
        ly = leave_one_year_out(dict(by_year[w]))
        print(f"  w={w}d  full={ly['full_exp_pct']:+.3f}%  peor año fuera="
              f"{ly['worst_drop_year']} → {ly['worst_drop_exp_pct']:+.3f}%  "
              f"años positivos {ly['years_positive']}/{len(by_year[w])}")

    print("\n── Por catalizador (w=1d, el desplegado) " + "─" * 16)
    for kind, rets in sorted(by_catalyst[1].items(), key=lambda kv: -len(kv[1])):
        s = event_study(rets)
        print(f"  {kind:20s} n={s['n']:5d}  exp={s['expectancy_pct']:+.3f}%  "
              f"hit={s['hit_rate']:.0%}  tail={s['tail_ratio']}")


def _selfcheck() -> None:
    """La lógica no trivial: extracción de eventos y alineación de retornos."""
    arts = [
        {"created_at": "2020-05-01T12:00:00Z", "headline": "Acme beats estimates", "symbols": ["ACME"]},
        {"created_at": "2020-05-01T15:00:00Z", "headline": "Acme raises guidance", "symbols": ["ACME"]},
        {"created_at": "2020-05-04T12:00:00Z", "headline": "Acme names new CFO", "symbols": ["ACME"]},
        {"created_at": "2020-05-05T12:00:00Z", "headline": "Acme beats estimates", "symbols": ["ACME", "BCME"]},
        {"created_at": "2020-05-06T12:00:00Z", "headline": "Acme misses estimates", "symbols": ["ACME"]},
    ]
    bull, ctrl = catalyst_days("ACME", arts)
    assert list(bull) == ["2020-05-01"], f"dedup por día falló: {bull}"
    assert bull["2020-05-01"] == "earnings_beat", "gana el primer catalizador del día"
    assert ctrl == ["2020-05-04", "2020-05-05", "2020-05-06"], f"control mal: {ctrl}"

    import pandas as pd
    idx = pd.to_datetime(["2020-05-01", "2020-05-04", "2020-05-05", "2020-05-06"])
    df = pd.DataFrame({"close": [100.0, 110.0, 121.0, 121.0]}, index=idx)
    got = long_returns(df, ["2020-05-01"], 1)
    assert got and abs(got[0][2] - 0.10) < 1e-9, f"retorno w=1 mal: {got}"
    got2 = long_returns(df, ["2020-05-02"], 1)   # sábado → entra el lunes
    assert got2 and got2[0][1] == "2020-05-04" and abs(got2[0][2] - 0.10) < 1e-9, \
        f"alineación de fin de semana mal: {got2}"
    assert long_returns(df, ["2020-05-06"], 1) == [], "sin salida futura no debe emitir"
    print("selfcheck ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--selfcheck":
        _selfcheck(); raise SystemExit(0)
    run([s.strip().upper() for s in args[0].split(",")] if args else None)
