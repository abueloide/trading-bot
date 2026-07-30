#!/usr/bin/env python3
"""
F4 — Cola gorda con PÉRDIDA ACOTADA (carril COLA GORDA, docs/RESEARCH-BACKLOG.md).

Tesis: F1 (pánico), F2 (VIX) y F3 (gap) murieron por el MISMO motivo — la cola
derecha existía, pero la izquierda pesaba igual o más porque la pérdida en spot
**no está acotada**. `tail_ratio` máximo de los tres estudios: 1.86, contra el
umbral ≥3 del carril. El backlog concluyó que la forma de payoff que pide la
tesis es una opción larga (riesgo definido) → cambio de alcance, decisión de Luis.

**Antes de escalar, falta probar lo barato:** un STOP DURO acota la pérdida por
construcción, sin opciones y sin datos nuevos. Si con stop alguna celda da
tail_ratio ≥3 CON expectativa neta positiva, el carril sigue vivo en spot. Si no,
el "spot no puede dar esta forma de payoff" deja de ser argumento y pasa a ser
resultado medido, y la escalada a opciones queda justificada con evidencia.

TRAMPA CENTRAL (por la que este estudio podría auto-engañarse):
un stop **infla mecánicamente** el tail_ratio — trunca las pérdidas y deja
intactas las ganancias. Un tail_ratio de 4 con stop de −3% no prueba nada por sí
solo. Por eso el placebo corre con **el mismo stop**: la pregunta no es "¿el stop
sube el tail?" (sí, siempre) sino "¿el evento con stop bate a un día random con
stop?". Y la expectativa neta manda: un stop que sube el tail y hunde la
expectativa es una pérdida disfrazada de asimetría.

Ejecución del stop sobre barras diarias (honesta, no optimista):
  - entrada al cierre del día del evento;
  - si en alguna sesión posterior el `open` ya abre bajo el stop → salida AL OPEN
    (gap-through: el stop no te salva del hueco);
  - si no, pero el `low` toca el stop → salida al precio del stop;
  - si nunca toca → salida al cierre de la sesión w.

Uso:
    python events/stop_study.py                 # barrido completo (panic+vix+gap)
    python events/stop_study.py panic           # una familia
    python events/stop_study.py --selfcheck
"""
from __future__ import annotations

import random
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from events.event_study import event_study, jackknife_by_group  # noqa: E402
from events.gap_study import GAP_UNIVERSE, dedup_overlap, gap_events, gap_pct  # noqa: E402
from events.panic_study import p90, panic_dates, percentile_of  # noqa: E402
from events.vix_study import VIX_SYMBOL, episodes, vix_spike_dates  # noqa: E402

INDEX_SYMBOLS: Tuple[str, ...] = ("SPY", "QQQ", "IWM")
WINDOWS: Tuple[int, ...] = (5, 10, 20)
STOPS: Tuple[float, ...] = (3.0, 5.0, 8.0)   # % bajo la entrada
REGIMES = {"OOS 2015-21": (2015, 2022), "IS 2022-24": (2022, 2025)}
PLACEBO_DRAWS = 200
PLACEBO_SEED = 20260730
MIN_TAIL_COLA_GORDA = 3.0    # umbral del carril: lo que este estudio busca batir


def stopped_return(df, i: int, window: int, stop_pct: float) -> float:
    """Retorno LARGO desde el cierre de `i` a w sesiones, con stop duro. Puro.

    `stop_pct` > 0 = distancia del stop bajo la entrada, en por ciento.
    Devuelve NaN-free float; el caller garantiza que i + window < len(df).
    """
    entry = float(df["close"].iloc[i])
    stop_price = entry * (1.0 - stop_pct / 100.0)
    o, lo, c = df["open"], df["low"], df["close"]
    for j in range(i + 1, i + window + 1):
        op = float(o.iloc[j])
        if op <= stop_price:            # abrió con hueco bajo el stop
            return op / entry - 1.0
        if float(lo.iloc[j]) <= stop_price:
            return -stop_pct / 100.0
    return float(c.iloc[i + window]) / entry - 1.0


def stopped_returns(df, event_ts: Sequence, window: int, stop_pct: float,
                    signs: Dict[object, float] | None = None) -> List[float]:
    """`stopped_return` por evento, saltando los que no caben en la serie. Puro.

    `signs` opcional: dirección por evento (gap-down → corto). El stop se aplica
    siempre CONTRA la posición, así que para el corto se invierte la serie.
    """
    idx = list(df.index)
    pos = {ts: i for i, ts in enumerate(idx)}
    out: List[float] = []
    for ts in event_ts:
        i = pos.get(ts)
        if i is None or i + window >= len(idx):
            continue
        sign = 1.0 if signs is None else signs.get(ts, 1.0)
        if sign > 0:
            out.append(stopped_return(df, i, window, stop_pct))
        else:
            out.append(stopped_return(_flipped(df), i, window, stop_pct))
    return out


_FLIP_CACHE: Dict[int, object] = {}


def _flipped(df):
    """Serie espejo (corto = largo sobre 2·entry − precio). Cacheada por id."""
    key = id(df)
    if key not in _FLIP_CACHE:
        base = float(df["close"].iloc[0]) * 2.0
        mirror = df.copy()
        mirror["open"] = base - df["open"]
        mirror["close"] = base - df["close"]
        mirror["high"] = base - df["low"]
        mirror["low"] = base - df["high"]
        _FLIP_CACHE[key] = mirror
    return _FLIP_CACHE[key]


def placebo_stopped(df, exclude: Sequence, window: int, stop_pct: float, n: int,
                    draws: int = PLACEBO_DRAWS, seed: int = PLACEBO_SEED) -> Dict[str, object]:
    """Mismo N de días random, MISMO stop. Neutraliza la inflación mecánica del tail."""
    idx = list(df.index)
    excl = set(exclude)
    pool = [ts for ts in idx[:-window] if ts not in excl]
    if n == 0 or len(pool) < n:
        return {"exp_dist": [], "tail_dist": []}
    rng = random.Random(seed)
    exps, tails = [], []
    for _ in range(draws):
        rets = stopped_returns(df, rng.sample(pool, n), window, stop_pct)
        if not rets:
            continue
        s = event_study(rets)
        exps.append(s["expectancy_pct"])
        tails.append(s["tail_ratio"])
    return {"exp_dist": exps, "tail_dist": tails}


def _pooled_placebo(bars: Dict[str, object], events_by_sym: Dict[str, List],
                    per_sym: Dict[str, List[float]], window: int, stop_pct: float,
                    draws: int = 60, seed: int = PLACEBO_SEED) -> Dict[str, object]:
    """Placebo POOLED: por sorteo, cada símbolo aporta tantos días random como
    eventos tuvo, y el tail se mide sobre el pool completo. Puro salvo el RNG.

    No sirve promediar placebos por-símbolo (muestras chicas → tails ruidosos):
    el real es pooled, el null tiene que serlo también.
    """
    rng = random.Random(seed)
    pools = {}
    for sym, evs in events_by_sym.items():
        idx = list(bars[sym].index)
        excl = set(evs)
        pools[sym] = [ts for ts in idx[:-window] if ts not in excl]
    exps, tails = [], []
    for _ in range(draws):
        sample: List[float] = []
        for sym, n in ((s, len(v)) for s, v in per_sym.items()):
            pool = pools.get(sym, [])
            if len(pool) < n:
                continue
            sample.extend(stopped_returns(bars[sym], rng.sample(pool, n), window, stop_pct))
        if not sample:
            continue
        s = event_study(sample)
        exps.append(s["expectancy_pct"])
        tails.append(s["tail_ratio"])
    return {"exp_dist": exps, "tail_dist": tails}


def _row(label: str, rets: List[float], pl: Dict[str, object]) -> Dict[str, float]:
    s = event_study(rets)
    pct_exp = percentile_of(s["expectancy_pct"], pl["exp_dist"])
    pct_tail = percentile_of(s["tail_ratio"], pl["tail_dist"])
    cola = "COLA ✅" if s["tail_ratio"] >= MIN_TAIL_COLA_GORDA else "cola-flaca"
    viva = "VIVA" if (s["tail_ratio"] >= MIN_TAIL_COLA_GORDA
                      and s["expectancy_pct"] > 0 and pct_tail >= 90) else "----"
    print(f"  {label:26s} {viva} {cola:10s} exp={s['expectancy_pct']:+.2f}% "
          f"hit={s['hit_rate']:.0%} tail={s['tail_ratio']:6.2f} p90={p90(rets):+.2f}% "
          f"maxL={s['max_loss_pct']:6.1f}% n={s['n']:4d} | placebo tail sup {pct_tail}% "
          f"exp sup {pct_exp}%")
    return {**s, "pct_tail": pct_tail, "pct_exp": pct_exp}


def _load(sym: str, years: Tuple[int, int]):
    from backtesting.engine import load_bars
    return load_bars(sym, date(years[0], 1, 1), date(years[1], 1, 1), "1d", source="yfinance")


def run_index_family(family: str) -> None:
    """panic (SPY ≤ −3%) o vix (^VIX +20%) sobre índices, largo con stop."""
    for rname, years in REGIMES.items():
        trig_sym = "SPY" if family == "panic" else VIX_SYMBOL
        trig = _load(trig_sym, years)
        if trig is None or trig.empty:
            print(f"{family} {rname}: sin datos de {trig_sym}"); continue
        raw = (panic_dates(trig, -3.0) if family == "panic"
               else vix_spike_dates(trig, 20.0))
        evs_all = episodes(raw, list(trig.index))
        print(f"\n### F4/{family} · {rname} · {len(raw)} días → {len(evs_all)} episodios")
        for sym in INDEX_SYMBOLS:
            df = _load(sym, years)
            if df is None or df.empty:
                print(f"  {sym}: sin datos"); continue
            own = set(df.index)
            evs = [ts for ts in evs_all if ts in own]
            for w in WINDOWS:
                for stop in STOPS:
                    rets = stopped_returns(df, evs, w, stop)
                    if len(rets) < 15:
                        continue
                    pl = placebo_stopped(df, evs, w, stop, len(rets))
                    _row(f"{sym} w={w}d stop={stop:.0f}%", rets, pl)


def run_gap_family() -> None:
    """gap UP ≥12% continuación — la celda que más cerca estuvo en F3, ahora con stop."""
    for rname, years in REGIMES.items():
        bars = {}
        for sym in GAP_UNIVERSE:
            df = _load(sym, years)
            if df is not None and not df.empty:
                bars[sym] = df
        print(f"\n### F4/gap-UP≥12% cont · {rname} · {len(bars)} símbolos")
        for w in WINDOWS:
            for stop in STOPS:
                per_sym: Dict[str, List[float]] = {}
                events_by_sym: Dict[str, List] = {}
                for sym, df in bars.items():
                    evs = dedup_overlap(df, gap_events(df, 12.0, "up"), w)
                    if not evs:
                        continue
                    rets = stopped_returns(df, evs, w, stop)
                    if rets:
                        per_sym[sym] = rets
                        events_by_sym[sym] = evs
                pooled = [r for v in per_sym.values() for r in v]
                if len(pooled) < 15:
                    continue
                pl = _pooled_placebo(bars, events_by_sym, per_sym, w, stop)
                _row(f"pooled w={w}d stop={stop:.0f}%", pooled, pl)
                jk = jackknife_by_group(per_sym, k=3)
                print(f"      jackknife −{jk['dropped']}: exp="
                      f"{jk['jackknifed']['expectancy_pct']:+.2f}% "
                      f"tail={jk['jackknifed']['tail_ratio']} "
                      f"({jk['groups_positive']}/{jk['groups']} símbolos +)")


def _selfcheck() -> None:
    import pandas as pd

    # Serie: entra a 100, el low del día 2 perfora el stop de 5% (93 < 95) sin
    # que el open lo haga (96 > 95), y luego rebota a 120.
    o = [100, 99, 96, 110, 120]
    h = [101, 100, 99, 112, 121]
    lo = [99, 96, 93, 105, 118]
    c = [100, 98, 96, 110, 120]
    idx = pd.bdate_range("2020-01-01", periods=5)
    df = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=idx)

    # sin stop (stop enorme) → retorno completo a w=4
    assert abs(stopped_return(df, 0, 4, 50.0) - 0.20) < 1e-9
    # stop 5%: el low del día 2 (93) perfora 95 → sale en el stop, no en el rebote
    assert abs(stopped_return(df, 0, 4, 5.0) - (-0.05)) < 1e-9
    # stop 1%: el día 1 abre a 99 = bajo 99.0 → gap-through, sale AL OPEN
    assert abs(stopped_return(df, 0, 4, 1.0) - (-0.01)) < 1e-9, stopped_return(df, 0, 4, 1.0)

    # gap-through real: open muy por debajo del stop → pierde MÁS que el stop
    gapdown = df.copy()
    gapdown.loc[idx[1], ["open", "high", "low", "close"]] = [80, 82, 78, 81]
    r = stopped_return(gapdown, 0, 4, 5.0)
    assert abs(r - (-0.20)) < 1e-9, f"el stop no salva del hueco: {r}"

    # el stop trunca la izquierda → tail_ratio sube MECÁNICAMENTE (la trampa)
    sin_stop = event_study([0.10, -0.30, 0.10, -0.30])
    con_stop = event_study([0.10, -0.05, 0.10, -0.05])
    assert con_stop["tail_ratio"] > sin_stop["tail_ratio"], "premisa del estudio"

    # eventos que no caben en la serie se descartan sin crash
    assert stopped_returns(df, [idx[-1]], 5, 5.0) == []
    assert len(stopped_returns(df, [idx[0]], 4, 5.0)) == 1

    # espejo (corto): una serie que cae 20% da +20% en la pata corta sin stop
    down = pd.DataFrame(
        {"open": [100, 95, 90, 85, 80], "high": [101, 96, 91, 86, 81],
         "low": [99, 94, 89, 84, 79], "close": [100, 95, 90, 85, 80]}, index=idx)
    short = stopped_returns(down, [idx[0]], 4, 50.0, signs={idx[0]: -1.0})
    assert abs(short[0] - 0.20) < 1e-9, short
    print("selfcheck ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selfcheck"]:
        _selfcheck()
    elif not args or args[0] in ("all", "panic", "vix", "gap"):
        which = args[0] if args else "all"
        if which in ("all", "panic"):
            run_index_family("panic")
        if which in ("all", "vix"):
            run_index_family("vix")
        if which in ("all", "gap"):
            run_gap_family()
    else:
        print(__doc__)
        raise SystemExit(2)
