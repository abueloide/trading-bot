#!/usr/bin/env python3
"""
F3 — Continuación de gap extremo (carril COLA GORDA, docs/RESEARCH-BACKLOG.md).

Tesis: un gap overnight >|X|% en un single-name es un repricing por catalizador
duro (earnings, guidance, M&A, FDA). Si el mercado *sub-reacciona*, el movimiento
CONTINÚA en las sesiones siguientes con cola derecha gorda.

**Desbloqueo del BLOCKED-DATA de F3.** F3 estaba parada por no tener calendario de
earnings verificado (feed de pago). Mismo truco que desbloqueó E4: el evento no se
*fetchea*, se *computa*. El gap ES la huella observable del catalizador — `open_t /
close_{t-1} - 1` sale del OHLC que ya tenemos. No identifica la CAUSA (earnings vs
M&A vs guidance) pero sí la MISMA población de eventos, y la regla tradeable solo
necesita la huella: cuando ves el gap al open, ya sabes que hubo catalizador.

Diferencia con `panic_study.py` (F1): allá el evento era de MERCADO (SPY ≤ −3%) y
por eso los días se agrupaban en ~6 episodios (COVID, bear-2022). Aquí el evento es
**idiosincrático por símbolo** → mucho menos clustering. Aun así se aplican las dos
lecciones de F1/F2: dedup de ventanas solapadas y reporte de concentración temporal.

Entrada: cierre del día del gap (el gap ya ocurrió al open → ejecutable, sin
lookahead). Salida: cierre a w sesiones.

Uso:
    python events/gap_study.py                      # barrido completo
    python events/gap_study.py AAPL,NVDA,AMD 8      # subset + umbral
    python events/gap_study.py --selfcheck
"""
from __future__ import annotations

import random
import statistics as st
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from events.event_study import event_study, gate_event, jackknife_by_group  # noqa: E402
from events.panic_study import p90, percentile_of  # noqa: E402

# Universo: single-names líquidos listados ANTES de 2015 (para que el OOS 2015-21
# tenga muestra) y con frecuencia real de gaps grandes. Mezcla mega-cap (gaps raros)
# + alta-beta/cíclicas (gaps frecuentes).
# ponytail: tupla estática. Screener dinámico solo si el universo empieza a importar.
# SESGO DE SUPERVIVENCIA CONOCIDO: son tickers vivos en 2026. Los gaps de las que
# quebraron/se deslistaron (peor cola izquierda) NO están. Sesga a favor de la tesis
# → si aun así falla, falla de verdad.
GAP_UNIVERSE: Tuple[str, ...] = (
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "NFLX",
    "AMD", "INTC", "MU", "QCOM", "CRM", "ADBE", "BABA", "BIDU",
    "YELP", "FSLR", "WDC", "STX", "LRCX", "AMAT", "EBAY", "SBUX",
    "NKE", "DIS", "BA", "GM", "F", "CLF", "UAL", "WYNN",
)

WINDOWS: Tuple[int, ...] = (1, 5, 10, 20)
THRESHOLDS: Tuple[float, ...] = (8.0, 12.0)
REGIMES = {"OOS 2015-21": (2015, 2022), "IS 2022-24": (2022, 2025)}
PLACEBO_DRAWS = 300
PLACEBO_SEED = 20260729
MIN_TAIL_COLA_GORDA = 3.0    # umbral del carril (informativo; gate_event exige 1.2)


def gap_pct(df) -> "object":
    """Serie de gaps overnight: open_t / close_{t-1} − 1. Puro."""
    return df["open"] / df["close"].shift(1) - 1.0


def gap_events(df, threshold_pct: float, direction: str) -> List:
    """Timestamps con gap ≥ +thr (up) o ≤ −thr (down). Puro."""
    g = gap_pct(df)
    thr = threshold_pct / 100.0
    if direction == "up":
        return [ts for ts, v in g.items() if v == v and v >= thr]
    return [ts for ts, v in g.items() if v == v and v <= -thr]


def dedup_overlap(df, events: Sequence, window: int) -> List:
    """Descarta eventos cuya ventana solapa la de un evento ya aceptado. Puro.

    Lección de F1: contar días solapados infla N sin añadir información
    independiente. Aquí es por-símbolo (el evento es idiosincrático).
    """
    pos = {ts: i for i, ts in enumerate(df.index)}
    kept: List = []
    last = -10**9
    for ts in sorted(events):
        i = pos.get(ts)
        if i is None:
            continue
        if i - last >= window:
            kept.append(ts)
            last = i
    return kept


def continuation_returns(df, events: Sequence, window: int, mode: str = "cont") -> List[float]:
    """sign(gap) × retorno forward de `window` sesiones desde el cierre del gap. Puro.

    mode='cont' compra la continuación; mode='rev' compra la reversión (espejo).
    """
    idx = list(df.index)
    pos = {ts: i for i, ts in enumerate(idx)}
    close, g = df["close"], gap_pct(df)
    sign_mult = 1.0 if mode == "cont" else -1.0
    out: List[float] = []
    for ts in events:
        i = pos.get(ts)
        if i is None or i + window >= len(idx):
            continue
        gv = g.iloc[i]
        if gv != gv or gv == 0:
            continue
        direction = 1.0 if gv > 0 else -1.0
        fwd = float(close.iloc[i + window] / close.iloc[i] - 1)
        out.append(sign_mult * direction * fwd)
    return out


def month_concentration(events: Sequence) -> float:
    """Fracción de eventos que cae en el mes-calendario más cargado. Puro.

    Lección de F1: N crudo miente si los eventos se apelotonan en una crisis.
    """
    if not events:
        return 0.0
    c = Counter((ts.year, ts.month) for ts in events)
    return round(max(c.values()) / len(events), 3)


def placebo_pool(df, events: Sequence, threshold_pct: float, window: int) -> List:
    """Días SIN gap extremo (el null: momentum genérico de gap chico). Puro."""
    g = gap_pct(df)
    thr = threshold_pct / 100.0
    excl = set(events)
    return [ts for ts, v in g.items()
            if ts not in excl and v == v and v != 0 and abs(v) < thr]


def run(symbols: Sequence[str] = GAP_UNIVERSE,
        thresholds: Sequence[float] = THRESHOLDS,
        windows: Sequence[int] = WINDOWS) -> None:
    from backtesting.engine import load_bars

    bars: Dict[str, object] = {}
    for sym in symbols:
        df = load_bars(sym, date(2015, 1, 1), date(2025, 1, 1), "1d", source="yfinance")
        if df is not None and not df.empty:
            bars[sym] = df
    print(f"F3 gap-continuation · {len(bars)}/{len(symbols)} símbolos con datos\n")
    if not bars:
        print("sin datos", file=sys.stderr)
        raise SystemExit(2)

    rng = random.Random(PLACEBO_SEED)
    for thr in thresholds:
        for direction in ("up", "down"):
            for mode in ("cont", "rev"):
                print(f"=== gap {direction.upper()} ≥{thr}%  ·  modo {mode} ===")
                for label, (y0, y1) in REGIMES.items():
                    for w in windows:
                        pooled: List[float] = []
                        pooled_evs: List = []
                        per_sym_rets: Dict[str, List[float]] = {}
                        breadth = 0
                        placebo_dist: List[List[float]] = []
                        for sym, full in bars.items():
                            df = full[(full.index >= f"{y0}-01-01") & (full.index < f"{y1}-01-01")]
                            if len(df) < 60:
                                continue
                            evs = dedup_overlap(df, gap_events(df, thr, direction), w)
                            rets = continuation_returns(df, evs, w, mode)
                            if not rets:
                                continue
                            pooled.extend(rets)
                            pooled_evs.extend(evs)
                            per_sym_rets[sym] = rets
                            if st.mean(rets) > 0:
                                breadth += 1
                            pool = placebo_pool(df, evs, thr, w)
                            if len(pool) >= len(rets):
                                placebo_dist.append(
                                    [continuation_returns(df, rng.sample(pool, len(rets)), w, mode)
                                     for _ in range(PLACEBO_DRAWS // 10)])
                        if not pooled:
                            continue
                        s = event_study(pooled)
                        g = gate_event(s)
                        # placebo pooled: draw d-ésimo de cada símbolo → distribución conjunta
                        exp_dist, tail_dist = [], []
                        for d in range(PLACEBO_DRAWS // 10):
                            merged = [r for per_sym in placebo_dist for r in per_sym[d]]
                            if merged:
                                ps = event_study(merged)
                                exp_dist.append(ps["expectancy_pct"])
                                tail_dist.append(ps["tail_ratio"])
                        verdict = "PASS ✅" if g["passed"] else "FAIL ❌"
                        cola = "COLA ✅" if s["tail_ratio"] >= MIN_TAIL_COLA_GORDA else "cola-flaca"
                        print(f"  {label:12s} w={w:2d}d  {verdict} {cola}  "
                              f"exp={s['expectancy_pct']:+.2f}%  hit={s['hit_rate']:.0%}  "
                              f"tail={s['tail_ratio']}  p90={p90(pooled):+.2f}%  "
                              f"maxL={s['max_loss_pct']:.1f}%  n={s['n']}  "
                              f"breadth={breadth}/{len(bars)}  conc={month_concentration(pooled_evs)}")
                        if exp_dist:
                            print(f"                placebo: exp={st.mean(exp_dist):+.2f}% "
                                  f"tail={round(st.mean(tail_dist), 2)}  →  real supera "
                                  f"{percentile_of(s['expectancy_pct'], exp_dist)}% (exp) / "
                                  f"{percentile_of(s['tail_ratio'], tail_dist)}% (tail)")
                        if g["passed"]:
                            # Solo donde importa: ¿el PASS sobrevive quitar los 3 nombres
                            # que más aportan? (killer test de estudios pooled — F3)
                            jk = jackknife_by_group(per_sym_rets, k=3)
                            print(f"                jackknife sin {'/'.join(jk['dropped'])}: "
                                  f"exp={jk['jackknifed']['expectancy_pct']:+.2f}% "
                                  f"tail={jk['jackknifed']['tail_ratio']}  "
                                  f"({jk['groups_positive']}/{jk['groups']} símbolos con exp>0)")
                print()


def _selfcheck() -> None:
    import pandas as pd
    # 10 días planos, luego gap up +10% al open que continúa, luego plano.
    idx = pd.bdate_range("2020-01-01", periods=16)
    close = [100.0] * 10 + [110.0, 115.0, 118.0] + [118.0] * 3
    open_ = [100.0] * 10 + [110.0, 110.0, 115.0] + [118.0] * 3
    df = pd.DataFrame({"open": open_, "close": close}, index=idx)

    ups = gap_events(df, 8.0, "up")
    assert ups == [idx[10]], f"un solo gap-up, no {ups}"
    assert gap_events(df, 12.0, "up") == [], "umbral más duro no dispara"
    assert gap_events(df, 8.0, "down") == [], "no hay gaps a la baja"

    # continuación: largo desde el cierre del día del gap
    r = continuation_returns(df, ups, 1, "cont")
    assert len(r) == 1 and abs(r[0] - (115.0 / 110.0 - 1)) < 1e-9, r
    assert abs(continuation_returns(df, ups, 1, "rev")[0] + r[0]) < 1e-12, "rev es el espejo"
    # ventana fuera de rango → evento descartado, no crash
    assert continuation_returns(df, [idx[-1]], 5, "cont") == []

    # gap DOWN: el signo se invierte (continuación = corto)
    df2 = pd.DataFrame({"open": [100.0, 90.0, 85.0], "close": [100.0, 88.0, 85.0]},
                       index=pd.bdate_range("2020-01-01", periods=3))
    downs = gap_events(df2, 8.0, "down")
    assert downs == [df2.index[1]], downs
    rd = continuation_returns(df2, downs, 1, "cont")
    assert rd and rd[0] > 0, "corto que sigue cayendo debe ser ganancia"

    # dedup: dos gaps a 2 sesiones con w=5 → solo sobrevive el primero
    idx3 = pd.bdate_range("2020-01-01", periods=8)
    df3 = pd.DataFrame({"open": [100, 120, 120, 145, 145, 145, 145, 145],
                        "close": [100, 120, 120, 145, 145, 145, 145, 145]},
                       dtype=float, index=idx3)
    ev3 = gap_events(df3, 8.0, "up")
    assert len(ev3) == 2, ev3
    assert dedup_overlap(df3, ev3, 5) == [ev3[0]], "ventanas solapadas se colapsan"
    assert dedup_overlap(df3, ev3, 1) == ev3, "sin solape se conservan ambos"

    assert month_concentration([]) == 0.0
    assert month_concentration(list(idx[:5])) == 1.0, "todos en el mismo mes"

    # placebo pool = días con gap chico NO nulo; el evento y los gaps 0 quedan fuera
    df4 = pd.DataFrame({"open": [100.0, 101.0, 102.0, 115.0], "close": [100.0, 100.0, 102.0, 115.0]},
                       index=pd.bdate_range("2020-01-01", periods=4))
    ev4 = gap_events(df4, 8.0, "up")
    pool = placebo_pool(df4, ev4, 8.0, 1)
    assert ev4 == [df4.index[3]], ev4
    assert pool == [df4.index[1], df4.index[2]], pool  # evento fuera; día 1 es NaN
    print("selfcheck ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selfcheck"]:
        _selfcheck()
    elif not args:
        run()
    elif len(args) in (1, 2):
        syms = [s.strip().upper() for s in args[0].split(",")]
        thrs = (float(args[1]),) if len(args) == 2 else THRESHOLDS
        run(syms, thrs)
    else:
        print(__doc__)
        raise SystemExit(2)
