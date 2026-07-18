#!/usr/bin/env python3
"""
Event-study harness — Fase 1 del pivote event-driven (docs/PLAN-event-driven.md).

Prueba si un catalizador AGENDADO (FOMC, CPI, OPEC…) deja un edge tradeable en la
ventana post-evento. NO persigue velocidad-a-la-noticia (HFT gana esa carrera);
mide **drift/continuación** sobre horas-días, que un bot retail SÍ puede cabalgar.

Señal tradeable = dirección del movimiento del día del evento; el "drift return"
por evento = sign(mov_dia_evento) × retorno_forward(window). Si el drift es real,
la distribución de esos retornos tiene expectativa positiva y cola derecha gorda.

Gate del carril-event (long-shot, NO Sharpe≥0.8): expectativa>0 + ratio de cola +
muestra mínima. Para un long-shot de cola gorda no se puede exigir p-value (N chico):
se exige asimetría y pérdida acotada. Ver docs/PLAN-event-driven.md.

Uso:
    python events/event_study.py FOMC SPY,QQQ,GLD,USO
    python events/event_study.py --selfcheck
"""
from __future__ import annotations

import statistics as st
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

# Allow `python events/event_study.py` from repo root (mirrors run_backtest.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- Calendario de catalizadores (hand-seeded; VERIFICAR contra fuente oficial) ---
# FOMC = día del ANUNCIO (2º día de junta). Fuente: federalreserve.gov. Alta confianza.
# ponytail: dict estático, no API. Es finito y público; un loader vive solo si crece.
FOMC_ANNOUNCEMENT_DATES = [
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
]

CALENDARS: Dict[str, List[str]] = {
    "FOMC": FOMC_ANNOUNCEMENT_DATES,
    # TODO Fase 1: CPI (mensual), OPEC (juntas), earnings por-símbolo (yfinance).
}

# Gate del carril-event. Long-shot: asimetría, no Sharpe.
MIN_EVENTS = 15          # muestra mínima para no leer ruido
MIN_EXPECTANCY_PCT = 0.0 # retorno esperado por evento debe ser positivo
MIN_TAIL_RATIO = 1.2     # ganancia media / |pérdida media| — la asimetría del long-shot


def _forward_drift_returns(
    df, event_dates: List[str], window_days: int
) -> List[float]:
    """Por cada evento: sign(mov del día del evento) × retorno de `window_days` después.

    Entra al cierre del día del evento, sale `window_days` sesiones después.
    El signo hace la señal tradeable (continuación en la dirección del evento).
    """
    idx = list(df.index)
    close = df["close"]
    out: List[float] = []
    for ds in event_dates:
        ts = _nearest_index(idx, ds)
        if ts is None:
            continue
        i = idx.index(ts)
        if i == 0 or i + window_days >= len(idx):
            continue
        event_move = float(close.iloc[i] / close.iloc[i - 1] - 1)
        fwd = float(close.iloc[i + window_days] / close.iloc[i] - 1)
        if event_move == 0:
            continue
        sign = 1.0 if event_move > 0 else -1.0
        out.append(sign * fwd)
    return out


def _nearest_index(idx, ds: str):
    """Índice de sesión == fecha del evento, o la primera sesión posterior (feriados)."""
    target = _to_ts(ds)
    for ts in idx:
        if ts.date() >= target:
            # solo aceptamos si cae dentro de 4 días (evento en fin de semana/feriado)
            return ts if (ts.date() - target).days <= 4 else None
    return None


def _to_ts(ds: str):
    import pandas as pd
    return pd.Timestamp(ds).date()


def event_study(returns: List[float]) -> Dict[str, float]:
    """Estadísticos de la distribución de drift-returns. Puro; sin I/O."""
    n = len(returns)
    if n == 0:
        return {"n": 0, "expectancy_pct": 0.0, "median_pct": 0.0, "hit_rate": 0.0,
                "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "tail_ratio": 0.0, "max_loss_pct": 0.0}
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    avg_win = st.mean(wins) if wins else 0.0
    avg_loss = st.mean(losses) if losses else 0.0
    tail = (avg_win / abs(avg_loss)) if losses and avg_loss != 0 else (float("inf") if wins else 0.0)
    return {
        "n": n,
        "expectancy_pct": round(st.mean(returns) * 100, 3),
        "median_pct": round(st.median(returns) * 100, 3),
        "hit_rate": round(len(wins) / n, 3),
        "avg_win_pct": round(avg_win * 100, 3),
        "avg_loss_pct": round(avg_loss * 100, 3),
        "tail_ratio": round(tail, 2) if tail != float("inf") else 999.0,
        "max_loss_pct": round(min(returns) * 100, 3),
    }


def gate_event(stats: Dict[str, float]) -> Dict[str, object]:
    """Gate del carril-event: asimetría + muestra, NO Sharpe. Puro."""
    checks = {
        "n_events": (stats["n"], stats["n"] >= MIN_EVENTS),
        "expectancy_pct": (stats["expectancy_pct"], stats["expectancy_pct"] > MIN_EXPECTANCY_PCT),
        "tail_ratio": (stats["tail_ratio"], stats["tail_ratio"] >= MIN_TAIL_RATIO),
    }
    passed = all(ok for _, ok in checks.values())
    failed = [k for k, (_, ok) in checks.items() if not ok]
    return {"passed": passed, "reason": "clean" if passed else "failed: " + ", ".join(failed), "checks": checks}


def run(event: str, symbols: List[str], windows=(1, 3, 5), years=(2022, 2025)) -> None:
    from backtesting.engine import load_bars
    dates = CALENDARS.get(event)
    if not dates:
        print(f"Evento desconocido: {event}. Conocidos: {list(CALENDARS)}", file=sys.stderr)
        raise SystemExit(2)
    start, end = date(years[0], 1, 1), date(years[1], 1, 1)
    print(f"Event-study: {event} ({len(dates)} eventos) · {', '.join(symbols)} · ventanas {windows}d\n")
    for sym in symbols:
        df = load_bars(sym, start, end, "1d", source="yfinance")
        if df is None or df.empty:
            print(f"  {sym}: sin datos"); continue
        for w in windows:
            rets = _forward_drift_returns(df, dates, w)
            stats = event_study(rets)
            g = gate_event(stats)
            verdict = "PASS ✅" if g["passed"] else "FAIL ❌"
            print(f"  {sym:5s} w={w}d  {verdict}  exp={stats['expectancy_pct']:+.2f}%  "
                  f"hit={stats['hit_rate']:.0%}  tail={stats['tail_ratio']}  "
                  f"maxL={stats['max_loss_pct']:.1f}%  n={stats['n']}  ({g['reason']})")
        print()


def _selfcheck() -> None:
    # Drift claro y asimétrico → PASS.
    good = [0.02, 0.03, 0.015, -0.005, 0.025, 0.01, -0.008, 0.02, 0.03, 0.012,
            0.018, -0.006, 0.022, 0.014, 0.02, 0.011]
    s = event_study(good)
    assert s["n"] == 16 and s["expectancy_pct"] > 0
    assert gate_event(s)["passed"] is True, "drift asimétrico debe pasar"

    # Simétrico sin edge → FAIL (expectativa ~0).
    flat = [0.01, -0.01, 0.012, -0.012, 0.008, -0.008, 0.01, -0.01,
            0.009, -0.009, 0.011, -0.011, 0.007, -0.007, 0.01, -0.01]
    assert gate_event(event_study(flat))["passed"] is False, "simétrico debe fallar"

    # Muestra chica → FAIL aunque sea positivo.
    assert gate_event(event_study([0.05, 0.04, 0.03]))["passed"] is False, "N chico debe fallar"
    print("selfcheck ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selfcheck"]:
        _selfcheck()
    elif len(args) == 2:
        run(args[0], [s.strip().upper() for s in args[1].split(",")])
    else:
        print(__doc__)
        raise SystemExit(2)
