#!/usr/bin/env python3
"""
News event-study — ¿un PICO de volumen de noticias sobre un activo deja edge?

Hipótesis news-específica (no reducible a precio): cuando un símbolo tiene un
día de buzz anormal (volumen de titulares en el decil alto), ¿el precio hace
drift (continúa) o fade (revierte) en los días siguientes? Señal tradeable =
dirección del movimiento del día del pico. Reusa el gate/stats de event_study.

Fuente: Alpaca/Benzinga news (histórico, ya autenticado con las llaves del repo).

Uso:
    python events/news_study.py AAPL,TSLA,NVDA,AMZN,MSFT,META drift
    python events/news_study.py AAPL,TSLA,NVDA,AMZN,MSFT,META fade
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from events.event_study import _forward_drift_returns, event_study, gate_event

BUZZ_PERCENTILE = 0.90   # día-evento = volumen de noticias en el top 10% del símbolo
MAX_PAGES = 40           # tope de paginación por símbolo (evita runaway)


def news_counts_by_day(symbol: str, start: datetime, end: datetime) -> Counter:
    """Cuenta titulares por día para un símbolo, paginando la API de Alpaca."""
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    # raw_data=True → get_news returns the dict incl. next_page_token (NewsSet buries it).
    client = NewsClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), raw_data=True)
    counts: Counter = Counter()
    token = None
    for _ in range(MAX_PAGES):
        req = NewsRequest(symbols=symbol, start=start, end=end, limit=50, page_token=token)
        raw = client.get_news(req)
        arts = raw.get("news", []) if isinstance(raw, dict) else []
        for a in arts:
            ts = a.get("created_at") or a.get("updated_at")
            if ts:
                counts[str(ts)[:10]] += 1
        token = raw.get("next_page_token") if isinstance(raw, dict) else None
        if not token:
            break
    return counts


def buzz_event_dates(counts: Counter) -> List[str]:
    """Días cuyo volumen de noticias está en el decil alto del símbolo."""
    if len(counts) < 10:
        return []
    vals = sorted(counts.values())
    thr = vals[int(len(vals) * BUZZ_PERCENTILE)]
    return sorted(d for d, c in counts.items() if c >= max(thr, 2))


def run(symbols: List[str], mode: str = "drift", years=(2023, 2025)) -> None:
    from backtesting.engine import load_bars

    load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))
    start_dt = datetime(years[0], 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(years[1], 1, 1, tzinfo=timezone.utc)
    start_d, end_d = date(years[0], 1, 1), date(years[1], 1, 1)
    print(f"News-study [{mode}]: buzz-spike (top {int((1-BUZZ_PERCENTILE)*100)}%) · "
          f"{', '.join(symbols)} · {years[0]}-{years[1]-1}\n")
    for sym in symbols:
        counts = news_counts_by_day(sym, start_dt, end_dt)
        events = buzz_event_dates(counts)
        if len(events) < 15:
            print(f"  {sym:5s}  muestra chica: {len(events)} eventos de buzz (news days={len(counts)})")
            continue
        df = load_bars(sym, start_d, end_d, "1d", source="yfinance")
        if df is None or df.empty:
            print(f"  {sym:5s}  sin precios"); continue
        for w in (1, 3, 5):
            stats = event_study(_forward_drift_returns(df, events, w, mode))
            g = gate_event(stats)
            verdict = "PASS ✅" if g["passed"] else "FAIL ❌"
            print(f"  {sym:5s} w={w}d  {verdict}  exp={stats['expectancy_pct']:+.2f}%  "
                  f"hit={stats['hit_rate']:.0%}  tail={stats['tail_ratio']}  n={stats['n']}  ({g['reason']})")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) in (1, 2):
        syms = [s.strip().upper() for s in args[0].split(",")]
        run(syms, args[1] if len(args) == 2 else "drift")
    else:
        print(__doc__); raise SystemExit(2)
