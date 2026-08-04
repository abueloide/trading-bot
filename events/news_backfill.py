#!/usr/bin/env python3
"""
Backfill de titulares históricos (Alpaca/Benzinga) con caché local.

**Por qué existe:** `events/news_study.py` y el docstring de `live/news_reactor.py`
asumen que "el histórico topa en 1 página" (medido 2026-07-21) y de ahí sale la
decisión de desplegar el reactor SIN backtest. Eso es un **bug de paginación**, no
un tope de tier: la API de Alpaca devuelve `next_page_token = None` siempre, pero
el histórico sí está ahí — se camina moviendo `end` al timestamp del artículo más
viejo de la página. Medido 2026-08-03: 2016→hoy, con filtro por símbolo.

Caché: un jsonl por símbolo en `data/news_cache/`. Re-correr es gratis.

Uso:
    python events/news_backfill.py AAPL,MSFT 2016 2025
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "news_cache"
PAGE = 50                 # tope duro de la API por request
MAX_PAGES_PER_SYMBOL = 1500  # runaway guard; AAPL 2016-24 son ~1.1k páginas


def _client():
    from alpaca.data.historical.news import NewsClient
    return NewsClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"),
                      raw_data=True)


def walk_news(symbol: str, start: datetime, end: datetime) -> Iterator[dict]:
    """Todos los titulares de `symbol` en [start, end), del más nuevo al más viejo.

    La API ignora `page_token` (siempre devuelve None), así que la paginación se
    hace moviendo `end` al `created_at` del artículo más viejo de cada página.
    """
    from alpaca.data.requests import NewsRequest

    client = _client()
    cursor = end
    seen: set = set()
    for _ in range(MAX_PAGES_PER_SYMBOL):
        raw = client.get_news(NewsRequest(symbols=symbol, start=start, end=cursor, limit=PAGE))
        arts = raw.get("news", []) if isinstance(raw, dict) else []
        fresh = [a for a in arts if str(a.get("id")) not in seen]
        if not fresh:
            return
        for a in fresh:
            seen.add(str(a.get("id")))
            yield a
        oldest = fresh[-1].get("created_at")
        if not oldest:
            return
        nxt = datetime.fromisoformat(str(oldest).replace("Z", "+00:00"))
        if nxt >= cursor:  # no avanzamos → cortar en vez de girar en falso
            return
        cursor = nxt
        if cursor <= start:
            return


def backfill(symbol: str, y0: int, y1: int, refresh: bool = False) -> Path:
    """Baja y cachea los titulares de `symbol` para [y0, y1). Devuelve la ruta."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{symbol}_{y0}_{y1}.jsonl"
    if path.exists() and not refresh:
        return path
    start = datetime(y0, 1, 1, tzinfo=timezone.utc)
    end = datetime(y1, 1, 1, tzinfo=timezone.utc)
    n = 0
    tmp = path.with_suffix(".partial")
    with tmp.open("w") as fh:
        for a in walk_news(symbol, start, end):
            fh.write(json.dumps({
                "id": a.get("id"),
                "created_at": a.get("created_at"),
                "headline": a.get("headline", ""),
                "symbols": a.get("symbols", []) or [],
            }, ensure_ascii=False) + "\n")
            n += 1
    tmp.rename(path)
    print(f"  {symbol:6s} {n:6d} titulares → {path.name}", flush=True)
    return path


def load_cached(symbol: str, y0: int, y1: int) -> List[dict]:
    path = CACHE / f"{symbol}_{y0}_{y1}.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _selfcheck() -> None:
    """La paginación es la lógica no trivial. Cobertura real: tests/test_news_catalyst_study.py."""
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", "-q",
                        str(REPO / "tests" / "test_news_catalyst_study.py"),
                        "-k", "pagination"], cwd=REPO)
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--selfcheck":
        _selfcheck(); raise SystemExit(0)
    if len(args) != 3:
        print(__doc__); raise SystemExit(2)
    from dotenv import load_dotenv
    load_dotenv(str(REPO / ".env"))
    syms = [s.strip().upper() for s in args[0].split(",")]
    for s in syms:
        backfill(s, int(args[1]), int(args[2]))
