#!/usr/bin/env python3
"""
News reactor — reacciona a catalizadores de noticias EN VIVO (paper).

Lo que hace: poll del feed Benzinga/Alpaca → clasifica el titular → si es un
catalizador alcista accionable, abre un LARGO en paper con salida por tiempo, y
registra la decisión en un journal para acumular evidencia hacia adelante.

Por qué así (decisiones y sus límites):
- **Long-only:** el ledger virtual no soporta cortos (ver VirtualPortfolio).
  Los catalizadores bajistas se registran en el journal como `skipped`, así el
  journal mide TAMBIÉN lo que no operamos (evidencia de si valdría el corto).
- **Reglas por keyword, no LLM:** determinista, auditable, cero costo y cero
  latencia. ponytail: si el journal muestra que la clasificación es el cuello de
  botella, ahí sí se mete un LLM — no antes.
- **Sin backtest previo:** el histórico de noticias del tier actual topa en 1
  página (medido 2026-07-21), así que esto se valida FORWARD, no hacia atrás.
  Por eso el journal es el entregable, no el P&L de la primera semana.
- **Salida por tiempo (1 día):** el drift post-noticia se disipa rápido; sin
  histórico no podemos optimizar la ventana, así que se fija y se mide.

Uso:
    python -m live.news_reactor --once       # un ciclo (dry-run si el mercado cerró)
    python -m live.news_reactor --loop       # ciclo continuo cada POLL_SECONDS
    python -m live.news_reactor --selfcheck
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
JOURNAL = REPO / "data" / "news_reactor" / "journal.jsonl"
SEEN_FILE = REPO / "data" / "news_reactor" / "seen_ids.json"

POLL_SECONDS = 60
MAX_CONCURRENT = 5          # posiciones abiertas simultáneas del reactor
DOLLARS_PER_TRADE = 1_000.0  # tamaño fijo por evento (paper)
HOLD_DAYS = 1               # el drift post-noticia se disipa rápido
LOOKBACK_MINUTES = 30       # ventana de titulares frescos en cada poll

# Catalizadores accionables. El primer match gana, por eso el orden importa:
# lo más específico y direccional arriba.
BULLISH_PATTERNS = [
    ("acquisition_target", r"\b(to be acquired|acquisition of|agrees to be acquired|takeover bid|buyout offer)\b"),
    ("fda_approval", r"\b(fda approval|fda approves|receives approval|granted approval)\b"),
    ("earnings_beat", r"\b(beats|tops|surpasses)\b.{0,30}\b(estimates|expectations|consensus|views)\b"),
    ("guidance_raise", r"\b(raises|boosts|lifts)\b.{0,25}\b(guidance|outlook|forecast)\b"),
    ("upgrade", r"\b(upgrades?|upgraded)\b.{0,30}\b(to buy|to outperform|to overweight)\b"),
]
BEARISH_PATTERNS = [
    ("earnings_miss", r"\b(misses|falls short of)\b.{0,30}\b(estimates|expectations|consensus)\b"),
    ("guidance_cut", r"\b(cuts|lowers|slashes)\b.{0,25}\b(guidance|outlook|forecast)\b"),
    ("downgrade", r"\b(downgrades?|downgraded)\b.{0,30}\b(to sell|to underperform|to underweight)\b"),
    ("investigation", r"\b(sec investigation|doj probe|fraud|delisting|bankruptcy)\b"),
]


@dataclass(frozen=True)
class Catalyst:
    kind: str
    direction: str  # "bullish" | "bearish"


@dataclass(frozen=True)
class Decision:
    ts: str
    news_id: str
    headline: str
    symbol: Optional[str]
    catalyst: Optional[str]
    direction: Optional[str]
    action: str      # "buy" | "skipped"
    reason: str
    dollars: float = 0.0


def classify(headline: str) -> Optional[Catalyst]:
    """Clasifica un titular en un catalizador direccional. Puro."""
    h = (headline or "").lower()
    for kind, pat in BULLISH_PATTERNS:
        if re.search(pat, h):
            return Catalyst(kind, "bullish")
    for kind, pat in BEARISH_PATTERNS:
        if re.search(pat, h):
            return Catalyst(kind, "bearish")
    return None


def pick_symbol(symbols: Sequence[str]) -> Optional[str]:
    """Un titular con muchos tickers es un roundup, no un catalizador de una empresa."""
    clean = [s for s in symbols if s.isalpha() and 1 <= len(s) <= 5]
    if len(clean) != 1:
        return None
    return clean[0]


def decide(
    *,
    news_id: str,
    headline: str,
    symbols: Sequence[str],
    ts: str,
    open_symbols: Sequence[str],
    n_open: int,
) -> Decision:
    """Regla completa de una noticia → decisión. Pura; sin I/O ni red."""
    base = dict(ts=ts, news_id=news_id, headline=headline[:180])
    cat = classify(headline)
    if cat is None:
        return Decision(**base, symbol=None, catalyst=None, direction=None,
                        action="skipped", reason="sin catalizador")
    sym = pick_symbol(symbols)
    if sym is None:
        return Decision(**base, symbol=None, catalyst=cat.kind, direction=cat.direction,
                        action="skipped", reason="0 o >1 ticker (roundup)")
    if cat.direction == "bearish":
        # Se registra igual: el journal mide qué habríamos ganado con cortos.
        return Decision(**base, symbol=sym, catalyst=cat.kind, direction=cat.direction,
                        action="skipped", reason="bajista y el ledger es solo-largo")
    if sym in open_symbols:
        return Decision(**base, symbol=sym, catalyst=cat.kind, direction=cat.direction,
                        action="skipped", reason="ya hay posición abierta")
    if n_open >= MAX_CONCURRENT:
        return Decision(**base, symbol=sym, catalyst=cat.kind, direction=cat.direction,
                        action="skipped", reason=f"tope de {MAX_CONCURRENT} concurrentes")
    return Decision(**base, symbol=sym, catalyst=cat.kind, direction=cat.direction,
                    action="buy", reason="catalizador alcista accionable",
                    dollars=DOLLARS_PER_TRADE)


# ----------------------------------------------------------------- I/O

def _load_seen() -> set:
    if SEEN_FILE.exists():
        try:
            return set(json.loads(SEEN_FILE.read_text()))
        except Exception:
            return set()
    return set()


def _save_seen(seen: set) -> None:
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Solo los últimos 5k ids: el archivo no debe crecer sin límite.
    SEEN_FILE.write_text(json.dumps(sorted(seen)[-5000:]))


def _journal(dec: Decision) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a") as fh:
        fh.write(json.dumps(asdict(dec), ensure_ascii=False) + "\n")


def fetch_recent(minutes: int = LOOKBACK_MINUTES) -> List[dict]:
    """Titulares de los últimos `minutes`. Devuelve dicts crudos."""
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    client = NewsClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"),
                        raw_data=True)
    req = NewsRequest(start=datetime.now(timezone.utc) - timedelta(minutes=minutes), limit=50)
    raw = client.get_news(req)
    return raw.get("news", []) if isinstance(raw, dict) else []


def run_once(executor=None, dry_run: bool = False) -> List[Decision]:
    """Un ciclo: lee titulares nuevos, decide, ejecuta y journaliza."""
    from dotenv import load_dotenv
    load_dotenv(str(REPO / ".env"))

    seen = _load_seen()
    # 1) Cerrar lo vencido ANTES de decidir: si no, el tope de concurrentes se
    #    llena el primer día y el reactor deja de operar (bug medido 2026-07-30).
    if executor is not None and not dry_run:
        closed = executor.close_due(HOLD_DAYS)
        if closed:
            logger.info("cerradas por tiempo: %s", ", ".join(closed))
    items = fetch_recent()
    open_symbols: List[str] = []
    if executor is not None:
        open_symbols = list(getattr(executor, "open_symbols", lambda: [])())

    decisions: List[Decision] = []
    for a in items:
        nid = str(a.get("id") or a.get("ID") or "")
        if not nid or nid in seen:
            continue
        seen.add(nid)
        dec = decide(
            news_id=nid,
            headline=a.get("headline", ""),
            symbols=a.get("symbols", []) or [],
            ts=str(a.get("created_at") or a.get("updated_at") or ""),
            open_symbols=open_symbols,
            n_open=len(open_symbols),
        )
        if dec.action == "buy" and dry_run:
            dec = Decision(**{**asdict(dec), "action": "skipped",
                              "reason": "dry-run (no se ejecutó)"})
        elif dec.action == "buy" and executor is not None:
            ok = executor.buy_dollars(symbol=dec.symbol, dollars=dec.dollars,
                                      hold_days=HOLD_DAYS, strategy="news_reactor")
            if ok:
                open_symbols.append(dec.symbol)
            else:
                dec = Decision(**{**asdict(dec), "action": "skipped",
                                  "reason": "ejecución rechazada (mercado cerrado o broker)"})
        _journal(dec)
        decisions.append(dec)
    _save_seen(seen)
    return decisions


def _selfcheck() -> None:
    assert classify("Acme Corp beats Q3 estimates").kind == "earnings_beat"
    assert classify("Acme raises full-year guidance").direction == "bullish"
    assert classify("Acme misses estimates").direction == "bearish"
    assert classify("Acme names new CFO") is None, "ruido no debe operar"

    kw = dict(ts="t", open_symbols=[], n_open=0)
    d = decide(news_id="1", headline="Acme beats estimates", symbols=["ACME"], **kw)
    assert d.action == "buy" and d.symbol == "ACME"

    d = decide(news_id="2", headline="Acme beats estimates",
               symbols=["ACME", "BCME", "CCME"], **kw)
    assert d.action == "skipped", "roundup multi-ticker no opera"

    d = decide(news_id="3", headline="Acme misses estimates", symbols=["ACME"], **kw)
    assert d.action == "skipped" and d.direction == "bearish"

    d = decide(news_id="4", headline="Acme beats estimates", symbols=["ACME"],
               ts="t", open_symbols=["ACME"], n_open=1)
    assert d.action == "skipped", "no duplica posición"

    d = decide(news_id="5", headline="Acme beats estimates", symbols=["ACME"],
               ts="t", open_symbols=["X"], n_open=MAX_CONCURRENT)
    assert d.action == "skipped", "respeta el tope de concurrentes"
    print("selfcheck ok")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck(); return 0

    executor = None
    if not args.dry_run:
        from live.reactor_executor import ReactorExecutor
        executor = ReactorExecutor()

    def cycle() -> None:
        decs = run_once(executor=executor, dry_run=args.dry_run)
        buys = [d for d in decs if d.action == "buy"]
        logger.info("ciclo: %d titulares nuevos, %d compras", len(decs), len(buys))
        for d in buys:
            logger.info("  BUY %s (%s) — %s", d.symbol, d.catalyst, d.headline[:70])

    if args.loop:
        while True:
            try:
                cycle()
            except Exception as e:  # el reactor no debe morirse por un poll fallido
                logger.warning("ciclo falló: %s: %s", type(e).__name__, e)
            time.sleep(POLL_SECONDS)
    else:
        cycle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
