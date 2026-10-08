#!/usr/bin/env python3
"""Reporte diario para Luis (apertura / cierre) — texto listo para Telegram.

Determinista: lee los libros (data/ledgers/state.json, data/ict/state.json) y
marca a precio actual del broker. El cron del bot solo retransmite la salida.

    python scripts/daily_report.py --open
    python scripts/daily_report.py --close
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from live.ict_signal import WINDOW_END, WINDOW_START, daily_bias  # noqa: E402

ET = ZoneInfo("America/New_York")
MX = ZoneInfo("America/Mexico_City")
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
STATE = REPO / "data/ledgers/state.json"
ICT_STATE = REPO / "data/ict/state.json"
RACE_INCEPTION = "2026-06-25"
NAMES = {
    "momentum_rotation": "Rotación por momentum",
    "confirmed_mr": "Reversión confirmada",
}
BIAS = {1: "alcista (solo compras)", -1: "bajista (solo ventas en corto)", 0: "sin dato"}


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _pct(x: float) -> str:
    return f"{x:+.2f}%"


def _fecha(d) -> str:
    return f"{d.day}-{MESES[d.month - 1]}"


def _daily_closes(data, symbol: str, start: datetime):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    df = data.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=[symbol], timeframe=TimeFrame.Day, start=start, feed="iex", adjustment="split",
    )).df.xs(symbol)
    closes = df["close"]
    closes.index = closes.index.tz_convert(ET).date
    return closes


def main() -> int:
    mode = "close" if "--close" in sys.argv else "open"
    load_dotenv(REPO / ".env")
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    from config import ALPACA_CONFIG

    trading = TradingClient(ALPACA_CONFIG["api_key"], ALPACA_CONFIG["secret_key"], paper=True)
    data = StockHistoricalDataClient(ALPACA_CONFIG["api_key"], ALPACA_CONFIG["secret_key"])
    now = datetime.now(ET)
    today = now.date()
    clock = trading.get_clock()
    if not clock.is_open and clock.next_open.astimezone(ET).date() != today and mode == "open":
        print(f"Hoy no abre el mercado. Próxima sesión: {_fecha(clock.next_open.astimezone(ET))}.")
        return 0

    positions = trading.get_all_positions()
    marks = {p.symbol: float(p.current_price) for p in positions}
    prev = {p.symbol: float(p.lastday_price) for p in positions}  # prior session close
    books = json.loads(STATE.read_text())
    spy = _daily_closes(data, "SPY", datetime.fromisoformat(RACE_INCEPTION).replace(tzinfo=ET))
    spy_pct = (float(spy.iloc[-1]) / float(spy.iloc[0]) - 1) * 100

    title = "Apertura" if mode == "open" else "Cierre"
    lines = [f"📊 {title} · {_fecha(today)} · cuenta de práctica", ""]
    total = 0.0
    for key, name in NAMES.items():
        b = books[key]
        unmarked = [s for s in b["lots"] if s not in marks]
        equity = b["cash"] + sum(l["qty"] * marks.get(s, l["avg_entry"]) for s, l in b["lots"].items())
        total += equity
        ret = (equity / b["starting_cash"] - 1) * 100
        line = f"• {name}: {_money(equity)} ({_pct(ret)} desde el 25-jun, {len(b['lots'])} posiciones)"
        if mode == "close":
            # vs the prior session's close, lot by lot (a name bought today counts
            # from yesterday's close, not its fill — small, and only on entry days).
            day_pnl = sum(l["qty"] * (marks[s] - prev[s]) for s, l in b["lots"].items() if s in marks)
            line += f" · hoy {_pct(day_pnl / (equity - day_pnl) * 100)}"
        if unmarked:
            line += f" ⚠️ sin precio: {', '.join(unmarked)}"
        lines.append(line)

    ict = json.loads(ICT_STATE.read_text()) if ICT_STATE.exists() else None
    if ict:
        equity = ict["starting_cash"] + ict["realized_pnl"]
        total += equity
        n = len(ict["trades"])
        wins = sum(1 for t in ict["trades"] if t["pnl"] > 0)
        lines.append(
            f"• Estrategia del reel (QQQ intradía): {_money(equity)} "
            f"({_pct((equity / ict['starting_cash'] - 1) * 100)} desde el {_fecha(datetime.fromisoformat(ict['inception']))}, "
            f"{n} operaciones, {wins} ganadas)"
        )
    lines += ["", f"S&P 500 desde el 25-jun: {_pct(spy_pct)}", f"Total de las tres: {_money(total)}"]

    if ict:
        lines.append("")
        if mode == "open":
            qqq = _daily_closes(data, "QQQ", datetime.now(ET) - timedelta(days=60))
            a, b = (datetime.combine(today, t, tzinfo=ET).astimezone(MX) for t in (WINDOW_START, WINDOW_END))
            lines.append(
                f"Reel hoy: sesgo {BIAS[daily_bias(qqq, today)]}. Busca entrada de "
                f"{a:%H:%M} a {b:%H:%M} hora de CDMX, máximo una operación."
            )
        else:
            order, trade = ict.get("order"), next((t for t in ict["trades"] if t["day"] == today.isoformat()), None)
            if ict.get("day") != today.isoformat() or not order:
                lines.append("Reel hoy: no hubo señal, no operó.")
            elif trade:
                side = "compra" if trade["side"] > 0 else "venta en corto"
                lines.append(
                    f"Reel hoy: {side} de {trade['qty']:.0f} QQQ a {trade['entry']:.2f}, "
                    f"salida a {trade['exit']:.2f}. Resultado: {trade['pnl']:+,.0f} dólares."
                )
            elif order["status"] == "expired":
                lines.append("Reel hoy: hubo señal, pero el precio no regresó a la entrada y la orden se canceló.")
            else:
                lines.append(f"Reel hoy: orden {order['status']} sin cierre registrado todavía. Revisar data/ict/journal.jsonl.")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
