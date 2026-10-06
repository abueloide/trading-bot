#!/usr/bin/env python3
"""Live (paper) runner for the ICT intraday strategy — QQQ, SMT pair SPY.

One process per session (LaunchAgent com.luisfer.ictrunner). Every 5 minutes it
pulls settled 5m bars (Alpaca IEX, real time), asks live/ict_signal.find_setup,
and on a setup rests ONE bracket limit order (entry + stop + target). The broker
manages the exit; this process only cancels a stale entry and forces flat at
15:55 ET.

QQQ belongs to this strategy alone: the daily horses trade S&P 500 constituents
and SPY is the benchmark that is never traded. That exclusivity is what makes
close_position("QQQ") safe here (the horses must never use it — see
live_executor_adapter.py).

Own books in data/ict/ (NOT data/ledgers/state.json: the 13:00 horse-race job
rewrites that file and the two processes would clobber each other).

    python -m live.ict_runner --loop     # run the session
    python -m live.ict_runner --once     # one evaluation (no waiting)
    python -m live.ict_runner --status   # print books as JSON
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time as time_mod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from live.ict_signal import (
    ENTRY_TTL_BARS, FLAT_TIME, RTH_END, RTH_START, Setup, daily_bias, find_setup, rth,
)

logger = logging.getLogger("ict_runner")

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "data" / "ict" / "state.json"
JOURNAL = REPO / "data" / "ict" / "journal.jsonl"
ET = ZoneInfo("America/New_York")

TRADED, PAIR = "QQQ", "SPY"
START_CASH = 25_000.0
RISK_PCT = 0.01
MAX_LEVERAGE = 4.0
BAR_MINUTES = 5
HISTORY_DAYS = 45      # calendar days of 5m bars: >20 sessions for the daily bias
SETTLE_SECONDS = 20    # wait after the bar boundary so the bar is final
CONTEXT_BARS = 160
ORDER_PREFIX = "ict-"


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"strategy": "ict_smt_ifvg", "starting_cash": START_CASH, "realized_pnl": 0.0,
            "inception": datetime.now(ET).date().isoformat(), "day": None, "order": None,
            "bias": 0, "trades": []}


def _save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(tmp, STATE)


def _journal(event: str, **fields) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a") as f:
        f.write(json.dumps({"ts": datetime.now(ET).isoformat(), "event": event, **fields}, default=str) + "\n")
    logger.info("%s %s", event, fields)


class Broker:
    """Thin Alpaca wrapper: only what this strategy needs."""

    def __init__(self) -> None:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient
        from config import ALPACA_CONFIG

        if not os.getenv("ALPACA_BASE_URL", "").startswith("https://paper-api.alpaca.markets"):
            raise SystemExit("Refusing to run: ALPACA_BASE_URL is not a paper endpoint.")
        key, secret = ALPACA_CONFIG["api_key"], ALPACA_CONFIG["secret_key"]
        self.trading = TradingClient(key, secret, paper=True)
        self.data = StockHistoricalDataClient(key, secret)

    def bars(self, now: datetime) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        req = StockBarsRequest(
            symbol_or_symbols=[TRADED, PAIR], timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
            start=now - timedelta(days=HISTORY_DAYS), feed="iex", adjustment="split",
        )
        return self.data.get_stock_bars(req).df

    def submit_bracket(self, setup: Setup, qty: int, client_id: str) -> str:
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, StopLossRequest, TakeProfitRequest

        order = self.trading.submit_order(LimitOrderRequest(
            symbol=TRADED, qty=qty,
            side=OrderSide.BUY if setup.side > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.DAY, limit_price=round(setup.entry, 2),
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(setup.target, 2)),
            stop_loss=StopLossRequest(stop_price=round(setup.stop, 2)),
            client_order_id=client_id,
        ))
        return str(order.id)

    def order(self, order_id: str):
        return self.trading.get_order_by_id(order_id)

    def cancel(self, order_id: str) -> None:
        try:
            self.trading.cancel_order_by_id(order_id)
        except Exception as e:  # already filled/cancelled is fine; anything else is logged
            logger.warning("cancel %s: %s", order_id, e)

    def position_qty(self) -> float:
        try:
            return float(self.trading.get_open_position(TRADED).qty)
        except Exception:
            return 0.0

    def flatten(self) -> None:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        for o in self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[TRADED])):
            self.cancel(str(o.id))
        if self.position_qty() != 0:
            self.trading.close_position(TRADED)

    def day_fills(self, day) -> list:
        """Filled QQQ orders (incl. bracket legs) submitted today, oldest first."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        after = datetime.combine(day, RTH_START, tzinfo=ET) - timedelta(hours=1)
        orders = self.trading.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.CLOSED, symbols=[TRADED], after=after, nested=False, limit=100))
        fills = [o for o in orders if o.filled_qty and float(o.filled_qty) > 0]
        return sorted(fills, key=lambda o: o.filled_at)


def settled_frames(raw: pd.DataFrame, now: datetime):
    """Aligned RTH frames of both symbols, without the in-progress bar."""
    q, s = rth(raw.xs(TRADED)), rth(raw.xs(PAIR))
    cutoff = now.astimezone(ET) - timedelta(minutes=BAR_MINUTES)
    idx = q.index.intersection(s.index)
    idx = idx[idx <= cutoff]
    return q.loc[idx], s.loc[idx]


def position_size(equity: float, entry: float, risk: float) -> int:
    return int(math.floor(min(equity * RISK_PCT / risk, equity * MAX_LEVERAGE / entry)))


def book_day(state: dict, fills: list, day) -> Optional[dict]:
    """Realized P&L of today's round trip from broker fills (flat at EOD)."""
    if not fills:
        return None
    signed = [(float(o.filled_qty) * (1 if o.side.value == "buy" else -1), float(o.filled_avg_price)) for o in fills]
    net_qty = sum(q for q, _ in signed)
    if abs(net_qty) > 1e-6:
        logger.warning("day %s not flat at booking (net %s) — P&L left unbooked", day, net_qty)
        return None
    pnl = -sum(q * p for q, p in signed)
    first_qty, entry_px = signed[0]
    trade = {"day": day.isoformat(), "side": 1 if first_qty > 0 else -1, "qty": abs(first_qty),
             "entry": entry_px, "exit": signed[-1][1], "pnl": round(pnl, 2)}
    state["realized_pnl"] = round(state["realized_pnl"] + pnl, 2)
    state["trades"].append(trade)
    return trade


def step(broker: Broker, state: dict, now: datetime) -> None:
    """One evaluation at `now` (just after a 5m boundary)."""
    now_et = now.astimezone(ET)
    today = now_et.date()
    if state["day"] != today.isoformat():
        state.update(day=today.isoformat(), order=None, bias=0, booked=False)

    order = state["order"]
    if now_et.time() >= FLAT_TIME:
        if not state.get("booked"):
            broker.flatten()
            time_mod.sleep(3)
            trade = book_day(state, broker.day_fills(today), today) if order else None
            state["booked"] = True
            _journal("eod", trade=trade, realized_pnl=state["realized_pnl"])
        return

    if order:  # one shot per day: only babysit the resting entry
        if order["status"] == "resting":
            o = broker.order(order["id"])
            if float(o.filled_qty or 0) > 0:
                order["status"] = "filled"
                _journal("entry_filled", price=float(o.filled_avg_price), qty=float(o.filled_qty))
            elif now_et >= datetime.fromisoformat(order["expiry"]):
                broker.cancel(order["id"])
                order["status"] = "expired"
                _journal("entry_expired")
        return

    q, s = settled_frames(broker.bars(now), now)
    if q.empty or q.index[-1].date() != today:
        return
    closes = q["close"].groupby(q.index.date).last()
    bias = daily_bias(closes, today)
    state["bias"] = bias
    setup = find_setup(q.iloc[-CONTEXT_BARS:], s.iloc[-CONTEXT_BARS:], bias)
    if setup is None:
        return
    equity = state["starting_cash"] + state["realized_pnl"]
    qty = position_size(equity, setup.entry, setup.risk)
    if qty <= 0:
        return
    order_id = broker.submit_bracket(setup, qty, f"{ORDER_PREFIX}{today:%Y%m%d}")
    expiry = setup.signal_time + timedelta(minutes=BAR_MINUTES * (ENTRY_TTL_BARS + 1))
    state["order"] = {"id": order_id, "status": "resting", "side": setup.side, "qty": qty,
                      "entry": setup.entry, "stop": setup.stop, "target": setup.target,
                      "expiry": expiry.isoformat()}
    _journal("setup", **state["order"])


def _sleep_to_next_bar() -> None:
    now = datetime.now(timezone.utc)
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=BAR_MINUTES - now.minute % BAR_MINUTES)
    time_mod.sleep(max(1.0, (nxt - now).total_seconds() + SETTLE_SECONDS))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    load_dotenv(REPO / ".env")
    state = _load_state()
    if args.status:
        print(json.dumps(state, indent=2, default=str))
        return 0
    broker = Broker()
    if not args.loop:
        step(broker, state, datetime.now(timezone.utc))
        _save_state(state)
        return 0
    clock = broker.trading.get_clock()
    if not clock.is_open and clock.next_open.astimezone(ET).date() != datetime.now(ET).date():
        logger.info("no session today (next open %s) — exiting", clock.next_open)
        return 0
    while True:
        now_et = datetime.now(ET)
        if now_et.time() >= RTH_END:
            break
        if now_et.time() >= RTH_START:
            try:
                step(broker, state, datetime.now(timezone.utc))
                _save_state(state)
            except Exception:
                # A data/API hiccup must not kill the session: the resting bracket
                # still needs its expiry check and the 15:55 flatten.
                logger.exception("step failed; retrying next bar")
        _sleep_to_next_bar()
    logger.info("session over")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
