#!/usr/bin/env python3
"""Forced-daily sibling of ict_runner — QQQ, same daily bias, ONE trade EVERY day.

The horse in ict_runner.py only trades when the ICT setup confirms (SMT + iFVG),
so most sessions are flat. This sibling answers a different question Luis asked:
"force a trade every day and watch the money move." Same bias (daily_bias, the
20d trend the reel calls for), but no setup gate — it enters market at the open
in the bias direction and exits market at the close. Isolates exactly one
variable vs its sibling: forcing the trade vs waiting for the signal.

Own books in data/ict_daily/ so the two never clobber each other; shares Broker,
position_size and book_day with ict_runner (same mechanics, one logic change).

    python -m live.ict_daily_runner --loop     # run the session
    python -m live.ict_daily_runner --once     # one evaluation (no waiting)
    python -m live.ict_daily_runner --status   # print books as JSON
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time as time_mod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from live.ict_runner import (
    Broker, REPO, START_CASH, _sleep_to_next_bar, book_day, position_size,
)
from live.ict_signal import FLAT_TIME, RTH_END, RTH_START, daily_bias, rth

logger = logging.getLogger("ict_daily_runner")

STATE = REPO / "data" / "ict_daily" / "state.json"
JOURNAL = REPO / "data" / "ict_daily" / "journal.jsonl"
ET = ZoneInfo("America/New_York")

TRADED = "QQQ"
BAR_MINUTES = 5
ORDER_PREFIX = "ictd-"
# Stop/target still bound the paper risk the same way the sibling does, but here
# they only cap a trade the close would otherwise exit flat: 1% equity risk.
RISK_PCT = 0.01
STOP_FRAC = 0.005   # 0.5% protective stop from entry; close exits first most days


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"strategy": "ict_forced_daily", "starting_cash": START_CASH, "realized_pnl": 0.0,
            "inception": datetime.now(ET).date().isoformat(), "day": None, "entered": False,
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


def _bias_now(broker: Broker, now: datetime, today) -> int:
    """daily_bias from settled QQQ closes — the same 20d trend the sibling uses."""
    raw = broker.bars(now)
    q = rth(raw.xs(TRADED))
    if q.empty:
        return 0
    closes = q["close"].groupby(q.index.date).last()
    return daily_bias(closes, today)


def step(broker: Broker, state: dict, now: datetime) -> None:
    """One evaluation at `now` (just after a 5m boundary)."""
    now_et = now.astimezone(ET)
    today = now_et.date()
    if state["day"] != today.isoformat():
        state.update(day=today.isoformat(), entered=False, bias=0, booked=False)

    if now_et.time() >= FLAT_TIME:
        if not state.get("booked"):
            broker.flatten()
            time_mod.sleep(3)
            trade = book_day(state, broker.day_fills(today), today) if state.get("entered") else None
            state["booked"] = True
            _journal("eod", trade=trade, realized_pnl=state["realized_pnl"])
        return

    if state["entered"]:
        return  # one shot per day; the bracket/close handles the exit

    bias = _bias_now(broker, now, today)
    state["bias"] = bias
    if bias == 0:
        return  # not enough history for a direction yet; try next bar
    price = broker.last_price(TRADED)
    if price <= 0:
        return
    equity = state["starting_cash"] + state["realized_pnl"]
    risk = price * STOP_FRAC
    qty = position_size(equity, price, risk)
    if qty <= 0:
        return
    stop = price * (1 - STOP_FRAC) if bias > 0 else price * (1 + STOP_FRAC)
    client_id = f"{ORDER_PREFIX}{today:%Y%m%d}"
    order_id = broker.submit_market_stop(TRADED, bias, qty, round(stop, 2), client_id)
    state["entered"] = True
    _journal("entry", id=order_id, side=bias, qty=qty, approx_price=price, stop=round(stop, 2))


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
                logger.exception("step failed; retrying next bar")
        _sleep_to_next_bar()
    logger.info("session over")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
