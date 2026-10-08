#!/usr/bin/env python3
"""Backtest of live/ict_signal.py on cached 5m bars (QQQ traded, SPY = SMT pair).

Bar-by-bar, calling the same find_setup() the live runner uses. Conservative
fills: a bar that touches both stop and target counts as the STOP; the entry bar
can also stop out; stops slip; forced flat at 15:55 ET. One trade per day.

    python backtesting/ict_backtest.py [--bars data/ict/bars_sip.pkl]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live.ict_signal import ENTRY_TTL_BARS, FLAT_TIME, daily_bias, find_setup, rth  # noqa: E402

START_CASH = 25_000.0
RISK_PCT = 0.01       # of equity per trade
MAX_LEVERAGE = 4.0    # intraday notional cap (Alpaca day-trading buying power)
STOP_SLIP = 0.02      # $/share, stops only (entry and target are limits)
CONTEXT_BARS = 160    # ~2 sessions of 5m bars handed to find_setup


def position_size(equity: float, entry: float, risk: float) -> int:
    return int(math.floor(min(equity * RISK_PCT / risk, equity * MAX_LEVERAGE / entry)))


def load(path: str, traded: str = "QQQ", pair: str = "SPY"):
    raw = pd.read_pickle(path)
    q, s = rth(raw.xs(traded)), rth(raw.xs(pair))
    idx = q.index.intersection(s.index)
    return q.loc[idx], s.loc[idx]


def run(q: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    closes = q["close"].groupby(q.index.date).last()
    equity = START_CASH
    trades = []
    days = q.index.date
    for day in sorted(set(days)):
        bias = daily_bias(closes, day)
        if bias == 0:
            continue
        mask = days == day
        first = int(mask.argmax())
        last = first + int(mask.sum()) - 1
        pending = position = None
        for t in range(first, last + 1):
            bar = q.iloc[t]
            if position is None and pending is None:
                lo = max(0, t + 1 - CONTEXT_BARS)
                setup = find_setup(q.iloc[lo:t + 1], s.iloc[lo:t + 1], bias)
                if setup is not None:
                    pending = (setup, t + ENTRY_TTL_BARS)
                continue
            if pending is not None:
                setup, expiry = pending
                tapped = bar["low"] <= setup.entry if setup.side > 0 else bar["high"] >= setup.entry
                if tapped:
                    qty = position_size(equity, setup.entry, setup.risk)
                    position, pending = (setup, qty), None
                    if qty == 0:
                        position = None
                        break
                elif t >= expiry:
                    break  # one shot per day
                else:
                    continue
            setup, qty = position
            exit_px = None
            if setup.side > 0:
                if bar["low"] <= setup.stop:
                    exit_px, why = min(setup.stop, bar["open"]) - STOP_SLIP, "stop"
                elif bar["high"] >= setup.target:
                    exit_px, why = setup.target, "target"
            else:
                if bar["high"] >= setup.stop:
                    exit_px, why = max(setup.stop, bar["open"]) + STOP_SLIP, "stop"
                elif bar["low"] <= setup.target:
                    exit_px, why = setup.target, "target"
            if exit_px is None and q.index[t].time() >= FLAT_TIME:
                exit_px, why = bar["close"], "eod"
            if exit_px is not None:
                pnl = (exit_px - setup.entry) * qty * setup.side
                equity += pnl
                trades.append({
                    "day": day, "side": setup.side, "entry": setup.entry, "exit": exit_px,
                    "qty": qty, "why": why, "pnl": pnl, "r": pnl / (setup.risk * qty),
                    "equity": equity,
                })
                break
    return pd.DataFrame(trades)


def summarize(tr: pd.DataFrame, q: pd.DataFrame) -> dict:
    if tr.empty:
        return {"trades": 0}
    curve = tr["equity"]
    dd = (curve / curve.cummax().clip(lower=START_CASH) - 1).min()
    first_day = q.index[q.index.date >= tr["day"].iloc[0]][0]
    return {
        "trades": len(tr),
        "sessions": len(set(q.index.date)),
        "win_rate_pct": round(100 * (tr["pnl"] > 0).mean(), 1),
        "avg_r": round(tr["r"].mean(), 3),
        "return_pct": round(100 * (curve.iloc[-1] / START_CASH - 1), 2),
        "max_dd_pct": round(100 * dd, 2),
        "qqq_buy_hold_pct": round(100 * (q["close"].iloc[-1] / q.loc[first_day, "open"] - 1), 2),
        "exits": tr["why"].value_counts().to_dict(),
        "long_avg_r": round(tr.loc[tr.side > 0, "r"].mean(), 3),
        "short_avg_r": round(tr.loc[tr.side < 0, "r"].mean(), 3),
        "n_long": int((tr.side > 0).sum()),
        "n_short": int((tr.side < 0).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default="data/ict/bars_sip.pkl")
    args = ap.parse_args()
    q, s = load(args.bars)
    tr = run(q, s)
    print(f"{q.index[0].date()} -> {q.index[-1].date()}")
    for k, v in summarize(tr, q).items():
        print(f"  {k:18s} {v}")
    if not tr.empty:
        tr["year_half"] = [f"{d.year}H{1 if d.month <= 6 else 2}" for d in tr["day"]]
        print(tr.groupby("year_half").agg(n=("r", "size"), avg_r=("r", "mean"), pnl=("pnl", "sum")).round(3))
        out = Path("backtesting/results/ict_trades.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        tr.to_csv(out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
