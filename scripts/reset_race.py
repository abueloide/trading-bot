#!/usr/bin/env python3
"""One-shot reset of the paper horse race to fresh equal $25k books.

Resets ALL horses at once — the ONLY time flattening the shared Alpaca paper
account is safe. The per-strategy SELL isolation guardrail (never close_position
mid-race) exists because four horses net into one account; but the instant every
virtual book goes to all-cash, every open position is an orphan, so close_all()
can't clobber a live horse's shares. Resetting one horse without the others would
violate that — this script intentionally resets the whole field.

Steps (backs up before clobbering, aborts before writing if the broker can't be
confirmed flat, so virtual books never desync from Alpaca):
  1. Back up state.json + equity_curve.jsonl with a dated suffix.
  2. close_all() on the Alpaca paper account (frees cash for the new baskets).
  3. Write fresh {strategy: starting_cash, no lots} books for every horse.
  4. Rotate equity_curve.jsonl out so the race % restarts at today's inception.

Usage:
    python scripts/reset_race.py          # dry run: print the plan, touch nothing
    python scripts/reset_race.py --yes    # actually flatten + reset
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor import Executor  # noqa: E402
from run_trading_system import EQUITY_CURVE_PATH, STATE_PATH, STRATEGIES  # noqa: E402


def _fresh_books() -> dict:
    return {
        c.strategy: {
            "strategy": c.strategy,
            "starting_cash": float(c.starting_cash),
            "cash": float(c.starting_cash),
            "realized_pnl": 0.0,
            "lots": {},
        }
        for c in STRATEGIES
    }


def _backup(path: Path, stamp: str) -> None:
    if path.exists():
        dest = path.with_suffix(path.suffix + f".bak-{stamp}")
        dest.write_bytes(path.read_bytes())
        print(f"  backed up {path} -> {dest.name}")


def main() -> int:
    load_dotenv()
    apply = "--yes" in sys.argv
    books = _fresh_books()

    base_url = os.getenv("ALPACA_BASE_URL", "")
    if not base_url.startswith("https://paper-api.alpaca.markets"):
        print(f"REFUSING: ALPACA_BASE_URL is not paper (got {base_url!r}).")
        return 1

    print(f"Reset plan — {len(books)} horses, each ${list(books.values())[0]['cash']:,.0f}:")
    for name in books:
        print(f"  - {name}")
    if not apply:
        print("\nDry run. Re-run with --yes to flatten the paper account and reset.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print("\n[1/4] backing up ledgers...")
    _backup(STATE_PATH, stamp)
    _backup(EQUITY_CURVE_PATH, stamp)

    print("[2/4] flattening Alpaca paper account...")
    ex = Executor()
    acct = ex.get_account()
    if acct is None:
        print("ABORT: cannot reach the Alpaca paper account; books left untouched.")
        return 2
    remaining = ex.close_all()
    print(f"  account equity ${acct['equity']:,.2f} — positions remaining after close_all: {remaining}")
    if remaining != 0:
        print("  WARN: positions still open (likely market closed / pending). "
              "Re-run after the next session to confirm flat.")

    print("[3/4] writing fresh books...")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(books, indent=2))
    os.replace(tmp, STATE_PATH)
    print(f"  wrote {STATE_PATH}")

    print("[4/4] rotating equity curve...")
    if EQUITY_CURVE_PATH.exists():
        EQUITY_CURVE_PATH.unlink()
        print(f"  cleared {EQUITY_CURVE_PATH} (backup kept)")

    print("\nDone. Race restarts at the next cron run with all horses at 0.00%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
