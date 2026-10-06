#!/usr/bin/env python3
"""One-shot: retire specific horses mid-race WITHOUT flattening the account.

Unlike reset_race.py (close_all, resets everyone), this sells only the retired
strategies' exact per-ledger qty via place_market_sell, so the surviving horses'
shares in the shared paper account are untouched. Retired ledgers are archived,
then dropped from state.json.

Usage:
    python scripts/retire_strategies.py rsi_mr donchian_breakout opex_drift          # dry run
    python scripts/retire_strategies.py rsi_mr donchian_breakout opex_drift --yes
    (add --news-reactor to also liquidate data/news_reactor/ledger.json)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor import Executor, is_market_open  # noqa: E402

STATE_PATH = Path("data/ledgers/state.json")
REACTOR_LEDGER = Path("data/news_reactor/ledger.json")
REACTOR_OPENED = Path("data/news_reactor/opened.json")


def _atomic_write(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def main() -> int:
    load_dotenv()
    apply = "--yes" in sys.argv
    with_reactor = "--news-reactor" in sys.argv
    retire = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not os.getenv("ALPACA_BASE_URL", "").startswith("https://paper-api.alpaca.markets"):
        print("REFUSING: ALPACA_BASE_URL is not paper.")
        return 1

    state = json.loads(STATE_PATH.read_text())
    unknown = [r for r in retire if r not in state]
    if unknown:
        print(f"ABORT: not in state.json: {unknown}")
        return 1

    books = {r: state[r] for r in retire}
    if with_reactor and REACTOR_LEDGER.exists():
        books["news_reactor"] = json.loads(REACTOR_LEDGER.read_text())

    sells = [(name, sym, lot["qty"]) for name, b in books.items() for sym, lot in b["lots"].items()]
    print(f"Retire {list(books)} — {len(sells)} sells:")
    for name, sym, qty in sells:
        print(f"  {name:20s} SELL {qty:.4f} {sym}")
    if not apply:
        print("\nDry run. Re-run with --yes.")
        return 0
    if not is_market_open():
        print("ABORT: market closed — sells would be rejected; nothing changed.")
        return 2

    ex = Executor()
    failed = []
    for name, sym, qty in sells:
        res = ex.place_market_sell(symbol=sym, qty=qty, strategy=name)
        if res is None:
            failed.append((name, sym, qty))
        else:
            books[name]["lots"][sym]["sell_order_id"] = res["id"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = STATE_PATH.parent / f"retired-{stamp}.json"
    _atomic_write(archive, books)
    print(f"archived retired ledgers -> {archive}")

    if failed:
        # Keep the failed lots on the books: dropping a ledger whose shares are
        # still at the broker would orphan them.
        print(f"WARN {len(failed)} sells failed; their ledgers stay in state.json: {failed}")
    failed_names = {f[0] for f in failed}
    for name in retire:
        if name not in failed_names:
            state.pop(name)
    _atomic_write(STATE_PATH, state)
    if with_reactor and "news_reactor" not in failed_names and REACTOR_LEDGER.exists():
        b = books["news_reactor"]
        _atomic_write(REACTOR_LEDGER, {**b, "lots": {}})
        _atomic_write(REACTOR_OPENED, {})
    print(f"state.json now holds: {list(state)}")
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
