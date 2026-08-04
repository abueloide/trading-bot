"""Persist/restore the per-strategy virtual ledgers as JSON.

The bot is the only actor on the paper account, so persist-and-trust is fine
for v1: each run loads the prior ledger state and saves the new state at the
end. Atomic write (tmp + replace) avoids corruption if the process is killed
mid-write.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from live.virtual_portfolio import VirtualPortfolio

logger = logging.getLogger(__name__)


def save_ledgers(portfolios: List[VirtualPortfolio], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {vp.strategy: vp.to_dict() for vp in portfolios}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)  # atomic on POSIX


def load_ledgers(path) -> Dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # Fail hard: a corrupt-but-PRESENT state file must NOT silently reset all
        # ledgers to starting cash while Alpaca still holds every position — the
        # next cycle would re-buy full slices (doubled broker positions) and wipe
        # the race history. A fresh start must be explicit (scripts/reset_race.py).
        raise SystemExit(
            f"FATAL: ledger state exists but is unreadable at {path}: {e}. "
            "Refusing to start fresh (would double broker positions). Restore the "
            "file from backup, or run scripts/reset_race.py to reset intentionally."
        )
