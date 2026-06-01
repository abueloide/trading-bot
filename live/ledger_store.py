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
        logger.error("ledger state unreadable at %s: %s — starting fresh", path, e)
        return {}
