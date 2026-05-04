#!/usr/bin/env python3
"""
Trade Journal — JSONL daily files plus DB log + weekly summaries.

Every executed action lands here with full strategy attribution so we can
answer: which strategy made which P&L over which window.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

JOURNAL_DIR = Path("data/journal")
JOURNAL_DIR.mkdir(parents=True, exist_ok=True)


def _serialize(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    return str(obj)


class TradeJournal:
    """Append-only JSONL journal with optional DB mirror."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or JOURNAL_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._db = None
        try:
            from database_manager import get_database_manager
            self._db = get_database_manager()
        except Exception:
            pass

    def _path_for_today(self) -> Path:
        return self.base_dir / f"{date.today().isoformat()}.jsonl"

    def write(self, entry: Dict[str, Any]) -> None:
        entry = dict(entry)
        entry.setdefault("timestamp", datetime.utcnow())
        try:
            with open(self._path_for_today(), "a") as f:
                f.write(json.dumps(entry, default=_serialize) + "\n")
        except Exception as e:
            logger.error(f"journal write failed: {e}")
        if self._db is not None:
            try:
                self._db.log_trade_journal(entry)
            except Exception as e:
                logger.warning(f"journal DB mirror failed: {e}")

    # ---------------------------------------------------------- reads

    def read_day(self, d: Optional[date] = None) -> List[Dict[str, Any]]:
        d = d or date.today()
        path = self.base_dir / f"{d.isoformat()}.jsonl"
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except Exception as e:
            logger.error(f"journal read failed: {e}")
        return out

    def read_range(self, start: date, end: date) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        d = start
        while d <= end:
            out.extend(self.read_day(d))
            d += timedelta(days=1)
        return out

    # ---------------------------------------------------------- summaries

    def weekly_summary(self, end: Optional[date] = None) -> Dict[str, Any]:
        end = end or date.today()
        start = end - timedelta(days=6)
        entries = self.read_range(start, end)
        return self._summarize(entries, label=f"{start} → {end}")

    def strategy_attribution(
        self, start: Optional[date] = None, end: Optional[date] = None
    ) -> Dict[str, Dict[str, float]]:
        end = end or date.today()
        start = start or (end - timedelta(days=30))
        entries = self.read_range(start, end)
        out: Dict[str, Dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
        for e in entries:
            strategy = e.get("strategy") or "unknown"
            pnl = float(e.get("pnl") or 0)
            if pnl != 0:
                out[strategy]["pnl"] += pnl
                out[strategy]["trades"] += 1
                if pnl > 0:
                    out[strategy]["wins"] += 1
        return dict(out)

    @staticmethod
    def _summarize(entries: Iterable[Dict[str, Any]], label: str) -> Dict[str, Any]:
        n = 0
        total_pnl = 0.0
        wins = 0
        losses = 0
        per_strategy: Dict[str, float] = defaultdict(float)
        for e in entries:
            n += 1
            pnl = float(e.get("pnl") or 0)
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
            strategy = e.get("strategy") or "unknown"
            per_strategy[strategy] += pnl
        return {
            "label": label,
            "n_entries": n,
            "total_pnl": total_pnl,
            "wins": wins,
            "losses": losses,
            "per_strategy": dict(per_strategy),
        }


_journal: Optional[TradeJournal] = None


def get_trade_journal() -> TradeJournal:
    global _journal
    if _journal is None:
        _journal = TradeJournal()
    return _journal
