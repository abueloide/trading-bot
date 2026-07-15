"""Persist a daily equity/alpha snapshot per horse as an append-only JSONL.

The horse-race report only ever prints the *latest* cut. To judge an edge you
need a time series: an equity curve gives drawdown and alpha *stability*, not
just a single end-of-window number. Each run appends one record per strategy
for the snapshot date; a re-run on the same day replaces that day's rows
(idempotent) so the cron firing twice never double-counts.

Schema (one JSON object per line):
    {"date", "strategy", "equity", "return_pct", "alpha_pct", "benchmark_pct"}

``alpha_pct`` and ``benchmark_pct`` are null when no benchmark was available.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def load_snapshots(path, *, strict: bool = False) -> List[dict]:
    """Read every snapshot record.

    ``strict=False`` (readers/reports): a missing or unreadable file → empty
    list, so a single bad line never breaks a display. ``strict=True`` (the
    rewrite path in ``append_snapshot``): raise instead of returning ``[]`` — a
    partial read must never be the basis for rewriting the equity curve, or one
    corrupt byte would replace the whole experiment history with today's rows.
    """
    path = Path(path)
    if not path.exists():
        return []
    records: List[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (json.JSONDecodeError, OSError) as e:
        if strict:
            raise SystemExit(
                f"FATAL: equity curve exists but is unreadable at {path}: {e}. "
                "Refusing to rewrite history from a partial read (would lose the "
                "curve). Restore the file from backup before the next run."
            )
        logger.error("equity snapshot unreadable at %s: %s", path, e)
        return []
    return records


def append_snapshot(
    rows: List[dict],
    benchmark_pct: Optional[float],
    snapshot_date: date,
    path,
) -> None:
    """Append one record per strategy for ``snapshot_date`` (idempotent per day).

    ``rows`` are the dicts produced by ``horse_race_report.build_report``. Any
    existing rows for the same date are dropped first so a same-day re-run
    overwrites rather than duplicates. Atomic write (tmp + replace).
    """
    path = Path(path)
    day = snapshot_date.isoformat()

    kept = [r for r in load_snapshots(path, strict=True) if r.get("date") != day]
    for r in rows:
        alpha = r.get("alpha_pct")
        kept.append(
            {
                "date": day,
                "strategy": r["strategy"],
                "equity": r["equity"],
                "return_pct": r["return_pct"],
                "alpha_pct": alpha if benchmark_pct is not None else None,
                "benchmark_pct": benchmark_pct,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, path)  # atomic on POSIX
