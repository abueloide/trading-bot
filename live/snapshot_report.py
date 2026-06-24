"""Canonical apertura/cierre snapshot — ONE source of truth, no network.

The morning (apertura) and evening (cierre) messages used to disagree with the
settled equity curve by a few dollars because they pulled a *live* Alpaca account
read while the curve is built from yfinance settled closes — same money, two
yardsticks. With the market closed at 7:30, a live read adds nothing but drift.

This reads the SAME persisted artifacts the 2-week checkpoint reads:
  * ``equity_curve.jsonl`` → latest settled equity + return per horse
  * ``state.json``         → position count per horse (lots held)

so apertura, cierre, standup and checkpoint can never quote different numbers for
the same day. No Alpaca call, no re-fetch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from live.equity_snapshot import load_snapshots

DEFAULT_CURVE_PATH = Path("data/ledgers/equity_curve.jsonl")
DEFAULT_STATE_PATH = Path("data/ledgers/state.json")


def _position_counts(state_path) -> Dict[str, int]:
    """Held-lot count per strategy from the persisted ledger; {} if unreadable."""
    path = Path(state_path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return {name: len(s.get("lots", {})) for name, s in state.items()}


def latest_snapshot(
    curve_path=DEFAULT_CURVE_PATH,
    state_path=DEFAULT_STATE_PATH,
) -> Optional[dict]:
    """Canonical numbers for the most recent settled day, or None if no curve.

    Returns ``{"date", "rows": [{strategy, equity, return_pct, alpha_pct, pos}],
    "total_equity", "total_return_pct"}``. ``pos`` is None when state is
    unreadable (numbers still come from the curve, so they stay consistent).
    """
    records = load_snapshots(curve_path)
    if not records:
        return None
    day = max(r["date"] for r in records)
    counts = _position_counts(state_path)
    rows: List[dict] = []
    for r in records:
        if r["date"] != day:
            continue
        rows.append(
            {
                "strategy": r["strategy"],
                "equity": r["equity"],
                "return_pct": r["return_pct"],
                "alpha_pct": r.get("alpha_pct"),
                "pos": counts.get(r["strategy"]),
            }
        )
    rows.sort(key=lambda x: x["equity"], reverse=True)
    total_equity = sum(x["equity"] for x in rows)
    # Total return is measured against the SAME base each horse started with, so a
    # plain mean of per-horse returns is the equal-weight book return (every horse
    # is funded with the identical $25k slice). Empty curve already returned None.
    total_return_pct = sum(x["return_pct"] for x in rows) / len(rows)
    return {
        "date": day,
        "rows": rows,
        "total_equity": total_equity,
        "total_return_pct": total_return_pct,
    }


if __name__ == "__main__":  # smoke / self-check
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        curve = Path(d) / "curve.jsonl"
        state = Path(d) / "state.json"
        curve.write_text(
            "\n".join(
                json.dumps(r)
                for r in [
                    {"date": "2026-06-22", "strategy": "a", "equity": 30000.0, "return_pct": 20.0, "alpha_pct": 19.0},
                    {"date": "2026-06-23", "strategy": "a", "equity": 27000.0, "return_pct": 8.0, "alpha_pct": 8.2},
                    {"date": "2026-06-23", "strategy": "b", "equity": 24000.0, "return_pct": -4.0, "alpha_pct": -3.8},
                ]
            )
            + "\n"
        )
        state.write_text(json.dumps({"a": {"lots": {"X": {}, "Y": {}}}, "b": {"lots": {"Z": {}}}}))
        snap = latest_snapshot(curve, state)
        assert snap["date"] == "2026-06-23", snap  # latest day only
        assert [r["strategy"] for r in snap["rows"]] == ["a", "b"], snap  # equity-sorted
        assert snap["rows"][0]["pos"] == 2 and snap["rows"][1]["pos"] == 1, snap
        assert abs(snap["total_equity"] - 51000.0) < 1e-6, snap
        assert abs(snap["total_return_pct"] - 2.0) < 1e-6, snap  # mean(8, -4)
        # Unreadable state -> pos None, numbers still from the curve.
        snap2 = latest_snapshot(curve, Path(d) / "missing.json")
        assert snap2["rows"][0]["pos"] is None, snap2
        assert snap2["date"] == "2026-06-23", snap2
        print("snapshot_report self-check OK")
