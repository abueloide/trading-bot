from __future__ import annotations

from datetime import date

import pytest

from live.equity_snapshot import append_snapshot, load_snapshots


def _rows():
    return [
        {"strategy": "momentum_rotation", "equity": 26000.0, "return_pct": 4.0, "alpha_pct": 1.0},
        {"strategy": "rsi_mr", "equity": 24000.0, "return_pct": -4.0, "alpha_pct": -7.0},
    ]


def test_append_writes_one_record_per_strategy(tmp_path):
    path = tmp_path / "equity_curve.jsonl"
    append_snapshot(_rows(), benchmark_pct=3.0, snapshot_date=date(2026, 6, 14), path=path)

    records = load_snapshots(path)
    assert len(records) == 2
    momentum = next(r for r in records if r["strategy"] == "momentum_rotation")
    assert momentum["date"] == "2026-06-14"
    assert momentum["equity"] == pytest.approx(26000.0)
    assert momentum["return_pct"] == pytest.approx(4.0)
    assert momentum["alpha_pct"] == pytest.approx(1.0)
    assert momentum["benchmark_pct"] == pytest.approx(3.0)


def test_append_accumulates_across_days(tmp_path):
    path = tmp_path / "equity_curve.jsonl"
    append_snapshot(_rows(), benchmark_pct=3.0, snapshot_date=date(2026, 6, 13), path=path)
    append_snapshot(_rows(), benchmark_pct=3.5, snapshot_date=date(2026, 6, 14), path=path)

    records = load_snapshots(path)
    assert len(records) == 4
    dates = {r["date"] for r in records}
    assert dates == {"2026-06-13", "2026-06-14"}


def test_rerun_same_day_replaces_not_duplicates(tmp_path):
    path = tmp_path / "equity_curve.jsonl"
    append_snapshot(_rows(), benchmark_pct=3.0, snapshot_date=date(2026, 6, 14), path=path)
    # Re-run later the same day with a fresh mark: must replace, not append.
    updated = [dict(r, equity=r["equity"] + 100.0) for r in _rows()]
    append_snapshot(updated, benchmark_pct=3.2, snapshot_date=date(2026, 6, 14), path=path)

    records = load_snapshots(path)
    assert len(records) == 2
    momentum = next(r for r in records if r["strategy"] == "momentum_rotation")
    assert momentum["equity"] == pytest.approx(26100.0)
    assert momentum["benchmark_pct"] == pytest.approx(3.2)


def test_benchmark_pct_none_persists_as_null(tmp_path):
    path = tmp_path / "equity_curve.jsonl"
    rows = [{"strategy": "rsi_mr", "equity": 25000.0, "return_pct": 0.0}]
    append_snapshot(rows, benchmark_pct=None, snapshot_date=date(2026, 6, 14), path=path)

    records = load_snapshots(path)
    assert records[0]["benchmark_pct"] is None
    assert records[0]["alpha_pct"] is None


def test_load_missing_file_returns_empty(tmp_path):
    assert load_snapshots(tmp_path / "nope.jsonl") == []
