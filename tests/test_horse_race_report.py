from __future__ import annotations

import pytest

from live.horse_race_report import build_report, format_table
from live.virtual_portfolio import VirtualPortfolio


def test_report_ranks_by_total_equity():
    a = VirtualPortfolio("momentum_rotation", 1000.0)
    a.record_buy("AAPL", 2.0, 100.0)  # cash 800, holds 2 AAPL
    b = VirtualPortfolio("rsi_mr", 1000.0)  # all cash
    rows = build_report([a, b], marks={"AAPL": 150.0})
    # a equity = 800 + 2*150 = 1100 ; b = 1000. a ranks first.
    assert rows[0]["strategy"] == "momentum_rotation"
    assert rows[0]["equity"] == pytest.approx(1100.0)
    assert rows[0]["return_pct"] == pytest.approx(10.0)
    assert rows[1]["strategy"] == "rsi_mr"
    assert rows[1]["return_pct"] == pytest.approx(0.0)


def test_report_computes_alpha_against_benchmark():
    a = VirtualPortfolio("momentum_rotation", 1000.0)
    a.record_buy("AAPL", 2.0, 100.0)  # equity 1100 -> +10%
    rows = build_report([a], marks={"AAPL": 150.0}, benchmark_pct=4.0)
    assert rows[0]["alpha_pct"] == pytest.approx(6.0)  # 10% return - 4% market


def test_report_no_alpha_when_benchmark_absent():
    a = VirtualPortfolio("rsi_mr", 1000.0)
    rows = build_report([a], marks={})
    assert "alpha_pct" not in rows[0]


def test_format_table_renders_benchmark_footer():
    a = VirtualPortfolio("momentum_rotation", 1000.0)
    a.record_buy("AAPL", 2.0, 100.0)
    rows = build_report([a], marks={"AAPL": 150.0}, benchmark_pct=4.0)
    benchmark = {"symbol": "SPY", "return_pct": 4.0, "start_date": "2026-06-05"}
    table = format_table(rows, benchmark=benchmark)
    assert "alpha%" in table
    assert "SPY" in table
    assert "2026-06-05" in table
