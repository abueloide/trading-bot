from __future__ import annotations

import pytest

from live.virtual_portfolio import VirtualPortfolio


def test_buy_reduces_cash_and_adds_position():
    vp = VirtualPortfolio("momentum_rotation", starting_cash=1000.0)
    vp.record_buy("AAPL", qty=2.0, price=100.0)
    assert vp.cash == pytest.approx(800.0)
    assert vp.qty("AAPL") == pytest.approx(2.0)
    assert vp.avg_entry("AAPL") == pytest.approx(100.0)


def test_buy_averages_entry_price():
    vp = VirtualPortfolio("s", starting_cash=1000.0)
    vp.record_buy("AAPL", qty=2.0, price=100.0)
    vp.record_buy("AAPL", qty=2.0, price=120.0)
    assert vp.qty("AAPL") == pytest.approx(4.0)
    assert vp.avg_entry("AAPL") == pytest.approx(110.0)


def test_sell_books_realized_pnl_and_reduces_qty():
    vp = VirtualPortfolio("s", starting_cash=1000.0)
    vp.record_buy("AAPL", qty=4.0, price=100.0)  # cash 600
    vp.record_sell("AAPL", qty=2.0, price=130.0)  # +260 cash, +60 realized
    assert vp.qty("AAPL") == pytest.approx(2.0)
    assert vp.cash == pytest.approx(860.0)
    assert vp.realized_pnl == pytest.approx(60.0)


def test_sell_more_than_owned_is_rejected():
    vp = VirtualPortfolio("s", starting_cash=1000.0)
    vp.record_buy("AAPL", qty=1.0, price=100.0)
    with pytest.raises(ValueError):
        vp.record_sell("AAPL", qty=2.0, price=100.0)


def test_to_portfolio_state_reflects_slice():
    vp = VirtualPortfolio("s", starting_cash=1000.0)
    vp.record_buy("AAPL", qty=2.0, price=100.0)
    state = vp.to_portfolio_state(marks={"AAPL": 110.0})
    assert state.cash == pytest.approx(800.0)
    # equity = cash + marked positions = 800 + 2*110
    assert state.equity == pytest.approx(1020.0)
    assert len(state.positions) == 1
    assert state.positions[0]["symbol"] == "AAPL"
    assert state.positions[0]["market_value"] == pytest.approx(220.0)
