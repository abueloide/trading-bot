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


def test_sell_symbol_never_owned_raises():
    vp = VirtualPortfolio("s", starting_cash=1000.0)
    with pytest.raises(ValueError):
        vp.record_sell("TSLA", qty=1.0, price=100.0)
    assert vp.cash == pytest.approx(1000.0)  # cash untouched


def test_sell_zero_or_negative_price_raises():
    vp = VirtualPortfolio("s", starting_cash=1000.0)
    vp.record_buy("AAPL", qty=1.0, price=100.0)
    with pytest.raises(ValueError):
        vp.record_sell("AAPL", qty=1.0, price=0.0)


def test_cross_strategy_isolation():
    # Two strategies hold the same symbol independently. Selling from one
    # must not touch the other's position — the core ledger invariant.
    a = VirtualPortfolio("momentum_rotation", starting_cash=1000.0)
    b = VirtualPortfolio("rsi_mr", starting_cash=1000.0)
    a.record_buy("AAPL", qty=3.0, price=100.0)
    b.record_buy("AAPL", qty=2.0, price=100.0)
    a.record_sell("AAPL", qty=3.0, price=120.0)
    assert a.qty("AAPL") == pytest.approx(0.0)
    assert b.qty("AAPL") == pytest.approx(2.0)  # untouched
    assert b.avg_entry("AAPL") == pytest.approx(100.0)


def test_full_liquidation_then_rebuy_resets_avg_entry():
    vp = VirtualPortfolio("s", starting_cash=1000.0)
    vp.record_buy("AAPL", qty=2.0, price=100.0)
    vp.record_sell("AAPL", qty=2.0, price=110.0)
    vp.record_buy("AAPL", qty=1.0, price=200.0)
    assert vp.qty("AAPL") == pytest.approx(1.0)
    assert vp.avg_entry("AAPL") == pytest.approx(200.0)  # old lot fully cleared


def test_to_dict_from_dict_roundtrip():
    vp = VirtualPortfolio("confirmed_mr", starting_cash=1000.0)
    vp.record_buy("AAPL", qty=2.0, price=100.0)
    vp.record_buy("MSFT", qty=1.0, price=50.0)
    vp.record_sell("AAPL", qty=1.0, price=130.0)  # realized +30, holds 1 AAPL @100
    data = vp.to_dict()
    restored = VirtualPortfolio.from_dict(data)
    assert restored.strategy == "confirmed_mr"
    assert restored.starting_cash == pytest.approx(1000.0)
    assert restored.cash == pytest.approx(vp.cash)
    assert restored.realized_pnl == pytest.approx(30.0)
    assert restored.qty("AAPL") == pytest.approx(1.0)
    assert restored.avg_entry("AAPL") == pytest.approx(100.0)
    assert restored.qty("MSFT") == pytest.approx(1.0)


def test_to_dict_is_json_serializable():
    import json
    vp = VirtualPortfolio("rsi_mr", 500.0)
    vp.record_buy("SPY", 1.0, 400.0)
    json.dumps(vp.to_dict())  # must not raise
