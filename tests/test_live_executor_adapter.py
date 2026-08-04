from __future__ import annotations

from live.live_executor_adapter import LiveExecutorAdapter


class FakeExec:
    def __init__(self):
        self.calls = []

    def place_market_order_with_time_exit(self, symbol, qty, max_hold_days, side="BUY", strategy=""):
        self.calls.append(("buy", symbol, qty, max_hold_days, side, strategy))
        return {"id": "1"}

    def place_market_sell(self, symbol, qty, strategy=""):
        self.calls.append(("sell", symbol, qty, strategy))
        return {"id": "2"}

    def close_position(self, symbol, reason=""):
        # MONEY GUARDRAIL: the adapter must NEVER route a strategy SELL here —
        # in the shared paper account this liquidates the whole net position
        # and clobbers the other horses' shares. Recorded so a reroute is loud.
        self.calls.append(("close_position", symbol, reason))
        return {"id": "danger"}


def test_buy_uses_backstop_hold_when_none():
    ex = FakeExec()
    ok = LiveExecutorAdapter(ex).buy(symbol="AAPL", qty=2.0, price=100.0,
                                     strategy="momentum_rotation",
                                     strategy_type="momentum", max_hold_days=None)
    assert ok is True
    kind, sym, qty, hold, side, strat = ex.calls[0]
    assert kind == "buy" and side == "BUY" and hold == 252 and strat == "momentum_rotation"


def test_buy_passes_explicit_hold():
    ex = FakeExec()
    LiveExecutorAdapter(ex).buy(symbol="AAPL", qty=1.0, price=10.0, strategy="rsi_mr",
                                strategy_type="mean_reversion", max_hold_days=7)
    assert ex.calls[0][3] == 7


def test_sell_uses_qty_specific_market_sell():
    ex = FakeExec()
    ok = LiveExecutorAdapter(ex).sell(symbol="AAPL", qty=3.0, price=120.0, strategy="confirmed_mr")
    assert ok is True
    kind, sym, qty, strat = ex.calls[0]
    assert kind == "sell" and sym == "AAPL" and qty == 3.0 and strat == "confirmed_mr"


def test_sell_never_routes_through_close_position():
    """Locks the money guardrail: a strategy SELL must hit place_market_sell,
    never close_position (which would liquidate the shared net position)."""
    ex = FakeExec()
    LiveExecutorAdapter(ex).sell(symbol="AAPL", qty=3.0, price=120.0, strategy="confirmed_mr")
    assert all(c[0] != "close_position" for c in ex.calls)
