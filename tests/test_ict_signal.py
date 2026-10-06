"""find_setup: a hand-built long setup fires once, mirrors to a short, and
respects bias / trade window."""
import pandas as pd

from live.ict_signal import daily_bias, find_setup


def _frame(rows, start="2026-10-05 15:00"):
    idx = pd.date_range(start, periods=len(rows), freq="5min", tz="America/New_York")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def _bars():
    # 12 flat bars of the prior session, then today's open at 09:30.
    flat = [(100.0, 100.5, 99.5, 100.0)] * 12
    today = [
        (100.0, 100.4, 99.8, 100.0),   # 09:30
        (100.0, 100.1, 99.0, 99.1),    # 09:35 drop
        (98.6, 98.7, 98.0, 98.2),      # 09:40 gap: high 98.7 < low[09:30] 99.8 ; sweeps 99.5
        (98.2, 98.9, 98.1, 98.8),      # 09:45
        (98.8, 100.2, 98.7, 100.1),    # 09:50 closes above 99.8 -> inversion
    ]
    q = _frame(flat, "2026-10-05 15:00")
    q = pd.concat([q, _frame(today, "2026-10-06 09:30")])
    # SPY holds its low while QQQ sweeps -> SMT divergence.
    s = q.copy()
    s["low"] = s["low"].clip(lower=99.6)
    return q, s


def test_long_setup_fires_on_inversion_bar():
    q, s = _bars()
    setup = find_setup(q, s, bias=1)
    assert setup is not None and setup.side == 1
    assert setup.entry == 99.8
    assert setup.stop == 98.0 - 0.02
    assert setup.entry + 2 * setup.risk <= setup.target <= setup.entry + 3 * setup.risk


def test_no_setup_before_inversion_or_against_bias():
    q, s = _bars()
    assert find_setup(q.iloc[:-1], s.iloc[:-1], bias=1) is None
    assert find_setup(q, s, bias=-1) is None
    assert find_setup(q, s, bias=0) is None


def test_no_setup_without_divergence():
    q, _ = _bars()
    assert find_setup(q, q.copy(), bias=1) is None


def test_short_is_the_mirror():
    q, s = _bars()
    flip = lambda d: pd.DataFrame(
        {"open": 200 - d["open"], "high": 200 - d["low"], "low": 200 - d["high"], "close": 200 - d["close"]},
        index=d.index,
    )
    setup = find_setup(flip(q), flip(s), bias=-1)
    assert setup is not None and setup.side == -1
    assert abs(setup.entry - 100.2) < 1e-9 and setup.stop > setup.entry > setup.target


def test_bias_uses_only_prior_sessions():
    closes = pd.Series(range(1, 31), index=pd.date_range("2026-09-01", periods=30).date)
    assert daily_bias(closes, closes.index[25]) == 1
    assert daily_bias(closes[::-1].set_axis(closes.index), closes.index[25]) == -1
    assert daily_bias(closes, closes.index[5]) == 0


def test_book_day_realizes_round_trip_and_refuses_open_position():
    from datetime import date
    from types import SimpleNamespace as NS

    from live.ict_runner import book_day, position_size

    def fill(side, qty, px):
        return NS(side=NS(value=side), filled_qty=str(qty), filled_avg_price=str(px))

    state = {"realized_pnl": 0.0, "trades": []}
    trade = book_day(state, [fill("sell", 100, 600.0), fill("buy", 100, 598.0)], date(2026, 10, 6))
    assert trade["side"] == -1 and trade["pnl"] == 200.0 and state["realized_pnl"] == 200.0
    # still holding shares -> nothing booked
    assert book_day(state, [fill("buy", 100, 600.0)], date(2026, 10, 7)) is None
    assert state["realized_pnl"] == 200.0
    # 1% risk would be 250 shares; the 4x notional cap (100k/600) binds first.
    assert position_size(25_000, 600.0, 1.0) == 166
    assert position_size(25_000, 600.0, 5.0) == 50
