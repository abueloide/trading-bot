"""ict_daily_runner.step: forces exactly one trade per session in the bias
direction, skips only when bias is unknown, and books flat at the close."""
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pandas as pd

from live import ict_daily_runner as dr


class FakeBroker:
    """Only the methods step() touches. Records the forced entry."""

    def __init__(self, closes: pd.Series, price: float = 600.0):
        self._closes = closes
        self._price = price
        self.submitted = None
        self.flattened = False

    def bars(self, now):
        # One session per close; rth() needs a tz-aware NY index with a 'close'.
        idx = pd.DatetimeIndex([pd.Timestamp(d, tz="America/New_York") + pd.Timedelta(hours=10)
                                for d in self._closes.index])
        q = pd.DataFrame({"open": self._closes.values, "high": self._closes.values,
                          "low": self._closes.values, "close": self._closes.values}, index=idx)
        return pd.concat({dr.TRADED: q}, names=["symbol"])

    def last_price(self, symbol):
        return self._price

    def submit_market_stop(self, symbol, side, qty, stop, client_id):
        self.submitted = NS(side=side, qty=qty, stop=stop, client_id=client_id)
        return "order-1"

    def flatten(self):
        self.flattened = True

    def day_fills(self, day):
        return []


def _closes(n=30, rising=True):
    vals = range(1, n + 1) if rising else range(n, 0, -1)
    return pd.Series([float(v) for v in vals], index=pd.date_range("2026-09-01", periods=n).date)


def _now(h, m=0):
    return datetime(2026, 10, 15, h, m, tzinfo=timezone.utc)  # 15:00 UTC = 11:00 ET


def test_forces_long_entry_when_bias_up():
    broker = FakeBroker(_closes(rising=True))
    state = {"starting_cash": 25_000.0, "realized_pnl": 0.0, "day": None, "trades": []}
    dr.step(broker, state, _now(14))  # 14 UTC = 10:00 ET, inside RTH
    assert state["entered"] is True
    assert broker.submitted.side == 1 and broker.submitted.qty > 0
    assert broker.submitted.stop < 600.0  # long stop below entry


def test_forces_short_entry_when_bias_down():
    broker = FakeBroker(_closes(rising=False))
    state = {"starting_cash": 25_000.0, "realized_pnl": 0.0, "day": None, "trades": []}
    dr.step(broker, state, _now(14))
    assert state["entered"] is True
    assert broker.submitted.side == -1
    assert broker.submitted.stop > 600.0  # short stop above entry


def test_only_one_entry_per_day():
    broker = FakeBroker(_closes(rising=True))
    state = {"starting_cash": 25_000.0, "realized_pnl": 0.0, "day": None, "trades": []}
    dr.step(broker, state, _now(14))
    first = broker.submitted
    broker.submitted = None
    dr.step(broker, state, _now(14, 30))  # same day, later bar
    assert broker.submitted is None and first is not None


def test_no_entry_when_bias_unknown():
    broker = FakeBroker(_closes(n=5))  # < BIAS_EMA history -> bias 0
    state = {"starting_cash": 25_000.0, "realized_pnl": 0.0, "day": None, "trades": []}
    dr.step(broker, state, _now(14))
    assert state["entered"] is False and broker.submitted is None


def test_flattens_and_books_at_close():
    broker = FakeBroker(_closes(rising=True))
    state = {"starting_cash": 25_000.0, "realized_pnl": 0.0, "day": None, "trades": [], "entered": True}
    state["day"] = "2026-10-15"
    dr.step(broker, state, _now(20))  # 20 UTC = 16:00 ET -> past FLAT_TIME 15:55
    assert broker.flattened is True and state["booked"] is True
