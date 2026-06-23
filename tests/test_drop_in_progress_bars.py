"""drop_in_progress_bars: settled-close discipline for the mid-session cron.

Regression guard for the 2026-06-22 taint — the 13:00 CST run stamped the daily
snapshot on a partial intraday bar, inflating momentum's return/alpha before the
go/no-go verdict went live.
"""
from datetime import date

import pandas as pd

from live.yfinance_bars import drop_in_progress_bars


def _df(dates, closes):
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1] * len(closes)},
        index=idx,
    )


def test_drops_today_partial_bar():
    today = date(2026, 6, 22)
    snap = {"AAA": _df(["2026-06-18", "2026-06-22"], [100.0, 120.0])}
    out = drop_in_progress_bars(snap, today)
    assert out["AAA"].index[-1].date() == date(2026, 6, 18)
    assert float(out["AAA"]["close"].iloc[-1]) == 100.0


def test_settled_snapshot_unchanged():
    # Last bar predates `today` (the normal 1-day lag) -> nothing dropped.
    today = date(2026, 6, 23)
    snap = {"AAA": _df(["2026-06-18", "2026-06-22"], [100.0, 120.0])}
    out = drop_in_progress_bars(snap, today)
    assert len(out["AAA"]) == 2
    assert out["AAA"].index[-1].date() == date(2026, 6, 22)


def test_symbol_emptied_by_drop_is_omitted():
    today = date(2026, 6, 22)
    snap = {"AAA": _df(["2026-06-22"], [120.0])}
    out = drop_in_progress_bars(snap, today)
    assert "AAA" not in out


def test_mixed_universe_only_today_bar_trimmed():
    today = date(2026, 6, 22)
    snap = {
        "AAA": _df(["2026-06-18", "2026-06-22"], [10.0, 11.0]),   # has partial
        "BBB": _df(["2026-06-17", "2026-06-18"], [20.0, 21.0]),   # already settled
    }
    out = drop_in_progress_bars(snap, today)
    assert out["AAA"].index[-1].date() == date(2026, 6, 18)
    assert len(out["BBB"]) == 2  # untouched
