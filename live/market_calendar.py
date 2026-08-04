"""NYSE trading-day helpers — weekday alone over-counts the trading calendar.

The gap and staleness guards in ``risk_metrics`` / ``checkpoint_report`` count
the days a healthy L–V curve *should* have. Counting Mon–Fri treats a weekday
market holiday (Juneteenth, Good Friday, …) as a skipped cron run, firing a
false ``⚠ GAP`` / ``⚠ STALE`` on the one readout that feeds the irreversible
money decision — cry-wolf that trains the operator to ignore a real guard.

pandas' ``USFederalHolidayCalendar`` is the wrong source: it counts Columbus
and Veterans Day (NYSE is OPEN) and omits Good Friday (NYSE is CLOSED), so it
would *under-warn* on a real skipped run. So we keep an explicit, dated NYSE
holiday set instead of a dependency or a wrong calendar.

ponytail: holidays listed through 2026 only (covers the paper window, inception
2026-06-05 → checkpoint ~06-30). Extend NYSE_HOLIDAYS by year if the experiment
runs longer; a missing year degrades to weekday-only counting (the old, slightly
noisier behaviour), never to a wrong trade.
"""
from __future__ import annotations

from datetime import date

# NYSE full-day closures. Half-days (early closes) still trade, so they are NOT
# holidays here. Source: NYSE published holiday calendar.
NYSE_HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2026
        date(2026, 1, 1),    # New Year's Day
        date(2026, 1, 19),   # MLK Jr. Day
        date(2026, 2, 16),   # Washington's Birthday
        date(2026, 4, 3),    # Good Friday
        date(2026, 5, 25),   # Memorial Day
        date(2026, 6, 19),   # Juneteenth
        date(2026, 7, 3),    # Independence Day (observed)
        date(2026, 9, 7),    # Labor Day
        date(2026, 11, 26),  # Thanksgiving
        date(2026, 12, 25),  # Christmas
    }
)


def is_trading_day(d: date) -> bool:
    """True if ``d`` is a NYSE trading day (a weekday that isn't a full closure)."""
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


if __name__ == "__main__":  # ponytail: smallest check that the holiday gate bites
    assert is_trading_day(date(2026, 6, 18))   # Thu, normal
    assert not is_trading_day(date(2026, 6, 19))  # Juneteenth (the live false GAP)
    assert not is_trading_day(date(2026, 6, 20))  # Sat
    assert is_trading_day(date(2026, 6, 22))   # Mon, normal
    assert not is_trading_day(date(2026, 4, 3))   # Good Friday (federal cal misses it)
    print("market_calendar self-check OK")
