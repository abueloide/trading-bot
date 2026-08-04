"""Regression: untradeable-at-broker symbols stay out of the traded universe.

SATS (EchoStar) sits in the static S&P 500 snapshot and yfinance prices it, but
Alpaca returns `asset "SATS" not found`. Before the fix, a horse that selected it
saw the BUY fail graceful and leaked the slot to cash, biasing its return. The
exclusion must apply to BOTH the symbol list and the sector map (single source),
or the two views drift apart.
"""
from stock_universe import (
    UNTRADEABLE_AT_BROKER,
    sp500_constituents,
    sp500_sector_map,
    sp500_symbols,
)


def test_sats_is_excluded_from_symbols():
    assert "SATS" in UNTRADEABLE_AT_BROKER  # guards the offender stays pinned
    assert "SATS" not in sp500_symbols()


def test_excluded_symbols_absent_everywhere():
    symbols = set(sp500_symbols())
    sectors = set(sp500_sector_map())
    rows = {row["symbol"] for row in sp500_constituents()}
    for sym in UNTRADEABLE_AT_BROKER:
        assert sym not in symbols
        assert sym not in sectors
        assert sym not in rows


def test_universe_still_substantial():
    # Sanity: dropping a handful must not gut the universe.
    assert len(sp500_symbols()) > 400
