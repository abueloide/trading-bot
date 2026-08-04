"""Guard the benchmark wiring: the alpha column silently went dark in prod.

SPY (the buy&hold yardstick) is NOT an S&P 500 constituent, so it was never in
the fetched universe — ``compute_benchmark`` returned None every run and every
equity-curve snapshot carried ``alpha_pct: null``. The whole point of the race
is alpha vs the market, so these invariants must hold:

  1. The benchmark symbol IS fetched (so alpha can be computed).
  2. The benchmark symbol is NOT a strategy symbol (so no horse trades it).
"""
from __future__ import annotations

import run_trading_system as rts


def test_benchmark_symbol_is_in_the_fetch_list():
    assert rts.BENCHMARK_SYMBOL in rts.FETCH_SYMBOLS


def test_rebalance_reference_is_in_the_fetch_list():
    # The monthly-rebalance calendar anchor must be fetched too, else it falls
    # back to an arbitrary symbol's calendar.
    assert rts.REBALANCE_REFERENCE in rts.FETCH_SYMBOLS


def test_benchmark_symbol_is_not_tradeable():
    # No strategy may carry the benchmark symbol in its universe — it is a
    # yardstick, not a position.
    for cfg in rts.STRATEGIES:
        assert rts.BENCHMARK_SYMBOL not in cfg.symbols


def test_fetch_list_has_no_duplicates():
    assert len(rts.FETCH_SYMBOLS) == len(set(rts.FETCH_SYMBOLS))


def test_fetch_list_is_universe_plus_benchmark():
    # Every tradeable name is still fetched; the only addition is the yardstick.
    assert set(rts.UNIVERSE).issubset(set(rts.FETCH_SYMBOLS))


def test_every_strategy_symbol_is_fetched():
    # Regresión: una estrategia con universo propio (opex_drift → IVV/QQQ) debe
    # tener barras. Si no se fetchea, nunca opera y falla en silencio.
    fetched = set(rts.FETCH_SYMBOLS)
    for cfg in rts.STRATEGIES:
        missing = set(cfg.symbols) - fetched
        assert not missing, f"{cfg.strategy} operaría {missing} sin barras"
