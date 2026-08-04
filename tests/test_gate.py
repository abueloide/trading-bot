"""Backtest-first gate: outlier mirage must be vetoed, broad edge must pass."""
from backtesting.gate import evaluate


def test_outlier_mirage_is_vetoed():
    # One huge winner (NVDA-style), rest negative -> FAIL on median + breadth.
    mirage = {
        "SPY": {"excess_return_pct": -20.2, "sharpe": 1.02, "n_trades": 13},
        "AAPL": {"excess_return_pct": -21.3, "sharpe": 0.61, "n_trades": 16},
        "MSFT": {"excess_return_pct": -19.2, "sharpe": 0.68, "n_trades": 10},
        "QQQ": {"excess_return_pct": -12.7, "sharpe": 1.24, "n_trades": 3},
        "NVDA": {"excess_return_pct": 118.7, "sharpe": 1.56, "n_trades": 5},
    }
    assert evaluate(mirage)["passed"] is False


def test_broad_positive_edge_passes():
    good = {s: {"excess_return_pct": 8.0, "sharpe": 1.1, "n_trades": 12}
            for s in "SPY AAPL MSFT QQQ NVDA".split()}
    assert evaluate(good)["passed"] is True


def test_thin_universe_fails():
    assert evaluate({"SPY": {"excess_return_pct": 8.0, "sharpe": 1.1, "n_trades": 12}})["passed"] is False
