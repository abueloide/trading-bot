#!/usr/bin/env python3
"""Statistical validation for backtest results.

Three independent, pure tools (no I/O, no run-dir assumptions):
  - monte_carlo_test: permutation test — is the strategy's Sharpe / max-drawdown
    significantly better than a random reordering of the same trades?
  - bootstrap_sharpe_ci: resample returns to get a Sharpe confidence interval
    and prob(Sharpe > 0).
  - walk_forward_analysis: split the equity curve into sequential windows and
    check consistency (profitable windows, Sharpe stability).

Ported from Vibe-Trading's agent/backtest/validation.py, adapted to this repo's
Trade/equity shapes (backtesting.engine.Trade + pd.Series equity curve). Field
names (`pnl`, `entry_time`) happen to match Vibe's TradeRecord, so no renaming
was needed there — only the import source changed.

These are pure functions: call them directly with a backtest's equity_curve
(pd.Series) and closed_trades (List[Trade]) from backtesting.engine.simulate().
Nothing here reads results/*.json or writes files.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Allow `python backtesting/validation.py` from repo root (mirrors run_backtest.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.engine import Trade  # noqa: E402


# ─── Monte Carlo Permutation Test ───


def monte_carlo_test(
    trades: List[Trade],
    initial_capital: float,
    n_simulations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Shuffle trade PnL order to test path significance.

    Null hypothesis: the observed Sharpe / max-drawdown is no better than
    a random ordering of the same trades.

    Args:
        trades: Closed trades from backtesting.engine.simulate() (need .pnl).
        initial_capital: Starting capital for the path being tested.
        n_simulations: Number of random permutations.
        seed: Random seed for reproducibility.

    Returns:
        Dict with actual_sharpe, p_value_sharpe, actual_max_dd,
        p_value_max_dd, simulated_sharpes (percentiles).
    """
    if len(trades) < 3:
        return {"error": "need at least 3 trades", "p_value_sharpe": 1.0}

    pnls = np.array([t.pnl for t in trades])
    actual = _path_metrics(pnls, initial_capital)

    rng = np.random.default_rng(seed)
    sharpe_count = 0
    dd_count = 0
    sim_sharpes = []

    for _ in range(n_simulations):
        shuffled = rng.permutation(pnls)
        sim = _path_metrics(shuffled, initial_capital)
        sim_sharpes.append(sim["sharpe"])
        if sim["sharpe"] >= actual["sharpe"]:
            sharpe_count += 1
        if sim["max_dd"] >= actual["max_dd"]:  # less negative = "better"
            dd_count += 1

    sim_arr = np.array(sim_sharpes)
    return {
        "actual_sharpe": round(actual["sharpe"], 4),
        "actual_max_dd": round(actual["max_dd"], 4),
        "p_value_sharpe": round(sharpe_count / n_simulations, 4),
        "p_value_max_dd": round(dd_count / n_simulations, 4),
        "simulated_sharpe_mean": round(float(sim_arr.mean()), 4),
        "simulated_sharpe_std": round(float(sim_arr.std()), 4),
        "simulated_sharpe_p5": round(float(np.percentile(sim_arr, 5)), 4),
        "simulated_sharpe_p95": round(float(np.percentile(sim_arr, 95)), 4),
        "n_simulations": n_simulations,
        "n_trades": len(trades),
    }


def _path_metrics(pnls: np.ndarray, initial_capital: float) -> Dict[str, float]:
    """Compute Sharpe and max drawdown from a PnL sequence."""
    equity = initial_capital + np.cumsum(pnls)
    returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0.0])
    std = returns.std()
    sharpe = float(returns.mean() / (std + 1e-10) * np.sqrt(252))
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(dd.min())
    return {"sharpe": sharpe, "max_dd": max_dd}


# ─── Bootstrap Sharpe CI ───


def bootstrap_sharpe_ci(
    equity_curve: pd.Series,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    bars_per_year: int = 252,
    seed: int = 42,
) -> Dict[str, Any]:
    """Resample daily returns to estimate a Sharpe confidence interval.

    Args:
        equity_curve: Equity time series (e.g. from engine.simulate()).
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        bars_per_year: Annualisation factor.
        seed: Random seed.

    Returns:
        Dict with observed_sharpe, ci_lower, ci_upper, median_sharpe,
        prob_positive (fraction of bootstrap samples with Sharpe > 0).
    """
    returns = equity_curve.pct_change().dropna().values
    if len(returns) < 5:
        return {"error": "need at least 5 return observations"}

    observed = _sharpe(returns, bars_per_year)

    rng = np.random.default_rng(seed)
    boot_sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(returns, size=len(returns), replace=True)
        boot_sharpes.append(_sharpe(sample, bars_per_year))

    arr = np.array(boot_sharpes)
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(arr, alpha * 100))
    upper = float(np.percentile(arr, (1 - alpha) * 100))
    prob_pos = float(np.mean(arr > 0))

    return {
        "observed_sharpe": round(observed, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "median_sharpe": round(float(np.median(arr)), 4),
        "prob_positive": round(prob_pos, 4),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def _sharpe(returns: np.ndarray, bars_per_year: int = 252) -> float:
    std = returns.std()
    return float(returns.mean() / (std + 1e-10) * np.sqrt(bars_per_year))


# ─── Walk-Forward Analysis ───


def walk_forward_analysis(
    equity_curve: pd.Series,
    trades: List[Trade],
    n_windows: int = 5,
    bars_per_year: int = 252,
) -> Dict[str, Any]:
    """Split a backtest into sequential windows and check consistency.

    Each window is evaluated independently (returns normalised to window start).
    Note: this is a *post-hoc* consistency check on a single backtest run — it is
    unrelated to engine.run_backtest()'s in-sample/out-of-sample walk-forward,
    which fits/evaluates on rolling date ranges before this ever runs.

    Args:
        equity_curve: Equity time series.
        trades: Closed trades (need .entry_time and .pnl).
        n_windows: Number of non-overlapping windows.
        bars_per_year: Annualisation factor.

    Returns:
        Dict with per_window stats, consistency metrics.
    """
    if len(equity_curve) < n_windows * 2:
        return {"error": f"need at least {n_windows * 2} bars for {n_windows} windows"}

    indices = equity_curve.index
    window_size = len(indices) // n_windows
    windows = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size if i < n_windows - 1 else len(indices)
        win_eq = equity_curve.iloc[start_idx:end_idx]
        win_start = indices[start_idx]
        win_end = indices[end_idx - 1]

        # Per-window trades
        win_trades = [t for t in trades if win_start <= t.entry_time <= win_end]

        # Per-window metrics
        ret = float(win_eq.iloc[-1] / win_eq.iloc[0] - 1) if win_eq.iloc[0] > 0 else 0.0
        win_returns = win_eq.pct_change().dropna().values
        sharpe = _sharpe(win_returns, bars_per_year) if len(win_returns) > 1 else 0.0

        peak = win_eq.cummax()
        dd = (win_eq - peak) / peak.replace(0, 1)
        max_dd = float(dd.min())

        win_pnls = [t.pnl for t in win_trades]
        win_rate = len([p for p in win_pnls if p > 0]) / len(win_pnls) if win_pnls else 0.0

        windows.append(
            {
                "window": i + 1,
                "start": str(win_start.date()) if hasattr(win_start, "date") else str(win_start),
                "end": str(win_end.date()) if hasattr(win_end, "date") else str(win_end),
                "return": round(ret, 6),
                "sharpe": round(sharpe, 4),
                "max_dd": round(max_dd, 6),
                "trades": len(win_trades),
                "win_rate": round(win_rate, 4),
            }
        )

    # Consistency metrics
    returns_list = [w["return"] for w in windows]
    sharpes_list = [w["sharpe"] for w in windows]
    profitable_windows = sum(1 for r in returns_list if r > 0)

    return {
        "n_windows": n_windows,
        "windows": windows,
        "profitable_windows": profitable_windows,
        "consistency_rate": round(profitable_windows / n_windows, 4),
        "return_mean": round(float(np.mean(returns_list)), 6),
        "return_std": round(float(np.std(returns_list)), 6),
        "sharpe_mean": round(float(np.mean(sharpes_list)), 4),
        "sharpe_std": round(float(np.std(sharpes_list)), 4),
    }


# ─── Self-check (synthetic data, no fixtures/run-dir needed) ───
#
# IMPORTANT — what monte_carlo_test actually measures: shuffling PRESERVES the
# multiset of pnls, so it can never detect "positive mean vs zero mean" (that
# is exactly what bootstrap_sharpe_ci / prob_positive is for). It only detects
# whether the historical ORDER of trades is special relative to random
# reshuffles — i.e. path/regime dependence (e.g. a choppy period followed by a
# genuine trend). Pure i.i.d. pnls, even with a strongly positive mean, are
# exchangeable with any shuffle, so p_value_sharpe stays high (~uniform) for
# them — that is correct behavior, not a bug. So the two synthetic cases below
# are built to actually exercise that distinction:
#   Case A: a regime shift (choppy period -> sustained trend) — genuine order
#           dependence, so monte_carlo_test should give a LOW p_value_sharpe;
#           bootstrap/walk-forward should also show a strong, consistent edge.
#   Case B: pure zero-mean i.i.d. noise — no order dependence (p_value_sharpe
#           stays high) and no directional edge (prob_positive ~0.5).

if __name__ == "__main__":
    from datetime import datetime, timedelta

    def _make_trades(pnls: List[float], start: datetime) -> List[Trade]:
        out = []
        for i, pnl in enumerate(pnls):
            t = Trade(
                symbol="TEST",
                entry_time=start + timedelta(days=i),
                entry_price=100.0,
                qty=10.0,
                strategy="synthetic",
                strategy_type="test",
            )
            t.exit_time = start + timedelta(days=i + 1)
            t.exit_price = 100.0 + pnl / 10.0
            t.exit_reason = "signal_exit"
            t.pnl = pnl
            t.pnl_pct = pnl / 1000.0 * 100
            t.hold_days = 1
            out.append(t)
        return out

    def _make_equity(pnls: List[float], initial_capital: float, start: datetime) -> pd.Series:
        equity = initial_capital + np.cumsum(pnls)
        idx = pd.date_range(start=start, periods=len(equity), freq="D")
        return pd.Series(equity, index=idx, name="equity")

    start = datetime(2024, 1, 1)
    rng = np.random.default_rng(3)

    print("=" * 70)
    print("Case A: regime shift — choppy period, then a sustained trend")
    print("=" * 70)
    choppy = rng.normal(20, 15, 60)
    trend = rng.normal(80, 10, 60)
    edge_pnls = list(np.concatenate([choppy, trend]))
    edge_trades = _make_trades(edge_pnls, start)
    edge_equity = _make_equity(edge_pnls, 5_000, start)

    mc_edge = monte_carlo_test(edge_trades, initial_capital=5_000, n_simulations=1000)
    boot_edge = bootstrap_sharpe_ci(edge_equity, n_bootstrap=1000)
    wf_edge = walk_forward_analysis(edge_equity, edge_trades, n_windows=4)

    print("monte_carlo_test:", mc_edge)
    print("bootstrap_sharpe_ci:", boot_edge)
    print("walk_forward_analysis: consistency_rate=", wf_edge["consistency_rate"],
          "sharpe_mean=", wf_edge["sharpe_mean"])
    assert mc_edge["p_value_sharpe"] < 0.05, "regime-shift order should look unusual vs random shuffles"
    assert boot_edge["prob_positive"] > 0.9, "consistently positive pnls should give high prob_positive"
    assert wf_edge["consistency_rate"] >= 0.75, "trending equity should be profitable in most windows"

    print()
    print("=" * 70)
    print("Case B: pure zero-mean i.i.d. noise — no edge, no order dependence")
    print("=" * 70)
    random_pnls = list(rng.normal(0, 100, size=120))
    random_trades = _make_trades(random_pnls, start)
    random_equity = _make_equity(random_pnls, 100_000, start)

    mc_rand = monte_carlo_test(random_trades, initial_capital=100_000, n_simulations=1000)
    boot_rand = bootstrap_sharpe_ci(random_equity, n_bootstrap=1000)
    wf_rand = walk_forward_analysis(random_equity, random_trades, n_windows=4)

    print("monte_carlo_test:", mc_rand)
    print("bootstrap_sharpe_ci:", boot_rand)
    print("walk_forward_analysis: consistency_rate=", wf_rand["consistency_rate"],
          "sharpe_mean=", wf_rand["sharpe_mean"])
    assert mc_rand["p_value_sharpe"] > 0.05, "i.i.d. noise order should NOT look unusual vs random shuffles"
    assert 0.3 < boot_rand["prob_positive"] < 0.7, "zero-mean noise prob_positive should be ~0.5"

    print()
    print("All self-checks passed.")
