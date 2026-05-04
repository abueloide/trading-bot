#!/usr/bin/env python3
"""
Backtesting engine — walk-forward validation with transaction costs.

Design principles:
- Transaction costs ALWAYS modeled at 0.05% slippage per side. No exceptions.
- Walk-forward: 24-month in-sample, 6-month out-of-sample, rolling.
- Mean-reversion strategies use TIME-BASED exits (max_hold_days), no SL.
- Always benchmark against SPY buy-and-hold over the same window.
- Models alternative-data delays: congress 45d, insider 2d (passed via series).

Data sources:
- Primary:  Alpaca historical bars (StockHistoricalDataClient.get_stock_bars)
- Fallback: yfinance (offline-friendly when Alpaca creds absent)

Output:
- ``results/<strategy>_<run_id>.json`` — summary metrics
- ``results/<strategy>_<run_id>_equity.csv`` — equity curve
- ``results/<strategy>_<run_id>_trades.csv`` — trade log
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from config import BACKTESTING_CONFIG
except ImportError:
    BACKTESTING_CONFIG = {
        "transaction_cost_pct": 0.0005,
        "in_sample_months": 24,
        "out_of_sample_months": 6,
        "results_dir": "backtesting/results",
        "baseline_symbol": "SPY",
        "data_source": "alpaca",
        "fallback_data_source": "yfinance",
    }


SLIPPAGE = float(BACKTESTING_CONFIG.get("transaction_cost_pct", 0.0005))
RESULTS_DIR = Path(BACKTESTING_CONFIG.get("results_dir", "backtesting/results"))
BASELINE = BACKTESTING_CONFIG.get("baseline_symbol", "SPY")


# =============================================================================
# Data loading
# =============================================================================

def load_bars(
    symbol: str,
    start: date,
    end: date,
    timeframe: str = "1d",
    source: str = "alpaca",
) -> Optional[pd.DataFrame]:
    """Load OHLCV bars for a symbol over [start, end].

    Returns a DataFrame indexed by date with columns open, high, low, close,
    volume (lowercase). Returns None on failure.
    """
    if source == "alpaca":
        df = _load_bars_alpaca(symbol, start, end, timeframe)
        if df is not None and not df.empty:
            return df
        logger.info(f"Alpaca data unavailable for {symbol}; falling back to yfinance")
    return _load_bars_yfinance(symbol, start, end, timeframe)


def _load_bars_alpaca(
    symbol: str, start: date, end: date, timeframe: str
) -> Optional[pd.DataFrame]:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        return None

    api_key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        return None

    tf = TimeFrame.Day if timeframe == "1d" else TimeFrame.Hour
    try:
        client = StockHistoricalDataClient(api_key, secret)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=datetime.combine(start, datetime.min.time()),
            end=datetime.combine(end, datetime.min.time()),
        )
        bars = client.get_stock_bars(req)
        if not hasattr(bars, "df"):
            return None
        df = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol]
        df = df.set_index(pd.to_datetime(df["timestamp"]).dt.tz_localize(None))
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        return df
    except Exception as e:
        logger.warning(f"Alpaca load failed for {symbol}: {e}")
        return None


def _load_bars_yfinance(
    symbol: str, start: date, end: date, timeframe: str
) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError:
        return None
    interval = {"1d": "1d", "1h": "1h"}.get(timeframe, "1d")
    try:
        df = yf.download(
            symbol, start=start, end=end, interval=interval,
            progress=False, auto_adjust=True,
        )
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        # yfinance multi-symbol returns multi-index columns; take symbol slice
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz is not None else pd.to_datetime(df.index)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.warning(f"yfinance load failed for {symbol}: {e}")
        return None


# =============================================================================
# Portfolio simulation
# =============================================================================

@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    entry_price: float
    qty: float
    strategy: str
    strategy_type: str
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_days: int = 0


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    start: str
    end: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    avg_hold_days: float
    n_trades: int
    benchmark_return_pct: float
    excess_return_pct: float
    transaction_cost_pct: float
    trades: List[Dict] = field(default_factory=list)


def _compute_metrics(equity: pd.Series, trades: List[Trade], baseline_return: float) -> Dict[str, float]:
    if len(equity) < 2:
        return {
            "total_return_pct": 0.0, "cagr_pct": 0.0, "sharpe": 0.0,
            "sortino": 0.0, "max_drawdown_pct": 0.0, "win_rate_pct": 0.0,
            "profit_factor": 0.0, "avg_hold_days": 0.0,
        }

    daily_ret = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(years, 1e-9)) - 1

    if daily_ret.std(ddof=0) > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std(ddof=0) * np.sqrt(252))
    else:
        sharpe = 0.0

    downside = daily_ret[daily_ret < 0]
    if len(downside) > 0 and downside.std(ddof=0) > 0:
        sortino = float(daily_ret.mean() / downside.std(ddof=0) * np.sqrt(252))
    else:
        sortino = 0.0

    running_max = equity.cummax()
    drawdown = (equity / running_max - 1)
    max_dd = float(drawdown.min())

    closed = [t for t in trades if t.exit_time is not None]
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl < 0]
    win_rate = (len(wins) / len(closed) * 100) if closed else 0.0
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (np.inf if gross_win > 0 else 0.0)
    avg_hold = float(np.mean([t.hold_days for t in closed])) if closed else 0.0

    return {
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd * 100,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor if not np.isinf(profit_factor) else 999.0,
        "avg_hold_days": avg_hold,
    }


def simulate(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    symbol: str,
    strategy: str,
    strategy_type: str,
    initial_capital: float = 10_000,
    position_size_pct: float = 0.25,
    max_open_positions: int = 4,
    max_hold_days: Optional[int] = None,
    slippage: float = SLIPPAGE,
) -> tuple[pd.Series, List[Trade]]:
    """Run a single-symbol simulation. Returns (equity_curve, trades)."""
    cash = initial_capital
    positions: List[Trade] = []
    closed_trades: List[Trade] = []
    equity_records: List[tuple] = []

    df = df.copy()
    df["date"] = df.index
    sig = signals.reindex(df.index).fillna(False)

    for ts, row in df.iterrows():
        price = float(row["close"])

        # 1) Check exits on existing positions before re-entering
        still_open: List[Trade] = []
        for pos in positions:
            should_exit = False
            reason = None

            # Strategy-driven exit
            if bool(sig.loc[ts, "exit"]) if "exit" in sig.columns else False:
                should_exit = True
                reason = "signal_exit"

            # Time-based exit (mean_reversion / momentum)
            if max_hold_days is not None and not should_exit:
                held = (ts - pos.entry_time).days
                if held >= max_hold_days:
                    should_exit = True
                    reason = "time_exit"

            if should_exit:
                exit_price = price * (1 - slippage)  # sell takes slippage
                proceeds = pos.qty * exit_price
                cash += proceeds
                pos.exit_time = ts
                pos.exit_price = exit_price
                pos.exit_reason = reason
                pos.pnl = proceeds - (pos.qty * pos.entry_price)
                pos.pnl_pct = (exit_price / pos.entry_price - 1) * 100
                pos.hold_days = (ts - pos.entry_time).days
                closed_trades.append(pos)
            else:
                still_open.append(pos)
        positions = still_open

        # 2) Entry on signal if room
        if (
            "entry" in sig.columns
            and bool(sig.loc[ts, "entry"])
            and len(positions) < max_open_positions
            and not any(p.symbol == symbol for p in positions)
        ):
            target_dollars = (cash + sum(p.qty * price for p in positions)) * position_size_pct
            target_dollars = min(target_dollars, cash)
            if target_dollars > 0 and price > 0:
                entry_price = price * (1 + slippage)  # buy takes slippage
                qty = target_dollars / entry_price
                cost = qty * entry_price
                if cost <= cash:
                    cash -= cost
                    positions.append(Trade(
                        symbol=symbol,
                        entry_time=ts,
                        entry_price=entry_price,
                        qty=qty,
                        strategy=strategy,
                        strategy_type=strategy_type,
                    ))

        equity = cash + sum(p.qty * price for p in positions)
        equity_records.append((ts, equity))

    # Force-close any open positions at last bar
    if positions and len(df):
        last_ts = df.index[-1]
        last_price = float(df.iloc[-1]["close"])
        for pos in positions:
            exit_price = last_price * (1 - slippage)
            proceeds = pos.qty * exit_price
            cash += proceeds
            pos.exit_time = last_ts
            pos.exit_price = exit_price
            pos.exit_reason = "end_of_period"
            pos.pnl = proceeds - (pos.qty * pos.entry_price)
            pos.pnl_pct = (exit_price / pos.entry_price - 1) * 100
            pos.hold_days = (last_ts - pos.entry_time).days
            closed_trades.append(pos)
        # Replace last equity record with cash-only post-liquidation
        equity_records[-1] = (last_ts, cash)

    equity_curve = pd.Series(
        [e for _, e in equity_records], index=[t for t, _ in equity_records], name="equity"
    )
    return equity_curve, closed_trades


# =============================================================================
# Walk-forward orchestrator
# =============================================================================

def run_backtest(
    *,
    strategy_name: str,
    strategy_fn: Callable[..., pd.DataFrame],
    strategy_type: str,
    max_hold_days: Optional[int],
    symbols: List[str],
    start: date,
    end: date,
    initial_capital: float = 10_000,
    position_size_pct: float = 0.25,
    walk_forward: bool = True,
    in_sample_months: int = 24,
    oos_months: int = 6,
    data_source: str = "alpaca",
    extra_data: Optional[Dict[str, Any]] = None,
    save: bool = True,
) -> Dict[str, BacktestResult]:
    """Run backtest across symbols. Returns dict mapping symbol -> BacktestResult.

    When ``walk_forward`` is True, each symbol is evaluated only on its
    rolling out-of-sample windows (the in-sample window is used for fitting if
    a strategy supports parameters, else just held out for reproducibility).
    """
    extra_data = extra_data or {}

    # Always load baseline once for the full window for benchmarking.
    baseline_df = load_bars(BASELINE, start, end, source=data_source)
    if baseline_df is None or baseline_df.empty:
        logger.warning("Baseline data unavailable — benchmark will be 0%")
        baseline_return = 0.0
    else:
        baseline_return = float(baseline_df["close"].iloc[-1] / baseline_df["close"].iloc[0] - 1)

    spy_close = baseline_df["close"] if baseline_df is not None and not baseline_df.empty else None

    results: Dict[str, BacktestResult] = {}
    run_id = uuid.uuid4().hex[:8]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        df = load_bars(symbol, start, end, source=data_source)
        if df is None or df.empty or len(df) < 50:
            logger.warning(f"Skipping {symbol}: insufficient data")
            continue

        windows = _walk_forward_windows(df.index, in_sample_months, oos_months) if walk_forward else [(df.index[0], df.index[-1])]

        equity_chunks: List[pd.Series] = []
        trades_total: List[Trade] = []
        running_equity = initial_capital

        for win_start, win_end in windows:
            sub = df.loc[win_start:win_end]
            if len(sub) < 30:
                continue
            try:
                params: Dict[str, Any] = {}
                if strategy_name in {"rsi_mr", "confirmed_mr"} and spy_close is not None:
                    params["spy_close"] = spy_close
                if strategy_name == "rsi_mr" and "vix_rank" in extra_data:
                    params["vix_rank_series"] = extra_data["vix_rank"]
                signals = strategy_fn(sub, **params)
            except Exception as e:
                logger.error(f"Strategy {strategy_name} crashed on {symbol}: {e}")
                continue

            equity, trades = simulate(
                sub, signals,
                symbol=symbol, strategy=strategy_name, strategy_type=strategy_type,
                initial_capital=running_equity,
                position_size_pct=position_size_pct,
                max_hold_days=max_hold_days,
            )
            if not equity.empty:
                running_equity = float(equity.iloc[-1])
                equity_chunks.append(equity)
                trades_total.extend(trades)

        if not equity_chunks:
            continue

        equity_full = pd.concat(equity_chunks)
        equity_full = equity_full[~equity_full.index.duplicated(keep="last")]

        metrics = _compute_metrics(equity_full, trades_total, baseline_return)
        result = BacktestResult(
            strategy=strategy_name,
            symbol=symbol,
            start=str(start),
            end=str(end),
            initial_capital=initial_capital,
            final_equity=float(equity_full.iloc[-1]),
            n_trades=len(trades_total),
            benchmark_return_pct=baseline_return * 100,
            excess_return_pct=metrics["total_return_pct"] - baseline_return * 100,
            transaction_cost_pct=SLIPPAGE * 100,
            trades=[asdict(t) for t in trades_total],
            **metrics,
        )
        results[symbol] = result

        if save:
            base = RESULTS_DIR / f"{strategy_name}_{symbol}_{run_id}"
            with open(f"{base}.json", "w") as f:
                json.dump({k: v for k, v in asdict(result).items() if k != "trades"}, f, indent=2, default=str)
            equity_full.to_frame().to_csv(f"{base}_equity.csv")
            if trades_total:
                pd.DataFrame([asdict(t) for t in trades_total]).to_csv(f"{base}_trades.csv", index=False)

    if save and results:
        summary = {
            "run_id": run_id,
            "strategy": strategy_name,
            "transaction_cost_pct": SLIPPAGE * 100,
            "baseline_return_pct": baseline_return * 100,
            "per_symbol": {
                s: {k: v for k, v in asdict(r).items() if k != "trades"}
                for s, r in results.items()
            },
        }
        with open(RESULTS_DIR / f"{strategy_name}_summary_{run_id}.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Backtest results saved to {RESULTS_DIR} (run_id={run_id})")

    return results


def _walk_forward_windows(
    index: pd.DatetimeIndex, in_sample_months: int, oos_months: int
) -> List[tuple]:
    """Generate rolling (in_sample_start, oos_end) windows over the index."""
    windows = []
    if len(index) < 30:
        return windows
    start = index[0]
    end = index[-1]
    cursor = start + pd.DateOffset(months=in_sample_months)
    while cursor < end:
        oos_end = min(cursor + pd.DateOffset(months=oos_months), end)
        windows.append((cursor, oos_end))
        cursor = oos_end
    if not windows:
        windows.append((start, end))
    return windows
