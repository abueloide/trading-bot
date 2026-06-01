# Paper Trading "Horse Race" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the three backtested strategies live in parallel on one Alpaca **paper** account, each with its own virtual ledger, so we can watch which actually performs on out-of-sample data — with zero real money.

**Architecture:** A thin `live/` package. `StrategyRunner` turns a backtest strategy fn into a per-bar live signal. `VirtualPortfolio` is a per-strategy ledger (cash, positions, realized P&L) that produces a `RiskManager.PortfolioState`. `Orchestrator` wires signal → risk → executor each cycle, depending on small Protocols (`BarProvider`, `ExecutorPort`) so it is fully testable offline. A `LiveExecutorAdapter` (wired only at the end) maps `ExecutorPort` onto the existing `executor.py`. The existing `risk_manager`, `executor`, `trade_journal`, `backtesting/strategies` are reused unchanged.

**Tech Stack:** Python 3.14, pandas/numpy, alpaca-py (live run only), pytest. Reuses existing repo modules.

---

## File Structure

**Create:**
- `live/__init__.py` — package marker
- `live/strategy_runner.py` — `Signal`, `StrategyRunner`
- `live/virtual_portfolio.py` — `VirtualPortfolio`
- `live/ports.py` — `BarProvider`, `ExecutorPort` Protocols
- `live/orchestrator.py` — `Orchestrator`, `StrategyConfig`
- `live/live_executor_adapter.py` — `LiveExecutorAdapter` (maps ExecutorPort → real executor.py)
- `live/horse_race_report.py` — `build_report`
- `tests/__init__.py`, `tests/conftest.py`
- `tests/test_strategy_runner.py`, `tests/test_virtual_portfolio.py`, `tests/test_orchestrator.py`, `tests/test_horse_race_report.py`

**Modify:**
- `run_trading_system.py` — replace the dead loop with an `Orchestrator` run
- `requirements-dev.txt` (create) — pytest

**Reuse unchanged:** `risk_manager.py`, `executor.py`, `trade_journal.py`, `backtesting/strategies.py`, `enhanced_alpaca_client.py`.

All commands assume repo root `~/Documents/proyectos-ia/Bots/Trading/trading-bot` and the venv `.venv-bt` created during the audit.

---

## Task 0: Dev environment + branch + test scaffold

**Files:**
- Create: `requirements-dev.txt`, `tests/__init__.py`, `tests/conftest.py`, `live/__init__.py`, `pytest.ini`

- [ ] **Step 1: Create the working branch**

```bash
git checkout -b feature/paper-horse-race
git add backtesting/strategies.py docs/superpowers/specs/2026-05-31-paper-horse-race-design.md
git commit -m "fix: pandas 3.0 ffill in backtest strategies; add paper horse-race spec"
```

- [ ] **Step 2: Install test deps into the existing venv**

Run:
```bash
.venv-bt/bin/pip install pytest
printf 'pytest>=8.0\n' > requirements-dev.txt
```
Expected: pytest installs without error.

- [ ] **Step 3: Create package + test scaffolding**

`live/__init__.py`:
```python
"""Live paper-trading layer: strategy runner, virtual ledgers, orchestrator."""
```

`tests/__init__.py`:
```python
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

`tests/conftest.py`:
```python
"""Shared fixtures: synthetic OHLCV bars for offline strategy/ledger tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _bars(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(closes, index=idx, dtype="float64")
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": pd.Series(1_000_000, index=idx, dtype="float64"),
        }
    )


@pytest.fixture
def rising_bars() -> pd.DataFrame:
    # 260 strictly rising bars — long enough for the 200d / momentum filters.
    return _bars([100.0 + i for i in range(260)])


@pytest.fixture
def oversold_then_bars() -> pd.DataFrame:
    # Long uptrend, then a sharp 5-day drop to force RSI(2) oversold on the last bar.
    base = [100.0 + i * 0.5 for i in range(255)]
    drop = [base[-1] * f for f in (0.96, 0.92, 0.88, 0.85, 0.82)]
    return _bars(base + drop)
```

- [ ] **Step 4: Verify the scaffold runs (zero tests collected is OK)**

Run: `.venv-bt/bin/python -m pytest -q`
Expected: `no tests ran` (exit 5) or `0 passed` — no import/collection errors.

- [ ] **Step 5: Commit**

```bash
git add live/__init__.py tests/__init__.py tests/conftest.py pytest.ini requirements-dev.txt
git commit -m "chore: pytest scaffold + synthetic bar fixtures for live layer"
```

---

## Task 1: StrategyRunner

Turns a backtest strategy fn into a per-bar live signal by reading the **last row's** `entry`/`exit`.

**Files:**
- Create: `live/strategy_runner.py`
- Test: `tests/test_strategy_runner.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_strategy_runner.py`:
```python
from __future__ import annotations

import pytest

from live.strategy_runner import Signal, StrategyRunner


def test_unknown_strategy_raises():
    with pytest.raises(KeyError):
        StrategyRunner("does_not_exist")


def test_hold_when_no_entry_or_exit(rising_bars):
    # confirmed_mr needs an oversold candle to enter; a pure uptrend → HOLD.
    runner = StrategyRunner("confirmed_mr")
    sig = runner.run("AAPL", rising_bars)
    assert isinstance(sig, Signal)
    assert sig.action == "HOLD"
    assert sig.symbol == "AAPL"
    assert sig.price == pytest.approx(float(rising_bars["close"].iloc[-1]))


def test_buy_on_oversold_last_bar(oversold_then_bars):
    # confirmed_mr: RSI(2) < 15 on the deep-drop last bar → BUY (SPY filter optional).
    runner = StrategyRunner("confirmed_mr")
    sig = runner.run("AAPL", oversold_then_bars)
    assert sig.action == "BUY"


def test_too_few_bars_is_hold():
    import pandas as pd
    df = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2],
                       "close": [1.0, 2.0], "volume": [1, 1]})
    runner = StrategyRunner("rsi_mr")
    assert runner.run("AAPL", df).action == "HOLD"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-bt/bin/python -m pytest tests/test_strategy_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'live.strategy_runner'`.

- [ ] **Step 3: Implement `live/strategy_runner.py`**

```python
"""StrategyRunner — wraps a backtest strategy fn as a per-bar live signal.

The backtest contract (see backtesting/strategies.py) is:
    fn(df, **params) -> DataFrame[index, "entry": bool, "exit": bool]
Live trading only cares about the LAST bar: if its `entry` is True → BUY,
elif its `exit` is True → SELL, else HOLD.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from backtesting.strategies import STRATEGY_REGISTRY

Action = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class Signal:
    action: Action
    price: float
    symbol: str
    strategy: str


class StrategyRunner:
    """Runs one registered strategy against a trailing window of bars."""

    def __init__(self, strategy_name: str) -> None:
        if strategy_name not in STRATEGY_REGISTRY:
            raise KeyError(
                f"Unknown strategy {strategy_name!r}. "
                f"Available: {list(STRATEGY_REGISTRY)}"
            )
        spec = STRATEGY_REGISTRY[strategy_name]
        self.name = strategy_name
        self._fn = spec["fn"]
        self.strategy_type = str(spec["type"])
        self.max_hold_days = spec.get("max_hold_days")

    def run(self, symbol: str, bars: pd.DataFrame) -> Signal:
        price = float(bars["close"].iloc[-1]) if len(bars) else 0.0
        try:
            signals = self._fn(bars)
        except Exception:
            return Signal("HOLD", price, symbol, self.name)
        if signals is None or len(signals) == 0:
            return Signal("HOLD", price, symbol, self.name)
        last = signals.iloc[-1]
        if bool(last.get("entry", False)):
            return Signal("BUY", price, symbol, self.name)
        if bool(last.get("exit", False)):
            return Signal("SELL", price, symbol, self.name)
        return Signal("HOLD", price, symbol, self.name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-bt/bin/python -m pytest tests/test_strategy_runner.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add live/strategy_runner.py tests/test_strategy_runner.py
git commit -m "feat(live): StrategyRunner — backtest fn to per-bar live signal"
```

---

## Task 2: VirtualPortfolio

Per-strategy ledger. Critical invariant: a SELL only disposes of shares THIS strategy bought.

**Files:**
- Create: `live/virtual_portfolio.py`
- Test: `tests/test_virtual_portfolio.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_virtual_portfolio.py`:
```python
from __future__ import annotations

import pytest

from live.virtual_portfolio import VirtualPortfolio


def test_buy_reduces_cash_and_adds_position():
    vp = VirtualPortfolio("momentum_rotation", starting_cash=1000.0)
    vp.record_buy("AAPL", qty=2.0, price=100.0)
    assert vp.cash == pytest.approx(800.0)
    assert vp.qty("AAPL") == pytest.approx(2.0)
    assert vp.avg_entry("AAPL") == pytest.approx(100.0)


def test_buy_averages_entry_price():
    vp = VirtualPortfolio("s", starting_cash=1000.0)
    vp.record_buy("AAPL", qty=2.0, price=100.0)
    vp.record_buy("AAPL", qty=2.0, price=120.0)
    assert vp.qty("AAPL") == pytest.approx(4.0)
    assert vp.avg_entry("AAPL") == pytest.approx(110.0)


def test_sell_books_realized_pnl_and_reduces_qty():
    vp = VirtualPortfolio("s", starting_cash=1000.0)
    vp.record_buy("AAPL", qty=4.0, price=100.0)  # cash 600
    vp.record_sell("AAPL", qty=2.0, price=130.0)  # +260 cash, +60 realized
    assert vp.qty("AAPL") == pytest.approx(2.0)
    assert vp.cash == pytest.approx(860.0)
    assert vp.realized_pnl == pytest.approx(60.0)


def test_sell_more_than_owned_is_rejected():
    vp = VirtualPortfolio("s", starting_cash=1000.0)
    vp.record_buy("AAPL", qty=1.0, price=100.0)
    with pytest.raises(ValueError):
        vp.record_sell("AAPL", qty=2.0, price=100.0)


def test_to_portfolio_state_reflects_slice():
    vp = VirtualPortfolio("s", starting_cash=1000.0)
    vp.record_buy("AAPL", qty=2.0, price=100.0)
    state = vp.to_portfolio_state(marks={"AAPL": 110.0})
    assert state.cash == pytest.approx(800.0)
    # equity = cash + marked positions = 800 + 2*110
    assert state.equity == pytest.approx(1020.0)
    assert len(state.positions) == 1
    assert state.positions[0]["symbol"] == "AAPL"
    assert state.positions[0]["market_value"] == pytest.approx(220.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-bt/bin/python -m pytest tests/test_virtual_portfolio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'live.virtual_portfolio'`.

- [ ] **Step 3: Implement `live/virtual_portfolio.py`**

```python
"""VirtualPortfolio — per-strategy ledger over a shared paper account.

Tracks the cash slice, per-symbol qty/avg-entry, and realized P&L for ONE
strategy. The real Alpaca account nets all strategies together; this ledger
keeps them separate so a SELL only ever disposes of shares THIS strategy
bought. Emits a RiskManager.PortfolioState for position sizing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from risk_manager import PortfolioState


@dataclass
class _Lot:
    qty: float = 0.0
    avg_entry: float = 0.0


class VirtualPortfolio:
    def __init__(self, strategy: str, starting_cash: float) -> None:
        self.strategy = strategy
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.realized_pnl = 0.0
        self._lots: Dict[str, _Lot] = {}

    def qty(self, symbol: str) -> float:
        return self._lots.get(symbol, _Lot()).qty

    def avg_entry(self, symbol: str) -> float:
        return self._lots.get(symbol, _Lot()).avg_entry

    def record_buy(self, symbol: str, qty: float, price: float) -> None:
        if qty <= 0 or price <= 0:
            raise ValueError("buy qty and price must be positive")
        cost = qty * price
        if cost > self.cash + 1e-9:
            raise ValueError("insufficient virtual cash for buy")
        lot = self._lots.setdefault(symbol, _Lot())
        new_qty = lot.qty + qty
        lot.avg_entry = (lot.avg_entry * lot.qty + price * qty) / new_qty
        lot.qty = new_qty
        self.cash -= cost

    def record_sell(self, symbol: str, qty: float, price: float) -> float:
        lot = self._lots.get(symbol, _Lot())
        if qty <= 0:
            raise ValueError("sell qty must be positive")
        if qty > lot.qty + 1e-9:
            raise ValueError(
                f"{self.strategy} cannot sell {qty} {symbol}; owns {lot.qty}"
            )
        proceeds = qty * price
        realized = (price - lot.avg_entry) * qty
        self.realized_pnl += realized
        self.cash += proceeds
        lot.qty -= qty
        if lot.qty <= 1e-9:
            self._lots.pop(symbol, None)
        return realized

    def to_portfolio_state(self, marks: Dict[str, float]) -> PortfolioState:
        positions = []
        invested = 0.0
        for sym, lot in self._lots.items():
            mark = float(marks.get(sym, lot.avg_entry))
            mv = lot.qty * mark
            invested += mv
            positions.append({"symbol": sym, "qty": lot.qty, "market_value": mv})
        equity = self.cash + invested
        return PortfolioState(
            equity=equity,
            cash=self.cash,
            positions=positions,
            today_starting_equity=self.starting_cash,
            week_starting_equity=self.starting_cash,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-bt/bin/python -m pytest tests/test_virtual_portfolio.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add live/virtual_portfolio.py tests/test_virtual_portfolio.py
git commit -m "feat(live): VirtualPortfolio per-strategy ledger with sell invariant"
```

---

## Task 3: Ports + Orchestrator

Wires signal → risk → executor for one cycle. Depends on Protocols so it is testable with fakes.

**Files:**
- Create: `live/ports.py`, `live/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write `live/ports.py` (no test — pure interface)**

```python
"""Ports the Orchestrator depends on, so it can be tested with fakes."""
from __future__ import annotations

from typing import Optional, Protocol

import pandas as pd


class BarProvider(Protocol):
    def get_bars(self, symbol: str, lookback: int) -> pd.DataFrame:
        """Trailing `lookback` daily OHLCV bars, lowercase columns, date index."""
        ...


class ExecutorPort(Protocol):
    def buy(self, *, symbol: str, qty: float, price: float, strategy: str,
            strategy_type: str, max_hold_days: Optional[int]) -> bool: ...

    def sell(self, *, symbol: str, qty: float, price: float, strategy: str) -> bool: ...
```

- [ ] **Step 2: Write the failing orchestrator tests**

`tests/test_orchestrator.py`:
```python
from __future__ import annotations

import pandas as pd

from live.orchestrator import Orchestrator, StrategyConfig
from live.virtual_portfolio import VirtualPortfolio


class FakeBars:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_bars(self, symbol: str, lookback: int) -> pd.DataFrame:
        return self._df


class RecordingExecutor:
    def __init__(self):
        self.buys = []
        self.sells = []

    def buy(self, *, symbol, qty, price, strategy, strategy_type, max_hold_days):
        self.buys.append((symbol, qty, strategy))
        return True

    def sell(self, *, symbol, qty, price, strategy):
        self.sells.append((symbol, qty, strategy))
        return True


def test_buy_signal_places_tagged_order_and_updates_ledger(oversold_then_bars):
    execu = RecordingExecutor()
    cfg = [StrategyConfig(strategy="confirmed_mr", symbols=["AAPL"], starting_cash=1000.0)]
    orch = Orchestrator(cfg, FakeBars(oversold_then_bars), execu,
                        risk_config={"max_position_pct": 1.0, "min_cash_reserve_pct": 0.0})
    orch.run_cycle()
    assert len(execu.buys) == 1
    assert execu.buys[0][0] == "AAPL"
    assert execu.buys[0][2] == "confirmed_mr"
    assert orch.portfolio("confirmed_mr").qty("AAPL") > 0


def test_hold_signal_places_no_order(rising_bars):
    execu = RecordingExecutor()
    cfg = [StrategyConfig(strategy="confirmed_mr", symbols=["AAPL"], starting_cash=1000.0)]
    orch = Orchestrator(cfg, FakeBars(rising_bars), execu)
    orch.run_cycle()
    assert execu.buys == []


def test_sell_signal_only_when_holding(oversold_then_bars):
    execu = RecordingExecutor()
    cfg = [StrategyConfig(strategy="confirmed_mr", symbols=["AAPL"], starting_cash=1000.0)]
    orch = Orchestrator(cfg, FakeBars(oversold_then_bars), execu,
                        risk_config={"max_position_pct": 1.0, "min_cash_reserve_pct": 0.0})
    orch.portfolio("confirmed_mr").record_buy("AAPL", qty=1.0, price=50.0)
    # Build an exit bar: RSI(2) high → exit True. Reuse a rising tail.
    bars = oversold_then_bars.copy()
    # Force exit by appending strong up bars so RSI(2) > 65 on last row.
    import numpy as np
    last_idx = pd.date_range(bars.index[-1], periods=6, freq="B")[1:]
    up = pd.DataFrame({c: [bars["close"].iloc[-1] * (1.05 ** (i + 1)) for i in range(5)]
                       for c in ["open", "high", "low", "close"]}, index=last_idx)
    up["volume"] = 1_000_000.0
    bars2 = pd.concat([bars, up])
    orch2 = Orchestrator(cfg, FakeBars(bars2), execu,
                         risk_config={"max_position_pct": 1.0, "min_cash_reserve_pct": 0.0})
    orch2.portfolio("confirmed_mr").record_buy("AAPL", qty=1.0, price=50.0)
    orch2.run_cycle()
    assert any(s[0] == "AAPL" and s[2] == "confirmed_mr" for s in execu.sells)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv-bt/bin/python -m pytest tests/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'live.orchestrator'`.

- [ ] **Step 4: Implement `live/orchestrator.py`**

```python
"""Orchestrator — one cycle of the paper horse race.

For each strategy and its symbols: fetch trailing bars, get a live signal,
size BUYs through the RiskManager against that strategy's virtual ledger,
and route BUY/SELL to the ExecutorPort with a strategy tag. SELLs only fire
for symbols the strategy currently holds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from live.ports import BarProvider, ExecutorPort
from live.strategy_runner import StrategyRunner
from live.virtual_portfolio import VirtualPortfolio
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

LOOKBACK_BARS = 260  # ~1y of daily bars; enough for 200d / momentum filters.


@dataclass
class StrategyConfig:
    strategy: str
    symbols: List[str]
    starting_cash: float


class Orchestrator:
    def __init__(
        self,
        configs: List[StrategyConfig],
        bars: BarProvider,
        executor: ExecutorPort,
        risk_config: Optional[dict] = None,
    ) -> None:
        self._bars = bars
        self._executor = executor
        self._risk = RiskManager(risk_config)
        self._runners: Dict[str, StrategyRunner] = {}
        self._portfolios: Dict[str, VirtualPortfolio] = {}
        self._symbols: Dict[str, List[str]] = {}
        for c in configs:
            self._runners[c.strategy] = StrategyRunner(c.strategy)
            self._portfolios[c.strategy] = VirtualPortfolio(c.strategy, c.starting_cash)
            self._symbols[c.strategy] = list(c.symbols)

    def portfolio(self, strategy: str) -> VirtualPortfolio:
        return self._portfolios[strategy]

    def run_cycle(self) -> None:
        for strategy, runner in self._runners.items():
            vp = self._portfolios[strategy]
            for symbol in self._symbols[strategy]:
                try:
                    df = self._bars.get_bars(symbol, LOOKBACK_BARS)
                except Exception as e:
                    logger.warning("bars fetch failed %s/%s: %s", strategy, symbol, e)
                    continue
                if df is None or len(df) == 0:
                    continue
                sig = runner.run(symbol, df)
                if sig.action == "BUY":
                    self._handle_buy(runner, vp, sig)
                elif sig.action == "SELL":
                    self._handle_sell(vp, sig)

    def _handle_buy(self, runner, vp, sig) -> None:
        if vp.qty(sig.symbol) > 0:
            return  # already holding for this strategy; no pyramiding in v1
        state = vp.to_portfolio_state(marks={sig.symbol: sig.price})
        proposed = state.equity  # let risk manager trim via caps/reserve
        decision = self._risk.evaluate_entry(
            symbol=sig.symbol,
            proposed_size_usd=proposed,
            proposed_price=sig.price,
            sector=None,
            portfolio=state,
            is_day_trade=False,
            pdt_can_day_trade=True,
        )
        if not decision.allowed or decision.adjusted_qty <= 0:
            logger.info("entry denied %s/%s: %s", runner.name, sig.symbol, decision.reason)
            return
        qty = decision.adjusted_qty
        if self._executor.buy(
            symbol=sig.symbol, qty=qty, price=sig.price, strategy=runner.name,
            strategy_type=runner.strategy_type, max_hold_days=runner.max_hold_days,
        ):
            vp.record_buy(sig.symbol, qty, sig.price)

    def _handle_sell(self, vp, sig) -> None:
        held = vp.qty(sig.symbol)
        if held <= 0:
            return
        if self._executor.sell(
            symbol=sig.symbol, qty=held, price=sig.price, strategy=vp.strategy
        ):
            vp.record_sell(sig.symbol, held, sig.price)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv-bt/bin/python -m pytest tests/test_orchestrator.py -v`
Expected: 3 passed. (If `test_sell_signal_only_when_holding` does not produce an exit, widen the up-tail multiplier until RSI(2) > 65 on the last bar — the strategy's exit threshold.)

- [ ] **Step 6: Commit**

```bash
git add live/ports.py live/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(live): Orchestrator wires signal->risk->executor per cycle"
```

---

## Task 4: Horse-race report

Aggregates the three ledgers into a side-by-side P&L view.

**Files:**
- Create: `live/horse_race_report.py`
- Test: `tests/test_horse_race_report.py`

- [ ] **Step 1: Write the failing test**

`tests/test_horse_race_report.py`:
```python
from __future__ import annotations

import pytest

from live.horse_race_report import build_report
from live.virtual_portfolio import VirtualPortfolio


def test_report_ranks_by_total_equity():
    a = VirtualPortfolio("momentum_rotation", 1000.0)
    a.record_buy("AAPL", 2.0, 100.0)  # cash 800, holds 2 AAPL
    b = VirtualPortfolio("rsi_mr", 1000.0)  # all cash
    rows = build_report([a, b], marks={"AAPL": 150.0})
    # a equity = 800 + 2*150 = 1100 ; b = 1000. a ranks first.
    assert rows[0]["strategy"] == "momentum_rotation"
    assert rows[0]["equity"] == pytest.approx(1100.0)
    assert rows[0]["return_pct"] == pytest.approx(10.0)
    assert rows[1]["strategy"] == "rsi_mr"
    assert rows[1]["return_pct"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-bt/bin/python -m pytest tests/test_horse_race_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'live.horse_race_report'`.

- [ ] **Step 3: Implement `live/horse_race_report.py`**

```python
"""Horse-race report — side-by-side equity/return per strategy ledger."""
from __future__ import annotations

from typing import Dict, List

from live.virtual_portfolio import VirtualPortfolio


def build_report(portfolios: List[VirtualPortfolio], marks: Dict[str, float]) -> List[dict]:
    rows = []
    for vp in portfolios:
        state = vp.to_portfolio_state(marks)
        ret = ((state.equity / vp.starting_cash) - 1.0) * 100.0 if vp.starting_cash else 0.0
        rows.append({
            "strategy": vp.strategy,
            "equity": state.equity,
            "cash": vp.cash,
            "realized_pnl": vp.realized_pnl,
            "return_pct": ret,
            "n_positions": len(state.positions),
        })
    rows.sort(key=lambda r: r["equity"], reverse=True)
    return rows


def format_table(rows: List[dict]) -> str:
    header = f"{'strategy':<20}{'equity':>12}{'return%':>10}{'realized':>12}{'pos':>5}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['strategy']:<20}{r['equity']:>12.2f}{r['return_pct']:>10.2f}"
            f"{r['realized_pnl']:>12.2f}{r['n_positions']:>5}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-bt/bin/python -m pytest tests/test_horse_race_report.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run the full suite**

Run: `.venv-bt/bin/python -m pytest -v`
Expected: all tests pass (Tasks 1–4).

- [ ] **Step 6: Commit**

```bash
git add live/horse_race_report.py tests/test_horse_race_report.py
git commit -m "feat(live): horse-race comparative report"
```

---

## Task 5: Live wiring — adapter + run_trading_system + paper smoke test

Maps `ExecutorPort` onto the real `executor.py` and replaces the dead loop. This is the only task that touches Alpaca; run it with **paper** keys only.

**Files:**
- Create: `live/live_executor_adapter.py`
- Modify: `run_trading_system.py`
- Test: manual paper smoke test (documented)

- [ ] **Step 1: Read the real executor's close/buy methods**

Run: `grep -n "def place_\|def close\|def sell\|ClosePositionRequest\|def buy" executor.py`
Expected: confirm `place_market_order_with_time_exit(symbol, qty, max_hold_days, side, strategy)` exists and find the position-close path (`ClosePositionRequest` / a close method). Use whatever close method exists; if none, close via `place_market_order_with_time_exit(..., side="SELL", max_hold_days=0)`.

- [ ] **Step 2: Implement `live/live_executor_adapter.py`**

```python
"""Adapter mapping the ExecutorPort onto the existing executor.py.

BUY  -> place_market_order_with_time_exit (no stop-loss; time/ signal exit).
SELL -> close this strategy's qty via a market SELL.
Isolates all real-executor specifics here so the Orchestrator stays testable.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_MOMENTUM_BACKSTOP_DAYS = 252  # exit primarily on SELL signal; this is a safety cap


class LiveExecutorAdapter:
    def __init__(self, executor) -> None:
        self._ex = executor

    def buy(self, *, symbol: str, qty: float, price: float, strategy: str,
            strategy_type: str, max_hold_days: Optional[int]) -> bool:
        hold = max_hold_days if max_hold_days else _MOMENTUM_BACKSTOP_DAYS
        res = self._ex.place_market_order_with_time_exit(
            symbol=symbol, qty=qty, max_hold_days=hold, side="BUY", strategy=strategy,
        )
        return res is not None

    def sell(self, *, symbol: str, qty: float, price: float, strategy: str) -> bool:
        res = self._ex.place_market_order_with_time_exit(
            symbol=symbol, qty=qty, max_hold_days=0, side="SELL", strategy=strategy,
        )
        return res is not None
```

(If Step 1 revealed a dedicated close method, call that instead of the `side="SELL"` market order.)

- [ ] **Step 3: Replace the dead loop in `run_trading_system.py`**

Replace the body of `run_trading_system.py` with a paper-only entry point:
```python
#!/usr/bin/env python3
"""Run the paper horse race: 3 backtested strategies, one paper account."""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from enhanced_alpaca_client import EnhancedAlpacaClient
from executor import Executor
from live.live_executor_adapter import LiveExecutorAdapter
from live.orchestrator import Orchestrator, StrategyConfig, LOOKBACK_BARS
from live.horse_race_report import build_report, format_table

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("horse_race")

UNIVERSE = ["SPY", "AAPL", "MSFT", "QQQ", "NVDA"]
SLICE = 33_000.0  # virtual cash per strategy (paper)

STRATEGIES = [
    StrategyConfig("momentum_rotation", UNIVERSE, SLICE),
    StrategyConfig("confirmed_mr", UNIVERSE, SLICE),
    StrategyConfig("rsi_mr", UNIVERSE, SLICE),
]


class AlpacaBars:
    def __init__(self, client: EnhancedAlpacaClient):
        self._c = client

    def get_bars(self, symbol: str, lookback: int):
        # Must return lowercase OHLCV columns, date index. Adapt to the
        # client's actual bar method discovered in executor/enhanced client.
        return self._c.get_daily_bars(symbol, lookback)


def main() -> int:
    load_dotenv()
    base_url = os.getenv("ALPACA_BASE_URL", "")
    if "paper" not in base_url:
        raise SystemExit("Refusing to run: ALPACA_BASE_URL is not a paper endpoint.")
    client = EnhancedAlpacaClient()
    executor = Executor()
    orch = Orchestrator(STRATEGIES, AlpacaBars(client), LiveExecutorAdapter(executor))
    orch.run_cycle()
    marks = {s: 0.0 for s in UNIVERSE}  # replace with last close per symbol if desired
    print(format_table(build_report(
        [orch.portfolio(s.strategy) for s in STRATEGIES], marks)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Reconcile `AlpacaBars.get_bars` with the real client**

Run: `grep -n "def get_.*bar\|StockBarsRequest\|TimeFrame" enhanced_alpaca_client.py`
Adjust `AlpacaBars.get_bars` to call the real method and rename/ lowercase columns to `open, high, low, close, volume` with a `DatetimeIndex`. Add a tiny helper if the client returns capitalized columns.

- [ ] **Step 5: Install live deps + set paper keys**

Run:
```bash
.venv-bt/bin/pip install alpaca-py python-dotenv
cp -n .env.example .env  # then edit .env: ALPACA_API_KEY / ALPACA_SECRET_KEY (PAPER),
                          # ALPACA_BASE_URL=https://paper-api.alpaca.markets, ENABLE_TRADING=true
```
Confirm `.env` is gitignored: `git check-ignore .env` → prints `.env`.

- [ ] **Step 6: Paper smoke test (manual)**

Run during US market hours: `.venv-bt/bin/python run_trading_system.py`
Expected: logs show signals; if any BUY fired, a tagged order appears via Alpaca paper and in `data/journal/<today>.jsonl`; the printed table shows three strategies. If market is closed, the executor logs "Market closed — rejecting order" and the table shows all cash — that is correct behavior.

- [ ] **Step 7: Verify journal attribution**

Run: `.venv-bt/bin/python -c "from trade_journal import get_trade_journal; print(get_trade_journal().strategy_attribution())"`
Expected: a dict keyed by strategy name (empty if no fills yet).

- [ ] **Step 8: Commit**

```bash
git add live/live_executor_adapter.py run_trading_system.py requirements-dev.txt
git commit -m "feat(live): paper horse-race entry point + executor adapter"
```

- [ ] **Step 9: Open PR**

```bash
git push -u origin feature/paper-horse-race
gh pr create --fill --title "Paper horse race: 3 strategies, one paper account"
```

---

## Scheduling (after smoke test passes)

Daily cadence via the existing pattern — `/loop 24h` or a cron that runs `run_trading_system.py` once per trading day after the close. Defer until the smoke test is green. Let it run for 2–4 weeks, then read `strategy_attribution()` to see which strategy leads on live data before any real-money conversation.

---

## Self-Review Notes

- **Spec coverage:** StrategyRunner (Task 1), VirtualPortfolio incl. sell invariant (Task 2), Orchestrator wiring the dead loop (Task 3), report (Task 4), Alpaca paper wiring + PDT-safe swing + paper-only guard (Task 5). All spec components covered.
- **Deferred per spec (YAGNI):** intraday cadence, crypto leg, signal_evaluator engine, cross-sectional top-N momentum rebalance (v1 uses per-symbol positive-momentum entry/exit as the strategy fn exposes it).
- **Type consistency:** `Signal(action, price, symbol, strategy)`, `StrategyConfig(strategy, symbols, starting_cash)`, `ExecutorPort.buy/sell` keyword-only signatures, and `VirtualPortfolio.record_buy/record_sell/to_portfolio_state/qty/avg_entry` are used identically across tasks.
- **Known reconcile points flagged for the implementer:** the real Alpaca bar method (Task 5 Step 4) and close method (Task 5 Step 1) — verified live rather than assumed.
