"""VirtualPortfolio — per-strategy ledger over a shared paper account.

Tracks the cash slice, per-symbol qty/avg-entry, and realized P&L for ONE
strategy. The real Alpaca account nets all strategies together; this ledger
keeps them separate so a SELL only ever disposes of shares THIS strategy
bought. Emits a RiskManager.PortfolioState for position sizing.
"""
from __future__ import annotations

from dataclasses import dataclass
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
        if qty <= 0 or price <= 0:
            raise ValueError("sell qty and price must be positive")
        if qty > lot.qty + 1e-9:
            raise ValueError(
                f"{self.strategy} cannot sell {qty} {symbol}; owns {lot.qty}"
            )
        qty = min(qty, lot.qty)  # absorb float noise; never subtract more than owned
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
