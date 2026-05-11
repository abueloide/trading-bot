#!/usr/bin/env python3
"""
Risk Manager — enforces all portfolio-level rules from RISK_CONFIG.

Checks performed before any new entry:
    - max_open_positions, max_position_pct
    - max_sector_exposure_pct
    - earnings_blackout (no entries within N days of earnings)
    - cash reserve floor (dynamic with macro uncertainty)
    - circuit breakers (daily/weekly losses)
    - PDT compliance (when equity < $25K)
    - Psychology of Money risk scaling: smaller portfolios → larger %, capped

Returns a ``RiskDecision`` with allow/deny + reasoning + adjusted size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from config import RISK_CONFIG
except ImportError:
    RISK_CONFIG = {}


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""
    adjusted_size_usd: float = 0.0
    adjusted_qty: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class PortfolioState:
    equity: float = 0.0
    cash: float = 0.0
    positions: List[Dict[str, Any]] = field(default_factory=list)
    sector_for_symbol: Dict[str, str] = field(default_factory=dict)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    today_starting_equity: float = 0.0
    week_starting_equity: float = 0.0


class RiskManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = {**RISK_CONFIG, **(config or {})}

    # ---------------------------------------------------------- public API

    def evaluate_entry(
        self,
        *,
        symbol: str,
        proposed_size_usd: float,
        proposed_price: float,
        sector: Optional[str],
        portfolio: PortfolioState,
        macro_risk_score: float = 0.5,
        next_earnings_date: Optional[date] = None,
        is_day_trade: bool = False,
        pdt_can_day_trade: bool = True,
    ) -> RiskDecision:
        notes: List[str] = []

        # 1) Circuit breakers — daily / weekly loss
        if portfolio.today_starting_equity > 0:
            daily_dd = portfolio.daily_pnl / portfolio.today_starting_equity
            if daily_dd <= -abs(self.cfg.get("max_daily_loss_pct", 0.05)):
                return RiskDecision(False, "daily loss limit hit, no new entries today")
        if portfolio.week_starting_equity > 0:
            weekly_dd = portfolio.weekly_pnl / portfolio.week_starting_equity
            if weekly_dd <= -abs(self.cfg.get("max_weekly_loss_pct", 0.08)):
                return RiskDecision(False, "weekly loss limit hit, no new entries this week")

        # 2) Earnings blackout
        if next_earnings_date is not None:
            days_to = (next_earnings_date - date.today()).days
            blackout = self.cfg.get("earnings_blackout_days", 3)
            if 0 <= days_to <= blackout:
                return RiskDecision(False, f"in earnings blackout (next: {next_earnings_date})")

        # 3) PDT compliance
        if is_day_trade and not pdt_can_day_trade:
            return RiskDecision(False, "PDT limit reached")

        # 4) Max open positions
        if len(portfolio.positions) >= self.cfg.get("max_open_positions", 4):
            return RiskDecision(False, "max_open_positions reached")

        # 5) Cash reserve — dynamic floor
        min_cash_pct = self.cfg.get("min_cash_reserve_pct", 0.20)
        max_cash_pct = self.cfg.get("max_cash_reserve_pct", 0.50)
        # Increase reserve linearly with macro risk above 0.5.
        dynamic_extra = max(0.0, (macro_risk_score - 0.5) * 0.6)
        target_cash_pct = min(min_cash_pct + dynamic_extra, max_cash_pct)
        required_cash = portfolio.equity * target_cash_pct
        if portfolio.cash - proposed_size_usd < required_cash:
            available = portfolio.cash - required_cash
            if available <= 0:
                return RiskDecision(False, "would breach dynamic cash reserve")
            proposed_size_usd = available
            notes.append(f"Trimmed to keep {target_cash_pct:.0%} cash reserve")

        # 6) Position-size cap
        max_pos_pct = self.cfg.get("max_position_pct", 0.25)
        max_pos_dollars = portfolio.equity * max_pos_pct
        if proposed_size_usd > max_pos_dollars:
            proposed_size_usd = max_pos_dollars
            notes.append(f"Capped at {max_pos_pct:.0%} max position size")

        # 7) Psychology of Money — risk scaling by portfolio size
        scaling = self.cfg.get("risk_scaling", {})
        if portfolio.equity <= 5_000:
            risk_pct = scaling.get("up_to_5k", 0.02)
        elif portfolio.equity <= 10_000:
            risk_pct = scaling.get("up_to_10k", 0.015)
        else:
            risk_pct = scaling.get("above_10k", 0.01)
        risk_dollars = portfolio.equity * risk_pct
        # When this is a stop-loss-bearing strategy, the caller passes the SL
        # implicitly via proposed_size_usd; here we cap so a 1-stop loss
        # equals at most risk_dollars.
        if proposed_size_usd * abs(self.cfg.get("default_stop_loss_pct", 0.02)) > risk_dollars:
            new_size = risk_dollars / abs(self.cfg.get("default_stop_loss_pct", 0.02))
            if new_size < proposed_size_usd:
                proposed_size_usd = new_size
                notes.append(f"Trimmed to per-trade {risk_pct:.1%} risk budget")

        # 8) Sector concentration
        max_sector_pct = self.cfg.get("max_sector_exposure_pct", 0.40)
        if sector:
            existing_sector_dollars = sum(
                p.get("market_value", 0.0)
                for p in portfolio.positions
                if portfolio.sector_for_symbol.get(p.get("symbol"), "") == sector
            )
            allowed_sector_dollars = portfolio.equity * max_sector_pct
            if existing_sector_dollars + proposed_size_usd > allowed_sector_dollars:
                room = allowed_sector_dollars - existing_sector_dollars
                if room <= 0:
                    return RiskDecision(False, f"sector cap reached for {sector}")
                proposed_size_usd = min(proposed_size_usd, room)
                notes.append(f"Trimmed by {sector} sector cap")

        if proposed_size_usd <= 0 or proposed_price <= 0:
            return RiskDecision(False, "size or price invalid after risk adjustment")

        qty = proposed_size_usd / proposed_price
        return RiskDecision(
            allowed=True,
            adjusted_size_usd=proposed_size_usd,
            adjusted_qty=qty,
            reason="passed all risk checks",
            notes=notes,
        )

    @staticmethod
    def correlation_warning(
        symbol: str, sector: str, portfolio: PortfolioState, threshold_pct: float = 0.30
    ) -> bool:
        """Lightweight correlation check: True if same-sector exposure is high."""
        same_sector_value = sum(
            p.get("market_value", 0.0)
            for p in portfolio.positions
            if portfolio.sector_for_symbol.get(p.get("symbol"), "") == sector
        )
        return portfolio.equity > 0 and same_sector_value / portfolio.equity > threshold_pct


_risk_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager
