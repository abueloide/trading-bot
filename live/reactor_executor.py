"""Ejecutor del news reactor — órdenes en paper con ledger propio.

Aislado del horse-race a propósito: el reactor abre y cierra por su cuenta y no
debe tocar los books de los 5 caballos. Reusa `executor.Executor` (que ya trae el
guard de cuenta paper y el rechazo fuera de horario) y `VirtualPortfolio` para
llevar su propio cash.

MONEY GUARDRAIL: el SELL va por `place_market_sell` con qty exacta, NUNCA por
close_position — la cuenta paper es compartida con el horse-race y close_position
liquidaría la posición NETA del símbolo, robándole acciones a los otros caballos.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import List

from live.virtual_portfolio import VirtualPortfolio

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "data" / "news_reactor" / "ledger.json"
OPENED = REPO / "data" / "news_reactor" / "opened.json"  # símbolo -> fecha de entrada
STARTING_CASH = 25_000.0


class ReactorExecutor:
    def __init__(self, executor=None) -> None:
        if executor is None:
            from executor import Executor
            executor = Executor()
        self._ex = executor
        self._vp = self._load()

    # ---------------------------------------------------------- ledger
    def _load(self) -> VirtualPortfolio:
        if STATE.exists():
            try:
                return VirtualPortfolio.from_dict(json.loads(STATE.read_text()))
            except Exception as e:
                logger.warning("ledger ilegible (%s); arranco limpio", e)
        return VirtualPortfolio("news_reactor", STARTING_CASH)

    def _save(self) -> None:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._vp.to_dict()))
        tmp.replace(STATE)  # atómico: nunca dejar el ledger a medias

    def open_symbols(self) -> List[str]:
        # VirtualPortfolio no expone los símbolos; los lots viven en to_dict().
        lots = self._vp.to_dict().get("lots", {})
        return [s for s, lot in lots.items() if float(lot.get("qty", 0)) > 0]

    # ------------------------------------------------- salidas por tiempo
    # El reactor DEBE cerrar lo suyo: `executor.check_time_exits()` está
    # deliberadamente sin llamar en el sistema (ver run_trading_system.py), así
    # que el time-exit registrado en el broker nunca dispara. Sin esto el reactor
    # llena su tope de concurrentes el primer día y deja de operar para siempre
    # (medido 2026-07-30: 5 posiciones abiertas, 72 catalizadores rechazados).
    def _load_opened(self) -> dict:
        if OPENED.exists():
            try:
                return json.loads(OPENED.read_text())
            except Exception:
                return {}
        return {}

    def _save_opened(self, opened: dict) -> None:
        OPENED.parent.mkdir(parents=True, exist_ok=True)
        tmp = OPENED.with_suffix(".tmp")
        tmp.write_text(json.dumps(opened))
        tmp.replace(OPENED)

    def close_due(self, hold_days: int) -> List[str]:
        """Cierra las posiciones cuya ventana de hold ya venció. Devuelve símbolos."""
        opened = self._load_opened()
        today = date.today()
        closed: List[str] = []
        for sym in list(self.open_symbols()):
            iso = opened.get(sym)
            if not iso:
                # Sin fecha (posición previa al fix): trátala como vencida.
                age = hold_days
            else:
                age = (today - date.fromisoformat(iso)).days
            if age >= hold_days and self.sell_all(sym):
                opened.pop(sym, None)
                closed.append(sym)
        self._save_opened(opened)
        return closed

    # ---------------------------------------------------------- órdenes
    def buy_dollars(self, *, symbol: str, dollars: float, hold_days: int,
                    strategy: str = "news_reactor") -> bool:
        price = self._last_price(symbol)
        if price is None or price <= 0:
            logger.warning("sin precio para %s — no opero", symbol)
            return False
        if dollars > self._vp.cash:
            logger.info("cash insuficiente para %s (%.0f > %.0f)", symbol, dollars, self._vp.cash)
            return False
        qty = round(dollars / price, 4)
        if qty <= 0:
            return False
        res = self._ex.place_market_order_with_time_exit(
            symbol=symbol, qty=qty, max_hold_days=hold_days, side="BUY", strategy=strategy,
        )
        if res is None:
            return False
        try:
            self._vp.record_buy(symbol, qty, price)
        except ValueError as e:
            logger.critical("LEDGER DRIFT news_reactor/%s: broker llenó, ledger rechazó: %s",
                            symbol, e)
            return False
        opened = self._load_opened()
        opened[symbol] = date.today().isoformat()
        self._save_opened(opened)
        self._save()
        logger.info("BUY %s qty=%s @ %.2f (%s)", symbol, qty, price, strategy)
        return True

    def sell_all(self, symbol: str) -> bool:
        qty = self._vp.qty(symbol)
        if qty <= 0:
            return False
        price = self._last_price(symbol)
        res = self._ex.place_market_sell(symbol=symbol, qty=qty, strategy="news_reactor")
        if res is None:
            return False
        if price:
            self._vp.record_sell(symbol, qty, price)
            self._save()
        return True

    def _last_price(self, symbol: str):
        try:
            from live.yfinance_bars import YFinanceBars
            df = YFinanceBars().get_bars(symbol, 5)
            if df is None or df.empty:
                return None
            return float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning("precio de %s falló: %s", symbol, e)
            return None
