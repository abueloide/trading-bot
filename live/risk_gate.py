"""Multi-perspective risk debate — patrón portado de TradingAgents (arXiv 2412.20138).

`RiskManager.evaluate_entry` ya da UN veredicto de sizing. TradingAgents añade una
capa ANTES de ejecutar: varios analistas de riesgo con sesgos distintos
(conservador / neutral / agresivo) "debaten" la orden ya dimensionada y un juez
agrega. El valor no es más reglas, es una SEGUNDA opinión multi-lente que puede
vetar un trade que el sizing dejó pasar — alineado con "riesgo y disciplina por
encima de todo".

Determinista (sin LLM): heurísticas reproducibles sobre `PortfolioState`, no
llamadas a un modelo. Así es testeable y no mete sorpresas en una decisión que
toca dinero. **NO está cableado al orchestrator** — se enchufa vía PR después de
`evaluate_entry` y antes de `executor.buy` (ver PLAN.md → Herramientas externas).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from risk_manager import PortfolioState, RiskDecision


@dataclass
class RiskStance:
    lens: str          # conservative | neutral | aggressive
    veto: bool
    reason: str = ""


@dataclass
class GateVerdict:
    approved: bool
    stances: List[RiskStance] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.approved:
            return "gate: approved (debate sin veto de mayoría)"
        return "gate: VETADO — " + "; ".join(self.vetoes)


def _concentration(size_usd: float, pf: PortfolioState) -> float:
    """Fracción del equity que ocuparía la posición propuesta."""
    return size_usd / pf.equity if pf.equity > 0 else 1.0


def _cash_left_frac(size_usd: float, pf: PortfolioState) -> float:
    """Fracción de cash que quedaría libre tras la compra."""
    return (pf.cash - size_usd) / pf.equity if pf.equity > 0 else 0.0


def _intraday_dd(pf: PortfolioState) -> float:
    """Drawdown intradía como fracción (≤0 = perdiendo hoy)."""
    base = pf.today_starting_equity
    return pf.daily_pnl / base if base > 0 else 0.0


def _conservative(size_usd: float, pf: PortfolioState) -> RiskStance:
    if _concentration(size_usd, pf) > 0.15:
        return RiskStance("conservative", True, "concentración >15% del equity")
    if _cash_left_frac(size_usd, pf) < 0.30:
        return RiskStance("conservative", True, "dejaría <30% de cash de reserva")
    if _intraday_dd(pf) < -0.02:
        return RiskStance("conservative", True, "ya >2% abajo hoy, no abrir riesgo nuevo")
    return RiskStance("conservative", False)


def _neutral(size_usd: float, pf: PortfolioState) -> RiskStance:
    if _concentration(size_usd, pf) > 0.25:
        return RiskStance("neutral", True, "concentración >25% del equity")
    if _cash_left_frac(size_usd, pf) < 0.15:
        return RiskStance("neutral", True, "dejaría <15% de cash")
    return RiskStance("neutral", False)


def _aggressive(size_usd: float, pf: PortfolioState) -> RiskStance:
    if _concentration(size_usd, pf) > 0.40:
        return RiskStance("aggressive", True, "concentración >40% — hard stop incluso para tesis agresiva")
    if _cash_left_frac(size_usd, pf) < 0.0:
        return RiskStance("aggressive", True, "no hay cash para fondear")
    return RiskStance("aggressive", False)


def debate_entry(
    *,
    proposed_size_usd: float,
    portfolio: PortfolioState,
    base_decision: RiskDecision,
) -> GateVerdict:
    """Corre el debate sobre una orden YA dimensionada por evaluate_entry.

    Regla del juez (conservadora): se VETA si
      (a) el sizing base ya negó, o
      (b) el lente agresivo vetó (hard stop que ni la tesis permisiva tolera), o
      (c) mayoría (≥2 de 3 lentes) vetó.
    Si no, se aprueba. El gate sólo puede ENDURECER, nunca relajar `base_decision`.
    """
    stances = [
        _conservative(proposed_size_usd, portfolio),
        _neutral(proposed_size_usd, portfolio),
        _aggressive(proposed_size_usd, portfolio),
    ]
    vetoes: List[str] = []
    if not base_decision.allowed:
        vetoes.append(f"sizing base negó: {base_decision.reason}")

    veto_stances = [s for s in stances if s.veto]
    if any(s.lens == "aggressive" and s.veto for s in stances):
        vetoes.append(next(s.reason for s in stances if s.lens == "aggressive" and s.veto))
    elif len(veto_stances) >= 2:
        vetoes.append("mayoría de lentes vetó: " + "; ".join(f"{s.lens}: {s.reason}" for s in veto_stances))

    return GateVerdict(approved=not vetoes, stances=stances, vetoes=vetoes)
