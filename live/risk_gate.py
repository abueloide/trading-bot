"""Risk Gate — segunda opinión que RECHAZA (no recorta) antes de ejecutar.

Patrón robado de TradingAgents (arXiv 2412.20138): un Risk Manager que revisa la
orden ANTES de ejecutar. `RiskManager.evaluate_entry` ya dimensiona y RECORTA;
este gate añade una capa que puede VETAR por completo una orden que el sizing
dejó pasar — alineado con "riesgo y disciplina por encima de todo".

Política (decisión Luis 2026-07-07): dos hard caps, modo RECHAZO —
  - posición > 20% del equity  → veta
  - sector   > 40% del equity  → veta
El gate sólo puede ENDURECER, nunca relajar `base_decision` (si el sizing base
negó, el gate también). Los caps son inyectables por config del horse-race.

Determinista (sin LLM): heurísticas reproducibles sobre `PortfolioState`, no
llamadas a un modelo. Cableado en `orchestrator._do_buy`, después de
`evaluate_entry` y antes de `executor.buy`.

Nota: la reserva de cash y los circuit-breakers de pérdida ya los enforcea el
RiskManager base — este gate NO los re-verifica (evita duplicar política).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from risk_manager import PortfolioState, RiskDecision

# Hard caps aprobados por Luis (2026-07-07). Defaults = su decisión; el
# orchestrator los puede sobreescribir vía RISK_CONFIG (gate_position_cap/…).
POSITION_HARD_CAP = 0.20   # máx 20% del equity en una sola posición
SECTOR_HARD_CAP = 0.40     # máx 40% del equity en un solo sector GICS


@dataclass
class GateVerdict:
    approved: bool
    vetoes: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.approved:
            return "gate: approved"
        return "gate: VETADO — " + "; ".join(self.vetoes)


def _concentration(size_usd: float, pf: PortfolioState) -> float:
    """Fracción del equity que ocuparía la posición propuesta."""
    return size_usd / pf.equity if pf.equity > 0 else 1.0


def _sector_frac_after(
    size_usd: float,
    pf: PortfolioState,
    sector: Optional[str],
    sector_of: Optional[Callable[[str], str]],
) -> float:
    """Fracción del equity que el sector del candidato ocuparía TRAS la compra.

    0.0 si no se pasó sector/lookup (sin data no se puede vetar por sector).
    Mismo cálculo que RiskManager step 8, pero aquí para RECHAZAR, no recortar.
    """
    if not sector or sector_of is None or pf.equity <= 0:
        return 0.0
    existing = sum(
        p.get("market_value", 0.0)
        for p in pf.positions
        if sector_of(p.get("symbol", "")) == sector
    )
    return (existing + size_usd) / pf.equity


def gate_entry(
    *,
    proposed_size_usd: float,
    portfolio: PortfolioState,
    base_decision: RiskDecision,
    sector: Optional[str] = None,
    sector_of: Optional[Callable[[str], str]] = None,
    position_cap: float = POSITION_HARD_CAP,
    sector_cap: float = SECTOR_HARD_CAP,
) -> GateVerdict:
    """Veta una orden YA dimensionada si viola los hard caps o si el sizing negó.

    ``sector``/``sector_of`` opcionales: sin ellos el cap de sector no aplica
    (no hay data de sector que evaluar).
    """
    vetoes: List[str] = []

    if not base_decision.allowed:
        vetoes.append(f"sizing base negó: {base_decision.reason}")

    conc = _concentration(proposed_size_usd, portfolio)
    if conc > position_cap:
        vetoes.append(f"posición {conc:.0%} > cap {position_cap:.0%}")

    sec_frac = _sector_frac_after(proposed_size_usd, portfolio, sector, sector_of)
    if sec_frac > sector_cap:
        vetoes.append(f"sector {sec_frac:.0%} > cap {sector_cap:.0%}")

    return GateVerdict(approved=not vetoes, vetoes=vetoes)
