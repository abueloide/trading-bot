"""Tests del debate multi-perspectiva (patrón TradingAgents portado)."""
from risk_manager import PortfolioState, RiskDecision
from live.risk_gate import debate_entry


def _pf(**kw) -> PortfolioState:
    base = dict(equity=100_000.0, cash=100_000.0, today_starting_equity=100_000.0, daily_pnl=0.0)
    base.update(kw)
    return PortfolioState(**base)


_OK = RiskDecision(True, "ok", adjusted_size_usd=1.0, adjusted_qty=1.0)


def test_small_clean_entry_approved():
    # 10% del equity, 90% cash libre, sin DD → ningún lente veta.
    v = debate_entry(proposed_size_usd=10_000, portfolio=_pf(), base_decision=_OK)
    assert v.approved, v.summary


def test_majority_veto_blocks_even_if_sizing_allowed():
    # 20% concentración: conservador veta (>15%) y deja 80% cash; neutral NO veta.
    # Subimos para que conservador+neutral veten (concentración >25%) → mayoría.
    v = debate_entry(proposed_size_usd=30_000, portfolio=_pf(), base_decision=_OK)
    assert not v.approved
    assert any("conservative" in x or "mayoría" in x for x in v.vetoes), v.vetoes


def test_aggressive_hard_stop_vetoes_alone():
    # 45% concentración: hard stop del lente agresivo solo basta.
    v = debate_entry(proposed_size_usd=45_000, portfolio=_pf(), base_decision=_OK)
    assert not v.approved
    assert any(">40%" in x for x in v.vetoes), v.vetoes


def test_gate_cannot_relax_a_base_denial():
    # Aunque el debate no encuentre problema, si el sizing base negó, el gate veta.
    denied = RiskDecision(False, "max_open_positions reached")
    v = debate_entry(proposed_size_usd=5_000, portfolio=_pf(), base_decision=denied)
    assert not v.approved
    assert any("sizing base negó" in x for x in v.vetoes), v.vetoes


def test_intraday_drawdown_makes_conservative_cautious_but_not_majority():
    # Ya 3% abajo hoy: conservador veta; neutral/agresivo no → 1 voto, no mayoría → pasa.
    v = debate_entry(
        proposed_size_usd=10_000,
        portfolio=_pf(daily_pnl=-3_000.0),
        base_decision=_OK,
    )
    assert v.approved, v.summary
    assert any(s.lens == "conservative" and s.veto for s in v.stances)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL GREEN")
