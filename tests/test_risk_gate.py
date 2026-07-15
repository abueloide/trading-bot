"""Tests del Risk Gate — dos hard caps, modo RECHAZO (decisión Luis 2026-07-07).

  - posición > 20% del equity → veta
  - sector   > 40% del equity → veta
  - si el sizing base negó, el gate no puede relajarlo.
"""
from risk_manager import PortfolioState, RiskDecision
from live.risk_gate import gate_entry


def _pf(**kw) -> PortfolioState:
    base = dict(equity=100_000.0, cash=100_000.0, today_starting_equity=100_000.0, daily_pnl=0.0)
    base.update(kw)
    return PortfolioState(**base)


_OK = RiskDecision(True, "ok", adjusted_size_usd=1.0, adjusted_qty=1.0)

_SECTORS = {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials"}


def _sector_of(sym: str) -> str:
    return _SECTORS.get(sym, "Unknown")


def test_small_clean_entry_approved():
    # 10% del equity → bajo ambos caps → pasa.
    v = gate_entry(proposed_size_usd=10_000, portfolio=_pf(), base_decision=_OK)
    assert v.approved, v.summary


def test_position_at_cap_passes_over_cap_rejected():
    # 20% exacto pasa; 21% rechaza (cap es estricto >).
    assert gate_entry(proposed_size_usd=20_000, portfolio=_pf(), base_decision=_OK).approved
    v = gate_entry(proposed_size_usd=21_000, portfolio=_pf(), base_decision=_OK)
    assert not v.approved
    assert any("posición" in x for x in v.vetoes), v.vetoes


def test_sector_over_40pct_rejected():
    # Ya 30% en Technology (MSFT), candidato Tech suma 15% → 45% sector → rechazo.
    pf = _pf(positions=[{"symbol": "MSFT", "qty": 1, "market_value": 30_000.0}])
    v = gate_entry(
        proposed_size_usd=15_000,
        portfolio=pf,
        base_decision=_OK,
        sector="Technology",
        sector_of=_sector_of,
    )
    assert not v.approved
    assert any("sector" in x for x in v.vetoes), v.vetoes


def test_sector_under_cap_passes():
    # 30% en Technology + candidato en Financials (15%) → sin choque de sector.
    pf = _pf(positions=[{"symbol": "MSFT", "qty": 1, "market_value": 30_000.0}])
    v = gate_entry(
        proposed_size_usd=15_000,
        portfolio=pf,
        base_decision=_OK,
        sector="Financials",
        sector_of=_sector_of,
    )
    assert v.approved, v.summary


def test_sector_cap_noop_without_sector_data():
    # Sin sector/lookup no se puede vetar por sector: sólo aplica el cap de posición.
    v = gate_entry(proposed_size_usd=15_000, portfolio=_pf(), base_decision=_OK)
    assert v.approved, v.summary


def test_caps_are_injectable():
    # El horse-race puede afinar caps por config: 10% de cap rechaza un 15%.
    v = gate_entry(proposed_size_usd=15_000, portfolio=_pf(), base_decision=_OK, position_cap=0.10)
    assert not v.approved, v.summary


def test_gate_cannot_relax_a_base_denial():
    # Si el sizing base negó, el gate veta aunque los caps se cumplan.
    denied = RiskDecision(False, "max_open_positions reached")
    v = gate_entry(proposed_size_usd=5_000, portfolio=_pf(), base_decision=denied)
    assert not v.approved
    assert any("sizing base negó" in x for x in v.vetoes), v.vetoes


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL GREEN")
