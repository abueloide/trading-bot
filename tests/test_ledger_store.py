from __future__ import annotations

import pytest

from live.ledger_store import save_ledgers, load_ledgers
from live.virtual_portfolio import VirtualPortfolio


def test_save_then_load_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    a = VirtualPortfolio("momentum_rotation", 1000.0)
    a.record_buy("AAPL", 2.0, 100.0)
    b = VirtualPortfolio("rsi_mr", 1000.0)
    save_ledgers([a, b], path)
    states = load_ledgers(path)
    assert set(states.keys()) == {"momentum_rotation", "rsi_mr"}
    assert states["momentum_rotation"]["lots"]["AAPL"]["qty"] == 2.0


def test_load_missing_file_returns_empty(tmp_path):
    assert load_ledgers(tmp_path / "nope.json") == {}


def test_save_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "state.json"
    save_ledgers([VirtualPortfolio("s", 1.0)], path)
    assert path.exists()


def test_corrupt_state_fails_hard_not_silent_reset(tmp_path):
    # A present-but-unreadable state file must abort the run, NOT reset ledgers
    # to starting cash (which would double broker positions). Audit #1.
    path = tmp_path / "state.json"
    path.write_text("{not valid json")
    with pytest.raises(SystemExit):
        load_ledgers(path)
