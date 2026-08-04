from __future__ import annotations

import pytest

from live.strategy_runner import StrategyRunner


def test_unknown_strategy_raises():
    with pytest.raises(KeyError):
        StrategyRunner("does_not_exist")
