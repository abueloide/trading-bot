#!/usr/bin/env python3
"""
Backtest-first gate — verdict PASS/FAIL on a walk-forward summary.

A strategy earns a paper-race slot ONLY if it clears this gate on the
out-of-sample (walk-forward) backtest. No more "deploy and pray".

Producer:  python backtesting/run_backtest.py --strategy X   ->  results/X_summary_*.json
Verdict:   python backtesting/gate.py results/X_summary_*.json

Why median + breadth, not mean (learned 2026-07-08):
    momentum_rotation showed mean excess +9.1% — but that was ONE outlier
    (NVDA +118.7%); the other 4 symbols were all negative. A mean-based gate
    passes that mirage; a median/breadth gate vetoes it. Live paper confirmed
    the veto (momentum was the worst horse). See docs/postmortems/.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from typing import Any, Dict, List, Optional

# Gate thresholds. Tuned to veto all 4 of the 2026-07 field on OOS backtest.
# ponytail: constants here, move to config only if a second consumer appears.
MIN_MEDIAN_EXCESS_PCT = 0.0   # median OOS return must beat its benchmark
MIN_BREADTH_FRAC = 0.60       # >=60% of symbols must have positive excess
MIN_MEDIAN_SHARPE = 0.80      # median risk-adjusted return floor
MIN_SYMBOLS = 4               # too few names = not a real test
MIN_TRADES_PER_SYMBOL = 5     # too few trades = luck, not signal


def evaluate(per_symbol: Dict[str, dict]) -> Dict[str, object]:
    """Apply the gate to a summary's per_symbol block. Pure; no I/O."""
    rows = [v for v in per_symbol.values() if "excess_return_pct" in v]
    if len(rows) < MIN_SYMBOLS:
        return {"passed": False, "reason": f"only {len(rows)} symbols (< {MIN_SYMBOLS})", "checks": {}}

    excess = [r["excess_return_pct"] for r in rows]
    sharpe = [r.get("sharpe", 0.0) for r in rows]
    trades = [r.get("n_trades", 0) for r in rows]
    breadth = sum(1 for e in excess if e > 0) / len(excess)

    checks = {
        "median_excess_pct": (round(st.median(excess), 2), st.median(excess) > MIN_MEDIAN_EXCESS_PCT),
        "breadth_frac": (round(breadth, 2), breadth >= MIN_BREADTH_FRAC),
        "median_sharpe": (round(st.median(sharpe), 2), st.median(sharpe) >= MIN_MEDIAN_SHARPE),
        "min_trades": (min(trades), min(trades) >= MIN_TRADES_PER_SYMBOL),
    }
    passed = all(ok for _, ok in checks.values())
    failed = [k for k, (_, ok) in checks.items() if not ok]
    return {"passed": passed, "reason": "clean" if passed else "failed: " + ", ".join(failed), "checks": checks}


def annotate_validation(
    result: Dict[str, object],
    validation: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, object]:
    """Optionally attach statistical-validation stats to an evaluate() verdict.

    Non-breaking / additive only: never changes `passed`, `reason`, or `checks`
    — it just stacks a `validation` block on top for extra context on the
    strategy that already cleared (or failed) the gate above.

    `validation` is caller-supplied: a dict keyed by symbol with the dict
    returned by backtesting.validation.monte_carlo_test() / bootstrap_sharpe_ci()
    (e.g. {"AAPL": {"p_value_sharpe": 0.01, ...}, ...}). This function does no
    I/O and makes no run-dir assumptions — the caller is responsible for
    loading each symbol's equity/trades artifacts and calling those pure
    functions; this just aggregates whatever it's given. If `validation` is
    omitted, `result` is returned unchanged, so existing call sites (and
    tests) are unaffected.
    """
    if not validation:
        return result

    p_values = [v["p_value_sharpe"] for v in validation.values() if "p_value_sharpe" in v]
    prob_positives = [v["prob_positive"] for v in validation.values() if "prob_positive" in v]

    annotated = dict(result)
    annotated["validation"] = {
        "symbols": sorted(validation.keys()),
        "median_p_value_sharpe": round(st.median(p_values), 4) if p_values else None,
        "median_prob_positive": round(st.median(prob_positives), 4) if prob_positives else None,
    }
    return annotated


def _load(path: str) -> Dict[str, dict]:
    return json.load(open(path))["per_symbol"]


def main(argv: List[str]) -> int:
    if len(argv) != 1:
        print("usage: python backtesting/gate.py results/<strategy>_summary_*.json", file=sys.stderr)
        return 2
    res = evaluate(_load(argv[0]))
    verdict = "PASS ✅" if res["passed"] else "FAIL ❌"
    print(f"{verdict}  ({res['reason']})")
    for name, (val, ok) in res["checks"].items():
        print(f"  {'✓' if ok else '✗'} {name:18s} = {val}")
    return 0 if res["passed"] else 1


def _selfcheck() -> None:
    # Outlier mirage: one huge winner, rest negative -> must FAIL (breadth + median).
    mirage = {
        "SPY": {"excess_return_pct": -20.2, "sharpe": 1.02, "n_trades": 13},
        "AAPL": {"excess_return_pct": -21.3, "sharpe": 0.61, "n_trades": 16},
        "MSFT": {"excess_return_pct": -19.2, "sharpe": 0.68, "n_trades": 10},
        "QQQ": {"excess_return_pct": -12.7, "sharpe": 1.24, "n_trades": 3},
        "NVDA": {"excess_return_pct": 118.7, "sharpe": 1.56, "n_trades": 5},
    }
    assert evaluate(mirage)["passed"] is False, "outlier mirage must be vetoed"

    # Genuine broad edge -> must PASS.
    good = {s: {"excess_return_pct": 8.0, "sharpe": 1.1, "n_trades": 12} for s in "SPY AAPL MSFT QQQ NVDA".split()}
    assert evaluate(good)["passed"] is True, "broad positive edge must pass"

    # Too few symbols -> FAIL regardless.
    assert evaluate({"SPY": good["SPY"]})["passed"] is False, "thin universe must fail"
    print("selfcheck ok")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selfcheck":
        _selfcheck()
    else:
        raise SystemExit(main(sys.argv[1:]))
