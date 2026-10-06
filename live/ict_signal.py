"""ICT-style intraday setup — mechanical reading of the "trading rizz" reel.

The reel (2026-10-06): "before the New York session open, drop into higher time
frames, determine a bias, go into lower time frames, look for an SMT divergence
and an inverse fair value gap, retrace in, tap in, target resting liquidity,
1:2 / 1:3 RR."

His method is discretionary. This is ONE fixed, un-tuned interpretation so it can
be backtested and run forward without anyone's eye in the loop:

  bias     prior daily close vs EMA(20) of daily closes (above = long-only day).
  SMT      one index sweeps its recent low while the other does not (bull);
           mirror with highs (bear).
  iFVG     a 3-bar gap against the bias, formed into the sweep, that price then
           CLOSES back through -> the gap flips into support/resistance.
  entry    limit at the gap edge (the "retrace in, tap in").
  stop     beyond the sweep extreme.
  target   session high/low (resting liquidity), clamped to 2R..3R.

Pure functions over DataFrames; the backtest and the live runner share them, so
there is no separate "live logic" to drift from what was measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Optional

import pandas as pd

SWEEP_LOOKBACK = 12   # bars (1h of 5m) that define the "recent" low/high
SWEEP_MAX_AGE = 12    # inversion must come within this many bars of the sweep
FVG_LOOKBACK = 12     # gap must have formed within this many bars before the sweep
ENTRY_TTL_BARS = 6    # resting limit is cancelled if not tapped in 30 min
MIN_RR, MAX_RR = 2.0, 3.0
MIN_RISK_PCT = 0.0005  # skip microscopic stops (< 0.05% of price): pure noise
STOP_PAD = 0.02
BIAS_EMA = 20
WINDOW_START, WINDOW_END = time(9, 35), time(11, 30)  # ET, signal-bar start
RTH_START, RTH_END = time(9, 30), time(16, 0)
FLAT_TIME = time(15, 55)  # bar that starts here is the forced exit


@dataclass(frozen=True)
class Setup:
    side: int          # +1 long, -1 short
    entry: float
    stop: float
    target: float
    signal_time: pd.Timestamp

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop)


def rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-session bars; index converted to America/New_York."""
    idx = df.index.tz_convert("America/New_York")
    out = df.copy()
    out.index = idx
    t = idx.time
    return out[(t >= RTH_START) & (t < RTH_END)]


def daily_bias(closes: pd.Series, day) -> int:
    """+1/-1 from the PRIOR session's close vs its EMA(20). 0 = not enough history.

    `closes` = one close per session, indexed by date. Only sessions strictly
    before `day` are used, so the bias is known before the open.
    """
    prior = closes[closes.index < day]
    if len(prior) < BIAS_EMA:
        return 0
    ema = prior.ewm(span=BIAS_EMA, adjust=False).mean().iloc[-1]
    return 1 if prior.iloc[-1] > ema else -1


def find_setup(q: pd.DataFrame, s: pd.DataFrame, bias: int) -> Optional[Setup]:
    """Setup confirmed by the LAST bar of `q`, or None.

    `q` (traded) and `s` (SMT pair) are aligned RTH 5m frames ending at the same
    bar and spanning at least the previous session. Only the last bar's close is
    new information, so calling this bar by bar has no lookahead.
    """
    if bias == 0 or len(q) < SWEEP_LOOKBACK + 3 or not q.index.equals(s.index):
        return None
    t = len(q) - 1
    now = q.index[t]
    if not (WINDOW_START <= now.time() <= WINDOW_END):
        return None
    # Mirror shorts onto the long logic: negate and swap high/low.
    if bias < 0:
        q = _flip(q)
        s = _flip(s)
    lo_q, lo_s, hi_q, cl_q = q["low"].values, s["low"].values, q["high"].values, q["close"].values
    day_start = int((q.index.date == now.date()).argmax())

    # 1) most recent SMT sweep today: exactly one of the pair takes its recent low.
    k = None
    for j in range(t - 1, max(day_start, t - SWEEP_MAX_AGE) - 1, -1):
        if j < SWEEP_LOOKBACK:
            break
        q_sweep = lo_q[j] < lo_q[j - SWEEP_LOOKBACK:j].min()
        s_sweep = lo_s[j] < lo_s[j - SWEEP_LOOKBACK:j].min()
        if q_sweep != s_sweep:
            k = j
            break
    if k is None:
        return None

    # 2) latest down-gap formed into the sweep: high[i] < low[i-2].
    zone_top = None
    for i in range(k, max(k - FVG_LOOKBACK, 2) - 1, -1):
        if hi_q[i] < lo_q[i - 2]:
            zone_top, gap_i = lo_q[i - 2], i
            break
    if zone_top is None:
        return None

    # 3) inversion: THIS bar is the first close back above the gap since it formed.
    if not (cl_q[t] > zone_top and (cl_q[gap_i:t] <= zone_top).all()):
        return None

    entry = zone_top
    stop = lo_q[k:t + 1].min() - STOP_PAD
    risk = entry - stop
    if risk <= 0 or risk < MIN_RISK_PCT * abs(entry):
        return None
    liquidity = hi_q[day_start:t + 1].max()
    target = min(max(liquidity, entry + MIN_RR * risk), entry + MAX_RR * risk)
    if bias < 0:
        entry, stop, target = -entry, -stop, -target
    return Setup(side=bias, entry=float(entry), stop=float(stop), target=float(target), signal_time=now)


def _flip(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": -df["open"], "high": -df["low"], "low": -df["high"], "close": -df["close"]},
        index=df.index,
    )
