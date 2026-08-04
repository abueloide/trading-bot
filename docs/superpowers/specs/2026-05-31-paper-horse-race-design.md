# Paper Trading "Horse Race" — Design Spec

**Date:** 2026-05-31
**Status:** Approved (design), pending implementation plan
**Author:** Luisfer + Claude

## Problem

`abueloide/trading-bot` is a solid scaffold but two gaps block any real use:

1. **The executor is not wired into the live loop.** `trading_bot.py` / `run_trading_system.py` compute a signal, log it, and stop. `executor.py` (which can place real Alpaca orders) is never called. Today the bot trades $0.
2. **No strategy has a proven edge.** Backtests (2022–2024, in-sample) show every strategy underperforms SPY buy-and-hold except when handed NVDA in hindsight. Deploying real money now = expected loss.

## Goal

Run a **paper-trading horse race**: the three backtested strategies operate live, in parallel, on the same real-time Alpaca data, in a **single paper account** (no real money). Each strategy's P&L is tracked separately so we can watch which one actually performs on out-of-sample, live data — before any real-money decision.

**Explicit non-goal:** real money. Live trading with $250 is out of scope until the horse race produces evidence. No code path in this work touches a live account.

## Strategies in the race

From `backtesting/strategies.py` (the code we backtested → paper mirrors the backtest):

| Strategy | Type | Hold | PDT risk |
|---|---|---|---|
| `momentum_rotation` | momentum | weeks (monthly rebalance) | none |
| `confirmed_mr` | mean_reversion | ≤7 days | low |
| `rsi_mr` | mean_reversion | ≤10 days | low–med (runs swing, no intraday day-trades) |

`rsi_mr` runs in **swing mode** (entries/exits on separate days) so it never triggers PDT, which under $25k would freeze the whole shared account.

## Architecture (Approach B: one account, three virtual ledgers)

```
                    ┌──────────────────────────────┐
                    │   Orchestrator loop           │
                    │   (run_trading_system.py)     │
                    └───────────────┬───────────────┘
                                    │ per cycle, per strategy
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │ StrategyRunner  │    │ StrategyRunner  │    │ StrategyRunner  │
   │ momentum_rot.   │    │ confirmed_mr    │    │ rsi_mr (swing)  │
   └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
            │ entry/exit signal on latest bar              │
            ▼                                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ VirtualPortfolio (one per strategy)                           │
   │  - owns a slice of paper equity (1/N split)                   │
   │  - tracks its own cash, positions (symbol→qty), realized P&L  │
   │  - emits a RiskManager.PortfolioState for sizing              │
   └───────────────────────────┬──────────────────────────────────┘
                                ▼
                    ┌────────────────────────┐
                    │ RiskManager             │  (existing, unchanged)
                    │ evaluate_entry(...)      │  sizes within the slice
                    └───────────┬─────────────┘
                                ▼
                    ┌────────────────────────┐
                    │ Executor                │  (existing)
                    │ place_*_order(strategy=) │  ONE paper Alpaca account
                    └───────────┬─────────────┘
                                ▼
                    ┌────────────────────────┐
                    │ TradeJournal            │  (existing)
                    │ strategy_attribution()   │  → horse-race report
                    └────────────────────────┘
```

## Components

### 1. `StrategyRunner` (NEW, thin)
- **Does:** wraps a backtest strategy fn. Given a trailing window of OHLCV bars (pulled from Alpaca), runs the fn and reads the **last row's** `entry`/`exit` flags to produce a live signal for the current bar.
- **Depends on:** `backtesting/strategies.STRATEGY_REGISTRY`, Alpaca historical bars (via `enhanced_alpaca_client`).
- **Interface:** `run(symbol, bars_df) -> {"action": "BUY"|"SELL"|"HOLD", "price": float}`.

### 2. `VirtualPortfolio` (NEW)
- **Does:** the ledger for one strategy. Holds allocated cash, per-symbol qty owned *by this strategy*, realized/unrealized P&L. Builds a `PortfolioState` for the risk manager. Reconciles against the real Alpaca account so the sum of virtual positions never exceeds the real account (bookkeeping owned here).
- **Key invariant:** an exit from strategy A only sells shares A bought. Two strategies may hold the same symbol; the real account nets them, the ledger keeps them separate.
- **Persists:** to sqlite / `data/journal` so state survives restarts.
- **Depends on:** `risk_manager.PortfolioState`, `trade_journal`.

### 3. Orchestrator loop (FIX `run_trading_system.py`)
- **Does:** the missing wiring. Each cycle: for each strategy → StrategyRunner produces a signal → VirtualPortfolio + RiskManager size it within the slice → Executor places the tagged paper order → check time-exits → update ledgers.
- Cadence: daily bar close for momentum/swing strategies (v1 = end-of-day, not intraday — keeps it simple and PDT-free).

### 4. Horse-race report (NEW, small)
- **Does:** reads `TradeJournal.strategy_attribution()` + the three ledgers, prints/serves the three equity curves and P&L side by side. Can reuse the existing dashboard or a simple CLI table for v1.

## What is reused unchanged
- `risk_manager.py`, `executor.py`, `trade_journal.py`, `backtesting/strategies.py`, `enhanced_alpaca_client.py`.

## Data flow per cycle
1. For each strategy with its symbol universe, fetch trailing bars from Alpaca.
2. StrategyRunner → signal on the latest bar.
3. On BUY: VirtualPortfolio proposes size → RiskManager.evaluate_entry(state=slice) → Executor.place_*(strategy=name).
4. On SELL/time-exit: Executor closes only this strategy's shares; VirtualPortfolio books realized P&L.
5. Journal every action with `strategy` tag.

## Error handling
- No Alpaca client / market closed → skip cycle, log, no crash (executor already guards this).
- Bar fetch fails for a symbol → skip that symbol this cycle.
- Ledger/real-account drift detected → halt new entries, alert, require manual reconcile (never silently trade on bad state).
- All config via `.env` (Alpaca PAPER keys only); fail fast if keys missing.

## Testing
- Unit: StrategyRunner signal extraction (last-row entry/exit) against fixture bars; VirtualPortfolio buy/sell/P&L math and the "don't sell another strategy's shares" invariant.
- Integration: one full cycle against Alpaca **paper** with a tiny universe; assert a tagged order appears in the journal and the ledger updates.
- No live-account test exists or is allowed.

## Open items deferred (YAGNI for v1)
- Intraday cadence, crypto (Binance) leg, the complex `signal_evaluator` engine, geopolitical allocation. All out of scope until the stock paper race shows signal.

## Risks / constraints
- **PDT applies to the whole paper account** (Alpaca simulates it). Mitigated by swing-only mean reversion.
- **Shared account nets positions** → the VirtualPortfolio reconciliation is the main net-new complexity and the main place bugs would hide. Covered by the invariant test above.
- Paper fills are idealized; results are directional evidence, not a profit guarantee.
