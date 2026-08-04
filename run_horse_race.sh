#!/bin/bash
# Paper horse-race daily run — invoked by LaunchAgent com.luisfer.horserace
# at ~13:00 CST Mon-Fri (= 14:00-15:00 ET year-round, inside RTH).
#
# Safety: this drives ONLY the Alpaca paper account. The executor's
# is_market_open() guard turns any off-hours / holiday run into a no-op,
# and run_trading_system.py refuses to start unless ALPACA_BASE_URL is paper.
#
# Lives in ~/dev (local, NOT iCloud/Documents) so launchd can execute it
# without hitting macOS TCC "Operation not permitted".
set -uo pipefail

REPO="/Users/luisfer/dev/trading-bot"
cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }

mkdir -p data
{
  echo "================ $(date '+%Y-%m-%d %H:%M:%S %Z') ================"
  "$REPO/.venv-bt/bin/python" run_trading_system.py
  echo "exit: $?"
} >> data/cron.log 2>&1
