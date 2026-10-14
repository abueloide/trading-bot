#!/bin/bash
# Forced-daily ICT runner (paper) — sibling of run_ict.sh, invoked by LaunchAgent
# com.luisfer.ictdailyrunner at 07:20 CST Mon-Fri. Forces ONE trade per session
# in the daily-bias direction so Luis can watch the money move every day and
# compare against the signal-selective horse. No report here: the close report
# from run_ict.sh already reads both books and shows them side by side.
cd /Users/luisfer/dev/trading-bot || exit 1
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
mkdir -p data/ict_daily
{
  echo "================ $(date '+%Y-%m-%d %H:%M:%S %Z') ================"
  ./.venv-bt/bin/python -m live.ict_daily_runner --loop
  echo "exit: $?"
} >> data/ict_daily/runner.log 2>&1
