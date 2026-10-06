#!/bin/bash
# ICT intraday runner (paper) — invoked by LaunchAgent com.luisfer.ictrunner at
# 07:20 CST Mon-Fri. That is before the NY open in both DST regimes (09:30 ET =
# 07:30 CST in summer, 08:30 CST in winter); the runner itself keys off the ET
# clock, idles until 09:30 ET and exits at 16:00 ET.
cd /Users/luisfer/dev/trading-bot || exit 1
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
mkdir -p data/ict
{
  echo "================ $(date '+%Y-%m-%d %H:%M:%S %Z') ================"
  ./.venv-bt/bin/python -m live.ict_runner --loop
  echo "exit: $?"
} >> data/ict/runner.log 2>&1
