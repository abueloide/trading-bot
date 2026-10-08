#!/bin/bash
# ICT intraday runner (paper) + reportes diarios — invoked by LaunchAgent
# com.luisfer.ictrunner at 07:20 CST Mon-Fri. That is before the NY open in both
# DST regimes (09:30 ET = 07:30 CST in summer, 08:30 CST in winter); the runner
# itself keys off the ET clock, idles until 09:30 ET and exits at 16:00 ET.
# The close report hangs off the runner's exit so it lands right after the bell
# in either regime without a second schedule.
cd /Users/luisfer/dev/trading-bot || exit 1
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
mkdir -p data/ict
{
  echo "================ $(date '+%Y-%m-%d %H:%M:%S %Z') ================"
  echo "open report: $(scripts/send_report.sh open)"
  START=$(date +%s)
  ./.venv-bt/bin/python -m live.ict_runner --loop
  echo "exit: $?"
  # A runner that returns within 10 min found no session today (holiday): the
  # open report already said so, don't send a close.
  if [ $(( $(date +%s) - START )) -gt 600 ]; then
    echo "close report: $(scripts/send_report.sh close)"
  fi
} >> data/ict/runner.log 2>&1
