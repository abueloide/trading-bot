#!/bin/bash
# send_report.sh open|close — genera el reporte y lo manda por el Telegram del
# bot de trading. Determinista: sin LLM en medio (el cron del gateway no tenía
# shell en su sesión aislada, y un script que solo retransmite no necesita modelo).
cd /Users/luisfer/dev/trading-bot || exit 1
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
MODE="${1:?usage: send_report.sh open|close}"
TEXT="$(./.venv-bt/bin/python scripts/daily_report.py "--$MODE" 2>>data/ict/report.err.log)"
[ -z "$TEXT" ] && TEXT="Reporte de $MODE falló: revisa data/ict/report.err.log"
openclaw --profile trading message send --channel telegram --target 7266808827 -m "$TEXT" 2>&1 | tail -1
