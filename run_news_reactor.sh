#!/bin/bash
# News reactor — corre en loop mientras el mercado está abierto (paper).
# El propio executor rechaza órdenes fuera de horario, así que el loop es seguro
# aunque arranque temprano; sale solo a las 15:05 CST para no dejar proceso vivo.
cd /Users/luisfer/dev/trading-bot || exit 1
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

END_EPOCH=$(date -j -f "%H:%M" "15:05" "+%s" 2>/dev/null)
while [ "$(date +%s)" -lt "$END_EPOCH" ]; do
  ./.venv-bt/bin/python -m live.news_reactor --once >> data/news_reactor/reactor.log 2>&1
  sleep 300   # 5 min: suficiente para un trade con salida a 1 día
done
echo "$(date) reactor: fin de sesión" >> data/news_reactor/reactor.log
