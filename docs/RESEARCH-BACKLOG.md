# Research backlog — cola de hipótesis para el loop backtest-first

El research loop (cron diario) drena esta cola en orden de prioridad: toma el
primer item **PENDIENTE**, lo backtestea con walk-forward + `gate.py`, escribe
postmortem si FAIL o lo marca candidata si PASS, y actualiza su estado aquí.
Método: `STRATEGY-METHOD.md`. Si la cola está vacía, el loop genera una idea nueva.

Estados: PENDIENTE · EN CURSO · HECHO (con veredicto).

## Prioridad

### H2 — Trend-following en commodities  · PENDIENTE
Tesis: el trend-following captura las tendencias largas de materias primas mejor
que en equities (donde el mean-reversion domina). Carrusel: "trend on gold & oil".
- Estrategia: `donchian_breakout` (ya existe).
- Símbolos (ETFs, ≥4): `GLD,SLV,USO,UNG,DBA,DBC` (oro, plata, petróleo, gas, agri, broad).
- `--data-source yfinance`, walk-forward on. Gate normal.

### H3 — Cartera multi-activo compuesta  · PENDIENTE (depende de H1/H2)
Tesis: diversificar por CLASE de activo (equities mean-reversion + cripto momentum
+ commodities trend) baja correlación y sube robustez vs el field equities-only.
Solo vale construirla si H1 y/o H2 pasan el gate por separado; si ambas fallan,
la composición no las salva. Marcar según resultados de H1/H2.

## Hecho

### H1 — Momentum en cripto  · HECHO · FAIL ❌ (2026-07-13)
Muerta en backtest, 0 edge OOS. Gate FAIL (median_excess −69.12 / breadth 0.0 /
median_sharpe −0.06 / min_trades 3). El smoke test in-sample (+33% BTC) era
overfit puro: no sobrevivió walk-forward (BTC OOS −2.04%, Sharpe −0.21). La
estrategia se sienta en cash durante el cripto-invierno y nunca captura el bull;
único positivo (XRP +14.7%) es idiosincrático, breadth 0/6. Engine NO se rompió
con calendario 7d/semana. No desplegada. Postmortem: `docs/postmortems/2026-07-13-crypto-momentum-h1.md`.
