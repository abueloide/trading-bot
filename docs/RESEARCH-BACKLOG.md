# Research backlog — cola de hipótesis para el loop backtest-first

El research loop (cron diario) drena esta cola en orden de prioridad: toma el
primer item **PENDIENTE**, lo backtestea con walk-forward + `gate.py`, escribe
postmortem si FAIL o lo marca candidata si PASS, y actualiza su estado aquí.
Método: `STRATEGY-METHOD.md`. Si la cola está vacía, el loop genera una idea nueva.

Estados: PENDIENTE · EN CURSO · HECHO (con veredicto).

## Prioridad

### H1 — Momentum en cripto  · PENDIENTE
Tesis: el momentum cross-sectional funciona mejor en activos jóvenes y volátiles
(cripto) que en índices maduros. El carrusel de @raycfu lo afirma; un smoke test
in-sample (2026-07-12) dio BTC-USD momentum +33%, Sharpe 0.77, +4.56 vs SPY —
único con excess positivo. Probar EN SERIO.
- Estrategia: `momentum_rotation` (ya existe) sobre universo cripto.
- Símbolos (≥4 para el gate): `BTC-USD,ETH-USD,SOL-USD,LTC-USD,BNB-USD,XRP-USD`.
- `--data-source yfinance`, walk-forward on. Gate normal.
- Ojo: cripto es 7 días/semana; validar que el engine no se rompa con calendario
  no-bursátil (si sesga, documentarlo — es un hallazgo, no un bug a esconder).

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

_(vacío — el loop mueve items aquí con su veredicto)_
