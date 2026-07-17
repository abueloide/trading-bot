# Research backlog — cola de hipótesis para el loop backtest-first

El research loop (cron diario) drena esta cola en orden de prioridad: toma el
primer item **PENDIENTE**, lo backtestea con walk-forward + `gate.py`, escribe
postmortem si FAIL o lo marca candidata si PASS, y actualiza su estado aquí.
Método: `STRATEGY-METHOD.md`. Si la cola está vacía, el loop genera una idea nueva.

Estados: PENDIENTE · EN CURSO · HECHO (con veredicto).

## Prioridad

_(cola vacía — el loop genera idea nueva en el próximo ciclo)_

## Hecho

### H5 — Donchian breakout con salida ATR-buffered  · HECHO · FAIL ❌ (2026-07-16)
`donchian_atr_ride` (entrada máximo 20d, salida mínimo 10d − 1.5×ATR14). Gate FAIL
(median_excess −61.3 / breadth 0.0 / **median_sharpe 0.54** / min_trades 5). El buffer
ATR **sí levantó el Sharpe** (mejor de todas las fallidas: bollinger 0.29, donchian
0.21) — la tesis del whipsaw era correcta — pero 0.54 < 0.80. Sigue perdiendo contra
B&H: un breakout se sienta en cash entre rupturas y en un bull sacrifica exposición.
excess/breadth contaminados por el defecto de benchmark; el killer limpio es el Sharpe.
Familia breakout long-only agotada en large-caps 2022-26. Próxima idea con chance:
**trend-hold always-in** (mantenerse invertido, solo cortar drawdowns), no timing de
ruptura. No desplegada. Postmortem: `docs/postmortems/2026-07-16-donchian-atr-ride-h5.md`.

### H4 — Mean-reversion de horizonte corto para régimen choppy  · HECHO · FAIL ❌ (2026-07-15)
`bollinger_reversion` (reclaim de banda inferior 20/2σ → salida a SMA20, sin 200d).
Gate FAIL (median_excess −64.99 / breadth 0.0 / median_sharpe 0.29 / min_trades 9).
La tesis del régimen estaba mal: 2022-26 en large-caps NO fue choppy, fue un bull
fuerte. MR de baja exposición captura migajas (casi todos verdes, PF>1) pero no le
gana a buy-and-hold; Sharpe mediana 0.29 falla independiente del defecto de benchmark.
Familia MR long-only en large-caps agotada bajo este gate. No desplegada.
Postmortem: `docs/postmortems/2026-07-15-bollinger-reversion-h4.md`.

### H3 — Cartera multi-activo compuesta  · HECHO · CANCELADA ❌ (2026-07-14)
Dependía de que H1 o H2 pasaran el gate por separado. **Ambas fallaron** (0 edge
OOS). Diversificar sobre dos fuentes sin edge no produce edge; la composición no
las salva. No se construye.

### H2 — Trend-following en commodities  · HECHO · FAIL ❌ (2026-07-14)
`donchian_breakout` sobre GLD/SLV/USO/UNG/DBA/DBC. Gate FAIL (median_excess
−64.56 / breadth 0.0 / median_sharpe 0.21 / min_trades 9). No hubo tendencia
limpia 2022-26: oro/broad choppy con whipsaws, UNG desastre (−9.92%, PF 0.17),
ganadores marginales (USO/SLV/DBA Sharpe 0.28-0.51, ninguno cerca de 0.80).
Familia trend commodities agotada con Donchian 20/10. No desplegada.
Postmortem: `docs/postmortems/2026-07-14-commodities-trend-h2.md`.

### H1 — Momentum en cripto  · HECHO · FAIL ❌ (2026-07-13)
Muerta en backtest, 0 edge OOS. Gate FAIL (median_excess −69.12 / breadth 0.0 /
median_sharpe −0.06 / min_trades 3). El smoke test in-sample (+33% BTC) era
overfit puro: no sobrevivió walk-forward (BTC OOS −2.04%, Sharpe −0.21). La
estrategia se sienta en cash durante el cripto-invierno y nunca captura el bull;
único positivo (XRP +14.7%) es idiosincrático, breadth 0/6. Engine NO se rompió
con calendario 7d/semana. No desplegada. Postmortem: `docs/postmortems/2026-07-13-crypto-momentum-h1.md`.
