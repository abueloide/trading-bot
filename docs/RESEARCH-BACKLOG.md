# Research backlog — cola de hipótesis para el loop backtest-first

El research loop (cron diario) drena esta cola en orden de prioridad: toma el
primer item **PENDIENTE**, lo backtestea con walk-forward + `gate.py`, escribe
postmortem si FAIL o lo marca candidata si PASS, y actualiza su estado aquí.
Método: `STRATEGY-METHOD.md`. Si la cola está vacía, el loop genera una idea nueva.

Estados: PENDIENTE · EN CURSO · HECHO (con veredicto).

## Prioridad — carril COLA GORDA (long-shot de Luis)

C2 ya está en paper. Lo que falta es lo que Luis realmente busca: **asimetría
extrema** (1% de probabilidad, pago enorme), no edges de 1-3%/año. Criterio del
carril: tail_ratio ALTO (≥3) importa más que hit rate; se acepta perder seguido.
Datos: yfinance (gratis). Método/gate: `PLAN-event-driven.md`.

### F1 — Rebote post-pánico  · PENDIENTE
Tesis: tras una caída extrema de 1 día (SPY ≤ −3%, o ≤ −4%), el rebote a 1-10
días tiene cola derecha gorda (liquidación forzada → sobre-venta). Eventos raros
(N chico por diseño) pero es exactamente la forma de payoff que Luis quiere.
- Símbolos: SPY/QQQ/IWM + high-beta (ARKK, SOXL si hay historia).
- Ventanas: 1, 5, 10, 20d. Reportar tail_ratio y percentil 90 del retorno.
- Placebo obligatorio: días random vs. días de pánico.

### F2 — Explosión de volatilidad  · PENDIENTE
Tesis: cuando el VIX salta >20% en un día, el movimiento subsecuente del índice
tiene varianza brutal. Buscar la pata con cola derecha (no el promedio).
- Datos: ^VIX + SPY por yfinance.

### F3 — Continuación de gap extremo en earnings  · PENDIENTE
Tesis: gaps >10% post-earnings continúan (underreaction en la cola).
- BLOQUEADA por datos: necesita fechas de earnings verificadas (feed de pago).

## Prioridad (histórico)

**PIVOTE 2026-07-17 → event-driven.** El daily-bar long-only está agotado (H1-H5
FAIL). El loop **NO debe generar más ideas daily-bar** (quemar Opus en pozo seco).
Dirección nueva en `PLAN-event-driven.md`. La Fase 1 (harness event-window +
E1/E2/E3) la construye el PO en sesión interactiva, no el loop autónomo (es
greenfield, no una variante de estrategia). El loop queda en hold hasta que exista
el harness event; luego se le encolan E1/E2/E3 para grindear.

- **E4** — OpEx (3er viernes/mes) drift/fade en SPY/QQQ/IWM → **HECHO · CANDIDATA ✅**
  (2026-07-24, ver abajo). **Desbloqueó el nudo BLOCKED-DATA:** OpEx es catalizador de
  ALTA frecuencia (120 eventos/10 años) pero su calendario **se computa** (3er viernes),
  no se baja de BLS/FRED → el loop autónomo SÍ lo drena en sandbox. Primera hipótesis
  event-driven que pasa el gate en AMBOS regímenes. No desplegada; falta placebo.
- **E2** — drift direccional post-sorpresa CPI (SPY/QQQ) → **PENDIENTE · BLOCKED-DATA**.
  Necesita calendario CPI verificado en `CALENDARS`. El loop autónomo NO lo puede drenar:
  BLS/FRED/ALFRED devuelven 403/timeout desde el sandbox (WebFetch y egress general los
  firewallean); solo Yahoo/DBnomics son alcanzables y DBnomics no da fechas de *publicación*.
  Cargar 120 fechas a mano sin fuente verificable repetiría el fallo que E1 nos enseñó a
  evitar. **Acción para Luis (sesión interactiva):** pegar el schedule de release CPI 2015-24
  (bls.gov/schedule/archives o FRED release_id=10 con API key) o dar una ruta de datos.
- **E3** — post-earnings drift en large-caps → **PENDIENTE · BLOCKED-DATA parcial**.
  yfinance `get_earnings_dates` solo trae ~4 trimestres recientes, no 10 años → muy pocos
  eventos por símbolo para un OOS largo. Necesita histórico de earnings por símbolo (fuente
  aparte) + soporte de calendario **por-símbolo** en el harness (hoy `CALENDARS` es lista
  compartida por evento). PO-interactivo, no loop.
- **E1c** — FOMC drift condicionado a magnitud de sorpresa (|mov|≥k×vol20d) → **HECHO ·
  FAIL ❌** (2026-07-23, ver abajo).
- **E1b** — SPY/QQQ post-FOMC drift **y** fade (mismo calendario verificado) → **HECHO ·
  FAIL ❌** (2026-07-22, ver abajo).
- **E1** — GLD/USO en ventana FOMC/OPEC → **HECHO · FAIL ❌** (ver abajo).

> **FOMC-equities agotado (3 variantes: E1b fija, E1c magnitud; E1 GLD fija).** El loop
> NO debe generar más variantes sobre el calendario FOMC — es un catalizador de baja
> frecuencia (≤79 fechas/9 años) y cualquier selector de evento thinnea la muestra bajo
> el piso N=15. La siguiente hipótesis de selección-de-evento necesita un catalizador de
> ALTA frecuencia. CPI/earnings siguen BLOCKED-DATA (ver E2/E3), **pero OpEx NO**: su
> calendario es puro cómputo (3er viernes), no fetch → drenable en sandbox (ver E4).

## Hecho

### E4 — OpEx (3er viernes) drift/fade en índices  · HECHO · CANDIDATA ✅ (2026-07-24)
Primer catalizador de **alta frecuencia drenable en sandbox**: OpEx = 3er viernes/mes,
calendario **computado** (`_third_fridays`, sin API) → 120 eventos/10 años. Corrido
drift+fade × SPY/QQQ/IWM × 1/3/5d, split OOS 2015-21 (ZIRP/COVID) vs IS 2022-24 (hikes).
**Fade muerto en todo.** 3d/5d drift = artefacto del rally de hikes (fuerte en IS, muerto
en OOS) → descartado. **Pero SPY y QQQ w=1d drift pasan el gate-event en AMBOS regímenes**
(SPY +0.15%→+0.34%, QQQ +0.26%→+0.24%; tails 1.27–1.96) — algo que NINGUNA variante FOMC
logró. Tesis: gamma de dealers pinnea en vencimiento, el flujo residual continúa ~1 sesión
y se disipa (por eso solo 1d sobrevive). Candidata **débil** (edge delgado ~2-4%/año bruto,
hit ~50% = todo cola, breadth 2/3, IWM falla). **NO desplegada.** Killer test pendiente para
veredicto semanal: **placebo vs días random no-OpEx** (¿es específico de OpEx o momentum 1d
genérico?). Ficha completa: `docs/CANDIDATES.md` C2. Filtro/harness sin cambios salvo el
calendario OPEX en `events/event_study.py`.

### E1c — FOMC drift condicionado a magnitud  · HECHO · FAIL ❌ (2026-07-23)
Rescate de E1b: filtrar a eventos con `|mov_día| ≥ 1.0×vol20d` (solo sorpresas grandes,
auto-calibrado por vol → regime-neutral). **No limpió el edge; thinneó la muestra.** N
cae a ~15-20 (roza el piso MIN_EVENTS=15) y el flip de régimen PERSISTE: hikes sigue
fadeando (3d/5d negativos) incluso restringido a movimientos grandes. Solo 1 celda pasa
(SPY 3d ZIRP) y ni coincide con los ganadores del baseline → azar de subconjunto. La
dependencia de régimen es estructural, no ruido de días chicos. Familia FOMC-equities
agotada en 3 variantes. No desplegada. Filtro `min_vol_mult` queda en el harness para
catalizadores de alta frecuencia futuros. Postmortem:
`docs/postmortems/2026-07-23-fomc-drift-magnitude-e1c.md`.

### E1b — SPY/QQQ post-FOMC drift+fade  · HECHO · FAIL ❌ (2026-07-22)
El otro lado del evento de E1: índices de equity en la ventana FOMC (calendario ya
verificado). **El signo se invierte con el régimen.** OOS 2015-21 (ZIRP): equities
*driftean* (SPY 1d/3d, QQQ 5d pasan gate-event). IS 2022-24 (hikes): equities *fadean*
(SPY/QQQ 3d/5d pasan, exp +0.44…+0.76%). **Cero celdas pasan en ambos regímenes** y el
modo ganador se voltea → no hay regla direccional fija desplegable; es reacción
condicional al régimen macro, igual que E1 con GLD. La familia "regla direccional fija
en ventana FOMC" queda agotada (GLD fade + equity drift/fade). No desplegada.
Postmortem: `docs/postmortems/2026-07-22-spy-qqq-fomc-drift-e1b.md`.

### E1 — Oro en ventana FOMC (candidata C1: GLD fade)  · HECHO · FAIL ❌ (2026-07-20)
El OOS real la mató. Se verificaron las 24 fechas 2022-24 contra federalreserve.gov
(24/24 correctas) y se cargaron las 55 fechas 2015-2021 verificadas. Misma regla, sin
retoques: exp 3d **−0.01%** / hit 44% y 5d +0.13% / hit 49%, contra +0.79%/67% y
+1.05%/71% in-sample. El hit rate colapsa a moneda al aire → no era edge, era el
régimen de hikes 2022-24. El 5d pasa el gate-event por tecnicismo (tail 1.21 vs 1.20)
pero ≈1% bruto anual antes de costos. C1 archivada, no desplegada.
Postmortem: `docs/postmortems/2026-07-20-gld-fade-fomc-c1.md`.
**Lección de método:** partir el in-sample por la mitad NO es validación temporal si
ambas mitades caen en el mismo régimen macro.

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
