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

### F1 — Rebote post-pánico  · HECHO · FAIL ❌ (2026-07-27, ver abajo)

### F2 — Explosión de volatilidad  · HECHO · FAIL ❌ (2026-07-28, ver abajo)

### F3 — Continuación de gap extremo  · HECHO · FAIL ❌ (2026-07-29, ver abajo)
**Se desbloqueó el BLOCKED-DATA** (el gap se computa del OHLC, no se fetchea) y aun
así murió.

### F4 — Cola gorda con pérdida acotada (stop duro)  · HECHO · FAIL ❌ (2026-07-30)
Harness nuevo: `events/stop_study.py`. Probó lo barato antes de escalar a opciones:
un **stop duro** acota la pérdida en spot, sin datos nuevos. Las tres familias ya
estudiadas (pánico / VIX / gap-UP≥12%) × ventanas 5/10/20d × stops 3/5/8% × split
OOS/IS, con ejecución honesta del stop sobre barras diarias (gap-through: si abre
bajo el stop, sale al open) y **placebo con el mismo stop** (pooled para el gap).
**Corrige la conclusión de F3:** el stop **sí** cruza el umbral ≥3 del carril —
gap-UP 20d/stop-3% da tail **5.44** OOS / **4.88** IS. Spot **sí** puede producir la
forma de payoff. Pero: (1) el `tail_ratio` resulta ser una **PERILLA** — mismo evento
y ventana, moviendo solo el stop, el tail va de 5.44 (3%) a 1.93 (8%) mientras la
expectativa se queda plana (±0.4pp) → **el gate `tail≥3` es gameable, cualquier
hipótesis muerta lo cruza apretando el stop**; y (2) el **jackknife mata las celdas
igual que en F3**: +1.81% → −1.15% sin TSLA/BA/STX, 8/23 símbolos positivos, y en IS
la expectativa jackknifeada es negativa en las 9 celdas. VIX nunca pasa de tail 1.72
y con stops 5-8% queda BAJO el placebo; pánico tiene 6-7 episodios en IS (bajo el
piso de 15) → invalidable por muestra, como F1/F2.
Postmortem: `docs/postmortems/2026-07-30-cola-gorda-stop-f4.md`.

> **Carril COLA GORDA agotado: F1, F2, F3 y F4 muertas.** Las tres primeras
> convergen en que la cola izquierda pesa igual o más que la derecha; F4 mostró que
> acotar la izquierda con un stop **sí** produce `tail_ratio` ≥3 en spot — y que ahí
> **no hay nada detrás**: la única fuente de cola derecha medida en todo el carril es
> **idiosincrática** (3 nombres en su parábola), y el jackknife la desarma en ambos
> regímenes.
>
> **Decisión pendiente de Luis (el loop no la toma):** abrir carril de **opciones**
> (straddle/strangle) es cambio de alcance. Pero el argumento cambió con F4: una
> opción larga **no crea edge**, compra la forma **pagando prima**. Si la única cola
> derecha que hemos medido es idiosincrática y no persiste, comprar la forma es
> comprar la prima. El carril opciones necesitaría un **edge nuevo**, no el mismo con
> otro envoltorio. Sin esa decisión, el loop no tiene carril COLA GORDA que drenar.
>
> **Deuda de método que dejó F4 (para cuando se retome cualquier carril con stop):**
> el gate `tail_ratio ≥ 3` está mal calibrado — mantenerlo como *descriptor de forma*
> y mover el gate a **expectativa neta que bata el placebo (≥90 pct) + jackknife que
> sobreviva**. Cualquier estudio con salida por stop debe correr su placebo **con el
> mismo stop**.

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

### F3 — Continuación de gap extremo  · HECHO · FAIL ❌ (2026-07-29)
Harness nuevo: `events/gap_study.py`. **Desbloqueó el BLOCKED-DATA** con el truco de
E4: el evento no se fetchea, se computa — el gap `open_t/close_{t-1}−1` es la huella
observable del catalizador y sale del OHLC que ya tenemos. 32 single-names × 10 años,
gaps ≥|8%| y ≥|12%|, ambas patas, modos cont/rev, w 1/5/10/20d, split OOS/IS, placebo.
Muestra sanísima (1,140+ eventos; gap-UP con conc mensual 0.09-0.12 = episodios de
verdad independientes, a diferencia de los gap-DOWN que son COVID-mar-2020, conc 0.40).
**`gap UP ≥12% cont` pasaba el gate en AMBOS regímenes en 3 de 4 ventanas** (OOS +2.65%
/ IS +3.32% a 20d, tails 1.31/1.57, placebo 80-97 pct) — se veía candidata. **Lo mató el
jackknife por símbolo:** sin los top-3 contribuyentes la expectativa se vuelve NEGATIVA
en ambos regímenes (−0.76% OOS / −0.82% IS). **TSLA sola es el 81% del edge en 5
observaciones**; solo 9-13 de 23 símbolos tienen expectativa positiva.
**Modo de falla NUEVO en el repo:** primer estudio *pooled cross-sectional* → juntar
retornos de N símbolos fabrica expectativa cuando unos pocos nombres venían en su
parábola de la década. Ni el placebo ni el split de régimen lo detectan (el placebo
compara contra días random del mismo símbolo → la parábola está en ambos lados). Solo
el jackknife lo ve → cableado en `event_study.jackknife_by_group` y auto-impreso en
toda celda PASS. `tail_ratio` máximo del barrido = **1.86** (umbral del carril: ≥3).
No desplegada. Postmortem: `docs/postmortems/2026-07-29-gap-continuation-f3.md`.

### F2 — Explosión de volatilidad (VIX spike)  · HECHO · FAIL ❌ (2026-07-28)
Harness nuevo: `events/vix_study.py` (trigger ^VIX +20%/+12% diario, **ambas patas**
largo y corto, **dedup por episodios** — lo que le faltó a F1). SPY/QQQ/IWM × w
1/5/10/20d, OOS 2015-21 vs IS 2022-24, placebo de 500 muestras random.
**Muestra sana esta vez** (118 días → 84 episodios en OOS, muy sobre el piso) y aun
así **cero celdas pasan en ambos regímenes**. Dos killers:
1. **El signo se invierte con el régimen.** ZIRP: largo 1d pasa en los 3 símbolos
   (+0.33/+0.43/+0.37%, supera 97-99% del placebo). Hikes: pasa el **corto** en los
   3 y el largo muere. El placebo confirma que el evento sí condiciona el movimiento
   — pero la dirección la decide el régimen macro, no el evento (enfermedad de E1b/E1c).
2. **El spike de VIX no es un evento, es un termómetro de régimen.** |mov| del evento
   vs días random: OOS 1.30x (w=1d) pero IS **0.77x** — en vol alta el evento predice
   un movimiento MENOR que un día al azar. La "explosión de varianza" solo existe
   medida contra una base calmada.
`tail_ratio` máximo del barrido = 2.35 (en celdas n≈7); con muestra real nunca pasa
de ~1.6 → **el umbral ≥3 del carril nunca se alcanza**, y la cola gorda vuelve a
estar a la izquierda (5d OOS: +1.79% a favor vs −2.63% en contra). No desplegada.
Postmortem: `docs/postmortems/2026-07-28-explosion-vix-f2.md`.

### F1 — Rebote post-pánico  · HECHO · FAIL ❌ (2026-07-27)
Harness nuevo: `events/panic_study.py` (evento **condicional al precio**, no de
calendario; señal fija LARGO, no signo-del-día). SPY ≤ −3%/−4% × SPY/QQQ/IWM/ARKK
× w 1/5/10/20d, OOS 2015-21 vs IS 2022-24, con placebo de 500 muestras random.
**La cola está del lado equivocado:** `tail_ratio` 0.30-1.17 en OOS (placebo
~1.0-1.4) — la cola derecha existe (p90 +4…+19%) pero la izquierda es más gorda
(maxL −20…−38%). **Cero celdas pasan en ambos regímenes** (única PASS en OOS,
ARKK 20d, se voltea en IS). A −4% el trigger da N=11 en 7 años → bajo el piso
MIN_EVENTS=15 antes de mirar un retorno.
**Killer estructural:** los eventos NO son independientes — los 24 días de pánico
del OOS son ~6 episodios (15 de ellos son COVID-2020); a −4%, 9 de 13 caen en
mar-jun 2020; los 8 del IS son todos el bear de 2022. **N efectivo ≈ 5-6.** El
piso por N crudo es inválido para eventos condicionales-al-precio (la vol se
autocorrelaciona → los días se agrupan por construcción); hay que contar
**episodios**. Familia "comprar la caída de 1 día" en daily bars: agotada. La vía
viva de la tesis de asimetría sería condicionar a agotamiento del pánico, o
comprar la cola con **opciones** (pérdida acotada por construcción, que el spot no
da) — fuera del harness actual. No desplegada.
Postmortem: `docs/postmortems/2026-07-27-rebote-post-panico-f1.md`.

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
