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

### R1 — Re-auditoría de C2 con la barra nueva · HECHO · MIXTO ⚠️ (2026-07-31)
Con el carril COLA GORDA agotado y sin carril nuevo (decisión de opciones es de
Luis), el loop tomó la **deuda de método de F4** y la aplicó al único caballo que
este carril tiene EN VIVO. Motivo: C2 pasó el gate el 07-24, y el método subió dos
veces después (F3 jackknife, F4 placebo condicionado) — C2 nunca enfrentó ninguno.
Tercer hueco encontrado leyendo el código: **C2 se gateó en su versión de dos patas,
pero lo desplegado es solo la pata larga** → la muestra real es la MITAD (solo días
verdes califican) y **QQQ-IS corre con n=14, bajo el piso MIN_EVENTS=15**.
**IVV (la pata S&P en vivo) no sobrevive en OOS:** sin los top-3 eventos de 36 la
expectativa es negativa (−0.026%) y contra un placebo que también compra solo días
verdes cae en pct **68** → indistinguible de comprar cualquier día verde. El placebo
viejo daba ~90 porque no condicionaba el control (justo el sesgo de F4).
**QQQ sí sobrevive todo** (jk +0.260%, **7/7 años positivos**, pct 100) — primera y
única celda del repo que aguanta un jackknife. C2 no muere; se parte en dos.
**Despliegue NO tocado** (cambio en vivo = decisión de Luis). Recomendación: dejar
IVV como **control interno**, no sacarlo — si replica a QQQ en vivo, la tesis de
microestructura era mentira, y eso se aprende gratis.
Harness: `events/opex_audit.py`; `jackknife_by_event` + `leave_one_year_out` en
`event_study.py` (con tests). Postmortem: `docs/postmortems/2026-07-31-c2-opex-reaudit.md`.

> **Regla de método nueva (generaliza F4):** si la regla condiciona la entrada
> (día verde, stop, umbral de gap), **el placebo condiciona igual**. Un control sin
> condicionar mide el condicionamiento, no el evento — e infla el percentil.

### R2 — Auditoría del news reactor (el backtest que "no existía") · HECHO · FAIL ❌ (2026-08-04)
Segunda aplicación del patrón R1: auditar lo DESPLEGADO contra la barra vigente. El
reactor corre en paper desde 07-30 con la justificación de que **no se podía
backtestear** ("el histórico topa en 1 página"). **Era un bug de paginación nuestro:**
con filtro por símbolo el histórico llega a 2016 → **214,162 titulares, 31 símbolos,
1,112 días-evento**. Se midió la regla que CORRE (importando `classify`/`pick_symbol`
del módulo vivo), no una reimplementación.
**FAIL en las 6 celdas** (2 regímenes × w=1/3/5d), `tail_ratio` 0.82–1.13 contra piso
3.0. Y dos killers independientes sobre el holding desplegado (w=1d): **percentil
40.8 en OOS contra placebo condicionado** — el catalizador rinde MENOS que un día en
que el símbolo salió en noticias sin catalizador, o sea el filtro resta valor — y el
**jackknife voltea el signo** (−0.109% OOS, −0.091% IS al quitar 3 nombres). Por tipo,
`earnings_beat` es el peor de los cinco (**−0.968%**, hit 40%): cuando sale el titular
el movimiento ya ocurrió. Sesgo de supervivencia (31 líquidos vs cualquier ticker de
Benzinga) **infla** el resultado → el FAIL es más fuerte que el número.
**Despliegue NO tocado** (cambio en vivo = decisión de Luis). **Recomendación: matar el
reactor** — no es afinable, un percentil <50 dice que el filtro resta, no que esté mal
calibrado; mover `HOLD_DAYS` a 5d sería girar la perilla contra el mismo dato que la
eligió (F4). Harness: `events/news_catalyst_study.py` + `events/news_backfill.py` (con
tests). Postmortem: `docs/postmortems/2026-08-04-news-catalyst-reactor-r2.md`.

> **Regla de método nueva:** **"no se puede backtestear" es un bug abierto, no una
> propiedad del mundo** — y nunca una licencia para saltarse el gate. Si esa frase es
> lo que justifica desplegar sin validar, atacar la frase es la primera tarea. Aquí
> costó 5 días de una estrategia viva sin edge. Corolario: un placebo condicionado con
> percentil **<50** no es solo "no pasó" — es que la condición de entrada es
> activamente peor que su ausencia; un placebo random es ciego a eso.

### R3 — Auditoría del field MR en vivo (`confirmed_mr`, `rsi_mr`) · HECHO · MIXTO ⚠️ (2026-08-04)
Tercera aplicación del patrón: los dos caballos de mean-reversion llevan en paper
desde el arranque, nunca enfrentaron el método actual, y arrastraban el defecto HIGH
#3 de la auditoría 07-02 (GATED): **`live/portfolio_targets.py` llama `fn(df)` pelón
y el engine inyecta `spy_close`** ⇒ el caballo que corre NO lleva el filtro
SPY>200dMA con el que se validó. Medido: **el filtro vale +0.10 a +0.21pp por trade**
en 3 de 4 celdas, y en OOS es todo — `rsi_mr` sin filtro cae a percentil **71.5**
contra su control condicionado+duration-matched (exp +0.312% vs control +0.267%:
indistinguible de comprar cualquier día del mismo nombre en tendencia); **con**
filtro, percentil 100. Segundo hallazgo de plomería: el **filtro VIX de `rsi_mr` está
muerto en AMBOS caminos** — `extra_data["vix_rank"]` no lo llena nadie en el repo; el
caballo nunca tuvo el filtro que lleva en el nombre.
**La señal no está muerta** (expectativa positiva neta en las 4 celdas, placebo ≥91
en 3 de 4) **pero no es candidata**: el control **sin sesgo de supervivencia** (ETFs
de índice) deja a `rsi_mr`-OOS en **pct 52.5**, el jackknife k=3 sobre 58 símbolos no
prueba nada (solo 38-45/58 símbolos positivos), y **el tranche que el vivo realmente
opera** (ordena por RSI2 asc. para llenar 10 slots) rinde MENOS que el menos
sobrevendido en 5 de 8 celdas. Sin constituyentes históricos con delistados el número
no se puede limpiar → BLOCKED-DATA, no pendiente.
**Despliegue NO tocado.** Recomendación: pasar `spy_close` en el vivo (plomería, no
cambio de tesis) y renombrar/borrar el "VIX filter" inexistente.
Harness: `events/field_audit.py` (+ `tests/test_field_audit.py`).
Postmortem: `docs/postmortems/2026-08-04-field-mr-live-audit-r3.md`.

> **Reglas de método nuevas:** (1) **el jackknife escala con el número de grupos** —
> k=3 sobre 23 mata edges, sobre 58 no prueba nada; con pools grandes informa la
> *fracción* de grupos positivos. (2) **Universo estático ⇒ celda de control sin
> supervivencia obligatoria**: un backtest sobre los constituyentes de HOY mide qué
> le funcionó a los que sobrevivieron, y comprar caídas es el trade que ese sesgo más
> adorna. (3) **El placebo iguala la DURACIÓN, no solo el condicionamiento** (si la
> salida es por señal, el control sale casi de inmediato y el percentil mide tiempo
> en el mercado). (4) **Medir la regla que corre incluye medir a QUIÉN elige**: con
> más señales que slots, lo desplegado es señal + ranking + capacidad.

### R4 — Auditoría de los caballos no-MR en vivo (`momentum_rotation`, `donchian_breakout`) · HECHO · momentum FAIL ❌ / donchian MIXTO ⚠️ (2026-08-05)
Cuarta aplicación del patrón. La divergencia live↔backtest resultó **mayor que la de
R3**: no es un parámetro que se pierde, es **otra regla**. Lo registrado y gateable de
momentum es una señal POR SÍMBOLO (`entry = momentum_score > 0`, que en un bull no
selecciona nada); lo desplegado es **cross-sectional** (`momentum_top(bars, 15,
sector_of, max_per_sector=3)`, rebalanceo mensual). El **ranking nunca pasó por un
gate** y el **cap sectorial existe SÓLO en el vivo** (no hay backtest de él en el repo).
Donchian sí llama `fn(df)` idéntico, pero raciona 10 slots por fuerza de ruptura.
**`momentum_rotation` MUERE con tres killers independientes.** Contra el control
honesto para un long-only —**15 nombres al azar del mismo universo, mismos periodos**—
da percentil **49.0** en OOS (+1.540% vs +1.549%: moneda al aire) y **16.0** en IS
(+0.740% vs +1.067%: por la regla de R2, la condición de entrada **resta**). Y el
**tranche opuesto gana en ambos regímenes** (bottom-15 por score: +1.755% vs +1.540%
OOS, +1.229% vs +0.740% IS) — el score no está débilmente correlacionado, está
**invertido**. La cadena `momentum (+1.540%) ≈ random survivors (+1.549%) > índice sin
supervivencia (+1.119%)` explica el número entero: **el edge aparente ES el sesgo de
supervivencia**. En el régimen vigente pierde contra no hacer nada (control ETF
+1.013% vs +0.740%) con **4× el drawdown** y 2× el turnover; LOYO peor año 2024 → −0.042%.
**`donchian_breakout` no muere: la señal bate su placebo condicionado** (días
casi-ruptura: cierre en el decil alto del rango 20d sin superarlo) en ambos regímenes,
**84.0 / 83.5** — pero no llega al piso de 90, el jackknife se lleva la mitad en IS
(+0.874% → +0.433%, 40/60 símbolos +), y **el ranking que el vivo opera destruye
valor**: en IS el quintil de ruptura más FUERTE —lo primero que compra— rinde
**−0.534%** contra **+0.659%** del más débil. Con ~248 rupturas/año, hold ≈28d y 10
slots, el libro está permanentemente lleno **racionando hacia el peor tranche**. La
única celda del estudio que cruza 90 pct es el control ETF (92.0, maxL −6.5%) — la que
el caballo NO opera.
**Despliegue NO tocado.** Recomendación: **matar `momentum_rotation`** (no es afinable;
invertir el score sería curve-fitting contra el mismo sesgo que lo produjo, error de
F4); **no matar donchian** pero medir rankings alternativos en un estudio aparte con su
propio OOS (cambio de tesis, no plomería).
Harness: `events/trend_audit.py` (+ `tests/test_trend_audit.py`).
Postmortem: `docs/postmortems/2026-08-05-trend-horses-live-audit-r4.md`.

> **Reglas de método nuevas:** (1) **para un long-only el placebo es otra canasta del
> mismo universo, no días random** — un control de días-random le regala el retorno de
> estar invertido en un bull y cualquier regla cruza el pct 90. (2) **Si el control
> random-name iguala a la regla y ambos baten al índice sin supervivencia, lo medido es
> el sesgo**: la firma es `regla ≈ random-survivors > índice`. (3) **Una regla
> desplegada cuyo backtest registrado tiene otra forma (per-symbol vs cross-sectional)
> NO tiene backtest** — no es "validada con un defecto", es no validada. (4) **El
> tranche que elige el racionamiento puede tener el signo contrario** (2ª confirmación
> de la lección 4 de R3, ahora con signo negativo explícito). (5) **Toda perilla que
> sólo existe en el vivo es deuda de validación** — el cap sectorial vale ~20 puntos de
> percentil en OOS y nadie lo gateó.

> **Estado del loop:** sin carril PENDIENTE que drenar. COLA GORDA agotado (F1-F4),
> E2/E3 BLOCKED-DATA, FOMC y daily-bar son pozos secos declarados. **Lo que el loop
> puede hacer sin decisión de Luis es auditar lo desplegado contra la barra vigente**
> (esto fue R1, R2, R3 y R4 — patrón: **lo desplegado sin backtest vigente es la primera
> cola a drenar**). **Con R4 el field desplegado queda 100% auditado:** los 5 caballos
> (`confirmed_mr`, `rsi_mr`, `momentum_rotation`, `donchian_breakout`, `opex_drift`) más
> el news reactor pasaron por la barra vigente. **Ningún caballo del field daily-bar
> sobrevive limpio**; la única celda del repo que aguanta todo sigue siendo QQQ-OpEx (R1).
> Lo que necesita decisión: matar el reactor (R2), matar `momentum_rotation` (R4), pasar
> `spy_close` al vivo (R3), abrir carril opciones, o desbloquear E2/E3 con datos.

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
