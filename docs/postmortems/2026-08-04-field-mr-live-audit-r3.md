# R3 — Auditoría del field MR EN VIVO: el caballo que corre no es el que se validó

**Fecha:** 2026-08-04 · **Veredicto:** MIXTO ⚠️ (bug confirmado con costo medido; señal
no muerta pero tampoco candidata) · **Harness:** `events/field_audit.py` (+ tests)
**Despliegue NO tocado** — cambiar lo que corre en paper es decisión de Luis.

## Por qué este item

Tercera aplicación del patrón que dejaron R1 y R2: **lo desplegado sin backtest
vigente es la primera cola a drenar**. Los dos caballos de mean-reversion
(`confirmed_mr`, `rsi_mr`) llevan en paper desde el arranque de la carrera, nunca
enfrentaron el método actual (placebo condicionado de R1, jackknife de F3/F4), y
arrastran un defecto HIGH abierto desde la auditoría del 2026-07-02 (#3, GATED):

    live/portfolio_targets.py:oversold_candidates  →  fn(df)                 (pelón)
    backtesting/engine.py:413                      →  fn(df, spy_close=SPY)  (filtro)

O sea: **el caballo que opera en paper no lleva el filtro SPY>200dMA con el que se
backtesteó.** El veredicto α-día 5 midió estrategias que no coinciden con sus
backtests, y nadie había medido cuánto vale esa diferencia.

## Qué se midió

Evento = día en que dispara `entry`. Holding = como el vivo: sale al primer día con
señal de salida (RSI2 > umbral) o a `max_hold_days`, sin re-entrada mientras está
dentro. Muestra: 60 símbolos S&P (seed 7), 2015-2024, split OOS 2015-21 / IS 2022-24.
Cada celda corre en las dos semánticas (LIVE = `fn(df)`, BACKTEST = con `spy_close`),
más una celda de control sobre ETFs de índice.

## Hallazgo 1 — el filtro que el vivo tira SÍ vale, y en OOS lo vale todo

| celda | LIVE (lo que corre) | BACKTEST (lo validado) |
|---|---|---|
| `confirmed_mr` OOS 15-21 | +0.280% · pct **91.0** | +0.384% · pct 96.5 |
| `confirmed_mr` IS 22-24 | +0.575% · pct 100 | +0.628% · pct 100 |
| `rsi_mr` OOS 15-21 | +0.312% · pct **71.5** | +0.525% · pct **100** |
| `rsi_mr` IS 22-24 | +0.389% · pct 99 | +0.330% · pct 98 |

(expectativa por trade, bruta; placebo condicionado + duration-matched, 200 sorteos)

El filtro suma entre **+0.10 y +0.21pp por trade** en 3 de 4 celdas. El caso que
importa es `rsi_mr` en OOS: **sin el filtro cae a percentil 71.5** — su expectativa
(+0.312%) queda casi encima de su propio control (+0.267%), o sea comprar RSI2<10 en
un nombre en tendencia es apenas distinguible de comprar cualquier día de ese mismo
nombre en tendencia, sostenido los mismos días. **Con** el filtro, percentil 100.

No es un detalle cosmético de plomería: es la diferencia entre una regla que bate a
su control y una que no. Arreglarlo es pasar `spy_close` en `oversold_candidates`.

## Hallazgo 2 — el filtro VIX de `rsi_mr` nunca existió

`strategy_rsi_mr_vix` recibe `vix_rank_series`, y `backtesting/engine.py:415` lo
inyecta desde `extra_data["vix_rank"]`… que **no lo llena nadie en todo el repo**
(`grep vix_rank` → la firma, esa línea, y un umbral en `config.py`). El filtro cae a
all-True en el backtest **y** en el vivo. El caballo se llama "RSI(2) MR + VIX
Filter" y nunca tuvo filtro VIX en ninguno de los dos caminos. La única divergencia
real live↔backtest es `spy_close`.

## Hallazgo 3 — la señal no está muerta, pero el número del universo estático no es interpretable solo

Per-trade, las dos reglas sobreviven la barra vigente en el universo S&P estático:
expectativa positiva neta de costos (0.10% ida y vuelta) en las 4 celdas, placebo
≥91 en 3 de 4, jackknife que aguanta, LOYO mayormente positivo. Es más de lo que
sobrevivió cualquier cosa en este repo desde C2/QQQ. **Y aun así no es candidata**,
por tres razones:

1. **Supervivencia.** El universo son los constituyentes de HOY, y comprar caídas es
   exactamente el trade que ese sesgo adorna: toda caída de un sobreviviente se
   recuperó. La celda de control sobre ETFs (sin ese sesgo) es mucho más floja —
   `rsi_mr` OOS cae a **pct 52.5** (nada), `confirmed_mr` IS a 67.5. Lo único que
   replica fuera del sesgo es `confirmed_mr` en OOS (pct 90.5, exp +0.396%).
2. **El jackknife aquí es débil.** Quitar 3 de 58 símbolos no es el killer test que
   fue en F3 (3 de 23). Sobrevivirlo no acredita nada. El dato honesto del mismo
   corte: solo **38-45 de 58 símbolos** son positivos (65-76%).
3. **El tranche que el vivo opera no es el mejor.** El vivo ordena por RSI(2)
   ascendente y llena 10 slots ⇒ opera la cola MÁS sobrevendida. En **5 de 8**
   celdas ese quintil rinde MENOS que el quintil menos sobrevendido (p.ej.
   `confirmed_mr` IS LIVE: +0.422% el más sobrevendido vs +0.897% el menos). La
   regla de asignación de slots no está justificada por el dato.

Además, expectativa por trade ≠ retorno de portafolio: 10 slots, cash drag y el
timing de ejecución (13:00 CST, no al cierre) viven fuera de esta medición. Que la
señal mida positivo per-trade y el caballo pierda contra SPY en vivo no es una
contradicción — son dos cosas distintas, y esta auditoría solo cubre la primera.

## Recomendaciones (ninguna se ejecuta sin Luis)

1. **Pasar `spy_close` en el vivo** (`oversold_candidates`) — es el defecto #3, ahora
   con precio: +0.10/+0.21pp por trade, y saca a `rsi_mr`-OOS de indistinguible.
   Es plomería, no un cambio de tesis: alinea el vivo con lo que se validó.
2. **Renombrar o borrar el "VIX filter"** de `rsi_mr`. Un nombre que promete un filtro
   inexistente es cómo se cuela un supuesto no validado (mismo mecanismo que "no se
   puede backtestear" en R2).
3. **No tocar la asignación por RSI(2) con este dato.** Girar esa perilla contra el
   mismo corte que la evaluó es F4 otra vez. Si se toca, se re-mide aparte.
4. **No abrir carril de daily-bar MR** sobre esto. Sin datos de constituyentes
   históricos (delistados incluidos) el número está sesgado y no hay forma de
   limpiarlo con yfinance. Eso es BLOCKED-DATA, no un pendiente.

## Reglas de método que deja R3

> **El jackknife tiene que escalar con el número de grupos.** k=3 sobre 23 símbolos
> mata edges; k=3 sobre 58 no prueba nada. Con pools grandes, el dato que informa es
> la **fracción de grupos positivos**, no el jackknife de top-k.

> **Estudio sobre universo estático ⇒ celda de control sin supervivencia, obligatoria.**
> Un backtest sobre los constituyentes de hoy mide "qué le funcionó a los que
> sobrevivieron". El control (ETFs de índice, o cualquier serie que exista completa)
> dice cuánto del número es el sesgo. Aquí el sesgo se comió el edge de `rsi_mr`.

> **El placebo debe igualar la DURACIÓN, no solo el condicionamiento.** Si la salida
> es por señal, un control no-condicionado sale casi de inmediato: comparar 1 día
> contra 5 en un mercado que sube mide tiempo en el mercado. Extensión directa de la
> regla de R1.

> **Medir la regla que corre incluye medir a QUIÉN elige.** Cuando hay más señales
> que slots, lo desplegado es señal + ranking + capacidad. Medir solo la señal
> promedio audita una regla que nadie está operando.
