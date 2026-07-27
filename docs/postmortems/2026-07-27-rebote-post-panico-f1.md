# F1 — Rebote post-pánico · FAIL ❌ (2026-07-27)

**Carril:** COLA GORDA (long-shot de Luis). **Veredicto: muerta.**
Harness: `events/panic_study.py` (nuevo). Datos: yfinance daily 2015-2024.

## La hipótesis

Tras una caída extrema de 1 día en el mercado (SPY ≤ −3% / −4%), la liquidación
forzada deja sobre-venta y el rebote a 1-20d tiene **cola derecha gorda**
(`tail_ratio ≥ 3`). Se acepta perder seguido si el pago es enorme.

Regla probada: LARGO al cierre del día de pánico, salida a w ∈ {1, 5, 10, 20}
sesiones. Trigger = SPY (pánico de mercado); medición en SPY/QQQ/IWM/ARKK.
Split OOS 2015-21 (ZIRP/COVID) vs IS 2022-24 (hikes).

## El resultado: la cola está del lado equivocado

`tail_ratio` (ganancia media / |pérdida media|) sale **por debajo de 1.0 en la
mayoría de celdas** — 0.30 a 1.17 en OOS — contra ~1.0-1.4 del placebo. No es que
la cola derecha no exista (p90 va de +4% a +19%, respetable): es que **la cola
izquierda es más gorda todavía** (maxL −20% a −38%). Comprar el primer −3% te
mete justo antes de los siguientes cinco.

De 32 celdas (4 símbolos × 4 ventanas × 2 regímenes) a −3%:

- **Cero pasan el gate-event en AMBOS regímenes.** Única celda que pasa en OOS
  (ARKK 20d: exp +8.23%, tail 1.41) se voltea en IS (exp −0.53%).
- **Cero llegan al umbral del carril** (tail ≥ 3) salvo ARKK 5d en IS (tail 3.46,
  n=8, exp +0.57%) — una celda aislada, en el régimen chico, con hit 25%. Ruido.
- A **−4%** el trigger da N=11 en 7 años OOS → **por debajo del piso MIN_EVENTS=15**
  antes de mirar un solo retorno. El umbral más duro no es testeable con daily
  bars y 10 años.

Donde la expectativa SÍ pega fuerte al placebo (SPY/QQQ/IWM 20d a −4%, percentil
100), es la ventana de 20 días — o sea, drift de mercado + la V de COVID, no un
rebote específico del pánico. Es el mismo error que mató a C1: leer el régimen
como si fuera edge.

## El killer real: los eventos no son independientes

Los 24 "eventos" de pánico del OOS son ~6 episodios:

```
2015-08-21/24 · 2016-06-24 (Brexit) · 2018-02-05/08 · 2018-10/12 ·
2019-08-05 · 2020-02-24 … 2020-10-28 (COVID: 15 de los 24)
```

A −4%, **9 de 13 eventos caen entre 2020-02-27 y 2020-06-11**. El IS 2022-24
son 8 eventos, todos del mismo bear de abr-sep 2022.

**N efectivo ≈ 5-6, no 24.** El piso `MIN_EVENTS=15` da falsa seguridad cuando
el evento es condicional al precio: los días de pánico se agrupan por
construcción (la vol se autocorrelaciona). Cualquier estadístico sobre esta
muestra está midiendo cuántas crisis distintas cayeron en la ventana, no una
distribución de payoff.

**Lección de método (nueva, para el harness):** en eventos condicionales-al-precio
hay que contar **episodios**, no días. Un gate por N crudo es inválido aquí. Ver
"Deuda" abajo.

## Deuda técnica detectada (no bloqueante)

`tail_ratio` es inestable con N chico: en el placebo de 2022-24 (n=8) la media
sale 5-21 porque muchas muestras random no tienen ni una pérdida y el ratio se
capea a 999. Los percentiles vs placebo de `tail` en ese régimen no son leíbles;
los de `expectancy` sí. Si el carril COLA GORDA sigue vivo, el placebo debería
reportar mediana en vez de media.

## Qué NO reintentar

- Familia "comprar la caída de 1 día" en índices/ETFs con daily bars: agotada.
  Ni el umbral (−3 vs −4), ni la ventana (1-20d), ni el símbolo (beta alta
  incluida) cambian el signo del problema — la cola izquierda gana.
- Umbrales más extremos (−5%, −6%): matemáticamente peor, N → 3-4.

## Qué queda vivo

Si el carril COLA GORDA se quiere seguir, la vía no es "comprar el pánico" sino
**condicionar a que el pánico ya haya parado** (señal de agotamiento) o **comprar
la cola con opciones** (pérdida acotada por construcción, que es lo que la tesis
de asimetría realmente pide y el spot no da). Ambas fuera del harness actual.
Siguiente item de la cola: **F2 — explosión de VIX**.
