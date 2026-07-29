# F3 — Continuación de gap extremo · FAIL ❌ (2026-07-29)

**Carril:** COLA GORDA · **Harness:** `events/gap_study.py` (nuevo)
**Veredicto:** MUERTA. El edge pooled existe pero es de **TSLA y NVDA**, no del gap.

---

## Lo primero: se desbloqueó el BLOCKED-DATA

F3 llevaba parada por no tener calendario de earnings verificado (feed de pago).
**Mismo truco que desbloqueó E4:** el evento no se *fetchea*, se *computa*. El gap
overnight `open_t / close_{t-1} − 1` sale del OHLC que ya tenemos y es la **huella
observable** del catalizador. No identifica la causa (earnings vs M&A vs guidance)
pero sí la misma población de eventos — y la regla tradeable solo necesita la
huella: cuando ves el gap al open, ya sabes que hubo catalizador.

Resultado: 32 símbolos × 10 años, **1,140+ eventos** en el barrido. La muestra más
sana que ha tenido el carril COLA GORDA (F1 tenía ~6 episodios efectivos).

## Diseño

- **Universo:** 32 single-names líquidos listados pre-2015 (mega-cap + alta-beta).
- **Evento:** gap ≥ |8%| y ≥ |12%|, ambas patas (up y down).
- **Señal:** `sign(gap) × retorno_forward` — modo `cont` (continuación) y `rev` (reversión).
- **Entrada:** cierre del día del gap (el gap ya ocurrió al open → ejecutable, sin lookahead).
- **Ventanas:** 1/5/10/20d. **Split:** OOS 2015-21 vs IS 2022-24.
- **Dedup:** ventanas solapadas colapsadas por símbolo (lección F1).
- **Placebo:** 30 muestras de días con gap chico no nulo, por símbolo, pooled.

## Lo que se vio primero (y era mentira)

`gap UP ≥12% · cont` pasaba `gate_event` en **ambos regímenes** en 3 de 4 ventanas —
algo que ninguna hipótesis del carril F había logrado:

| celda | OOS 2015-21 | IS 2022-24 |
|---|---|---|
| w=10d | +0.99% tail 1.28 (n=53) | +2.11% tail 1.86 (n=22) |
| w=20d | +2.65% tail 1.31 (n=51) | +3.32% tail 1.57 (n=22) |

Placebo 80-97 percentil. Concentración mensual 0.09-0.12 (los gap-UP **no** se
apelotonan en una crisis — a diferencia de los gap-DOWN, conc 0.33-0.40, que son
COVID-marzo-2020 y repiten la enfermedad de F1). Se veía como candidata.

## El killer: jackknife por símbolo

`jackknife_by_group` — quitar los k símbolos que más aportan al pool:

| celda | régimen | full | sin el #1 | sin top-3 |
|---|---|---|---|---|
| ≥12% w=20d | OOS | **+2.65%** | +0.54% (sin TSLA) | **−0.76%** |
| ≥12% w=20d | IS | **+3.32%** | +1.57% (sin TSLA) | **−0.82%** |
| ≥12% w=10d | OOS | **+0.99%** | −0.64% (sin TSLA) | **−1.95%** |
| ≥12% w=10d | IS | **+2.11%** | +0.94% (sin FSLR) | **−0.83%** |
| ≥8% w=20d | OOS | +3.66% | +3.16% | +2.69% (tail 1.18) |
| ≥8% w=20d | IS | **+1.45%** | +0.84% (sin NVDA) | **−0.29%** |

**TSLA sola aporta +110% de retorno acumulado sobre 5 eventos** en la celda ≥12%
w=20d OOS, cuando el total pooled es +135%. Es el **81% del edge en 5 observaciones
de un solo nombre.** Y solo 9-13 de 23 símbolos tienen expectativa positiva: a nivel
símbolo es un volado.

El gap no predijo la continuación. **Ser TSLA en 2020 la predijo.**

## Por qué esto es un modo de falla NUEVO en el repo

H1-H5, E1-E4, F1, F2 fueron todos estudios a nivel índice o de un símbolo a la vez.
F3 es el **primer estudio pooled cross-sectional** y trajo su propia trampa: juntar
retornos de N símbolos en una distribución fabrica expectativa cuando unos pocos
nombres estuvieron en su parábola de la década. El momentum secular se filtra
disfrazado de edge de evento — y ni el placebo ni el split de régimen lo detectan
(el placebo compara contra días random *del mismo símbolo*, así que la parábola está
en ambos lados; el split de régimen no ayuda porque TSLA/NVDA subieron en los dos).

Solo el **jackknife por grupo** lo ve. Queda cableado en `event_study.py` y se
imprime automáticamente en toda celda PASS de `gap_study.py`.

Nota: el universo tiene **sesgo de supervivencia** declarado (32 tickers vivos en
2026; los gaps de las que quebraron no están). El sesgo empuja a favor de la tesis
y aun así la tesis muere — el veredicto es conservador.

## Y además: la cola nunca estuvo ahí

`tail_ratio` máximo del barrido completo con muestra real = **1.86**. El umbral del
carril COLA GORDA es **≥3**. Las únicas celdas con tail >3 tenían n=9-10 en subsets
cherry-picked. La forma de payoff que Luis busca (perder seguido, pagar enorme) **no
aparece en spot ni en el evento más violento del tape** (gap >12% en single-name).

Esto cierra el tercer intento del carril con el mismo hallazgo estructural que F1 y
F2: **spot no paga asimetría**. La cola derecha existe (p90 +12 a +19%) pero la
izquierda pesa lo mismo o más (maxL −15 a −58%), porque la pérdida no está acotada
por construcción. Lo único que acota la pérdida por construcción es una **opción
larga**. Ese es cambio de alcance → decisión de Luis.

## Estado

No desplegada. Sin ficha en `CANDIDATES.md`. Harness `events/gap_study.py` queda en
el repo (selfcheck verde) por si el carril de opciones se abre: el detector de
eventos por gap es reusable tal cual como trigger de entrada.
