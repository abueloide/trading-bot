# R2 — El news reactor EN VIVO no tiene edge · 2026-08-04

**Veredicto: la regla que corre en paper desde 2026-07-30 es indistinguible del
ruido. FAIL en las dos ventanas de régimen, en las tres ventanas de holding, y
por debajo de la mediana del placebo condicionado en el holding desplegado.**

**Acción recomendada a Luis (NO ejecutada — apagar algo en vivo es decisión suya):
matar el news reactor.** Ver §Decisión.

## El backtest que "no existía" sí existía

El reactor se desplegó a paper el 2026-07-30 (`6dc28fc`) con una justificación
explícita en el código: no se podía validar hacia atrás porque *"el histórico de
noticias del tier actual topa en 1 página (medido 2026-07-21)"*.

Eso era **un bug de paginación nuestro**, no un límite del proveedor. Con el
filtro por símbolo y el cursor bien pasado, el histórico llega a 2016:
**214,162 titulares sobre 31 símbolos** (`events/news_backfill.py`).

La lección no es sobre noticias. Es que **"no se puede medir" se aceptó como
propiedad del mundo cuando era una propiedad de nuestro código**, y sobre esa
aceptación se desplegó capital (ficticio) a producción. Un "no se puede
backtestear" debe tratarse como un bug abierto hasta que alguien intente
romperlo, no como una licencia para saltarse el gate.

## Qué se midió

La **regla desplegada**, importando `classify` y `pick_symbol` de
`live/news_reactor.py` — no una reimplementación parecida (lección R1: auditar lo
que CORRE). Evento = titular con catalizador alcista y exactamente 1 ticker.
Señal = largo al cierre de la sesión del titular, salida a `w` sesiones.
**w=1 es el `HOLD_DAYS` desplegado.**

Killer tests completos: split OOS 2016-21 / IS 2022-24, **placebo condicionado**
(control = días en que el símbolo SÍ tuvo noticias pero SIN catalizador alcista),
jackknife por símbolo y por evento, leave-one-year-out, episodios-no-titulares.

## Los números

**1,112 días-evento alcistas.** Muestra grande — esto no muere por falta de N.

| Split | w | expectativa | hit | tail | percentil vs placebo | jackknife símbolos (−3) |
|---|---|---|---|---|---|---|
| OOS 2016-21 | **1d** | **+0.003%** | 51% | 0.94 | **40.8** | **−0.109%** |
| OOS 2016-21 | 3d | +0.194% | 50% | 1.13 | 46.6 | +0.054% |
| OOS 2016-21 | 5d | +0.300% | 52% | 1.10 | 44.4 | +0.112% |
| IS 2022-24 | **1d** | **+0.036%** | 55% | 0.82 | **48.9** | **−0.091%** |
| IS 2022-24 | 3d | +0.409% | 59% | 0.97 | 60.0 | +0.186% |
| IS 2022-24 | 5d | +0.485% | 56% | 1.06 | 57.3 | +0.291% |

Tres lecturas, todas malas para la regla desplegada:

1. **El percentil del holding desplegado (w=1d) está POR DEBAJO de 50 en OOS
   (40.8).** El catalizador alcista rinde *menos* que un día cualquiera en que el
   mismo símbolo salió en noticias sin catalizador. No es que el edge sea chico:
   es que el filtro de catalizador **destruye** valor respecto a simplemente
   estar en el flujo de noticias.
2. **El jackknife voltea el signo en w=1d en ambos regímenes.** Quitar 3 nombres
   (TSLA, NVDA, KO en OOS) lleva la expectativa a negativo. El poco número que
   hay vive en un puñado de símbolos — exactamente el patrón que mató a F3.
3. **`tail_ratio` falla en las 6 celdas** (0.82–1.13, piso 3.0 del carril cola
   gorda). No hay pago asimétrico que compense el hit rate de moneda al aire.

Las ventanas largas (3d/5d) se ven "mejor" pero tampoco pasan, y **no son lo que
corre**: el reactor sale en 1 sesión. Medir 5d y reportar eso sería auditar la
regla que nos gustaría tener.

## El catalizador más intuitivo es el peor

Por tipo, en el holding desplegado:

| Catalizador | n | expectativa | hit |
|---|---|---|---|
| upgrade | 690 | +0.082% | 53% |
| fda_approval | 180 | +0.094% | 54% |
| guidance_raise | 134 | −0.033% | 55% |
| acquisition_target | 66 | −0.209% | 46% |
| **earnings_beat** | **42** | **−0.968%** | **40%** |

`earnings_beat` — el caso de manual, "la empresa superó estimados, la acción
sube" — es el **peor** de los cinco, casi −1% por evento. Cuando el titular sale,
el movimiento ya ocurrió; comprar el titular es comprar la salida de alguien más.

## Sesgo que hace el FAIL más fuerte

El universo son 31 supervivientes líquidos. El reactor en vivo opera **cualquier**
ticker de Benzinga, incluidas small caps y empresas luego adquiridas. El sesgo de
supervivencia **infla** este resultado. La regla real corre en un universo peor
que el que aquí falló.

## Decisión (de Luis, no ejecutada)

El reactor sigue corriendo en paper. No lo apagué: es un cambio en vivo.

- **Recomendación: matarlo.** No es "afinable" — el percentil <50 en OOS dice que
  el filtro resta, no que esté mal calibrado. Mover `HOLD_DAYS` a 5d para
  perseguir el +0.30% sería ajustar la perilla contra el mismo dato que la eligió
  (lección F4: el stop es una perilla, no una medición).
- Riesgo de dejarlo: cero en dinero (es paper), pero **contamina el horse-race**
  — un caballo sin edge dentro de la carrera mete ruido en la comparación contra
  SPY, que es justo lo que el experimento intenta medir.

## Lecciones de método

1. **"No se puede backtestear" es un bug hasta que se demuestre lo contrario.**
   Si esa frase es la que justifica saltarse el gate, la frase misma es lo primero
   que hay que atacar. Aquí costó 5 días de una estrategia viva sin validar.
2. **El placebo condicionado puede dar percentil <50, y eso es información.** No
   es solo "no pasó el corte": significa que la condición de entrada es
   activamente peor que su ausencia. Un placebo random nunca lo habría mostrado.

Harness: `events/news_catalyst_study.py` (+ `--selfcheck`),
backfill `events/news_backfill.py`, tests `tests/test_news_catalyst_study.py`.
