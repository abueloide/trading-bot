# F4 — Cola gorda con pérdida acotada (stop duro) · FAIL ❌

**Fecha:** 2026-07-30 · **Carril:** COLA GORDA · **Harness:** `events/stop_study.py`
**Veredicto:** FAIL — pero el modo de falla **corrige la conclusión que dejó F3**.

## Por qué se corrió

F1 (pánico), F2 (VIX) y F3 (gap) murieron con el mismo diagnóstico escrito en el
backlog: *"la cola derecha existe pero la izquierda pesa igual o más porque la
pérdida no está acotada; la forma de payoff que pide la tesis es una opción larga →
decisión de Luis"*. `tail_ratio` máximo de los tres estudios: **1.86**, contra el
umbral ≥3 del carril.

Esa conclusión tenía un hueco: **antes de escalar a opciones (cambio de alcance,
gasto, decisión de Luis) faltaba probar lo barato** — un **stop duro** acota la
pérdida por construcción, en spot, sin datos nuevos. Si con stop alguna celda daba
tail ≥3 con expectativa neta positiva, el carril seguía vivo sin opciones.

## Qué se midió

Las tres familias de evento ya estudiadas, ahora con salida por stop:
pánico (SPY ≤ −3%), spike de VIX (+20%) y gap UP ≥12% continuación.
Ventanas 5/10/20d × stops **3% / 5% / 8%** × split OOS 2015-21 / IS 2022-24.
Ejecución del stop sobre barras diarias, sin optimismo: si el `open` abre bajo el
stop se sale **al open** (gap-through, el stop no salva del hueco); si no, y el
`low` toca, se sale al precio del stop; si nunca toca, al cierre de la sesión w.

**Placebo con el mismo stop** (y para el gap, placebo *pooled*: cada símbolo aporta
tantos días random como eventos tuvo, y el tail se mide sobre el pool completo —
promediar placebos por-símbolo compara muestras chicas y ruidosas contra una grande).

## Resultado 1 — el stop SÍ rompe la barrera del ≥3 (la conclusión de F3 era falsa)

Primera vez en el carril que `tail_ratio` cruza 3, y no por poco:

| celda | régimen | exp | hit | tail | n |
|---|---|---|---|---|---|
| gap-UP w=20d stop=3% | OOS | +1.81% | 24% | **5.44** | 51 |
| gap-UP w=20d stop=3% | IS | +1.14% | 23% | **4.88** | 22 |
| gap-UP w=10d stop=3% | OOS | +1.77% | 32% | **3.70** | 53 |
| pánico QQQ w=20d stop=3% | OOS | +1.83% | 38% | **3.05** | 16 |

O sea: **"spot no puede dar esta forma de payoff" era incorrecto.** Sí puede —
con riesgo definido por stop. Lo que no aparece es el edge detrás.

## Resultado 2 — el `tail_ratio` es una PERILLA, no una medición

Mismo evento, misma ventana, moviendo solo el stop (gap-UP w=20d, OOS):

| stop | exp | hit | tail |
|---|---|---|---|
| 3% | +1.81% | 24% | **5.44** |
| 5% | +2.02% | 35% | 2.89 |
| 8% | +1.58% | 41% | 1.93 |

El tail se **triplica** mientras la expectativa se queda plana (±0.4pp). El stop no
mejora el payoff: recorta la pérdida media y baja el hit rate en la misma
proporción. **Un `tail_ratio` alto obtenido apretando el stop no es evidencia de
nada** — y el gate del carril (`tail_ratio ≥ 3`) es, por sí solo, **gameable**:
cualquier hipótesis muerta lo cruza con un stop suficientemente apretado.

Consecuencia para el método: en cualquier estudio con salida por stop,
(a) el placebo tiene que correr **con el mismo stop**, y (b) el binding constraint
es la **expectativa neta**, no el tail. El tail solo informa la *forma*.

## Resultado 3 — el edge sigue siendo el mismo fake de F3

Las celdas que cruzan 3 baten al placebo en tail (98.3% OOS) — pero el **jackknife
por símbolo las mata igual que en F3**:

- gap-UP w=20d stop=3% OOS: exp **+1.81% → −1.15%** sin TSLA/BA/STX; **8 de 23**
  símbolos con expectativa positiva.
- gap-UP, régimen IS: la expectativa jackknifeada es **negativa en las nueve celdas**
  (−1.5% a −2.5%); 5-7 de 15 símbolos positivos.

El stop cambia la *forma* de la distribución; no cambia de dónde viene el dinero.
Sigue viniendo de tres nombres en su parábola, no del evento.

## Resultado 4 — pánico y VIX ni siquiera llegan al banquillo

- **VIX:** ninguna celda alcanza tail 3 (máximo **1.72**), y con stops de 5-8% el
  real queda **por debajo** del placebo (percentiles de 3-20%). Con stop de 3% a 5d
  la expectativa es negativa en los tres símbolos.
- **Pánico:** las dos celdas con tail ≥3 (QQQ e IWM, 20d, stop 3%) tienen una
  expectativa que **no se distingue del placebo** (percentil 72% y 30%) — el tail
  viene del stop, no del pánico. IWM 20d/3%: tail 3.02 con expectativa **+0.01%**.
- Ambas familias son **invalidables por muestra en IS**: 6 y 7 episodios en 2022-24,
  bajo el piso de 15. Igual que en F1/F2, no hay validación cruzada de régimen
  posible para eventos de mercado raros.

## Veredicto

FAIL. Ninguna celda cumple las tres cosas a la vez (tail ≥3 **y** expectativa
positiva que bata al placebo **y** sobrevivir el jackknife) en ambos regímenes.

## Lo que cambia para el carril

1. **La razón para escalar a opciones ya no es "spot no puede dar la forma"** — sí
   puede, medido. Es que **no hay edge que llenar la forma**: en spot lo único que
   produjo cola derecha gorda fue un puñado de nombres en su parábola, y eso el
   jackknife lo desarma en los dos regímenes. Una opción larga tampoco crea edge:
   compra la forma *pagando prima*. Si la única fuente de cola derecha medida es
   idiosincrática y no persiste, comprar la forma es comprar la prima. **Esto no
   cierra la decisión de Luis, la abarata: el carril opciones necesitaría un edge
   NUEVO, no el mismo con otro envoltorio.**
2. **El gate del carril está mal calibrado.** `tail_ratio ≥ 3` como criterio
   principal es gameable con el stop. Propuesta: mantenerlo como descriptor de forma
   y mover el gate a **expectativa neta que bata el placebo (≥90 pct) + jackknife
   que sobreviva**, con el tail como filtro secundario.

## Archivos

- Harness: `events/stop_study.py` (+ `tests/test_stop_study.py`)
- Reproducir: `.venv-bt/bin/python events/stop_study.py [panic|vix|gap]`
