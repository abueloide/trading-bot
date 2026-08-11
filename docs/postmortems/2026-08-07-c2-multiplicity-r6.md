# R6 — C2 (OpEx) contra el null best-of-k: sobrevive, pero pierde su titular

**Fecha:** 2026-08-07
**Veredicto:** **PASS ✅** para C2 (primera cosa del repo que sobrevive la corrección
por multiplicidad) · **MUERE el titular** "primera hipótesis que cruza el gate en
AMBOS regímenes"
**Harness:** `events/opex_multiplicity.py` (+ `tests/test_opex_multiplicity.py`, 13 tests)
**Despliegue:** NO tocado. No hay nada que cambiar: el resultado confirma lo desplegado.

---

## Por qué este estudio

R5 (2026-08-06) dejó una regla de método marcada como **retroactiva a todos los
barridos del repo**:

> con k hipótesis, el null es el máximo de k — el mejor de 5 cruza el pct 90 el 41%
> de las veces contra un null de una sola corrida.

C2 (OpEx 1-day drift) es **la única celda del repo que aguanta todo** (placebo
condicionado de F4, jackknife de F3, leave-one-year-out, costos) y **está desplegada
en paper** (IVV + QQQ, long-only). Nació de un barrido de **3 símbolos × 3 ventanas ×
2 modos = 18 celdas** y todos sus percentiles se midieron contra el null de su propia
celda. Nunca enfrentó la corrección. Era la deuda de método más cara abierta: si el
edge se disolvía, el repo se quedaba con **cero** celdas vivas.

Patrón de siempre (R1-R5): **cuando sube la barra, se re-audita lo desplegado**.

## Método

- **El null corre el barrido COMPLETO sobre cada calendario-placebo** (1,000 sorteos).
  Diferencia con `best_of_k_null` de R5, que remuestreaba el máximo de una sola
  distribución (asume independencia): aquí las 18 celdas comparten los mismos días
  sorteados, así que la correlación entre ellas (SPY/QQQ/IWM se mueven juntos; w=1
  está contenido en w=3) queda **dentro** del null. Un null independiente sería
  demasiado exigente y el número no sería interpretable.
- **Calendario-placebo:** mismo N por régimen (84 OOS / 36 IS), sesiones no-OpEx,
  excluyendo ±3 días de un vencimiento (para que el control no caiga en la resaca del
  propio evento).
- **Score de una celda = `min(exp_OOS, exp_IS)`** — el criterio real del carril es
  "pasa en AMBOS regímenes", así que el score es su régimen más flojo.
- **Un sorteo que no produce candidata no aporta al null.** Si ninguna celda pasa el
  gate en ambos regímenes, en ese mundo el investigador declara la familia muerta y no
  reporta nada; contar su máximo inflaría el null en lugar de corregir.
- **Cuatro familias**, porque "cuál es k" es la decisión que hace o rompe este test:
  A) k=18, la familia con la que C2 se seleccionó (2 patas, w∈{1,3,5}).
  B) k=3, los símbolos de la regla long-only en vivo (la que auditó R1).
  C) k=9, la familia **honesta** del vivo: el w=1 no cayó del cielo, se heredó de la
     selección de A, así que contar sólo símbolos vuelve a subestimar la multiplicidad.
  D) k=6, **ancla homogénea**: sólo w=1, ambas patas, 3 símbolos.

## El defecto que apareció a mitad del estudio: el máximo crudo no es invariante a escala

El primer corte, con el máximo del score crudo, dio **83.6** para C2 (k=18) — reprobado.
Mirando la distribución, el máximo de cada sorteo casi siempre lo gana una celda de
**5 días**: una ventana de 5d tiene ~5× la media y ~2× el desvío de una de 1d por pura
exposición al mercado, no por ser más rara. Comparar el score de una celda de 1d contra
un máximo dominado por celdas de 5d es apples-to-oranges, y **la dirección del sesgo no
depende del resultado**: rechaza celdas chicas aunque sean genuinas.

El arreglo es estándar (max-T de Westfall-Young): **estandarizar cada celda contra su
propio null antes de tomar el máximo**. Mide "qué tan rara es esta celda para sí misma",
que es exactamente lo que la multiplicidad debe comparar.

**Para no autoengañarse** (el estadístico se cambió DESPUÉS de ver un número feo), se
agregó la familia D: sólo w=1, donde todas las celdas comparten escala y el **máximo
crudo es válido sin estudentizar**. Si el veredicto sólo apareciera al cambiar de
estadístico, sería sospechoso. No es el caso — D pasa con el crudo.

## Resultados (1,000 calendarios-placebo; estables en 3 semillas)

| Familia | k | pct ingenuo (null de su propia celda) | pct crudo best-of-k | **pct max-t** |
|---|---|---|---|---|
| A · selección de C2 (2 patas) | 18 | 99.8 | 83.6 ⚠️ sesgado por escala | **97.8** ✅ |
| B · vivo long-only (símbolos) | 3 | 99.8 | 99.7 | **99.7** ✅ |
| C · vivo long-only honesto (+ventanas) | 9 | 99.8 | 87.0 ⚠️ sesgado | **99.6** ✅ |
| D · ancla homogénea w=1 | 6 | 99.8 | **95.0** ✅ (crudo válido aquí) | 99.2 ✅ |

Semillas 20260807 / 777 / 4242 → A: 97.8 / 96.1 / 97.8 · C: 99.6 / 99.2 / 99.8 ·
D (crudo): 95.0 / 94.8 / 96.6. El veredicto no depende de la semilla.

**La celda en vivo (QQQ long-only w=1) queda en z = +2.52 contra su propio null**, y
ese z sobrevive el máximo de la familia hasta k=9.

## Lo que SÍ murió: el titular de C2

`docs/CANDIDATES.md` vendía C2 como *"la primera hipótesis event-driven que cruza el
gate en los DOS regímenes"*, y ese fue el argumento que la separó de las 3 variantes
FOMC. El null lo mide directo:

> **~50% de los calendarios-placebo producen ≥1 celda de 18 que "pasa el gate en ambos
> regímenes".**

Con k=18 celdas, cruzar el gate en ambos regímenes es **una moneda al aire**. El badge
no valía nada; era una función de cuántas celdas se probaron. Lo que sí sostiene a C2
es otra cosa: la **magnitud** de su celda contra su propio null (z=+2.5), que es lo que
mide el max-t. La conclusión sobrevive, pero **por una razón distinta a la que estaba
escrita** — y esa distinción importa porque el badge se seguía usando para justificar
el despliegue.

Nota de contexto: 50% aplica al gate del carril-event, que es **laxo a propósito**
(n≥15, exp>0, tail≥1.2). No es un defecto del gate — es que un gate laxo × k grande no
es evidencia, y el repo lo estaba leyendo como si lo fuera.

## Qué NO prueba esto

- **No agranda el edge.** C2 sigue siendo ~1-3%/año neto (~$50-150/año sobre $5k). No
  es el long-shot de cola gorda que Luis busca; sobrevivir un test no lo convierte en
  otra cosa.
- **No rehabilita a IVV.** R1 lo dejó en pct 68 contra placebo condicionado y jackknife
  negativo; R6 no lo toca. Sigue como control interno.
- **No cubre la selección del propio evento.** El calendario OpEx se eligió después de
  que FOMC muriera 3 veces. La familia honesta de "catalizadores probados" es más
  grande que 18 celdas; R6 corrige la multiplicidad **dentro** del evento, no **entre**
  eventos. Con 4 catalizadores probados (FOMC×3 + OpEx) y ~50% de tasa de falso positivo
  por familia, la corrección entre-eventos no es cosmética. Queda anotado, no medido.

## Reglas de método que deja

1. **El máximo crudo de k sólo es válido si la familia es HOMOGÉNEA en escala.** R5 lo
   usó bien (5 rankings sobre el mismo libro, mismas unidades). Cuando la familia mezcla
   ventanas/holdings/apalancamientos, el máximo lo gana siempre la celda de mayor escala
   y el test rechaza celdas chicas legítimas. **Estandarizar contra el null de cada celda
   antes de maximizar (max-t).** Aplica retroactivamente igual que la regla de R5.
2. **Si cambias de estadístico después de ver un número, ancla con una sub-familia
   donde el estadístico viejo sea válido.** Sin la familia D, este postmortem sería
   indistinguible de haber buscado el número que gusta.
3. **"Pasa en ambos regímenes" no es evidencia si no viene con su k.** Medir siempre la
   tasa a la que el null produce ≥1 celda que pasa; con gate laxo y k grande esa tasa
   se acerca a 1 y el badge no informa nada.
4. **Un hallazgo que confirma lo desplegado también se escribe.** Este loop lleva 5
   auditorías seguidas terminando en "matar/neutralizar"; registrar el PASS es lo que
   hace que el proceso sea una medición y no una máquina de matar caballos.

## Reproducir

```bash
.venv-bt/bin/python -m events.opex_multiplicity --draws 1000 [--seed S]
.venv-bt/bin/python -m events.opex_multiplicity --selfcheck
.venv-bt/bin/python -m pytest tests/test_opex_multiplicity.py -q
```
