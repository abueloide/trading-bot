# R5 — ¿el ranking de `donchian_breakout` informa? El racionamiento está al revés

**Fecha:** 2026-08-06
**Veredicto:** **FAIL ❌** para "existe un ranking mejor" · **el ranking del vivo está en el lado equivocado en ambos regímenes**
**Harness:** `events/ranking_study.py` (+ `tests/test_ranking_study.py`, 10 tests)
**Despliegue:** NO tocado (cambio en vivo = decisión de Luis)

---

## Por qué este estudio

Es el único item que R4 (2026-08-05) dejó abierto y que el loop puede drenar sin
decisión de Luis: *"no matar donchian pero medir rankings alternativos en un estudio
aparte con su propio OOS"*.

El hecho que lo motiva: con ~248 rupturas/año, hold ≈28d y 10 slots, el libro está
**permanentemente lleno**. Este estudio lo mide directo: **el caballo ejecuta el
8.8-10.4% de las señales que dispara**. Nueve de cada diez rupturas nunca se compran.
Lo que el caballo entrega no es la señal — es `señal + ranking + capacidad`, y el
ranking (fuerza de ruptura DESC, `live/portfolio_targets.breakout_candidates`) nunca
pasó por un gate.

**La pregunta NO era "cuál ranking gana".** Elegir el ganador de un barrido sobre el
mismo dato que produjo el hallazgo de R4 es el error de F4 (girar la perilla contra el
dato que la eligió). La pregunta fue:

> ¿algún criterio bate a **repartir los slots al azar entre las señales del día**, en
> AMBOS regímenes, después de corregir por haber probado k criterios?

## Método

- **Se simula el libro que corre**, no trades pooled: 10 slots, cierre por señal de
  salida (mínimo de 10d), sin re-entrada estando dentro, y cuando hay más señales que
  slots libres el ranking decide. Réplica de `Orchestrator._run_slot_filler` (cierra
  salidas → llena slots libres con el top del ranking). Señales importadas del módulo
  vivo (disciplina de R2).
- **NULL = prioridad al azar entre los candidatos del mismo día**, 300 corridas.
  Control más estricto que el de R4: mismo universo, mismas señales, misma capacidad,
  mismos días — **lo único que cambia es a quién le toca el slot**.
- **k=5 criterios pre-registrados** (antes de mirar resultados): `strength` (el vivo),
  `strength_inv`, `atr_strength` (fuerza ÷ ATR14), `lowvol`, `trend` (distancia a la
  SMA200).
- **Corrección por multiplicidad:** el mejor de los 5 se compara contra la
  distribución del **máximo de 5 rankings al azar**, no contra la de uno solo. Sin
  esto, el mejor de 5 cruza el "pct 90" el **41%** de las veces por puro azar
  (1−0.9⁵) — está cuantificado en `test_best_of_k_null_is_stricter_than_single_draw`.
- Dos métricas que tienen que coincidir: expectativa por trade y **retorno anual del
  libro** (`Σ retornos / slots / años`). Un criterio puede subir la expectativa por
  trade simplemente tomando menos trades; el libro no se deja.
- Muestra: 60 símbolos S&P (seed 7, la misma de R3/R4), 2014-2024. Split OOS 2015-21 /
  IS 2022-24.

## Resultados

Retorno anual del libro, y percentil contra el null de racionamiento neutral:

| criterio | OOS 2015-21 | pct | IS 2022-24 | pct |
|---|---|---|---|---|
| **`strength` (EL VIVO)** | **+16.15%** | **33.0** | **+1.70%** | **15.0** |
| `strength_inv` | +18.34% | 86.7 | +5.08% | 74.0 |
| `atr_strength` | +17.38% | 67.3 | +2.14% | 21.0 |
| `lowvol` | +15.60% | 22.0 | +1.45% | 12.3 |
| `trend` | +17.86% | 79.3 | +5.05% | 72.7 |
| **NULL (al azar)** | **+16.75%** | — | **+3.72%** | — |

**1. Ningún criterio informa.** El mejor (`strength_inv`) queda en percentil **47.5**
(OOS) y **23.2** (IS) contra el máximo de 5 rankings al azar. Contra el null simple se
veía en 86.7 / 74.0 — es decir, **el criterio ganador es exactamente lo que produce
probar cinco cosas**. Y ni siquiera con el null generoso llega a 90 en ninguno de los
dos regímenes. Encima, `strength_inv` no sobrevive sus killer tests en IS: jackknife
−3 lo vuelve **negativo** (−0.339%, 26/59 símbolos +).

**2. El ranking del vivo está en el lado equivocado, en los dos regímenes.**
`strength` es el peor o penúltimo de los cinco, y queda **debajo del racionamiento
neutral en ambos** (pct 33 y 15). El signo es consistente con el hallazgo
independiente de R4 (quintil más fuerte −0.534% vs +0.659% del más débil en IS), ahora
medido con otro harness y sobre el libro completo. En el régimen vigente el costo es
**~2.0 puntos porcentuales anuales**: +1.70%/año contra +3.72%/año de tirar una moneda.

**3. Lo que está en juego es el racionamiento, no la señal.** Con fill del 9-10%, la
diferencia entre el mejor y el peor criterio (+16.15% vs +18.34% OOS; +1.45% vs +5.08%
IS) sale entera de *a quién le toca el slot*, con la misma señal y la misma capacidad.

## Conclusión

**No hay ranking desplegable.** Pero la conclusión no es "déjalo como está": la única
afirmación que el dato sostiene en ambos regímenes es que **el criterio que corre está
del lado malo**, y la acción que se deriva de eso **no elige nada del dato** —
racionar neutral (al azar / primero en llegar entre las señales del día) es la
hipótesis nula misma. Por eso no es curve-fitting, y por eso es distinto de "usa
`strength_inv`", que sí lo sería (pct 47.5 corregido, jackknife negativo en IS).

Esto **no resucita a `donchian_breakout`**: R4 ya lo dejó en placebo 84.0/83.5 contra
un piso de 90, con el control ETF (92.0) ganándole a la celda que el caballo opera.
R5 sólo dice que, mientras siga corriendo, su perilla de racionamiento está girada al
revés.

**Recomendación (decisión de Luis, el loop no toca el vivo):** neutralizar el ranking
de `breakout_candidates` — orden arbitrario estable entre las señales del día en lugar
de fuerza DESC. Es plomería con dirección validada en dos regímenes y dos harnesses
independientes, no un cambio de tesis. Alternativa igual de válida: matar el caballo
por lo de R4 y ahorrarse la perilla.

## Reglas de método nuevas

1. **Con k hipótesis, el null es el máximo de k.** Probar 5 criterios y quedarse con el
   mejor cruza el pct 90 el 41% de las veces contra un null de una sola corrida. Todo
   barrido de variantes que reporte el percentil del ganador contra un null simple está
   reportando el percentil equivocado — aplica retroactivamente a cualquier estudio del
   repo que haya barrido parámetros y leído el mejor.
2. **Cuando la capacidad muerde, el placebo se sortea DENTRO del evento.** Con fill del
   9%, el control correcto no es otra canasta ni otros días: son las mismas señales,
   los mismos días y los mismos slots, cambiando sólo la prioridad. Aísla el
   racionamiento de la señal, que es lo único que la perilla controla.
3. **Un hallazgo negativo puede tener acción desplegable sin ser curve-fitting** — si
   la acción es *volver al null* y no *elegir al ganador*. La prueba es que la acción
   no necesita mirar el dato para especificarse.
4. **Medir el fill rate antes de discutir la señal.** 9% de fill dice que el estudio
   per-trade de la señal describe una regla que el vivo casi nunca ejecuta.
