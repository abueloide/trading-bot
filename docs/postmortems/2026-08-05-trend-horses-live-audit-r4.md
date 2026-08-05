# R4 — Auditoría de los caballos NO-MR en vivo (`momentum_rotation`, `donchian_breakout`)

**Fecha:** 2026-08-05
**Veredicto:** `momentum_rotation` **FAIL ❌** · `donchian_breakout` **MIXTO ⚠️**
**Harness:** `events/trend_audit.py` (+ `tests/test_trend_audit.py`, 9 tests)
**Despliegue:** NO tocado (cambio en vivo = decisión de Luis)

---

## Por qué esta auditoría

Cuarta aplicación del patrón R1/R2/R3: **auditar la regla que CORRE contra la barra
vigente**. Los dos caballos no-MR llevan en paper desde el arranque (momentum) y
desde el 2026-06-25 (donchian) y nunca enfrentaron el método actual: placebo
condicionado (R1), duration-matched (R3), jackknife (F3/F4), control sin
supervivencia (R3), y medición del tranche que el vivo realmente opera (R3, lección 4).

`donchian_breakout` además arrastra una anomalía de secuencia: **su familia se
declaró agotada DESPUÉS de desplegarla** — H2 (Donchian en commodities, 07-14) y H5
(variante ATR, 07-16) fallaron el gate, pero la celda que corre en paper (Donchian
20/10 sobre large-caps) nunca se gateó.

## Divergencia live↔backtest encontrada leyendo el código

Mayor que la de R3 (`spy_close`). No es un parámetro que se pierde: **es otra regla**.

| | `momentum_rotation` |
|---|---|
| **Registrado / backtesteable** | `strategy_momentum_rotation(df)` → señal POR SÍMBOLO: `entry = momentum_score > 0` |
| **Desplegado** | `orchestrator._run_momentum` → `momentum_top(bars, 15, sector_of=_sector_of, max_per_sector=3)`, rebalanceo el primer día hábil del mes, equal-weight |

`score > 0` **no selecciona nada**: en un bull la mayoría del universo lo cumple. Lo
que decide el P&L es el **ranking cross-sectional**, y el ranking nunca pasó por un
gate. El **cap sectorial `MAX_PER_SECTOR=3` existe SÓLO en el vivo** — no hay
backtest de él en ninguna parte del repo.

`donchian_breakout` sí llama `fn(df)` idéntico en ambos caminos, pero el vivo ordena
por **fuerza de ruptura** (`breakout_candidates`) para llenar 10 slots ⇒ opera el
tranche extremo, no la señal promedio.

## Método

- Muestra: 60 símbolos S&P (seed 7, la misma de R3), 2014-2024, `load_bars`/yfinance.
- Split OOS 2015-21 (ZIRP/COVID) vs IS 2022-24 (hikes).
- **Momentum:** evento = un periodo de tenencia entre rebalanceos mensuales; retorno
  = equal-weight de la canasta. Vista truncada `df.loc[:t0]` en cada rebalanceo (no
  mirar el futuro al rankear). Se importa `momentum_top` y `_sector_of` **del módulo
  vivo** (disciplina de R2: medir la regla que corre, no una reimplementación).
  - **Placebo condicionado y duration-matched:** 15 nombres **al azar del mismo pool
    elegible**, mantenidos los MISMOS periodos. Es el único control honesto para un
    long-only: un placebo de días-random mediría "estar invertido en un bull", no el
    momentum.
  - Tranche opuesto: **bottom-15 por score**, mismos periodos.
  - Celda **CON y SIN cap sectorial** (la perilla que sólo existe en el vivo).
- **Donchian:** evento = día de ruptura del máximo de 20d; salida en el primer mínimo
  de 10d (sin cap de tiempo — `max_hold_days=None` en el registry); sin re-entrada
  estando dentro.
  - **Placebo condicionado:** días **casi-ruptura** (cierre en el decil alto del rango
    previo de 20d **sin superar** el máximo). Aísla LA RUPTURA de ESTAR CERCA DEL
    MÁXIMO; duration-matched.
  - Tranche: quintil de ruptura más fuerte (lo que el vivo elige) vs el más débil.
- **Control sin supervivencia** en ambos: la misma regla sobre ETFs de índice
  (QQQ/IWM/DIA/IVV), que existieron todo el periodo.
- Jackknife k=3 por símbolo + leave-one-year-out.

---

## Resultado 1 — `momentum_rotation`: FAIL ❌

| celda | exp/mes | placebo (15 random, mismos periodos) | **pct** | bottom-15 |
|---|---|---|---|---|
| OOS 2015-21 LIVE (cap 3) | +1.540% | +1.549% | **49.0** | **+1.755%** |
| OOS 2015-21 sin cap | +1.446% | +1.549% | 29.5 | +1.755% |
| IS 2022-24 LIVE (cap 3) | +0.740% | +1.067% | **16.0** | **+1.229%** |
| IS 2022-24 sin cap | +0.771% | +1.067% | 17.0 | +1.229% |
| OOS control ETF (sin supervivencia) | +1.119% | — | — | +0.977% |
| IS control ETF (sin supervivencia) | **+1.013%** | — | — | +1.524% |

**Tres killers independientes:**

1. **Percentil 49.0 en OOS.** Contra 15 nombres al azar del mismo universo,
   mantenidos los mismos periodos, el ranking de momentum es **una moneda al aire**
   (+1.540% vs +1.549%). No es un edge delgado: es cero.
2. **Percentil 16.0 en IS.** En el régimen de hikes el momentum es **activamente
   peor** que elegir al azar. Por la regla de R2, un percentil <50 no dice "no pasó"
   — dice que la condición de entrada **resta**.
3. **El tranche opuesto gana en AMBOS regímenes.** Bottom-15 por score rinde más que
   top-15 (+1.755% vs +1.540% OOS; +1.229% vs +0.740% IS). El score no está
   débilmente correlacionado con el retorno futuro en esta muestra: está **invertido**.

**La cadena que explica el número entero:**

```
momentum (+1.540%)  ≈  15 survivors al azar (+1.549%)  >  índice sin supervivencia (+1.119%)
```

El "edge" aparente de momentum sobre el índice es **exactamente** el sesgo de
supervivencia del snapshot estático del S&P — su propio control random-name lo
iguala al decimal. Regla 2 de R3, confirmada por tercera vez.

**Y en el régimen actual pierde contra no hacer nada:** el control ETF rinde
**+1.013%/mes en IS contra +0.740%** del caballo, con **un tercio del drawdown**
(maxL −5.8% vs −7.8%) y **un quinto del turnover** (14.2% vs 31.2% mensual).
LOYO en IS: peor año 2024 → **−0.042%**.

**La única perilla que mueve el número es la que nadie validó.** El cap sectorial
—que existe sólo en el vivo— mueve el percentil OOS de 29.5 a 49.0 (≈20 puntos) y no
hace nada en IS (16 vs 17). Perilla, no edge (lección de F4). No se toca: girarla
contra este mismo dato sería el error que F4 documentó.

## Resultado 2 — `donchian_breakout`: MIXTO ⚠️

| celda | n | exp | placebo casi-ruptura | pct | quintil FUERTE (lo que opera) | quintil débil |
|---|---|---|---|---|---|---|
| OOS 2015-21 | 1734 | +1.609% | +1.395% | 84.0 | +1.466% | +1.494% |
| IS 2022-24 | 734 | +0.874% | +0.577% | 83.5 | **−0.534%** | +0.659% |
| OOS control ETF | 130 | +1.530% | +1.474% | 51.0 | +2.448% | +1.295% |
| IS control ETF | 51 | +1.424% | +0.487% | **92.0** | +0.640% | +1.651% |

**La señal no está muerta:** bate su placebo condicionado en ambos regímenes
(84.0 / 83.5) — la ruptura sí aporta sobre estar cerca del máximo sin romperlo. Pero
**no llega al piso de 90 pct** en ninguno de los dos, el jackknife se lleva la mitad
de la expectativa en IS (+0.874% → +0.433%, 40/60 símbolos positivos), y el número
single-name está inflado por supervivencia igual que el de momentum.

**El killer es el ranking que el vivo opera.** `breakout_candidates` ordena por
fuerza de ruptura DESC y llena 10 slots. En IS, ese quintil más fuerte —**lo primero
que el vivo compra**— rinde **−0.534%** contra **+0.659%** del quintil más débil. En
OOS es empate (+1.466 vs +1.494). Con ~248 rupturas/año en el universo, hold ≈28d y
10 slots, **el libro está permanentemente lleno y racionando**: lo desplegado no es
"la señal Donchian", es la señal filtrada por un criterio de racionamiento que en el
régimen actual selecciona el peor tranche.

**Y el control ETF se ve mejor que el universo desplegado:** en IS, Donchian sobre
índices da +1.424% con pct **92.0** y maxL −6.5%, contra +0.874%/pct 83.5 y maxL
−30.4% en single-names. La única celda del estudio que cruza 90 es la que el caballo
**no** opera.

---

## Qué se hace con esto (decisión de Luis, el loop no la toma)

1. **`momentum_rotation`: matar.** No es afinable — pct 49 (OOS) y 16 (IS) contra su
   propio control, tranche opuesto ganando en ambos regímenes, y perdiendo contra el
   índice en el régimen vigente con 4× el drawdown. No hay perilla honesta que girar:
   el score está invertido en esta muestra, e invertirlo sería curve-fitting contra el
   mismo sesgo que produjo el número (F4).
2. **`donchian_breakout`: no matar, arreglar el racionamiento.** La señal bate su
   placebo condicionado en ambos regímenes; lo que destruye valor es el **ranking por
   fuerza**, que nunca se validó. Camino honesto: medir rankings alternativos **en un
   estudio aparte con su propio OOS**, no girar la perilla contra estos números.
   Cambio de tesis, no plomería ⇒ decisión de Luis.
3. **Lo barato y sin tesis:** el control ETF es la celda más fuerte del estudio en
   ambos caballos. Si algo del carril trend merece paper, se parece más a
   índices que a single-names.

---

## Reglas de método que deja R4

1. **Para un long-only, el placebo es "otra canasta del mismo universo", no días
   random.** Un control de días-random le regala al long-only el retorno de estar
   invertido en un bull y cualquier regla cruza el percentil 90. El control tiene que
   competir por el mismo capital en los mismos días.
2. **Cuando el control random-name iguala a la regla y ambos baten al índice sin
   supervivencia, lo medido es el sesgo, no la regla.** La firma es la cadena
   `regla ≈ random-survivors > índice`. Ninguna celda del universo estático es
   interpretable sin la celda de ETFs al lado.
3. **Una regla desplegada cuyo backtest registrado es de otra forma (per-symbol vs
   cross-sectional) NO tiene backtest.** No es "validada con un defecto": es
   **no validada**. El gate midió una función que el vivo no llama.
4. **El tranche que el racionamiento elige es parte de la regla, y puede tener el
   signo contrario.** Con más señales que slots (aquí ~25× más), el criterio de
   desempate es tan determinante como la señal — y aquí es el que pierde el dinero.
   Segunda confirmación de la lección 4 de R3, esta vez con signo negativo explícito.
5. **Una perilla que sólo existe en el vivo mueve el resultado y no está validada.**
   El cap sectorial vale ~20 puntos de percentil en OOS. Toda perilla live-only es
   deuda de validación, no una mejora.
