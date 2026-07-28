# Postmortem — F2 Explosión de volatilidad (VIX spike) · FAIL ❌

**Fecha:** 2026-07-28 · **Carril:** COLA GORDA · **Harness:** `events/vix_study.py`
**Veredicto:** MUERTA. No desplegada. Cero celdas pasan en ambos regímenes.

## Tesis probada

Cuando el ^VIX salta ≥X% en un día, el movimiento subsecuente del índice tiene
varianza brutal → buscar la **pata con cola derecha gorda** (largo o corto), no el
promedio. Criterio del carril: `tail_ratio ≥ 3` pesa más que hit rate.

## Diseño

- **Trigger:** `^VIX` pct_change ≥ +20% (y ≥ +12% para robustez), diario, yfinance.
- **Medidos:** SPY/QQQ/IWM, ventanas 1/5/10/20d, **ambas patas** (largo y corto).
- **Split:** OOS 2015-21 (ZIRP/COVID) vs IS 2022-24 (hikes).
- **Episodios, no días** (learning de F1 → `[[feedback-event-sample-independence]]`):
  saltos separados por <5 sesiones colapsan en un episodio; piso `MIN_EPISODES=8`.
- **Placebo obligatorio:** 500 muestras random de días no-evento, mismo N y ventana.

## Qué mató la hipótesis

### 1. El signo se invierte con el régimen (mismo trigger, 12%)

| Ventana 1d | OOS 2015-21 (84 ep.) | IS 2022-24 (29 ep.) |
|---|---|---|
| SPY largo | **+0.33% PASS** (supera 99% del placebo) | −0.21% FAIL (9.6%) |
| QQQ largo | **+0.43% PASS** (99.4%) | −0.22% FAIL (15.4%) |
| IWM largo | **+0.37% PASS** (97.8%) | −0.11% FAIL (29.6%) |
| SPY/QQQ/IWM corto | FAIL (0.6–2.2%) | **PASS** (84–90%) |

En ZIRP el índice *rebota* tras el susto; en hikes *sigue cayendo*. El placebo
confirma que el evento SÍ condiciona el movimiento (percentiles 90-99 en ambos
lados) — pero lo que decide la dirección es el **régimen macro**, no el evento.
No hay regla direccional fija desplegable. Es exactamente la enfermedad de E1b/E1c.

### 2. El killer estructural: el spike de VIX no es un evento, es un termómetro

Movimiento absoluto de SPY tras el evento vs. días random (mismo N, 300 draws):

| | OOS 2015-21 | IS 2022-24 |
|---|---|---|
| w=1d | 0.92% vs 0.70% = **1.30x** | 0.63% vs 0.81% = **0.77x** |
| w=5d | 2.13% vs 1.51% = 1.41x | 1.96% vs 1.82% = 1.07x |
| w=10d | 3.09% vs 2.17% = 1.43x | 3.21% vs 2.65% = 1.21x |

**En el régimen de vol alta el evento predice un movimiento MENOR que un día al
azar (0.77x).** El "salto del VIX" es un cambio *relativo* a las últimas semanas:
solo marca algo cuando la base está tranquila. Cuando la vol ya es alta —justo el
entorno donde uno querría la cola gorda— el trigger no aporta información. La
"explosión de varianza" de la tesis es un artefacto de medir contra una base
calmada, no una propiedad del evento.

### 3. La cola sigue del lado equivocado (igual que F1)

`tail_ratio` máximo observado en todo el barrido: **2.35** (y solo en celdas de
n≈7). En las celdas con muestra real nunca pasa de ~1.6. El umbral del carril es
**≥3: cero celdas lo alcanzan.** Y la asimetría apunta a la izquierda: en OOS a
5d/10d el movimiento medio a favor es +1.79%/+2.66% contra −2.63%/−4.17% en contra.
Igual que F1: la cola derecha existe pero la izquierda es más gorda.

## Learnings (aplicables al carril, no solo a F2)

1. **Un trigger de vol relativa no es un evento; es una etiqueta de régimen.** Su
   contenido informativo se agota precisamente cuando el régimen ya es volátil.
   Antes de gastar un ciclo en un trigger derivado de vol, comparar el |mov| del
   evento contra random **por régimen** — si el ratio cae bajo 1.0 en alguno, ya
   murió y no hace falta el barrido direccional completo.
2. **La dedup por episodios funcionó y no salvó nada.** 118 días crudos → 84
   episodios en OOS (muestra sana, muy por encima del piso). Que F2 muera con N
   grande y placebo limpio es más fuerte que la muerte de F1 por N chico: aquí no
   hay excusa de muestra, el edge direccional simplemente no existe fuera del
   régimen.
3. **Segunda hipótesis seguida que apunta a opciones.** F1 y F2 concluyen lo mismo
   desde ángulos distintos: en spot no se puede cobrar una expansión *simétrica*
   de varianza, y la pérdida no está acotada. La forma de payoff que la tesis de
   Luis pide (perder seguido, pagar enorme, riesgo acotado por construcción) es
   literalmente un **straddle/strangle largo**, no una posición direccional en
   spot. El harness actual no lo puede probar. **Esto es decisión de Luis**, no del
   loop: abrir o no el carril de opciones (datos de opciones históricos + modelo de
   costo/IV) es un cambio de alcance, no una variante más.

## Qué queda vivo del trabajo

- `events/vix_study.py` (172 LOC, selfcheck verde): harness reusable para cualquier
  trigger condicional al precio con **dedup por episodios** y ambas patas. La
  función `episodes()` es lo que le faltó a F1.
- El backlog COLA GORDA queda sin items drenables por el loop: F1 y F2 muertas, F3
  BLOCKED-DATA (necesita fechas de earnings verificadas).
