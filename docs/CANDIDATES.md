# Candidatas event-driven

Hipótesis que **pasaron el gate del carril-event** y sobrevivieron chequeos de
disciplina. NO son edge probado ni están en paper — son las que ganan el derecho
a la siguiente validación. Método: `PLAN-event-driven.md` / `event_study.py`.

---

## C1 — GLD (oro) fade post-FOMC · 3-5 días · ❌ MUERTA EN OOS (2026-07-20)

> **Archivada.** El OOS real 2015-2021 (55 FOMC verificados) tumbó la tesis:
> exp 3d **−0.01%** / hit 44% (vs +0.79% / 67% in-sample), 5d +0.13% / hit 49%.
> Era efecto del régimen de hikes 2022-24, no anomalía del evento. NO va a paper.
> Postmortem: `docs/postmortems/2026-07-20-gld-fade-fomc-c1.md`.
> Lo de abajo queda como registro de la tesis original.


**Regla:** el día del anuncio FOMC, al cierre, tomar posición en GLD **contra** el
movimiento del día (si GLD subió en el anuncio → corto; si bajó → largo). Salir a
3-5 sesiones. Tesis: el oro (sensible a tasas/USD) **sobre-reacciona** al FOMC y
revierte en los días siguientes.

**Números (24 FOMC, 2022-2024, in-sample):**
| Ventana | Expectativa/evento | Hit | Tail ratio |
|---|---|---|---|
| 3d | +0.79% | 67% | 2.12 |
| 5d | +1.05% | 71% | 1.88 |

**Por qué sobrevive (no es espejismo):**
1. **Placebo PASA:** fadear días *random* en GLD da exp ~0.00% (hit ~48%). El oro
   NO revierte en general — revierte **específico a FOMC**. Es edge de evento, no
   mean-reversion genérica ya arbitrada. (SPY tiene señal chica; QQQ/USO son
   mayormente reversión genérica → descartados como edge de evento.)
2. **Estable en el tiempo:** aguanta en ambas mitades — 2022-1S23 (+0.73/+0.85%,
   hit 67-75%) y 2S23-2024 (+0.84/+1.24%, hit 67%). No es un solo periodo.
3. **Mecanismo económico plausible:** oro cotiza tasas reales/USD; el knee-jerk
   del anuncio sobrepasa y corrige. No es puro data-mining.

**Caveats honestos (por qué NO toca paper aún):**
- **N=24** (12 por mitad). Chico. La consistencia ayuda, no lo elimina.
- Ambas mitades caen en 2022-24 (misma era macro de hikes/cuts). **Falta OOS real
  pre-2022.**
- Fechas FOMC **hand-seeded** — verificar contra federalreserve.gov.
- Se probaron 4 símbolos × 3 ventanas × 2 modos → hay riesgo de multiple-testing;
  GLD sobrevive placebo + split, pero la barra del gate-event es laxa a propósito.

**Siguientes pasos antes de paper:** (ejecutados 2026-07-20 → mataron la candidata)
1. ~~OOS real: fechas FOMC verificadas 2015-2021 → correr GLD-fade ahí.~~ ❌ FAIL.
2. ~~Verificar las 24 fechas 2022-24 contra la Fed.~~ ✅ 24/24 correctas.
3. Definir regla ejecutable + sizing para long-shot (cuánto por evento).
4. Paper valida cableado → luego el $5k real (gatillo de Luis).

---

## C2 — OpEx 1-day drift (SPY, QQQ) · ✅ CANDIDATA · pasa gate en AMBOS regímenes (2026-07-24)

**Regla:** el día de vencimiento mensual de opciones (**3er viernes del mes**), al
cierre, tomar posición **a favor** del movimiento del día (drift/continuación) en
SPY y QQQ. Salir **1 sesión** después. El calendario es puro cómputo (3er viernes),
no API → el loop autónomo lo drena en sandbox (a diferencia de CPI/earnings).

**Por qué importa:** es la **primera hipótesis event-driven que cruza el gate en los
DOS regímenes** (ZIRP/COVID 2015-21 *y* hikes 2022-24). Las 3 variantes FOMC murieron
justo porque *ninguna celda* pasaba en ambos regímenes; aquí SPY/QQQ w=1d sí.

**Números (drift, w=1d):**
| Símbolo | OOS 2015-21 (N≈84) | IS 2022-24 (N≈35) |
|---|---|---|
| SPY | exp +0.15% · tail 1.27 · PASS | exp +0.34% · tail 1.96 · PASS |
| QQQ | exp +0.26% · tail 1.87 · PASS | exp +0.24% · tail 1.86 · PASS |

**Tesis:** cerca del vencimiento el gamma de dealers *pinnea* el precio; el flujo de
cobertura residual del día OpEx **continúa ~1 sesión** antes de disiparse. Es
microestructura (posicionamiento de dealers), no macro — por eso es más robusto al
régimen que el FOMC. La disipación se ve en los datos: 3d/5d drift es fuerte solo en
IS (rally 2022-24) y muere en OOS → efecto de régimen, se descarta; solo el w=1d
sobrevive limpio.

**Caveats honestos (por qué NO toca paper aún — es candidata débil, un peldaño BAJO C1):**
- **Edge delgado:** +0.15–0.34%/evento, ~12 eventos/año = ~2–4% bruto anual. Sensible
  a costos; falta sizing net-of-cost.
- **hit rate ~49–55% (moneda al aire):** el edge vive en la asimetría de cola, no en
  acertar dirección. Frágil.
- **Breadth 2/3:** IWM falla tail en ambos regímenes. No es amplio.
- **FALTA EL PLACEBO (lo que salvó/mató a C1):** aún no se corrió el control
  vs. días random no-OpEx. Sin eso no sabemos si es efecto **específico de OpEx** o
  mero momentum de 1 día genérico ya arbitrado. **Este es el killer test pendiente.**
- Multiple-testing: 3 símbolos × 3 ventanas × 2 modos.

**PLACEBO: ✅ SOBREVIVE (2026-07-26).** 30 remuestreos de días random no-OpEx (mismo N)
por celda. El drift de 1d en días random es **~0 o negativo en las 4 celdas** → el
efecto es **específico de OpEx**, no momentum genérico de 1 día:

| Celda | OpEx exp | Placebo medio | z | placebos ≥ real |
|---|---|---|---|---|
| SPY 2015-21 | +0.144% | −0.024% | +1.49 | 3/30 |
| SPY 2022-24 | +0.305% | −0.035% | +1.57 | 1/30 |
| QQQ 2015-21 | +0.251% | −0.076% | +2.74 | 0/30 |
| QQQ 2022-24 | +0.186% | −0.005% | +0.74 | 8/30 (débil) |

Es el **primer test-asesino que algo pasa en este proyecto** (mató a C1). Reserva
honesta: 3/4 celdas fuertes, QQQ-IS floja (pudo ser azar); signo consistente en las 4.


**COSTOS: ✅ SOBREVIVE (2026-07-26).** Neto de round-trip a 2/4/6 bps, positivo en
las 4 celdas incluso en el escenario caro (6 bps): SPY +0.084%/+0.245% por evento
(OOS/IS), QQQ +0.191%/+0.126% → **≈ +1.0% a +2.9% anual neto**.

**Reserva de expectativa (importante):** ~1-3%/año neto sobre $5k = **$50-150/año**.
C2 es un edge REAL pero PEQUEÑO y constante — NO es el long-shot de cola gorda que
busca Luis. Decisión de si vale la pena desplegarlo es suya, no técnica.

**Siguientes pasos antes de paper (para veredicto semanal de Luis):**
1. ~~**Placebo/control** vs. días random no-OpEx.~~ ✅ PASA (arriba).
2. ~~Sizing net-of-cost SPY/QQQ.~~ ✅ PASA (arriba).
3. Si sobrevive placebo → paper valida cableado. ✅ DESPLEGADA (commit `4b67a82`,
   universo **IVV + QQQ**, pata **long-only**).

---

### RE-AUDITORÍA 2026-07-31 — la barra subió después de que C2 pasó ⚠️

C2 se gateó antes de que existieran el jackknife (F3) y el placebo condicionado
(F4), y **se gateó la versión de dos patas mientras lo que corre en paper es solo
la pata larga**. Re-auditada la regla desplegada (`events/opex_audit.py`):

**La muestra real es la mitad** (solo días verdes califican): QQQ IS pasa de n=36
a **n=14 — bajo el piso `MIN_EVENTS=15`**. La expectativa por evento sube (la
pata larga es la buena) pero el N se parte.

| Celda (ticker desplegado) | n | exp | jk −top3 | años+ | placebo cond. | |
|---|---|---|---|---|---|---|
| IVV OOS 2015-21 | 36 | +0.124% | **−0.026%** | 6/7 | pct **68** | ❌ |
| IVV IS 2022-24 | 16 | +0.667% | +0.358% | 3/3 | pct 100 | ✅ |
| QQQ OOS 2015-21 | 35 | +0.426% | +0.260% | **7/7** | pct 100 | ✅ |
| QQQ IS 2022-24 | **14** | +0.727% | +0.291% | 3/3 | pct 98 | ⚠️ n<15 |

- **IVV-OOS no sobrevive:** sin los 3 mejores eventos de 36 la expectativa es
  negativa, y contra un control que también compra solo días verdes cae en el
  percentil 68 → **indistinguible de comprar cualquier día verde y salir mañana**.
  El placebo original daba ~90 porque no condicionaba el control (sesgo de F4).
- **QQQ es la celda más sólida del repo:** sobrevive jackknife, positiva 7/7 años,
  percentil 100. Único hallazgo del proyecto que aguanta un jackknife.

**C2 NO está muerta y NO se tocó el despliegue.** Decisión abierta para Luis:
sacar IVV del universo, o dejarlo como **control interno** (si IVV replica a QQQ
en vivo, la tesis de microestructura específica era mentira). Recomendación:
dejarlo como control — no cuesta nada y vale más como control que como caballo.
Detalle: `docs/postmortems/2026-07-31-c2-opex-reaudit.md`.

---

### MULTIPLICIDAD 2026-08-07 (R6) — sobrevive ✅, pero el titular de arriba es falso ❌

La regla de R5 (*con k hipótesis el null es el máximo de k*) aplicada a C2, que nació
de un barrido de **18 celdas** y midió todos sus percentiles contra el null de su
propia celda. Null = **el barrido completo corrido sobre 1,000 calendarios-placebo**.

| Familia | k | pct max-t | |
|---|---|---|---|
| La que **seleccionó** a C2 (2 patas, 3 sym × 3 w × 2 modos) | 18 | **97.8** | ✅ |
| La regla **en vivo** long-only (3 símbolos) | 3 | **99.7** | ✅ |
| La regla en vivo, familia **honesta** (+ ventanas) | 9 | **99.6** | ✅ |
| Ancla **homogénea** (sólo w=1, máximo crudo válido sin estudentizar) | 6 | 95.0 crudo | ✅ |

**C2 es lo primero del repo que sobrevive la corrección por multiplicidad** (z=+2.52
contra su propio null; estable en 3 semillas). El despliegue queda confirmado.

**Lo que muere es el argumento de venta.** Arriba se lee *"la primera hipótesis
event-driven que cruza el gate en los DOS regímenes"*, y ese fue el motivo por el que
C2 se separó de las 3 variantes FOMC. El null lo mide directo: **~50% de los
calendarios-placebo producen ≥1 celda de 18 que "pasa el gate en ambos regímenes"**.
Con un gate laxo a propósito y k=18, ese badge es una moneda al aire y no debe volver a
usarse como justificación. Lo que sostiene a C2 es la **magnitud de su celda contra su
propio null**, no el cruce de regímenes.

**Deuda anotada, no medida:** la multiplicidad **entre eventos** (OpEx se eligió después
de que FOMC muriera 3 veces). R6 corrige dentro del evento, no entre eventos.
Detalle: `docs/postmortems/2026-08-07-c2-multiplicity-r6.md`.
