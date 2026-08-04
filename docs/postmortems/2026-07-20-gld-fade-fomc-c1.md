# Postmortem — C1 GLD fade post-FOMC · MUERTA en OOS real (2026-07-20)

**Veredicto: NO procede a paper.** El edge era del régimen 2022-24, no del evento FOMC.

## Tesis original (docs/CANDIDATES.md, 2026-07-18)

El oro sobre-reacciona al anuncio FOMC (sensible a tasas reales/USD) y revierte en
3-5 sesiones. Regla: al cierre del día del anuncio, posición **contra** el movimiento
del día; salir a 3-5 sesiones. In-sample (24 FOMC, 2022-2024): exp +0.79% (3d) /
+1.05% (5d), hit 67-71%, tail 1.88-2.12. Pasó placebo (días random en GLD ≈ 0.00%)
y split de mitades.

## Qué se hizo ahora

1. **Verificación del calendario** — las 24 fechas 2022-24 hand-seeded se contrastaron
   contra federalreserve.gov: **correctas, 24/24**. No había error de fechas.
2. **OOS real pre-2022** — se cargaron las 55 fechas de anuncio 2015-2021 verificadas
   contra la Fed (`fomchistorical<año>.htm`). Solo juntas regulares; las de emergencia
   de marzo 2020 se excluyen (la regla necesita fecha sabida de antemano).
3. Se corrió la misma regla, sin tocar parámetros, sobre ese periodo.

## Números reales

GLD, modo fade:

| Ventana | IS 2022-24 (n=24) | **OOS 2015-21 (n=55)** | Full 2015-25 (n=79) |
|---|---|---|---|
| 1d | +0.03% · hit 58% | **−0.17% · hit 46%** | −0.11% · hit 49% |
| 3d | +0.79% · hit 67% · tail 2.12 | **−0.01% · hit 44% · tail 1.26** | +0.23% · hit 51% |
| 5d | +1.05% · hit 71% · tail 1.88 | **+0.13% · hit 49% · tail 1.21** | +0.41% · hit 56% |

## Por qué murió

- **La expectativa se cae 8x y el hit rate se vuelve moneda al aire** (71% → 49%).
  Eso no es "edge más débil fuera de muestra"; es la firma de que no había edge.
- **El 5d OOS "pasa" el gate por tecnicismo** (exp>0, tail 1.21 vs mínimo 1.20). El
  gate-event es laxo a propósito. Económicamente: +0.13%/evento × 8 eventos/año ≈
  **1% bruto anual**, antes de spread y comisiones. Muerto aunque fuera real.
- **El control drift invierte de signo entre periodos**: en OOS, drift-1d pasa
  (+0.17%) y fade-5d apenas pasa. Que la dirección "ganadora" cambie según el
  periodo y la ventana es ruido, no mecanismo.
- **Diagnóstico:** 2022-24 fue el ciclo de hikes más agresivo en 40 años; el oro
  reaccionaba violento a cada anuncio y corregía. Eso es un **efecto de régimen
  macro**, no una anomalía estable del evento FOMC. Los splits de mitades no lo
  detectaron porque **ambas mitades vivían dentro del mismo régimen** — lección de
  método: partir in-sample por la mitad NO es OOS si las dos mitades comparten
  régimen.

## Qué se salva

- Harness `events/event_study.py` con **calendario FOMC verificado 2015-2024** (79
  eventos) y rango de años en CLI (`... fade 2015:2022`) → toda hipótesis FOMC
  futura nace con OOS real disponible desde el día uno.
- Regla de método nueva para `STRATEGY-METHOD.md`: *el split IS debe cruzar régimen
  macro; si no, no cuenta como validación temporal.*

## Estado

C1 archivada. Fase 1 del pivote event-driven sigue viva — quedan **E2 (drift
post-sorpresa CPI)** y **E3 (post-earnings drift)** sin probar. No se desplegó nada.
