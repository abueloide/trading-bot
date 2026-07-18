# Candidatas event-driven

Hipótesis que **pasaron el gate del carril-event** y sobrevivieron chequeos de
disciplina. NO son edge probado ni están en paper — son las que ganan el derecho
a la siguiente validación. Método: `PLAN-event-driven.md` / `event_study.py`.

---

## C1 — GLD (oro) fade post-FOMC · 3-5 días · PROMETEDORA (2026-07-18)

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

**Siguientes pasos antes de paper:**
1. OOS real: fechas FOMC verificadas 2015-2021 → correr GLD-fade ahí.
2. Verificar las 24 fechas 2022-24 contra la Fed.
3. Definir regla ejecutable + sizing para long-shot (cuánto por evento).
4. Paper valida cableado → luego el $5k real (gatillo de Luis).
