# Brief — Field risk-first que quepa en el Risk Gate

> **Estado:** propuesta para Luis (rumbo A, decidido 2026-07-03). NO ejecutado.
> **Autor:** loop autónomo, 2026-07-04. Cero lógica de trading tocada — esto es diseño.
> **Norte:** edge real en paper con *riesgo y disciplina por encima de todo*. El Risk Gate ES esa disciplina; el field nuevo se diseña para caber en él, no al revés.

---

## 0. Contexto (por qué existe este brief)

El primer veredicto real (α-día 5, 2026-07-03): **ningún caballo le ganó a SPY** (+1.43%). Hipótesis nula confirmada — el field actual (momentum_rotation, confirmed_mr, rsi_mr, donchian_breakout) no tiene edge. Luis eligió **rumbo A**: matar el field + cablear el `live/risk_gate.py` (patrón robado de TradingAgents) como gate previo a toda orden.

El spike de enganche (rama `feature/risk-gate-enganche`, commit `d5a80e3`) **midió** la colisión: al cablear `debate_entry` entre `evaluate_entry` y `executor.buy`, 4 tests de momentum fallan porque el gate vetea las órdenes del field. Este brief diagnostica *por qué* y diseña un field que pase.

---

## 1. Diagnóstico: qué exige el gate, exactamente

El juez de `debate_entry` vetea si **(a)** el sizing base ya negó, **(b)** el lente agresivo vetea (hard stop), o **(c)** mayoría (≥2 de 3 lentes) vetea. Los tres lentes, sobre `size_usd` y `cash` como fracción del equity:

| Lente | Vetea concentración | Vetea reserva de cash | Vetea DD intradía |
|-------|--------------------|-----------------------|-------------------|
| conservative | > 15% | deja < 30% cash | < −2% hoy |
| neutral | > 25% | deja < 15% cash | — |
| aggressive (hard stop) | > 40% | deja < 0% cash | — |

**Aprobación limpia (0 vetos)** ⇔ cada orden ≤ **15% del equity** Y deja ≥ **30% de cash**.

### El hallazgo real: no es (solo) concentración, es la reserva de cash

El sizing nominal del field YA respeta el 15% por nombre:

- `DEFAULT_SLOTS = {momentum: 15, mean_reversion: 10, breakout: 10}` (`orchestrator.py:41`)
- Cada compra = `starting_cash / slots` = $25k/15 = **6.7%** (momentum) o $25k/10 = **10%** (MR/breakout).

Ninguna orden individual pasa de 10% → la concentración por-orden **no** es el problema principal.

El problema es la **reserva de cash**. Los 4 caballos son estrategias equal-weight de despliegue ~100%: buscan llenar sus 10–15 slots. Pero el lente conservador exige dejar ≥30% de cash *tras cada compra*. Un caballo que despliega su capital secuencialmente:

- tras ~7 posiciones (70% invertido) → cash = 30% → la 8ª compra deja < 30% → **conservador vetea** (1 voto, aún aprueba);
- tras ~8.5 posiciones (< 15% cash) → **neutral se suma** → mayoría → **VETADO**.

**Conclusión:** una estrategia diseñada para estar ~totalmente invertida es estructuralmente incompatible con un gate que exige 15–30% de cash permanente. Los tests de momentum que fallan son la evidencia: escenarios de rebalanceo concentrado (pocos nombres, tamaño grande) chocan con el hard stop del 40% o con el piso de cash.

---

## 2. Rumbo recomendado: mantener el gate estricto, rediseñar el field

El gate estricto ES el activo — codifica "riesgo y disciplina por encima de todo" y una reserva de cash (dry powder) es *buena* higiene, no una limitación arbitraria. **No aflojar el gate** (la alternativa contradice rumbo A). En su lugar, el field nuevo se diseña con estas invariantes duras:

### Invariantes de diseño del field risk-first

1. **≤ 15% del equity por nombre.** Con $25k/caballo ⇒ ≤ $3,750 por posición. Sizing = `starting_cash / slots` con **slots ≥ 8** (→ ≤ 12.5%, margen bajo el 15%).
2. **Despliegue objetivo ≤ 70%.** Mantener ≥ 30% de cash de reserva por diseño, no como accidente. Con slots=8 y target 6 posiciones activas ⇒ ~75% invertido; para caber holgado, **target 5–6 posiciones de las 8 slots**, o slots=10 con deploy tope de 7.
3. **Sin apuestas concentradas.** El sector-cap actual (`MAX_PER_SECTOR = 3`) se mantiene; ningún nombre ni tema puede colarse por encima del 15%.
4. **El gate se cablea de una vez** (el PR de enganche mergea cuando el field quepa) — así toda orden pasa por la segunda opinión multi-lente antes de `executor.buy`.

### Traducción a parámetros concretos (punto de partida, no final)

| Caballo nuevo | Tesis | slots | tamaño/nombre | deploy máx | ¿pasa gate? |
|---------------|-------|-------|---------------|-----------|-------------|
| `momentum_capped` | top-N momentum, sector-cap 3 | 10, deploy ≤7 | ~10% | 70% | ✅ conservador OK |
| `mr_diversified` | mean-reversion, ≥8 nombres o no entra | 8–10, deploy ≤7 | 10–12.5% | ≤70% | ✅ |
| `breakout_buffered` | donchian con reserva de cash explícita | 10, deploy ≤7 | 10% | 70% | ✅ |
| `equal_risk` (opcional) | equal-weight defensivo, benchmark interno | 10, deploy ≤7 | 10% | 70% | ✅ |

El cambio de fondo vs el field viejo: **las estrategias ahora tienen un target de despliegue < 100% como parte de la tesis** (dry powder deliberado), en vez de intentar invertir todo el cash. Eso resuelve la colisión de raíz y añade disciplina real (capacidad de comprar dips sin vender).

---

## 3. Plan de ejecución (fases, cada una PR para Luis)

> Cambio de arquitectura + de qué opera la carrera ⇒ **NUNCA auto-ship**. Cada fase es rama + PR.

- **Fase 1 — Enganche del gate (rama `feature/risk-gate-enganche`, ya spiked).** Cablear `debate_entry` en `orchestrator._do_buy`. **Bloqueado hasta que el field quepa** (si se mergea con el field viejo, vetea órdenes reales). Reescribir los 4 tests de momentum a la semántica nueva (deploy ≤70%), no borrarlos.
- **Fase 2 — Field risk-first.** Implementar los caballos nuevos con las invariantes de §2. Añadir un tope de despliegue explícito al orchestrator (`max_deploy_frac` por caballo) — hoy no existe; es el mecanismo que garantiza la reserva de cash.
- **Fase 3 — Reset de la carrera.** `scripts/reset_race.py` con el field nuevo. Mantener los 4 caballos viejos corriendo como **baseline nulo / control** hasta que el reemplazo tenga ventana propia (no matar la señal de control antes de tener la nueva).
- **Fase 4 — Ventana de alpha nueva.** Toda la batería de guards de `checkpoint_report` (alpha-days, gate de ruido, STALE/GAP/INTRADAY, selection-bias, sample-bar) aplica igual. GATE #1 sigue pidiendo 3–6 meses; esto no acelera la decisión de dinero, solo mejora la disciplina del experimento.

---

## 4. Decisiones que son de Luis (no las tomo yo)

1. **¿Se aprueba mantener el gate estricto y rediseñar el field a deploy ≤70%?** (Rec: sí — es la lectura fiel de rumbo A.)
2. **¿Cuántos caballos en el field nuevo?** (Rec: 3–4, uno por tesis; el 4º `equal_risk` es opcional como control.)
3. **¿Se mata el field viejo al resetear, o corre en paralelo como control hasta cerrar su ventana?** (Rec: paralelo — no perder la baseline nula que ya tenemos.)

**Lo que sigue → esto:** Luis decide §4.1 (gate estricto + field a deploy ≤70%) → luego implemento Fase 2 (caballos nuevos + `max_deploy_frac`) en rama+PR → luego mergea el enganche de Fase 1.
