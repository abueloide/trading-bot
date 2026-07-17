# Plan — Pivote event-driven (long-shot asimétrico)

> Vigente 2026-07-17. Reemplaza el research de daily-bar long-only (pozo seco:
> H1-H5 todas FAIL). Frame: **long-shot** — $5k MXN que Luis acepta perder, con
> cola derecha gorda. No es un vehículo de preservación de capital. Norte real de
> Luis = SaaS+seguros; esto es lotería con boleto barato + inteligencia del bot.

## La verdad que ordena el plan (leer antes de construir)

El instinto de Luis — "misil → petróleo, reacciona en segundos" — apunta al activo
correcto (eventos) pero a la **fuente de edge equivocada**. Hay que separar tres:

1. **Velocidad a la noticia (latency edge). MUERTO para nosotros.** Cuando sale la
   nota, HFT la precia en microsegundos. Un bot retail (lee API → manda orden = 
   segundos) SIEMPRE llega tarde. No competimos aquí. Reconocerlo nos ahorra meses.
2. **Drift / continuación post-evento. VIVO y testeable.** Tras un shock, ¿el
   movimiento *continúa* horas/días (underreaction)? Fenómeno documentado
   (post-earnings drift, momentum post-noticia). Aquí un bot retail SÍ juega: no
   le gana al mercado a la noticia, **cabalga la continuación**. Backtesteable en
   barras horarias.
3. **Catalizadores agendados. VIVO, y GRATIS de probar.** FOMC, CPI, NFP, OPEC,
   earnings — las fechas son **públicas y sabidas de antemano**. No necesitas feed
   de noticias en tiempo real: necesitas un calendario económico (gratis) y probar
   patrones de posicionamiento alrededor del evento.

**Conclusión de secuencia:** empezamos por lo #3 (gratis, tractable), luego #2
(necesita el feed de pago), y NUNCA perseguimos #1. Así el "sí entro" al feed de
pago se gasta sólo cuando la Fase 1 gratis lo justifique.

## La epistemología del long-shot (por qué el gate cambia)

Los eventos gordos (misil→petróleo) son **raros** → muestra chica → **no se puede
probar estadísticamente** como una estrategia daily. Para un long-shot de cola
gorda, la disciplina NO es "prueba el edge con p<0.05" (imposible con N pequeño);
es **acotar la pérdida y dimensionar para asimetría**. El gate del carril event
mide: expectativa positiva, ratio de cola (ganancia media / pérdida media),
pérdida máxima acotada, y que no sea un solo trade con suerte — NO Sharpe≥0.8.
(El gate daily estricto se queda para su carril; no aplica aquí.)

## Fases

### Fase 1 — Catalizadores agendados (sin gasto, empieza ya)
- **Datos:** calendario económico gratis (FOMC/CPI/NFP/OPEC + earnings dates) +
  barras horarias yfinance (~2 años) / daily alrededor de ventanas de evento.
- **Minar:** `salvage/event-infra/geopolitical_engine.py` (759 LOC) y
  `news_intelligence.py` (485) como mapa de taxonomía de eventos, NO wired.
- **Hipótesis a probar (carril event, gate asimétrico):**
  - **E1** — Oro/petróleo (GLD/USO) en ventana ±N días de FOMC/OPEC.
  - **E2** — Índices (SPY/QQQ) drift direccional post-sorpresa CPI.
  - **E3** — Post-earnings drift en large-caps (dirección del gap → continuación).
- **Entregable:** harness de backtest por ventanas-de-evento + veredicto de las 3.

### Fase 2 — Reacción a noticias (usa el feed de pago que Luis aprobó)
- Solo si Fase 1 muestra vida. Feed de noticias timestamped (~$20-50/mes).
- Se prueba **drift post-noticia (#2)**, jamás velocidad (#1).

### Fase 3 — Paper valida el cableado → luego el $5k real (gatillo de Luis)
- Paper primero para confirmar que la plomería jala (no prender $5k sobre un bug).
- El $5k real lo decide/dispara Luis, no el bot.

## Guardrails (intactos)
Siempre paper hasta que Luis dispare real. Nunca secretos en cuarentena. El loop
autónomo no manda órdenes reales ni mensajes a humanos.

## Estado del build
- [ ] Fase 1: harness event-window + calendario gratis + E1/E2/E3. ← SIGUIENTE
- [ ] Gate carril-event (expectativa/cola, no Sharpe).
- [ ] Fase 2 (gated por Fase 1).
