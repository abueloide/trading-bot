# Postmortem — momentum_rotation en cripto (Backlog H1)

**Fecha:** 2026-07-13 · **Veredicto:** muerta en backtest, 0 edge OOS. No desplegada.
**Gate:** FAIL (median_excess −69.12 / breadth 0.0 / median_sharpe −0.06 / min_trades 3).
**Muestra:** walk-forward 2022-01-01→2026-07-13, 6 símbolos cripto (BTC/ETH/SOL/LTC/BNB/XRP), yfinance, OOS.

## Tesis

El momentum cross-sectional funcionaría mejor en activos jóvenes y volátiles
(cripto) que en índices maduros. Origen: carrusel de @raycfu + un **smoke test
in-sample** (2026-07-12) que dio BTC-USD +33%, Sharpe 0.77, +4.56 vs SPY — el
único con excess positivo del lote. La hipótesis era que esa señal in-sample
sobreviviera al walk-forward.

## Números OOS reales (per-símbolo)

| Símbolo | Return% | Sharpe | N | vs SPY% |
|---|---|---|---|---|
| XRP-USD | 14.70 | 0.53 | 7 | −53.26 |
| BNB-USD | 0.90 | 0.08 | 5 | −67.06 |
| LTC-USD | −1.14 | −0.02 | 5 | −69.09 |
| ETH-USD | −1.18 | −0.11 | 5 | −69.14 |
| BTC-USD | −2.04 | −0.21 | 5 | −70.00 |
| SOL-USD | −6.69 | −0.47 | 3 | −74.64 |

Mediana Sharpe −0.06 · breadth 0/6 · benchmark SPY +67.96%.

## Por qué murió

**El smoke test in-sample era overfit puro.** El BTC +33% Sharpe 0.77 NO
sobrevivió al walk-forward: en OOS el mismo símbolo hizo −2.04% Sharpe −0.21.
Clásico: la señal que brilla mirando toda la serie de golpe se evapora cuando
sólo puedes usar datos pasados para decidir.

**Mecánica del colapso:** momentum_rotation (lookback 6m, skip 1m, rebalanceo
mensual) se va a **cash cuando el momentum es negativo**. El cripto-invierno
2022-2023 tuvo momentum negativo casi todo el tramo → la estrategia se sentó en
cash → retornos pegados a 0% con drawdowns pequeños (−4 a −10%). Nunca capturó
el bull 2023-2024. Resultado: ni pierde feo ni gana; simplemente no hay edge.
El único positivo (XRP) es idiosincrático —el pump 2025-26— no una señal de
momentum replicable; breadth 0/6 lo confirma.

**N bajo (3-7 trades/símbolo en 4.5 años):** consecuencia directa de estar en
cash la mayor parte del tiempo, no un bug. `min_trades` falla por eso.

## Calendario 7 días/semana (chequeo del backlog)

El engine **no se rompió** con el calendario no-bursátil de cripto: yfinance
entrega barras diarias incluyendo fines de semana y el backtest corrió limpio.
La comparación vs SPY (5 días/semana) mezcla calendarios, pero el gate lee por
Sharpe/breadth (no por excess), así que el veredicto FAIL es sólido con o sin
ese sesgo. No es un bug a esconder: documentado, sin impacto en la conclusión.

## Implicación para H3 (cartera multi-activo)

H1 falla → la pata "cripto momentum" de la cartera compuesta no aporta edge.
H3 sólo tiene sentido si H2 (commodities trend) pasa. Si H2 también falla, H3
se cancela: no hay nada que diversificar.
