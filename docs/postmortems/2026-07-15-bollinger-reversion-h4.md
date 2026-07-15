# Postmortem — H4: Mean-reversion de horizonte corto (`bollinger_reversion`)

**Fecha:** 2026-07-15 · **Veredicto:** FAIL ❌ (gate) · **No desplegada.**

## Tesis

Todos los caballos direccionales murieron en 2022-26 (trend_pullback, pullback_ride,
donchian_breakout, momentum cripto/commodities). La hipótesis: si el tape era
range-bound/whipsaw —lo que rompe a los trend-followers— entonces el régimen espejo
favorece a la reversión a la media, porque lo estirado a corto plazo revierte. Las MR
viejas (rsi_mr/confirmed_mr) nunca testearon esta celda: gatean sobre una línea 200d
que jamás acumula barras dentro de una ventana walk-forward OOS de 6 meses. Idea:
MR con lookbacks cortos solamente.

**Señal:** reclaim de la banda inferior de Bollinger (20/2σ) — cerró por debajo ayer,
la recupera hoy (esperar el giro, no el extremo → no cuchillo cayendo).
**Salida:** vuelta a la media (SMA20), con time-stop de 10 días.

## Números OOS reales (walk-forward, yfinance, 14 large-caps, 2022-01 → 2026-07)

Gate **FAIL**: `median_excess_pct −64.99` · `breadth_frac 0.0` · `median_sharpe 0.29`
· `min_trades 9 ✓`.

- Retornos por símbolo positivos pero minúsculos (mediana ~2-3%): AAPL +8.05,
  NVDA +8.15, PG +5.82, JPM +5.12; perdedores UNH −9.16 (WinRt 26.7%, PF 0.20),
  HD −2.46, JNJ −2.16, GOOGL −0.37.
- **Nadie** le ganó a SPY (breadth 0/14). Excess mediana −65%.
- Sharpe mediana 0.29, muy lejos de 0.80. Algunos limpios (PG 1.89, AAPL 1.49)
  pero la mayoría <0.7.

## Por qué murió

La premisa del régimen estaba **mal**. 2022-26 para large-caps NO fue choppy: fue un
bull fuerte tras el pozo de 2022 (SPY ripeó ~+65% en la ventana de benchmark). Una
estrategia de baja exposición que solo sostiene durante rebotes oversold captura
migajas y se queda sentada mientras el índice corre. La reversión a la media no perdió
plata (casi todos los símbolos verdes, PF>1 en la mayoría) — simplemente **no puede
ganarle a buy-and-hold en un mercado de una sola dirección hacia arriba**. UNH fue el
único desastre real: nombre en downtrend estructural, donde "oversold" siguió más
oversold (el failure mode clásico de MR contra tendencia bajista).

El `median_sharpe 0.29 < 0.80` falla **independiente** del defecto conocido de
alineación de ventanas del `excess_return_pct` (`project-backtest-benchmark-defect`):
aún corrigiendo el benchmark, el retorno ajustado a riesgo no llega. Veredicto limpio.

## Aprendizaje transferible

- El eje "choppy vs trending" fue el supuesto no validado: asumí régimen lateral
  porque los trend-followers murieron, pero **murieron por otra razón** (entradas/
  salidas malas, no por ausencia de tendencia). El bull era real; solo no lo capturaban.
- MR de baja exposición está estructuralmente vetada por un gate de excess-vs-SPY en
  cualquier mercado alcista. Para que MR pase este gate necesitaría o (a) un régimen
  genuinamente lateral/bajista en la ventana OOS, o (b) apalancamiento/exposición alta
  que aquí no existe. Familia MR long-only en large-caps: agotada bajo este gate.
- Próxima dirección: si se insiste en MR, testear en un universo/ventana con régimen
  lateral verdadero (p.ej. 2015-2016 o sectores defensivos en drawdown), no en
  large-caps 2022-26. O aceptar que el gate excess-vs-SPY exige capturar el beta, lo
  que empuja de vuelta hacia estrategias con exposición sostenida.
