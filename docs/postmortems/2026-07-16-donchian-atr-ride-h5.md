# Postmortem — H5: Donchian breakout con salida ATR-buffered

**Fecha:** 2026-07-16 · **Estrategia:** `donchian_atr_ride` · **Veredicto:** FAIL ❌ (gate) · **No desplegada**

## Tesis
`donchian_breakout` (H2) murió por whipsaw: su salida a mínimo de 10 días era
demasiado ajustada y cada pullback superficial lo sacaba antes de que la tendencia
reanudara. La *entrada* (comprar máximo local nuevo) no fue lo que lo mató — el
postmortem de H4 estableció que el tape de large-caps 2022-26 fue un **bull fuerte**,
así que los breakouts sí atrapan tendencias reales. La celda no probada era la
salida: **ensancharla con un buffer de ATR** para que se adapte a la volatilidad de
cada nombre. El stop solo dispara cuando el precio rompe el mínimo reciente por más
de 1.5×ATR — el ruido dentro de la tendencia se aguanta, una ruptura real sí sale.

- **Entrada:** close > máximo de los 20 días previos (idéntica a donchian).
- **Salida:** close < (mínimo 10 días − 1.5 × ATR(14)), usando ATR/mínimo de la
  barra previa (sin lookahead). Estrictamente más suelta que donchian.
- **Régimen esperado:** uptrends persistentes con pullbacks superficiales.

## Números OOS reales (walk-forward, 14 large-caps BALANCED, oos=6m, slippage 0.05%/lado)
Gate **FAIL**: `median_excess −61.3 / breadth 0.0 / median_sharpe 0.54 / min_trades 5`.

| Métrica | Valor | Umbral | |
|---|---|---|---|
| median_sharpe | **0.54** | ≥ 0.80 | ✗ |
| breadth_frac | 0.0 | ≥ 0.60 | ✗ |
| median_excess_pct | −61.3 | > 0 | ✗ (contaminado, ver abajo) |
| min_trades | 5 | ≥ 5 | ✓ |

Ganadores por Sharpe: GOOGL 1.45, CAT 1.36, AAPL 1.26, WMT 1.24, NVDA 0.98.
Perdedores: PG −0.89, HD −0.33, META −0.23. Drawdowns contenidos (máx −11.75% UNH).

## Por qué murió
1. **Sharpe 0.54 es el mejor de cualquier estrategia fallida hasta hoy** (bollinger
   0.29, donchian-commodities 0.21, momentum −0.06). El buffer ATR **sí levantó el
   Sharpe** frente a donchian vanilla — la tesis del whipsaw era direccionalmente
   correcta. Pero 0.54 < 0.80: no alcanza.
2. **Sigue perdiendo contra buy-and-hold** (breadth 0.0). En un bull fuerte, una
   estrategia de breakout se sienta en cash entre rupturas y sacrifica demasiada
   exposición; el baseline B&H subió +67.9% en la ventana. Comprar máximos nuevos
   captura tramos, no el compounding completo.
3. **median_excess/breadth están contaminados** por el defecto de benchmark conocido
   (`excess_return_pct` compara el retorno de la ventana OOS de 6m contra el SPY de
   rango completo multi-año → columna "vs SPY%" toda −40 a −76). No leer excess como
   señal; el killer limpio e independiente es **median_sharpe 0.54 < 0.80**.

## Aprendizaje transferible
- El buffer de volatilidad en la salida es una mejora **real** de Sharpe sobre stops
  de canal fijos — vale para futuras variantes de trend-following.
- Familia breakout long-only (donchian, donchian+ATR, ema_crossover) **agotada bajo
  este gate en large-caps 2022-26**: el problema estructural no es el timing de salida
  sino que estar-fuera-del-mercado en un bull pierde contra B&H por Sharpe. Para pasar
  el gate haría falta *más exposición* (always-in / trend-hold), no mejor timing de
  ruptura.
- Confirma el patrón de 9 hipótesis: ninguna estrategia de baja-exposición le gana al
  bull. La siguiente idea con chance debería mantenerse invertida y solo cortar
  drawdowns (trend-hold), no operar rupturas.
