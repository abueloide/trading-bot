# Postmortem — trend_pullback (Field 02, idea 1)

**Fecha:** 2026-07-09 · **Veredicto:** muerta en backtest, 0 edge. No desplegada.
**Gate:** FAIL (median_excess −56.24 / breadth 0.0 / median_sharpe −0.02 / min_trades 4).
**Muestra:** walk-forward 2022-01-01→2026-07-09, 14 símbolos (preset BALANCED), sizing 25%.

## Tesis (y por qué debía ser distinta de lo que ya murió)

Las dos MR muertas (`rsi_mr`, `confirmed_mr`) **peleaban la tendencia**: compraban
sobreventa extrema (RSI2<10/15) sin exigir que el pullback hubiera girado al alza →
cuchillos cayendo. `trend_pullback` compra **con** la tendencia: sólo cuando el tramo
intermedio es alcista (SMA20>SMA50) **y** el precio acaba de reconquistar la línea
rápida desde abajo (dip resuelto), y aguanta hasta que la tendencia se rompe
(close<SMA50). Sin gatillo de sobreventa, sin apuesta contra-tendencia. Régimen
esperado: **uptrends intermedios persistentes** (el propio OOS ~2024).

Lookbacks cortos (20/50) a propósito: una línea de régimen de 200d nunca junta
suficientes barras dentro de una ventana OOS de 6 meses, así que el signal sería
inválido. 20/50 sí es evaluable dentro de la ventana.

## Números OOS reales (per-símbolo)

| Símbolo | Return% | Sharpe | N | vs SPY% |
|---|---|---|---|---|
| GOOGL | 9.99 | 1.01 | 7 | −46.04 |
| CAT | 12.98 | 0.93 | 7 | −43.06 |
| JNJ | 4.20 | 0.74 | 6 | −51.84 |
| AAPL | 3.12 | 0.52 | 9 | −52.92 |
| MSFT | 0.60 | 0.12 | 4 | −55.44 |
| WMT | 0.10 | 0.03 | 7 | −55.93 |
| AMZN | −0.14 | −0.00 | 10 | −56.17 |
| UNH | −0.28 | −0.04 | 5 | −56.31 |
| META | −1.15 | −0.19 | 4 | −57.19 |
| HD | −1.36 | −0.28 | 7 | −57.40 |
| JPM | −2.00 | −0.31 | 16 | −58.03 |
| PG | −3.71 | −0.90 | 9 | −59.75 |
| XOM | −4.00 | −0.49 | 17 | −60.03 |
| NVDA | −7.54 | −0.60 | 10 | −63.58 |

**Mediana Sharpe ≈ −0.02, breadth de returns positivos 4/14 (29%).**

## Por qué murió

1. **El signal no tiene edge propio.** Aun ignorando el benchmark, la mediana de
   Sharpe es ~0 y sólo 4/14 nombres dan return positivo. La reconquista de SMA20 en
   uptrend no bate al ruido: entra tarde (ya rebotó) y sale en cada dip a SMA50, así
   que recorta ganadores y come costos. GOOGL/CAT salvan la cara; el resto es flat.
2. **min_trades falla (4 < 5)** en MSFT y META: pocos dips resueltos limpios → señal
   escasa en algunos nombres. Confirma que el gatillo es demasiado específico.

## Hallazgo de método (transferible, más valioso que la estrategia)

El `excess_return_pct` de ESTE arnés **no es alpha**: compara el return de la
estrategia (sólo tramos OOS, sizing 25%) contra SPY buy-and-hold del **rango completo
2022→2026 (~+56%, 4.5 años, 100% invertido)**. Ventanas y exposición desalineadas →
todo da −43% a −64% por construcción, gane o pierda el signal. El `vs SPY%` es
**inservible como medida de edge** tal como está.

→ El que sí discrimina hoy es **Sharpe/return crudo + breadth**, no el excess.
Recomendación para revisión semanal con Luis (decisión de método, no la tomo solo):
benchmarkear SPY sobre **las mismas ventanas OOS** que opera la estrategia (y a
exposición comparable) antes de que `median_excess_pct` signifique algo. Hasta
entonces, leer el gate por `median_sharpe` + `breadth` de returns, que sí mataron
esta idea correctamente.

## Learnings

1. **Con-tendencia tampoco basta si el gatillo entra tarde.** Comprar la reconquista
   de la SMA rápida ya perdió el rebote; el edge, si existe, está antes (en el dip) o
   en aguantar más (no salir en cada toque de SMA50).
2. **El benchmark del arnés mide ventana equivocada.** Documentado arriba; el gate
   sigue siendo válido vía Sharpe/breadth, pero `excess` necesita fix de método.
