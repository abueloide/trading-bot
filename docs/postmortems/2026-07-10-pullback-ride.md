# Postmortem — pullback_ride (Field 02, idea 2)

**Fecha:** 2026-07-10 · **Veredicto:** muerta en backtest, 0 edge. No desplegada.
**Gate:** FAIL (median_excess −56.61 / breadth 0.0 / **median_sharpe 0.09** / min_trades 7 ✓).
**Muestra:** walk-forward 2022-01-01→2026-07-10, 14 símbolos (preset BALANCED), sizing 25%, OOS 6m.

## Tesis (y por qué debía ser distinta de lo que ya murió)

El postmortem de `trend_pullback` nombró dos fallas concretas: entraba **tarde**
(compraba sólo tras reconquistar SMA20, ya con el rebote hecho) y salía **temprano**
(bailaba en cada toque de SMA50, recortando ganadores). Donchian probó la entrada
opuesta (comprar máximos nuevos, aguantar hasta mínimo de N días) — tampoco edge.
La celda sin probar era: comprar **el dip mismo** —mientras el precio sigue bajo la
línea rápida pero se sostiene sobre la lenta dentro de un uptrend (debilidad dentro
de fuerza, más temprano que una reconquista)— y aguantar con un stop de mínimo móvil
en vez de salir en SMA50, para que los ganadores corran hasta una ruptura real.

- **Entrada:** SMA20>SMA50 (uptrend) ∧ close≤SMA20 (en el dip, no extendido) ∧ close>SMA50 (dip contenido, no cuchillo).
- **Salida:** close < mínimo de los `trail_lookback`=10 bars previos (ruptura de tendencia), no el primer beso de SMA50.

Régimen esperado: uptrends persistentes con pullbacks someros. Lookbacks 20/50/10
para que todo gate sea válido dentro de la ventana OOS de ~6 meses.

## Números OOS reales (per-símbolo)

| Símbolo | Return% | Sharpe | N | vs SPY% |
|---|---|---|---|---|
| GOOGL | 9.93 | 0.90 | 15 | −47.43 |
| CAT | 7.64 | 0.54 | 12 | −49.72 |
| JNJ | 3.70 | 0.58 | 11 | −53.66 |
| JPM | 3.39 | 0.38 | 20 | −53.96 |
| NVDA | 3.02 | 0.22 | 12 | −54.34 |
| WMT | 1.96 | 0.26 | 13 | −55.40 |
| AMZN | 1.69 | 0.20 | 16 | −55.67 |
| MSFT | −0.19 | −0.02 | 9 | −57.54 |
| AAPL | −0.57 | −0.06 | 12 | −57.92 |
| HD | −0.85 | −0.13 | 13 | −58.21 |
| UNH | −1.30 | −0.20 | 7 | −58.65 |
| XOM | −3.36 | −0.45 | 18 | −60.72 |
| PG | −4.26 | −0.86 | 15 | −61.62 |
| META | −6.11 | −0.73 | 17 | −63.47 |

**Mediana Sharpe 0.09, breadth de returns positivos 7/14 (50%).**

## Por qué murió

1. **Entrar antes + aguantar más NO crea edge donde no lo hay.** Este era el test
   directo de las dos hipótesis del postmortem anterior, ejecutadas juntas. Resultado:
   mediana Sharpe subió de −0.02 → **0.09** y min_trades pasó (7 vs 4: el dip da más
   señal que la reconquista) — pero 0.09 sigue a un orden de magnitud del piso 0.80,
   y sólo la mitad de los nombres da return positivo. La mejora es de ruido, no de señal.
2. **Los mismos dos nombres cargan la cara** (GOOGL, CAT) que en `trend_pullback`.
   El resto es flat-a-negativo. Es un edge concentrado en 2/14, no amplio — exactamente
   lo que el gate por breadth existe para vetar.
3. **excess −56.61 es el defecto de arnés conocido** (ventana/exposición desalineada,
   ver postmortem trend_pullback); el gate discriminó bien por Sharpe+breadth.

## Hallazgo de método (transferible)

**La familia pullback está agotada.** Tres ideas muertas cubren sus tres celdas:
- `rsi_mr`/`confirmed_mr`: comprar sobreventa extrema contra-tendencia → cuchillos.
- `trend_pullback`: comprar la reconquista (tarde) + salir en SMA50 (temprano).
- `pullback_ride`: comprar el dip (temprano) + aguantar con trailing-low (tarde salir).

Las dos hipótesis del postmortem de `trend_pullback` ("el edge está antes, en el dip"
y "el edge está en aguantar más") quedan **ambas falsadas**: probadas juntas, mueven
la aguja de ruido pero no producen edge amplio. Comprar pullbacks a la media móvil
sobre esta canasta de 14 large-caps, en 2022–2026, no tiene alpha replicable — el poco
retorno positivo se concentra en GOOGL/CAT, no en la señal.

→ **Recomendación para revisión semanal (decisión de Luis, no la tomo solo):** dejar de
iterar sobre pullback-a-la-media. La siguiente idea debería salir de una familia
distinta (volatilidad/breakout de compresión, cross-sectional relativo, o estacional),
no otra variación de "comprar el dip". Y sigue en pie el fix de método pendiente:
benchmarkear SPY sobre las mismas ventanas OOS antes de que `median_excess_pct`
signifique algo.

## Learnings

1. **Barrer las dos hipótesis de un postmortem en una sola idea ahorra un ciclo.**
   En vez de dos corridas (una por entrada, otra por salida), este test combinó ambas
   y cerró la familia entera con evidencia. Cuando un postmortem deja 2 hipótesis
   baratas y ortogonales, probarlas juntas primero es más eficiente: si el combo no
   da edge, ninguna sola lo dará.
2. **Un edge concentrado en 2/14 nombres reaparece idéntico entre variaciones** — es
   señal de que el alpha vive en esos activos (GOOGL/CAT trending), no en la estrategia.
   El gate por breadth es el que lo caza; sin él, el mean lo disfrazaría de éxito.
</content>
</invoke>
