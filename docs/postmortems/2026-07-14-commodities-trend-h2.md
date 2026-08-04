# Postmortem — donchian_breakout en commodities (Backlog H2)

**Fecha:** 2026-07-14 · **Veredicto:** muerta en backtest, sin edge OOS. No desplegada.
**Gate:** FAIL (median_excess −64.56 / breadth 0.0 / median_sharpe 0.21 / min_trades 9).
**Muestra:** walk-forward 2022-01-01→2026-07-14, 6 ETFs de commodities (GLD/SLV/USO/UNG/DBA/DBC), yfinance, OOS. Slippage 0.05%/lado.

## Tesis

El trend-following captura las tendencias largas de materias primas mejor que en
equities (donde el mean-reversion domina). Canal Donchian 20/10 sobre oro, plata,
petróleo, gas natural, agri y broad-basket. Origen: carrusel "trend on gold & oil".

## Números OOS reales (per-símbolo)

| Símbolo | Return% | Sharpe | MaxDD% | WinRt% | PF | N | vs SPY% |
|---|---|---|---|---|---|---|---|
| USO (petróleo) | 4.91 | 0.30 | −9.08 | 55.6 | 1.48 | 9 | −61.76 |
| SLV (plata) | 3.84 | 0.28 | −9.91 | 36.4 | 1.30 | 11 | −62.83 |
| DBA (agri) | 3.18 | 0.51 | −3.66 | 50.0 | 1.86 | 10 | −63.50 |
| GLD (oro) | 1.05 | 0.14 | −5.78 | 46.2 | 1.14 | 13 | −65.62 |
| DBC (broad) | 0.32 | 0.06 | −5.08 | 28.6 | 1.06 | 14 | −66.36 |
| UNG (gas nat) | −9.92 | −0.52 | −16.11 | 11.1 | 0.17 | 9 | −76.59 |

Median Sharpe 0.21 (floor 0.80). Breadth 0/6: ningún símbolo clarea el gate.

## Por qué murió

- **No hubo tendencia limpia que capturar.** 2022-2026: el oro/broad se movieron
  de lado con whipsaws que el canal Donchian cobró como falsos breakouts (GLD
  Sharpe 0.14 con 13 trades, DBC 0.06 con 14 — pura fricción). El régimen no fue
  trending sostenido sino choppy.
- **UNG (gas natural) fue un desastre** (−9.92%, PF 0.17, 11% winrate): el gas
  spikea y colapsa; el breakout entra tarde arriba y sale tarde abajo.
- **Los "ganadores" (USO/SLV/DBA) son marginales y positivos-pero-flojos:**
  Sharpe 0.28-0.51, ninguno cerca del 0.80. No es edge, es ruido con signo.
- El `median_excess −64.56` está inflado por el defecto de benchmark conocido
  (ventana OOS ~25% vs SPY full-range 100%), pero **Sharpe 0.21 y breadth 0/6
  matan la hipótesis sin depender de ese número**.

## Conclusión

La tesis "commodities trendean mejor que equities" no se sostiene con Donchian
20/10 en este período. El trend-following clásico necesita tendencias seculares
sostenidas (2000s super-ciclo, 2021 energía) que no estuvieron presentes 2022-26.
Familia trend-following en commodities agotada con este parámetro; una variante
con canal más largo (55/20 estilo Turtle) podría reducir whipsaws pero la muestra
OOS ya quedaría muy corta (N<6). No vale reabrir sin más data.

## Impacto en H3 (cartera multi-activo)

H3 dependía de que H1 (cripto momentum) **o** H2 (commodities trend) pasaran el
gate por separado. **Ambas fallaron.** La composición multi-activo no las salva
(diversificar sobre dos fuentes sin edge no produce edge). H3 se cancela.
