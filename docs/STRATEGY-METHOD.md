# Método de research de estrategias — backtest-first

> Vigente desde 2026-07-08. Reemplaza el ciclo viejo "deploy y mira un mes".
> Norte: encontrar edge real en paper, con disciplina, sin quemar foco de Luis.

## Por qué cambiamos

+1 mes iterando estrategias a mano, 0 edge. El fallo **no fue mala suerte, fue
método**: metíamos una estrategia a paper y esperábamos semanas a ver qué pasaba.
Lento, ruidoso, y —peor— desplegábamos estrategias que el backtest **ya sabía
perdedoras**. Aplicado retroactivamente, el gate de abajo veta las 5 estrategias
que existieron. Ninguna debió llegar a paper. (Ver `postmortems/`.)

## El gate (obligatorio antes de paper)

Una idea nueva NO toca la carrera de paper hasta pasar el walk-forward:

```
python backtesting/run_backtest.py --strategy <name>   # produce results/<name>_summary_*.json
python backtesting/gate.py results/<name>_summary_*.json  # PASS/FAIL
```

Umbrales (`backtesting/gate.py`):
| Check | Regla | Por qué |
|---|---|---|
| `median_excess_pct` | mediana OOS > 0 | mediana, no promedio: resiste outliers |
| `breadth_frac` | ≥60% de símbolos con excess+ | edge amplio, no un símbolo con suerte |
| `median_sharpe` | mediana ≥ 0.80 | retorno ajustado a riesgo |
| `min_trades` | ≥5 trades/símbolo | pocos trades = azar, no señal |

**Por qué mediana + amplitud y no promedio:** momentum_rotation dio excess
promedio **+9.1%** — pero era **un solo outlier (NVDA +118.7%)**; los otros 4
símbolos negativos. Un gate por promedio aprueba ese espejismo. El gate por
mediana/amplitud lo veta — y en vivo se confirmó (momentum fue el peor caballo).

## Ciclo (carril verde — lo corro yo, Luis no lo babysitea)

1. **Genero** ideas nuevas (registro en `backtesting/strategies.py`).
2. **Backtest + gate.** FAIL → postmortem directo, no se despliega.
3. **PASS → paper.** Solo las que pasan entran a la horse-race.
4. **Paper es la confirmación final:** backtest edge que no sobrevive en vivo =
   overfit/régimen. Muere → postmortem.
5. **Commit + tests verdes** cada ciclo.

## Cadencia y stop

- **Evaluación semanal** con Luis (digest, no decisión diaria). Standups/cierres
  diarios: apagados.
- **Stop condition:** si tras el timebox nada pasa el gate + sobrevive paper,
  veredicto duro. (Timebox a fijar; default sugerido: revisión semanal, corte a
  reevaluar el rumbo global si 8 semanas sin candidato PASS-y-vivo.)
- **Invariante:** siempre paper. Nada de dinero real sin GATE #1
  (`docs/ROADMAP-real-money.md`).

## Postmortems

Cada estrategia muerta deja entrada en `postmortems/` con: tesis, params, números
OOS reales, y **por qué murió**. Es el activo transferible aunque nunca haya edge.
