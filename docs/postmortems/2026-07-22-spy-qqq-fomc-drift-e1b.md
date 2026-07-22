# Postmortem — E1b SPY/QQQ post-FOMC drift · MUERTA (sign-flip por régimen) (2026-07-22)

**Veredicto: NO procede a paper.** No hay regla direccional estable en la ventana FOMC
para índices de equity: el signo de la reacción se **invierte** entre régimen ZIRP y
régimen de hikes. Ninguna celda pasa el gate en ambos regímenes con la misma regla.

## Tesis original

E1 mató GLD *fade* post-FOMC, pero nadie había gateado el otro lado obvio del mismo
evento verificado: **índices de equity que CONTINÚAN (drift) la dirección del día del
anuncio** los 1-5 días siguientes, bajo la hipótesis de *underreaction* a la guía de
política (el mercado digiere el dot-plot/press-conf en sesiones, no en un tick).
Régimen esperado: mejor cuando la Fed sorprende y hay repricing sostenido.

## Qué se hizo

Mismo harness event-study, calendario FOMC ya verificado (55 fechas 2015-21 + 24 de
2022-24, contrastadas contra federalreserve.gov en E1). Se corrió drift **y** fade,
ventanas 1/3/5d, SPY y QQQ, partiendo **IS 2022-24 vs OOS 2015-21** — el split que
cruza régimen macro (la lección de método de E1).

## Números reales

`exp` = expectativa por evento; `tail` = ganancia media / |pérdida media|. Gate-event:
n≥15, exp>0, tail≥1.20. **PASS en negrita.**

### Modo DRIFT (continuación)
| | SPY 1d | SPY 3d | SPY 5d | QQQ 1d | QQQ 3d | QQQ 5d |
|---|---|---|---|---|---|---|
| **OOS 2015-21** (n=55) | **+0.18% t1.88** | **+0.13% t1.46** | −0.04% t1.05 | +0.10% t0.91 | +0.10% t0.94 | **+0.17% t1.22** |
| IS 2022-24 (n=24) | +0.10% t0.59 | −0.44% t0.78 | −0.74% t0.61 | +0.11% t0.57 | −0.65% t0.72 | −0.76% t0.43 |

### Modo FADE (reversión)
| | SPY 1d | SPY 3d | SPY 5d | QQQ 1d | QQQ 3d | QQQ 5d |
|---|---|---|---|---|---|---|
| OOS 2015-21 (n=55) | −0.18% t0.53 | −0.13% t0.69 | +0.04% t0.95 | −0.10% t1.1 | −0.10% t1.06 | −0.17% t0.82 |
| **IS 2022-24** (n=24) | −0.10% t1.69 | **+0.44% t1.29** | **+0.74% t1.65** | −0.11% t1.75 | **+0.65% t1.39** | **+0.76% t2.34** |

## Por qué murió

**El signo se invierte con el régimen.** En 2015-21 (ZIRP, baja vol, QE) los índices
**driftean** post-FOMC (SPY 1d/3d, QQQ 5d pasan). En 2022-24 (hikes, alta vol) los
índices **fadean** (SPY/QQQ 3d/5d pasan con exp +0.44…+0.76%). Y el drift 2022-24 es
netamente negativo a 3-5d. **Cero celdas (símbolo × modo × ventana) pasan en ambos
regímenes**; cada PASS en un régimen es FAIL en el otro, y el MODO ganador se voltea.

La única celda "fuerte" en un régimen — SPY drift 1d OOS, tail 1.88 — tiene hit 47%
(por debajo de moneda al aire): la carga un tail derecho gordo específico del régimen,
y +0.18% bruto/1d se lo come el costo. No es edge, es la reacción condicional del
régimen a la política, igual que E1 con GLD.

## Lección

Refuerza y generaliza la lección de E1: **la ventana FOMC no deja edge direccional
estable en daily-drift** — ni fade (GLD, equity) ni drift (equity) sobreviven el cruce
de régimen. El evento sí mueve precio, pero el signo del movimiento post-evento está
condicionado por el régimen macro (ZIRP vs hikes), así que cualquier regla direccional
fija es overfit de régimen. **La familia "regla direccional fija en ventana FOMC" queda
agotada** (GLD fade E1, equity drift+fade E1b). Si algo vive aquí, tendría que ser
condicional al régimen (state-aware), no una regla fija — y eso sube mucho el riesgo de
overfit con n≈55/régimen. Prioridad baja; mejor grindear E2 (CPI) / E3 (earnings) cuando
haya fechas verificadas.
