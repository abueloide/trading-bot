# E1c — FOMC drift condicionado a magnitud de sorpresa · FAIL ❌ (2026-07-23)

## Tesis
E1b mató la **regla direccional fija** en ventana FOMC (equities driftean en ZIRP,
fadean en hikes; cero celdas pasan en ambos regímenes). Hipótesis de rescate: el null
de E1b venía de operar **días chicos** (movimiento del día del evento = ruido, no
sorpresa). Si se filtra a eventos donde `|mov_día| ≥ k × vol_diaria_20d_previa` — o sea
solo repricings GRANDES y genuinos — el drift/continuación debería limpiarse y, al ser
un filtro auto-calibrado por vol (regime-neutral), sobrevivir en ambos regímenes.

Régimen esperado: cualquiera con repricing real; la señal la lleva la **magnitud**, no
el signo ni el régimen macro. Datos: calendario FOMC ya verificado (79 fechas 2015-24) +
Yahoo. Sin dato firewalled. Filtro implementado en `events/event_study.py` (`min_vol_mult`,
vol de los 20d ANTERIORES al evento, sin lookahead).

## Números OOS reales (SPY/QQQ, drift, filtro k=1.0×vol20d)
| Régimen | Símbolo·ventana | exp | hit | tail | n |
|---|---|---|---|---|---|
| OOS ZIRP 2015-21 | SPY 3d | **+0.19%** | 45% | 1.56 | 20 | ← única que pasa |
| OOS ZIRP 2015-21 | SPY 1d | −0.01% | 45% | 1.18 | 20 |
| OOS ZIRP 2015-21 | QQQ 1d/3d/5d | −0.21/−0.25/−0.48% | — | <1.2 | 17 |
| IS hikes 2022-24 | SPY 3d/5d | −0.36 / −0.49% | 47/53% | 0.84/0.59 | 15 |
| IS hikes 2022-24 | QQQ 3d/5d | −0.55 / −0.50% | 50/62% | 0.63/0.43 | 16 |

Baseline SIN filtro (E1b): en ZIRP pasaban SPY 1d+3d y QQQ 5d; en hikes 0 celdas.

## Por qué murió
1. **El filtro no limpió nada — thinneó la muestra.** N cae de 55→~20 (ZIRP) y 24→~15
   (hikes), rozando el piso `MIN_EVENTS=15`. A k=1.5× caería por debajo → auto-fail por
   muestra. El presupuesto estadístico se agota antes de encontrar señal.
2. **El flip de régimen persiste.** Hikes sigue fadeando (3d/5d negativos) incluso
   restringido a sorpresas grandes. La dependencia de régimen NO era ruido de días
   chicos contaminando una señal fija; es **estructural**.
3. **Una sola celda pasa, y se mueve.** El único PASS (SPY 3d ZIRP) ni siquiera coincide
   con los ganadores del baseline (SPY 1d, QQQ 5d) → es azar de subconjunto, no un edge
   que el filtro haya revelado.

## Veredicto
FAIL. La familia **"regla direccional en ventana FOMC para equities"** queda agotada en
las tres variantes probadas: fija (E1b), y ahora condicionada a magnitud (E1c). No
desplegada. La reacción post-FOMC en índices es condicional al régimen macro y no se
rescata con un selector de eventos por tamaño de sorpresa.

## Lección de método
Filtrar eventos por magnitud en un calendario ya chico (≤79 fechas/9 años) choca de
frente con el piso de muestra: cada filtro que sube la "calidad" del evento baja N hacia
el ruido. Para hipótesis de selección-de-evento hace falta un catalizador de **mayor
frecuencia** (CPI mensual, earnings por-símbolo, OpEx) — que hoy están BLOCKED-DATA en el
sandbox. El pozo FOMC-equities está seco; el loop no debe generar más variantes sobre él.
