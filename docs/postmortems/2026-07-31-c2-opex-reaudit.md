# Re-auditoría de C2 (OpEx 1d drift long-only) — 2026-07-31

**Veredicto: C2 NO muere, pero se parte en dos. QQQ sobrevive todo. IVV (la pata
S&P que corre en paper) NO pasa la barra nueva en el régimen OOS.**
**Acción recomendada a Luis (NO ejecutada: es cambio en vivo, decisión suya):
sacar IVV del universo de `opex_drift`, dejar QQQ solo.**

## Por qué re-auditar algo que ya pasó el gate

C2 pasó el gate el 2026-07-24 y se desplegó a paper (commit `4b67a82`). Desde
entonces el método subió dos veces, y C2 nunca enfrentó ninguno de los dos tests
nuevos porque no existían cuando pasó:

| Lección | De dónde | Qué le faltaba a C2 |
|---|---|---|
| **Jackknife** — quitar los top contribuyentes | F3 (2026-07-29) | Nunca se le corrió. Mató un edge que ya había pasado placebo y split de régimen. |
| **El placebo lleva el MISMO condicionamiento que la regla** | F4 (2026-07-30) | El placebo de C2 (2026-07-26) muestreó días random **sin condicionar a verde**; la regla real solo opera días verdes. |

Y un tercer hueco que apareció al leer el código en vez del doc: **C2 se validó
en su versión de dos patas, pero lo que corre en paper es solo la pata larga**
(`strategy_opex_drift` entra solo si el 3er viernes cerró verde). La regla
desplegada nunca fue medida contra el gate; se gateó a su hermana.

Harness: `events/opex_audit.py` + `jackknife_by_event` / `leave_one_year_out` en
`events/event_study.py`.

## La muestra desplegada es la MITAD de la que se gateó

Solo los días verdes califican, y son ~44% de los OpEx:

| Celda | Validado (2 patas) | Desplegado (long-only) |
|---|---|---|
| SPY OOS 2015-21 | n=84 · +0.153% | n=37 · +0.144% |
| SPY IS 2022-24 | n=35 · +0.343% | n=16 · +0.653% |
| QQQ OOS 2015-21 | n=83 · +0.263% | n=35 · +0.426% |
| QQQ IS 2022-24 | n=36 · +0.243% | **n=14** · +0.727% |

La expectativa **por evento sube** (confirma lo medido el 2026-07-26: la pata
larga es la buena) pero el N se parte a la mitad. **QQQ en el régimen de hikes
corre con n=14, por debajo del piso `MIN_EVENTS=15` del propio proyecto** — algo
invisible mientras el gate leía el N=36 de la versión de dos patas.

## Resultado sobre los tickers que REALMENTE corren (IVV, QQQ)

`run_trading_system.py:87` → `StrategyConfig("opex_drift", ["IVV", "QQQ"], ...)`.
Barra: n≥15 · exp>0 · jackknife−top3 >0 · leave-one-year-out >0 · placebo
verde-condicionado ≥p90.

| Celda | n | exp | hit | jk −top3 | peor año fuera | años+ | placebo pct | |
|---|---|---|---|---|---|---|---|---|
| **IVV OOS 2015-21** | 36 | +0.124% | 64% | **−0.026%** | +0.049% | 6/7 | **68** | ❌ |
| IVV IS 2022-24 | 16 | +0.667% | 75% | +0.358% | +0.515% | 3/3 | 100 | ✅ |
| **QQQ OOS 2015-21** | 35 | +0.426% | 69% | +0.260% | +0.360% | **7/7** | 100 | ✅ |
| QQQ IS 2022-24 | **14** | +0.727% | 71% | +0.291% | +0.632% | 3/3 | 98 | ⚠️ n<15 |

Los dos hallazgos:

1. **IVV no sobrevive el jackknife en OOS.** Quitando los 3 mejores eventos de 36,
   la expectativa se vuelve **negativa** (−0.026%). Siete años de S&P, y el edge
   son 3 días. Además su placebo verde-condicionado cae en el percentil **68** —
   contra un control que también solo opera días verdes, IVV-OOS **no se
   distingue de comprar cualquier día verde y salir mañana**. El placebo original
   daba ~90 porque comparaba una regla condicionada contra un control sin
   condicionar: exactamente el sesgo que F4 nos enseñó a cerrar.
2. **QQQ es la celda más sólida que ha producido este proyecto.** Sobrevive el
   jackknife (+0.260% sin los top-3), es positiva en **7 de 7 años** del OOS,
   ninguna caída de año la voltea, y bate el placebo condicionado en el percentil
   100. Es lo contrario de F3: ahí el edge era 3 nombres, aquí está repartido.
   Su única mancha es el n=14 del régimen de hikes.

## Qué NO cambió (y por qué esto no es un F5)

C2 sigue siendo el único hallazgo del repo que sobrevive un jackknife. No aplica
el patrón de F3/F4: aquí no hay 3 símbolos ni 3 días cargando la distribución en
el lado QQQ. La tesis de microestructura (gamma de dealers pinnea, el flujo
residual continúa ~1 sesión) sigue en pie y sigue explicando por qué solo el w=1d
sobrevive.

Lo que cambió es la **confianza por pata**: la mitad IVV del despliegue está
apoyada en 3 días, y la mitad QQQ está apoyada en algo real pero delgado
(~+0.43%/evento × ~5 eventos verdes/año ≈ 2%/año bruto en el OOS).

## Decisión que le toca a Luis (no la tomo yo)

**Sacar IVV de `opex_drift` y dejar QQQ solo.** Es un cambio en una estrategia
que ya corre en paper → no lo ejecuto en el loop. Argumentos:

- A favor: IVV falla 2 de 4 checks en el régimen largo; su aporte esperado es
  ruido con costo de ejecución.
- En contra: quedarse con 1 símbolo × ~5 eventos/año es una muestra viva
  minúscula; el paper tardaría años en decir algo. Alternativa: dejar las dos y
  tratar IVV como control interno (si IVV y QQQ se comportan igual en vivo, la
  tesis de microestructura específica de QQQ era mentira).

Mi lectura: la segunda. IVV como **control** vale más que IVV como caballo, y no
cuesta nada porque ya está cableado. Pero eso es preferencia, no técnica.

## Deuda de método saldada (la que dejó F4)

Cableados y con tests (`tests/test_event_study.py`):

- `jackknife_by_event(returns, k)` — hermano intra-símbolo de
  `jackknife_by_group`. F3 cubrió el fraude entre símbolos; este cubre el fraude
  entre eventos de una misma serie, que es el que casi mata a C2.
- `leave_one_year_out(by_year)` — ¿el edge es un año? Breadth temporal.
- Placebo condicionado (`opex_audit.conditioned_placebo`) — el control replica el
  filtro de entrada de la regla. **Regla general nueva: si la regla condiciona
  (día verde, stop, umbral de gap), el placebo condiciona igual.** Sin eso el
  percentil mide el condicionamiento, no el evento.

Sigue pendiente lo otro que dejó F4: el `tail_ratio` como gate. Aquí se usó como
descriptor, no como gate — correcto según esa lección. Nótese que IVV-OOS tiene
tail 0.88 (bajo 1.2) y aun así hubiera pasado el gate viejo por otras vías; una
razón más para no gatear por forma.

## Estado

- C2 **sigue en paper sin cambios** (no toqué el despliegue).
- Ficha C2 actualizada en `docs/CANDIDATES.md` con la barra nueva.
- Auditoría reproducible: `.venv-bt/bin/python -m events.opex_audit`.
