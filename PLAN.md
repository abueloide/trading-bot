# PLAN — Paper horse-race → edge probado

> Norte: encontrar un **edge real en paper** (estrategia que le gane al mercado, no solo a las otras), con riesgo y disciplina por encima de todo. NUNCA dinero real sin edge probado. Rama: `feature/paper-horse-race` (NO mergeada).
> Estado vivo: 3 caballos × $25k sobre 1 cuenta Alpaca paper, corriendo L-V 13:00 CST vía LaunchAgent `com.luisfer.horserace`. Epoch limpio desde **2026-06-05**.

## Prioridad

1. **[EN CURSO] Medir edge contra benchmark, no solo entre caballos.**
   - [x] Benchmark SPY buy&hold + columna `alpha%` en el reporte (`live/benchmark.py`, anclado al 2026-06-05). Hecho 2026-06-14, 63/63 tests verde.
   - [x] **BUG CRÍTICO arreglado 2026-06-15:** el alpha NUNCA se calculaba en prod. SPY no es constituyente del S&P 500 → no estaba en el universo descargado → `compute_benchmark` devolvía `None` en cada corrida → `alpha_pct: null` en toda la curva y la columna `alpha%` se caía del reporte. Los tests pasaban porque inyectaban SPY en el fixture (test-verde/prod-roto). Fix: `FETCH_SYMBOLS = UNIVERSE + [SPY]` en `run_trading_system.py` (SPY se descarga pero el orchestrator solo pide símbolos de estrategia, así que NUNCA se opera). Regresión cerrada con `tests/test_benchmark_wiring.py` (5 tests). Verificado end-to-end: SPY +2.35% desde 2026-06-05. **El alpha real empieza a registrarse en la próxima corrida (mar 2026-06-16).**
   - [x] Lectura de checkpoint on-demand: `live/checkpoint_report.py` (+ `tests/test_checkpoint_report.py`, 13 tests) consolida por caballo equity/return/alpha + max-DD/vol + un veredicto descriptivo en UNA tabla, leyendo solo la curva persistida (sin red, sin Alpaca). Correr: `.venv-bt/bin/python -m live.checkpoint_report`. NO automatiza ninguna decisión.
   - [ ] Dejar acumular alpha hasta cerrar la ventana de 2 semanas. OJO: la curva de alpha real arranca el 2026-06-16 (el bug la tuvo en null hasta hoy), así que la ventana útil de alpha corre ~2026-06-16 → 2026-06-30. Leer con `checkpoint_report`: si ningún caballo es alpha-positivo y estable, NO hay edge.
   - [x] (Mejora) Persistir snapshot diario de equity/alpha por caballo para una curva, no solo el último corte. Hecho 2026-06-14: `live/equity_snapshot.py` (JSONL append-only, idempotente por día) cableado en `run_trading_system.py` → `data/ledgers/equity_curve.jsonl`. 68/68 tests verde. Empieza a acumular en la próxima corrida L-V.

2. **[HECHO — esperando data] Métrica de riesgo, no solo retorno.** Un caballo que gana con drawdown brutal no es edge.
   - [x] `live/risk_metrics.py`: max-drawdown (peak-to-trough) + volatilidad (stdev de retornos diarios) por caballo, leyendo `equity_curve.jsonl`. Cableado en `run_trading_system.py` → imprime tabla de riesgo tras el snapshot. Hecho 2026-06-15, 77/77 tests verde (9 nuevos en `test_risk_metrics.py`).
   - [ ] Los números solo son útiles con ~varios días de curva acumulada (empieza a llenarse en la próxima corrida L-V). Leer en el checkpoint junto a alpha.

3. **[PENDIENTE — decisión operador] Checkpoint de 2 semanas.** Al cierre de la ventana: leer attribution + alpha y decidir con Luis. Solo si hay alpha real y consistente → recién evaluar $250 vivo con tope + kill-switch (ver `docs/ROADMAP-real-money.md`). NO antes. Esta decisión es de Luis (dinero/irreversible).

4. **[PENDIENTE] Vigilancia operativa.** Confirmar en `data/cron.log` que las corridas L-V se ejecutan sin error tras 13:00 CST.

5. **[DIFERIDO post-edge] Deuda técnica conocida:**
   - `check_time_exits` no cableado (necesita reconciliación de ledger por estrategia — mismo invariante CORE que SELL).
   - Revival de `momentum_news` (overlay en `live/news_overlay.py`) requiere fuente de noticias per-ticker o pagada; free-tier AlphaVantage no sirve.
   - Cripto, refresco manual del snapshot S&P 500 cuando cambie membresía.

## NO romper (guardrails)
- Solo corre si `ALPACA_BASE_URL=paper-api.alpaca.markets`. Sizing contra cash (no equity). Vender por qty específica, NUNCA `close_position`. NUNCA tocar secretos en cuarentena.

## Pendiente del operador (fuera del loop del bot)
- Rotar secretos en cuarentena (`~/.trading-secrets-quarantine-20260531/`): Binance, KuCoin, Telegram token, instancias EC2. Ver `CLAWBOT-BACKLOG.md`.
