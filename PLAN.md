# PLAN — Paper horse-race → edge probado

> Norte: encontrar un **edge real en paper** (estrategia que le gane al mercado, no solo a las otras), con riesgo y disciplina por encima de todo. NUNCA dinero real sin edge probado. Rama: `feature/paper-horse-race` (NO mergeada).
> Estado vivo: 3 caballos × $25k sobre 1 cuenta Alpaca paper, corriendo L-V 13:00 CST vía LaunchAgent `com.luisfer.horserace`. Epoch limpio desde **2026-06-05**.

## Prioridad

1. **[EN CURSO] Medir edge contra benchmark, no solo entre caballos.**
   - [x] Benchmark SPY buy&hold + columna `alpha%` en el reporte (`live/benchmark.py`, anclado al 2026-06-05). Hecho 2026-06-14, 63/63 tests verde.
   - [ ] Dejar acumular alpha hasta cerrar la ventana de 2 semanas del epoch limpio (~2026-06-19). Leer `alpha%`: si ningún caballo es alpha-positivo y estable, NO hay edge.
   - [ ] (Mejora) Persistir snapshot diario de equity/alpha por caballo para una curva, no solo el último corte.

2. **[PENDIENTE] Métrica de riesgo, no solo retorno.** Un caballo que gana con drawdown brutal no es edge. Añadir max-drawdown y/o vol al reporte una vez haya serie temporal de equity.

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
