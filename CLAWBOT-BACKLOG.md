# Backlog — bot trading

> Pendientes operativos de este workspace (paper horse-race en prod). Editable por operador y bot. Sacado de memoria 2026-06-02. **NO dinero real sin edge probado.**

## Vigilancia activa (paper)

- [ ] **Verificar `data/cron.log`** tras las 13:00 CST (LaunchAgent `com.luisfer.horserace`). Confirmar que las corridas L-V se ejecutan.
- [ ] **Dejar correr 2 semanas** (decisión 2026-05-31) y leer `strategy_attribution()` / `data/ledgers/state.json` para ver qué estrategia gana.
- [ ] **SOLO si hay edge real** → recién entonces evaluar $250 vivo con tope + kill-switch. NO antes.

## HECHO 2026-06-04 — universo amplio + ranking real (rama feature/paper-horse-race)

Plan confirmado el 3-jun ("dale con esos números"), construido y testeado (40/40 verde):
- [x] **Universo S&P 500 estático** (503 nombres) commiteado en `sp500_constituents.py` (build-time, sin red runtime). Helper `stock_universe.sp500_symbols()`. Veto Wikipedia respetado.
- [x] **Descarga en lote** `YFinanceBars.get_bars_batch()` (chunks 100). 503/503 en ~14s, sin rate-limit.
- [x] **Momentum cross-sectional real**: `live/portfolio_targets.py` rankea por momentum 6m (skip 1m), top-15 equal-weight, rebalanceo solo 1er día hábil del mes. Adiós compra alfabética.
- [x] **MR diversificada**: escanea 503, rankea por RSI(2) asc, hasta 10 concurrentes a ~10% c/u. Hoy confirmed_mr=2 / rsi_mr=11 candidatos (antes: 0).
- [x] Orchestrator reescrito a modelo portafolio-objetivo. Invariantes intactas (sizing vs cash, venta por qty, guard paper).
- [ ] **DECISIÓN PENDIENTE operador**: resetear ledgers a $33k limpio + flatten de las 3 posiciones huérfanas (SPY/AAPL/QQQ del código viejo) para arrancar el horse-race corregido desde cero. Si NO se resetea, momentum sostiene las 3 viejas hasta el 1er día hábil del próximo mes.
- [ ] Commit/push de la rama (sigue NO mergeada).

## PENDIENTE del operador — rotar secretos (cuarentena)

Llaves reales viejas en `~/.trading-secrets-quarantine-20260531/` (chmod 700). Rotar y luego `shred`:
- [ ] **Binance API key** (`cJh2…ZoPj`, reutilizada en 4 carpetas — #1, puede mover fondos)
- [ ] **KuCoin**
- [ ] **Telegram bot token** (@BotFather `/revoke`)
- [ ] **Revisar/terminar instancias EC2** (mx-central-1, 6 `.pem` SSH)

## Diferido (post-edge)

`check_time_exits` no cableado (necesita reconciliación de ledger) · push/PR de la rama `feature/paper-horse-race` (NO mergeada) · refresco manual del snapshot S&P 500 (script build-time) cuando cambie la membresía.

### Nuevos mercados — crypto / forex (idea Luis 2026-06-15, PARKED hasta checkpoint de edge)

Regla: **no se abre mercado nuevo hasta que las acciones muestren aunque sea un asomo de edge** (alpha vs SPY positivo y no concentrado). Multiplicar mercados antes dispersa el foco sin responder la única pregunta (¿hay edge?).
- **Crypto = candidato razonable.** Plumbing a medias: `crypto_client.py` (Binance, soporta testnet/paper, misma interfaz que Alpaca) PERO no cableado al orquestador vivo. Bloqueos: (1) llaves Binance en cuarentena → necesita creds **testnet** nuevas, NUNCA las de mainnet; (2) 24/7 rompe el modelo de cron diario L-V; (3) extender el framework horse-race+alpha a crypto. Encaja en el mismo rig si se hace bien.
- **Forex = NO.** Bróker nuevo desde cero, apalancamiento = riesgo de reventar cuenta, stats de retail las peores. Fuera de alcance.

## NO romper

Guard: solo corre si `ALPACA_BASE_URL=paper-api.alpaca.markets`. Sizing contra cash (no equity). Vender por qty específica, NUNCA `close_position`.
