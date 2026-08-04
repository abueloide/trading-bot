# Roadmap — del paper al dinero real (plan oficial)

> Acordado 2026-06-13 entre Luis y el asistente PO. Este doc es el contrato.
> Si en 3 semanas se nos olvida el porqué de un candado, se relee esto — no se
> improvisa. Reglas de dinero por encima de ganas.

## Premisa honesta

- El bot **no es la palanca** de los $200k/mes (esa es SaaS + seguros). Es un
  laboratorio + un satélite chico de capital de riesgo. Luis lo sabe y aun así
  quiere explorarlo — válido, mientras siga gated.
- A hoy **no hay edge probado**. El retail no gana en corto plazo con señales
  accesibles (ver auditoría 2026-05-31 y la conversación del 13-jun).
- Tesis correcta de Luis: **la constancia compone más que los trades de suerte.**
  El plan se construye sobre eso.

## Modelo: core + satélite

| Motor | Rol | Tamaño | Riesgo | Estado |
|-------|-----|--------|--------|--------|
| 🛞 **Core** | Crecimiento real vía aportaciones + interés compuesto. Índice amplio aburrido o CETES/SOFIPO. | La mayor parte del capital | Bajo | Decisión separada de Luis; el bot NO lo necesita para arrancar |
| 🎲 **Satélite** | Estrategias activas del bot. Aprendizaje + intento de alfa. | Solo los **$10k MXN** (~$550 USD) que Luis puede perder | Alto, acotado | **Paper hoy. NO arranca real hasta pasar el gate de abajo.** |

El "combinar no cae mal" = core grande y constante + satélite chico y activo.
El satélite NUNCA crece con dinero nuevo hasta probarse; si pierde, se perdió
solo capital de apuesta, no ahorro.

## EL GATE — qué tiene que pasar para encender dinero real en el satélite

Ninguno es negociable. Se encienden TODOS o no hay dinero real.

### 1. Evidencia de edge (estadística, no corazonada)
- **Muestra mínima: ~3–6 meses de paper en vivo.** Las "2 semanas" originales eran
  para validar la plomería, NO para probar edge. 2 semanas ganando = suerte.
- Idealmente la ventana cruza un tramo que sube y uno que baja (régimen mixto).
- Al menos un caballo debe **ganarle al S&P (SPY) neto de costos realistas**
  (comisión + slippage ~0.1%/trade) en:
  - retorno total de la ventana, **y**
  - ajustado por riesgo (Sharpe ≥ SPY, o al menos max drawdown no peor que SPY).
- **Atribución:** la ventaja no puede venir de 1–2 nombres con suerte. Se revisa
  `strategy_attribution()`. Si es concentrado, no cuenta.

### 2. Código de dinero real probado (hoy NO existe)
- [ ] Arreglar bug de `close_position` / time-exits (cierra posición completa, no
      la porción de la estrategia — invariante de dinero rota).
- [ ] Reemplazar el guard duro "solo paper" por un **flag explícito y deliberado**
      de dinero real, con enforcement de tope + kill-switch, todo testeado.
- [ ] Reconciliación ledger virtual ↔ cuenta real (que los números cuadren).
- [ ] Bróker real fondeado + KYC (Alpaca real, o GBM+/Kuspit en MX si son pesos).

### 3. Candados de riesgo (cuando ya corra real)
- **Tope de capital:** SOLO los $10k MXN. Cero dinero nuevo hasta que el satélite
  se pruebe en vivo-real por otra ventana.
- **Kill-switch:** si el satélite real cae **-20% desde su pico**, se detiene auto
  y regresa a paper. Sin excepción, sin "ya va a rebotar".
- Caps por posición/trade vía `risk_manager` (ya existen).
- Reporte transparente: equity, drawdown, costos, en cada checkpoint.

## Qué promete el asistente (honesto)

- **SÍ:** correr un sistema disciplinado, automatizado, de bajo costo; reportar
  con transparencia; **proteger el downside** (que no se pierda feo por un bug o
  una racha).
- **NO:** inventar alfa que no existe. Si algún día prometo 30% anual tradeando,
  es mentira y este doc lo desmiente.

## Estado actual (2026-06-13)

- Paper horse-race con **3 caballos × $25k** (momentum_rotation, confirmed_mr,
  rsi_mr). momentum_news retirado (la API gratis de noticias no lo podía servir).
- Reloj de validación corriendo. Próximo checkpoint de plomería: ~2 semanas desde
  el reinicio del 5-jun. Checkpoint de **edge**: meses, no semanas.
- Dinero real: **OFF**. Gate sin abrir.

## Lo que sigue

1. Dejar correr el paper (cero intervención = correcto en esta fase).
2. A las ~2 semanas: reporte de plomería (¿corre limpio? ¿algún caballo se
   despega?) — NO es el gate de edge todavía.
3. Mientras, si Luis quiere que su dinero trabaje YA sin riesgo de código sin
   probar: CETES/SOFIPO o índice (decisión de core, separada del bot).
4. Revisar este doc en cada hito. No mover un candado sin releerlo.
