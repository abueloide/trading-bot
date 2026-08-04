# Postmortem — Spark spread vía acciones de generadoras (proxy) · FAIL ❌ (2026-07-31)

**Fuente:** "Power 2026" (power2026.ai) de Neel Somani, ex-quant researcher de un
hedge fund cubriendo power y gas. Material serio, con citas a EIA/DOE/ISO-NE/NREL
— la mejor fuente que ha entrado al proyecto (llegó por un reel que mandó Luis).

## Qué se probó
El libro define el margen del generador de gas:
`spark spread = Precio Power − (Heat Rate × Precio Gas)`.
Como no tenemos acceso a los mercados de power, se probó el **proxy accionario**:
si el spark spread es el margen, un movimiento del gas debería moverlas.
Correlación de retornos diarios de `NG=F` contra VST, NRG, CEG, TLN, PEG
(2021-2026, n=770–1377).

## Resultado
| Símbolo | corr mismo día | corr predictiva (gas hoy → equity mañana) |
|---|---|---|
| VST | +0.040 | −0.006 |
| NRG | +0.025 | +0.013 |
| CEG | +0.074 | −0.010 |
| TLN | −0.004 | +0.010 |
| PEG | +0.069 | −0.019 |

Todas indistinguibles de cero. Sin señal, ni contemporánea ni predictiva.

## Por qué murió (lo explica la propia fuente)
El libro dice que el dueño de una planta de gas **vende un forward strip de power
y compra uno de gas justamente para fijar su economía**. O sea: las generadoras
**se cubren hacia adelante por diseño**, así que su acción está deliberadamente
aislada del gas spot. El modelo del libro es correcto — y precisamente por eso el
proxy accionario no puede funcionar. La cobertura es la que rompe la transmisión.

## Lo que NO se puede probar (muro regulatorio, no de datos)
Los trades reales del libro — spark/dark spread en forwards, basis, FTRs,
congestion, virtuals DA/RT — exigen registro como participante del ISO, colateral
y crédito. **Cerrado para una cuenta retail.** No es cuestión de pagar un feed.

## Learning transferible
Cuando una fuente institucional describe cómo un actor **se cubre**, ese mismo
mecanismo dice *dónde NO va a haber señal* en el proxy líquido. Leer la cobertura
como filtro de hipótesis ahorra el backtest.
