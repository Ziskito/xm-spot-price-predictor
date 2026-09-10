# Mockups visuales del dashboard (motor de decisión) — OE3

> **Estos archivos son una propuesta visual, no la interfaz final. Todo aquí está sujeto a cambios.**
> Muestran *cómo se vería* el dashboard rediseñado con enfoque de operador; la implementación real
> vive en [`../app.py`](../app.py) y se corre con `streamlit run dashboard/app.py`.

## Qué hay aquí

Un **lienzo de diseño editable** (canvas multi-artboard) con tres tableros:

| Archivo | Tablero | Qué muestra |
|---|---|---|
| `Main.dc.html` | **Operador** | La vista "¿qué hago ahora?": tarjeta grande de recomendación (COMPRAR / VENDER / ESPERAR), franja del día hora por hora, indicadores y mini-gráfico. Sin perillas: el método de umbral lo elige el backtest. |
| `Analista.dc.html` | **Analista** | Los controles finos de siempre: percentiles, método de umbral, rango de fechas, tabla comparativa de métodos y gráfica de backtest. |
| `Sistema.dc.html` | **Sistema** | Las muestras de color con su hex (los mismos valores del diccionario `T` en [`../app.py`](../app.py) y de [`../../.streamlit/config.toml`](../../.streamlit/config.toml)) y la tarjeta de recomendación en sus 3 estados / 2 roles. |
| `canvas.json` | — | Disposición de los tableros en el lienzo. |
| `motor-decision-dashboard.html` | — | El lienzo completo, listo para abrir en el navegador (ver / exportar PNG-PDF). ~2.4 MB, autocontenido. |

## Cómo verlo / editarlo

- **Rápido:** abrir `motor-decision-dashboard.html` en el navegador. Se puede navegar (pan/zoom),
  seleccionar elementos y exportar a PNG/PDF.
- **Editable en línea (con guardado):** el mismo lienzo está publicado como Artifact privado:
  <https://claude.ai/code/artifact/2faa1dea-bdc7-40ba-8f72-9066c1d008f9>
- **Regenerar** `motor-decision-dashboard.html` tras editar los `.dc.html`: se re-empaqueta con la
  herramienta de Claude Design (`/design` en Claude Code).

## Notas

- Los datos de los mockups son los de una corrida real (día 05/08/2026, señal ESPERAR,
  ventaja +346 COP/kWh…), pero **las gráficas son SVG ilustrativos**, no los plots de Plotly.
- El puente con Figma son los tokens de color: mismos nombres y valores en el tablero *Sistema*,
  en `T` (dentro de `app.py`) y en `.streamlit/config.toml`. Cambiar la paleta = editar esos hex.
