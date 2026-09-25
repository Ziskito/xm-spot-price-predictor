# Nota para Rafael — cambios que afectan al motor de decisión (OE3)

*De Juan, 24 de septiembre de 2026. Detalle completo en la bitácora del `README.md` (entradas del 22 al 24 de septiembre).*

Ninguno de tus archivos se modificó: `05_modelo_xgboost_Rafa.ipynb`, `11_modelo_lightgbm_Rafa.ipynb`, `12_motor_decision_Rafa.ipynb`, `src/motor_decision.py` y `dashboard/` están intactos. Esto es solo lo que conviene que tengas en cuenta cuando integres.

## 1. Lo más importante: la exactitud depende de la HORA en que se lanza el pronóstico

El precio de bolsa es escalonado. Cada día tiene en promedio solo 7,2 precios distintos en sus 24 horas, porque cada planta oferta un precio diario y el precio de cada hora lo fija la planta marginal. El día de ofertas va de **00:00 a 23:00**: es la única transición horaria donde el precio **nunca** se repite exactamente.

Todas las cifras de 24 h del proyecto se miden lanzando el pronóstico **a las 00:00**. Esa es la hora más favorable, porque en ese momento ya se observó la primera hora del día de ofertas y quedan 23 horas de ese mismo día por pronosticar.

| si el pronóstico se lanza a las... | MAE del ensamble | MAPE |
|---|---|---|
| **00:00** (lo que reporta el proyecto) | **41,29** | **10,74%** |
| **23:00** (el caso más difícil) | **51,82** | **14,22%** |
| cualquier otra hora | entre ambos | — |
| persistencia (referencia) | 56,41 | 15,81% |

**Qué implica para el motor:**
- Si el motor consume el pronóstico lanzado a las 00:00, las cifras reportadas son las que aplican.
- Si lo lanza a otra hora, espera más error. A las 23:00 las horas de la madrugada pasan de un MAE de ~8-20 a ~40-44.
- **Los umbrales de confianza o de riesgo que dependan del error esperado deberían depender de la hora de lanzamiento**, no ser fijos.
- Las predicciones con el lanzamiento a las 23:00 ya existen, por si las necesitas: `data/processed/resultados/*_corte23.csv` y `ensamble_protocolo_honesto.csv`.
- **Las horas pico de la tarde (18-20 h) casi no cambian con la hora de lanzamiento.** Ahí el error es alto en cualquier caso: MAE ~90-107.

## 2. Composición vigente del ensamble

**24 h** — 6 votantes: Persistencia, XGBoost, ARX+GARCH, N-BEATSx, N-HiTS y GARCH-ged. El combinador es QRA ponderado para sMAPE, con pesos por franja de 6 horas (`combinador_optimo_mape.py`). Da MAE 41,29 / MAPE 10,74% / sMAPE 10,07%, verificado en 10 particiones.

**72 h** — con el puente 24h→72h (`puente_24h_a_72h.py`): MAE 58,84 / MAPE 16,94%.

Sigue pendiente, como habíamos quedado, regenerar los contratos oficiales de OE3 con estas composiciones cuando integres después de la sustentación.

## 3. Lo que NO hay que integrar

Se probaron varias ideas para bajar el error en los puntos de cambio intradía (las rampas). **Ninguna quedó**, así que no hay votantes nuevos que incorporar:

- **Selección de ventana por puntos de cambio (NOT):** mejoró 2026, pero en el walk-forward de 6 años solo gana en 3 de 6 frente al LEAR con histórico completo, y en El Niño 2023-24 empeora. **No se adopta.**
- Modelo de rampas, calibración de amplitud, análisis funcional (Hyndman), compensación de sesgo (STBC), precio "pegajoso" y oráculo de demanda: ninguno mejora las rampas.

La conclusión es que el error en las rampas es sobre todo un **límite de información**. Ni conociendo la demanda real de cada hora baja más de 3-5%. Lo que mueve las rampas son las ofertas y la disponibilidad por planta, que el Anexo 1 no nos deja desagregar.

## 4. Corrección pendiente en la señal de régimen

La señal de confianza por régimen hidrológico debería usar **umbrales relativos móviles de 90 días**, no umbrales absolutos del histórico. En 2026 no hay **ninguna** hora de "embalse bajo" en términos absolutos, así que con umbral fijo esa categoría nunca se activa.

## 5. Dónde está cada cosa

- Bitácora completa: `README.md`
- Scripts de los experimentos: `scripts_experimento/`
- Resultados: `data/processed/resultados/`
- Papers citados, con cita y enlace: `docs/papers/REFERENCIAS.md`. Los PDF no están en el repo porque es público; si los quieres, te los paso.
