# Augur: pronóstico del precio de bolsa y apoyo a la decisión

Plataforma de procesamiento y analítica de datos para el pronóstico del precio de bolsa de la energía en Colombia y el
apoyo a la toma de decisiones de comprar, vender o esperar.

Proyecto final (Major Capstone Design), Departamento de Ingeniería Eléctrica y Electrónica, Universidad del Norte.

**Autores:** Juan David Barceló Barraza y Rafael Andrés Dede Perdomo
**Asesores:** Daniela María Charris Stand y José Daniel Soto Ortiz

**Dashboard publicado:** https://augurpredictor.streamlit.app

## Descripción

El precio de bolsa en Colombia se fija cada hora sobre una matriz mayoritariamente hidroeléctrica y puede multiplicarse
durante semanas en presencia del fenómeno de El Niño. XM publica el día anterior un predespacho ideal que solo cubre el
día siguiente y no indica su confiabilidad, y el precio de bolsa se publica con unos tres días de rezago.

Augur adquiere los datos públicos del mercado (XM/SIMEM) y el índice climático ONI (NOAA), los caracteriza en tiempo y
frecuencia, pronostica el precio a 24 y 72 horas con bandas de incertidumbre y traduce el pronóstico en recomendaciones
de comprar, vender o esperar para generadores y comercializadores, en un dashboard público que se actualiza tres veces
al día.

## Resultados

Validación fuera de muestra en 2026 y en cinco años históricos, con solo la información pública disponible al momento
de pronosticar:

| Indicador | Resultado |
|---|---|
| Error absoluto medio a 24 h | 33,7 COP/kWh (MAPE 10,0 %) |
| Frente a la persistencia | 68 % menos error |
| Frente al predespacho ideal de XM | 11 % menos error (Diebold-Mariano, p < 0,001) |
| Error absoluto medio a 72 h | 52,2 COP/kWh |
| Cobertura de la banda de incertidumbre | 80,6 % (objetivo 80 %) |
| Todo 2026 hasta el 31 de octubre | 38,6 COP/kWh frente a 42,2 del predespacho |
| Motor de decisión (backtesting) | Ventaja positiva en las dos mitades del periodo, para los dos roles y los dos horizontes |
| Validación con usuarios | Cuatro profesionales del sector; nota promedio de 7,8 de 10 |

El diseño definitivo corrige el predespacho ideal de XM con tres modelos CatBoost reentrenados cada mes. Se eligió tras
comparar más de quince familias de modelos (estadísticos, árboles de decisión potenciados y redes neuronales) frente a
la persistencia.

## Cómo funciona

1. **Adquisición:** descarga diaria de precio, demanda, generación, volumen de embalses, aportes hídricos y predespacho
   ideal (XM/SIMEM) y del índice ONI (NOAA), sincronizados a resolución horaria desde enero de 2019.
2. **Caracterización:** análisis temporal y espectral del precio (componente dominante de 24 h, armónicos de 12 y
   168 h) y filtro de Savitzky-Golay para la tendencia.
3. **Pronóstico:** predespacho corregido con CatBoost a 24 h y producto a 72 h, con bandas por calibración conforme.
4. **Motor de decisión:** señales de comprar, vender o esperar por percentiles del pronóstico, con filtro intradía,
   filtro de precio de contrato y selección del método por estabilidad.
5. **Dashboard:** semáforo, banda de incertidumbre, mapa de calor día-hora y precio real frente a pronóstico, con
   metadatos de cada imagen.

## Estructura del repositorio

```
src/motor_decision.py          motor de decisión: umbrales, señales, backtesting
dashboard/app.py               aplicación web (Streamlit)
scripts_experimento/           adquisición, experimentos, pronóstico de producción y generación de documentos
  actualizar_produccion.py     actualización diaria: datos, reentrenamiento mensual y pronóstico
notebooks/                     análisis exploratorio, caracterización y modelado
data/processed/                datasets horarios sincronizados y predespacho ideal de XM
data/processed/resultados/     predicciones y métricas de cada prueba
data/produccion/               serie de la operación diaria desde el 6 de agosto de 2026
.github/workflows/             actualización automática del pronóstico (GitHub Actions)
docs/                          informes de avance, manual de usuario, matriz de riesgos y referencias
```

## Instalación y uso

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

La actualización diaria se corre con `python scripts_experimento/actualizar_produccion.py`; en producción la ejecuta
GitHub Actions tres veces al día. La guía del motor de decisión está en `docs/motor_decision_guia.md` y el manual de
usuario en `docs/Manual_Usuario_augur.pdf`.

## Bitácora de avances

| Fecha | Avance |
|---|---|
| 2026-04-20 | Estructura inicial del repositorio y dependencias |
| 2026-06-17 | Descarga del precio de bolsa 2019-2024 desde la API de XM |
| 2026-07-15 | Datos 2019-2025 de precio, demanda, generación, volumen de embalses y aportes hídricos |
| 2026-08-08 | Dataset maestro horario, índice ONI y caracterización temporal y espectral |
| 2026-08-10 | Primer modelo XGBoost |
| 2026-09-03 | Extensión a 2026, variables predictoras compartidas, modelos Prophet, XGBoost, ARX+GARCH, N-BEATSx y N-HiTS, y validación walk-forward |
| 2026-09-08 | Modelo LightGBM |
| 2026-09-09 | Prueba de la relación hidráulica/térmica y de la extensión a 2017 |
| 2026-09-10 | Festivos en el pipeline, modelo LEAR, bandas de incertidumbre, prueba de Diebold-Mariano, motor de decisión v1, matriz de riesgos y bocetos del dashboard |
| 2026-09-11 | Horizonte de 72 h, CatBoost, ensambles y calibración adaptativa de las bandas |
| 2026-09-17 | Comparación con la literatura de mercados hidrodominantes y mejoras del ensamble a 24 y 72 h |
| 2026-09-18 | Análisis del error en la primera hora del día e informe comparativo de modelos |
| 2026-09-22 | Diagnóstico de los puntos de cambio intradía y biblioteca de referencias |
| 2026-09-23 | Selección de ventana por puntos de cambio (NOT) y revisión de la hora de corte del pronóstico |
| 2026-09-24 | Barrido de parámetros del motor, método híbrido y caracterización extendida a 2026 |
| 2026-09-25 | Informe de avance (plantilla ABET) |
| 2026-09-28 | Comentarios de la profesora aplicados al informe de avance |
| 2026-09-30 | Capturas del dashboard en el informe y presentación de avance |
| 2026-10-04 | Motor conectado al ensamble de 24 y 72 h; modo claro y oscuro del dashboard |
| 2026-10-06 | Ensamble y motor en orígenes históricos; bandas por hora y mapa de calor |
| 2026-10-07 | Predespacho ideal de XM en el pronóstico, filtro intradía del motor, detector de picos y desplomes, y escenario de información pública |
| 2026-10-14 | Dashboard publicado con actualización diaria automática y manual de usuario |
| 2026-10-15 | Rediseño del dashboard y primer borrador del informe final |
| 2026-10-19 | Vista Metodología, gráficas interactivas y revisión de accesibilidad |
| 2026-10-20 | Benchmark y walk-forward del escenario de información pública; reentrenamiento por la Resolución CREG 101 127 de 2026 |
| 2026-10-23 | Exportación de gráficas y reportes en PDF; referencias verificadas |
| 2026-10-26 | Videos del proyecto y validación de todo 2026 |
| 2026-10-27 | Manual de usuario rev. 1.5 |
| 2026-11-05 | Notebooks de caracterización actualizados |
| 2026-11-09 | Ventana de comparación elegible y filtro de precio de contrato en el motor; manual rev. 1.6 |
| 2026-11-10 | Pruebas de la curva de la Resolución CREG 101 127 y del índice SOI; referencias depuradas |
| 2026-11-12 | Caracterización del precio en el informe final |
| 2026-11-13 | Figuras de la presentación final y video explicativo |
| 2026-11-19 | Ajustes del dashboard tras la validación con usuarios |
| 2026-11-20 | Entrega del informe final, con la validación de cuatro profesionales del sector |

El detalle de cada cambio está en el [historial de commits](https://github.com/Ziskito/xm-spot-price-predictor/commits/main)
y las actualizaciones automáticas en [Actions](https://github.com/Ziskito/xm-spot-price-predictor/actions).
