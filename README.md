# xm-spot-price-predictor

Plataforma de procesamiento y analítica de datos para el pronóstico del precio de bolsa de energía en Colombia (mercado administrado por XM) y el apoyo a la toma de decisiones de compra/venta/espera, con horizonte de 24 a 72 horas.

Proyecto final — Depto. de Ingeniería Eléctrica y Electrónica, Universidad del Norte.

**Equipo:** Juan David Barceló Barraza, Rafael Andrés Dede Perdomo
**Asesores:** José Daniel Soto Ortiz, Daniela María Charris Stand

## Descripción

El mercado eléctrico colombiano opera con precios horarios altamente dependientes de la hidrología (cerca del 70% de la generación es hidroeléctrica). El proyecto construye un pipeline reproducible que adquiere, sincroniza y caracteriza temporal/espectralmente las variables públicas del mercado (precio de bolsa, demanda, generación, volumen de embalses, aportes hídricos, índice ONI de El Niño/La Niña), entrena y compara modelos de pronóstico, y traduce esos pronósticos —junto con su incertidumbre— en señales accionables para agentes generadores y comercializadores.

## Estructura del repositorio

```
data/
  precio_bolsa_*.csv, demanda_real_*.csv, generacion_total_*.csv,
  volumen_util_embalses_*.csv, aportes_hidricos_*.csv   # crudos, por rango de años
  external/oni_index.csv                                # índice ONI mensual (NOAA CPC)
  processed/
    dataset_maestro_2019_2025.csv, dataset_maestro_2026.csv   # series sincronizadas a resolución horaria
    dataset_features_2019_2025.csv, dataset_features_2026.csv # + features de ingeniería (lags, medias, armónicos, hidrología rezagada)
    resultados/                                          # métricas y predicciones guardadas de los modelos entrenados

notebooks/
  01_prueba_conexion_juan.ipynb              # conexión a las APIs de XM/SIMEM
  02_datos_embalses_hidrologia_juan.ipynb    # adquisición de embalses y aportes hídricos
  03_cracterizacion_precio_juan.ipynb        # caracterización temporal y espectral (FFT/periodograma, filtros, correlación)
  04_modelo_prophet_juan.ipynb               # modelo Prophet (el más débil de los 5, en todas las pruebas)
  05_features_compartidas_juan.ipynb         # extracción de características compartida (insumo de 04, 06, 07, 08, 09, 10)
  06_modelo_xgboost_juan.ipynb               # modelo XGBoost (puntual, cuantiles, SHAP, permutation importance, PDP, horizonte 72h)
  07_validacion_walkforward_juan.ipynb       # validación walk-forward, 6 orígenes temporales / regímenes ENSO, 5 modelos
  08_modelo_arima_garch_juan.ipynb           # ARX+GARCH, grid search de 27 configs, bandas de incertidumbre nativas
  09_modelos_deep_learning_juan.ipynb        # N-BEATSx y N-HiTS (neuralforecast), grid search, N-BEATSx "pesado", ensamble de 5 semillas
  10_diebold_mariano_juan.ipynb              # prueba de significancia estadística sobre el walk-forward -- decide el modelo final
  05_modelo_xgboost_Rafa.ipynb               # variante de XGBoost de Rafael (en desarrollo, pipeline de features propio)

src/         # (pendiente) módulos reutilizables fuera de notebooks
dashboard/   # (pendiente) dashboard de integración de pronósticos, recomendaciones e imágenes de decisión
```

## Datos

- **Fuentes:** API REST pública de XM y SIMEM, IDEAM, NOAA CPC (índice ONI).
- **Ventana:** enero de 2019 a agosto de 2026, resolución horaria. Se excluye 2015-2018 por el cambio estructural de la regulación CREG de 2015; se maneja la pandemia (2020) con una variable indicadora en vez de excluir esos datos.
- **Variables:** precio de bolsa, demanda, generación, volumen de embalses, aportes hídricos, ONI, indicador de pandemia.
- `pydataxm` no es compatible con pandas 3.0 (4 incompatibilidades documentadas); la adquisición se implementó con una función propia sobre `requests`.
- Demanda y generación **contemporáneas** (sin rezagar) se excluyen como predictoras porque se determinan en el mismo proceso de despacho que el precio; solo sus versiones rezagadas ≥ horizonte de pronóstico se usan.

## Estado del pipeline

- [x] Adquisición y sincronización de las 5 fuentes en un dataset maestro horario (`dataset_maestro_*.csv`)
- [x] Caracterización temporal y espectral (tendencia, estacionalidad, periodograma/FFT, comparación de filtros)
- [x] Extracción de características compartidas: 33 variables (rezagos, medias móviles, armónicos de calendario seno/coseno, hidrología rezagada y sus derivadas, volatilidad, ONI) — `05_features_compartidas_juan.ipynb`
- [x] Modelo Prophet entrenado y evaluado (`04`)
- [x] Modelo XGBoost entrenado, tuneado por grid search y evaluado (`06`)
- [x] Regresión por cuantiles (10/50/90%) sobre XGBoost, con verificación de calibración
- [x] Explicabilidad (SHAP) sobre el modelo XGBoost final, más permutation importance y Partial Dependence Plots como métodos complementarios (`06`)
- [x] Evaluación honesta a 72 horas, con todas las variables (incluida hidrología) re-rezagadas al horizonte real
- [x] Validación walk-forward en 6 orígenes temporales / regímenes hidrológicos distintos, 5 modelos (`07`)
- [x] ARX+GARCH: grid search de 27 configuraciones + bandas de incertidumbre desde su propia varianza condicional (`08`)
- [x] N-BEATSx y N-HiTS: entrenamiento, grid search de `input_size`, prueba de "más pesado" (perdió) y ensamble de 5 semillas (ganó) (`09`)
- [x] Prueba de significancia estadística (Diebold-Mariano) sobre el walk-forward — resultado: **N-BEATSx/N-HiTS le ganan a ARX+GARCH de forma significativa en 6/6 orígenes; ARX+GARCH es significativamente PEOR que la persistencia en 3/6; XGBoost peor que persistencia en 5/6** (`10`)
- [ ] Motor de decisión (reglas por percentil, señales compra/venta/espera) — OE3, no iniciado
- [ ] Biblioteca de imágenes de apoyo a la decisión y dashboard — OE3, no iniciado
- [ ] Backtesting del motor de decisión y validación con usuarios — OE4, no iniciado
- [ ] Bandas de incertidumbre para N-BEATSx (el modelo recomendado ahora) — todavía no tiene, solo ARX+GARCH y XGBoost las tienen construidas
- [ ] Informe comparativo de modelos (documento formal, Fase 3) — resultados existen pero no están consolidados en un documento aparte de este README

## Modelo recomendado (actualizado 2026-09-03, con respaldo estadístico)

**N-BEATSx** (o su ensamble de 5 semillas) — no ARX+GARCH. Ver la sección "Modelos de deep learning" más abajo y la bitácora del final del día para el detalle completo de por qué cambió la recomendación a lo largo de la sesión.

## Resultados actuales (evaluación única y honesta contra 2026)

Entrenamiento: 2019–2025 (60,625 filas). Prueba: 2026, enero–5 de agosto (5,208 filas).

**Horizonte 24 horas:**

**Actualizado con el pipeline de sincronización mejorado** (hidrología interpolada + ONI con rampa causal, ver Bitácora 2026-09-03 15:16):

| Modelo | MAE | RMSE | MAPE |
|---|---|---|---|
| Persistencia (t-24h) | 56.31 | 112.22 | 15.75% |
| Naive estacional (t-168h) | 124.53 | 180.60 | 36.75% |
| Prophet (regresores: hidrología, ONI, pandemia, precio t-24h; log-transform) | 95.84 | 150.22 | 19.93% |
| XGBoost (depth=3, lr=0.01, 33 features; log-transform) | 61.15 | 107.32 | 15.89% |
| N-HiTS (Challu et al. 2023, `neuralforecast`, input_size=168h) | 57.08 | 101.35 | 16.36% |
| N-BEATSx (Olivares et al. 2023, `neuralforecast`, input_size=168h) | 56.11 | **100.92** | 15.93% |
| **ARX+GARCH(1,1)** (mismos regresores que Prophet + armónicos de calendario; log-transform; confirmado por grid search de 27 configuraciones) | **55.76** | 109.97 | **15.43%** |

**Nota de lectura:** ARX+GARCH gana en MAE y MAPE; N-BEATSx gana claramente en RMSE (mejor manejo de errores grandes/picos extremos). No hay un único "mejor modelo" según todas las métricas — ver sección de deep learning más abajo para el detalle.

<sub>Los números de XGBoost/ARX+GARCH previos a este pipeline (60.66/107.18/15.81% y 55.67/108.18/15.33% respectivamente) siguen siendo válidos como referencia histórica pero ya no reflejan el pipeline actual. El walk-forward de 6 orígenes (sección siguiente) todavía corresponde a la versión anterior del pipeline — pendiente de re-ejecutar con los datos nuevos.</sub>

**Horizonte 72 horas** (todas las variables, incluida hidrología, rezagadas correctamente a 72h):

| Modelo | MAE | RMSE | MAPE |
|---|---|---|---|
| Persistencia (t-72h) | 94.32 | 157.03 | 26.87% |
| XGBoost (misma configuración) | 89.56 | 144.06 | 24.40% |

A 24h, XGBoost queda prácticamente empatado con la persistencia (le gana en RMSE, pierde por poco en MAE/MAPE); a 72h sí le gana con margen claro. Prophet no supera la persistencia en ninguna métrica.

### Hallazgo principal: el error se concentra en régimen El Niño

Separando el error por régimen ONI (mediana como corte alto/bajo) en el test 24h de 2026:

| Modelo | Error medio, ONI alto | Error medio, ONI bajo | Razón |
|---|---|---|---|
| XGBoost | 86.83 | 23.73 | 3.66× |
| Prophet | 139.24 | 25.94 | 5.37× |
| **ARX+GARCH(1,1)** | 78.84 | 22.98 | **3.43×** |

Este es el hallazgo central: el problema no es la precisión promedio, es la varianza del error condicionada al régimen hidrológico — consistente con lo reportado en la literatura para mercados con alta dependencia hidroeléctrica (Lago et al., 2021).

### Modelo ARX+GARCH (referencia: Muñoz-Santiago et al., 2017 — ARIMA con IGARCH)

Tercera familia de modelo, agregada para atacar directamente el mecanismo diagnosticado arriba: heterocedasticidad condicional (varianza del error que cambia con el régimen), no solo sesgo promedio. Ecuación de la media: regresión log-lineal sobre los mismos regresores de Prophet (hidrología rezagada, ONI, pandemia, precio t-24h) más armónicos de calendario, estandarizada para estabilidad numérica del optimizador. Ecuación de varianza: GARCH(1,1) sobre los residuales — `04`/`06`/`08_modelo_arima_garch_juan.ipynb`.

En el holdout de 2026, **es el único de los tres modelos que le gana a la persistencia en las tres métricas** (MAE, RMSE y MAPE) y tiene la razón El Niño/La Niña más baja. Diagnóstico adicional: la persistencia del GARCH (`alpha[1] + beta[1] = 0.9961`) confirma empíricamente el comportamiento cuasi-IGARCH que reporta la literatura de referencia — los choques de volatilidad prácticamente no se disipan, lo cual es consistente con que el error de XGBoost y Prophet se mantenga alto durante todo el régimen de El Niño, no solo en el instante del choque.

### Validación walk-forward (6 orígenes temporales, `07`) — actualizada con ARX+GARCH y Origen 6 (2026)

Se repitió el entrenamiento (misma configuración final de cada modelo, sin re-tunear) y evaluación en orígenes independientes con ventana de entrenamiento creciente, cada uno cayendo en un régimen ENSO distinto según `data/external/oni_index.csv`. Los primeros 5 son el compromiso formal del Anexo 1 ("5 orígenes temporales"); el Origen 6 se agregó aparte para ver si el patrón se sostiene también en el tramo más reciente — entrena con el mismo 2019-2025 completo que el holdout oficial de arriba (`04`/`06`/`08`), pero mide todo 2026 disponible con el mismo protocolo walk-forward, en vez de como una corrida aparte.

**Nota sobre 2025:** no se agregó un origen con test en 2025 porque ese año ya se usó como set de validación para elegir los hiperparámetros de XGBoost y Prophet (grid search en `06` y `04` respectivamente) — un origen ahí les daría ventaja artificial a esos dos modelos, no sería una medición limpia como las demás. ARX+GARCH no fue tuneado contra 2025, así que para él sí hubiera sido válido, pero se dejó fuera para no complicar la tabla.

| Origen | Régimen (ONI) | Persistencia MAE | XGBoost MAE | Prophet MAE | ARX+GARCH MAE |
|---|---|---|---|---|---|
| 1 (jul–sep 2020) | La Niña, inicio (-0.8 a -0.3) | 15.71 | 16.02 | 20.84 | 33.10 |
| 2 (jul–sep 2021) | La Niña, continuación (-0.6 a -0.3) | 6.32 | 8.06 | 36.20 | 7.51 |
| 3 (oct–dic 2022) | La Niña, triple-dip (-0.9 a -0.7) | 36.07 | 42.25 | 38.27 | 44.50 |
| 4 (oct–dic 2023) | El Niño fuerte (1.7 a 2.0) | 90.42 | 129.97 | 204.69 | 90.35 |
| 5 (ene–mar 2024) | El Niño, pico (1.2 a 1.8) | 42.49 | 46.96 | 52.12 | 43.66¹ |
| 6 (ene–ago 2026) | El Niño 2026, neutral→fuerte (-0.4 a 1.4) | 56.40 | 60.72 | 89.69 | **55.77** |

¹ Origen 5: el optimizador de ARX+GARCH no convergió limpiamente en ese origen (`convergence_flag=8`, "positive directional derivative for linesearch"); el número se reporta tal cual mas se trata con cautela. Los demás orígenes (incluido el 6) convergieron sin advertencias (`flag=0`).

**Origen 6 confirma, con un pipeline independiente, el resultado que ya se había visto en el holdout oficial de 2026** (MAE 55.67/60.66/92.25 en `08`/`06`/`04` vs 55.77/60.72/89.69 aquí — la pequeña diferencia es solo por el límite de fechas del test, 5185 filas vs 5208): ARX+GARCH es el único modelo entrenado que le gana a la persistencia en ese tramo. Pero, igual que con los 5 orígenes históricos, **eso no es la norma** — XGBoost y Prophet pierden en los 6 orígenes; ARX+GARCH pierde en 4 de 6 y gana o empata en los otros 2 (Origen 4 y Origen 6, ambos en regímenes de El Niño). Entre los tres modelos entrenados, ARX+GARCH sigue ganando en la mayoría de los orígenes (4 de 6) y solo pierde feo en el Origen 1.

Estabilidad (coeficiente de variación del MAE entre los 6 orígenes) y razón El Niño/La Niña:

| Modelo | MAE medio | CV | Razón El Niño / La Niña |
|---|---|---|---|
| Persistencia | 41.24 | 0.73 | 3.26× |
| XGBoost | 50.66 | 0.86 | 3.58× |
| Prophet | 73.64 | 0.93 | 3.64× |
| **ARX+GARCH** | 45.81 | **0.60** | **2.23×** |

El patrón se mantiene, y se ve incluso un poco más marcado con el sexto punto: XGBoost y Prophet siguen siendo menos estables que la persistencia y amplifican la brecha El Niño/La Niña; **ARX+GARCH sigue siendo el único que es más estable que la propia persistencia** y reduce esa brecha a poco más de la mitad de lo que muestra XGBoost (2.23× vs 3.58×). Resultado completo guardado en `data/processed/resultados/walkforward_5origenes.csv` (nombre de archivo sin actualizar, contiene los 6 orígenes).

**Salvedad importante — ¿es el régimen o es el tamaño del entrenamiento?** Cada origen posterior no solo cambia de régimen, también tiene más filas de entrenamiento (Origen 1: 12,385; Origen 6: 60,625) — las dos cosas están mezcladas en la tabla de arriba. Para aislar el efecto, se fijó el mismo test (2026) y se varió solo cuánto historial se usa para entrenar ARX+GARCH:

| Ventana de entrenamiento | Filas | MAE | vs. Persistencia (56.31) |
|---|---|---|---|
| 2019–2025 completo | 60,625 | 55.67 | gana (-1.1%) |
| Solo 2023–2025 | 26,304 | 58.74 | pierde (+4.3%) |
| Solo 2024–2025 | 17,544 | 73.52 | pierde (+30.6%) |
| Solo 2025 | 8,760 | 139.31 | pierde (+147%) |

La relación es clara y casi monótona: con menos historial de entrenamiento el desempeño se degrada fuertemente, y con solo un año (2025, un año tranquilo sin El Niño) el modelo queda muy por debajo de la persistencia al toparse con el Niño de 2026. **La victoria de ARX+GARCH en el Origen 6 depende en buena parte de haber entrenado con el histórico completo, no de que 2026 sea un régimen intrínsecamente fácil.** Hipótesis a explorar más adelante (no comprobada rigurosamente todavía): lo que importa no es solo el volumen de datos, sino si el modelo ya vio un régimen análogo en entrenamiento — 2019-2025 incluye el súper Niño de 2023-2024, así que el modelo "ya conoce" ese tipo de evento antes de que le toque predecir el Niño de 2026; entrenado solo con 2025 (año sin ningún Niño) nunca vio ese patrón. Esto matiza, pero no invalida, la conclusión de estabilidad de arriba: sigue siendo cierto que ARX+GARCH es el modelo que mejor se comporta bajo cambio de régimen *dado un entrenamiento con historia suficiente*, pero esa condición ("historia suficiente") es parte necesaria de la conclusión, no un detalle menor.

### Cuantificación de incertidumbre

La regresión por cuantiles (XGBoost, α=0.1/0.5/0.9) logra una cobertura empírica del **80.1%** dentro de la banda [q10, q90] (objetivo: ~80%), y el ancho de esa banda correlaciona con el error real (correlación 0.41; terciles de banda angosta/media/ancha producen error medio de 19.4 / 53.4 / 106.7 respectivamente). Esta banda es la base natural para el motor de decisión de OE3.

## Correcciones aplicadas

- Se eliminó una fuga de datos temprana (`precio_lag_1h` como predictor, inconsistente con el horizonte de 24–72h) del pipeline de features compartido.
- Se corrigió que las variables de hidrología (embalses, aportes) quedaran rezagadas también en su columna cruda, no solo en sus derivadas.
- Se unificó la referencia cruzada entre notebooks: `06` citaba una versión antigua (con fuga) de las métricas de Prophet; ahora cita el resultado corregido (92.25 | 144.73 | 19.55%).

## Instalación y entorno

Entorno virtual Python (`venv/`) con `requirements.txt`. Librerías clave: pandas, numpy, requests, matplotlib, seaborn, Prophet, XGBoost, SHAP, scikit-learn, scipy.

### Dependencia especial: Prophet requiere CmdStan

Prophet necesita CmdStan (motor de inferencia bayesiana en C++), que en Windows no se pudo compilar desde el código fuente (falla conocida de RTools/mingw32-make). Se instaló en su lugar como binario precompilado vía conda-forge, en un entorno separado del venv del proyecto:

    conda create -n cmdstan_env -c conda-forge cmdstan -y

La variable de entorno CMDSTAN debe apuntar a la carpeta resultante, por ejemplo:

    C:\Users\<usuario>\miniconda3\envs\cmdstan_env\Library\bin\cmdstan

En este equipo quedó fijada de forma permanente. En un equipo nuevo, hay que repetir la instalación de conda y volver a fijar esa variable.

## Bitácora de avances

Cada vez que se complete un avance real (notebook ejecutado, corrección aplicada, resultado nuevo), se agrega una entrada aquí con fecha y hora.

### 2026-09-09 16:18 — experimento reversible: variable relación H/T y extensión 2017-2025

Experimento hecho en rama aparte `experimento-2017-hidrotermica` (main intacto en `4b01570`), a partir del paper de la propia referencia [1] del Anexo 1 (Agudelo, López-Lezama, Velilla, 2015 — NARX con 4 entradas exógenas: demanda, probabilidad El Niño, **relación generación hidráulica/térmica**, volumen útil diario). De las 4, la relación H/T era la única que el proyecto no tenía representada — la restricción de no desglosar generación por tipo de recurso está en el Alcance del Anexo 1 (no en Identificación/Objetivos/Cumplimiento ABET, que son las 3 secciones formalmente bloqueadas), así que se probó bajo el criterio de reversibilidad total: todo en archivos nuevos, ninguno del pipeline existente tocado, borrable sin dejar rastro.

**Construcción de la variable**: reconstruida desde cero porque XM no publica "generación térmica del sistema" como métrica directa — se descargó `Gene` por `Entity=Recurso` (por planta, hora a hora, 2017-2026) y se cruzó con el catálogo `ListadoRecursos` (`Type`: HIDRAULICA/TERMICA/...) para agregar. Validada contra la Figura 3 del propio paper (rango 0-15): en enero 2017 dio media 9.92, rango 6.29-13.69 — coincide bien. Rezagada 24h igual que el resto de variables hidrológicas para no filtrar información futura.

**También se descargaron 2017-2018** (precio, demanda, generación, embalses, aportes) para poder probar, además de la variable, si ampliar la ventana de entrenamiento de 2019-2025 a 2017-2025 ayuda.

**Diseño**: factorial 2×2 (con/sin variable H/T × ventana 2019-2025/2017-2025) sobre los 3 mejores modelos del proyecto (ARX+GARCH, N-BEATSx, N-HiTS), holdout fijo en 2026. BASELINE recalculado desde cero (no comparado contra números viejos cacheados) como control de consistencia del dataset reconstruido — salió casi idéntico al pipeline actual (N-BEATSx 55.46 vs 56.11, ARX+GARCH 55.78 vs 55.76), confirmando que el dataset del experimento es equivalente.

**Resultado (MAE, holdout 2026):**

| Modelo | BASELINE (2019-2025, sin H/T) | A_HT (2019-2025, con H/T) | B_2017 (2017-2025, sin H/T) | C_2017_HT (2017-2025, con H/T) |
|---|---|---|---|---|
| ARX+GARCH | 55.78 | 55.68 | 55.84 | 57.10 |
| N-BEATSx | **55.46** | 57.66 | 57.75 | 57.19 |
| N-HiTS | **57.56** | 58.45 | 57.96 | 57.91 |

**Ninguna de las tres variantes mejora al BASELINE actual** en los modelos de deep learning; en ARX+GARCH la variable sola da una mejora marginal (-0.10) que desaparece al combinarla con la ventana 2017-2025 (empeora +1.32). Nota: la fila C_2017_HT de ARX+GARCH salió con `convergencia=0`, distinto del `convergencia=8` de las demás — no se investigó más a fondo por no cambiar la conclusión.

**Por qué se descartó la extensión a 2017**: la caracterización del precio por año (`notebooks_experimento/caracterizacion_2017_2026_juan.ipynb`, réplica de las Celdas I/J de `03` sobre 2017-2026) muestra que 2017 (media 106.1, CV 0.357) y 2018 (media 116.2, CV 0.427) son un régimen de precio estructuralmente más bajo y comprimido que todo lo demás desde 2019 (228.3 en adelante) — meterlos diluye el aprendizaje de N-BEATSx en vez de aportarle historia útil.

**Por qué se descartó la variable H/T (diagnóstico adicional, `scripts_experimento/diagnostico_variable_ht.py`)**: la correlación cruda de `gen_termica` con el precio es +0.744 (la más fuerte de toda la sesión, coincide con lo que encuentra el paper), pero un XGBoost con el set completo de features le da a las 6 variables H/T los puestos 22-38 de 39 en importancia por ganancia — casi al fondo. La señal es real pero redundante: en datos horarios, `precio_lag24h`/`precio_media_24h`/`precio_media_7d` ya capturan esa misma dinámica de forma más directa e inmediata que la generación térmica rezagada. El paper trabajaba con datos mensuales sin ese nivel de autorregresión de precio disponible, por eso a su modelo NARX sí le aportaba.

**Veredicto**: se mantiene el pipeline actual (2019-2025, sin variable H/T) sin cambios. Todo el experimento queda en la rama `experimento-2017-hidrotermica` (carpetas `scripts_experimento/`, `notebooks_experimento/`, archivos `*_EXP.csv`, `*_2017_2018.csv`, `generacion_por_tipo_2017_2026.csv`, `oni_index_2017_2026.csv`) — pendiente decidir si se archiva o se borra la rama.

### 2026-09-03 22:00 — cierre de la sesión

Resumen del día para retomar mañana sin tener que releer todo:

**Lo que cambió de fondo:** empezó como "¿mejora el pipeline con interpolación?" y terminó en una revalidación completa de qué modelo es realmente el mejor. La respuesta cambió tres veces a lo largo del día:
1. Holdout único de 2026 → parecía ganar ARX+GARCH.
2. Walk-forward de 6 orígenes → N-BEATSx/N-HiTS le ganan a ARX+GARCH cabeza a cabeza en los 6, y ARX+GARCH en promedio queda peor que la persistencia simple.
3. Diebold-Mariano (prueba de significancia) → confirma con estadística formal que N-BEATSx/N-HiTS son significativamente mejores que ARX+GARCH en 6/6 orígenes, y que ARX+GARCH nunca le gana a la persistencia de forma significativa.

**Modelo recomendado ahora: N-BEATSx**, idealmente su versión de ensamble (5 semillas, MAE 54.89 en el holdout 2026 — el mejor resultado de toda la sesión).

**Pendiente para mañana, en orden sugerido de prioridad:**
1. Decidir con los asesores si se saca formalmente a XGBoost y Prophet de la comparación final (recomendado, respaldado por Diebold-Mariano: ambos pierden significativamente contra la persistencia en la mayoría de los orígenes).
2. Construir bandas de incertidumbre para N-BEATSx (no las tiene todavía — usar `MQLoss` de `neuralforecast`, que soporta pérdida por cuantiles nativa) antes de decidir qué modelo alimenta el motor de decisión de OE3.
3. Consolidar el informe comparativo de modelos como documento formal (Fase 3 del Anexo 1) — todos los resultados ya existen, en el Excel (`resultados_completos_sesion.xlsx`) y en este README, falta redactarlo como entregable aparte.
4. Empezar el motor de decisión (OE3) — es lo más urgente por cronograma, no ha arrancado y ya se acerca la fecha de Fase 3.
5. Revisar la nota de convergencia irregular del ARX+GARCH en el walk-forward (Orígenes 4 y 5 con `convergence_flag=8`) si se decide seguir usando ese modelo para algo (ej. su GARCH para volatilidad, aunque ya no sea el modelo puntual recomendado).

**Archivos clave para retomar:** `data/processed/resultados/resultados_completos_sesion.xlsx` (todo consolidado y verificado), `data/processed/resultados/diebold_mariano.csv` y `walkforward_predicciones_crudas.csv` (crudos de la prueba de significancia), notebooks `07`–`10`.

### 2026-09-03 21:50

- **Prueba de Diebold-Mariano completa** (`10_diebold_mariano_juan.ipynb`), la más importante de la sesión para decidir el modelo final. Repite el entrenamiento de los 5 modelos en los 6 orígenes del walk-forward, guardando las predicciones hora por hora (`data/processed/resultados/walkforward_predicciones_crudas.csv`) para poder correr la prueba (`data/processed/resultados/diebold_mariano.csv`). Nota técnica: el primer intento se colgó 90 minutos sin completar ni un origen (archivo no llegó a guardarse); un chequeo de memoria/procesos no mostró nada anormal, un smoke-test de un solo modelo confirmó tiempos normales (194s), y el segundo intento sí completó sin problema en el mismo tiempo esperado (~40 min) -- causa del primer colgón no identificada, posiblemente contención transitoria de CPU justo después del job anterior.
  - **N-BEATSx y N-HiTS le ganan a ARX+GARCH de forma estadísticamente significativa en los 6 de 6 orígenes.**
  - **N-BEATSx nunca pierde de forma significativa contra la persistencia** (gana en 3/6, empata en 3/6); N-HiTS gana en 3/6, pierde en 1/6.
  - **ARX+GARCH es significativamente PEOR que la persistencia en 3 de 6 orígenes, y nunca le gana de forma significativa** -- confirma con prueba estadística (no solo comparación de puntos) que su ventaja del holdout 2026 no era robusta.
  - **XGBoost es significativamente peor que la persistencia en 5 de 6 orígenes.**
  - Conclusión para OE2: el modelo defendible como mejor ya no es ARX+GARCH -- es N-BEATSx (o su ensamble de 5 semillas), con respaldo estadístico, no solo descriptivo.
- Corregida una omisión propia: no había enfatizado que **ARX+GARCH en promedio (walk-forward) queda peor que la persistencia simple** -- señalado por una segunda opinión externa (otra sesión de Claude consultada por Juan David con el Excel de resultados), confirmado contra los datos crudos.

### 2026-09-03 18:17

- **Corrección de sincronización**: el notebook `06` (XGBoost) estaba desactualizado (última corrida antes de que los datos se restauraran a la versión final "ambos interpolados"). Reejecutado; ahora coincide exactamente con lo reportado. Se verificó numéricamente que `dataset_maestro_2019_2025.csv` actual es idéntico (diferencia 0.0 en todas las columnas) al usado por `07`, `08` y `09`.
- **Excel consolidado con todos los resultados verificados**: `data/processed/resultados/resultados_completos_sesion.xlsx` (8 hojas: holdout 2026, walk-forward, estabilidad, ambos grid search, bandas de incertidumbre, notas de verificación).
- **Segunda opinión externa (otra sesión de Claude, consultada por Juan David con el Excel) encontró un punto real que no se había enfatizado**: en el walk-forward, **ARX+GARCH en promedio (MAE 45.40) queda peor que la persistencia simple (41.24)** — los verdaderos ganadores son N-BEATSx (37.11) y N-HiTS (36.90). Confirmado contra los datos crudos. Se aclaró además que los hiperparámetros de cada modelo se aplicaron FIJOS en los 6 orígenes (no se reajustaron por origen) — el resultado no es sobreajuste local, aunque los hiperparámetros sí se eligieron usando el escenario de historial completo, no revalidados por origen.
- **N-BEATSx "más pesado" (mismo modelo, más capacidad) — perdió, cuarta vez que pasa esto en la sesión.** Ni más pasos de entrenamiento (2000 vs 1000, MAE 54.10 vs 51.48 en validación) ni más bloques por stack ([2,2,2] vs [1,1,1], MAE 53.78 vs 51.48) mejoraron sobre la configuración base. Mismo patrón que el grid search de ARX+GARCH y el de input_size de N-BEATSx/N-HiTS: la complejidad extra no ayuda en este dominio.
- **Ensamble de N-BEATSx (5 semillas, misma configuración base) — sí funcionó, y es ahora el mejor resultado de toda la sesión**: MAE 54.89, RMSE 97.61, MAPE 16.15%, razón ONI 2.83x. Le gana a ARX+GARCH en MAE, RMSE y estabilidad de régimen (solo pierde en MAPE). Confirma la estrategia del propio paper de N-BEATS (Oreshkin et al. 2020): ensamblar semillas distintas del mismo modelo bien ajustado, no agrandar la arquitectura.
- **Prueba de Diebold-Mariano en curso** (`10_diebold_mariano_juan.ipynb`) para confirmar si las diferencias del walk-forward son estadísticamente significativas o están dentro del ruido — pendiente de resultado.

### 2026-09-03 16:46

- **Bandas de incertidumbre construidas desde la propia varianza condicional de ARX+GARCH** (`08`), en vez de usar prestadas las de la regresión por cuantiles de XGBoost (inconsistencia detectada por Juan David: no tiene sentido mezclar el punto central de un modelo con la incertidumbre de otro). Detalle técnico: la recursión nativa de GARCH usa el residual de la hora inmediatamente anterior, lo cual violaría el rezago de 24h ya establecido en todo el pipeline; se usó el estado de varianza de hace 24h como proxy, aproximación razonable dado que la persistencia de volatilidad es ~1.0 (casi IGARCH puro, cambia lento).
  - Resultado honesto: **las bandas de XGBoost calibran mejor y son más informativas** que las de GARCH — cobertura 80.1% (XGBoost) vs 81.8% (GARCH) contra el objetivo de 80%; correlación ancho-de-banda↔error-real 0.410 (XGBoost) vs 0.282 (GARCH). La intuición de usar la fuente "nativa" y consistente con el modelo recomendado no se tradujo en mejor desempeño empírico — probablemente por la aproximación de 24h que las bandas de GARCH necesitan y las de XGBoost no.
  - Queda como decisión de diseño pendiente para el motor de decisión (OE3): priorizar consistencia (bandas de ARX+GARCH) o mejor señal empírica (bandas de XGBoost).

### 2026-09-03 16:41

- **Walk-forward completo reejecutado con el pipeline interpolado, 5 modelos (Prophet excluido a petición, por su desempeño consistentemente débil en todas las pruebas anteriores), incluyendo N-BEATSx y N-HiTS por primera vez.** Hallazgo más importante de la sesión: **N-BEATSx le gana a la persistencia en 4 de los 6 orígenes**, con márgenes grandes (12-17%) en los tres regímenes de El Niño (Origen 4: 75.42 vs 90.42; Origen 5: 36.51 vs 42.49; Origen 6: 49.19 vs 56.40). N-HiTS gana en 3 de 6, también con margen fuerte en El Niño. Es la primera vez en toda la sesión que un modelo entrenado supera consistentemente a la persistencia en pruebas independientes de regímenes distintos, no solo en un corte único. ARX+GARCH sigue siendo el más estable en general (CV 0.60, razón El Niño/La Niña 2.29x), pero N-HiTS (2.41x) y N-BEATSx (2.62x) ya se acercan. Esto eleva a N-BEATSx a candidato principal junto con ARX+GARCH, no un experimento secundario.
- Se limpiaron artefactos de los entrenamientos de hoy: archivo `nul` (vacío, accidental) eliminado; `lightning_logs/` (checkpoints de PyTorch Lightning) agregado a `.gitignore`.

### 2026-09-03 15:29

- **`requirements.txt` reescrito** — venía en UTF-16 (un caracter por cada dos bytes, ilegible como texto plano para la mayoría de herramientas) y le faltaban `xgboost`, `prophet`, `cmdstanpy`, `shap`, `arch`, `neuralforecast`, `torch`, `scikit-learn` y todo lo instalado en esta sesión. Ahora refleja el entorno real (`pip freeze`), en UTF-8 plano.
- **Nota de entorno**: instalar `neuralforecast` bajó `pandas` de 3.0.2 a 2.3.3 automáticamente (una de sus dependencias todavía no soporta pandas 3.0 completo). No rompió nada verificado hasta ahora, pero es un cambio real de versión que quedó fijado en el `requirements.txt` nuevo.

### 2026-09-03 15:27

- **Sincronizados los notebooks `02` y `03` con el pipeline de datos real** (interpolación lineal + rampa causal de ONI) — hasta ahora el código documentado seguía diciendo "ffill"/escalón aunque los archivos en disco ya usaban el método nuevo. `03` reejecutado completo (sin llamadas a API, seguro); `02` corregido en código pero no reejecutado completo para no disparar descargas reales a la API de XM otra vez — los datos que produce ya están verificados correctos vía script equivalente.
- **Dos modelos de deep learning nuevos**: `09_modelos_deep_learning_juan.ipynb`, N-BEATSx (Olivares et al. 2023) y N-HiTS (Challu et al. 2023), vía `neuralforecast` (Nixtla), CPU-only (sin GPU local), 1000 pasos de entrenamiento, evaluados con `cross_validation` en ventanas de 24h no solapadas cubriendo todo 2026. Corrección necesaria: las variables exógenas (embalses ~1e10, demanda ~1e6) tuvieron que estandarizarse antes de entrenar — sin eso la red diverge numéricamente (mismo tipo de problema que ya había aparecido con ARX+GARCH).
  - **N-BEATSx tiene el mejor RMSE de los 6 modelos probados en el proyecto** (100.92, vs 109.97 de ARX+GARCH) — maneja mejor los picos extremos de precio que cualquier otro modelo. En MAE (56.11), MAPE (15.93%) y razón ONI (3.00x) queda muy cerca pero no supera a ARX+GARCH (55.76 / 15.43% / 2.89x).
  - N-HiTS queda un poco por debajo de N-BEATSx en todo, pero también con RMSE mucho mejor que el resto (101.35).
  - Conclusión: no es "deep learning gana" ni "no sirve" — ayuda específicamente en el manejo de errores grandes/casos extremos (RMSE), sin superar a ARX+GARCH en precisión típica, estabilidad de régimen, interpretabilidad ni costo computacional (ARX+GARCH: ~1-2s de entrenamiento; N-BEATSx/N-HiTS: minutos, 4.5-5.3M de parámetros).
- **Grid search de ARX+GARCH** (ver detalle en la sección de resultados): confirma que GARCH(1,1) normal, la especificación simple ya usada, generaliza mejor que la alternativa "ganadora" en validación — se mantiene sin cambios.

### 2026-09-03 15:16

- **Pipeline de sincronización mejorado**: embalses/aportes ahora se suben a resolución horaria con interpolación lineal (antes: escalón/ffill); ONI ahora usa una rampa causal dentro de cada mes (desde el valor oficial del mes anterior hasta el del mes actual, sin usar nunca el mes siguiente) en vez de un escalón mensual. Ambos cambios son seguros frente a fuga de datos porque el hueco que necesitan (≤24h para hidrología, ≤1 mes para ONI vía la rampa causal) queda cubierto por el rezago ya existente en el pipeline. `dataset_maestro_2019_2025.csv` y `dataset_maestro_2026.csv` regenerados; `dataset_features_*` también.
- **Ablación factorial (2×2) sobre XGBoost y ARX+GARCH**: separando el efecto de cada cambio. Hallazgo limpio y consistente en ambos modelos — el ONI suavizado mejora la estabilidad de régimen (razón ONI alto/bajo) y casi no afecta nada más; la hidrología interpolada mejora el horizonte de 72h y casi no afecta la razón ONI. Ninguno de los dos cambia la precisión a 24h de forma relevante. Detalle completo en la sección de resultados de arriba.
- **Prophet, con ambos cambios**: empeoró (MAE 92.25→95.84, RMSE 144.73→150.22, MAPE 19.55%→19.93%), aunque la razón ONI mejoró un poco (5.37x→4.93x). Sigue siendo el modelo más débil de los tres. (Nota técnica: la primera corrida de este resultado salió contaminada por correr las ablaciones de datos en paralelo mientras Prophet aún ejecutaba —relee el CSV varias veces durante su ejecución—; se identificó por los timestamps de log de cmdstanpy, se descartó, y se corrió de nuevo en aislamiento.)
- **Grid search real de ARX+GARCH** (27 configuraciones: 3 tipos de volatilidad × 3 órdenes × 3 distribuciones, validado en 2025). El ganador de validación (GJR-GARCH(2,1) normal) generalizó peor en el holdout de 2026 que el GARCH(1,1) simple ya usado (MAE 56.20 vs 55.76) y quedó con persistencia de volatilidad 1.02 (por encima de 1 = proceso no estacionario/explosivo). **Se mantiene GARCH(1,1) normal como configuración final** — el grid search confirma formalmente, en vez de solo asumir, que la especificación simple era la correcta.

### 2026-09-03 14:12

- **Dos bugs de reproducibilidad encontrados y corregidos en `03_cracterizacion_precio_juan.ipynb`** al ejecutarlo desde un kernel limpio (`Restart & Run All` real, vía nbconvert): (1) dos celdas muertas que dependían de variables (`df_maestro`, `datos`) nunca creadas dentro de ese notebook — solo funcionaban si antes se corría `02` en la misma sesión de kernel viva; eliminadas por ser además duplicado exacto de lo que las celdas siguientes ya hacían bien de forma autónoma. (2) Bug de indexado de pandas (`.index[slice_con_fechas]` en vez de `.loc[slice]`) en las celdas nuevas de comparación de filtros.
- **Implementada la comparación de filtros pendiente de Fase 1** (promedio móvil vs. Savitzky-Golay), sobre el tramo del súper Niño 2023-2024. Savitzky-Golay le gana al promedio móvil en las dos métricas: llega al 90% de la subida real 376h (~16 días) más rápido, y deja menos ruido residual (std 115.86 vs 136.34).
- **Implementado el análisis de correlación precio-hidrología-demanda pendiente de Fase 1.** Correlación contemporánea: oni +0.36, demanda +0.32, generación +0.32, aportes_hidricos -0.30, volumen_embalses -0.19 (todos los signos correctos físicamente). Correlación cruzada por rezago: `aportes_hidricos` tiene su correlación más fuerte a ~7 días de rezago (-0.316), no en el instante; `volumen_embalses` es más fuerte en el instante (-0.188) y se debilita con el rezago — evidencia empírica adicional (más allá del argumento de fuga de datos) de por qué las features usan rezagos de 24h+.

### 2026-09-03 14:02

- Agregados permutation importance y Partial Dependence Plots a `06_modelo_xgboost_juan.ipynb`, como métodos de interpretabilidad complementarios a SHAP (sugerencia de los asesores: buscar otras métricas de explicabilidad). Top 3 por permutation importance: `precio_lag24h` (0.425), `precio_media_24h` (0.103), `precio_lag168h` (0.036) — coincide exactamente con el top 3 de SHAP. Correlación de Spearman entre ambos rankings: 0.791, buena validación cruzada entre dos técnicas con lógica distinta.
- **Gap encontrado al verificar la Fase 1 de la metodología del Anexo 1**: la comparación de filtros (promedio móvil vs. Savitzky-Golay) y el análisis de correlación precio-hidrología-demanda, ambos prometidos explícitamente en Fase 1, no están implementados en ningún notebook (`03_cracterizacion_precio_juan.ipynb` llega hasta el periodograma/FFT, que sí está bien implementado, y se detiene ahí). Pendiente de decidir quién lo implementa.
- Aclarado con ejemplo numérico por qué interpolar variables diarias (embalses, aportes) es seguro dado el rezago de 24-72h ya existente, pero interpolar ONI (mensual) no lo sería sin protección adicional — el hueco de interpolación (~30 días) excede el rezago disponible.

### 2026-09-03 13:19

- Prueba de ablación (script suelto, no en un notebook todavía): fijado el test en 2026, variado solo el tamaño de la ventana de entrenamiento de ARX+GARCH. Resultado: MAE se degrada de forma casi monótona con menos historial (55.67 con 2019-2025 completo → 139.31 con solo 2025), y con poco historial pierde feo contra la persistencia. Conclusión: la victoria de ARX+GARCH en el Origen 6 (2026) depende en buena parte de haber entrenado con el histórico completo, no de que 2026 sea un régimen fácil. Matiza la conclusión de estabilidad del walk-forward: sigue siendo el modelo que mejor aguanta el cambio de régimen, pero con la condición de que tenga historia de entrenamiento suficiente (idealmente que incluya un evento El Niño previo comparable).

### 2026-09-03 12:43

- Agregado un Origen 6 a `07_validacion_walkforward_juan.ipynb`, que llega hasta 2026 (train 2019-2025 completo, test ene-ago 2026, mismo protocolo walk-forward que los otros 5). Resultado: confirma, con un pipeline independiente, lo que ya se había visto en el holdout oficial (`04`/`06`/`08`) — ARX+GARCH le gana a la persistencia ahí (55.77 vs 56.40), XGBoost y Prophet pierden. Con los 6 orígenes, ARX+GARCH sigue siendo el más estable (CV 0.60) y con la menor razón de error El Niño/La Niña (2.23x vs 3.58x de XGBoost). Se descartó agregar un origen con test en 2025 porque ese año ya se usó como validación para elegir hiperparámetros de XGBoost y Prophet — no sería una medición limpia para esos dos modelos.

### 2026-09-03 12:22

- Extendido y reejecutado `07_validacion_walkforward_juan.ipynb` con ARX+GARCH en los mismos 5 orígenes. Resultado: ARX+GARCH pierde contra persistencia en 4/5 orígenes (casi empata en el Origen 4, El Niño fuerte: 90.35 vs 90.42) pero es el único modelo entrenado más estable que la propia persistencia (CV 0.68 vs 0.86) y con la menor razón de error El Niño/La Niña (2.36x vs 3.43x de persistencia, 4.00x de XGBoost, 4.04x de Prophet). Entre los tres modelos entrenados, gana en 3 de 5 orígenes. Caveat: el ajuste del Origen 5 no convergió limpiamente (`convergence_flag=8`), ese número puntual se reporta con reserva.

### 2026-09-03 12:19

- Nuevo modelo: `08_modelo_arima_garch_juan.ipynb` — ARX+GARCH(1,1), tercera familia de modelo (Muñoz-Santiago et al. 2017, ARIMA con IGARCH), reemplazando a Prophet como referencia de mejor desempeño. En el holdout de 2026 le gana a la persistencia en las 3 métricas (MAE 55.67 vs 56.31, RMSE 108.18 vs 112.22, MAPE 15.33% vs 15.75%) y tiene la razón de error El Niño/La Niña más baja de los tres modelos (3.43x). Persistencia GARCH (alpha+beta=0.9961) confirma comportamiento cuasi-IGARCH.
- Instalado el paquete `arch` en el venv (no estaba en `requirements.txt`, falta agregarlo formalmente).
- Extendido `07_validacion_walkforward_juan.ipynb` para incluir ARX+GARCH en los mismos 5 orígenes — en ejecución, pendiente de confirmar si el resultado del holdout 2026 se sostiene fuera de ese corte único (mismo tipo de sorpresa que ya tuvimos con XGBoost).

### 2026-09-03 12:04

- Corregida referencia cruzada obsoleta en `06_modelo_xgboost_juan.ipynb`: citaba la métrica de Prophet con fuga de datos (89.09/140.52/19.29%); ahora cita la corregida (92.25/144.73/19.55%). Notebook reejecutado completo para regenerar las salidas.
- Construida y ejecutada la validación walk-forward (`07_validacion_walkforward_juan.ipynb`), 5 orígenes temporales sobre regímenes ENSO reales (`data/external/oni_index.csv`). Resultado: con la configuración de hiperparámetros ya fijada, ni XGBoost ni Prophet le ganan a la persistencia en ningún origen; ambos son menos estables que la persistencia misma (CV 0.99 y 1.08 vs 0.86) y amplifican, en vez de atenuar, la brecha de error El Niño/La Niña. XGBoost le gana a Prophet en 4 de los 5 orígenes (pierde solo en el Origen 3, La Niña triple-dip, por un margen chico). Resultado consolidado en `data/processed/resultados/walkforward_5origenes.csv`.
