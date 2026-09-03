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
  03_cracterizacion_precio_juan.ipynb        # caracterización temporal y espectral (FFT/periodograma)
  04_modelo_prophet_juan.ipynb               # modelo Prophet
  05_features_compartidas_juan.ipynb         # extracción de características compartida (insumo de 04, 06 y 07)
  06_modelo_xgboost_juan.ipynb               # modelo XGBoost (puntual, cuantiles, SHAP, horizonte 72h)
  07_validacion_walkforward_juan.ipynb       # validación walk-forward, 5 orígenes temporales / regímenes ENSO
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
- [x] Explicabilidad (SHAP) sobre el modelo XGBoost final
- [x] Evaluación honesta a 72 horas, con todas las variables (incluida hidrología) re-rezagadas al horizonte real
- [x] Validación walk-forward en 5 orígenes temporales / regímenes hidrológicos distintos (`07`)
- [ ] Motor de decisión (reglas por percentil, señales compra/venta/espera) — OE3, no iniciado
- [ ] Biblioteca de imágenes de apoyo a la decisión y dashboard — OE3, no iniciado
- [ ] Backtesting del motor de decisión y validación con usuarios — OE4, no iniciado

## Resultados actuales (evaluación única y honesta contra 2026)

Entrenamiento: 2019–2025 (60,625 filas). Prueba: 2026, enero–5 de agosto (5,208 filas).

**Horizonte 24 horas:**

| Modelo | MAE | RMSE | MAPE |
|---|---|---|---|
| Persistencia (t-24h) | 56.31 | 112.22 | 15.75% |
| Naive estacional (t-168h) | 124.53 | 180.60 | 36.75% |
| Prophet (regresores: hidrología rezagada, ONI, pandemia, precio t-24h; log-transform) | 92.25 | 144.73 | 19.55% |
| XGBoost (depth=3, lr=0.01, 33 features; log-transform) | 60.66 | 107.18 | 15.81% |
| XGBoost, mediana de la regresión por cuantiles (q50) | 59.83 | 109.88 | 16.07% |

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

Este es el hallazgo central: el problema no es la precisión promedio, es la varianza del error condicionada al régimen hidrológico — consistente con lo reportado en la literatura para mercados con alta dependencia hidroeléctrica (Lago et al., 2021).

### Validación walk-forward (5 orígenes temporales, `07`)

Se repitió el entrenamiento (misma configuración final, sin re-tunear) y evaluación en 5 orígenes independientes con ventana de entrenamiento creciente, cada uno cayendo en un régimen ENSO distinto según `data/external/oni_index.csv` (no se tocó 2025-2026, que sigue siendo el holdout oficial de arriba):

| Origen | Régimen (ONI) | Persistencia MAE | XGBoost MAE | Prophet MAE |
|---|---|---|---|---|
| 1 (jul–sep 2020) | La Niña, inicio (-0.8 a -0.3) | 15.71 | 16.02 | 20.84 |
| 2 (jul–sep 2021) | La Niña, continuación (-0.6 a -0.3) | 6.32 | 8.06 | 36.20 |
| 3 (oct–dic 2022) | La Niña, triple-dip (-0.9 a -0.7) | 36.07 | 42.25 | 38.27 |
| 4 (oct–dic 2023) | El Niño fuerte (1.7 a 2.0) | 90.42 | 129.97 | 204.69 |
| 5 (ene–mar 2024) | El Niño, pico (1.2 a 1.8) | 42.49 | 46.96 | 52.12 |

**Hallazgo:** con la configuración de hiperparámetros ya fijada (tuneada una sola vez sobre el split 2019–2024/2025), **ni XGBoost ni Prophet le ganan a la persistencia en ninguno de los 5 orígenes** — a diferencia del holdout único de 2026, donde XGBoost quedaba prácticamente empatado. La brecha es peor justo en el régimen que más importa: en el Niño fuerte de 2023 (Origen 4), XGBoost pierde por 44% y Prophet por 126%.

Estabilidad (coeficiente de variación del MAE entre los 5 orígenes) y razón El Niño/La Niña:

| Modelo | MAE medio | CV | Razón El Niño / La Niña |
|---|---|---|---|
| Persistencia | 38.20 | 0.86 | 3.43× |
| XGBoost | 48.65 | 0.99 | 4.00× |
| Prophet | 70.42 | 1.08 | 4.04× |

Ninguno de los dos modelos entrenados reduce la inestabilidad frente al régimen hidrológico — ambos son *menos* estables que la persistencia misma (CV más alto), y amplifican ligeramente, en vez de atenuar, la razón de error El Niño/La Niña. Esto refuerza que el valor del proyecto no está en ganarle a la persistencia en precisión puntual, sino en la cuantificación de incertidumbre (sección anterior), que si conecta directamente con el motor de decisión de OE3. Resultado completo guardado en `data/processed/resultados/walkforward_5origenes.csv`.

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

### 2026-09-03 12:04

- Corregida referencia cruzada obsoleta en `06_modelo_xgboost_juan.ipynb`: citaba la métrica de Prophet con fuga de datos (89.09/140.52/19.29%); ahora cita la corregida (92.25/144.73/19.55%). Notebook reejecutado completo para regenerar las salidas.
- Construida y ejecutada la validación walk-forward (`07_validacion_walkforward_juan.ipynb`), 5 orígenes temporales sobre regímenes ENSO reales (`data/external/oni_index.csv`). Resultado: con la configuración de hiperparámetros ya fijada, ni XGBoost ni Prophet le ganan a la persistencia en ningún origen; ambos son menos estables que la persistencia misma (CV 0.99 y 1.08 vs 0.86) y amplifican, en vez de atenuar, la brecha de error El Niño/La Niña. XGBoost le gana a Prophet en 4 de los 5 orígenes (pierde solo en el Origen 3, La Niña triple-dip, por un margen chico). Resultado consolidado en `data/processed/resultados/walkforward_5origenes.csv`.
