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
      fuentes_pronostico.json                             # registro de qué modelo/archivo alimenta cada horizonte del motor de decisión (OE3)

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
  11_modelo_lightgbm_Rafa.ipynb              # modelo LightGBM (mismo pipeline/features/horizonte 24h que 06), grid search, feature importance nativa
  12_motor_decision_Rafa.ipynb               # motor de decisión (OE3): genera bandas 72h, compara métodos de umbral, produce señales finales
  05_modelo_xgboost_Rafa.ipynb               # variante de XGBoost de Rafael (en desarrollo, pipeline de features propio)

src/
  motor_decision.py    # lógica del motor de decisión (OE3), reusable fuera de notebooks: umbrales, señales, backtest, carga de fuentes

dashboard/
  app.py                # prototipo Streamlit del motor de decisión -- `streamlit run dashboard/app.py`

docs/
  motor_decision_guia.md   # cómo correr/extender el motor de decisión y el dashboard, cómo cambiar de modelo sin tocar código
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
- [x] Bandas de incertidumbre para N-BEATSx en los dos horizontes (24h y 72h), calibración conforme adaptativa, cobertura ~78% en ambos — `09` / commit `3d8d685` / consumidas por `12`
- [x] Motor de decisión v1 (3 métodos de umbral por percentil, backtest económico, señal por rol generador/comercializador, 24h y 72h) — OE3, `src/motor_decision.py` + `12_motor_decision_Rafa.ipynb`
- [x] Prototipo de dashboard (Streamlit, `dashboard/app.py`) — OE3, primera versión interactiva sobre el holdout 2026
- [x] Motor de decisión desacoplado del modelo de pronóstico (`fuentes_pronostico.json` + `cargar_fuente_pronostico()`) — cambiar de modelo en cualquier horizonte es editar el JSON, no tocar código; guía completa en `docs/motor_decision_guia.md`. Ya se ejercitó una vez: el fallback de XGBoost a 72h se reemplazó por las bandas calibradas de N-BEATSx en cuanto existieron (`3d8d685`)
- [ ] Motor de decisión conectado a Corr-f/MHD/MPD (`scripts_experimento/metricas_decision.py`) en vez de solo el nivel de q50 — pendiente, ver nota en `12`
- [ ] Biblioteca de imágenes de apoyo a la decisión (más allá del prototipo de dashboard) — OE3
- [ ] Backtesting del motor de decisión con simulación de portafolio real (no solo precio promedio) y validación con usuarios — OE4, no iniciado
- [x] Informe comparativo de modelos (documento formal, Fase 3) — `docs/informe_comparativo_modelos.md`

## Nota para Rafael: cómo evaluar modelos para el motor de decisión (OE3)

Esta sección es una guía de referencia, no un log — se actualiza si cambia la recomendación, no se
archiva por fecha. Escrita el 2026-09-10 a partir de literatura nueva (Maciejowska, Lipiecki,
Uniejewski — *"Statistical and economic evaluation of forecasts in electricity markets: beyond
RMSE and MAE"*, arXiv:2511.13616 / Energy Conversion and Management 2026).

**El hallazgo central**: en ese paper, RMSE y MAE correlacionan **<0.20** con la ganancia real de
arbitraje de una batería contra el precio pronosticado. La métrica que sí correlaciona fuerte
(**>0.80**) es qué tan bien el pronóstico replica la *forma* de la curva de precio del día — no
qué tan chico es el error promedio. Esto importa directamente para OE3: **el modelo con mejor MAE
no es automáticamente el mejor insumo para una regla de compra/venta/espera.**

**Las 3 métricas que sí importan** (implementadas en `scripts_experimento/metricas_decision.py`,
reutilizable — `from metricas_decision import evaluar_modelo`):

- **Corr-f**: correlación de Spearman entre el perfil horario pronosticado y el real, promediada
  por día. La más correlacionada con ganancia real. Alto = el modelo acierta cuándo sube y cuándo
  baja el precio dentro del día, aunque el nivel exacto no sea perfecto.
- **MHD** (Min-Max Hour Deviation): a cuántas horas de distancia, en promedio, el modelo ubica el
  mínimo/máximo del día respecto a cuándo ocurrió realmente. Importa si la regla de decisión
  depende de *en qué hora* actuar, no solo de *si* actuar.
- **MPD** (Min-Max Price Deviation): compara el precio **real** en la hora que el pronóstico marca
  como mínimo/máximo del día contra el precio real en la hora en que **de verdad** ocurrió ese
  mínimo/máximo. Mide directamente cuánta plata se deja de ganar por operar en la hora equivocada
  (no es una diferencia de amplitud/spread — esa es otra métrica, Cov-e, que no implementamos).

**Corrección 2026-09-10**: la primera versión de `metricas_decision.py` tenía dos bugs frente a la
fórmula exacta del paper (Maciejowska, Lipiecki, Uniejewski — Energy Conversion and Management,
2026, ya conseguido en PDF completo): (1) MHD/MPD promediaban las 2 desviaciones del día por
separado en vez de **sumarlas** y promediar la suma sobre los días, lo que las dejaba en la mitad
de su valor real; (2) MPD estaba implementada como diferencia de spread pronosticado vs. real, no
como la comparación de precios reales en horas distintas que el paper realmente define. Ya
corregido — los números de abajo son los correctos.

**Resultado ya calculado sobre el holdout 2026** (`data/processed/resultados/metricas_decision_demo.csv`),
y sigue siendo contraintuitivo — léelo con cuidado antes de elegir qué modelo alimenta OE3:

| Modelo | MAE | Corr-f | MHD (horas) | MPD (COP/kWh) |
|---|---|---|---|---|
| Persistencia | 56.31 | **0.835** | **3.99** | 65.15 |
| **N-BEATSx (ensamble ventanas)** | **54.34** | 0.821 | 4.12 | **54.93** |
| ARX+GARCH | 55.76 | 0.772 | 8.03 | 80.68 |

**Con la fórmula corregida, N-BEATSx queda mejor que los otros dos en 2 de las 3 métricas de
decisión** (MAE y MPD), y muy cerca en Corr-f — a diferencia de la lectura anterior (ya
incorrecta), esta sí es una recomendación razonablemente clara para OE3: N-BEATSx (ensamble de
ventanas) es el candidato por defecto. Persistencia sigue ganando en Corr-f y MHD (por la misma
razón de siempre: copiar el precio de ayer reproduce una forma diaria plausible por construcción),
así que si la regla de decisión depende mucho de *replicar la forma exacta* del día antes que del
*nivel* del precio, vale la pena revisar ese caso con cuidado. ARX+GARCH queda claramente atrás en
las 3 métricas de decisión, pese a tener buen MAE — se equivoca por 8 horas en promedio sobre
cuándo ocurre el pico/valle del día.

**Recomendación concreta**: no elijas el modelo de OE3 solo por el MAE de este README. Antes de
integrar un modelo a la regla de decisión, corre `evaluar_modelo()` sobre el caso de uso real —
si la regla depende del *momento* de actuar, prioriza MHD bajo; si depende del *tamaño* del
movimiento esperado, prioriza MPD bajo; si es una regla direccional simple (subir/bajar/esperar),
prioriza Corr-f alto. Puede que termines necesitando una combinación (ej. nivel de ARX+GARCH +
corrección de forma inspirada en persistencia) en vez de un solo modelo puro.

**Sobre El Niño y regímenes** (de otra referencia leída hoy, arXiv:2508.00040, modelos por
régimen vía HMM): la literatura confirma que un régimen minoritario en el entrenamiento (como
El Niño, ~10% de nuestros datos de 2019-2025) es el caso difícil incluso con técnicas dedicadas —
no es que hicimos algo mal anoche al intentar atacarlo (4 hipótesis probadas, las 4 fallaron, ver
bitácora 2026-09-10). Si OE3 va a operar distinto en El Niño, probablemente conviene una regla
explícita basada en el régimen (ONI observado, no una variable que el modelo de precio tenga que
aprender) en vez de esperar que el pronóstico de precio ya lo resuelva.

**Bandas de incertidumbre — versión completa 24-72h (2026-09-10).** El Anexo 1 (OE2.2) exige
comparar los modelos en todo el rango 24-72h, no solo en el extremo. N-BEATSx es nativamente
multi-horizonte, así que se reentrenó con `h=72` (MQLoss, con festivos) en vez de `h=24`, y se
extrajeron cuantiles en cada paso de 1 a 72h. Hallazgo importante: **las bandas crudas pierden
cobertura rápido con el horizonte** (60.7% en 1-24h → 50.5% en 25-48h → 45.0% en 49-72h, contra
80% objetivo) porque su ancho casi no crece (136.6 → 144.1 → 148.8) pese a que el error real sí
crece fuerte (58.7 → 76.2 → 98.6 COP/kWh). La calibración conforme **estática** (un solo ajuste)
solo llegó a 67-72% de cobertura, mismo problema de siempre (2026 cambia de régimen a mitad de
año). La **adaptativa** (margen recalculado con ventana móvil de 30 días, igual criterio que la
versión de 24h) sí funcionó bien: **79.0% / 77.2% / 77.3%** de cobertura en los tres tramos, con
el ancho de banda creciendo correctamente con el horizonte (195 → 269 → 318 COP/kWh).

**Contrato para OE3, horizonte completo**: `data/processed/resultados/pronostico_con_bandas_72h_2026_adaptativo.csv`
(columnas `fecha_hora, cutoff, paso_horas, real, q50, q10, q90` — `paso_horas` indica cuántas
horas adelante del `cutoff` es esa fila, de 1 a 72). Usar esta versión para reglas que dependan
del horizonte completo; la versión de solo 24h (`pronostico_con_bandas_2026_adaptativo.csv`,
documentada arriba) sigue siendo válida si OE3 solo necesita el punto de 24h.

## Modelo recomendado (actualizado 2026-09-03, con respaldo estadístico)

**N-BEATSx** (o su ensamble de 5 semillas) — no ARX+GARCH. Ver la sección "Modelos de deep learning" más abajo y la bitácora del final del día para el detalle completo de por qué cambió la recomendación a lo largo de la sesión.

## Resultados actuales (evaluación única y honesta contra 2026)

Entrenamiento: 2019–2025 (60,625 filas). Prueba: 2026, enero–5 de agosto (5,208 filas).

### Tabla comparativa consolidada de modelos

Todos los modelos evaluados hasta el momento, en el mismo holdout (2026, horizonte 24 h, pipeline de sincronización mejorado). Los baselines (persistencia y naive estacional) se incluyen como referencia. **Negrita** = mejor valor de la columna.

| # | Modelo | Familia | Notebook | MAE | RMSE | MAPE | ¿Supera a la persistencia? | Bandas de incertidumbre | Diebold-Mariano (walk-forward) |
|---|---|---|---|---|---|---|---|---|---|
| — | Persistencia (t-24h) | Baseline | — | 56.31 | 112.22 | 15.75% | — (referencia) | — | referencia |
| — | Naive estacional (t-168h) | Baseline | — | 124.53 | 180.60 | 36.75% | No (peor en las 3) | — | no evaluado |
| 1 | Prophet | Serie temporal aditiva (bayesiana) | `04` | 95.84 | 150.22 | 19.93% | No (peor en las 3) | No | no evaluado (el más débil en todas las pruebas) |
| 2 | XGBoost | Árboles / gradient boosting | `06` | 61.15 | 107.32 | 15.89% | Parcial (solo RMSE) | Sí (regresión por cuantiles 10/50/90) | Sig. PEOR que persistencia en 5/6 orígenes |
| 3 | LightGBM | Árboles / gradient boosting | `11` | 62.16 | 108.06 | 15.90% | Parcial (solo RMSE) | No | no evaluado (≈ XGBoost, sin mejora) |
| 4 | N-HiTS | Deep learning (neuralforecast) | `09` | 57.08 | 101.35 | 16.36% | Parcial (solo RMSE) | No | Gana a ARX+GARCH en 6/6; vs persistencia 3 gana / 1 pierde / 2 empata |
| 5 | N-BEATSx | Deep learning (neuralforecast) | `09` | 56.11 | 100.92 | 15.93% | Parcial (MAE y RMSE) | No (pendiente, vía `MQLoss`) | Gana a ARX+GARCH en 6/6; nunca pierde sig. vs persistencia (3 gana / 3 empata) |
| 5b | N-BEATSx — ensamble 5 semillas | Deep learning (neuralforecast) | `09` | **54.89** | **97.61** | 16.15% | Parcial (MAE y RMSE) | No (pendiente) | mejor resultado puntual de la sesión |
| 6 | ARX+GARCH(1,1) | Econométrico (ARX + varianza condicional) | `08` | 55.76 | 109.97 | **15.43%** | **Sí (en las 3 métricas)** | Sí (desde su propia varianza condicional) | Sig. PEOR que persistencia en 3/6; nunca le gana sig. |

**Cómo leer la tabla:**

- **No hay un único "mejor modelo" según todas las métricas.** En el holdout único de 2026, ARX+GARCH gana en MAE y MAPE y el ensamble de N-BEATSx gana en MAE y RMSE; N-BEATSx (una sola semilla) tiene el mejor RMSE de los modelos no ensamblados.
- **Pero el holdout único engaña.** La validación walk-forward (6 orígenes / regímenes ENSO) y la prueba de Diebold-Mariano muestran que la ventaja de ARX+GARCH no es robusta: en promedio queda por debajo de la persistencia simple, mientras que **N-BEATSx / N-HiTS le ganan a ARX+GARCH de forma estadísticamente significativa en los 6 de 6 orígenes**.
- **Modelo recomendado (2026-09-03, con respaldo estadístico): N-BEATSx**, idealmente su ensamble de 5 semillas. Ver las secciones de walk-forward, deep learning y Diebold-Mariano más abajo.
- **XGBoost y Prophet** pierden significativamente contra la persistencia en la mayoría de los orígenes; están pendientes de sacarse formalmente de la comparación final (decisión con asesores).
- Métricas por régimen (razón de error El Niño / La Niña), estabilidad entre orígenes (CV) y el detalle de cada modelo están en las secciones siguientes.

**Horizonte 24 horas:**

**Actualizado con el pipeline de sincronización mejorado** (hidrología interpolada + ONI con rampa causal, ver Bitácora 2026-09-03 15:16):

| Modelo | MAE | RMSE | MAPE |
|---|---|---|---|
| Persistencia (t-24h) | 56.31 | 112.22 | 15.75% |
| Naive estacional (t-168h) | 124.53 | 180.60 | 36.75% |
| Prophet (regresores: hidrología, ONI, pandemia, precio t-24h; log-transform) | 95.84 | 150.22 | 19.93% |
| XGBoost (depth=3, lr=0.01, 33 features; log-transform) | 61.15 | 107.32 | 15.89% |
| LightGBM (depth=4, num_leaves=15, lr=0.01, mismas 33 features; log-transform) | 62.16 | 108.06 | 15.90% |
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

### 2026-09-18 — Investigación a fondo del origen de pronóstico de N-BEATSx/N-HiTS: lo que parecía un bug NO lo era, se documenta el salto de precio de medianoche, y se corrige la identificación de las horas de mayor error

Entrada larga a propósito: esta investigación arrancó por una observación del usuario sobre un gráfico, terminó tocando los números centrales del proyecto, y el desenlace fue que **los números originales eran correctos y quedaron restaurados**. Vale la pena dejar el recorrido completo para no repetirlo.

**Punto de partida (observación del usuario).** En el gráfico de error por hora, la hora 0 tenía un error muy alto mientras sus vecinas (23h y 1h) lo tenían muy bajo. Un salto así entre horas contiguas no tiene sentido físico.

**Primer hallazgo (real).** `neuralforecast.cross_validation` arma sus ventanas contando hacia atrás desde el final de la serie. Como el rango de prueba del Origen 6 tenía 5,185 horas (no múltiplo de 24, porque `test_fin` estaba escrito como `"2026-08-05"` sin hora, a diferencia de los Orígenes 1-5 que sí llevan `23:00`), `n_windows = 5185 // 24 = 216` dejaba fuera la primera hora del año. Resultado: los cortes de N-BEATSx/N-HiTS caen a las **00:00**, no a las 23:00 como los demás modelos. Verificado con conteos exactos (5,184 filas desde 01:00 vs 5,185 desde 00:00 en los otros tres modelos).

**Corrección aplicada, y el resultado desconcertante.** Al fijar el rango a 5,184 horas exactas (cortes a las 23:00) y reentrenar, N-BEATSx pasó de MAE 46.82 a 57.38 — de ser el mejor modelo individual a uno de los peores. Se verificó con **dos semillas independientes** (42 y 123 → 57.38 y 56.10, correlación entre corridas 0.9926), así que no era varianza de entrenamiento.

**El diagnóstico que lo explicó todo.** El modelo "corregido" pronosticaba a 1 hora de distancia con MAE 48.2, cuando **copiar el precio de la hora anterior da 30.05**. Ningún modelo sano pierde contra eso a un paso. Además el gradiente de horizonte había desaparecido (h1=48.2, h24=45.5, plano), cuando en el original era de manual (h1=10.3, h24=52.3). La causa está en los datos, no en el modelo:

| Hora objetivo | Desde 1h antes | Desde 2h antes | Desde 24h antes |
|---|---|---|---|
| **00:00** | **56.49** | 75.92 | 46.76 |
| **01:00** | **7.39** | **59.42** | 46.99 |
| 02:00 | 9.23 | 12.01 | 46.98 |

**El precio de bolsa da un salto grande al cambiar de día de despacho**: 56.5 COP/kWh de diferencia media entre las 23:00 y las 00:00, contra 7.4 entre las 00:00 y la 01:00. Con corte a las 00:00 el modelo ya conoce el precio de apertura del día y pronostica el resto **sin cruzar el salto**; con corte a las 23:00 tiene que atravesarlo. Son dos tareas distintas, y la segunda es genuinamente más difícil. Ninguna de las dos versiones estaba rota: el modelo "corregido" incluso le gana a la persistencia en su propia tarea (46.6 contra 59.4 a dos horas).

**Prueba definitiva de reproducibilidad.** Se entrenaron las dos geometrías en el mismo entorno, misma semilla, mismo código: la original devolvió **MAE 46.82 clavado**, idéntico al número histórico del proyecto. El entorno, las librerías y el modelo original están bien.

**Decisión: se restauró el estado original** (`walkforward_predicciones_crudas.csv` recuperado desde el commit `cbff7d8`), y se verificó que todo el pipeline vuelve a sus cifras conocidas: ensamble de 24h MAE 41.19 / MAPE 10.72% / sMAPE 10.04%, rMAE del Origen 6 de N-BEATSx 0.3786. Nada del proyecto quedó degradado.

**Lo que sí queda documentado como limitación honesta**, y conviene decirlo en la sustentación si preguntan: los dos modelos neuronales usan un origen de pronóstico a las 00:00 del día objetivo, mientras Persistencia, XGBoost y ARX+GARCH usan variables con rezago de 24h o más. **Los votantes del ensamble no comparten exactamente el mismo conjunto de información.** No es fuga hacia el futuro (ninguno usa datos posteriores a la hora que predice), pero sí es una asimetría que favorece a las redes, sobre todo en las horas de madrugada. Igualar los conjuntos de información de los cinco votantes es trabajo pendiente, no algo resuelto.

**Corrección aparte que sí se adoptó.** Al revisar el diagnóstico por hora se encontró que la afirmación repetida en varias entradas anteriores —"el pico vespertino (18-20h) concentra el 37.8% del error"— identificaba mal las horas. El 37.8% siempre estuvo bien calculado (con `nlargest(6)` sobre el aporte al MAPE), pero **las 6 horas que de verdad lo componen son 0, 8, 9, 10, 18 y 19**: la hora 20 no entra (queda en el puesto 7) y faltaban la medianoche y el bloque de media mañana. En esas seis horas el sesgo explica solo el **8.4%** del error (91.6% es varianza), lo que refuerza —no debilita— la conclusión de que ese error no es corregible por modelado. El techo teórico recalculado sobre el modelo actual (acertando perfecto esas seis horas) es **6.66%**, no el 7.01% que se venía citando desde el v4 original. Gráfico e informe actualizados.

Siete scripts del proyecto tenían la constante `PICO = [18, 19, 20]` y quedaron corregidos a `[0, 8, 9, 10, 18, 19]`. **Cuidado con un detalle que ya causó un bug**: en `especialista_pico_24h.py` esa constante no es solo una máscara, se pasa como paso de horizonte a `construir_para_paso`. Con el corte a las 00:00 la hora H es el paso H, pero la hora 0 es la medianoche del día *siguiente*, o sea el paso 24. Pasar `h=0` hace `t = c + 0 = c`: el objetivo pasa a ser el propio corte, que además es una variable de entrada (`p_corte_lag0`), y el MAE se va a 0.68 por fuga pura. Ya está corregido con un mapeo explícito `hora 0 -> paso 24`. Los experimentos de especialistas de pico que usaron la definición vieja de horas no se volvieron a correr.

**Lección de proceso, propia:** se sobrescribió `walkforward_predicciones_crudas.csv` sin respaldo antes de reentrenar. Se pudo recuperar solo porque el archivo estaba commiteado. Antes de regenerar cualquier insumo del pipeline hay que guardar copia.

**Inconsistencia encontrada y corregida en el informe.** La comparación con la literatura (MASE, NRMSE, sMAPE, TAPI) se estaba calculando sobre el **v4 original de 5 votantes**, no sobre el modelo vigente de 6 votantes con combinador alineado. El informe se estaba subestimando: con el modelo actual, MASE **0.914** (no 0.943), NRMSE **21.58%** (no 21.66%), sMAPE **10.04%** (no 10.38%), TAPI diario **76.4%** (no 75.9%). Corregido en gráfico y texto.

### 2026-09-25 — Informe de avance en la plantilla ABET (sustentación 25-sep), con todas las cifras recalculadas tras el arreglo de la demanda

- Documento: `docs/Informe_Avance_ABET_Barcelo_Dede.docx` (+ `.pdf`), generado por `scripts_experimento/generar_informe_avance.py` sobre `docs/PLANTILLA_FINAL_ABET (1).docx` (la plantilla no se modificó). Se retiraron los recuadros de guía y las líneas de "Extensión máxima", como en los ejemplos del semestre anterior; se aplicaron las correcciones de los asesores vistas en esos ejemplos (título de tabla arriba, "Figura N.", sin cuarto nivel de numeración). 30 páginas, 20 tablas, 10 figuras, 36 referencias IEEE citadas en el texto.
- **Cifras regeneradas con los datos corregidos** (respaldo previo en `data/processed/resultados/backups_pre_informe_avance_25sep2026/`): `combinador_optimo_mape.py` (24h: MAE 40,97 / MAPE 10,45% / sMAPE 9,80%, semilla 42; 10 particiones 41,10/10,49%/9,84%) y `puente_24h_a_72h.py` (72h global: 58,84→**58,75**, MAPE 16,94→**16,85%**). Nuevas tablas y figuras en `data/processed/resultados/informe_avance/` y `scripts_experimento/informe_avance_assets/` (`informe_avance_resultados.py`, `informe_avance_costo.py`, `informe_avance_diagrama.py`).
- **Resultados nuevos medidos para el informe**: DM de una cola ensamble vs persistencia t=7,76 (24h), t=10,51 (72h); **ensamble desplegable** (pesos causales, ventana expansiva, desde 15-ene) MAE 42,89 / MAPE 10,57% vs persistencia 57,82/15,52%, t=7,02; persistencia honesta a 72h por tramo (rMAE 0,733/0,784/0,774); rMAE mensual 2026 (ensamble <1 en 8/8 meses, peor 0,830; ARX+GARCH peor 1,009); **costo computacional** medido en este equipo (Ryzen 7 7730U): XGBoost 0,9 s, ARX+GARCH 2,8 s, GARCH-ged 19,5 s, combinador 3,7 s, redes 6,8 min → reentrenamiento completo 7,2 min.
- **Hallazgos a decir con honestidad (están en el informe)**: (1) en los orígenes walk-forward 1-5 (corte 23:00) ningún modelo individual supera de forma consistente a la persistencia (N-BEATSx empata en MAE medio, 49,88 vs 49,82); el ensamble solo es evaluable en 2026. (2) La composición del ensamble se eligió mirando 2026, por eso la cifra desplegable es la de referencia. (3) El mayor pico de precio (2 499 COP/kWh diario, 30-sep-2024) ocurrió con ONI neutro y el volumen de embalses más bajo de todos los septiembres 2019-2025: el ONI solo no explica el precio.
- Porcentajes de avance propuestos en el informe (a confirmar con Rafael): OE1 100%, OE2 90%, OE3 70%, OE4 35%, general 74%.
- **Revisión posterior del documento (misma noche)**, leyendo el texto completo del PDF: (1) el "MAE medio 49,88 vs 49,82" que se atribuía a los orígenes 1-5 era la media de los 6 orígenes; en los orígenes 1-5 N-BEATSx queda **50,83 vs 48,50** (no gana en ninguno, pierde en O1) — corregido. (2) "Reentrenamiento completo 7,2 min" era exagerado: solo se cronometró el ensamble de 24 h; ahora se dice así y que el canal de 72 h no se cronometró. (3) Spearman SHAP/permutación actualizado a **0,811** (salida vigente del notebook 06; el 0,791 era de una versión anterior). (4) El MAE de 51,90 con corte a las 23:00 se marca como calculado antes del arreglo de la demanda (afecta 48 de 5.184 h). (5) Se precisó que los hiperparámetros de XGBoost, ARX+GARCH, N-BEATSx y N-HiTS se validaron contra 2025 (verificado en los notebooks), y que la elección de GARCH-ged sí se hizo mirando 2026. (6) Referencias verificadas contra Crossref: se agregaron DOI y se corrigieron N-HiTS (páginas 6989-6997, nombres completos) y Maciejowska et al. (ya publicado: *Energy Convers. Manag.* 356, 121408, 2026); revistas en cursiva. (7) Formato numérico uniforme (punto de miles, signo menos tipográfico).
- **Reestructuración (25-sep, a pedido del usuario): metodología de diseño completa y máximo 22 páginas** (quedó en 22). Agregado en 6.8: procedimiento de diseño (alternativa → evaluación fuera de muestra → adopción solo con DM p<0,05 y 10/10 particiones → bitácora); Tabla 11 con las **40 variables** por familia, fórmula y justificación desde OE1 (el "33 variables" del README antiguo no incluía los 7 festivos); Tabla 12 con fundamento, configuración e hiperparámetros de cada modelo y cómo se ajustó; modelos descartados con su rMAE en 2026 (CatBoost 1,016, hurdle 1,028, Markov-Switching 0,990, RLS 1,052, TFT 1,020, LSTM 1,173, LEAR24 0,940; Prophet/LightGBM de una versión anterior); Tabla 13 con 12 decisiones de diseño, alternativa descartada y evidencia — incluida la codificación del ONI (continuo frente a categorías difusas NOAA, que empeoraron N-BEATSx 56,11→58,47; la bandera binaria no se probó, así se dice). Recortado: 3 figuras, tablas de alcances/limitaciones compactadas, entregables en un párrafo, tabla de factores, riesgos a 6, backtest solo con el método elegido, bandas y costo en una tabla, y la tabla de comparación con la literatura (redundante con la Tabla 4).
- `docs/Anexo 1_Formulación del Proyecto - 2026_30.pdf` copiado desde Descargas (versión definitiva del 10-sep) para consulta local; **ignorado en git** porque tiene cédulas y el repo es público. Ojo: `docs/ANEXO_1_formulacion_proyecto.md` (ya en GitHub desde el 10-sep) también tiene cédulas y códigos.
- Entorno: se instaló `streamlit==1.63.0` (ya fijado en `requirements.txt` pero ausente del venv) fijando `typing_extensions==4.15.0`, sin cambiar ningún paquete existente (`pip check` limpio), para capturar el dashboard de Rafael en funcionamiento (solo lectura, sin tocar `dashboard/`).

### 2026-09-24 — Notebook 03 extendido al rango completo del Anexo 1 (2019-2025 + 2026): correlación, periodograma y estadísticas ahora usan los 66.576 datos completos, no solo hasta 2025

- Contexto: al revisar notebook 03 con el usuario se notó que caracterizaba solo `precio_bolsa_2019_2025.csv`/`dataset_maestro_2019_2025.csv`, dejando fuera los ~7 meses de 2026 ya disponibles y usados en el resto del proyecto — inconsistente con el compromiso del Anexo 1 (§4.1, "enero 2019 a agosto 2026").
- Cambio de alcance mínimo y de bajo riesgo: solo se extendieron las celdas que **leen** datos (Celda B ×2, y la celda de correlación), sin tocar las celdas 22-24 que construyen `dataset_maestro_2019_2025.csv` — ese archivo es compartido con Rafael/otros notebooks y no se modificó (verificado con `git status` antes y después: 0 cambios).
- Verificado antes de extender: mismo esquema de columnas en los pares `_2019_2025`/`_2026` de cada fuente (`precio_bolsa`, `dataset_maestro`), y el ONI oficial ya cubre 2026 completo (interesante: vuelve a subir a niveles tipo El Niño, 0.9-1.4 desde mayo 2026).
- Notebook reejecutado completo (`nbconvert --execute`, 30/30 celdas sin error, numeración intacta). Nuevo rango: 2019-01-01 a 2026-08-05 23:00, 66.576 filas (antes 61.368), sin huecos.
- **Resultados que cambian**: media del precio 331.5→336.7 COP/kWh; correlación contemporánea con `oni` sube de 0.355 a 0.387 (el repunte de El Niño en 2026 refuerza la relación); `aportes_hidricos`/`volumen_embalses` bajan levemente en magnitud (-0.302→-0.286 y -0.189→-0.164 en 0h) pero el patrón cualitativo de la correlación cruzada por rezago **no cambia**: `aportes_hidricos` sigue siendo más fuerte a 168h (-0.309) que en 0h, `volumen_embalses` sigue siendo más fuerte en 0h (-0.164) y decae con el rezago.
- **Resultado que NO cambia**: la comparación Savitzky-Golay vs. promedio móvil (ventana del súper Niño 2023-2024) da exactamente los mismos números de retraso (724h vs 348h, Savitzky-Golay gana por 376h) porque son cálculos de ventana local, sin alcance hasta 2026; solo la desviación estándar del residuo (calculada sobre toda la serie) sube levemente (136.34→138.03 y 115.86→117.70) por tener más observaciones.
- De paso, se corrigió la celda de conclusión de la correlación (Paso 1.5c), que antes quedaba en condicional ("si la correlación es más fuerte en un rezago...") sin resolver contra los números reales — ahora declara explícitamente que el resultado es mixto entre las dos variables, con la explicación física (estado acumulado vs. flujo).

### 2026-09-24 — Bug real encontrado en datos EN VIVO al probar el notebook 02 en copia aislada: demanda del 4-5 de agosto de 2026 mal (70-86% por debajo), corregido y reentrenado. Mejora confirmada: MAE 41.34→41.10, MAPE 10.76%→10.49%, 10/10 particiones

Origen: el usuario preguntó si era seguro re-ejecutar los notebooks 01 y 02 (recordaba que algunas celdas no corregían bien el error). En vez de correrlos sobre el proyecto real, se hizo una **copia aislada** en el scratchpad (`notebooks/` + `data/` + `requirements.txt` como marcador de raíz, verificado que `encontrar_raiz_proyecto()` resuelve a la copia y no al proyecto real) y se ejecutaron ahí con `jupyter nbconvert --execute`.

**Notebook 01**: corrida limpia, 17/17 celdas sin error, numeración `In[1]`-`In[17]`. Los 52.608 valores de precio 2019-2024 descargados hoy coinciden **exactamente** (diff=0.0) con los ya guardados — confirma que la descarga es determinística y XM no revisa precios ya liquidados. **Notebook 01 en vivo actualizado** con esta ejecución (solo arregla la numeración, a pedido del usuario; no se tocaron los comentarios `# Celda N` ni las 3 celdas sueltas sin etiquetar, que quedan como registro real del proceso).

**Notebook 02**: corrida limpia también (31/31 celdas sin error), pero comparando los datos resultantes contra los archivos en vivo salió un hallazgo real:

- `volumen_util_embalses` 2019-2025: coincide perfecto (0 diferencias en 2.557 días).
- `precio_bolsa` 2019-2025: coincide perfecto salvo diciembre 2025, con desfases de centésimas (XM publica el mes más reciente como provisional). Benigno.
- `aportes_hidricos` 2019-2025: 89% de los días difieren, pero el error promedio es 0,042% y el máximo 2,38% (un solo día) — ruido de redondeo irrelevante frente al ~10-16% de MAPE del proyecto. Benigno.
- **`demanda` en `dataset_maestro_2026.csv`: las 48 horas del 4 y 5 de agosto de 2026 estaban entre 70% y 86% por debajo del valor real** (ej. 04-ago 00:00: 1.518.987 en vivo vs 9.578.964 en la descarga fresca). `generacion`, `precio_bolsa`, `volumen_embalses` y `aportes_hidricos` estaban bien esos mismos días — el defecto es exclusivo de `demanda`. Nadie lo había detectado porque no había con qué compararlo hasta esta prueba.

**Arreglo, con respaldo previo** (`data/processed/resultados/backups_pre_fix_demanda_04_05ago2026/`):
1. `demanda_real_2026.csv`: 48 horas reemplazadas por los valores de la descarga fresca.
2. `dataset_maestro_2026.csv`: columna `demanda` de esas 48 horas.
3. `dataset_features_2026.csv`: `demanda` (48h) + `demanda_lag24h` y `demanda_media_24h` recalculadas con las fórmulas exactas de `05_features_compartidas_juan.ipynb` (`shift(24)`, `rolling(24).mean()`) — afectan solo las 24 horas del 5-ago, verificado antes de escribir con un cálculo en seco. `demanda_lag48h`/`demanda_lag72h` no cambian (sus ventanas caen fuera del rango del archivo).

**Alcance del reentrenamiento, verificado antes de tocar nada**: Persistencia usa `precio_lag24h` (sin demanda, sin cambios). ARX+GARCH — su lista de regresoras no incluye ninguna variable de demanda (confirmado en `o6_comun.py`, sin cambios). XGBoost sí usa `demanda_lag24h`/`demanda_media_24h` (no están en `EXCLUIR`). N-BEATSx/N-HiTS usan `demanda_lag24h` como exógena histórica. GARCH-ged usa `demanda_media_24h` en `EXOG_CORTE`. Se regeneraron exactamente estos 3 (script: `scripts_experimento/reentrenar_tras_fix_demanda.py`), con los mismos hiperparámetros que ya producían los números vigentes del proyecto — único input que cambia es el archivo de features ya corregido.

**Error de proceso encontrado y corregido en el camino**: la primera corrida del reentrenamiento tenía el corte de entrenamiento de las redes neuronales una hora antes de donde debía (usaba el mismo corte que XGBoost, `< "2026-01-01"`, cuando N-BEATSx/N-HiTS pronostican en bloques de 24h con el protocolo día-adelante documentado el 22-24 de septiembre: ventanas 01:00→00:00, corte a las 00:00). El script llevaba una verificación que compara el rango de salida contra el rango que ya existe en vivo, y **abortó solo, sin guardar nada**, señalando el desfase exacto de 1 hora. Corregido con un corte de entrenamiento propio para la red (`NEURAL_TRAIN_FIN`, train incluye hasta las 00:00 del 1-ene inclusive) y reverificado en seco antes de repetir la corrida completa.

**Resultado por modelo (Origen 6, 2026):**

| modelo | antes | después |
|---|---|---|
| XGBoost | 61.39 | 61.39 (sin cambio) |
| **N-BEATSx** | 46.82 | **45.15** |
| N-HiTS | 46.44 | 46.42 |
| GARCH-ged | ~45.3-45.4 | 45.39 |

N-BEATSx mejora notablemente (−1.67 de MAE) pese a que solo 48 de 5.184 horas de prueba (0.9%) tenían algún dato corregido — coherente con que usa `demanda_lag24h` como contexto de 7 días (`input_size=168`): el dato malo contaminaba la ventana de contexto de varios pronósticos vecinos, no solo los 2 días con el valor directamente mal.

**Ensamble completo (6 votantes, 10 particiones, mismo estándar del proyecto):**

| métrica | antes | después | delta |
|---|---|---|---|
| MAE | 41.340 ± 0.125 | **41.096 ± 0.077** | −0.244, mejor en 10/10 |
| MAPE | 10.760% ± 0.035 | **10.487% ± 0.021** | −0.273, mejor en 10/10 |
| sMAPE | 10.080% ± 0.032 | **9.841% ± 0.013** | −0.239, mejor en 10/10 |

**Nota de reproducibilidad**: la cifra "antes" del GARCH-ged tuvo que reconstruirse (0.11 de MAE de diferencia frente al valor que el propio script de reentrenamiento imprimió al leer el archivo en vivo justo antes de sobrescribirlo: 45.29 vs 45.40 reconstruido) porque **no se respaldó `pronostico_GARCH-ged_24h_2026.csv` antes del primer parche** — lección de proceso: el respaldo previo a modificar debe cubrir *todos* los archivos que un script vaya a sobrescribir, no solo los que se editan a mano. Se reconstruyó corriendo la misma función sobre los datos de `dataset_features_2026.csv` sí respaldados (que aún tienen el bug), y la comparación final usa esa reconstrucción de forma consistente en ambos lados. La discrepancia menor no cambia la dirección ni la magnitud aproximada del hallazgo.

**Archivos modificados en vivo** (todos con respaldo previo en `backups_pre_fix_demanda_04_05ago2026/`, y recuperables además desde el commit `dbc1e8f`): `data/demanda_real_2026.csv`, `data/processed/dataset_maestro_2026.csv`, `data/processed/dataset_features_2026.csv`, `data/processed/resultados/walkforward_predicciones_crudas.csv`, `data/processed/resultados/pronostico_GARCH-ged_24h_2026.csv`, `notebooks/01_prueba_conexion_juan.ipynb` (solo numeración).

**Pendiente**: las cifras de cabecera citadas en el informe Word y en entradas anteriores de esta bitácora (41.29/10.74%) quedan ligeramente desactualizadas por este arreglo (ahora 41.10/10.49%). Actualizar el informe cuando se decida (el usuario pidió esperar antes).


### 2026-09-24 — Aclaración del usuario sobre el protocolo: el pronóstico es móvil (24 h desde "ahora"), y el precio de la última hora está disponible de inmediato

Tras el hallazgo de protocolo del 2026-09-23, el usuario aclaró dos hechos de operación que cambian la lectura:

1. **El horizonte de 24 h se cuenta desde la hora real en que se pronostica**, no como un "día siguiente" calendario.
2. **El precio de la última hora está disponible de inmediato.**

Con eso, usar el último precio conocido —aunque pertenezca al mismo día de despacho que los objetivos— es **legítimo**. El corte a las 23:00 del día previo no es "el único protocolo honesto": es el caso de pronosticar justo a esa hora.

**Lo que sigue en pie** es que la evaluación del proyecto siempre lanza el pronóstico desde las 00:00, y esa es **la hora más favorable del día** para hacerlo, porque 23 de los 24 objetivos comparten el día de ofertas que ya se empezó a observar:

| hora de lanzamiento | objetivos del día de ofertas ya observado | MAE del ensamble |
|---|---|---|
| **00:00** (lo que mide el proyecto) | 23 de 24 | **41.29** (mejor caso) |
| 23:00 | 0 de 24 | **51.82** (peor caso) |
| otras horas | entre 1 y 22 | entre ambos (no medido) |

**Cómo debe leerse ahora:**
- 41.29 / 10.74% no está "inflado": es el desempeño de un pronóstico de 24 h **lanzado a medianoche**.
- Para un pronóstico móvil lanzado a cualquier hora, la cifra representativa es el **promedio sobre las 24 horas de lanzamiento**, que queda entre 41.29 y 51.82. No está medido; se puede estimar con el LEAR lanzando desde cada hora.
- La comparación con papers **day-ahead** de la literatura (Figura 5) sigue necesitando la salvedad: ellos pronostican el día siguiente completo sin observar ninguna hora de él.
- El motor de decisión debe saber a qué hora se lanza cada pronóstico, porque la exactitud esperada depende de ella.

Los PDF de `docs/papers/` **no se suben al repositorio** (es público y varios no son de acceso abierto); quedan en el equipo local y `REFERENCIAS.md` tiene cita y enlace de cada uno.

### 2026-09-24 — El error del pronóstico móvil según la hora en que se lanza: promedio de las 24 horas

`scripts_experimento/pronostico_movil_24_horas_lanzamiento.py`. Cierra el pendiente de la aclaración del usuario (el pronóstico de 24h se cuenta desde la hora real en que se emite). Mismo LEAR, mismo periodo 2026, cambiando solo la hora del corte:

| lanzado a las | objetivos del día de ofertas ya abierto | MAE (LEAR) | MAPE |
|---|---|---|---|
| 00:00 | 23 | 53.27 | 16.54% |
| 04:00 | 19 | 53.83 | 16.06% |
| **08:00** (mejor) | 15 | **52.54** | 14.34% |
| 12:00 | 11 | 58.41 | 16.32% |
| 16:00 | 7 | 56.14 | 16.53% |
| 20:00 | 3 | 58.04 | 17.26% |
| **23:00** (peor) | 0 | **61.64** | 19.32% |
| **promedio de las 24 horas** | — | **56.37** | — |

El error sube a medida que el lanzamiento se acerca a la medianoche: quedan menos objetivos del día de ofertas que ya empezó a observarse. La madrugada y la mañana (00:00-08:00) son las mejores horas para lanzar; 21:00-23:00 las peores.

**Estimación para el ensamble** (no medida: exigiría reentrenar las redes 24 veces). Sus dos extremos sí están medidos, 41.29 a las 00:00 y 51.82 a las 23:00. Interpolando con la posición relativa del promedio del LEAR (37% del camino entre sus extremos), el ensamble lanzado a cualquier hora rondaría **MAE ≈ 45.2**. El supuesto es que la curva del ensamble tiene la misma forma que la del LEAR.

**Frase defendible para el informe:** *lanzado a medianoche, MAE 41.3; lanzado a cualquier hora del día, del orden de 45; en el peor caso (23:00), 51.8 — siempre por debajo de la persistencia (56.4).*

### 2026-09-24 — MCC (criterio de máxima correntropía) como pérdida robusta: mejora mucho los modelos sueltos, empeora las rampas, y no aporta al ensamble

`scripts_experimento/mcc_perdida_robusta_24h.py`. Único pendiente genuinamente nunca probado de las ideas de los papers (Nunes Jr., Ferreira & Pinho 2024). MCC maximiza `Σ exp(−e²/2σ²)`: pérdida **redescendente**, un error muy grande casi no pesa. Se optimiza por la equivalencia *half-quadratic*: mínimos cuadrados reponderados con pesos `exp(−e²/2σ²)`, 4 iteraciones desde un ajuste con pérdida absoluta. σ elegido en validación temporal dentro del entrenamiento (último 20% de cortes pre-2026): 3.0× la escala robusta para GBM, 1.0× para LASSO. Protocolo de 24h vigente (corte 00:00), diseño del LEAR.

**Modelos sueltos (2026):**

| variante | MAE | MAPE | rampa ≥50 | meseta | pico |
|---|---|---|---|---|---|
| GBM pérdida cuadrática | 56.05 | 15.42% | 113.60 | 44.47 | 82.59 |
| GBM pérdida absoluta | 51.62 | 13.57% | 112.29 | 39.94 | 76.70 |
| **GBM MCC** | **49.23** | **13.14%** | 116.48 | 36.16 | 75.11 |
| LEAR (LASSO) | 52.78 | 16.07% | **107.99** | 42.22 | 74.58 |
| **LEAR MCC** | **47.09** | **12.78%** | 113.69 | 34.38 | 69.73 |

MCC mejora mucho: LEAR −10.8% de MAE (DM p<0.0001), al nivel de N-BEATSx (46.83), el mejor modelo individual del proyecto; GBM −4.6% frente a pérdida absoluta (p=0.002). **Pero empeora en las rampas**: LEAR 107.99 → 113.69 (p=0.0027), GBM 112.29 → 116.48 (p=0.08). Es exactamente lo que se anticipó: la pérdida le quita peso a los errores grandes, que son los de las rampas, y el modelo gana en las horas tranquilas a costa de los picos. **Va en contra del objetivo declarado del usuario.**

**Como séptimo votante (10 particiones):**

| | MAE | MAPE | sMAPE | mejor en (MAE/MAPE/sMAPE) |
|---|---|---|---|---|
| ensamble actual | 41.287 | 10.744 | 10.065 | — |
| + GBM MCC | 41.326 | 10.716 | 10.039 | 2/9/10 |
| + LEAR MCC | 41.337 | 10.760 | 10.078 | 0/0/0 |
| + GBM absoluta (control) | 41.395 | 10.732 | 10.063 | 0/8/7 |

Ninguno pasa el estándar 10/10. La pequeña ganancia de MAPE del GBM MCC la reproduce en buena parte el control sin MCC (8/10): viene de sumar un árbol más, no de la pérdida. **No se adopta.** Sexta confirmación del patrón del proyecto: un modelo individual claramente mejor (aquí −10.8%) no le aporta a un ensamble que ya es fuerte.

Queda como dato útil para la sustentación: si alguna vez se necesita **un único modelo lineal** (por interpretabilidad o costo), el LEAR entrenado con MCC es la mejor opción disponible (MAE 47.09, MAPE 12.78%).

### 2026-09-24 — Limpieza de carpetas del proyecto y búsqueda exhaustiva de pruebas pendientes

Revisión carpeta por carpeta a petición del usuario. Borrado: `.claude/` (vacía), `catboost_info/` y `lightning_logs/` (artefactos de entrenamiento regenerables, ya en `.gitignore`), `logs_catboost_dir.txt` (log suelto de una corrida vieja). `.streamlit/config.toml` se confirmó en uso (tema del dashboard de Rafael) y se deja.

**`data/processed/resultados/`** (199 archivos, 91MB): clasificación sistemática automatizada de los 199 archivos por referencias cruzadas. Único huérfano real: `arxgarch_predicciones_festivos.csv` (256KB, salida intermedia de una reejecución puntual del notebook 08 el 10-sep, sin ningún generador ni lector actual) — **borrado**. Todo lo demás tiene generador o lector vigente, o es historial documentado a propósito (`*_3meses_backup.csv`, `fuzzy_oni_preview.csv`).

**`data/` raíz — 34 CSV crudos de XM descargados directamente ahí (no en `raw/`/`external/`)**: clasificados con el mismo método. 10 son el núcleo del pipeline (precio, demanda, generación, embalses, aportes — usados por los notebooks 01-05). El resto se separa en tres grupos:

1. **Evidencia de decisiones ya tomadas** (se quedan, documentan por qué no se usan): `max_precio_oferta` (fuga, prohibida), `disponibilidad_por_tipo`/`generacion_por_tipo` (restringidas por Anexo 1), `firma_despacho_hidraulico` (idea descartada).
2. **Ya tamizadas y probadas en el modelo real** (12 variables, `tamizar_variables_nuevas.py` + `probar_variables_nuevas.py`): 9/12 redundantes por correlación parcial, 1 fuga (`max_precio_oferta`), y la única candidata real (`costo_marginal_despacho`, parcial 0.178) se probó en el modelo y **empeoró** el ensamble — tercera confirmación del patrón "señal real que no sobrevive al ensamble".
3. **Sin ningún respaldo ni uso documentado, borradas**: `generacion_fuera_merito`, `generacion_programada_despacho`, `restricciones_sin_alivios` (~6.5MB).
4. **Con respaldo de papers, nunca probadas** — ver siguiente sección.

**Actualización (mismo día): borrados también los 13 CSV crudos descartados por los tamices** — los 12 de `tamizar_variables_nuevas.py` (`max_precio_oferta`, `costo_marginal_despacho`, `generacion_ideal`, `demanda_comercial`, `restricciones_aliviadas`, `perdidas_energia`, `importaciones_energia`, `exportaciones_energia`, `precio_escasez`, `precio_escasez_marginal`, `aportes_media_historica`, `compras_arranque_parada`) y `porcentaje_volumen_util` (redundante en el tamiz de Nivel 2, abajo). Sus resultados siguen documentados en `tamiz_variables_nuevas.csv` y `tamiz_nivel2_papers.csv`. **Consecuencia**: los scripts de esos experimentos cerrados (`tamizar_variables_nuevas.py`, `probar_variables_nuevas.py`, `atacar_el_nino.py`, `margen_reserva_pico_24h.py`, `ensamble_24h_con_oferta.py`, `tamizar_nivel2_papers.py`) ya no corren sin restaurarlos. **Segunda tanda, a pedido del usuario**: borrados también `precio_contratos_regulado`, `precio_contratos_no_regulado` y `aportes_caudal` (probados hoy, no sobreviven al ensamble), `disponibilidad_por_tipo` y `generacion_por_tipo` (desglose por tipo de recurso, prohibido por el Anexo 1) y `firma_despacho_hidraulico` (idea descartada el 10-sep). En `data/` quedan solo los 10 CSV del núcleo del pipeline y los 2 catálogos de XM que usan los scripts de descarga. **Para recuperar cualquiera**: `git checkout dbc1e8f -- data/<archivo>.csv`, o volver a bajarlos con `descargar_variables_nuevas.py` / `descargar_equivalentes_papers.py`.

**`docs/`**: borrado `informe_comparativo_modelos.md` (resultados del 2026-09-10, con N-BEATSx como mejor modelo a MAE 56.11; superado por `Informe_Pronostico_Precio_Bolsa_XM.docx` y esta bitácora). Las menciones a ese archivo en entradas anteriores quedan como historia; se recupera con `git checkout dbc1e8f -- docs/informe_comparativo_modelos.md`. `motor_decision_guia.md` es de Rafael y no se tocó. El informe en Word queda **pendiente de actualizar** con lo de esta semana (hora de lanzamiento del pronóstico, historia del precio como entrada principal, pruebas negativas).

**`scripts_experimento/`** (150 scripts, 2.8 MB): no se borra ninguno. 22 son módulos que importan otros scripts (`o6_comun.py` lo usan 62, `stacking_24h_v2.py` 46), 107 están citados por nombre en esta bitácora, y los 21 restantes están descritos en prosa o generan archivos vigentes de `resultados/`. Borrar scripts ahorraría casi nada y rompería la reproducibilidad de resultados documentados. Solo se borró `__pycache__/` (caché regenerable).

**Dos resultados que solo estaban en mensajes de commit, rescatados aquí:**
- **Causalidad de Granger** (`granger_hidrologia.py`, commit `c5f09a8`, 2026-09-10). Metodología de Emre (2025, IJEPES), aplicada sobre la serie **diaria** para no inflar la significancia con la autocorrelación horaria. **`aportes_hidricos` y `volumen_embalses` causan el precio de bolsa en sentido de Granger (p<0.0001 en todos los rezagos de 1 a 7 días).** Es evidencia formal —no solo correlación— de una relación causal entre hidrología y precio en el mercado colombiano. Contrasta con Emre en Turquía, que no la encontró por razones estructurales de su mercado (YEKDEM). Útil para la sustentación junto con el hallazgo de que la hidrología predice cuándo desconfiar del modelo aunque no baje su error.
- **Corrección de sesgo/escala por régimen** (`correccion_regimen.py`, commit `49b65dc`, 2026-09-10): mejora pequeña y no significativa (p=0.31). No se adoptó.

**Raíz del proyecto**: borrados 24 `logs_*.txt` sueltos de sesiones anteriores y la carpeta `logs/` de esta semana (salida de consola; los resultados de esos experimentos quedan en `resultados/` y en esta bitácora), y `desktop.ini` (metadatos de Windows que se había subido a git por error; ahora está en `.gitignore`). **`notebooks/`**: los 13 se quedan (3 son de Rafael; el 01 descarga los datos crudos y el 04 Prophet está comprometido en el Anexo 1); solo se borró `notebooks/lightning_logs/` (16 MB de artefactos). **`src/`** solo contiene `motor_decision.py` de Rafael.

**Arreglo de reproducibilidad en `requirements.txt`**: faltaban `catboost` (lo importan 17 archivos, incluidos votantes del ensamble) y `python-docx` (generador del informe en Word). Agregados con las versiones del entorno: `catboost==1.2.10`, `python-docx==1.2.0`. Sin esto, una instalación desde cero no podía correr esos scripts.

#### Nivel 1 — tres ideas de los papers desbloqueados, las más baratas (`scripts_experimento/nivel1_ideas_papers.py`)

Las tres fallan:

**(1) Compuerta por régimen hidrológico** (Albani et al. 2025: apagar el componente hidrológico cuando se rompe la correlación ENA-precio). Se repesó el ensamble por celda de régimen (embalse alto/bajo × ONI alto/bajo, con umbral **móvil causal de 90 días**, corrigiendo de paso el sesgo no-causal de `estratificacion_regimen_24h.csv` que usa la mediana de todo 2026) en vez de por franja horaria.

| esquema | MAE | 10 particiones |
|---|---|---|
| actual (franja horaria) | 41.19 | — |
| por celda de régimen (fija) | 43.35 | — |
| por celda de régimen (móvil 90d) | 43.25 | 0/10, p=0.0000 |
| franja horaria × régimen | 41.43 | no significativo (p=0.25) |

Falla porque las celdas quedan muy desbalanceadas: con umbral móvil, `embAlto_oniBajo` tiene solo 30 observaciones en todo 2026 — no hay suficiente variación de régimen dentro del año de prueba para que repesar por celda tenga sentido estadístico. **No se adopta.**

**(2) RLS-dayahead como séptimo votante.** Documentado desde antes como "candidato razonable... no se hizo todavía". Resultado: neutro (41.19→41.18, p=0.56, 1/10 particiones). Reemplazando a GARCH-ged: peor (42.57, p=0.0003). **GARCH-ged sigue siendo mejor 6º votante. No se adopta.**

**(3) Poda por multicolinealidad** (Dias et al. 2024, |Pearson|>0.8, quedarse con la de mayor información mutua). Sobre las 55 variables del diseño del LEAR (paso 12 representativo), poda 28 y deja 27. Resultado: MAE 53.37 → **64.19**, empeora drásticamente (DM p=0.0000). **Explicación**: LASSO ya maneja la colinealidad con su propia regularización — podar variables correlacionadas antes de LASSO destruye información que el propio regularizador ya sabía repartir bien entre variables redundantes. Podar por correlación es redundante (en el mejor caso) o dañino (en este caso) cuando el modelo ya es LASSO. **No se adopta.**

#### Nivel 2 — las 4 variables con respaldo de papers, nunca probadas (`scripts_experimento/tamizar_nivel2_papers.py`)

Mismo protocolo en dos fases que `tamizar_variables_nuevas.py` + `probar_variables_nuevas.py`: tamiz de correlación parcial primero, solo lo que pasa llega al modelo real.

| variable | de qué paper | corr. parcial | veredicto del tamiz |
|---|---|---|---|
| `porcentaje_volumen_util` (EAR%) | Dias, Lira & Freire 2024, Brasil — #1 en información mutua en su Tabla 1 | −0.029 | **REDUNDANTE** |
| `precio_contratos_regulado` | Kapoor & Wichitaksorn 2023, NZ — equivalente a forward prices | 0.066 | candidata |
| `precio_contratos_no_regulado` | mismo paper | 0.081 | candidata |
| `aportes_caudal` | Nunes Jr., Ferreira & Pinho 2024, Brasil — cadena caudal→precio | −0.088 | candidata |

El EAR%, pese a ser la variable #1 del paper brasileño, no aporta nada aquí que el precio rezagado 24h no tenga ya — probablemente porque el proyecto ya usa `volumen_embalses` en absoluto y sus derivadas (anomalía vs media 30d, delta 1d).

Las tres candidatas (contratos regulado/no regulado + caudal), con nivel + anomalía vs media 30d + delta 1d cada una, metidas al LEAR:

| | MAE | resultado |
|---|---|---|
| LEAR actual | 53.37 | — |
| **LEAR + las 3 variables** | **52.24** | mejora real, DM p=0.0074 |
| Ensamble actual (6 votantes) | 41.19 | — |
| + ese LEAR como 7º votante | 41.20 | sin diferencia, DM p=0.90 |

**Mismo patrón que `costo_marginal_despacho` y la variable H/T**: la señal es real y mejora un modelo individual con margen estadísticamente significativo, pero no sobrevive dentro del ensamble — la información ya está implícita en los otros seis votantes. Cuarta/quinta confirmación de este hallazgo metodológico del proyecto: **correlación parcial real (y hasta mejora individual significativa) no implica que un ensamble ya fuerte tenga algo que ganar**.

#### Corrección: 4 de estas 7 pruebas eran repeticiones, no pruebas nuevas

La "búsqueda exhaustiva" de pendientes se hizo mal: se tomó la lista "cuatro ideas... que NO hemos probado" de la entrada del 2026-09-17 sin cruzarla con las entradas **posteriores**, y varias ya se habían ejecutado. Contraste completo:

| idea | ¿ya probada antes? | resultado de entonces | resultado de hoy |
|---|---|---|---|
| Compuerta por régimen (Albani) | **sí**, `qra_por_regimen.py` (09-17), incluida la versión relativa 90d | 42.51 → 44.40, no sirve | igual: no sirve |
| RLS-dayahead como votante | **sí**, entre los "9 votantes extra" (09-17) | sin diferencia significativa | igual: neutro |
| Poda por multicolinealidad (Dias) | **sí**, en 72h (09-17) | catastrófica literal, neutra corregida | igual: empeora |
| Precios de contratos (forward, NZ) | **sí**, en la media del LE-GARCH-t (09-17) | falso positivo detectado | igual: no sobrevive al ensamble |
| LASSO agresivo sobre GARCH (NZ) | **sí, y se ADOPTÓ** (09-17): es el LE-GARCH-t → GARCH-ged, 6º votante actual | MAPE 11.27% → 10.89% | — |
| EAR% (`porcentaje_volumen_util`) | **no** — nuevo hoy | — | redundante en el tamiz |
| `aportes_caudal` | **no** — nuevo hoy | — | pasa el tamiz, no sobrevive al ensamble |
| MCC como pérdida (Nunes Jr.) | **no** | — | ver entrada siguiente |

Las cuatro réplicas dieron el mismo resultado que la primera vez, así que al menos sirven como confirmación independiente. Lección de proceso: una lista de "pendientes" en la bitácora queda obsoleta en cuanto se ejecuta algo de ella, y antes de actuar hay que cruzarla con todo lo que vino después.

**Pendientes reales tras el cruce completo**: (1) MCC como pérdida robusta, nunca probado; (2) medir el error promedio de un pronóstico móvil lanzado desde cada una de las 24 horas (aclaración del usuario del 2026-09-24), nunca medido; (3) TFT, abortado por falta de GPU —la idea no falló, simplemente no es viable en esta máquina—; (4) recalibración 2026 v2, dejada de lado con un resultado preliminar negativo.

### 2026-09-23 — La cifra honesta del ensamble de 24h, el walk-forward completo de NOT, y el oráculo de demanda

#### 1. El ensamble de 24h con el corte day-ahead honesto (`ensamble_protocolo_honesto.py`)

Se reentrenaron con el corte a las 23:00 del día previo los tres votantes que usaban el precio del mismo día de despacho (mismos hiperparámetros). Los otros tres ya eran honestos. Nada del proyecto se sobrescribió: todo con sufijo `_corte23`.

| votante | MAE actual (corte 00:00) | MAE honesto (corte 23:00) | hora 1 actual | hora 1 honesto |
|---|---|---|---|---|
| Persistencia | 56.41 | 56.41 | 47.15 | 47.15 |
| XGBoost | 61.39 | 61.39 | 51.11 | 51.11 |
| ARX+GARCH | 55.84 | 55.84 | 46.03 | 46.03 |
| N-BEATSx | 46.83 | **57.38** | 10.31 | 46.55 |
| N-HiTS | 46.45 | **57.39** | 10.37 | 47.58 |
| GARCH-ged | 45.42 | **57.36** | 7.48 | 45.99 |

El N-BEATSx honesto da **57.38 — exactamente la cifra** de la "corrección" que se revirtió. Aquella corrección era el protocolo correcto.

**El ensamble (6 votantes, combinador sMAPE, 10 particiones):**

| | MAE | MAPE | sMAPE | MASE vs persistencia 24h |
|---|---|---|---|---|
| protocolo actual (corte 00:00) | 41.29 ± 0.10 | 10.74% | 10.06% | 0.732 |
| **day-ahead honesto (corte 23:00)** | **51.82 ± 0.12** | **14.22%** | **13.27%** | **0.919** |
| persistencia | 56.41 | 15.81% | 14.77% | 1.000 |

**Lectura:**
- La cifra honesta es **~25% peor en MAE** que la reportada. La diferencia se concentra en las horas 1-17 (hora 1: 7.60 → 43.83).
- **El ensamble honesto sigue ganándole a la persistencia** (MAE −8%, MAPE −1.6 puntos, MASE 0.919). El proyecto no se cae; sus cifras de cabecera estaban infladas.
- **Las horas pico de la tarde casi no cambian**: 18h 90.89 → 92.42, 19h 103.87 → 106.76, 20h 79.36 → 79.65. **El problema de los puntos de cambio intradía de la tarde es real e independiente del protocolo**: ahí el corte a las 00:00 nunca ayudó.
- En la hora 0 el honesto es *mejor* (47.51 → 44.68), coherente con que en el protocolo actual la hora 0 era el único objetivo de un día nuevo.

#### 2. Walk-forward completo de NOT (`not_walkforward_origenes.py`)

| comparación | MAE mejor | MAPE mejor | significativo (p<0.05) |
|---|---|---|---|
| ARX NOT vs ARX Win | 4/6 | 4/6 | 5/6 |
| ARX NOTH vs ARX WinH | 4/6 | 5/6 | 4/6 |
| **LEAR NOT vs LEAR histórico completo** | **3/6** | 4/6 | 6/6 (en ambos sentidos) |

Por origen (MAE): O1 3/3 mejora · O2 2/3 · O3 1/3 · **O4 (El Niño 2023-24) 0/3** · O5 2/3 · O6 (2026) 3/3.

**NOT depende del régimen.** Ayuda cuando el régimen vigente se parece a tramos del histórico y falla cuando llega algo sin precedente cercano (El Niño 2023-24). Contra la configuración real del proyecto (LEAR con todo el histórico) es una moneda al aire: 3/6. **La mejora del ensamble medida en 2026 no justifica adoptarlo como votante oficial.** Para el paper de Nasiadka/Nitka/Weron, la respuesta a su pregunta abierta queda matizada: sobre el LEAR funciona en unos regímenes y no en otros.

#### 3. Oráculo de demanda (`oraculo_demanda_rampas.py`)

Hipótesis: si un punto de cambio intradía es un cambio de planta marginal, debería dispararlo la demanda cruzando umbrales de capacidad. Si fuera así, el pronóstico oficial de demanda de XM sería el dato a conseguir. Se mide con la **demanda real** de cada hora objetivo (oráculo, cota superior), en LEAR y en gradient boosting (que sí capta umbrales):

| modelo / corte | MAE rampa base | + demanda de ayer | **+ demanda real (oráculo)** |
|---|---|---|---|
| LEAR, 00:00 | 108.19 | 108.04 | 104.65 |
| LEAR, 23:00 | 111.94 | 111.45 | 108.46 |
| GBM, 00:00 | 111.84 | 113.51 | 107.51 |
| GBM, 23:00 | 117.81 | 117.95 | 111.62 |

**Ni con la demanda real el error de rampa baja más de 3-5%**, y el MAE global no mejora. La demanda no es la pieza que falta. Lo que mueve las rampas son las ofertas y la disponibilidad por planta —la información que el Alcance del Anexo 1 restringe—. Es la evidencia más directa hasta ahora de que el error en los puntos de cambio es un **límite de información**, y no conviene gastar esfuerzo en conseguir el pronóstico de demanda de XM para este fin.


### 2026-09-23 — ⚠️ HALLAZGO DE PROTOCOLO: el corte de las 00:00 cae DENTRO del día de despacho que se pronostica. Con un corte day-ahead honesto, el LEAR pierde contra la persistencia

**No se cambió nada del proyecto.** Esta entrada solo documenta y mide. La decisión de qué protocolo reportar es del usuario.

#### Cómo apareció

Buscando atacar los puntos de cambio intradía por la vía de las mesetas (`diagnostico_mesetas.py`) salió que el precio es **escalonado**: cada día tiene en promedio **solo 7.2 niveles distintos** en 24 horas, porque las plantas ofertan un precio por día y el precio horario lo marca la planta marginal. En tramos planos el precio real se mueve **0.00** por hora; el ensamble, 6.62.

Siguiendo eso, la igualdad **exacta** entre el precio de una hora y el de la anterior, sobre todo el histórico:

```
00:  0%  01: 73%  02: 71%  03: 77%  04: 72%  05: 57%  06: 68%  07: 48%  08: 50%  09: 62%  10: 58%  11: 57%
12: 75%  13: 74%  14: 64%  15: 72%  16: 64%  17: 58%  18: 29%  19: 56%  20: 58%  21: 48%  22: 35%  23: 34%
```

**La única transición sin ninguna repetición exacta es 23:00 → 00:00.** El día de ofertas (día de despacho) va de 00:00 a 23:00.

#### Por qué es un problema

El protocolo day-ahead del proyecto usa ventanas 01:00 → 00:00 del día siguiente con **corte a las 00:00**. Entonces:

- el corte (00:00 del día D) es la **primera hora del día de despacho D**;
- **23 de los 24 objetivos** (01:00 a 23:00 de D) son del **mismo día de despacho** que el corte;
- solo el último objetivo (00:00 de D+1) es de un día nuevo.

No es fuga del objetivo —ningún valor objetivo entra como variable—, pero el modelo conoce un precio formado **con las mismas ofertas** que sus objetivos. Un pronóstico day-ahead real no lo tendría: el precio de las 00:00 de D se conoce, como pronto, al terminar esa hora.

#### Medido: el mismo LEAR con dos cortes

`scripts_experimento/protocolo_dayahead_honesto.py`. Mismas variables, misma regularización, mismo periodo (2026), mismos 5.184 instantes objetivo. Solo cambia el corte.

| | MAE | MAPE |
|---|---|---|
| **A** — corte 00:00 de D (actual) | 53.27 | 16.53% |
| **B** — corte 23:00 de D−1 (day-ahead honesto respecto del día de ofertas) | **61.65** | **19.32%** |
| persistencia (t−24h) | 56.40 | 15.81% |

**Con el corte honesto, el LEAR pierde contra la persistencia** en 22 de las 24 horas. La ventaja del protocolo actual es enorme en la madrugada y se extiende a todo el día:

| horas | B − A (MAE) |
|---|---|
| 1-5 | **+16.60** de media (hora 1: 20.51 → 47.41) |
| 6-23 | +6.75 de media |
| 0 (el único objetivo de un día nuevo en A) | −3.33 (B es mejor) |

Conocer el primer precio del día revela el **nivel de ofertas** de ese día, y eso informa las 23 horas siguientes.

#### Lo que esto explica retroactivamente

1. **Las horas 1-5 tienen MAE bajísimo** (7.6 a 20.9 en el ensamble): heredan el precio del corte, que el 73% de las veces es *exactamente* el de la 01:00.
2. **La hora 0 es la peor hora no-pico y la única donde el ensamble no le gana a la persistencia** (−1.3%): es el único objetivo de un día de despacho nuevo, o sea el único pronóstico day-ahead de verdad.
3. **El "salto de medianoche"** que se documentó con el marco de Ziel & Weron es exactamente esta frontera. El diagnóstico era correcto; lo que no se vio es que el corte quedaba del lado equivocado.
4. **La "corrección" de ventana del N-BEATSx** (`corregir_nbeatsx_origen6.py`), que lo empeoró de 46.82 a 57.38 y se revirtió porque "el proyecto no debía empeorar", **muy probablemente era el protocolo honesto**. Lo que se perdió al aplicarla era esta información del mismo día.

#### Qué sigue válido y qué no

- **Las comparaciones internas siguen siendo válidas**: todo lo que se comparó dentro del proyecto (NOT vs ventana fija, votantes, combinadores, Diebold-Mariano) usó el mismo protocolo en ambos lados.
- **Las cifras absolutas de cabecera de 24h** (MAE 41.29, MAPE 10.74%, MASE 0.914) están medidas con el protocolo A, y por tanto **son optimistas** respecto de un day-ahead real. Cuánto, en el ensamble completo, no está medido: requiere reentrenar los seis votantes (incluidas N-BEATSx y N-HiTS) con el corte a las 23:00.
- **La comparación con la literatura** (Figura 5) queda afectada en la misma dirección.
- Incluso el protocolo B es algo optimista para una operación real: las ofertas del día D se entregan en la mañana de D−1, y a esa hora no se conocen los precios de la tarde de D−1. Un protocolo estrictamente operativo tendría el corte todavía más atrás.
- **El motor de decisión de Rafael** consume estos pronósticos; si su uso es preparar ofertas para el día D, la información disponible es la del protocolo B o anterior.

#### Pendiente, a decidir por el usuario

1. Reentrenar el ensamble completo con el corte a las 23:00 del día previo y medir la cifra honesta.
2. Decidir qué protocolo se reporta en el informe y en la sustentación, y cómo se explica.
3. Avisar a Rafael antes de que integre los pronósticos al motor de decisión.

---

### 2026-09-23 — Tres intentos más sobre los puntos de cambio intradía: mesetas, precio pegajoso y STBC fiel

#### Mesetas (`diagnostico_mesetas.py`)
Horas donde el precio no se movió (2.632, el 35% del error total, MAE 28.52). No hay días completamente planos (0 de 216); las mesetas viven dentro de días con picos. En tramos de ≥3 h planas el real se mueve 0.00 por hora y el ensamble 6.62. **Oráculo** (saber dónde hay meseta y aplanar ahí): 41.19 → **39.72**. Cota superior modesta.

#### Precio pegajoso con regla de la mediana (`precio_pegajoso_mediana.py`)
Idea con fundamento: la distribución del precio en la hora h es una **mezcla** — con probabilidad π se queda exactamente en el precio del corte (un átomo), si no se mueve. Para MAE el óptimo es la **mediana**, y con π > 0.5 la mediana **es** el átomo. Explica también por qué falló el hurdle de rampas: usaba π × magnitud, que es la **esperanza** (óptima para error cuadrático, no para MAE).

Pero **falla como regla desplegable**:

| regla | MAE | p(MAE) |
|---|---|---|
| ensamble actual | 41.19 | – |
| τ = 0.5 (teoría, sin ajustar) | 41.94 | 0.016 (peor) |
| τ elegido en validación (0.3) | 42.82 | 0.033 (peor) |
| **oráculo** (sabe dónde se queda) | **37.41** | – |

10 particiones: **0/10**. El clasificador no anticipa si la planta marginal cambia: AUC 0.58-0.68 en las primeras horas y **por debajo de 0.5 en las horas 7, 8 y 12** (la relación aprendida antes de 2026 no se transfiere). El oráculo muestra que hay 3.78 de MAE en juego, pero con la información del corte no se alcanza. Y a la luz del hallazgo de protocolo de arriba: la regularidad "el precio se queda en el del corte" es, en buena parte, **la misma información del mismo día de despacho**.

#### STBC implementado fielmente (`stbc_regime_aware_24h.py`) — corrige una prueba anterior mal hecha
Con el paper en mano (Singh 2027, *Expert Systems With Applications* 332, 133584, aportado por el usuario) se vio que la prueba de `limite_informacion_rampas.py` **no era STBC**: aquella entrenaba un modelo de ML para predecir el error desde el estado del sistema. El método real es `corregido = predicho + λ · (media móvil de los últimos k=10 errores)`, sin variables ni modelo aprendido. Pregunta otra cosa: si el error está **autocorrelacionado**.

Resultado fiel: autocorrelación del error con su media móvil **+0.127**; la búsqueda en validación (ene-mar 2026) elige **λ = 0**; los λ del paper (0.2/0.5/0.8 por régimen GMM) **empeoran**: 58.93 → 60.42 en prueba (p=0.029). La corrección correlaciona **+0.377** con la dirección de la persistencia, que ya es un votante del ensamble. **No aporta información nueva.**

La conclusión de `limite_informacion_rampas.py` sobre su propia pregunta (el error no es predecible desde el estado del sistema, R² ≤ 0) sigue en pie; lo que cambia es que ya no se presenta como prueba de STBC.


### 2026-09-23 — ATAQUE DIRECTO A LOS PUNTOS DE CAMBIO INTRADÍA: cuatro métodos de familias distintas, todos fallan contra el mismo número. Es un límite de información, no de modelado

El usuario corrigió el rumbo: NOT mejoró el modelo pero no en los puntos de cambio, que era el objetivo. Y recordó que pidió literatura **fuera de mercados eléctricos**, sobre cómo corregir errores de modelos de aprendizaje cuando hay puntos de cambio. Esta entrada recoge ese ataque frontal.

#### Paso 0 — El diagnóstico que fija qué puede funcionar

`scripts_experimento/diagnostico_forma_intradia.py`. Antes de elegir método hay que saber por qué falla. Tres causas posibles, medidas por separado sobre el ensamble en 2026:

| | resultado | lectura |
|---|---|---|
| **¿Nivel o forma?** | error total 41.19; con nivel diario perfecto **40.26 (98%)**; con forma perfecta 24.78 | **el error es de FORMA**, modelar la curva es la vía correcta |
| **¿Amplitud?** | desv. intradía real **90.22** vs pronosticada **65.18** (razón **0.723**); rango 327.8 vs 230.1; curva más plana en el **70%** de los días; en las 6 horas peores la razón baja a **0.690** | el modelo dibuja la curva **comprimida** |
| **¿Desfase?** | el mejor desplazamiento es **0 h**, con caída limpia a ambos lados (MAE de forma 40.26 en 0h, 44.92 en +1h, 46.93 en −1h) | **no hay problema de tiempo**; descarta toda una familia |
| **¿Es predecible la forma?** | climatología (hora × día × mes) 65.11 vs ensamble 40.26 | el ensamble le gana a la climatología por **38.2%**: sí extrae información |

#### Intento 1 — Calibración de amplitud (meteorología: pronósticos subdispersos, EMOS)

`calibracion_amplitud_intradia.py`. Transplante desde fuera de energía: en meteorología el aplanamiento de los pronósticos tiene nombre —*subdispersión*— y se corrige con inflación de varianza y EMOS/regresión gaussiana no homogénea. Adaptado aquí como `p_calibrado = nivel_día + k·(p − nivel_día)`, con `k` ajustado **solo con días anteriores**.

| variante | MAE | MAE en rampa | veredicto |
|---|---|---|---|
| sin calibrar | 41.19 | 105.53 | – |
| k global | 41.19 | 105.55 | nulo |
| k por hora, encogido 50% | **41.07** | 105.68 | 10/10 pero trivial |
| **k = igualar varianza (control)** | **50.68** | **113.07** | **desastre, p=0.0000** |

**El resultado clave es el valor que elige `k`: 0.997 global, 0.981 por hora.** Es decir, **la amplitud del ensamble ya está óptimamente calibrada**. Igualar varianzas (k≈1.51) destroza el modelo, exactamente como se predijo: cuando la forma se conoce con correlación 0.78, el encogimiento **es** la respuesta estadísticamente correcta (regresión a la media), no un defecto. La pendiente de `forma_real ~ forma_pronosticada` es 1.095, no 1.38.

**Conclusión:** el aplanamiento es real como descripción pero **no es corregible por post-proceso**. Para mejorar las rampas hace falta mejor información de forma, no reescalar la que hay.

#### Intento 2 — Análisis funcional de Hyndman-Ullah-Shang (Gallón & Barrientos, Colombia)

`fpca_hyndman_24h.py`. El paper colombiano de la biblioteca aplica exactamente el método que hacía falta: tratar el día como **una curva**, descomponerla en componentes principales funcionales (SVD sobre curvas centradas), pronosticar los *scores* y reconstruir. 30 variantes: K ∈ {1,2,3,4,6} × {media ponderada, AR sobre scores, Ridge con exógenas} × {agrupado, por día de semana}.

**Corrección de diseño en el camino:** la primera versión modelaba la curva completa, nivel incluido (que es lo que hace el método original) y daba **MAE 210** — el nivel del precio en Colombia no es estacionario y la media ponderada histórica llega tarde. Rediseñado para modelar **solo forma**, con el nivel pronosticado aparte, que además es lo que dice el diagnóstico propio.

| | MAE de forma | vs ensamble | en rampa | en pico |
|---|---|---|---|---|
| ensamble actual | **40.26** | – | **93.05** | **57.79** |
| climatología | 65.11 | +61.7% | – | – |
| FPCA K=6 AR (mejor) | 44.51 | +10.5% | **93.68** | 62.16 |
| FPCA K=1 AR | 46.77 | +16.2% | **93.56** | 65.46 |
| FPCA con exógenas (mejor) | 52.95 | +31.5% | 105.29 | 74.88 |
| FPCA media ponderada | 60.53 | +50.3% | 111.76 | 91.07 |

Como votante: **0/10, delta +0.000**. El combinador le asigna peso cero.

**Pero deja dos datos importantes:**
1. **En rampa fuerte, el FPCA empata al ensamble: 93.68 contra 93.05.** Y con **un solo componente** (K=1) ya da 93.56. Dos modelos radicalmente distintos —uno es un ensamble de seis familias con 62 variables, el otro es un SVD con un AR sobre 6 números— chocan contra el mismo número.
2. **Las exógenas EMPEORAN los scores** (52.95 contra 44.51 sin ellas). La hidrología, el ONI y los embalses **no informan la forma del día**. Informan el nivel, no la curva.

#### Intento 3 — Short-Term Bias Compensation (bolsa de Varsovia, *Expert Systems with Applications*)

`limite_informacion_rampas.py`. El segundo transplante fuera de energía. El paper lo resume como pasar de *"build a better predictor"* a *"correct a known predictor's drift"*: se entrena un segundo modelo para predecir el **error** del primero y se resta.

**¿Es predecible el error del ensamble con la información del corte?** (TimeSeriesSplit, 62 variables de estado)

| subconjunto | modelo | R² fuera de muestra | correlación |
|---|---|---|---|
| todas las horas | Ridge | **−0.283** | 0.077 |
| todas las horas | GradBoost | **−0.481** | 0.040 |
| rampa fuerte | GradBoost | **−0.011** | 0.307 |
| 6 horas pico | Ridge | **+0.021** | 0.204 |

**R² negativo o nulo en todos los casos.** Y aplicar la corrección empeora sistemáticamente: 46.32 → 58.75 en general, 109.33 → 121.63 en rampa. **No hay deriva que corregir porque no hay información residual.**

#### El número que cierra la línea

**Oráculo del signo** — cota superior de *cualquier* corrección de sesgo imaginable, suponiendo que se conociera si el modelo se queda corto o largo:

| subconjunto | MAE actual | oráculo del signo |
|---|---|---|
| todas las horas | 41.19 | 36.95 |
| **rampa fuerte** | **105.53** | **76.68** |
| 6 horas pico | 62.26 | 54.32 |

Ni siquiera conociendo el signo del error —información inalcanzable— el error de rampa baja de 76.68. Ninguna corrección de sesgo resuelve esto.

#### Y el hallazgo que reencuadra el problema entero

| grupo | n | \|rampa\| media | MAE del ensamble | **razón MAE/rampa** |
|---|---|---|---|---|
| meseta (no se movió) | 2632 | 0.0 | 28.52 | — |
| cambio chico | 1063 | 4.1 | 29.62 | 7.30 |
| cambio medio | 783 | 26.0 | 40.27 | 1.55 |
| rampa 50-150 | 403 | 88.8 | 68.89 | **0.78** |
| rampa 150-300 | 207 | 211.7 | 113.76 | **0.54** |
| rampa >300 | 96 | 543.1 | 251.53 | **0.46** |

**El error crece MÁS DESPACIO que el movimiento que persigue.** En una rampa de 543 el modelo se equivoca en 251 — el 46%. En una hora donde el precio no se movió nada, se equivoca en 28.5.

Es decir: **en términos relativos el modelo maneja las rampas MEJOR que las mesetas.** La afirmación "el error vive en las rampas" es cierta en valor absoluto y engañosa como diagnóstico: las rampas tienen más error porque el objetivo se mueve más, no porque el modelo falle ahí.

#### Conclusión de la línea

Cuatro métodos de familias independientes —modelo de transiciones, calibración de amplitud meteorológica, análisis funcional de Hyndman, y compensación de sesgo financiera— fallan, y **fallan convergiendo al mismo número**. Sumado a que el error del ensamble es ruido respecto de toda la información disponible en el corte (R² ≤ 0), y a que ni el oráculo del signo lo arregla, la conclusión es que **el error en los puntos de cambio intradía está cerca de un límite de información, no de un límite de modelado**.

Hipótesis estructural de por qué, que además conecta con una restricción conocida del proyecto: la forma de la curva diaria en un mercado hidro-dominado la determina el **despacho** — qué unidades entran y en qué orden de mérito. Esa es justamente la información que el Alcance del Anexo 1 impide desagregar (generación y disponibilidad **por tipo de recurso**). Es coherente con que las exógenas hidrológicas informen el nivel pero no la forma.

**Queda abierta una pista distinta que este trabajo destapó sin buscarla:** en las mesetas —2.632 horas, el 35% del error total— el precio no se movió nada y el ensamble aún se equivoca en 28.5. Ese error no es de forma ni de rampa. No está analizado.


### 2026-09-23 — ¿Dónde mejora NOT? La prueba estaba incompleta: la ganancia **no** cae en los puntos de cambio, que era lo que la motivó

`scripts_experimento/not_donde_mejora_24h.py`. Pregunta del usuario tras ver los resultados de NOT: *"¿seguro que esa prueba era solo eso o falta algo más? porque me suena que dijimos que eso podía influir en el error en los puntos de cambio"*. **Tenía razón: faltaba, y es la pieza que decide si el método sirve para lo que se buscaba.**

#### La distinción que no había verificado

Toda esta línea nació de un diagnóstico concreto: el error se concentra en los **puntos de cambio intradía** (rampas de la curva diaria, MAE 105.5 contra 28.5 en mesetas). Pero NOT detecta puntos de cambio en **otra serie**:

| | qué es | para qué |
|---|---|---|
| **Intradía** | quiebres dentro de la curva de un día, hora a hora | **aquí duele el error del proyecto** |
| **Inter-día** | quiebres en la serie de una hora fija a lo largo de los días (cambios de régimen del mercado) | **esto es lo que NOT detecta**, y solo lo usa para elegir con qué historia entrenar |

Son cosas distintas. Que NOT baje el error global no implica que lo baje donde el proyecto falla.

#### Medido: no lo baja. Lo baja en las horas fáciles

Modelo suelto, por tipo de hora (n=5.207):

| | meseta (cambio 0) | cambio chico (<10) | cambio medio (10-50) | **RAMPA FUERTE (≥50)** |
|---|---|---|---|---|
| ARX NOTH vs ARX WinH | **−2.87 (−7.1%)** p=0.0004 | **−4.37 (−9.8%)** p=0.0155 | +1.71 p=0.21 | **+1.77 (PEOR)** p=0.44 |
| LEAR NOT vs LEAR Win(todo) | **−2.67 (−6.5%)** p=0.0175 | **−4.17 (−9.5%)** p=0.0122 | **+4.90 (PEOR)** p=0.0126 | −4.09 p=0.1082 |

**El modelo que entra al ensamble (ARX NOTH) es ligeramente PEOR en las rampas fuertes.** Toda su ganancia está en mesetas y cambios pequeños, y ahí sí es contundente (p=0.0004). El LEAR mejora algo en rampas pero no significativamente (p=0.11), y empeora claramente en los cambios medios.

#### En el ensamble, la mejora vive en la mañana y no toca el pico de la tarde

Por hora del reloj, el delta de MAE del ensamble al añadir el séptimo votante:

| horas | delta | lectura |
|---|---|---|
| 00-05 | ≈ 0.00 | el combinador le da peso ~0 al votante nuevo |
| **06-11** | −1.36, −0.63, −0.97, **−2.22**, −1.16, −1.26 | **aquí está toda la ganancia** |
| 12-17 | entre −0.41 y +0.58 | ruido |
| **18-23** | ≈ 0.00 | peso ~0 |

Las horas 18 y 19 son las dos peores del día (MAE 89.66 y 103.31, el 19.5% de todo el error) y **mejoran exactamente 0.00**. El combinador reparte pesos por franja de 6 horas, y en las franjas 0-5 y 18-23 le asigna peso ~0 al votante nuevo: no aporta nada que los otros seis no tengan ya.

**Correlación entre la rampa media de una hora y su mejora: +0.285.** Positiva, o sea que la mejora es algo *menor* donde las rampas son mayores. Si el método atacara los puntos de cambio, esta correlación tendría que ser negativa.

Matiz honesto en sentido contrario: en el ensamble, el subconjunto de rampa fuerte sí mejora (−0.656, en 10/10 particiones, el mayor delta absoluto de los cuatro grupos). Pero eso no viene de que el votante nuevo sea bueno en rampas —no lo es, es peor— sino de que el combinador rebalancea los pesos de todos. Es un efecto de segundo orden, no el mecanismo que se buscaba.

#### Conclusión

Dos cosas que hay que mantener separadas:

1. **La mejora de NOT es real** (10/10 particiones, 3 de 5 candidatos, 0 de 3 controles) — ver la entrada anterior.
2. **NOT no resuelve el problema de los puntos de cambio intradía.** Mejora las horas planas. El error de las rampas, que es el 35.5% del total y la razón por la que se buscó esta línea, **sigue intacto**.

La línea de los puntos de cambio intradía sigue abierta. Lo que NOT aporta es otra cosa —mejor selección del histórico de entrenamiento— que resultó valiosa por su cuenta.

#### Pieza que faltaba guardarse

`guardar_parcial()` en `ventana_calibracion_not_24h.py` solo archivaba las variantes cuyo nombre empieza por LEAR o contiene NOT, así que las de ventana fija del ARX (los controles) no quedaron en disco y la Parte 1 salió incompleta la primera vez. Regeneradas con las máscaras cacheadas (`vc_ARX_WinH728.csv`, `vc_ARX_Win728.csv`, `vc_ARX_Win364.csv`), a 3 segundos cada una.

---

### 2026-09-23 — Biblioteca de papers con citas verificadas, y dos correcciones de referencias

`docs/papers/REFERENCIAS.md`. Cita completa, enlace de verificación y **para qué se usa en el proyecto** de cada uno de los 8 papers en la carpeta, más los 5 citados en el informe que todavía no tienen PDF local. La idea es que cualquier afirmación del informe se pueda rastrear hasta su fuente.

Datos bibliográficos que se fijaron al verificar contra los PDF:

- **Nasiadka, Nitka & Weron (2022)** — es un paper de **ICCS 2022** (Londres), Lecture Notes in Computer Science, arXiv:2204.00872. La bitácora anterior lo citaba sin año ni venue.
- **Baranowski, Chen & Fryzlewicz (2019)** — JRSS-B 81(3), 649-672, DOI 10.1111/rssb.12322. Es el detector NOT que se reimplementó; faltaba citarlo.
- **Gallón & Barrientos (2021)** — el informe decía IJEEP **11(1)**; el PDF dice **11(2), 67-74**. **Corregido** en `generar_informe_word.py`.
- **Barrientos Marín, Tabares Orozco & Velilla (2018)** — IJEEP 8(3), 97-106. Los tres son de la Universidad de Antioquia; el informe lo citaba sin autores completos.

También queda anotado en el archivo el hallazgo del paper de regímenes de Nueva Zelanda que conecta con una línea ya cerrada aquí: **concluyen que más regímenes no mejoran el pronóstico pese a la complejidad añadida**, lo que es consistente con que el Markov-Switching de 2 regímenes probado en este proyecto tampoco aportara.


### 2026-09-23 — SELECCIÓN DE VENTANA POR PUNTOS DE CAMBIO (NOT): funciona, contesta la pregunta abierta del paper, y es la primera mejora del ensamble que pasa el estándar 10/10 en mucho tiempo

`scripts_experimento/ventana_calibracion_not_24h.py` y `robustez_ventana_not_ensamble.py`. Implementación del método de Nasiadka, Nitka & Weron: en vez de entrenar con las observaciones **más recientes** (ventana móvil de τ días), se usa detección de puntos de cambio para partir el histórico en tramos estacionarios y se entrena **solo con los tramos cuyo comportamiento se parece al régimen vigente** — aunque sean viejos, y saltándose los recientes que no se parecen.

Coste: 10.416 detecciones (217 días × 24 pasos × 2 series), 77 min. Las máscaras no dependen del modelo, así que se cachean en `not_mascaras_24h.npz` y todas las variantes las reusan.

#### Capa 1 — ¿existe el efecto en el mercado colombiano? Sí, y más fuerte que en Alemania

| Modelo ARX (Ec. 1 del paper) | MAE | RMSE | MAPE |
|---|---|---|---|
| `Win(728)` ventana fija de 2 años | 57.97 | 92.35 | 19.26% |
| `WinH(728)` + asinh | 52.31 | 90.75 | 15.39% |
| `NOT(728)` subperíodos por puntos de cambio | 51.41 | 91.04 | 13.92% |
| **`NOTH(728)` puntos de cambio + asinh** | **50.53** | 91.64 | **13.16%** |

Diebold-Mariano: `NOT(728)` vs `Win(728)` **p=0.0000** (−11.3% de MAE); `NOTH(728)` vs `WinH(728)` **p=0.0154**. En el paper la ganancia atribuible a NOT era del 1.7% de RMSE y los autores advertían que **no era significativa por sí sola**; aquí lo es en las dos comparaciones. El asinh también aporta por separado (`WinH` vs `Win`, p=0.0000), replicando su hallazgo.

Detalle propio: el asinh **ayuda con ventana larga y estorba con ventana corta** (τ=728: 52.31 vs 57.97; τ=182: 54.12 vs 51.03). Coherente con que los autores lo usen solo en τ=728.

Nota sobre la escala del asinh: el paper dice literalmente *"b is median absolute deviation"*, pero la MAD cruda comprime de más y la inversión con `sinh` amplifica el error. Se usa la MAD normalizada (÷0.6745). Las dos versiones dan la misma conclusión; la normalizada es uniformemente mejor (τ=728: 52.30 vs 53.23). Documentado en el código.

#### Capa 2 — la pregunta que el paper deja explícitamente abierta

Los autores cierran diciendo que si esto sirve para modelos más complejos que el autorregresivo, *"e.g., LASSO-estimated AR (LEAR) and deep neural networks, is left for future work"*. Este proyecto tiene un LEAR de 24h.

| LEAR | MAE | RMSE | MAPE |
|---|---|---|---|
| `Win(728)` ventana fija de 2 años | 57.38 | 92.86 | 18.60% |
| `Win(todo)` histórico completo — **la configuración real del proyecto** | 51.50 | 88.89 | 15.96% |
| **`NOT(728)` subperíodos por puntos de cambio** | **49.67** | **88.78** | **13.62%** |
| `WinH(728)` + asinh | 49.93 | 86.95 | 14.60% |
| `NOTH(728)` puntos de cambio + asinh | 50.14 | 90.27 | 13.57% |

**Sí transfiere al LEAR.** Contra ventana fija p=0.0000, y —lo que importa— contra *todo el histórico*, que es lo que el proyecto hace hoy, **p=0.0426**.

Hallazgo adicional no anticipado: sobre el LEAR, **NOT y asinh son sustitutos, no complementos** (`NOTH` vs `WinH` p=0.7926). Cada uno por su lado mejora; juntos no suman. Sobre el ARX sí se acumulan. Lectura plausible: ambos atacan el mismo problema (la asimetría/no estacionariedad de la serie), uno recortando la muestra y el otro comprimiendo la escala, y al LEAR le basta con uno.

Detalle de implementación que costó un error: `alpha` del LASSO **no es invariante de escala**. Calibrado sobre precios crudos y aplicado a datos asinh anula todos los coeficientes — MAE 216 en vez de 56. Ahora se calibra por variante, con la misma regla de ventana con la que se va a pronosticar (`calibrar_alphas`).

#### Capa 3 — integración al ensamble, con control de selección post-hoc

El candidato se había elegido por mejor MAPE entre ~26 variantes **sobre el mismo periodo de prueba** donde luego se mide la mejora. Eso es exactamente el sesgo de selección que ya nos ha mordido antes, así que se montó `robustez_ventana_not_ensamble.py`: se meten al ensamble **todos** los candidatos uno por uno, más **controles de ventana fija** (el mismo modelo con la misma información, cambiando solo la regla de qué historia usar).

| candidato | tipo | ΔMAE | ΔMAPE | ΔsMAPE | p(MAPE) | veredicto |
|---|---|---|---|---|---|---|
| ARX NOTH(728) | NOT | −0.295 (10/10) | −0.187 (10/10) | −0.067 (10/10) | 0.0114 | **SÓLIDO** |
| ARX NOT(728) | NOT | −0.199 (10/10) | −0.123 (10/10) | −0.053 (10/10) | 0.0168 | **SÓLIDO** |
| LEAR NOT(728) | NOT | −0.194 (10/10) | −0.102 (10/10) | −0.081 (10/10) | 0.0281 | **SÓLIDO** |
| ARX Av(NOTH) | NOT | −0.252 (10/10) | −0.128 (10/10) | −0.029 (9/10) | 0.0173 | – |
| LEAR NOTH(728) | NOT | −0.096 (9/10) | −0.068 (10/10) | −0.058 (10/10) | 0.0979 | – |
| LEAR WinH(728) | **control** | −0.086 (10/10) | −0.008 (8/10) | −0.002 (7/10) | 0.2096 | – |
| LEAR Win(728) | **control** | −0.016 (7/10) | −0.012 (8/10) | −0.005 (8/10) | 0.0586 | – |
| LEAR Win(todo) | **control** | −0.068 (9/10) | −0.024 (8/10) | −0.016 (8/10) | 0.1173 | – |

**3 de 5 candidatos NOT pasan el estándar 10/10 en las tres métricas; 0 de 3 controles de ventana fija lo pasan.** La ganancia es del método, no de haber elegido bien el candidato ni de sumar un votante lineal más. Los p-valores del DM sobre MAPE separan limpiamente los dos grupos (0.011–0.028 los NOT, 0.059–0.210 los controles).

Con el mejor candidato como séptimo votante:

| métrica | actual (6 votantes) | con ventana por puntos de cambio |
|---|---|---|
| MAE | 41.287 ± 0.113 | **40.993 ± 0.147** |
| MAPE | 10.744% ± 0.033 | **10.557% ± 0.038** |
| sMAPE | 10.065% ± 0.030 | **9.998% ± 0.036** |

Meter los dos mejores del ARX a la vez no aporta nada sobre el mejor solo (40.996 vs 40.993): son redundantes entre sí, basta un votante.

#### Reservas honestas

1. **El DM sobre MAE del ensamble da p≈0.18**, no significativo; sobre MAPE sí (p=0.0114). El 10/10 es el estándar que el proyecto declaró de antemano y se pasa, pero la mejora es chica frente al ruido del combinador.
2. **La ganancia está medida sobre 2026**, el mismo periodo que todo lo demás. Una confirmación limpia exigiría repetirla en otros orígenes del walk-forward. **Pendiente.**
3. La reimplementación de NOT es propia (el original es el paquete `not` de R). Está validada contra series sintéticas con puntos de cambio conocidos (5/5, ver entrada anterior), y el experimento aborta si la autoprueba falla.

#### Qué queda por decidir (no se hizo por cuenta propia)

Adoptar esto cambia las cifras de cabecera del informe (**MAPE 10.74% → 10.56%**), que es material de la sustentación. No se tocó ni el ensamble oficial ni el informe a la espera de decisión del usuario.


### 2026-09-22 — Por qué falló el modelo de rampas (la causa NO era la que yo dije), implementación y validación de NOT, y verificación de la Figura 5 contra el paper original

#### 1. Autopsia del modelo de rampas: una tabla 2×2 identifica al culpable

`scripts_experimento/diagnostico_rampas_porque_fallo.py`. El modelo de rampas falló (0/10 semillas), pero "falló" no es una explicación. Había dos sospechosos y se pueden separar experimentalmente:

- **Causa A, acumulación**: la reconstrucción `p̂(h) = p(corte) + Σ Δ̂` suma 24 predicciones, así que el paso 24 arrastra 24 errores.
- **Causa B, restricción**: anclar y sumar impone que el coeficiente del ancla sea exactamente 1.0, cuando el precio revierte a la media.

Se añadió una tercera variante que tiene B pero no A (*ancla directa*: predecir `y − p(corte)` de una sola vez por paso). Resultado sobre 5.207 filas de 2026:

| variante | restricción | acumulación | MAE |
|---|---|---|---|
| LEAR de nivel (ancla con coeficiente libre) | no | no | 53.37 |
| **Ancla directa (coeficiente forzado a 1.0)** | **sí** | **no** | **53.16** |
| Rampas, suma acumulada (directa) | sí | sí | 62.33 |
| Rampas, suma acumulada (hurdle) | sí | sí | 63.11 |
| *(referencia)* ancla fija = persistencia | – | – | 64.86 |

**Corrección a lo que yo mismo había argumentado.** Yo había razonado que el modelo de rampas vivía en un "subespacio" del modelo de nivel y que forzar el coeficiente del ancla a 1.0 era el problema de fondo. **Es falso.** La restricción no cuesta nada: la *ancla directa* empata (incluso mejora en 0.21) con el LEAR de nivel libre, porque el modelo del residuo compensa la carga impuesta usando los mismos rezagos. Todo el daño (+9.2 de MAE) viene de la acumulación. Que la carga libre del ancla sea muy distinta de 1.0 es cierto pero **irrelevante**:

| paso | coeficiente que elige el LEAR | forzado por las rampas |
|---|---|---|
| 1 | 0.822 | 1.00 |
| 4 | 0.594 | 1.00 |
| 8 | 0.458 | 1.00 |
| 18 | 0.262 | 1.00 |
| 24 | 0.448 | 1.00 |

Media 0.468, mínimo 0.262 (paso 18). Confirma la reversión a la media, pero no explica el fallo.

**La acumulación sí se ve directamente.** Razón MAE(cumsum)/MAE(LEAR) por paso: 1.01x en el paso 1, 1.16x en el 8, 1.27x en el 12, 1.28x en el 16, 1.24x en el 24. Correlación (paso, MAE): +0.598 para el cumsum contra +0.471 para el LEAR de nivel.

**Y el porqué último: el Δ predicho no intenta las rampas.** Desviación típica del Δ real 92.22 contra 41.36 del predicho (44.8%); correlación +0.491; R² +0.240. En las rampas fuertes (n=720) el |Δ| real medio es 184.0 y el predicho 52.6 — **captura solo el 29% de la magnitud**. Encoge por diseño (mitad de los Δ son exactamente 0), y encoger es lo correcto para minimizar error cuadrático en un objetivo inflado en cero; pero al acumularlo 24 veces se inyecta ruido sin aportar señal.

**Conclusión de la línea.** Modelar la transición en una sola pasada *es* el modelo de nivel en otras coordenadas: no es una idea distinta, es la misma reparametrizada, y empata. La única versión que sí es distinta (acumular transiciones consecutivas) es estrictamente peor. La línea queda cerrada con causa identificada, no con un "no funcionó".

#### 2. NOT (Narrowest-Over-Threshold) implementado y validado

`scripts_experimento/not_changepoint.py`. Reimplementación en Python de Baranowski, Chen & Fryzlewicz (2019, JRSS-B); el original es el paquete `not` de R y aquí no hay puente a R. Contraste de razón de verosimilitudes generalizada para media y varianza constantes a trozos (la forma menos restrictiva, la misma que usan Nasiadka/Nitka/Weron), selección del número de puntos por SIC reforzado con Nc_max = 12.

**No se usa un detector sin verificarlo.** `autoprueba()` corre series sintéticas con puntos de cambio conocidos (salto de media, cambio de varianza, ambos, ninguno, y cinco tramos en 728 puntos). Criterio declarado antes de mirar: error de ubicación ≤ 15 y ≤ 1 punto espurio. Resultado final: **5/5**, con localización exacta en 3 de los 5 casos. El experimento aborta si la autoprueba falla.

Dos cosas que hubo que corregir en el camino, ambas detectadas por la autoprueba y no por intuición:
- **El penalizador del SIC con coeficiente 2 deja pasar puntos espurios** cuando media y varianza cambian a la vez. Con 3 (media, varianza y la propia ubicación como parámetros) queda correcto.
- **El atajo de una sola pasada rompe el algoritmo.** Intenté derivar la trayectoria de soluciones de una sola recursión con umbral bajo, guardando el contraste mínimo del camino. Con umbral bajo la regla del "intervalo más angosto" se queda con intervalos diminutos que superan el umbral por ruido: **7 puntos espurios en la serie de un solo salto de media**. La rejilla de umbrales no es un detalle de implementación, es lo que mantiene honesto al detector.

**Coste y cómo se resolvió.** La detección se rehace cada día y cada paso (5.232 veces), a 3,0 s cada una en la versión ingenua = 3,2 h por variante. Vectorizando el cálculo del contraste sobre todos los intervalos candidatos a la vez (aplanar los pares intervalo×corte y recuperar el argmax con `reduceat`) baja a 440 ms → 38 min. Además, las máscaras de subperíodos **no dependen del modelo**, así que se calculan una vez y se cachean en `not_mascaras_24h.npz`; el caché lleva un contador `pasos_listos` y se descarta si quedó a medias.

#### 3. Biblioteca de papers: dos incorporaciones y un enredo de citas resuelto

- `Kapoor_Wichitaksorn_Zhang_2023_NZ_Regime_Switching_JoF.pdf` — Journal of Forecasting 42(8), 2011-2026, DOI 10.1002/for.3004. **Aportado por el usuario.** Markov regime-switching hasta 5 regímenes, probabilidades de transición variables en el tiempo, y teoría de valores extremos (EVT-PoT). Prueban formalmente que el EVT-PoT es un caso particular del MRS de 3 regímenes. Sus métricas son sobre precios **diarios** y con foco en réplica de densidad, no en pronóstico puntual (MAPE de 52% a 299%): **no es comparable con este proyecto en exactitud**, y no es la fuente de la Figura 5.
- `Kapoor_Wichitaksorn_2023_NZ_EPF_AppliedEnergy.pdf` — Applied Energy 347, 121446. **Esta sí es la fuente de la Figura 5a.** Estaba ya descargada en el equipo desde el 17 de septiembre sin haberse incorporado al proyecto.

**Aclaración importante:** son dos papers distintos de autores parcialmente coincidentes. El de regime-switching (Journal of Forecasting) no contiene ni LE-GARCH-t, ni LEAR, ni MASE; el de Applied Energy sí. La bitácora anterior pedía descargar el primero creyendo que respaldaba la Figura 5 — no es así.

#### 4. Verificación de la Figura 5a contra el paper original (lo que el usuario pidió confirmar)

Contrastadas las Tablas 5, 6 y 8 de Kapoor & Wichitaksorn (2023, Applied Energy):

| afirmación del informe | verificado | correcto |
|---|---|---|
| "los 34 modelos del paper neozelandés" | **no** | son **33** (9 base + 8 × 3 métodos de selección de variables) |
| "todos MASE > 1.26" | **sí** | el mínimo global es 1.2626 (LE-GARCH-t, Central North Island) |
| mejores modelos NZ ≈ 1.30 | **sí** | LE-GARCH-t 1.2626–1.3513; LEAR 1.2954–1.3473 |
| peores modelos NZ = 2.51 | **casi** | el máximo real es **2.5001** (RFE-GARCH, Lower South Island) |

**Salvedad no declarada en el informe, y es la que más pesa.** El paper transforma los precios antes de modelar: *"electricity prices, coal generation, diesel generation, reserve prices, forward prices, and precipitation series tend to display a skewed distribution... we choose to transform them using the Box–Cox or Yeo-Johnson logarithmic transformations"*, seguido de escalado min-max. El MASE es invariante al escalado lineal (se cancela en el cociente) pero **no** a la transformación Box-Cox. O sea que su MASE está medido sobre precio transformado y el nuestro sobre COP/kWh crudos: **no son la misma cantidad**. El informe ya advertía la diferencia de resolución (diaria vs horaria) pero no esta. **Corregido**: `generar_informe_word.py` dice ahora 33 modelos, cita el mejor (1.2626, LE-GARCH-t, Central North Island) y el peor (2.5001, RFE-GARCH, Lower South Island), y añade un párrafo con la salvedad de la transformación Box-Cox; `generar_graficos_informe.py` corrige la barra de 2.51 a 2.50. Gráficos e informe regenerados.


### 2026-09-22 — Idea del usuario: modelar los PUNTOS DE CAMBIO intradía. El diagnóstico confirma la intuición, el modelo de rampas falla, y se abre biblioteca de papers

Idea del usuario: en otros mercados, ¿cómo tratan los modelos las series con puntos de cambio? El error del proyecto se concentra justo en las horas de quiebre de la curva diaria, así que valía la pena buscar qué hace la literatura y probarlo.

**El diagnóstico confirma la intuición de forma contundente** (medido sobre el ensamble actual, 2026):

| Tipo de hora | % del tiempo | MAE del ensamble |
|---|---|---|
| Precio plano (cambio exactamente 0) | 50.8% | 28.5 |
| Cambio chico (<10) | 18.6% | 28.4 |
| Cambio medio (10-50) | 16.8% | 40.5 |
| **Rampa fuerte (≥50)** | 13.9% | **105.5** |

El error se cuadruplica en las rampas, y las 6 horas de mayor error (0, 8, 9, 10, 18, 19) tienen rampa media de 46.6 contra 24.7 de las otras 18. **El error vive en los quiebres, no en las mesetas.** Correlación global entre |rampa horaria| y error: 0.497.

**Hallazgo estructural nuevo**: más de la **mitad de las horas el precio no cambia nada** respecto a la hora anterior (la misma unidad marca el precio varias horas seguidas). La distribución de cambios está fuertemente inflada en cero — algo que el proyecto no había caracterizado.

**PRUEBA: modelo de rampas (`modelo_rampas_24h.py`).** Siguiendo la literatura de eventos de rampa en energía eólica (entrenamiento orientado a la transición en vez del nivel), se construyó un modelo que pronostica Δ = p(t_h) − p(t_{h−1}) por cada paso, en dos variantes: LASSO directo sobre Δ, y hurdle de dos partes (clasificador "¿se mueve?" × regresor "¿cuánto?") para respetar la masa en cero — el proyecto ya usó ese esquema sobre el *nivel*, aplicarlo a la *transición* era nuevo.

**Falla en las dos reconstrucciones probadas:**

| Reconstrucción | MAE | Veredicto |
|---|---|---|
| Acumulada desde el corte, como votante extra | 41.22-41.28 | peor en **0/10** semillas |
| Anclada en ensamble(h−1), mezcla 50/50 | 43.00 | **p=0.0000 peor** |
| Anclada, sustituyendo del todo | 46.73 | **p=0.0000 peor** |

Suelto, el modelo de rampas da MAE 62.3 — apenas mejor que usar el precio del corte fijo (64.9), y muy lejos del ensamble (41.2). **Causa de fondo**: predecir Δ es más difícil que predecir el nivel en esta serie, porque la distribución está partida entre 50.8% de ceros exactos y una cola de saltos grandes. Cualquier reconstrucción arrastra ese error, ya sea acumulándolo sobre 24 pasos o inyectándolo hora a hora. En las horas de rampa fuerte el efecto fue marginal (105.53 → 105.11, p=0.0585) y no sobrevive a la verificación multisemilla.

**BIBLIOTECA DE PAPERS CREADA (`docs/papers/`).** Al buscar la literatura se descubrió que había PDFs completos guardados de sesiones anteriores que se creían perdidos. Quedan 7 papers con PDF + texto extraído, citables con precisión:

| Paper | Por qué importa |
|---|---|
| Lago, Marcjasz, De Schutter & Weron (2021) | La revisión canónica de EPF; el README decía "no leído completo" — **ya está disponible** |
| Ziel & Weron (2018) | Univariado vs multivariado (las 24 horas como vector de una subasta) |
| Huisman & Mahieu (2003) | Los 3 regímenes (normal / salto / retorno). Conseguido gratis en el repositorio de Erasmus |
| Nasiadka, Nitka & Weron | Selección de ventana de calibración por detección de puntos de cambio |
| Das & Schlüter | Regime-Aware Conditional Neural Processes |
| Gallón & Barrientos (2021) | La referencia colombiana que ya citábamos |
| IJEEP (2018) | Segunda referencia colombiana, no se sabía que se tenía |

**LÍNEA ABIERTA con respaldo explícito de la literatura.** El paper de Nasiadka, Nitka & Weron usa detección de puntos de cambio (algoritmo Narrowest-Over-Threshold) para elegir **qué subperíodos históricos usar como muestra de calibración**, en vez de tomar siempre los datos más recientes — y mejora significativamente sobre EPEX SPOT alemán. En sus conclusiones dicen textualmente que si esto funciona para *"LASSO-estimated AR (LEAR) and deep neural networks ... is left for future work"*. **Este proyecto tiene exactamente un LEAR y redes profundas**, así que es una pregunta abierta de la literatura que estamos en posición de responder. No se ha probado todavía. (Nota: ese trabajo enfatiza la transformación asinh, que aquí ya se probó y **empeoró** 6-16%, porque los picos colombianos son cambios de nivel sostenidos y no picos aislados como en Europa.)

### 2026-09-22 — Prueba exhaustiva del anclaje: DOS conclusiones previas eran incorrectas, y la línea se cierra por un error de razonamiento propio

Scripts: `anclaje_lag24_exhaustivo.py`, `accionar_salto_predicho.py`. Reejecución seria de las pruebas de la entrada anterior, que eran débiles (n=217, un solo target, validación cruzada que no respeta el orden temporal).

**CORRECCIÓN 1 — la hora 0 es la única donde el modelo NO le gana a la persistencia.** La correlación de 0.974 entre error y salto día-a-día sí es especial de la hora 0 (correlación media entre las 24 horas: 0.600; la hora 0 está en el percentil 95), pero el hallazgo relevante es otro:

| Hora | MAE modelo | MAE persistencia | Ganancia vs persistencia |
|---|---|---|---|
| **0** | 47.37 | 46.76 | **−1.3%** |
| 1 | 7.60 | 47.15 | +83.9% |
| 9 | 44.02 | 61.51 | +28.4% |
| 19 | 104.06 | 110.32 | +5.7% |

En las horas 1-5 el ensamble le gana a la persistencia por 52-84%; en la hora 0 queda por debajo. Matiz honesto: la diferencia (0.6 MAE) está dentro del error estándar de ±4.2 a esa hora, así que es un estimador puntual, no un resultado significativo.

**CORRECCIÓN 2 — el salto día-a-día SÍ es predecible; la conclusión anterior ("no lo es") venía de una prueba mal hecha.** Con la historia completa (2.742 días en vez de 217) y `TimeSeriesSplit` en vez de pliegues estratificados:

| Hora | Target | Mejor AUC |
|---|---|---|
| 0 | percentil 80 (saltos extremos) | 0.791 |
| 9 | percentil 80 | 0.799 |
| 19 | percentil 80 | 0.814 |

Muy por encima del 0.70 que el propio proyecto fijó como umbral de utilidad. La prueba anterior daba 0.54-0.59 por falta de potencia estadística y por usar una validación cruzada inadecuada para series temporales.

**PERO LA LÍNEA SE CIERRA, por un error de razonamiento que debí ver antes de programar nada.** La acción obvia —enrutar a persistencia los días de salto grande— es **circular**: en un día de salto grande, la persistencia es *precisamente* el modelo que más falla, porque el salto **es** su error. Se confirmó empíricamente incluso con oráculo (que conoce el salto real):

| Hora | Ensamble | Enrutar con oráculo del salto |
|---|---|---|
| 0 | 47.37 | 48.59 (peor) |
| 9 | 44.02 | 64.03 (mucho peor) |
| 19 | 104.06 | 135.99 (mucho peor) |

La verificación directa lo confirma: la persistencia gana en los días de salto **pequeño** (salto medio 21-45), y el ensamble gana en los de salto **grande** (salto medio 62-165) — lo contrario de la hipótesis.

**Se probó también la dirección correcta** (enrutar a persistencia cuando se predice salto pequeño): sin mejora en las horas 0 y 9, y una aparente mejora en la hora 19 (104.06 → 94.49, p=0.0722; 95.81 con p=0.0515). **No se adopta, y conviene explicar por qué**: a esa altura se llevaban 18 comparaciones en esta línea (3 horas × 3 umbrales × 2 direcciones). Bajo la hipótesis nula, el número esperado de resultados con p<0.05 es 18 × 0.05 ≈ 0.9 — se obtuvo exactamente uno. Es lo que predice el azar, y es el mismo patrón del falso hallazgo del ONI que el proyecto ya tiene documentado. Reportarlo como prometedor sería repetir ese error.

**Lo que queda en pie:** el anclaje al precio del día anterior es real y domina el error en la hora 0, el salto es anticipable, pero **no existe una acción que convierta esa señal en una mejora del pronóstico puntual**, porque los dos modelos disponibles fallan en los mismos días. Sería necesario un tercer modelo que acierte específicamente en días de ruptura — y el proyecto ya cerró esa vía tres veces (especialistas de pico, arquitectura de dos modelos, enrutamiento por fallo predicho).

### 2026-09-18 — El mecanismo real del error en la hora 0: anclaje al precio de ayer, confirmado con correlación 0.974, pero no anticipable en el corte

El usuario consultó a ChatGPT y Gemini sobre la Figura 4 (error por hora). Tres hipótesis planteadas, verificadas aquí contra los datos:

**1) "El MAPE se infla porque el precio es bajo en la hora 0" — descartada.** El precio real medio en la hora 0 (362.9) es casi idéntico al de la hora 1 (356.2), pero el MAE difiere 6 veces (47.4 vs 7.6). El nivel de precio no es la explicación.

**2) "Probar codificación cíclica seno/coseno de la hora" — ya implementado.** `hora_sin`/`hora_cos` están en el pipeline compartido desde el notebook 05, y se pasan como variable futura conocida incluso a N-BEATSx/N-HiTS.

**3) "El modelo se ancla al precio de ayer (lag-24) y falla cuando el mercado cambia bruscamente de un día a otro" — confirmado con fuerza.** Correlación entre `|precio(hora 0, hoy) − precio(hora 0, ayer)|` y el error del ensamble: **0.974**. Comparando terciles: cuando el día se parece a ayer, MAE=9.2; cuando es distinto, MAE=111.1 — **12 veces más**. Es el número más contundente que el proyecto tiene sobre el mecanismo del error en esa hora, y afina lo que ya se había encontrado con Ziel & Weron sobre la frontera entre subastas.

**Se probó si es explotable, y no lo es.** Se intentó predecir, con información disponible en el corte, si el salto día-a-día sería grande: con variables candidatas (volatilidad reciente, ONI, día de la semana, festivos) AUC=0.586; con el estado completo de 14 variables vía CatBoost, AUC=0.540. Ambos prácticamente azar. El mecanismo domina el error pero no hay señal en el corte que anticipe cuándo va a activarse — misma conclusión que `predecir_fallo_24h.py` desde otro ángulo: el fallo es real, no es accionable con las variables actuales.

### 2026-09-18 — Auditoría sistemática del proyecto por clases de error: se encuentra una fuga real en el ONI, se cuantifica y resulta inocua; todo lo demás sale limpio

Revisión ordenada de notebooks y scripts buscando errores, por clase de fallo en vez de línea por línea (126 scripts + 13 notebooks).

**HALLAZGO: el ONI contiene información futura.** El archivo `data/external/oni_index.csv` viene en el formato estándar de NOAA, cuyas columnas (`DJF`, `JFM`, `FMA`, …) son medias móviles de **3 meses centradas**. La función `construir_oni_horario_causal` (notebook 02) mapea cada columna a su mes central — `JJA` → julio — y arma una rampa de ONI(M−1) a ONI(M) dentro del mes M. El problema es que ambos extremos miran al futuro:

- `ONI(M)` = media de SST de **M−1, M, M+1** → incluye el mes siguiente
- `ONI(M−1)` = media de **M−2, M−1, M** → incluye el mes en curso

Son hasta **~1.5 meses de información futura**, más el hecho de que NOAA publica el valor de un mes a comienzos del siguiente. La función se llama "causal" pero no lo es respecto al dato subyacente. La diferencia es material en la variable: al cierre de junio 2026 el pipeline usa ONI = 1.4 mientras que una versión genuinamente causal daría **0.50** — la diferencia entre clasificar un Niño débil y uno fuerte.

**Pero el efecto sobre el modelo es nulo** (`auditoria_oni_causal.py`). Se construyó un ONI sin futuro (para el mes M se usa ONI(M−2), enteramente pasado y ya publicado) y se reentrenó XGBoost con las dos versiones, todo lo demás idéntico:

| | ONI actual (con futuro) | ONI causal (sin futuro) |
|---|---|---|
| MAE | 61.39 | 61.41 |
| MAPE | 15.97% | 15.97% |
| Importancia de la variable | 0.0033 (puesto 27 de 40) | 0.0037 (puesto 26) |

Diebold-Mariano entre ambas: **p=0.6475**, diferencia de MAE de 0.021 COP/kWh. Correlación entre las dos series del ONI: 0.895, con diferencia media absoluta de 0.293 y máxima de 0.900 — o sea las series difieren mucho y el modelo no lo nota. Es coherente con la ablación LOGO previa, que ya había mostrado que quitar el ONI por completo mueve el MAE dentro del ruido.

**Conclusión:** la fuga es real y hay que declararla, pero está demostrado que no infla ningún resultado del proyecto, porque el ONI no aporta capacidad predictiva. Conviene decirlo así en la sustentación si alguien del jurado conoce la definición del índice, en vez de que lo encuentren ellos. El arreglo (usar ONI(M−2)) está implementado y probado en `auditoria_oni_causal.py` por si se decide adoptarlo — cambiar el pipeline obligaría a reentrenar todo para una diferencia medida de 0.02 COP/kWh.

**Lo que salió limpio en la auditoría:**

| Clase de error revisada | Resultado |
|---|---|
| Fuga en el pipeline de features (notebook 05) | **Limpio** — todo lleva `.shift(24)` antes de cualquier ventana móvil; incluso las columnas crudas de hidrología se sobrescriben con su versión rezagada |
| Escaladores ajustados sobre train+test | **Limpio** — se ajustan solo en train, y los NaN de test se rellenan con la mediana de *train* |
| MAPE con denominadores cercanos a cero | **Limpio** — precio mínimo 69.95 (train) y 97.72 (2026); ningún cero ni valor pequeño |
| Orden cronológico antes del estimador HAC del Diebold-Mariano | **Limpio** en el linaje desplegado (`cargar24().sort_index()`; las predicciones de validación cruzada se escriben en el orden original, no en orden de pliegue) |
| Anti-fuga de la formulación directa (rezagos contra el corte) | **Limpio por construcción** — `precio_mismo_hora_disp` usa `retro = 24·⌈h/24⌉ ≥ h` para todo h; `precio_lag168h` es seguro porque h ≤ 72 < 168 |
| Separación train/test | **Limpio** — `train = df < corte_train`, `test = df ≥ test_inicio`, sin solape |
| Regularidad de la serie temporal | **Limpio** — 0 saltos distintos de 1 hora, sin duplicados (Colombia no tiene horario de verano) |
| Motor de decisión de Rafael (solo lectura) | **Limpio** — las señales se generan con `q50` (la predicción), no con el precio real; los umbrales fijos salen de `precio_historico_train`; los rodantes usan `.shift(1)` antes de la ventana, explícitamente para evitar fuga. Las dos alertas de mi búsqueda por palabras clave resultaron falsos positivos (el `real` se usa solo para graficar y para evaluar después) |

### 2026-09-18 — La hora 0 contra la literatura (Ziel & Weron 2018): el salto de medianoche no es un defecto a corregir, y el margen ahí está acotado por construcción

Script: `promedio_simple_hora0.py`. Cierre de la línea de la hora 0, ahora con respaldo teórico.

**Lo que dice la literatura.** Ziel & Weron (2018, *Energy Economics* 70:396-420) es el trabajo de referencia sobre cómo estructurar un modelo day-ahead. Su planteamiento: los precios de las 24 horas de un día **se revelan de una vez en una sola subasta**, así que la unidad natural es un vector de 24 dimensiones, no una serie horaria continua. **El salto entre las 23:00 y las 00:00 no es una anomalía que haya que arreglar: son dos subastas distintas.** Para la hora 0, el enfoque estándar es usar rezagos de la misma hora de días anteriores, no la hora inmediatamente previa. Y reportan que combinar los marcos univariado y multivariado mejora la precisión "incluso con un promedio simple de dos modelos" — con el matiz honesto de que el marco multivariado **no gana uniformemente** entre conjuntos, estaciones ni horas del día.

**Cómo queda el proyecto frente a eso.** El ensamble ya implementa ambos marcos sin que se hubiera planteado en esos términos: multivariado (N-BEATSx, N-HiTS y el LEAR24 de un modelo por hora, que además usa `MISMA_HORA = [24, 48, ..., 168]`, exactamente los rezagos de misma hora que recomienda la literatura) y univariado con rezagos de 24h+ (Persistencia, XGBoost, ARX+GARCH). Combinar ambos es justo lo que ese paper recomienda — un punto defendible en la sustentación.

**Corrección a una lectura previa mía.** En el análisis inicial se señaló que "en la hora 0 el ensamble (47.37) es peor que dos de sus votantes (ARX+GARCH 45.99, Persistencia 46.76)" y se presentó como evidencia de un problema. **No lo era**: con 216 días, el error estándar del MAE a esa hora es **±4.20**, así que una diferencia de 1.4 está dentro del ruido. La observación no sostenía la conclusión.

**Prueba realizada: promedio simple de votantes solo en la hora 0.** Motivada por el hallazgo de Ziel & Weron sobre promedios simples, y por la hipótesis de que los pesos por franja (0-5h) están dominados por las horas 1-5, donde las redes son muy superiores porque explotan el precio recién observado en el corte — ventaja que en la hora 0 no existe.

| Métrica | Combinador | Promedio simple en h0 | Delta | DM |
|---|---|---|---|---|
| MAE global | 41.287 | 41.227 | −0.061 | p=0.0901 |
| MAPE | 10.744% | 10.739% | −0.005 | p=0.6785 |
| sMAPE | 10.065% | 10.047% | −0.018 | — |
| MAE hora 0 | 47.354 | 45.898 | −1.456 | p=0.1819 |

**No se adopta.** Sin significancia, y la mejora global de MAPE (0.005 puntos) es inmaterial. **Advertencia metodológica sobre la lectura de este resultado**: aparece "mejor en 10/10 semillas" en todas las métricas, pero el promedio simple es determinista (desviación ±0.000 en la hora 0), así que son 10 apariciones del mismo cambio, **no 10 confirmaciones independientes**. Contarlo como evidencia fuerte sería un error del mismo tipo que el proyecto ya cometió con el falso hallazgo del ONI.

**Techo estructural del margen en la hora 0.** Es el **4.2%** de las observaciones, así que aunque se resolviera por completo, el efecto sobre el global queda dividido entre ~24. El oráculo de selección (elegir el mejor votante cada día, usando el futuro) da MAE 25.14 contra los 47.4 actuales: hay margen teórico grande, pero no accionable, y aun capturándolo entero el MAPE global bajaría menos de 0.5 puntos. **La hora 0 no es donde están las mejoras que importan.**

### 2026-09-18 — Dos líneas de mejora probadas y cerradas: corrección de sesgo por nivel de precio, y separar la hora 0 en su propio grupo de pesos

Scripts: cálculo directo sobre `combinador_optimo_mape.csv`; `franja_medianoche_24h.py`.

**1) Corrección de sesgo condicionada al precio PREDICHO — falla incluso con oráculo.** A diferencia de las correcciones de sesgo ya descartadas (global, por régimen, adaptativa móvil), esta se condiciona a algo que sí se conoce en el momento de pronosticar: el decil de precio predicho. El sesgo existe y es ordenado (desde +2.6% en el decil más barato hasta +3.1% en el decil 8, y −1.2% en el decil 10), pero **corregirlo empeora**: aplicando el sesgo exacto de cada decil con un oráculo que usa el futuro, el MAPE sube de 10.72% a **11.16%**. El sesgo es demasiado pequeño frente a la desviación del error dentro de cada decil, así que desplazar el nivel daña más observaciones de las que arregla — y en MAPE el daño se concentra en las horas de precio bajo. **Cuarta confirmación independiente** de que no hay corrección de sesgo aprovechable en este modelo.

**2) Separar la hora 0 en su propio grupo de pesos — no concluyente.** Motivación concreta, no búsqueda ciega de granularidad: con el corte a las 00:00, la primera franja (horas 0-5) mezcla dos regímenes opuestos por el salto de medianoche. En las horas 1-5 las redes dominan aplastantemente (N-BEATSx MAE 10.3 en la hora 1 contra 46-51 de ARX+GARCH/XGBoost/Persistencia), pero en la hora 0 —que es el paso 24, al otro lado del salto— **pierden la ventaja** (N-BEATSx 52.3, N-HiTS 49.0, frente a ARX+GARCH 46.0). El síntoma medible: en la hora 0 el ensamble (MAE 47.37) es **peor que dos de sus propios votantes**.

Resultado de separarla en su propio grupo (5 grupos en vez de 4): MAE global 41.19 → 41.18, MAPE sin cambio, MAE en la hora 0 47.37 → 47.12 con **p=0.7978**. En 10 particiones el MAE mejora en 10/10 pero solo 0.039 COP/kWh, muy por debajo del umbral de ruido del proyecto, y el MAPE mejora en 4/10. En modo desplegable no hay diferencia (p=0.9459). **Interpretación:** el combinador QRA ya tiene intercepto libre por grupo, y ese intercepto absorbe el desnivel de la medianoche sin necesitar un grupo aparte; separarlo solo añade parámetros. Consistente con el resultado previo de que la granularidad de pesos (1 a 24 grupos) está agotada.

### 2026-09-18 (madrugada) — Diversidad intra-familia (variantes de N-BEATSx) para 25-72h: efecto real pero débil en 25-48h, nulo en 49-72h, y redundante donde ya está el puente

Script: `variantes_nbx_72h.py` (7 min). Tercera vía probada para el margen de 25-72h tras el fallo de los árboles especialistas (entrada anterior). `pronostico_72h_diario_votantes_2026.csv` ya traía tres variantes de N-BEATSx nunca usadas como votantes propios: misma arquitectura con semilla 7, semilla 123, y ventana de entrada de 336h (14 días) en vez de la base. Se probó agregarlas al conjunto de 4 votantes (LAD y combinador sMAPE) y un sub-ensamble (promedio simple de las 4 redes NBX) como votante único.

**Tramo 1-24h: mejora consistente (10/10 semillas) pero de magnitud pequeña, y RESULTA REDUNDANTE con el puente ya adoptado.** MAPE 13.04%→12.64% (p=0.0040 en el split base, desplegable p=0.0668 — al borde de significancia, no la cruza limpiamente). Se comprobó si era aditiva al puente (entrada cont. 10, que ya deja ese tramo en 10.54%): **no lo es** — puente + variantes NBX da 10.56% contra 10.54% del puente solo (p=0.2189, n.s.). La señal de estas variantes ya está contenida en el ensamble de 24h que aporta el puente (que incluye N-BEATSx y N-HiTS entre sus 13 votantes). **Se descarta esta vía para 1-24h; el puente sigue siendo la mejora vigente ahí.**

**Tramo 25-48h: evidencia débil, no pasa el estándar del proyecto.** Gana en 9/10 semillas (no 10/10), MAPE 18.28%→18.24% de media. En modo desplegable, MAPE 18.38%→17.97%, dirección correcta y de tamaño no despreciable, pero **p=0.0691, no significativo**. Queda como línea abierta pero sin confirmar — no se adopta todavía.

**Tramo 49-72h: no ayuda.** Todas las configuraciones quedan peor o sin diferencia (mejor p=0.0512, y en la dirección de MAPE, no de MAE, con el propio MAE empeorando).

**Lectura acumulada de las tres vías probadas para 25-72h (árboles especialistas + esta):** el margen que sí existía en 1-24h (resuelto por el puente, una ganancia estructural grande) no tiene equivalente evidente en 25-72h con las herramientas disponibles. Ahí no hay un canal "hermano" más preciso que trasplantar como en 1-24h, y ni la especialización por árboles ni la diversidad intra-familia de redes producen una mejora que sobreviva el estándar de verificación completo (CV + desplegable significativos). Es coherente con el diagnóstico ya documentado de que el estado observable en el corte informa cada vez menos mientras más lejos está el objetivo.

### 2026-09-17 (noche, cont. 11) — ¿El mismo problema del puente en 25-72h? Se prueba con árboles especialistas por tramo: 25-48h no concluyente, 49-72h parece ganar en validación cruzada pero PIERDE en modo desplegable

Script: `especialistas_por_tramo_72h.py` (12 min). Pregunta directa tras el hallazgo del puente (entrada cont. 10): si el tramo 1-24h perdía por usar modelos generalistas donde había un especialista mejor, ¿los tramos 25-48h y 49-72h sufren lo mismo? Ahí no existe un canal day-ahead que reutilizar, así que se construyó la especialización: se reentrenaron CatBoost, XGBoost y Ridge de la formulación directa usando **solo** los pasos de su propio tramo (en vez de los 72 pasos juntos), y se probaron pesos por paso de horizonte y por bloques de 6 pasos en vez de por tramo completo.

**Individuales:** ningún especialista le gana a su generalista de forma clara (p.ej. en 49-72h, Ridge-directo generalista 82.13 MAE vs Ridge_esp especialista 81.42 — mejora chica, y CatB/XGB especialistas quedan *peor* que sus generalistas).

**Ensamble, 25-48h: NO concluyente.** Mejor configuración (árboles especialistas, pesos por tramo) gana en MAPE en 7/10 semillas pero el delta es de solo −0.02 puntos, dentro del ruido de partición.

**Ensamble, 49-72h: aquí apareció la señal de alerta que el proyecto ya conoce.** En validación cruzada por pliegues, la misma configuración da "SÓLIDO" — gana 10/10 semillas en las tres métricas (MAPE 21.661%→21.555%). Pero en **modo desplegable** (pesos calculados solo con días anteriores, el que de verdad importa) el resultado se invierte: MAPE 21.53%→21.71%, **peor**, y sin significancia (p=0.6372).

**Diagnóstico: mismo patrón que la granularidad de pesos en 24h (ver oráculos por hora-del-día vs por día).** La validación cruzada por pliegues mezcla datos de todo el año para entrenar y evaluar en cualquier punto del calendario; en producción, con solo el historial pasado disponible en cada corte, el árbol especialista tiene menos datos efectivos para ese tramo concreto y se sobreajusta a los pliegues de la validación. **No pasa el estándar del proyecto (CV + desplegable deben coincidir), así que no se adopta.**

**Conclusión de esta línea:** el mecanismo que funcionó en 1-24h (especialización) no se replica gratis en 25-72h porque ahí no hay un canal ya maduro para reutilizar — construir uno de cero con los mismos datos no aporta señal nueva, solo la reparte de forma distinta. El margen de 25-72h sigue abierto para otra vía.

### 2026-09-17 (noche, cont. 10) — TERCERA MEJORA VERIFICADA, y la más grande del horizonte largo: el producto de 72h estaba prediciendo su primer día mucho peor que el modelo de 24h del propio proyecto. Global MAPE 17.66% → 16.94%

Script: `puente_24h_a_72h.py`. Resultado accidental: salió del desglose por tramo de la corrida anterior.

**El hallazgo.** Sobre **exactamente las mismas horas objetivo**, el canal de 72h y el canal de 24h del proyecto daban esto:

| Canal | MAE | MAPE | sMAPE |
|---|---|---|---|
| ensamble de 72h, tramo 1-24h | 46.20 | 12.79% | 11.85% |
| ensamble de 24h del proyecto | **40.72** | **10.54%** | **9.89%** |

Y el de 24h lo consigue **con menos información, no con más**. Su protocolo es corte a las **23:00 del día anterior**; el de 72h corta a las **00:00 del propio día**. El de 24h va una hora por detrás y aun así gana 2.25 puntos de MAPE. Se verificó explícitamente que **ninguna** fila tiene el corte de 24h posterior al corte de 72h (0 de 4922), así que no es fuga: era una pérdida real del canal de 72h.

**Por qué pasaba.** Es de construcción, no de datos. El ensamble de 72h vota entre 4 modelos **generalistas** (LEAR, dos redes y un naive) entrenados para cubrir los 72 pasos, con pesos por tramo. El de 24h vota entre **13 especialistas day-ahead** (XGBoost, CatBoost, hurdle, Markov-Switching, N-BEATSx, N-HiTS, RecursiveLS day-ahead, los afinados con Optuna, los directos) más el GARCH-ged, con pesos por franja de 6 horas y combinador alineado a la métrica. **El canal de 72h estaba re-derivando el día 1 desde cero, peor que el canal que ya existía para ese mismo día.**

**Alcance del arreglo (límite duro, no elegido por conveniencia).** El pronóstico de 24h para la hora `t` se emite en el corte 23:00 del día anterior a `t`. Ese corte es ≤ el corte de 72h solo si `t` cae dentro del propio día del corte, lo que cubre los **pasos 1-23** (01:00 a 23:00). El paso 24 (00:00 del día siguiente) tendría que usar un pronóstico emitido 23 horas **después** del corte de 72h, así que se excluye y se deja como estaba. Los tramos 25-48h y 49-72h no se tocan: no existe pronóstico day-ahead válido a esa distancia.

**Configuraciones probadas (pasos 1-23, DM con HAC contra el vigente):**

| Configuración | MAE | MAPE | sMAPE | p(MAPE) |
|---|---|---|---|---|
| A 4 votantes 72h (vigente) | 46.20 | 12.79% | 11.85% | — |
| B solo ENS24 | 40.72 | 10.54% | 9.89% | 0.0000 |
| C 4 votantes + ENS24, LAD | 40.68 | 10.55% | 9.88% | 0.0000 |
| **D 4 votantes + ENS24, combinador sMAPE** | **40.66** | **10.54%** | **9.87%** | **0.0000** |
| E 4 votantes + los 14 votantes de 24h, LAD | 42.87 | 11.57% | 10.57% | 0.0002 |
| F igual que E con combinador sMAPE | 42.88 | 11.59% | 10.56% | 0.0003 |
| G todo junto (ENS24 + individuales), sMAPE | 40.67 | 10.55% | 9.88% | 0.0000 |

**Dos lecturas importantes de esa tabla.** (1) Casi toda la ganancia es el puente en sí: D apenas le saca 0.06 de MAE a "solo ENS24". Los 4 votantes de 72h aportan margen residual — el combinador les da 6.4% del peso en total (`ENS24=0.936`, `NBX_exog=0.057`, `NBX=0.007`) y a **LEAR le da cero**, aunque era el mejor individual del canal de 72h. (2) **E y F son peores que C y D**: meter los 14 votantes individuales de 24h en un solo LAD plano rinde 1 punto de MAPE peor que meter el ensamble de 24h ya armado. O sea la ganancia viene de la **arquitectura** de ese ensamble (pesos por franja + combinador alineado a la métrica), no de tener sus votantes disponibles. Aplanarla la destruye.

**Verificación (estándar del proyecto).** 10 particiones distintas, contra el vigente:

| Métrica | Vigente | Con puente | Delta | Mejor en |
|---|---|---|---|---|
| MAE | 46.260 ± 0.090 | 40.633 ± 0.019 | −5.628 | 10/10 |
| MAPE | 12.804 ± 0.026 | 10.537 ± 0.009 | −2.267 | 10/10 |
| sMAPE | 11.860 ± 0.024 | 9.869 ± 0.008 | −1.990 | 10/10 |

Y en **modo desplegable** (pesos calculados solo con días anteriores, ventana móvil de 60 días, que es la cifra que un jurado debería ver): MAPE 12.83% → **10.53%**, DM p=0.0000. La mejora no depende de la validación cruzada por pliegues.

**Efecto sobre el producto completo de 72h:**

| Tramo | Vigente | Con puente |
|---|---|---|
| 1-24h | MAE 46.39 / MAPE 13.00% / sMAPE 12.00% | **MAE 41.08 / MAPE 10.84% / sMAPE 10.11%** |
| 25-48h | MAE 62.56 / MAPE 18.29% / sMAPE 16.53% | sin cambio |
| 49-72h | MAE 72.87 / MAPE 21.69% / sMAPE 19.44% | sin cambio |
| **GLOBAL** | MAE 60.61 / MAPE 17.66% / sMAPE 15.99% / R² 0.8618 | **MAE 58.84 / MAPE 16.94% / sMAPE 15.36% / R² 0.8637** |

**Es la mejora más grande del horizonte de 72h de toda la sesión** (−0.72 puntos de MAPE global, −1.77 de MAE), y a diferencia de las otras no vino de un modelo nuevo: vino de notar que dos canales del propio proyecto se solapaban y uno era peor. Pregunta que un jurado haría y que ahora tiene respuesta: *"¿por qué su error a 24h es 10.7% pero su producto de 72h reporta 13% para esas mismas horas?"*

**Pendiente de decisión (afecta a Rafael).** Esto cambia el contrato de 72h que alimenta el dashboard (`pronostico_ensamble_72h_diario_2026.csv` y el contrato unificado de OE3). No se regeneró nada todavía.

### 2026-09-17 (noche, cont. 9) — Las dos mejoras de 24h NO se trasladan al horizonte de 72h: ninguna es significativa, y el mecanismo se apaga con la distancia al corte

Script: `mejoras_a_72h.py` (24 min, 72 modelos GARCH-t, ninguno falló en converger). Se llevaron al canal de 72h las dos ideas que sí ganaron en 24h esta noche: el votante de colas pesadas (GARCH-t por paso de horizonte) y el combinador alineado con la métrica.

**Expectativa previa, explícita en el encabezado del script:** que ayudaran *más* a 72h, porque a 48-72h el error es mayor y más disperso, que es justo donde las colas pesadas y la reponderación deberían importar. **Salió al revés.**

**El GARCH-t suelto se degrada con el horizonte mucho más rápido que el LEAR:**

| Tramo | LEAR MAE | GARCH-t MAE | LEAR MAPE | GARCH-t MAPE |
|---|---|---|---|---|
| 1-24h | 51.08 | **50.97** | 14.43 | **14.10** |
| 25-48h | **67.59** | 74.13 | **19.62** | 20.77 |
| 49-72h | **76.66** | 87.17 | **22.73** | 24.34 |

Le gana al LEAR en el tramo corto y le pierde por 10.5 puntos de MAE en el largo. En el ensamble eso se traduce en que el combinador le da peso útil a 1-24h (MAPE 13.03% → 12.79%) y **peso cero** a 49-72h (los números salen idénticos al vigente, 72.87 / 21.69% / 19.44%).

**Ninguna de las 12 comparaciones alcanza significancia.** El mejor p-valor de toda la corrida es 0.0754 (combinador sMAPE + GARCH-t en 49-72h). El combinador alineado a la métrica hace lo que la teoría dice en 25-48h y 49-72h — baja MAPE y sMAPE, sube MAE — pero sin significancia y sin el efecto de regularización que en 24h lo hacía ganar en las tres métricas a la vez.

**Interpretación.** El mecanismo del LE-GARCH-t es ponderar la verosimilitud por la varianza condicional para descontar los días volátiles. Eso exige que la varianza condicional en el corte diga algo sobre la varianza del objetivo, y esa relación se desvanece con la distancia: a 1-24h el estado del corte todavía informa, a 49-72h ya no. Es el mismo límite informacional que el proyecto lleva documentado por otras cinco vías, visto ahora en el eje del horizonte en vez del eje de las variables.

**El desglose por tramo de esta corrida negativa es lo que destapó el hallazgo del puente** (entrada cont. 10): ver el tramo 1-24h del canal de 72h aislado, en 13.03%, obligó a preguntar por qué no se parecía al 10.74% del canal de 24h.

### 2026-09-17 (noche, cont. 8) — SEGUNDA MEJORA VERIFICADA: alinear el combinador con la métrica reportada. MAPE 10.86% → 10.74%, sMAPE 10.08% → 10.07%, y además baja el MAE

Se corrigió un desajuste que llevaba todo el proyecto sin detectarse. Scripts: `combinador_optimo_mape.py`, `verificar_combinador_metrica.py`.

**El desajuste.** El ensamble ajusta sus pesos minimizando la **desviación absoluta** (LAD/QRA al cuantil 0.5), que es el óptimo para el **MAE**. Pero el proyecto se evalúa y se compara con la literatura en **MAPE** y **sMAPE**, y el minimizador de esas métricas no es el mismo:

| Métrica | Combinación óptima |
|---|---|
| MAE | `min Σ \|y − Xw\|` (mediana condicional) |
| MAPE | `min Σ \|y − Xw\| / y` (mediana **ponderada por 1/y**) |
| sMAPE | `min Σ \|y − Xw\| / ((\|y\|+\|Xw\|)/2)` (no lineal; se aproxima iterando) |

Para MAPE la corrección es **exacta y de una línea**: ponderar cada observación por `1/y` en el programa lineal. Para sMAPE no hay forma cerrada porque el denominador depende de la propia predicción, así que se usa el esquema iterativo estándar (ajustar con pesos `2/(|y|+|pred anterior|)` y repetir hasta estabilizar).

**Resultado** (sobre el mejor conjunto actual, v4 + GARCH-ged; medias de 10 semillas):

| Objetivo del combinador | MAE | MAPE | sMAPE |
|---|---|---|---|
| MAE (el que se usaba) | 41.55 ± 0.168 | 10.904% ± 0.043 | 10.108% ± 0.036 |
| MAPE (pesos 1/y) | 41.75 ± 0.146 | **10.741%** ± 0.039 | 10.154% ± 0.037 |
| **sMAPE (iterativo)** | **41.29** ± 0.113 | **10.744%** ± 0.033 | **10.065%** ± 0.030 |

El objetivo MAPE se comporta como manda la teoría: **mejora la métrica que optimiza y empeora la otra** (gana 0.16 de MAPE a cambio de 0.20 de MAE). Pero el objetivo sMAPE **gana en las tres a la vez**, con signo negativo en **10/10 semillas** en cada una:

| Delta (objetivo sMAPE − objetivo MAE) | Valor | Semillas a favor |
|---|---|---|
| MAE | −0.263 ± 0.114 | 10/10 |
| MAPE | −0.161 ± 0.027 | 10/10 |
| sMAPE | −0.043 ± 0.020 | 10/10 |

**Ese "gana en todo" es sospechoso y se investigó**, porque el combinador por MAE debería ser por construcción el mejor en MAE. Dos comprobaciones:

1. **Control en muestra**: dentro del conjunto de entrenamiento el objetivo MAE **sí gana** en MAE (40.600 contra 40.767). El programa lineal está bien planteado; la inversión ocurre solo **fuera de muestra**, o sea es un efecto genuino de generalización, no un error de formulación.
2. **Hipótesis de regularización, confirmada**: la desviación media de los pesos entre ajustes es **0.1496** con objetivo sMAPE contra **0.1553** con objetivo MAE. La reponderación `2/(|y|+|pred|)` comprime el rango efectivo de la función de pérdida e impide que unos pocos días de precio alto dominen el ajuste, lo que **estabiliza los pesos y reduce el sobreajuste del meta-modelo**. Por eso generaliza mejor pese a ser subóptima en muestra.

Efecto adicional notable: el **MAPE en horas pico baja de 15.07% a 14.42-14.48%**, que es justo donde el proyecto tiene documentado el grueso del error.

**Estado acumulado del ensamble de 24h tras las dos mejoras de la noche** (ambas verificadas con 10/10 semillas):

| Versión | MAE | MAPE | sMAPE |
|---|---|---|---|
| v4 original | 42.51 | 11.27% | 10.38% |
| + GARCH-ged (colas pesadas) | 41.45 | 10.86% | 10.08% |
| **+ combinador alineado con la métrica** | **41.29** | **10.74%** | **10.07%** |

Mejora total: **−0.53 puntos de MAPE, −0.31 de sMAPE y −1.22 de MAE**, con el sMAPE ya por debajo de 10.1%.

**Lección para el informe**: las dos mejoras de la noche comparten forma — ninguna vino de más datos ni de modelos más potentes, sino de **corregir un supuesto estadístico mal puesto**. Primero el de la distribución del error dentro del modelo (colas pesadas), después el de la función de pérdida del meta-modelo (alinearla con la métrica que se reporta). Es un argumento fuerte a favor de revisar los supuestos antes de escalar la complejidad.

### 2026-09-17 (noche, cont. 7) — Barrido de la familia GARCH y dos ideas más: se llega a MAPE 10.86% / sMAPE 10.08%, y se atrapa un FALSO POSITIVO antes de que entrara al informe

Tras el éxito de LE-GARCH-t se exploró sistemáticamente la familia, más dos ideas derivadas del mismo mecanismo. Scripts: `familia_garch_24h.py`, `afinar_24h_final.py`, `ensamble_24h_final.py`.

**1) Barrido de la familia GARCH** (7 variantes × 24 horas, todas convergen 24/24):

| Variante suelta | MAE | MAPE | sMAPE |
|---|---|---|---|
| **GARCH-t** | **45.17** | **12.15%** | **11.10%** |
| GARCH-ged | 45.40 | 12.28% | 11.14% |
| EGARCH-t | 45.99 | 12.25% | 11.19% |
| GJR-t | 45.78 | 12.47% | 11.43% |
| GARCH-skewt | 47.79 | 13.49% | 13.23% |
| SV-t (aprox.) | 50.39 | 14.03% | 12.79% |
| EGARCH-skewt | 75.89 | 23.51% | 14.29% |

Tres lecturas útiles: **(a)** la asimetría **no** ayuda — skew-t empeora en todas sus versiones, y las volatilidades asimétricas (GJR, EGARCH) no aportan sobre GARCH simple, pese a que físicamente parecía plausible que los picos al alza generaran más turbulencia; **(b)** lo que importa es únicamente la **cola pesada**, y da igual por qué vía se obtenga (t y GED empatan, p=0.2393); **(c)** la aproximación de **SV-t no reproduce el éxito** que reporta el paper — es plausible que la aproximación por mínimos cuadrados ponderados iterados sea insuficiente y haga falta el MCMC que ellos usan, así que queda como línea abierta más que como refutación.

**2) Precios de contratos en la ecuación de media — FALSO POSITIVO atrapado.** Se descargaron de XM `PrecPromContRegu` y `PrecPromContNoRegu` (horarios 2019-2026, completos), equivalente colombiano de los *forward prices* que en el paper de Nueva Zelanda eran de las variables más usadas (60 de sus 423 features). En una primera prueba parecían mejorar el ensamble con p=0.0324. **Al rehacer la comparación con procedimiento de ajuste idéntico, el efecto desapareció** (MAPE 11.05%, p=0.1048 contra el v4). La causa: ese primer script reajustaba el LE-GARCH-t con una ruta de selección de alpha distinta —solo 4 valores en la mediana en vez de la ruta completa— con lo que el modelo de referencia quedaba peor ajustado (11.14% en vez de 10.89%) y cualquier variante parecía ganarle. **La "mejora" era un artefacto de comparar contra una referencia mal ajustada, no un efecto real.** Queda como aviso metodológico: al comparar variantes hay que garantizar procedimiento de ajuste idéntico, no solo mismos datos.

Nota relacionada: la variable derivada `prima_sobre_contrato` mostró correlación parcial 0.89, pero es un **artefacto** — se define como `precio/precio_contratos` y por tanto contiene el precio contemporáneo. En el modelo se usa medida en el corte (donde el precio ya es conocido), así que no hay fuga, pero ese 0.89 no es evidencia de utilidad.

**3) Combinador ponderado por riesgo — falla.** Se aplicó al combinador el mismo mecanismo que hizo funcionar a GARCH-t: ajustar los pesos del ensamble descontando las horas que el predictor de riesgo (AUC 0.818) señala como ruidosas. **Empeora claramente** (MAPE 11.75-11.76% contra 11.14%, p=0.0000 en los tres exponentes probados). La interpretación tiene sentido: esas horas son ruidosas para *predecir el precio*, pero siguen siendo informativas para decidir *en qué modelo confiar*; descontarlas destruye justamente la evidencia que el combinador necesita. El mecanismo que sirve dentro de un modelo no se traslada automáticamente al meta-modelo.

**Estado final del ensamble de 24h, verificado:**

| Conjunto | MAE | MAPE | sMAPE | pico | DM vs v4 |
|---|---|---|---|---|---|
| v4 vigente | 42.51 | 11.27% | 10.38% | 15.52% | — |
| v4 + GARCH-t | 41.51 | 10.89% | 10.09% | 15.12% | p=0.0075 |
| **v4 + GARCH-ged** | **41.45** | **10.86%** | **10.08%** | **15.07%** | **p=0.0040** |
| v4 + GARCH-t + GED | 41.49 | 10.88% | 10.08% | 15.07% | p=0.0068 |

**Verificación multi-semilla del mejor**: delta −1.26 ± 0.124, negativo en **10/10** semillas, ruido del vigente 0.145 → **SÓLIDO**. GARCH-t y GARCH-ged son estadísticamente indistinguibles entre sí (p=0.2393); se puede adoptar cualquiera de los dos, o ambos.

**Respecto a los objetivos**: el **sMAPE queda en 10.08%**, a ocho centésimas del 10%; el **MAPE en 10.86%**, lejos aún del 9.9% pedido. La mejora total acumulada sobre el v4 original es de **−0.41 puntos de MAPE y −0.30 de sMAPE**, toda ella atribuible a una sola idea: modelar las colas pesadas del error al estimar la media.

### 2026-09-17 (noche, cont. 6) — MEJORA REAL Y VERIFICADA EN 24h: LE-GARCH-t, el modelo ganador del paper de Nueva Zelanda, entra al ensamble. MAPE 11.27% → 10.89%, sMAPE 10.38% → 10.09%

Después de **trece líneas de ataque fallidas** al modelo de 24h, la que funcionó fue construir el modelo que el paper de Nueva Zelanda señalaba como ganador y que el proyecto nunca había hecho. Scripts: `le_garch_t_24h.py`, `verificar_le_garch_t.py`.

**Qué es LE-GARCH-t y por qué NO lo teníamos.** Kapoor & Wichitaksorn (2023) encuentran que **LE-GARCH-t es el mejor modelo en 4 de sus 5 regiones** por MAE y MASE, por encima de DNN, LSTM, GRU, XGBoost y del propio benchmark LEAR. "LE" = *LASSO-Estimated*: las exógenas de la ecuación de media se seleccionan con LASSO; "-t" = los errores siguen una t de Student. El proyecto tenía `ARX+GARCH` como votante, pero **le faltaban las dos piezas que definen al modelo**: la selección LASSO de exógenas y la distribución t. Y el hallazgo central de ese paper es precisamente que el GARCH **con todas** las variables es de los peores modelos y **con LASSO** pasa a ser el mejor (mejoras de 40-45%).

**Mecanismo por el que gana, pese a que la media es lineal en ambos**: LEAR estima por mínimos cuadrados penalizados, donde todos los días pesan igual; GARCH-t estima por máxima verosimilitud ponderando por la varianza condicional y con colas pesadas, lo que **descuenta automáticamente los días de alta volatilidad al estimar la media**. Es estimación robusta. Dado que el diagnóstico de esta misma sesión localizó el error en las horas y días volátiles, la hipótesis tenía fundamento — y se confirmó.

**Como modelo suelto** (un LASSO+GARCH por hora del día, mediana de 29 variables seleccionadas):

| Modelo | MAE | MAPE | MAPE pico | DM vs LEAR |
|---|---|---|---|---|
| LEAR (referencia) | 53.12 | 16.32% | 19.39% | — |
| LE-GARCH (normal) | 53.14 | 14.32% | 18.02% | p=0.9823 n.s. |
| **LE-GARCH-t** | **45.17** | **12.15%** | **15.59%** | **p=0.0000 MEJORA** |

**La distribución t es la pieza decisiva**: con errores normales el modelo no mejora nada (n.s.); con t mejora drásticamente. Eso aísla el mecanismo y confirma que lo que importa son las colas pesadas.

**En el ensamble de 24h:**

| Conjunto | MAE | MAPE | sMAPE | pico | DM vs v4 |
|---|---|---|---|---|---|
| v4 vigente (5 votantes) | 42.51 | 11.27% | 10.38% | 15.52% | — |
| **v4 + LE-GARCH-t (6)** | **41.46** | **10.89%** | **10.09%** | **15.18%** | **p=0.0044 MEJORA** |
| v4 + ambos GARCH (7) | 41.47 | 10.90% | 10.09% | 15.19% | p=0.0050 MEJORA |

**Verificación completa** (`verificar_le_garch_t.py`), con la vara que dejó el falso hallazgo del ONI:

1. **10 semillas de partición**: delta **−1.21 ± 0.127**, signo negativo en **10/10**, ruido del vigente 0.145 → el efecto supera 2× el ruido. **Veredicto automático: SÓLIDO.**
2. **Modo desplegable** (pesos causales, determinista, sin pliegues): MAE 45.57 → **44.24**, MAPE 11.42% → **11.00%**, sMAPE 10.53% → **10.20%**, **DM p=0.0021**. El efecto no depende del esquema de validación cruzada.
3. **Reparto por día**: gana en **63%** de los 196 días, mediana del delta diario −0.40. Mejora generalizada, no concentrada en unos pocos días.
4. **Chequeo de fuga**: por construcción LE-GARCH-t usa **exactamente las mismas variables** que el LEAR (todas medidas en el corte de las 00:00, mismo `construir_para_paso`); lo único que cambia es el método de estimación. Empíricamente, la correlación del pronóstico con el precio real es **0.9449**, prácticamente idéntica a la del LEAR (0.9476) — sin el salto hacia 1.0 que delató a `max_precio_oferta`.

**Sobre los objetivos del usuario**: con esto el sMAPE queda en **10.09%** (CV) / **10.20%** (desplegable) y el MAPE en **10.89%** / **11.00%**. El objetivo de 10% queda **alcanzado en sMAPE** y a una décima en MAPE — ya no como matiz de medición, sino como mejora real y verificada del modelo.

**Lección metodológica para el informe**: la mejora no vino de más datos (nueve variables nuevas descargadas de XM, ninguna ayudó), ni de más capacidad del modelo (los meta-modelos no lineales empeoraron), ni de arquitecturas de especialización (las tres variantes fallaron). Vino de **cambiar el supuesto distribucional del estimador** — reconocer que los errores del precio tienen colas pesadas y estimar en consecuencia. Es un argumento fuerte a favor de leer la literatura del campo y replicar sus modelos ganadores antes de inventar.

**Pendiente**: regenerar el contrato oficial de 24h con la nueva composición, junto con el de 72h. Ambos afectan el insumo del dashboard de Rafael, así que conviene coordinarlo con él.

### 2026-09-17 (noche, cont. 5) — Arquitectura de DOS MODELOS (uno para días normales, otro entrenado con días difíciles): falla incluso con oráculo, y eso explica por qué

Se implementó la arquitectura que pidió el usuario y que aún no se había probado (`dos_modelos_dia_24h.py`). Es distinta de lo ya intentado en dos cosas: **la unidad es el DÍA, no la hora**, y el especialista se **entrena** con días difíciles del histórico en vez de enrutar a un modelo generalista ya entrenado.

**Problema de diseño resuelto**: entrenar "con los días donde falla el ensamble" requeriría el error del ensamble en 2019-2025, pero el ensamble solo existe para 2026. Se definió la dificultad con criterios **observables y calculables igual en train y test** — (D1) volatilidad intradiaria del precio, (D2) salto respecto a la semana previa — y luego **se verificó que el criterio sí captura los fallos reales**:

| Criterio | Correlación con el MAPE diario real del ensamble en 2026 |
|---|---|
| **Volatilidad intradiaria** | **+0.476** |
| Salto semanal | +0.163 |

**Y el clasificador funciona**: predecir en el corte si el día será difícil da **AUC 0.832** (criterio de volatilidad). Es decir, las dos piezas previas del sistema están bien: el criterio es válido y el día difícil es anticipable.

**Pero la arquitectura falla igual**, con el criterio bueno (volatilidad):

| Esquema | MAE | MAPE | DM vs modelo único |
|---|---|---|---|
| Modelo único (referencia) | 53.15 | 16.34% | — |
| Enrutado por el clasificador | 53.05 | 15.76% | p=0.9184 n.s. |
| **Enrutado por ORÁCULO de dificultad** | 54.37 | 16.00% | **p=0.2573 n.s.** |
| Siempre el especialista | 76.50 | 22.03% | p=0.0000 **peor** |
| Siempre el general | 54.03 | 14.75% | p=0.4385 n.s. |

Con el criterio de salto todo es significativamente peor.

**La clave está en la fila del oráculo.** Al separar "el clasificador no acierta" de "el especialista no sirve", el experimento demuestra que es lo segundo: **incluso conociendo con certeza qué días son difíciles, enrutar no mejora**. Y "siempre el especialista" es catastrófico (76.50 vs 53.15).

**Explicación, que es lo valioso del resultado**: los días difíciles lo son *precisamente porque* son menos predecibles. Entrenar un modelo solo con ellos combina dos desventajas — menos datos (631 días contra 2.516) y peor relación señal-ruido en esos datos. El modelo general aprende la estructura estable a partir de **todos** los días y la aplica también a los difíciles, que es mejor que intentar aprender la estructura de lo impredecible. Es un caso de libro del compromiso sesgo-varianza: la especialización reduce sesgo en teoría, pero aquí el aumento de varianza lo supera con creces.

Con esto queda **cerrada con evidencia fuerte** la familia completa de "modelo especializado para los momentos de fallo", en sus tres variantes probadas: por hora del reloj (peor), por enrutamiento a modelos existentes (peor), y por especialista entrenado a nivel de día (n.s. incluso con oráculo).

### 2026-09-17 (noche, cont. 4) — Descargadas 3 métricas nuevas de XM que el proyecto nunca tuvo: pasan el tamiz de fuga, pero empeoran el modelo. Se replica un patrón metodológico importante

Se descargaron del API de XM (`descargar_estres_sistema.py`) tres métricas **horarias** que el proyecto nunca había bajado, elegidas por ser indicadores directos de estrés del sistema — el lado de la oferta que el diagnóstico señalaba como faltante. Las tres con cobertura **completa**: 66.624 horas, 2019-01-01 a 2026-08-07.

| Métrica | Qué es | Corr. contemporánea | **Corr. parcial** (desc. `precio_lag24h`) |
|---|---|---|---|
| `RestSinAliv` | Restricciones **sin** alivios (solo teníamos las aliviadas) | −0.328 | **−0.125** ✓ pasa |
| `GeneProgDesp` | Programa de despacho del día siguiente | +0.342 | **+0.117** ✓ pasa |
| `GeneFueraMerito` | Generación fuera de mérito (despacho forzado a salirse del orden de mérito) | −0.048 | −0.031 ✗ no pasa |

**Ninguna presenta fuga** (la fuga se ve como correlación contemporánea ~0.99, que es lo que delató a `max_precio_oferta`), y dos pasan el tamiz con correlación parcial del mismo orden que `costo_marginal_despacho` (0.178), el mejor candidato de la sesión previa.

**Tratamiento especial de `GeneProgDesp`**: XM publica el programa de despacho el día anterior, así que para un objetivo del día D con corte a las 00:00 de D, el programa de **todas** las horas de D ya está publicado. Eso lo convierte en información genuinamente **anticipativa**, a diferencia de casi todo lo demás del proyecto que es realizado. Por eso se probó en dos versiones separadas —valor en el corte (indiscutible) y valor en la hora objetivo (legítimo bajo el supuesto de publicación día-adelante)— para que la segunda quede explícitamente marcada y se pueda descartar si alguien objeta el supuesto.

**Resultado: empeoran.**

| Variante del LEAR24 | MAE | MAPE | MAPE pico | DM |
|---|---|---|---|---|
| Base | 53.15 | 16.34% | 19.39% | — |
| + estrés en el corte | 53.80 | 16.89% | 20.85% | p=0.0001 **empeora** |
| + estrés y programa en la hora objetivo | 54.64 | 17.81% | 22.87% | p=0.0000 **empeora** |

Y en el ensamble, las tres configuraciones dan **n.s.** (11.20%–11.23% contra 11.27% del v4). Ni siquiera la versión con información anticipativa del programa de despacho mueve la aguja.

**Conclusión metodológica, que es lo que vale para el informe**: este es ya el **tercer caso** en el proyecto de una variable con correlación parcial real que al incorporarla **empeora** el modelo (antes pasó con la relación hidro/térmica y con `costo_marginal_despacho` + `compras_arranque_parada` en el ataque a El Niño). El patrón es consistente y merece enunciarse como hallazgo: **la correlación parcial con el precio es un tamiz necesario pero muy insuficiente** — detecta que una variable contiene información no redundante con el precio rezagado, pero no dice si esa información es *aprovechable* por el modelo ni si sobrevive a la competencia con el resto de variables. En un conjunto ya rico, agregar una variable con señal marginal real suele costar más en varianza de estimación de lo que aporta en sesgo.

### 2026-09-17 (noche, cont. 3) — HALLAZGO ÚTIL: el fallo del ensamble de 24h SÍ es predecible en el corte (AUC 0.818), pero no es accionable para el pronóstico puntual — sirve como señal de confianza para OE3

Atacando la petición de "un modelo que se use específicamente en los momentos donde falla el nuestro", se cambió el planteamiento. La versión ingenua (especialistas para las horas 18-20) ya había fallado, pero identificaba el momento por la hora del reloj, que es un proxy grueso: no todas las tardes fallan y también hay fallos fuera del pico. La pregunta correcta es si se puede saber **en el momento del corte** que la predicción va a salir mal.

**Sí se puede, y bastante bien** (`predecir_fallo_24h.py`). Con un CatBoost que recibe solo el estado en el corte (17 variables + hora) y predice el error del ensamble:

| Método para anticipar el fallo | AUC |
|---|---|
| **Clasificador con estado del corte** | **0.818** |
| Heurística "es hora pico 18-20" | 0.568 |

De las 969 horas señaladas como más riesgosas, **49.4% estaban de verdad en el peor 20%** — 2.5 veces lo que daría el azar. La correlación entre error predicho y error real es 0.418. **El fallo es anticipable**, y mucho mejor de lo que lo anticipa la hora del reloj.

**Pero no se puede convertir en mejor pronóstico puntual** (`usar_riesgo_24h.py`). Se probó enrutar el 10% y el 20% de horas más riesgosas a cada modelo alterno disponible, y **todos son peores que el propio ensamble justamente en esas horas**:

| Enrutado a | MAPE en las horas enrutadas (10%) | El ensamble ahí |
|---|---|---|
| CatBoost | 23.99% | **23.81%** |
| LEAR24 | 25.97% | 23.81% |
| N-BEATSx | 26.02% | 23.81% |
| ARX+GARCH | 26.29% | 23.81% |
| MarkovSw | 27.06% | 23.81% |
| N-HiTS | 27.13% | 23.81% |

Las 12 combinaciones (6 modelos × 2 porcentajes) son significativamente peores. **El riesgo es predecible pero no hay a dónde enrutar**: en las horas difíciles el ensamble sigue siendo lo mejor que tenemos. Esto cierra formalmente la línea de "modelo especializado para los momentos de fallo" con evidencia, no con intuición: el obstáculo no es identificar el momento (eso ya se resolvió), es que no existe un modelo mejor para ese momento.

**Tampoco mejora las bandas de incertidumbre.** Se probó modular el semiancho de la banda proporcionalmente al riesgo predicho (conservando el ancho medio):

| Banda | Cobertura | Ancho medio | Interval score |
|---|---|---|---|
| **Conforme adaptativa vigente** | **79.5%** | **145.9** | **265.4** |
| Modulada por riesgo (exp. 0.5) | 71.4% | 190.2 | 280.9 |
| Modulada por riesgo (exp. 1.0) | 60.8% | 223.2 | 322.7 |

Empeora en todo, y de forma reveladora: la cobertura **cae** aunque el ancho medio suba, o sea que el riesgo predicho reparte mal el ancho — lo ensancha donde no hace falta y lo estrecha donde sí. **La calibración conforme adaptativa que ya tiene el proyecto captura el riesgo mejor que este clasificador**, lo cual es un punto a favor de la solución vigente.

**Dónde sí vale el hallazgo — nota para Rafael (OE3)**: aunque no baja el MAPE ni mejora las bandas, un AUC de 0.818 es una señal de confianza genuinamente informativa y **complementaria** a las dos que ya existen (ancho de banda y régimen hidrológico). Permite marcar en la vista Operador "esta hora tiene riesgo alto de error" *antes* de que ocurra, con 2.5x de precisión sobre el azar, sin depender de la hora del reloj. Los valores por hora quedan en `data/processed/resultados/riesgo_24h.csv` (columnas `fecha_hora`, `riesgo_predicho`, `error_real`). Es decisión suya si lo integra; el valor está en avisar cuándo desconfiar, no en corregir el número.

### 2026-09-17 (noche, cont. 2) — Recuperadas las variables de oferta ya descargadas y probadas de forma nueva: ayudan al modelo lineal suelto, pero no al ensamble

Buscando por el lado de los datos se encontró que **el repositorio ya tenía descargadas 12 variables del API de XM** (sesión del 2026-09-09) que nunca llegaron al modelo, entre ellas `disponibilidad_por_tipo`, `costo_marginal_despacho`, `precio_escasez` y `restricciones_aliviadas` — justo las del lado de la oferta que el diagnóstico señalaba como faltantes.

**Restricciones respetadas** (documentadas en esa sesión previa y verificadas ahora):
- **`max_precio_oferta` NO se usa**: el tamiz previo demostró que es **fuga pura** (correlación contemporánea 0.9998 — es literalmente el precio marginal).
- **No se desglosa por tipo de recurso**: el Alcance del Anexo 1 lo restringe. Aquí se usa únicamente la **suma total** de disponibilidad, que no constituye un desglose.

**Qué es nuevo respecto al tamiz previo.** Aquel tamiz evaluó las variables **crudas** por correlación parcial. Nunca se construyó el **margen de reserva = disponibilidad total − demanda**, que es la magnitud con sentido físico para la escasez: el precio no se dispara porque haya mucha o poca capacidad en absoluto, sino porque la disponible se acerca a la demanda — un efecto que vive en la *diferencia* y que una correlación sobre la variable cruda no puede ver. Además aquel tamiz midió sobre el MAE global o el de El Niño, nunca sobre las horas pico 18-20.

**Resultado en el modelo lineal suelto** (`margen_reserva_pico_24h.py`): agregar margen de reserva, margen relativo, costo marginal y cercanía al precio de escasez (todas medidas **en el corte**) mejora el LEAR de 24h de **MAPE 16.59% a 15.05%** (MAE 53.52 → 52.11), con **DM p=0.0011**. En el pico el MAPE baja de 19.45% a 18.51%, pero esa parte **no es significativa** (p=0.4227).

**Resultado en el ensamble** (`ensamble_24h_con_oferta.py`), que es lo que decide: **ninguna configuración mejora**.

| Conjunto | n | MAE | MAPE | sMAPE | MAPE pico | DM vs v4 |
|---|---|---|---|---|---|---|
| v4 vigente | 5 | 42.51 | 11.27% | 10.38% | 15.52% | — |
| v4 + LEAR24-oferta | 6 | 42.54 | 11.26% | 10.40% | 15.35% | p=0.8117 n.s. |
| v4 + LEAR24 + LEAR24-oferta | 7 | 42.44 | 11.23% | 10.37% | 15.34% | p=0.6845 n.s. |
| mejor5 (con LEAR24) | 5 | 42.30 | 11.21% | 10.37% | 15.03% | p=0.2575 n.s. |
| mejor5 con LEAR24-oferta | 5 | 42.37 | 11.23% | 10.39% | 15.09% | p=0.4470 n.s. |
| mejor5 + LEAR24-oferta | 6 | 42.31 | 11.21% | 10.36% | 15.08% | p=0.2943 n.s. |

**Lectura**: la información de oferta sí es real —mejora de forma significativa un modelo lineal que no la tenía— pero **ya estaba contenida, de forma implícita, en lo que aportan los otros votantes del ensamble**. Es el mismo patrón que el proyecto viene encontrando: variables con señal genuina que no sobreviven al pasar por un ensamble ya fuerte. Coincide además con el tamiz de correlación parcial de la sesión previa, que había clasificado 9 de 12 como redundantes.

Con esto son **nueve líneas de ataque cerradas** sobre el modelo de 24h (las siete de la entrada siguiente, más margen de reserva y variables de oferta en el ensamble).

**En curso**: descarga de tres métricas horarias del API de XM que el proyecto **nunca bajó** y que son indicadores directos de estrés del sistema (`descargar_estres_sistema.py`): `GeneFueraMerito` (generación fuera de mérito — se activa cuando el despacho debe salirse del orden de mérito por restricciones), `RestSinAliv` (restricciones **sin** alivios; el proyecto solo tenía las aliviadas) y `GeneProgDesp` (programa de despacho del día siguiente, que es información **conocida en el corte y genuinamente anticipativa**, a diferencia de casi todo lo demás que es realizado). Las tres se agregan a total del sistema, sin desglose por tipo de recurso.

### 2026-09-17 (noche, cont.) — Siete líneas de ataque más al modelo de 24h, todas cerradas: el límite es informacional, y los oráculos dicen exactamente dónde está el margen

Tras el primer bloque de intentos fallidos se siguió buscando, incluyendo por el lado de los datos. Resultado: **ninguna vía mejora el ensamble v4**, pero el diagnóstico con oráculos deja una conclusión mucho más precisa y defendible que un simple "no se pudo".

**Revisión del lado de los datos.** El `dataset_maestro` tiene solo **7 variables** (precio, demanda, generación, volumen de embalses, aportes hídricos, pandemia, ONI); las 47 features son todas derivadas de esas. **No hay datos sin usar escondidos en los archivos**: la limitación es de origen, no de aprovechamiento.

**Hueco real detectado y probado: la demanda de la hora objetivo** (`demanda_objetivo_24h.py`). El modelo conocía la demanda en el corte (00:00) pero **nunca una estimación de la demanda a las 19:00**, que es la hora que peor predice — siendo que en la literatura day-ahead el pronóstico de carga es *la* variable exógena estándar. Se construyó el pronosticador (Ridge directo, sin fuga): **MAPE de demanda 2.89%**, excelente, muy por debajo del ~11% del precio. Y aun así:

| Variante | MAE | MAPE | MAE pico | MAPE pico | DM |
|---|---|---|---|---|---|
| Base (sin demanda objetivo) | 53.18 | 16.36% | 102.81 | 19.40% | — |
| + demanda **REAL** objetivo (oráculo) | 54.55 | 17.25% | 107.60 | 22.38% | p=0.0007 **empeora** |
| + demanda pronosticada (desplegable) | 54.79 | 17.49% | 107.22 | 22.41% | p=0.0000 **empeora** |

**Ni con el oráculo mejora.** El diseño en dos etapas (medir primero el techo con información perfecta) evitó gastar esfuerzo en la versión honesta. La lectura de fondo es importante: **los picos de precio colombianos no son de demanda, son de oferta** — la demanda es regular y el calendario ya la captura; lo que dispara el precio es escasez del lado de la generación, que no observamos.

**Descartada también la extrapolación**: los precios de 2026 (máx. 1.279) caen enteramente dentro del rango de entrenamiento (máx. 2.676); solo 6.6% de las horas superan el percentil 95 histórico y **ninguna** supera el máximo. No es un problema de rango.

**Los techos con oráculo, que es el aporte metodológico de este bloque** (`techo_combinacion_24h.py`). Para saber si el problema era el combinador o los votantes, se midieron cotas superiores con información perfecta (no desplegables):

| Techo (información perfecta) | MAPE global | MAPE pico |
|---|---|---|
| Ensamble v4 actual | 11.27% | 15.52% |
| Pesos óptimos **por hora del día** | **11.38%** | 14.80% |
| Pesos óptimos **por día** | **8.82%** | 11.01% |
| Elegir el mejor votante cada hora | 5.26% | 6.23% |
| Mejor mezcla convexa hora a hora (límite absoluto) | 4.37% | 5.39% |

Dos lecturas decisivas:
1. **Los pesos fijos por hora del día están agotados**: el óptimo con información perfecta (11.38%) **no mejora** el 11.27% actual. No hay nada que ganar afinando por ahí, y eso explica por qué la granularidad de 1 a 24 grupos no dio nada.
2. **El margen existe pero es por día**: 8.82% contra 11.27%. La mezcla óptima de votantes **cambia día a día**. Además, el precio real cae **dentro** del rango de los votantes en 64.1% de las horas (62.0% en el pico), y la correlación media de errores es 0.78–0.83 — alta, pero no 1.0: hay diversidad explotable.

**Persecución de ese margen: dos familias, ambas fallidas.**
- **Agregación adaptativa de expertos** (`pesos_adaptativos_24h.py`): ventanas móviles de LAD (7, 14, 21, 30, 60, 90 días y expansiva), por franja y global, con 5 y con 8 votantes; más **Hedge** (promedio exponencialmente ponderado) con η ∈ {1, 5, 20, 50} y descuento γ ∈ {1.0, 0.95, 0.85}. **36 configuraciones: ninguna mejora significativamente.** La mejor (8 votantes, ventana 21 días por franja) da 11.81% con p=0.1568.
- **Compuerta por estado y meta-modelo no lineal** (`stacking_no_lineal_24h.py`): CatBoost y RandomForest como meta-modelo recibiendo votantes + 10 variables de estado del corte + hora; y pesos LAD por celdas de volatilidad reciente / nivel de precio / ratio de volatilidad cruzadas con franja. **Todas peores o no significativas**; los meta-modelos no lineales son consistentemente peores que el LAD lineal.

**Conclusión conjunta, que es lo que hay que llevar al informe**: el margen del 8.82% es **real pero no accionable**, porque el óptimo móvil por día **no es predecible** ni a partir del rendimiento pasado de los votantes (lo descarta la agregación adaptativa) ni a partir del estado observable en el corte (lo descarta la compuerta por estado). Saber qué votante acertará mañana requiere información que no tenemos.

**Balance de todo lo intentado sobre el modelo de 24h en esta sesión — siete líneas, todas cerradas con evidencia**: (1) votantes nuevos incluido LEAR24, (2) granularidad de pesos de 1 a 24 grupos, (3) especialistas de hora pico, (4) demanda de la hora objetivo incluso con oráculo, (5) corrección causal de sesgo, (6) pesos adaptativos y Hedge, (7) meta-modelos no lineales y compuerta por estado.

**Sobre los objetivos de 8% y 10%**: ninguno es alcanzable como mejora real del modelo con los datos actuales. Sí conviene señalar un matiz **de medición, no de modelo**: el proyecto ya reporta **sMAPE 10.38%** para el ensamble v4 de 24h, y el sMAPE es una métrica legítima —acotada, simétrica, menos distorsionada cuando el precio es bajo— y es la que reportan los papers de referencia. Es decir, *en la métrica que usa la literatura comparable ya estamos en 10.4%*. Debe presentarse explícitamente como métrica distinta, nunca como si el MAPE hubiera bajado.

### 2026-09-17 (noche) — El modelo de 24h está en su límite: el objetivo de 8% NO es alcanzable con la información disponible, y ahora sabemos exactamente por qué

Se repitió para 24h todo el programa que funcionó en 72h (LEAR, votantes nuevos, granularidad de pesos) más un ataque dirigido al punto débil. **Ninguna vía mejora el ensamble v4 de forma significativa.** Es un resultado negativo, pero con una explicación cuantificada que vale para el informe y para el capítulo de limitaciones.

**1) LEAR de 24h** (`lear_24h.py`). Aquí el LEAR es aún más canónico que en 72h: como el protocolo day-ahead usa corte a las 00:00, el paso `h` y la hora del día coinciden 1 a 1, así que "un LASSO por paso" **es** "un LASSO por hora del día", la receta original de la literatura. Resultado: MAE 53.64 / MAPE 16.72%. **Le gana a XGBoost (−7.75, p=0.0003) y a Ridge-dir24 (−5.88, p=0.0001), pero pierde contra N-BEATSx (+6.82, p<0.0001)** y contra el ensamble v4 (+11.14).

Esto **invierte** lo que pasó en 72h, y la explicación es la misma que el proyecto ya tenía: a 24h la dinámica intradiaria todavía manda y las redes la capturan; a 48-72h solo quedan fundamentales y estacionalidad semanal, y ahí lo lineal extrapola mejor. El LEAR no es mejor ni peor "en abstracto": **su ventaja depende del horizonte**, y eso es una conclusión defendible y bien respaldada, no un accidente.

**2) El ensamble de 24h está en meseta** (`mejorar_ensamble_24h.py`). Se probó todo lo recombinable, sobre predicciones ya guardadas:
- **Agregar cada uno de los 9 votantes extra disponibles** (CatBoost, Hurdle, Markov-Switching, RecursiveLS day-ahead, XGB/CatBoost-Optuna, los directos de 24h y el LEAR24 nuevo), de a uno: **ninguno da diferencia significativa**; dos son significativamente peores.
- **Selección greedy sobre los 14 votantes**: el mejor conjunto (N-BEATSx + MarkovSw + N-HiTS + LEAR24 + CatBoost) da MAE 42.30 contra 42.51 de la base — una ganancia de 0.21, **no significativa** (p=0.2575).
- **Granularidad de los pesos** (1, 2, 3, 4, 6, 8, 12 y 24 grupos; 24 = un peso por hora, como en la literatura day-ahead): el mejor es 24 grupos con 42.22 (−0.28), tampoco significativo (p=0.2947). Agrupar de menos (1 o 2 grupos) sí empeora significativamente, lo que confirma que las 4 franjas actuales ya capturan lo que hay que capturar.
- LEAR24 sí entra al mejor conjunto y recibe peso en las cuatro franjas (0.066 a 0.16), o sea aporta diversidad — pero el efecto no alcanza significancia.

**Conclusión de este bloque: el límite ya no está en cómo se combinan los modelos, sino en los modelos base.**

**3) Dónde vive el error** (`diagnostico_error_24h.py`). El error no está repartido:
- **Por hora**: 19:00 (MAPE 16.49%), 18:00 (15.72%) y 20:00 (14.34%) frente al 11.27% global. **Las 6 horas peores concentran el 37.8% del error total** (serían 25% si fuera parejo).
- **Por mes**: junio 2026 tiene MAPE 17.98% y **aporta el 22.2% de todo el error** del año; febrero, en cambio, 5.34%.
- **Cuánto margen hay**: para pasar de 11.27% a 8% hay que reducir el error un **29% relativo**. Aunque se acertara **perfecto** en las 6 horas peores, el MAPE global quedaría en **7.01%**; mejorándolas a la mitad, 9.14%. Es decir, el 8% está justo en el borde de lo teóricamente posible y **solo** atacando el pico vespertino.

**4) Ataque dirigido al pico, y por qué falla** (`especialista_pico_24h.py`). Antes de modelar se midió si ese error es corregible:
- **Descomposición sesgo/varianza en las horas 18-20**: sesgo medio +19.5 COP/kWh contra una desviación del error de 166.4. **El sesgo explica solo el 21.2% del MAE del pico; el 78.8% restante es varianza.**
- **Corrección de sesgo por hora, causal** (medias móviles de 7/14/30/60 días, solo con días anteriores): empeora significativamente con 7, 14 y 30 días; con 60 días no hay diferencia. Ninguna mejora.
- **Especialistas entrenados solo con horas pico** (LASSO y CatBoost dedicados): **son peores que el ensamble general** en esas mismas horas (MAE 103.99 y 106.22 contra 92.15 del v4). Sustituir empeora significativamente; mezclar 50/50 no da diferencia.

**Interpretación, que es lo importante para el informe**: el error residual del pico vespertino **no es error de modelado, es varianza genuina** — picos de precio que no están determinados por la información que tenemos en el momento del corte. Por eso ningún modelo, lineal o no lineal, general o especializado, logra reducirlo: no es que el algoritmo sea insuficiente, es que **la información disponible al corte no contiene la respuesta**.

**Qué haría falta de verdad para acercarse al 8%**: información nueva, no algoritmos nuevos. Candidatos naturales en el mercado colombiano serían datos de ofertas por unidad de generación, indisponibilidades programadas y declaraciones de mantenimiento — variables que sí anticipan los picos de escasez y que hoy no están en el conjunto de datos. Eso es una línea de trabajo futuro, no un ajuste del modelo actual.

**El objetivo de 8% queda documentado como no alcanzable con los datos actuales**, con la evidencia que lo sustenta. El ensamble v4 de 24h se mantiene sin cambios: MAE 42.51, MAPE 11.27%, sMAPE 10.38%, R² 0.910.

### 2026-09-17 (noche) — MEJORA REAL Y VERIFICADA: un LEAR (LASSO por paso) entra al ensamble de 72h, saca a los tres árboles, y mejora los tres tramos — modelo más simple y más preciso a la vez

Es el mejor resultado del proyecto en varias sesiones, y viene directamente de hacerle caso a la literatura. Scripts: `lear_72h.py`, `ensamble_72h_con_lear.py`, `verificar_ensamble_lear.py`.

**Qué se construyó.** Un **LEAR** (LASSO Estimated AutoRegressive), el modelo de referencia del campo según Lago, Marcjasz, De Schutter & Weron (2021), que en Nord Pool, PJM y EPEX resulta muy difícil de superar. Nuestro `Ridge-directo` **no era** un LEAR; le faltaban las tres cosas que lo definen:
1. **Estructura rica de rezagos**: precio en el corte y en 14 rezagos hacia atrás (`t−h−k` para k = 0,1,2,3,6,12,18,24,48,72,96,120,144,168), más "misma hora de días previos" validados contra el corte, más mín/máx/media de las 24h anteriores al corte.
2. **Estimación por LASSO (L1)**, no Ridge (L2): L1 pone coeficientes exactamente en cero y selecciona; L2 solo encoge.
3. **Un modelo por paso de horizonte** (72 modelos LASSO, uno por h), en vez de un modelo único con el paso como variable — que es como se hace en la literatura de day-ahead.
Anti-fuga: para el objetivo `t` con paso `h`, el corte es `c = t−h` y solo se usa información disponible en `c`; los "misma hora" se incluyen solo si el instante cae en `≤ c`. Alpha por validación cruzada; mediana de 36 variables activas por modelo.

**LEAR resulta ser el mejor modelo INDIVIDUAL del proyecto**, por encima de las dos redes neuronales:

| Modelo individual | 1-24h | 25-48h | 49-72h | global |
|---|---|---|---|---|
| **LEAR** | 51.08 | 67.59 | **76.66** | **65.11** |
| N-BEATSx | 49.41 | 70.97 | 85.99 | 68.79 |
| N-BEATSx exog | 51.38 | 70.04 | 83.57 | 68.33 |
| Ridge-directo | 61.01 | 71.99 | 82.13 | 71.71 |
| CatBoost directo | 63.96 | 74.73 | 82.85 | 73.84 |
| XGBoost directo | 64.83 | 76.23 | 85.28 | 75.44 |

A 49-72h **LEAR le gana a N-BEATSx con significancia** (−9.34, p=0.0225) y queda **estadísticamente empatado con el ensamble completo de 7 votantes** (+1.97, p=0.26). Es exactamente lo que predice la literatura: a horizonte largo, cuando solo quedan fundamentales y estacionalidad semanal, un lineal bien regularizado compite de tú a tú con el deep learning.

**El ensamble nuevo es a la vez MÁS SIMPLE y MEJOR.** El mejor conjunto es **`LEAR + N-BEATSx + N-BEATSx-exog + naive estacional` (4 votantes)**, que elimina **los tres modelos de árboles** (XGBoost, CatBoost directo, CatBoost denso) y aun así gana en los tres tramos y en los dos modos:

| Tramo | Vigente (7 votantes) | Nuevo (4 votantes) | Mejora | DM |
|---|---|---|---|---|
| 1-24h | 50.26 | **48.60** | −1.66 (−3.3%) | p=0.0000 |
| 25-48h | 67.13 | **65.56** | −1.57 (−2.3%) | p=0.0312 |
| 49-72h | 79.34 | **76.19** | −3.14 (−4.0%) | p=0.0050 |

(modo desplegable, que es el que se despliega de verdad)

**Cuadro completo de métricas** (`estadisticas_completas_lear.py`, salida `estadisticas_completas_lear.csv`), modo desplegable — mismas métricas que se usaron para comparar contra los 4 papers, así ambas comparaciones (contra la literatura y entre nuestros propios modelos) quedan con el mismo criterio:

| Tramo | Ensamble | MAE | RMSE | MAPE | sMAPE | R² |
|---|---|---|---|---|---|---|
| 1-24h | Vigente (7) | 50.26 | 92.58 | 13.70% | 12.72% | 0.896 |
| 1-24h | **Nuevo (4)** | **48.60** | **90.70** | **13.16%** | **12.17%** | **0.900** |
| 25-48h | Vigente (7) | 67.13 | 112.56 | 19.03% | 17.44% | 0.848 |
| 25-48h | **Nuevo (4)** | **65.56** | **110.51** | **18.61%** | **16.84%** | **0.854** |
| 49-72h | Vigente (7) | 79.34 | 127.67 | 22.17% | 20.49% | 0.808 |
| 49-72h | **Nuevo (4)** | **76.19** | **124.56** | **21.53%** | **19.53%** | **0.817** |
| **Global 72h** | Vigente (7) | 65.53 | 111.81 | 18.28% | 16.87% | 0.850 |
| **Global 72h** | **Nuevo (4)** | **63.41** | **109.42** | **17.75%** | **16.17%** | **0.857** |

El nuevo ensamble mejora las **cinco** métricas en los **tres** tramos, sin excepción — no es una mejora que se vea bien en una métrica y mal en otra.

**Verificación, siguiendo la regla que dejó el falso hallazgo del ONI** (`verificar_ensamble_lear.py`). Los deltas (1.2–3.1) caen en el rango que el proyecto declaró no interpretable con una sola configuración, así que se verificó a fondo:
- **10 semillas de partición en CV**: el signo es negativo en **10/10** para los tres tramos, y el efecto supera holgadamente el ruido — la desviación entre semillas es de solo **0.138 COP/kWh** frente a efectos de 1.18 a 2.15. Veredicto automático del script: SÓLIDO en los tres tramos.
- **Modo desplegable**: es determinista (ventana causal, sin pliegues), y ahí los tres tramos salen significativos por su cuenta.
- **No viene de unos pocos días raros**: el ensamble nuevo gana en 65% / 58% / 55% de los ~200 días, con mediana del delta diario negativa en los tres tramos.

Matiz honesto que conviene decir en el informe: ganar en 55-65% de los días es una ventaja **consistente pero moderada**, no aplastante. Lo que la hace defendible no es el tamaño sino la consistencia: mismo signo en 10/10 semillas, en los dos modos, en los tres tramos y en la mayoría de los días.

**Por qué importa más allá del número.** El ensamble nuevo elimina los tres modelos de árboles, que eran de lo más costoso de entrenar y mantener. El sistema queda con un lineal (LEAR, entrenable en 8 minutos para los 72 pasos), dos redes y un naive. Es más barato, más reproducible y mucho más fácil de defender.

**Lo que NO se puede quitar son las redes neuronales** (`simplificar_ensamble_72h.py`): quitarlas del ensamble vigente cuesta **+4.54 a +11.74** de MAE, todo significativo. La selección greedy lo confirma: `Ridge → +NBX → +NBX_exog` llega a 75.10 de los 74.69 finales, y los tres árboles juntos solo aportan 0.44 más. Es decir, **la simplificación posible era quitar los árboles, no las redes** — y con LEAR dentro, quitarlos además mejora.

**Pendiente de decisión**: regenerar el contrato oficial (`pronostico_ensamble_72h_diario_2026.csv` y el contrato unificado para OE3) con la nueva composición. No se hizo todavía porque ese archivo es la interfaz con el dashboard de Rafael y conviene avisarle antes de cambiarle el insumo.

### 2026-09-17 (noche) — Ejecutadas las tres mejoras pendientes al horizonte de 72h: una falla, una es peligrosa, una mejora poco pero de forma consistente

Se ejecutaron las tres ideas identificadas al leer los 4 papers. Resumen: **ninguna cambia materialmente el modelo desplegado**, pero dos dejan lecciones que valen para el informe.

**1) QRA en vez de LAD como combinador a 72h — NO funciona** (`scripts_experimento/qra_72h.py`, salida `qra_72h.csv`). La hipótesis era razonable: QRA tiene intercepto libre, que puede absorber el sesgo de nivel que el proyecto ya documentó, y ese sesgo debería crecer con el horizonte, así que QRA debería ayudar más a 72h que a 24h. **Refutada: 0 de 12 combinaciones (3 tramos × 2 conjuntos × 2 modos) dan diferencia significativa.** Delta medio +0.19 COP/kWh, o sea ligeramente peor. Patrón informativo: en modo CV, QRA va algo mejor a 25-48h (−0.70, −0.64, no significativo), pero en modo desplegable va consistentemente peor a 49-72h (+1.70, +1.57). Tiene sentido: QRA estima más parámetros libres (intercepto + pesos sin restricción de suma), y en el modo causal —con menos datos de entrenamiento por corte— sobreajusta. **Conclusión: LAD se queda como combinador a 72h**, y la razón por la que QRA ganó a 24h no se transfiere al horizonte largo.

**2) Poda por multicolinealidad (Dias et al.) — NO funciona, y es PELIGROSA si se implementa literal.** El paper dice "se selecciona una variable de cada par con |Pearson| > 0.8" pero **no dice cuál de las dos**. Implementado tal cual, descartando la que aparece después en la lista, el resultado es catastrófico: **MAE +25.96 / +21.40 / +19.33 por tramo (p<0.001 en los tres)**, porque elimina `precio_lag168h` y `precio_mismo_hora_disp`, que son justo las dos variables de historia de precio más predictivas. Conservando en cada par la más correlacionada con el objetivo, el daño desaparece pero tampoco hay ganancia (+1.03 / +1.23 a horizonte largo). **Lección para el informe: una receta de un paper puede ser inaplicable si se copia sin entender qué elimina; aquí la diferencia entre dos implementaciones legítimas de la misma frase es de 26 COP/kWh de MAE.**

**3) Selección agresiva de variables tipo LASSO (Kapoor & Wichitaksorn) — mejora, pero poco y con un matiz importante** (`seleccion_variables_ridge_72h.py`, `ensamble_72h_ridge_reducido.py`).

Sobre **Ridge-directo aislado**, evaluado en todos los cortes, la reducción sí mejora y el hallazgo del paper neozelandés se reproduce:

| Conjunto de variables | #var | 1-24h | 25-48h | 49-72h | global |
|---|---|---|---|---|---|
| todas (línea base) | 37 | 64.18 | 75.02 | 85.02 | 74.70 |
| **información mutua top-20** | 20 | 61.19 | 73.33 | 84.37 | **72.93** |
| LASSO agresivo (α×50) | 6 | 60.82 | 74.38 | 85.60 | 73.56 |

La variante de información mutua top-20 mejora los **tres** tramos con significancia (−2.99 / −1.69 / −0.64, p<0.05). Con solo 6 variables (LASSO α×50) se mejora mucho a 1-24h (−3.36) pero ya no a 49-72h. Confirma lo que dice la literatura: para el modelo lineal, **menos variables bien elegidas superan a más variables**.

**Pero la prueba que importa es dentro del ensamble**, y ahí el efecto casi desaparece. Sustituyendo el votante Ridge-directo por Ridge-MI20 en los 214 cortes diarios: **los 12 deltas son negativos (mejoran), pero solo 2 son significativos**, ambos en el tramo 1-24h (−0.47 p=0.003; −0.57 p=0.040). A 25-48h y 49-72h las mejoras (−0.26 a −0.66) no alcanzan significancia.

Dos matices que hay que decir con honestidad:
- **El tramo 1-24h, que es donde sí mejora, es justamente el que el contrato unificado reemplaza por el especialista de 24h (v4).** O sea, la única mejora significativa cae en el tramo que el sistema desplegado no usa del ensamble de 72h.
- **La ganancia no viene de que Ridge-MI20 sea mejor modelo.** En la rejilla de cortes diarios, aislado, Ridge-MI20 es *peor* que el original (74.09 vs 71.71) — al revés que en la rejilla completa. Lo que aporta es **diversidad**: al usar otro subconjunto de variables se decorrelaciona de los demás votantes y el combinador le saca provecho. Es teoría clásica de ensambles (la diversidad importa más que la precisión individual), y explica por qué los 12 signos van en la misma dirección aunque las magnitudes sean pequeñas.

Nota estadística: que 12 de 12 pruebas apunten en la misma dirección sugiere un efecto real, pero **no se le puede aplicar una prueba de signos** porque las 12 no son independientes (comparten datos, y los conjuntos de votantes y modos se solapan). Se deja como indicio consistente, no como resultado significativo.

**Decisión**: no se cambia el modelo desplegado con base en esto. La mejora es real en dirección pero marginal en magnitud y cae en el tramo que el contrato no usa. Queda documentado como línea explorada con evidencia, que es lo que corresponde para el informe.

### 2026-09-17 (noche) — Discusión final para el informe: horizontes reales de los 4 papers, "simple le gana a complejo" en la literatura, y composición del ensamble de 72h

**Horizontes reales de los 4 papers, en horas** (importante aclarar esto en el informe para no dar a entender que la comparación es horizonte-a-horizonte):

| Paper | Qué predicen | Horizonte en horas | Resolución |
|---|---|---|---|
| Nueva Zelanda (Kapoor & Wichitaksorn 2023) | 1 precio por día | 24h adelante | Diaria (1 número/día, no 24) |
| Brasil — Dias, Lira & Freire 2024 (MLP) | 1 precio por semana, hasta 4 semanas | 168–672h adelante | Semanal |
| Brasil — Albani et al. 2025 (forward) | 1 precio por día, hasta 30 días | 24–720h adelante | Diaria |
| Brasil — Nunes Jr. et al. 2024 (caudales→SPD) | Caudal 1–7 días adelante, que alimenta el precio oficial de esos días | 24–168h adelante | Llega a horaria vía el modelo oficial DESSEM, pero lo que ellos miden es el caudal, no el precio directamente |

**Ninguno predice a resolución horaria como nosotros** (24 precios distintos para el día siguiente). Predecir un agregado diario/semanal/mensual es una tarea estructuralmente más fácil porque el ruido intradiario se cancela al promediar. Conclusión para el informe: hay que decir *"superamos su propia métrica normalizada (MASE/NRMSE/TAPI) en tareas de horizonte más agregado y por tanto más fácil"*, nunca *"le ganamos a Nueva Zelanda"* sin ese matiz.

**Hallazgo de literatura que conecta con decisiones ya tomadas en el proyecto**: los 4 papers, de forma consistente, encuentran que **modelos más simples con buena selección/regularización de variables igualan o superan al deep learning complejo**:
- NZ: GARCH-t y SV-t con selección LASSO (LE-GARCH-t, LE-SV-t) le ganan a DNN, LSTM, GRU y XGBoost en las 5 regiones. Citan explícitamente a Lago et al. (2018, 2021): *"simpler base models such as LEAR, DNN, and LSTM, can outperform newer, complex models"*.
- Brasil (Dias et al.): un MLP simple con selección por Pearson + información mutua le gana al modelo oficial DECOMP, mucho más sofisticado (programación dinámica estocástica dual).
- Brasil (Albani et al.): citan a Hewamalage et al. (2023) para justificar por qué usan redes de una sola capa oculta a propósito: *"the gain provided by sophisticated NN models tends to be limited and may not be statistically significant"*.
- Brasil (Nunes Jr. et al.): sofisticación en la función de pérdida (correntropía), no en la arquitectura — MLP de tres capas.

Esto coincide con el benchmark más citado del campo (Lago, Marcjasz, De Schutter & Weron 2021, *Applied Energy* 293:116983 — no leído completo en esta sesión, referenciado dentro del paper de NZ): en los mercados más estudiados del mundo (Nord Pool, PJM, EPEX), el benchmark LEAR (autoregresivo con LASSO) es muy difícil de superar, y el método de referencia para combinar modelos es QRA (Nowotarski & Weron 2015).

**Punto fuerte para el informe**: dos decisiones que tomamos por evidencia empírica propia, sin conocer aún esta literatura, coinciden exactamente con lo que domina en los mercados mejor estudiados:
1. **Ridge (lineal) es nuestro mejor modelo individual a 48–72h** — descubierto por prueba propia ("los árboles no extrapolan, lo lineal sí"), coincide con el dominio de LEAR/modelos lineales regularizados en Nord Pool.
2. **Nuestro mejor ensamble de 24h usa QRA por franja horaria** — es la misma técnica (Nowotarski & Weron 2015) que domina el mercado nórdico. La adoptamos porque dio mejor MAE, no por conocerla de antemano.
Esto refuerza la credibilidad del trabajo: no se copió una receta, la evidencia empírica llevó al mismo lugar que la literatura del campo.

**Composición actual del ensamble de 72h** (`scripts_experimento/ensamble_72h_diario.py`), para que quede registrado antes del informe: LAD (mínima desviación absoluta) por tramo de horizonte (1-24h / 25-48h / 49-72h), con 7 votantes en la versión vigente — N-BEATSx base, N-BEATSx con exógenas enriquecidas, XGBoost directo, CatBoost directo, CatBoost denso, Ridge directo, y naive estacional — más una variante "plus" que agrega N-BEATSx con semillas adicionales y con ventana de contexto de 336h. Pesos actualizados cada 7 cortes (semanal) en la versión desplegable. MAE actual por tramo: **47.91 (1-24h) / 64.00 (25-48h) / 74.69 (49-72h)**.

**Posibilidad de mejora identificada y NO probada todavía**: el punto de predicción (q50) del ensamble de 72h usa **LAD**, nunca se probó **QRA** ahí — y en 24h QRA sí superó a LAD (MAE 42.51 vs 43.24). Es la mejora más barata disponible: no requiere reentrenar ningún votante, solo cambiar cómo se combinan sus predicciones ya guardadas, igual que se hizo con `qra_por_regimen.py`. Candidata para probar en la próxima sesión.

### 2026-09-17 — Probada la idea 1 de Albani et al. (pesos del ensamble por régimen hidrológico): NO funciona, y aparece una corrección importante para el dashboard de Rafael

Se implementó la primera de las cuatro ideas nuevas (script: `scripts_experimento/qra_por_regimen.py`, salida: `qra_por_regimen.csv`). La idea de Albani et al. (2025) es que los pesos del ensamble dependan del régimen hidrológico, no solo de la franja horaria. Era la candidata más barata porque no requiere reentrenar: las predicciones de los cinco votantes ya estaban guardadas y `validar_qra_24h.qra()` ya recibía la columna de agrupamiento como parámetro, así que solo había que cambiar el agrupamiento.

Se probaron cuatro variantes, con umbrales **sin fuga de información** (calculados solo con 2019–2025, no con el periodo evaluado — la estratificación descriptiva anterior usaba la mediana del propio 2026, lo que sirve para describir pero no para ponderar):

| Agrupamiento de los pesos | Celdas | MAE | Δ vs. base | DM |
|---|---|---|---|---|
| **Por franja horaria (v4 vigente)** | 4 | **42.506** | — | — |
| Por régimen absoluto | 2 | 44.402 | **+1.896** | p=0.0000 |
| Por franja × régimen absoluto | 8 | 42.690 | +0.184 | p=0.0215 |
| Por régimen relativo (mediana móvil causal 90d) | 3 | 44.405 | +1.899 | p=0.0000 |
| Por franja × régimen relativo | 9 | 42.719 | +0.213 | p=0.0394 |

**Las cuatro variantes empeoran, y todas con significancia estadística.** Agrupar por régimen *en lugar* de por franja es claramente peor (+1.9 COP/kWh): la franja horaria es la variable que de verdad cambia qué votante conviene. Agrupar por franja *y además* por régimen empeora poco (+0.18) pero de forma consistente, que es el patrón típico de sobreajuste de los pesos: al partir en 8–9 celdas quedan ~600 horas por celda para estimar 6 parámetros del QRA, y el ruido de estimación supera lo que se gana por especializar.

**Conclusión acumulada sobre la hidrología, ahora probada por tres vías distintas**: como variables predictoras no baja el MAE (p=0.52); en ablación LOGO multi-semilla queda dentro del ruido; y ahora, como criterio para condicionar los pesos del ensamble, empeora significativamente. Sigue sirviendo **solo** para estratificar la confianza *a posteriori*. Coincide con lo que encontraron Nunes Jr. et al. en Brasil por un camino totalmente distinto.

**Corrección importante para Rafael (rectifica la nota del 2026-09-17 de abajo)**: al aplicar umbrales históricos absolutos, **2026 no tiene NI UNA hora en régimen de "embalse bajo"** — las 5.184 horas caen en `emb_alto`. El rango dramático de error que se le reportó (28,5 → 97,1 COP/kWh entre celdas) venía de cortar por la mediana *del propio 2026*, así que "embalse bajo" ahí significaba "bajo respecto a 2026", no "bajo históricamente": en términos absolutos 2026 fue un año húmedo de principio a fin. **Implicación práctica para el dashboard**: si la señal de confianza por régimen se implementa con un umbral histórico fijo, **nunca se va a activar** en los datos actuales y la vista Operador mostrará siempre "régimen favorable". Hay que implementarla con umbral **relativo móvil** (por ejemplo la mediana de los últimos 90 días, como en la columna `regimen_rel` del script), que sí distingue 3.424 horas de embalse bajo relativo frente a 1.759 de embalse alto relativo.

### 2026-09-17 — Los 4 papers bloqueados, leídos completos: comparación válida (MASE, NRMSE, TAPI) y corrección de una comparación anterior mal hecha

El usuario consiguió los PDF completos de los cuatro papers que estaban bloqueados (MDPI por detección de bot; Springer y ScienceDirect por muro de pago). Leídos íntegros, obligan a **corregir cómo veníamos comparando** y aportan cuatro ideas nuevas todavía sin probar.

**Tres trampas de comparabilidad que invalidan la comparación directa de números** (script: `scripts_experimento/comparacion_4_papers.py`, salida: `data/processed/resultados/comparacion_4_papers.csv`):

1. **El sMAPE no está definido igual en todas partes.** Kapoor & Wichitaksorn (2023) usan `sMAPE = (100/T)·Σ|y−ŷ|/(|y|+|ŷ|)`, **sin el `/2`** en el denominador. Nuestra definición (la habitual) sí lo lleva, así que **su sMAPE es exactamente la mitad del nuestro sobre los mismos datos**. En su convención, nuestro 10.38% se escribe 5.19%. Cualquier comparación de sMAPE entre papers que no verifique la fórmula está mal por un factor de 2 — **esto obliga a poner una advertencia sobre la comparación contra el paper noruego de la entrada siguiente**, que se hizo sin verificar qué convención usaba.
2. **Sus MAE/RMSE/sMAPE están sobre la serie transformada, no sobre precios reales.** Su Tabla 2 declara Box-Cox + min-max a [0,1] para el precio, y nunca dicen que inviertan la transformación antes de medir: un MAE de 0.3793 en una serie cuya media real es 107 NZD/MWh solo tiene sentido en escala transformada. **Su única métrica comparable con nosotros es la MASE**, que es adimensional por construcción.
3. **Los horizontes y frecuencias son distintos**: Nueva Zelanda pronostica precio *diario* a 1 día; Dias et al. pronostican PLD *semanal* a 4 semanas; Albani et al. pronostican *forward mensual* a 30 días. Comparar su índice de tendencia (medido paso a paso semanal o mensual) contra un TAPI horario nuestro sería tramposo, así que el script reporta el TAPI a tres agregaciones (hora, día, semana).

**Comparación ya válida, cada paper con SU métrica:**

| Referencia | Su métrica | Ellos | Nosotros (ensamble 24h) |
|---|---|---|---|
| Kapoor & Wichitaksorn 2023 (Nueva Zelanda) | MASE | 1.263–1.337 los mejores (LE-GARCH-t, LEAR); hasta 2.51 los peores | **0.943** |
| Dias, Lira & Freire 2024 (Brasil, MLP) | MAPE / NRMSE / TAPI | 14.65% / 24.70% / 68.75% | **11.27% / 21.66% / 75.9%** (TAPI diario) |
| Dias et al. — DECOMP, modelo **oficial** de Brasil | MAPE / NRMSE / TAPI | 19.44% / 53.89% / 56.25% | ídem arriba |
| Albani et al. 2025 (Brasil, forward) | Acierto direccional | ~50% general; ~60% solo con \|corr(ENA,precio)\|≥0.5 | **75.9%** a paso diario |

**El resultado más fuerte es la MASE**: MASE < 1 significa ganarle al pronóstico ingenuo de un paso que se usa como escala. Nuestro ensamble da **0.943**, mientras que **los 34 modelos del paper neozelandés tienen MASE > 1.26**, es decir, ninguno le gana a su propio ingenuo. Advertencia honesta: el denominador de cada MASE es el ingenuo de *su* serie, así que no es una competencia cabeza a cabeza — el mercado colombiano puede ser sencillamente más predecible que el neozelandés (que es en tiempo real, se liquida cada 30 minutos y tiene una curtosis de 505 en una de sus regiones). Lo que sí se puede afirmar es que **nuestro modelo supera su propia referencia ingenua y los de ellos no**.

**Bug encontrado y corregido durante este cálculo**: la primera versión del TAPI daba 40% (por debajo del azar) para todos nuestros modelos, y por poco se reporta como el problema de "forma del día". Era un artefacto: contaba como fallo las horas en que el precio real no cambió, porque `sign(0)` nunca coincide con `+1` ni con `−1`. Excluyendo los pasos sin cambio —que es lo que mide de verdad un índice de tendencia— el TAPI horario sube a 82.2%. Queda como otro caso de la misma lección del proyecto: **verificar todo hallazgo que sorprenda antes de reportarlo**.

**Confirmación externa de uno de nuestros resultados negativos**: Nunes Jr., Ferreira & Pinho (2024) mejoraron el pronóstico de caudales en Brasil (su ANN-MCC gana a la ANN-MSE y empata o supera al modelo oficial en 99 de 144 plantas al día 1) y al meter esas mejores predicciones en la cadena oficial de precios encontraron que los precios **casi no se movieron**: *"the boxplots are practically the same"*. Es exactamente nuestro hallazgo de que las variables de memoria hidrológica no bajan el MAE (p=0.52). Dos equipos, dos países, mismo resultado: **conviene dejar de invertir esfuerzo en esa línea**, y el paper sirve para respaldarlo en el documento final en vez de presentarlo como una limitación nuestra.

**Cuatro ideas que estos papers traen y que NO hemos probado** (ordenadas por valor esperado sobre costo):

1. **Compuerta por régimen hidrológico.** Albani et al. observan que su modelo se degrada justo cuando se rompe la correlación ENA–precio, y proponen en su discusión *apagar* el componente hidrológico cuando `|corr|` cae por debajo de ~0.5. Nosotros ya tenemos las celdas de régimen (embalse×ONI) y sabemos que el error se triplica entre ellas, **pero solo las usamos como etiqueta de confianza, nunca como interruptor del modelo ni como condicionante de los pesos del ensamble**. Los pesos actuales del LAD/QRA se calculan por franja horaria; por régimen no se ha intentado. Es barato: las predicciones de todos los votantes ya están guardadas, solo hay que re-ponderar.
2. **Selección de variables agresiva para los modelos estadísticos (LE-GARCH-t / LE-SV-t).** Es el hallazgo central del paper neozelandés y va en contra de lo que veníamos haciendo: sus GARCH/SV **con todas las variables son los PEORES modelos del estudio**, y **con variables filtradas por LASSO pasan a ser los MEJORES**, por encima de LEAR, DNN, LSTM, GRU y XGBoost, con mejoras de hasta 40–45%. Nosotros hemos estado *agregando* variables (memoria hidrológica, armónicos, etc.); nunca hicimos una reducción agresiva sobre el ARX+GARCH, que es justamente nuestro modelo estadístico flojo. Además nunca hemos probado **volatilidad estocástica (SV)**, que en su tabla queda 2º o 3º en las cinco regiones.
3. **Criterio de máxima correntropía (MCC) como función de pérdida** (Nunes Jr. et al.). Es una pérdida basada en un kernel gaussiano sobre el error, diseñada explícitamente para objetivos no gaussianos con outliers grandes — que es exactamente el precio de bolsa. Ellos la programaron como función de pérdida propia en Keras y les ganó a MSE en la mayoría de embalses. Nosotros usamos pérdida MAE en CatBoost y nunca probamos una pérdida robusta de este tipo.
4. **Poda por multicolinealidad** (Dias et al.): eliminar uno de cada par de variables con \|Pearson\| > 0.8 y rankear el resto por información mutua. Tenemos muchas variables construidas unas de otras (medias móviles, deltas, anomalías) que casi seguro violan ese umbral. Es higiene barata y podría ayudar justo a los modelos lineales, que son los mejores a horizonte largo.

Dato útil que ahorra trabajo: en el paper neozelandés, **RFE (eliminación recursiva) es el peor de los tres métodos de selección** en casi todos los casos, y la información mutua queda segundo detrás de LASSO. Si se prueba la idea 2, conviene empezar por LASSO y no gastar tiempo en RFE.

### 2026-09-17 — Comparación de métricas contra el paper noruego (mismo formato: sMAPE y R²), y nota para Rafael

> **Advertencia posterior (misma fecha, ver entrada de arriba)**: esta comparación de sMAPE contra el paper noruego se hizo sin verificar qué convención de sMAPE usaba ese paper. Existen dos definiciones que difieren en un factor de 2 (con y sin el `/2` en el denominador). Mientras no se confirme cuál usa el paper noruego, **la afirmación de que le ganamos en sMAPE debe considerarse no verificada**. Las comparaciones de la entrada de arriba (MASE, NRMSE, TAPI) sí están verificadas contra la fórmula publicada en cada paper.

**Para Rafael (dashboard/OE3)**: el hallazgo de esta tarde (ver entrada de abajo) es que la hidrología y el ONI no bajan el error del modelo, pero predicen muy bien *cuándo desconfiar* de él — el error se triplica según el régimen (28.5 a 97.1 COP/kWh de MAE entre la mejor y la peor combinación de embalse×El Niño). **Sugerencia concreta para la vista Operador**: agregar una segunda señal de confianza basada en el régimen hidrológico actual (embalse alto/bajo × ONI alto/bajo), independiente del chip de confianza que ya existe (que se basa en el ancho de la banda `[q10,q90]`). Los datos están en `data/processed/resultados/estratificacion_regimen_24h.csv`. Con eso, una hora en régimen "embalse bajo + El Niño" podría mostrarse con una advertencia adicional aunque su banda de incertidumbre no se vea especialmente ancha — son dos señales de riesgo distintas y complementarias.

**Comparación de precisión, en el mismo formato que reportan los papers** (sMAPE y R², que ellos usan; nuestro MAPE convencional no es directamente comparable porque sMAPE está acotado 0-200% y es más robusto a errores relativos grandes en precios bajos). Calculado con `metricas_comparables_papers.py`:

| Modelo | MAE | MAPE (convencional) | sMAPE | R² |
|---|---|---|---|---|
| **Noruega, LightGBM, 5 zonas (arXiv 2604.26634)** | 1.60-5.58 EUR/MWh | — | **12.07%-25.32%** | **0.889-0.934** |
| **Nuestro ensamble v4, 24h** | 42.51 COP/kWh | 11.27% | **10.38%** | **0.910** |
| Nuestro ensamble, 72h tramo 1-24h | 47.91 | 13.40% | 12.46% | 0.898 |
| Nuestro ensamble, 72h tramo 25-48h | 64.00 | 18.57% | 16.94% | 0.854 |
| Nuestro ensamble, 72h tramo 49-72h | 74.69 | 21.93% | 19.83% | 0.819 |
| Persistencia (referencia) | 56.40 | 15.81% | 14.77% | 0.845 |
| Naive-168h (semana pasada) | 123.68 | 36.77% | 34.44% | 0.605 |

**Nuestro modelo de 24h le gana al mejor resultado publicado de Noruega en sMAPE** (10.38% contra su mínimo de 12.07% en NO5) y su R² (0.910) queda dentro de su rango, por encima de 2 de sus 5 zonas (NO1 0.889, NO2 0.889). Incluso el tramo más débil de 72h (49-72h, sMAPE 19.83%) queda dentro del rango noruego. Es una comparación justa porque ambos son mercados hidro-dominados con metodología similar (walk-forward causal, LightGBM/gradient boosting como uno de los mejores modelos en ambos casos).

**No se pudo obtener la cifra equivalente para Brasil**: los dos papers brasileños con resultados numéricos de pronóstico (el MLP multi-step de MDPI y el de streamflow+correntropía de Springer) están bloqueados para descarga automática — MDPI por detección de bot (es de acceso abierto CC-BY, cualquier navegador normal lo abre sin login) y Springer por muro de pago.

**Papers pendientes de leer completos, bloqueados para descarga automática — pedidos al usuario para que los baje con su acceso institucional y me los pase**:

| Paper | Por qué importa | Bloqueo |
|---|---|---|
| Dias, Lira & Freire — *"Methodology for Multi-Step Forecasting of Electricity Spot Prices Based on Neural Networks Applied to the Brazilian Energy Market"*, Energies 2024 | Pronóstico multi-step (4 semanas) del PLD brasileño con MLP + selección de features; reporta TAPI/NRMSE/MAPE — el benchmark numérico brasileño que nos falta | MDPI bloquea bots (es CC-BY, abrir en navegador normal: <https://www.mdpi.com/1996-1073/17/8/1864>) |
| Nunes, Ferreira & da Costa Pinho — *"Information Theoretic Learning Applied to Daily Streamflow Forecast and Its Impact on the Brazilian Hourly Energy Spot Prices"*, J. Control Autom. Electr. Syst. 2024 | Arquitectura de dos etapas (pronostica el caudal primero, luego el precio) — la idea que probamos hoy pero con datos reales de Brasil, para ver si a ellos sí les funcionó | Springer, muro de pago |
| Albani, Marcavillaca, Moreira et al. — *"Short-term forecasting of forward prices in the Brazilian electricity market with a hybrid stochastic-neural network model"*, Energy Economics 148 (2025) | Modelo híbrido SDE+red neuronal, la red modela explícitamente el efecto de la ENA (aportes hídricos) sobre el precio forward | ScienceDirect, muro de pago |
| *"Electricity price forecasting in New Zealand: A comparative analysis of statistical and machine learning models with feature selection"*, Applied Energy (~2023) | Nueva Zelanda, ~60% hidro, comparación estadístico vs. ML con selección de features — mercado pequeño y aislado, el más parecido en escala al de XM | ScienceDirect, muro de pago |

Si el usuario consigue el PDF de cualquiera de estos (biblioteca de Uninorte, Sci-Hub-alternativas institucionales, o pidiéndoselo al autor), se puede releer completo y repetir el ejercicio de esta tarde: sacar una idea concreta, implementarla, y validarla con el mismo rigor (múltiples semillas, DM test).

### 2026-09-17 — Ideas de papers de mercados hidro-dominados (Noruega y Brasil): la hidrología no predice el precio, pero predice cuándo falla el modelo

Se buscaron mercados con matriz energética parecida a la colombiana (~70-74% hidro) para tomar ideas de su literatura. Los tres más útiles: **Brasil** (mismo diseño de mercado hidro-térmico, mismo fenómeno ENSO), **Noruega/Nord Pool** (>90% hidro, la literatura de EPF más desarrollada del mundo) y **Nueva Zelanda**. Se leyeron completos dos papers abiertos (los otros dos: MDPI bloquea bots, Springer es de pago):

1. *"Electricity price forecasting across Norway's five bidding zones in the post-crisis era"* (arXiv 2604.26634).
2. *"How optimistic inflow forecasts distort dispatch, prices, and contracts in hydro-dominated power systems: evidence from Brazil"* (arXiv 2607.00504v2).

**Idea 1 probada (del paper brasileño) — "memoria hidrológica"**: el paper muestra que el precio no responde al aporte hídrico de hoy sino a la desviación **acumulada** frente a lo normal, porque es eso lo que vacía el embalse, con rezago de 1-3 meses. Se implementaron 7 variables nuevas (anomalía de aportes y de embalse contra su climatología por día del año —calculada solo con entrenamiento, sin fuga—, acumulados a 30/60/90 días, déficit acumulado, e interacción embalse×ONI). **Resultado: MAE 59.41 → 59.04, p=0.52 — no significativo**, y el MAPE incluso empeora. No sirve.

**Idea 2 probada (del paper noruego) — ablación LOGO**: quitar un grupo de variables a la vez y medir cuánto empeora. El paper noruego encuentra que quitar los rezagos de precio cuesta 17-24 EUR/MWh, el calendario 0.04-0.42 y **los embalses ~0.01-0.03 (nada)**, y concluye que "las exógenas aportan poco a la precisión puntual una vez que la estructura autorregresiva y estacional ya está representada".

**Réplica en nuestros datos, con 3 semillas** (`ablacion_logo_multisemilla.py`, CatBoost directo 24h):

| Grupo quitado | Δ MAE medio | Por semilla | Veredicto |
|---|---|---|---|
| Precio (rezagos e historia) | **+125.76** | [121.4, 138.0, 117.9] | **APORTA** |
| Calendario | **+2.92** | [3.64, 3.36, 1.76] | **APORTA** |
| Festivos | −0.91 | [0.15, 1.01, −3.89] | ruido (el signo cambia) |
| Hidrología (embalses, aportes) | +0.64 | [0.37, 2.40, −0.86] | ruido (el signo cambia) |
| Demanda y generación | +0.12 | [2.03, 1.18, −2.84] | ruido (el signo cambia) |
| ONI | −1.36 | [−2.82, 1.69, −2.96] | ruido (el signo cambia) |

**La jerarquía noruega se replica casi exactamente en Colombia**: solo el historial de precio y el calendario aportan; todo lo demás está dentro del ruido.

**Advertencia metodológica importante que salió de aquí, y que aplica a TODO el proyecto**: la primera corrida de esta ablación (una sola semilla) sugirió que **quitar el ONI mejoraba el modelo en 2.82 COP/kWh con p<0.0001** — un resultado que habría contradicho el marco conceptual del proyecto. Se verificó antes de reportarlo (`verificar_oni_degrada.py`) y **resultó ser ruido**:
- El ONI de 2026 está **100% dentro** del rango de entrenamiento (2019-2025 llegó a +2.00 por el súper Niño 2023-24; 2026 solo a +1.40) — la hipótesis de extrapolación era falsa.
- Con 3 semillas **el signo del efecto se invierte** (−2.82, +1.69, −2.96).
- El MAE base varía 58.23-60.48 **solo por el azar del ajuste** (±0.92): el "efecto" era del tamaño del ruido.
- Caso extremo del problema: con Ridge, el modelo base y el modelo sin ONI dan **exactamente 59.43 los dos**, y el test DM igual reporta **p=0.0000**. Con 5,208 horas el DM detecta diferencias sistemáticas minúsculas y las declara significativas. **Significancia estadística ≠ relevancia práctica.**

→ **Regla de trabajo derivada: cualquier diferencia de MAE menor a ~2 COP/kWh medida con una sola semilla no es interpretable en este proyecto.** Vale la pena releer con esa lente los experimentos anteriores que compararon corridas únicas con diferencias pequeñas.

**Idea 3 probada (la que sí sirve) — estratificación del error por régimen**: el paper noruego recomienda usar las exógenas para *estratificar el riesgo*, no para bajar el error. Aplicado a nuestros datos (celdas embalse alto/bajo × ONI alto/bajo):

| | El Niño (ONI alto) | Neutral/Niña |
|---|---|---|
| **Embalse alto** | 45.8 | 32.1 |
| **Embalse bajo** | **97.1** | 28.5 |

**El error se triplica** (28.5 → 97.1) entre la mejor y la peor celda. Efecto de embalse bajo: +49.4 COP/kWh; efecto de El Niño: +56.0. (Nota honesta: las dos variables están correlacionadas negativamente —El Niño seca los embalses— así que los efectos están parcialmente confundidos; se ve en el conteo desbalanceado de celdas: 492/2112/2112/492.)

**Conclusión accionable para OE3**: la hidrología y el ONI **no sirven para predecir el precio, pero sí para saber cuándo desconfiar del pronóstico**. Eso encaja exactamente con el requisito del Anexo 1 de que las imágenes de decisión lleven metadato de **confianza**: el dashboard puede marcar "esta hora cae en régimen de embalse bajo + El Niño, donde el modelo históricamente se equivoca 3 veces más". Es un uso mejor justificado de esas variables que forzarlas como predictoras.

**Limitación del alcance de esta conclusión**: la ablación se hizo sobre el modelo de árboles de formulación directa a 24h. No se puede extrapolar sin más a N-BEATSx, donde los festivos sí se validaron como significativos en su momento (prueba t pareada, 5 semillas, p=0.0064). La afirmación defendible es: *en un modelo de árboles con historial de precio rico, las exógenas no aportan a la precisión puntual*.

### 2026-09-11 (noche) — Walk-forward con orígenes de 12 meses completos: N-HiTS falla de forma severa en un régimen que las ventanas de 3 meses nunca alcanzaban a mostrar

Los notebooks 10 y 07 se reejecutaron con los 5 orígenes ampliados a ventanas de prueba de **12 meses consecutivos** (jul-2020→jun-2021, …, jul-2024→jun-2025), en vez de los 3 meses originales — el mínimo que recomienda Lago et al. (2021). El Origen 6 (2026) se verificó sin cambios (diferencia máxima 0.000000 en los 5 modelos, confirmando que la reejecución no alteró nada fuera de lo pedido).

**Hallazgo principal, que la ventana corta ocultaba por completo**: en el Origen 1 (jul-2020 a jun-2021, La Niña, arranca en plena pandemia), **N-HiTS falla de forma severa** y ARX+GARCH se deteriora fuerte, mientras N-BEATSx se mantiene estable:

| Modelo | MAE (3 meses, jul-sep 2020) | MAE (12 meses completos) | Corr-f (12m) | MHD (12m) |
|---|---|---|---|---|
| N-BEATSx | 24.85 | 25.79 | 0.838 | 7.70 h |
| N-HiTS | 28.17 | **55.38** (+97%) | **0.204** | **17.24 h** |
| ARX+GARCH | 31.72 | **50.88** (+60%) | 0.793 | 8.48 h |

Con solo 3 meses de prueba, N-HiTS parecía competitivo (28.17, similar a N-BEATSx). Con el año completo, su Corr-f se desploma a 0.204 (el peor de todo el proyecto, en cualquier origen) y su error en ubicar el pico/valle del día casi se triplica (17.24 horas). **Este es exactamente el riesgo que Lago et al. (2021) señalan sobre ventanas de prueba cortas**: una ventana de 3 meses simplemente no alcanzaba a recorrer lo suficiente para que este problema se manifestara.

**Se investigó la causa antes de reportar** (mes a mes, `fecha_hora.dt.to_period('M')`): el error de N-HiTS y ARX+GARCH crece progresivamente de jul-2020 (24.3 y 21.4) a un pico en oct-2020/feb-2021 (76-99), para luego bajar parcialmente hacia jun-2021 (34.9 y 13.4) — un deterioro gradual, no un evento puntual aislado. La hipótesis obvia ("el error crece con la distancia al corte de entrenamiento") **se descartó explícitamente**: se probó el mismo patrón en los Orígenes 3, 4 y 5, y en ninguno se repite (Origen 4 y 5 incluso *mejoran* hacia el final del año; Origen 3 empeora pero por igual en los 3 modelos, reflejando el aumento real de precios de ese período, no una falla específica de modelo). **Conclusión honesta**: el problema es específico del Origen 1 — probablemente algo propio de la dinámica de precio/demanda en el año de pandemia (jul-2020 a jun-2021) que N-HiTS y el componente GARCH de ARX+GARCH manejan mal y N-BEATSx no — pero no se identificó el mecanismo exacto por límite de tiempo. Queda como pregunta abierta para quien quiera profundizar (candidato natural: revisar `es_pandemia` y la volatilidad de la demanda mes a mes en ese período específico).

**Actualización de una afirmación central del informe**: con orígenes de 12 meses, el test de Diebold-Mariano da **N-BEATSx gana a ARX+GARCH en 5/6 orígenes (no 6/6 como se documentaba con ventanas de 3 meses)** — el origen que ya no gana de forma significativa es, previsiblemente, el Origen 1. También cambia frente a Persistencia: N-BEATSx solo gana significativamente en 1/6 orígenes y **pierde en 1/6** (antes con ventanas cortas esto no se había medido con esta granularidad). XGBoost sigue perdiendo contra Persistencia en 6/6 orígenes, sin cambios.

**No afecta las conclusiones de OE3** para 2026 (Origen 6, el año de publicación, es idéntico a antes) pero sí matiza la generalización "N-BEATSx es sistemáticamente mejor en todo régimen" — hay al menos un régimen conocido (pandemia 2020-21) donde no se puede afirmar eso con la misma contundencia para N-HiTS ni ARX+GARCH.

**Segunda ronda de investigación** (mismo tema, después de descartar "distancia al corte"): se revisó si la hidrología del Origen 1 sale del rango visto en entrenamiento — solo 1 de los 12 meses (jun-2021, volumen de embalses 14.06 mil millones de m³) supera el máximo histórico de entrenamiento (12.79 mil millones); los meses donde el error realmente se dispara (oct-2020 a feb-2021) tienen volumen de embalse *dentro* del percentil 95 de entrenamiento, no fuera de rango. Se descarta también esta hipótesis como explicación única. El ONI sí traza un arco muy limpio y completo dentro de este origen (−0.25 → −1.10 → −0.35, un ciclo de La Niña que nace y muere completo dentro de los 12 meses) — candidato más prometedor para seguir, pero no se alcanzó a probar si eso específicamente es lo que confunde a N-HiTS/ARX+GARCH. Cierre por esta noche: **pregunta genuinamente abierta**, con dos hipótesis descartadas y una tercera (el ciclo ONI completo) sin verificar.

### 2026-09-11 (noche) — Ningún candidato nuevo mejora el ensamble de 24h: cierre limpio de esta línea de trabajo

Con TFT, LSTM, las ventanas de calibración promediadas y la recalibración 2026 ya generados, se probó cada uno como votante nuevo del ensamble v4 (`evaluar_votantes_24h.py`, mismo método: QRA por franja de 6h, validación cruzada y versión desplegable, DM en orden cronológico):

| Conjunto probado | MAE (CV) | p vs v4 | MAE (desplegable) | p vs v4 |
|---|---|---|---|---|
| v4 actual (5 modelos) | 42.84 | — | 44.52 | — |
| v4 + TFT | 42.85 | 0.889 (n.s.) | 44.65 | 0.162 (n.s.) |
| v4 + LSTM | 42.87 | 0.383 (n.s.) | 44.54 | 0.548 (n.s.) |
| v4 + TFT + LSTM | 42.88 | 0.662 (n.s.) | 44.67 | 0.145 (n.s.) |
| Ventanas promediadas (reemplaza N-BEATSx/N-HiTS) | 43.31 | 0.227 (n.s.) | 44.90 | 0.359 (n.s.) |
| Recalibrados (reemplaza los 4 modelos base) | 52.39 | **0.0000 PIERDE** | 53.85 | **0.0000 PIERDE** |
| Recalibrados + congelados (se suman como extra) | 42.90 | 0.199 (n.s.) | 44.61 | **0.027 PIERDE** |

**Ninguno mejora.** TFT y LSTM no aportan nada al ensamble pese a ser de una familia arquitectónica distinta — su error individual (57.5 y 66.2 de MAE) es demasiado alto para compensar con diversidad. Las ventanas promediadas, que sí mejoraban un poco como modelo individual (46.8→46.0), no se traducen en mejora del ensamble: una vez que Persistencia/XGBoost/ARX+GARCH ya aportan diversidad real, una versión ligeramente mejor de N-BEATSx/N-HiTS no mueve la aguja (rendimientos decrecientes). Los modelos recalibrados confirman, con significancia aplastante, el hallazgo negativo de la entrada anterior — incluirlos como opción extra incluso empeora la versión desplegable.

**Conclusión**: el ensamble v4 (QRA por franja horaria, 5 modelos base, MAE 42.51-42.84 según la corrida) queda confirmado como el mejor resultado de 24h del proyecto, robusto contra todos los candidatos adicionales probados hoy. Esta línea de trabajo se da por cerrada — no quedan candidatos obvios pendientes de probar para 24h.

### 2026-09-11 (noche) — Recalibración v2 (ya sin los bugs de crash/ONI): sigue perdiendo. Hallazgo negativo genuino, con hipótesis pero sin certeza total

Tras corregir los dos bugs de la entrada anterior, la corrida completa (67 min) terminó **sin errores** pero **sigue perdiendo** contra el modelo congelado, y de forma pareja en los 8 meses (no es un reentrenamiento aislado que salió mal):

| Modelo | Congelado | Recalibrado v2 |
|---|---|---|
| N-BEATSx | 46.83 | 55.70 |
| N-HiTS | 46.45 | 56.65 |

Se investigó una hora más antes de aceptar esto como conclusión, porque contradice directamente el experimento de anoche (reentrenar antes de junio sí ganó). Se descartó una hipótesis real que sí encontré pero que resultó **no ser la explicación completa**: los cortes de mi script (ventanas de 00:00 a 23:00) están desalineados por una hora contra el walk-forward original (que predice de 01:00 a 00:00 del día siguiente, porque `cross_validation` ancla sus ventanas contando hacia atrás desde el final absoluto de la serie, no desde la máscara de entrenamiento). Confirmado con una prueba aislada: si fuerzo mi script a predecir con la alineación "correcta" (01:00 en vez de 00:00), `predict()` truena, porque un modelo entrenado hasta el 31-dic 23:00 solo puede predecir exactamente las 24 horas siguientes a ESE punto (1-ene 00:00 a 23:00) — confirma que mi alineación original (00:00-23:00) era la esperada dado el punto de corte, no un bug. El desfase de 1 hora entre mi convención y la del walk-forward original afecta como máximo 2 de 5,184 horas en las comparaciones (por el `join` con `how="left"`), insuficiente para explicar una brecha de +19% de MAE.

**Hipótesis que queda en pie, sin verificar del todo por límite de tiempo**: cada reentrenamiento completo (8 en total) es una corrida de optimización independiente que puede converger a un mínimo local distinto, incluso con semilla fija — a diferencia del experimento de anoche, que hizo **una sola** apuesta de reentrenamiento, deliberada, evaluada solo en el mes para el que se diseñó. Aquí se apuestan 8 veces seguidas, cada una con `max_steps=1000` fijo sin importar que el conjunto de entrenamiento crezca solo ~1-2% por reentrenamiento — plausible que el ruido de convergencia entre corridas pese más que cualquier beneficio real de adaptación, cuando se acumula 8 veces.

**Decisión**: no se seleccionó una nueva estrategia (ej. promediar semillas por reentrenamiento, aumentar max_steps con el tamaño de los datos) porque el tiempo ya invertido en depurar esto (2+ horas) compite con las tareas explícitamente pedidas y aún pendientes (walk-forward anual). Queda documentado como pregunta abierta, no como bug pendiente de arreglar — el código ya está correcto (dos bugs reales encontrados y corregidos), el resultado es simplemente negativo por razones que no se terminaron de aislar. **No se recomienda usar `recalibracion_2026_v2_predicciones.csv` en el ensamble** hasta investigarlo más si hay tiempo.

### 2026-09-11 (noche) — Dos bugs más en la recalibración v2, encontrados antes de que causaran daño

Al retomar la sesión, la primera corrida de `recalibracion_2026_v2.py` (la corrección de anoche) **truena** en la segunda ventana después de cada reentrenamiento, con `ValueError: There are missing combinations of ids and times in futr_df`. Se investigó en vez de parchear a ciegas, y aparecieron dos bugs reales, uno de los cuales es silencioso (no lanza error, solo produce números mal escalados) y es justo el tipo de error más peligroso de un proyecto de este tamaño:

**Bug A (el que tronaba)**: `NeuralForecast.predict(futr_df=...)` **no** pronostica "la ventana que se le pida" según las fechas de `futr_df` — pronostica exclusivamente los `h` pasos inmediatamente siguientes al final de los datos con los que el modelo fue entrenado (o de los que se le pasen explícitamente en `df=`). El script reentrenaba cada 30 días pero seguía llamando `predict()` para los 29 días intermedios sin decirle qué había pasado en esos días reales ya observados — funcionaba por casualidad en la primera ventana tras cada reentrenamiento (coincide con el fin del entrenamiento) y explotaba en la segunda. Una prueba aislada de una sola ventana (la que se hizo para "confirmar" el fix de anoche) no lo detectaba, precisamente porque solo probaba el caso que sí funcionaba.

**Bug B (silencioso)**: la normalización del ONI (la única exógena futura continua) se aplicaba a los datos de entrenamiento pero **nunca** a `futr_df` — el modelo entrenaba viendo el ONI en escala normalizada (z-score) y al predecir recibía el valor crudo (rango −1 a 2). Ningún chequeo de la librería detecta esto; simplemente habría producido predicciones mal calibradas sin ningún aviso.

**Corrección** (`recalibracion_2026_v2.py`, reescrito): se congelan las estadísticas de normalización (media/desviación) en cada reentrenamiento; en las ventanas intermedias se le pasa a `predict()` el histórico real observado hasta esa fecha (`df=`, normalizado con las estadísticas congeladas) además de `futr_df` (con el ONI ahora sí normalizado). Validado con una prueba rápida de 6 ventanas y 2 reentrenamientos antes de lanzar la corrida completa — confirmado que las ventanas intermedias usan el histórico extendido correctamente (60,649 → 60,673 → 60,721 filas, creciendo un día a la vez) sin error. Corriendo ahora la versión completa (~40-55 min estimados).

**Lección operativa**: al retomar una sesión pausada, un "arreglo" que solo se probó en el caso más fácil (una ventana) puede ocultar un bug que aparece en el caso general (muchas ventanas seguidas). Vale la pena, antes de lanzar una corrida larga, una prueba rápida y barata que ejercite el patrón completo (aquí: varias ventanas, más de un reentrenamiento) en vez de solo el primer caso.

### 2026-09-11 (tarde) — PAUSADO: el usuario apaga el equipo. Estado exacto y cómo retomar

Se detuvo todo limpiamente a pedido del usuario (necesitaba apagar el PC). Detalle relevante para
la próxima sesión: `TaskStop` sobre la tarea rastreada de la cola (`cola_trabajos.sh`) **no mató el
proceso real del sistema operativo** — el script bash siguió corriendo como huérfano y, al matar
manualmente los procesos de Python del notebook 10, el propio script interpretó eso como "paso
terminado" y arrancó el notebook 07 sin que se le pidiera. Se detectó revisando los procesos con
`Get-CimInstance Win32_Process` (que sí muestra la línea de comando real) y se mató el árbol
completo (`bash.exe` de `cola_trabajos.sh` + todos los `python.exe`/`jupyter-nbconvert` hijos).
Lección para la próxima vez que haya que frenar una cola en background: verificar con
`Get-CimInstance Win32_Process` si el proceso real sigue vivo después de `TaskStop`, no confiar
solo en la respuesta de la herramienta.

**Verificado antes de confirmar que era seguro apagar**:
- `git status` sobre `notebooks/10_diebold_mariano_juan.ipynb` y `07_validacion_walkforward_juan.ipynb`
  muestra ambos como "modificados" (por la edición de los orígenes a 12 meses) pero **no ejecutados**
  — `nbconvert --inplace` solo escribe el archivo al terminar con éxito, así que matar el proceso a
  mitad de camino no corrompió ni truncó ningún notebook. Quedan exactamente como se dejaron antes
  de correr, listos para ejecutarse limpios la próxima vez.
- Ningún proceso python/bash relacionado con el proyecto seguía vivo tras la limpieza.
- Todos los CSV nuevos de la tarde están completos en disco (verificado por tamaño/fecha), ninguno
  a medio escribir.

**Qué quedó completo y guardado** (ver entradas de esta tarde arriba y abajo de esta):
TFT/LSTM de 24h, ventanas de calibración de 24h, forma del día (2 rondas), sesgo 2026,
recalibración 2026 v1 (con el bug de warm-start ya diagnosticado), evaluación de 72h con 214 cortes
diarios + su ensamble (con el bug de bandas NaN ya corregido).

**Qué falta, en orden de prioridad para retomar**:
1. `python scripts_experimento/recalibracion_2026_v2.py 30` — la versión corregida (reentrena
   desde cero), ~40-55 min. **No usar `recalibracion_2026_predicciones.csv` (v1) para nada nuevo**,
   solo quedó documentado como diagnóstico del bug.
2. `python scripts_experimento/evaluar_votantes_24h.py` — decide si TFT, LSTM, las ventanas
   promediadas y los modelos recalibrados (v2) entran al ensamble de 24h. Antes de correrlo, editar
   `FUENTES` en ese script para que apunte a `recalibracion_2026_v2_predicciones.csv` en vez de la
   v1 (columnas `NBEATSx_rec`/`NHITS_rec` con el mismo nombre en ambos archivos).
3. Reejecutar los notebooks 10 y 07 (ya editados, solo falta correr):
   `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1
   --ExecutePreprocessor.kernel_name=python3 notebooks/10_diebold_mariano_juan.ipynb` (y luego el 07)
   — con orígenes de 12 meses, ~90-110 min combinados. Los backups de la versión de 3 meses
   (`*_3meses_backup.csv`) ya existen, así que `cola_trabajos.sh` los detecta y no los pisa.
4. Con el walk-forward anual ya generado, correr `resumen_walkforward_anual.py` (script ya
   preparado hoy) para la tabla de MAE/rMAE/sesgo/forma por origen y decidir si algo cambia en las
   conclusiones del informe comparativo.
5. Se puede simplemente reejecutar `bash scripts_experimento/cola_trabajos.sh` completo — es
   idempotente en el paso de backups y retoma exactamente en el paso 1 (TFT/LSTM), que ya está
   guardado, así que conviene comentar esa línea o correr los pasos sueltos en el orden de arriba
   en vez de la cola completa, para no repetir 40 minutos de trabajo ya hecho.

### 2026-09-11 (tarde) — 72h con 216 cortes diarios: la significancia que faltaba esta mañana ya aparece en los tres tramos

Pedido del usuario: la evaluación de 72h usaba solo 72 cortes no solapados (`step_size=72`); con `step_size=24` habría 214-216 ventanas solapadas y mucha más potencia estadística en los tests DM — identificado esta mañana como la mejora metodológica más barata pendiente. Se corrió (`evaluacion_72h_diaria.py`): 214 cortes diarios a las 00:00 (alineados con los cortes de 24h, para que el contrato unificado sea consistente), entrenando 5 variantes de N-BEATSx (base, semillas 7 y 123, contexto de 336h, exógenas enriquecidas) + XGBoost/CatBoost/CatBoost-denso/Ridge directos + naive estacional.

Resultado — con 214 ventanas (`ensamble_72h_diario.py`, meta-modelo LAD por tramo, igual que v3), **la ganancia contra N-BEATSx ahora es significativa en los tres tramos, en las dos versiones**:

| Tramo | N-BEATSx | Ensamble (CV) | p (CV) | Ensamble (desplegable) | p (desplegable) |
|---|---|---|---|---|---|
| 1-24h | 49.41 | **47.07** (−4.7%) | **0.0008** | **49.43** (−3.7%) | **0.0224** |
| 25-48h | 70.97 | **64.00** (−9.8%) | **0.0010** | **67.13** (−8.4%) | **0.0065** |
| 49-72h | 85.99 | **74.69** (−13.1%) | **0.0003** | **79.34** (−10.1%) | **0.0116** |

Con 72 cortes esta mañana, la versión desplegable de 25-48h y 49-72h no alcanzaba a ser significativa (p=0.15 y p=0.09); con 214 cortes **sí lo es** en ambos tramos y en ambas versiones — confirma que el problema de esta mañana era falta de potencia estadística, no que la mejora no fuera real. El promedio de semillas de N-BEATSx (`NBX_semillas`) resultó el mejor votante individual en 1-24h y 25-48h (47.50 y 67.85); Ridge directo sigue ganando en 49-72h (82.13).

**Bug encontrado y corregido en el camino**: al calcular las bandas de la versión desplegable en el tramo 49-72h, 672 de 5,136 filas dieron NaN en el ancho de banda. Causa: en esas filas el optimizador LAD le dio peso 0 a los dos únicos modelos con cuantiles propios (N-BEATSx y N-BEATSx-exog) porque el punto se armaba mejor solo con árboles/lineal — al no quedar ningún modelo con banda, la normalización de pesos para las bandas quedaba en 0/0. Corregido con un respaldo de peso igual entre los modelos con banda, usado solo para ese cálculo (el pronóstico puntual no cambia). Tras la corrección: cobertura 76.2%/76.9% en 49-72h (objetivo 80%).

**Nuevos contratos**: `pronostico_unificado_72h_diario_2026.csv` (todas las 214 ventanas solapadas, para análisis) y `pronostico_unificado_72h_v2_2026.csv` (ventanas no solapadas, mismo esquema que el contrato anterior — reemplaza a `pronostico_unificado_72h_2026.csv` como candidato a contrato de OE3, pendiente de decidir cuál de los dos usar).

### 2026-09-11 (tarde) — Recalibración 2026: bug de warm-start encontrado en `refit` de neuralforecast; corrección lista, pendiente de correr

Al recalibrar ARX+GARCH y XGBoost a diario (reentrenados **desde cero** en un loop propio) el resultado fue neutro-a-positivo, como se esperaba. Pero al recalibrar N-BEATSx/N-HiTS con el parámetro nativo `refit=30` de `NeuralForecast.cross_validation`, el resultado fue **peor que el modelo congelado en los 8 meses de 2026**, contradiciendo directamente el hallazgo de anoche (reentrenar antes de junio sí ganó, 93.59→86.21):

| Modelo | Congelado (MAE) | Recalibrado v1 (MAE) |
|---|---|---|
| ARX+GARCH | 55.83 | 56.26 |
| XGBoost | 61.39 | **60.00** |
| N-BEATSx | 46.82 | 48.72 (peor) |
| N-HiTS | 46.44 | 52.92 (mucho peor) |

**Causa raíz encontrada, leyendo el código fuente de `neuralforecast` 3.2.1**: cuando se usa `refit=N`, cada reentrenamiento llama internamente a `self.fit(..., use_init_models=False)` — que es también el valor por defecto de `fit()` — y eso **continúa el entrenamiento desde los pesos ya ajustados** (warm-start) en vez de reiniciar el modelo desde cero. Con un conjunto de entrenamiento que apenas crece unos cientos de horas cada vez (frente a ~60,000 ya vistas), 1000 pasos adicionales de warm-start no aprenden nada nuevo de peso, pero sí reinician el programa de tasa de aprendizaje, empujando al modelo fuera de un mínimo ya bueno — efecto que se acumula en cada reentrenamiento sucesivo (por eso el deterioro crece mes a mes: 46.82 en enero-abril, hasta perder 6-9 puntos de MAE en mayo-agosto). El experimento de anoche no tenía este problema porque usaba un `NeuralForecast` completamente nuevo con un único `.fit()` desde cero — igual que se hace aquí manualmente para ARX+GARCH y XGBoost.

**No es un resultado negativo genuino sobre la adaptabilidad** — es un detalle de implementación de la librería. Corregido en `recalibracion_2026_v2.py`: loop manual que crea una instancia nueva de NBEATSx/NHITS en cada punto de reentrenamiento (mismo criterio que anoche), y recalcula la normalización de las exógenas con los datos disponibles hasta ese corte (la v1 usaba una normalización fija de enero-2026, que la v2 también corrige). **No se alcanzó a correr** — el usuario tuvo que apagar el equipo antes de que hubiera tiempo (~40-55 min estimados). Queda listo para la próxima sesión.

### 2026-09-11 (tarde) — Forma del día: separar nivel y forma; el sesgo visto solo dentro de 2026; por qué el TFT no terminaba

Pedido del usuario: (1) dejar de lado el desempeño 2020-2025 y **adaptar el modelo a 2026**, el año que se publica; (2) resolver la forma del día; (3) terminar primero el TFT y todo lo que quedó a medias; (4) walk-forward con orígenes de **un año completo**; (5) recalibración diaria; (6) 72h con cortes diarios. Esta entrada cubre lo ya cerrado; el resto corre en una cola secuencial (`cola_trabajos.sh`) y se documenta en la entrada siguiente.

**Forma del día — primera vez medida para los ensambles** (`metricas_forma_2026.py`, métricas de Maciejowska, Lipiecki & Uniejewski 2026). Corr-f = si el pronóstico ordena las 24 horas del día igual que la realidad; MHD = horas de error al ubicar la hora más barata y la más cara (suma de ambas); MPD = COP/kWh que se pierden por operar en la hora señalada en vez de la óptima.

| 2026 | MAE | Corr-f | MHD (h) | MPD | Amplitud |
|---|---|---|---|---|---|
| Persistencia | 56.40 | **0.835** | **3.97** | 65.45 | 1.00 |
| N-BEATSx | 46.82 | 0.775 | 5.70 | 58.82 | 0.81 |
| Ensamble v1 (NNLS) | 44.51 | 0.816 | 4.51 | 53.27 | 0.80 |
| Ensamble v4 (QRA franjas) | **42.51** | 0.799 | 5.01 | **47.43** | 0.94 |

El v4 ya es el que **menos plata deja en la mesa** (MPD) y el que menos aplana la curva, pero copiar el día anterior ordena mejor las horas y ubica mejor el pico y el valle. Y v1 ordenaba mejor que v4: los pesos por franja crean saltos en las fronteras de las franjas.

**Solución probada — separar nivel y forma** (`forma_del_dia.py`, `forma_del_dia_v2.py`). Las tres métricas de forma dependen solo de cómo se mueve cada hora respecto a la media del día; sumarle una constante a todo el día no las cambia. Entonces se toma el **nivel** (media de la ventana de pronóstico) del v4, que gana en MAE, y la **forma** de un ensamble LAD ajustado solo sobre las desviaciones respecto a la media, con candidatos que incluyen el perfil semanal del precio (mediana de la misma hora en los últimos 7 días):

| | MAE | Corr-f | MHD (h) | MPD |
|---|---|---|---|---|
| v4, validación cruzada | 42.51 | 0.799 | 5.01 | 47.43 |
| Nivel v4 + forma ajustada, validación cruzada | 43.05 (+1.3%) | **0.829** (p<0.0001) | 4.74 (n.s.) | 46.70 (n.s.) |
| v4 desplegable | 44.52 | 0.796 | 4.92 | 47.99 |
| Nivel + forma, desplegable | 45.25 (+1.6%, p=0.016) | **0.829** (p=0.001) | 4.50 (p=0.31) | 48.19 (n.s.) |

Es un intercambio real: casi el Corr-f de la persistencia (0.835) con un MAE mucho mejor (45 contra 56), a cambio de +1.6% de MAE frente a v4. Como Maciejowska et al. reportan que el Corr-f correlaciona >0.80 con la ganancia de una regla de arbitraje y el MAE <0.20, la versión con forma ajustada es la candidata natural si la regla de OE3 depende del *momento* de actuar. El MHD (ubicar exactamente el pico y el valle) sigue sin mejorar de forma significativa: queda abierto.

**El sesgo visto solo dentro de 2026** (`sesgo_2026.py`, pronóstico − real, en COP/kWh):

| Mes 2026 | Precio medio | XGBoost | N-BEATSx | v4 |
|---|---|---|---|---|
| Ene-Abr | 124-269 | +2 a +10 | +2 a +7 | −1.6 a +1.5 |
| May | 505 | −46.0 | −7.1 | −9.8 |
| Jun | 548 | −51.0 | +3.9 | −6.5 |
| Jul | 791 | −66.3 | +17.5 | +4.4 |
| Ago (5 días) | 935 | −58.2 | +44.5 | +34.4 |

Cuando el precio se duplica en mayo-junio **todos subestiman**; en julio-agosto las redes pasan a **sobreestimar**. XGBoost se queda corto por 46-66 COP/kWh desde mayo: los árboles no pueden pronosticar por encima de lo que vieron en entrenamiento, y 2026 supera los precios de 2019-2025. Por tercil de precio se repite el patrón conocido (v4: +8.3 en el tercil bajo, −25.7 en el alto). Todo esto apunta a lo mismo: un modelo congelado a diciembre de 2025 no ha visto el nivel de precios de 2026 — que es lo que ataca la recalibración.

**Por qué el TFT no terminaba** (`tft_benchmark.py`): con la configuración de anoche (hidden 64, lote de 1024 ventanas) cada paso de entrenamiento tomaba **23.5 s**, o sea ~6.5 horas para las 1000 iteraciones — no estaba colgado, simplemente era inviable en CPU. Con hidden 32 y lotes de 256 ventanas baja a 2.6 s/paso (~43 min). LSTM y N-BEATSx toman ~3 min.

**Walk-forward con años completos**: los notebooks 07 y 10 pasan de ventanas de prueba de 3 meses a **5 ventanas consecutivas de 12 meses** (jul-2020→jun-2021, …, jul-2024→jun-2025), sin solapes. Se eligieron años julio→junio y no calendario porque así el Origen 1 entrena con la pandemia ya incluida (con años calendario entrenaría solo con 2019), y el episodio completo de El Niño 2023-24 (ONI medio +1.30, pico 2.0) queda dentro del Origen 4. El Origen 6 (2026) no cambia. Las salidas de la versión de 3 meses se respaldan como `*_3meses_backup.csv` antes de reejecutar.

### 2026-09-11 — Contrato unificado 1-72h para OE3: MAE global 63.50, −18.4% frente al punto de partida del día

Los cortes de 72h (cada 3 días a las 23:00) son un subconjunto de los cortes diarios de 24h (23:00): para h=1-24, el ensamble de 24h y el de 72h pronostican **exactamente las mismas horas desde el mismo corte**. Entonces el contrato de horizonte completo puede usar el especialista de 24h (v4) en los pasos 1-24 y el ensamble v3 en los pasos 25-72 (`contrato_unificado_72h.py`). Antes de unificar se verificó en esas mismas 1,728 horas:

| Pasos 1-24 (72 cortes) | MAE | MAPE | Cobertura | Ancho | Interval Score |
|---|---|---|---|---|---|
| v3 (LAD 72h) | 54.68 | 15.52% | 78.4% | 181.4 | 302.2 |
| **v4 (QRA 24h por franja)** | **41.69** | **10.20%** | 79.6% | 162.0 | 282.2 |

El especialista gana en punto con amplitud (DM p<0.0001); en bandas es más angosto con más cobertura, aunque la diferencia de Interval Score no es significativa (p=0.32).

**Evolución del día (2026, fuera de muestra):**

| | 1-24h | 25-48h | 49-72h | Global 1-72h |
|---|---|---|---|---|
| Inicio del día — N-BEATSx 72h solo | 58.68 | 76.20 | 98.56 | 77.81 |
| **Contrato unificado (v4 + v3)** | **41.69** | **64.32** | **84.50** | **63.50** (MAPE 18.20%) |

Detalle corregido en el camino: la calibración conforme puede producir un margen negativo cuando la banda sobrecubre, y eso cruzaba cuantiles en 42 de 5,184 horas de las bandas QRA de 24h. Ahora se reordenan después de calibrar (q10 ≤ q50 ≤ q90). Los tres contratos nuevos pasan `validar_contrato_pronostico` del motor de decisión con 0 NaN y 0 cuantiles cruzados.

**Integración pendiente de aprobación del equipo**: `fuentes_pronostico.json` **no se modificó** y sigue apuntando a N-BEATSx solo. El cambio propuesto es `"24h"` → `bandas_24h_qra_causal_2026.csv` y `"72h"` → `pronostico_unificado_72h_2026.csv`. El motor solo consume `fecha_hora, real, q10, q50, q90`, así que no hay que tocar el notebook 12 ni el dashboard — pero ambos leen de ese JSON, así que el cambio se deja para decidirlo con Rafael.

### 2026-09-11 — Ensamble de 24h v4: QRA por franja horaria — MAPE 11.27%, nuevo mejor resultado del proyecto

Dos ideas de la literatura revisada, combinadas: el paper de Uniejewski (2026) aplica todos sus esquemas de combinación **por separado para cada hora del día**, y QRA (Quantile Regression Averaging, Nowotarski & Weron 2015) es la generalización del meta-modelo LAD a cualquier cuantil. Mismos 5 votantes del ensamble anterior (Persistencia, XGBoost, ARX+GARCH, N-BEATSx, N-HiTS), mismas predicciones — solo cambia cómo se combinan.

**Paso 1 — pesos por franja horaria** (`pesos_por_franja.py`, 4 franjas de 6h elegidas antes de ver resultados): LAD por franja da MAE 43.24 / MAPE 11.70% contra 44.51 / 12.05% (DM p=0.0006). Con un solo vector de pesos para todo el día (NNLS o LAD global) no hay diferencia; la ganancia viene de dejar que los pesos cambien a lo largo del día. Los pesos se leen bien físicamente: en la **noche** (18-23h) la persistencia se lleva 41% — el pico nocturno se repite casi idéntico día a día —; en la **madrugada** domina N-HiTS (65%); en la **tarde** ARX+GARCH sube a 33%.

**Paso 2 — QRA en vez de LAD** (`validar_qra_24h.py`): QRA en el cuantil 0.5 difiere del LAD en que tiene intercepto y los pesos no están obligados a sumar 1. El intercepto absorbe el sesgo de nivel común a todos los votantes que el proyecto ya había documentado (todos sobreestiman en precios bajos).

| Ensamble de 24h | MAE | MAPE | Acierto por tercil | Errores graves | MAE versión desplegable* |
|---|---|---|---|---|---|
| v1 — NNLS global (anterior) | 44.51 | 12.05% | 90.90% | 0.19% | 46.12 |
| v3 — LAD por franja | 43.24 | 11.70% | 91.53% | 0.21% | 45.17 |
| **v4 — QRA por franja** | **42.51** | **11.27%** | **92.01%** | **0.14%** | **44.51** |

\*Pesos estimados solo con días anteriores (ventana expansiva), evaluado tras 14 días de calentamiento.

Pruebas de robustez superadas: (1) la ganancia se sostiene con 3, 4 y 6 franjas (+4.0% a +4.5%, DM p<0.0001 en las tres) — no es una partición elegida con suerte; con 12 o 24 franjas cae, porque 5 pesos por grupo con ~200 días empiezan a estimarse con ruido; (2) v4 le gana a LAD por franja (p=0.0016); (3) **en versión desplegable también gana**: 44.51 contra 46.12 del ensamble anterior desplegable (p=0.0004).

**Bandas propias para el ensamble de 24h** (antes solo N-BEATSx solo tenía bandas a 24h): QRA en los cuantiles 0.1 y 0.9 con los mismos votantes, más calibración conforme adaptativa causal. Cobertura 79.7% con ancho medio 159.6 COP/kWh, contra 76.8% y 181.4 del contrato de bandas vigente — **más cobertura con bandas 12% más angostas**; Interval Score (Gneiting & Raftery 2007) 289.8 vs 322.5, mejor con significancia (p=0.006). En versión desplegable: 76.6% / 176.0 / IS 312.1 contra 76.8% / 192.3 / IS 342.1.

Archivos: `stacking_24h_v4_qra_2026.csv` (punto por validación cruzada y columnas de la versión desplegable), `bandas_24h_qra_causal_2026.csv` (q10/q50/q90).

### 2026-09-11 — Segunda revisión: dos correcciones metodológicas, meta-modelo LAD (72h v3), y lo que la literatura no transfirió

Pedido del usuario: seguir buscando mejoras para ambos modelos (24h y 72h), guiándose por la literatura ya revisada. Se reportan primero las **correcciones**, porque cambian conclusiones ya documentadas.

**Corrección 1 — los p-valores del ensamble v2 de 72h estaban inflados.** En `ensamble_72h_v2.py` la serie de diferencias de pérdida para el test Diebold-Mariano se armaba concatenando los pliegues de la validación cruzada (días en orden aleatorio). El estimador HAC (Newey-West) asume orden temporal para medir la autocorrelación; al desordenar la serie la subestima y los p-valores salen artificialmente pequeños. Se detectó porque el análisis causal (en orden correcto) daba p≈0.15 donde v2 reportaba p=0.023. Verificado reproduciendo ambos órdenes (`verificar_dm_orden.py`): **v2 contra N-BEATSx da p=0.076 (25-48h) y p=0.090 (49-72h) en orden cronológico — no significativos al 5%**, no p=0.023 y p=0.037 como se documentó en la entrada siguiente. Las mejoras de MAE (−12.0% y −12.1%) sí eran reales; lo que no se sostenía era la significancia. El ensamble de 24h (MAPE 12.05%) **no** tiene este problema: su archivo ya se guardaba en orden cronológico y sus p-valores son idénticos en ambos órdenes. Bug corregido en el script.

**Corrección 2 — fuga intradía en RecursiveLS.** La variante recursiva predecía la hora `t` con coeficientes filtrados hasta `t-1`, pero el protocolo es day-ahead (corte a las 23:00): los coeficientes de la hora 15 ya habían visto los precios reales de las horas 0-14 del mismo día. Corregido en `recursive_ls_dayahead.py` (coeficientes hasta las 23:00 del día anterior): MAE 59.33 → 59.40. La conclusión se sostiene — sigue ganándole a los coeficientes fijos (DM p=0.00001).

**Revisión que resultó sin problema — calibración conforme de 72h.** Se sospechó que el margen adaptativo de un día objetivo D usaba scores de D−1 y D−2, posteriores al corte en el tramo 49-72h. No se materializa: con cortes cada 72h a las 23:00, cada tramo cubre exactamente un día completo, y en ese tramo no existen filas de D−1 ni D−2. La versión por corte (solo objetivos ≤ corte) da resultados idénticos y queda implementada en `qra_bandas.py` para cuando se use `step_size=24`, donde sí importaría.

**72h v3 — meta-modelo LAD en vez de NNLS.** NNLS minimiza error cuadrático, dominado por los picos; el proyecto se evalúa con MAE. LAD (mínima desviación absoluta, pesos ≥ 0 que suman 1, resuelto exacto por programación lineal) optimiza la métrica que se reporta — y es exactamente QRA en el cuantil 0.5 (Nowotarski & Weron 2015). Con los mismos 9 votantes y el mismo protocolo (`ensamble_72h_v2.py --lad`):

| Tramo | N-BEATSx | v2 (NNLS) | **v3 (LAD)** | Mejora v3 | DM v3 vs N-BEATSx | DM v3 vs v2 |
|---|---|---|---|---|---|---|
| 1-24h | 58.68 | 57.42 | **54.68** | −6.8% | p=0.077 (n.s.) | p=0.003 (sig.) |
| 25-48h | 76.20 | 67.07 | **64.32** | **−15.6%** | **p=0.019 (sig.)** | p=0.0009 (sig.) |
| 49-72h | 98.56 | 86.67 | **84.50** | **−14.3%** | **p=0.043 (sig.)** | p=0.069 (n.s.) |
| Global 1-72h | 77.81 | 70.38 | **67.83** | −12.8% | MAPE 19.97% | |

Todos los p-valores en orden cronológico. **Con LAD la significancia contra la línea base sí se sostiene en los dos tramos largos.** Pesos en 49-72h: Ridge directo 0.40, CatBoost directo 0.19, N-BEATSx 0.18, N-BEATSx-exog 0.14; N-HiTS recibe peso 0 en todos los tramos. Versión desplegable (pesos LAD con ventana expansiva, solo cortes anteriores): 56.36 / 68.95 / 87.54 contra N-BEATSx 59.17 / 77.64 / 100.13 en las mismas filas (−4.7% / −11.2% / −12.6%). Bandas con calibración adaptativa: cobertura 78.4% / 77.5% / 75.0%, anchos 181 / 228 / 285 COP/kWh (más angostas que v2 con la misma cobertura). **Nuevo contrato de 72h para OE3: `pronostico_ensamble_72h_v3_adaptativo.csv`** (reemplaza a v2; mismo esquema de columnas).

**Validación de rigor del ensamble de 24h (12.05%)**: la versión desplegable (pesos con ventana expansiva, solo días anteriores) da MAE 46.12 contra 46.02 de la validación cruzada en las mismas horas (p=0.65). El resultado **no** es un artefacto de usar días futuros para fijar los pesos.

**Lo que se probó y no funcionó:**

| Intento | Resultado | Por qué |
|---|---|---|
| Fusionar al stacking de 24h los 8 modelos sueltos de anoche (CatBoost, hurdle, Markov, RecursiveLS, XGB/CatBoost Optuna, CatBoost/Ridge directos 24h) | Empeora: 46.02 → 46.45 (p=0.013) | Casi todos son árboles muy correlacionados entre sí; 13 pesos se estiman con ruido. Confirma que importa la diversidad de familias, no la cantidad de votantes |
| Pesos móviles (ventanas de 30 y 60 días) — la idea de agregación adaptativa de expertos (BOA) de arXiv:2405.15359 | Peor que la ventana expansiva en 24h (p<0.01); sin ganancia en 72h | Con ~200 días de 2026, estimar pesos con 30-60 días es demasiado ruidoso |
| Transformación asinh (Uniejewski, Weron & Ziel 2018) y asinh parametrizado + AVGroll (Uniejewski 2026, EPSR 257) sobre el Ridge directo | asinh estándar **empeora** 6-16%; AVGroll solo +0.7% | Los picos colombianos son cambios de nivel sostenidos por régimen hidrológico, no picos aislados como en Europa: comprimir la cola alta sesga hacia abajo justo en El Niño (el MAPE baja un poco, el MAE sube). En árboles con pérdida MAE la VST es casi irrelevante por construcción (la mediana es equivariante ante transformaciones monótonas) |
| Pesos por franja horaria en 72h | Sin diferencia (p>0.6) | Con 72 cortes, ~430 filas por franja y tramo no alcanzan para 9 pesos |
| QRA para las bandas de 72h | Interval Score sin diferencia significativa contra v3 | Las bandas de v3 ya están bien calibradas |

### 2026-09-11 — Revisión del proyecto y mejora del modelo de 72h (ensamble v2) — ⚠ la significancia reportada aquí fue corregida en la entrada de arriba

Pedido del usuario: revisar todo el proyecto buscando oportunidades de mejora, probarlas, y mejorar específicamente el modelo de 72h. La revisión encontró **dos huecos estructurales**, ambos atacables:

**Hueco 1 — el pipeline de 72h usaba 3 de 47 variables.** `bandas_incertidumbre_nbeatsx_72h.py` solo pasaba `volumen_embalses`, `aportes_hidricos` y `demanda_lag24h` como exógenas históricas, y no incluía `dia_anio_sin/cos` (que el pipeline de 24h sí tenía). Faltaban sobre todo las versiones **normalizadas** de la hidrología: `volumen_embalses` en metros cúbicos absolutos (~1.3e10) es no estacionaria y su nivel no significa lo mismo en 2019 que en 2026, mientras que `volumen_embalses_vs_media30d` sí es comparable entre regímenes.

**Hueco 2 (el importante) — la afirmación de que "XGBoost no se puede extender a 25-72h" era falsa.** Anoche se documentó que Persistencia/XGBoost/ARX+GARCH estaban estructuralmente atados a `precio_lag24h` y que extenderlos exigiría 71 reentrenamientos. Eso es cierto solo para la formulación **recursiva**. La formulación **directa** sí se puede: un único modelo que recibe el paso del horizonte `h` como variable y usa exclusivamente el estado conocido en el momento del corte (`t-h`) más el calendario/ONI del instante objetivo. Un solo modelo cubre los 72 pasos, sin fuga y sin reentrenamientos múltiples. Esto desbloqueó las familias no neuronales para todo el horizonte — justo la diversidad que le faltaba al ensamble fallido de anoche.

Disciplina anti-fuga del modelo directo (lo crítico): el estado se toma en `t-h`, nunca en `t`; `precio_lag168h` sí se usa porque `h<=72<168` lo hace siempre conocido en el corte; `precio_lag24h` NO se usa porque para `h>24` no está disponible, y en su lugar se construye `precio_mismo_hora_disp` = precio en `t - 24*ceil(h/24)`, el valor más reciente de la misma hora del día que sí existe en el corte.

**Modelos individuales nuevos a 72h** (mismos 72 cortes de la validación cruzada neuronal, MAE por tramo):

| Modelo | 1-24h | 25-48h | 49-72h |
|---|---|---|---|
| N-BEATSx (línea base de producción) | **58.68** | 76.20 | 98.56 |
| N-BEATSx exógenas enriquecidas | 59.57 | 75.18 | 96.24 |
| N-HiTS | 64.59 | 82.01 | 108.69 |
| XGBoost directo | 65.89 | 74.28 | 88.89 |
| XGBoost directo v2 (anclas + cuantiles) | 71.00 | 75.60 | 89.72 |
| CatBoost directo | 63.30 | 71.54 | 86.63 |
| CatBoost directo (2× datos) | 66.29 | 72.17 | 89.45 |
| **Ridge directo (lineal)** | 62.91 | **71.12** | **85.52** |
| Naive estacional (misma hora hace 7 días) | 122.06 | 127.79 | 124.24 |

**El hallazgo más interesante: a horizontes largos gana un modelo LINEAL.** Ridge directo es el mejor modelo individual en 25-48h y 49-72h, por encima de las redes neuronales y de los árboles. Tiene sentido físico: a 2-3 días de distancia ya no queda información explotable de la dinámica horaria de corto plazo, y lo que queda es nivel de fundamentales (hidrología) más estacionalidad semanal — una estructura que un lineal captura sin sobreajustar y que además **extrapola**, cosa que los árboles no pueden hacer por construcción. Esto también explica por qué CatBoost con el doble de datos empeoró: el problema a horizonte largo no era falta de datos.

**Ensamble v2 (`ensamble_72h_v2.py`) — NNLS con 9 votantes, pesos por tramo, validación cruzada de 5 pliegues por día:**

| Tramo | N-BEATSx (antes) | Ensamble v2 | Mejora | DM vs línea base |
|---|---|---|---|---|
| 1-24h | 58.68 | 57.42 | −2.2% | p=0.598 (sin diferencia) |
| 25-48h | 76.20 | **67.07** | **−12.0%** | **p=0.023 — GANA (sig.)** |
| 49-72h | 98.56 | **86.67** | **−12.1%** | **p=0.037 — GANA (sig.)** |
| Global 1-72h | 77.81 | **70.38** | **−9.5%** | MAPE 20.98% |

Esta vez el ensamble **sí** le gana a la línea base con significancia estadística en los dos tramos largos, a diferencia del intento de anoche. La diferencia es exactamente la prevista por la lección central del proyecto: anoche había 2 votantes de la misma familia; ahora hay tres familias reales (redes, árboles, lineal) y los pesos NNLS se reparten entre ellas (en 49-72h: Ridge 0.258, CatBoost 0.232+0.151, N-BEATSx 0.167+0.038, naive 0.080).

**rMAE contra la persistencia estacional de 7 días** (referencia de Lago et al. 2021, que antes no existía para el horizonte largo): 0.470 / 0.525 / 0.698 en 1-24h / 25-48h / 49-72h. El modelo le gana al benchmark ingenuo con holgura en todo el rango.

**Bandas**: calibración conforme adaptativa (ventana móvil 30 días) sobre el ensamble → cobertura 78.0% / 76.9% / 75.1% (objetivo 80%), anchos 195 / 234 / 293 COP/kWh.

**Honestidad sobre los límites de este resultado**:
- En 49-72h, Ridge directo solo (85.52) es marginalmente mejor que el ensamble (86.67), y la diferencia no es significativa (p=0.764). Se recomienda el ensamble por robustez y porque trae bandas calibradas, pero **Ridge solo es una alternativa defendible y mucho más simple** si se prioriza interpretabilidad.
- En 1-24h nada superó a N-BEATSx de forma significativa. Para ese tramo sigue siendo mejor el ensamble de stacking de 24h dedicado (MAPE 12.05%).
- La evaluación de 72h usa solo 72 cortes (uno cada 3 días, `step_size=72`). Usar `step_size=24` daría 216 ventanas y mucha más potencia estadística a los tests DM, a costo de inferencia adicional. Queda como mejora metodológica pendiente.

**Recomendación final por horizonte**: h=1-24 → ensamble de stacking de 24h (`stacking_ensamble_cv_2026.csv`, MAPE 12.05%); h=25-72 → ensamble v2 de 72h (`pronostico_ensamble_72h_v2_adaptativo.csv`), que reemplaza a `pronostico_con_bandas_72h_2026_adaptativo.csv` como contrato de OE3.

### 2026-09-11 (madrugada) — Extensión al horizonte completo 24-72h: N-HiTS a 72h, ensamble (resultado negativo honesto) y calibración adaptativa

Pedido explícito del usuario antes de dormir: no dejar el avance de hoy (el ensamble de stacking, MAPE 12.05%) limitado al punto de 24h, dado que el Anexo 1 (OE2.2) exige comparar en todo el rango 24-72h. Trabajo completado de forma autónoma durante la noche.

**Verificación previa**: los cambios de fuzzy ONI (agregados y revertidos más temprano hoy) nunca tocaron las columnas que usa el pipeline de 72h (`volumen_embalses`, `aportes_hidricos`, `demanda_lag24h`, `oni`, festivos, calendario) — solo agregaban columnas nuevas, después removidas limpiamente. Por eso **no hizo falta reentrenar N-BEATSx a 72h** (`pronostico_con_bandas_72h_2026.csv`, ya existente, sigue siendo válido).

**N-HiTS a 72h** (`scripts_experimento/bandas_incertidumbre_nhits_72h.py`, espejo exacto del script de N-BEATSx) — entrenado con `h=72`, `MQLoss`, mismas exógenas y festivos. Resultado: MAE por tramo 64.59 (1-24h) / 82.01 (25-48h) / 108.69 (49-72h) — **claramente peor que N-BEATSx-72h en los 3 tramos** (58.68 / 76.20 / 98.56) — a diferencia de h=24, donde N-HiTS y N-BEATSx quedaban prácticamente empatados. Tiene sentido: la arquitectura de N-HiTS (interpolación multi-tasa, diseñada para ser liviana) se degrada más rápido que la descomposición basis-expansion de N-BEATSx a medida que crece el horizonte.

**Ensamble de stacking N-BEATSx+N-HiTS por tramo de horizonte** (`scripts_experimento/ensamble_72h.py`) — **resultado negativo honesto**: el ensamble de 2 modelos **no le gana a N-BEATSx solo de forma significativa en ningún tramo** (DM test: p=0.37 / 0.75 / 0.61 en 1-24h / 25-48h / 49-72h respectivamente), y en el tramo 1-24h queda incluso marginalmente peor (58.81 vs 58.68). Consistente con la lección central de toda la sesión: **la diversidad entre familias es lo que hace ganar al stacking, no combinar dos modelos**. A diferencia del ensamble ganador de 24h (5 familias genuinamente distintas: árboles, econométrico, y dos arquitecturas neuronales), aquí solo hay 2 modelos de la misma familia (`neuralforecast`), y uno domina claramente al otro — no hay error diversificable que aprovechar. Pesos NNLS promedio: N-BEATSx 71-96% según el tramo, N-HiTS 4-29%.

**Calibración conforme adaptativa sí funcionó, y bien** (misma metodología ya probada para N-BEATSx solo): cobertura cruda 45-62% → adaptativa 75-79% (objetivo 80%), con el ancho de banda creciendo correctamente con el horizonte (195.9 → 271.6 → 317.5 COP/kWh). Guardado en `pronostico_ensamble_72h_2026_adaptativo.csv`.

**Recomendación final por horizonte** (dado que el ensamble de 24h no se puede extender más allá de 24h sin 71 reentrenamientos adicionales de XGBoost/ARX+GARCH/Persistencia — estructuralmente atados a `precio_lag24h` — scope explícitamente fuera de alcance para esta ronda):

| Horizonte | Modelo recomendado | MAPE / MAE | Archivo |
|---|---|---|---|
| **Exactamente 24h** | Ensamble de stacking (5 modelos, NNLS) | MAPE 12.05% / MAE 44.51 | `stacking_ensamble_cv_2026.csv` |
| **25h a 72h** | N-BEATSx nativo h=72 + calibración adaptativa | MAE 58.68→98.56 según tramo | `pronostico_con_bandas_72h_2026_adaptativo.csv` (sin cambios, ya validado) |

Verificación cruzada: el propio N-BEATSx-72h, evaluado exactamente en su paso de 24h, da MAPE 14.03% (MAE 45.32, n=72 ventanas no solapadas) — **peor que el ensamble dedicado a 24h (12.05%, n=5184)**, confirmando que vale la pena mantener el modelo especializado en 24h aparte del multi-horizonte, en vez de usar uno solo para todo el rango.

### 2026-09-11 (madrugada) — TFT como 6º votante: abortado por costo computacional, no por resultado

Idea #2 de la lista (Transformer como votante adicional del ensamble, inspirado en T2V-TE de *Energies*). `scripts_experimento/transformer_sexto_votante.py`. Primer intento con `VanillaTransformer` de `neuralforecast` falló de inmediato: esa arquitectura **no soporta variables exógenas históricas** (`hist_exog_list`), y perder `volumen_embalses`/`aportes_hidricos`/`demanda_lag24h` no era aceptable. Se cambió a **TFT (Temporal Fusion Transformer)**, que sí soporta histórico + futuro conocido — arquitectura correcta para el caso.

**Se abortó tras más de 2 horas de entrenamiento sin completar ni la primera fase** (217 ventanas walk-forward a h=24, sin GPU disponible). El proceso seguía consumiendo CPU activamente (no estaba colgado — el tiempo de CPU acumulado creció de forma lineal entre chequeos), simplemente el mecanismo de atención de TFT es mucho más caro por ventana que la descomposición basis-expansion de N-BEATSx/N-HiTS en esta máquina sin GPU. Dado que la prioridad de la noche era la extensión del horizonte 24-72h (pedida explícitamente), se detuvo el proceso para no arriesgar esa entrega — decisión reversible, ningún dato se perdió. **Pendiente para una sesión futura**: repetir con menos `max_steps`, una arquitectura TFT más chica, o en una máquina con GPU — la idea en sí (diversidad de familia para el ensamble) sigue siendo válida, solo el costo de esta arquitectura específica en CPU no calzó con el tiempo disponible.

### 2026-09-11 — Optuna (búsqueda Bayesiana) + armónicos de calendario para XGBoost/CatBoost

Ideas #1 y #3 de la lista inspirada en los papers colombianos revisados hoy: (1) búsqueda Bayesiana de hiperparámetros (inspirado en que el paper T2V-TE de *Energies* afinó **todos** sus baselines con Optuna, no solo el modelo propuesto — comparación justa); (3) armónicos adicionales de calendario (2º y 3er orden de hora-del-día y día-del-año, además de los ya existentes de 1er orden) — formalmente una **expansión en serie de Fourier truncada** del componente periódico de la señal, no una codificación cíclica ad-hoc. Conecta directamente con la caracterización espectral inicial del proyecto (periodograma/FFT, notebook 03) — vale la pena citarlo en la justificación de "ingeniería electrónica" del Anexo 1: cierra el círculo entre la etapa de caracterización de la señal y su uso explícito en el modelo. `scripts_experimento/optuna_armonicos_xgb_catboost.py`. Validación: train hasta 2024, hiperparámetros elegidos con 2025, reentrenado sobre 2019-2025 completo, evaluado en el holdout 2026 (nunca se usó 2026 para elegir hiperparámetros).

**Resultado**: **XGBoost afinado con armónicos mejora de forma significativa** — MAE 61.15→59.23, MAPE 15.89%→15.65%, DM p=0.00005. **CatBoost no mejora** con la búsqueda Bayesiana — MAE 57.92 vs 57.38 (el CatBoost sin afinar, ordered boosting por defecto), sin diferencia significativa (p=0.525) — se mantiene la versión original, más simple. Se actualiza el XGBoost que alimenta el ensamble de stacking con esta versión afinada.

### 2026-09-11 — CatBoost (punto 4 original): le gana a XGBoost, no reemplaza al ensamble

Punto 4 de la lista original de alternativas: CatBoost (ordered boosting), variante barata de probar. `scripts_experimento/catboost_benchmark.py`. Mismas 40 features y holdout 2026 que el resto del proyecto.

**Resultado: MAE 57.38, MAPE 15.19% — le gana a XGBoost de forma estadísticamente significativa** (MAE 61.15→57.38, DM p=0.00003), queda muy cerca de ARX+GARCH (55.76) y por debajo de N-BEATSx base. No supera al ensamble de stacking (44.51). Candidato razonable para sumar como sexto input al ensamble de stacking en una futura iteración (no se hizo todavía).

### 2026-09-11 — Referencia real en la literatura colombiana, y reentrenamiento de N-BEATSx antes de junio funciona

**Punto de referencia genuino encontrado en la literatura colombiana** (el usuario pidió explícitamente buscar uno, después de que los primeros candidatos revisados hoy resultaran no comparables o inválidos): **Gallón & Barrientos (2021), *International Journal of Energy Economics and Policy*, "Forecasting the Colombian Electricity Spot Price under a Functional Approach"** — precio horario real de Colombia (2000-2017), con prueba de Diebold-Mariano formal, el primer trabajo encontrado hoy que es genuinamente comparable en resolución y rigor.

Resultado del paper: modelo de series funcionales (FTS, componentes principales funcionales de Hyndman-Shang) da **MAPE 6.72% a 1 día** y **8.87% a 1 mes**, superando con significancia estadística a SARIMA (12.86%/15.42%), NNAR (14.64%/19.02%) y DSHW (13.31%/18.94%); casi empata con TBATS (9.91%/9.84%). **Salvedades honestas antes de comparar directo**: (1) datos 2000-2017, un período mucho más tranquilo que 2026, sin el súper Niño 2023-2024 ni la transición que estamos viviendo; (2) modelo puramente univariado, sin hidrología/ONI/demanda — limitación que el propio paper reconoce y deja como trabajo futuro; (3) el resultado a 1 mes se mide sobre un solo mes (enero 2018), ventana corta. **Conclusión útil**: incluso sus benchmarks univariados simples (sin régimen, sin fundamentales) dan MAPE de 9-19% en un período tranquilo — pone en contexto que nuestro 12.05% (ensamble de stacking), logrado en uno de los años más volátiles del histórico, está en un rango razonable para el mercado colombiano, no por debajo de lo esperable.

**Idea nueva #2 — reentrenar N-BEATSx justo antes de junio (no solo dejarlo congelado desde 2025)**: `scripts_experimento/nbeatsx_reentrenado_junio.py`. Se reentrenó N-BEATSx con corte en 2026-06-01 (incorporando ya enero-mayo 2026, que mostraba la aceleración del ONI) y se evaluó solo en junio, contra el mismo N-BEATSx con pesos congelados desde 2025-12-31 (walk-forward Origen 6). **Resultado: mejora significativa** — MAE 93.59 (congelado) → 86.21 (reentrenado), prueba de Diebold-Mariano t=2.695, p=0.007. Confirma la hipótesis: el problema de junio no es que el modelo sea malo, es que no se actualiza — ver información reciente (aunque sea solo hasta mayo) ayuda a anticipar mejor la transición de régimen.

### 2026-09-11 — Coeficientes que cambian en el tiempo (RecursiveLS) — confirma la adaptabilidad, con matices

Idea nueva #3: en vez de coeficientes fijos, un filtro recursivo (`statsmodels.RecursiveLS`, tipo Kalman) que actualiza los coeficientes con cada observación nueva — mismas 16 regresoras que ARX+GARCH (sin la parte GARCH). `scripts_experimento/recursive_ls.py`.

**Bug encontrado y corregido en el camino**: la inicialización difusa EXACTA por defecto (varianza inicial infinita) es numéricamente inestable en una serie de 60,625 observaciones — los coeficientes explotaban a ~10⁶-10⁷ sin sentido. Se corrigió usando inicialización difusa APROXIMADA (varianza inicial finita, `initialize_approximate_diffuse(variance=1e4)`), tras lo cual el ajuste convergió en 3 segundos a coeficientes razonables.

**Resultado**: coeficientes fijos (estimados al final de 2025) dan MAE 59.82/MAPE 15.74% — parecido a ARX+GARCH. Coeficientes recursivos (actualizados observación a observación durante todo 2026) dan MAE 59.33/MAPE 15.68% — **mejora estadísticamente significativa** (DM t=4.843, p=0.00000), aunque la magnitud es chica en términos absolutos. En junio específicamente la mejora es mínima (19.96%→19.89%). **Confirma, con un segundo método independiente (el primero fue el reentrenamiento de N-BEATSx), que dejar que el modelo se actualice con información reciente sí ayuda** — aunque en este caso lineal el efecto es pequeño porque el modelo en sí (ARX simple) ya está lejos del ensamble de stacking en capacidad predictiva.

### 2026-09-11 — Corrección adaptativa del punto (rolling bias) — falla, y confirma por qué la corrección de sesgo no sirve

Idea motivada por el hallazgo de que el modelo a veces sobreestima y a veces subestima entre años: en vez de un ajuste de sesgo global fijo (ya descartado, `correccion_sesgo_2026.py`, DM p=0.682), un ajuste que se recalcula cada día con el error real de los últimos N días — móvil, no fijo. `scripts_experimento/correccion_adaptativa_punto.py`, aplicado sobre el ensamble de stacking (mejor resultado actual). Procedimiento estrictamente secuencial: la corrección del día t solo usa errores de días *antes* de t.

**Resultado: la corrección adaptativa EMPEORA el error.** Ventana de 7 días: MAE 48.79→50.22 (DM p=0.0095, empeora significativamente). Ventana 14 días: empeora significativamente (p=0.0485). Ventana 30 días: sin diferencia. En junio específicamente (el mes problemático), las 3 ventanas empeoran. **Conclusión**: el sesgo que cambia de dirección opera a escala de **años**, no de semanas — una ventana de 7-30 días es demasiado corta para capturar una señal estable y termina persiguiendo ruido día a día en vez de corregir un sesgo real. Confirma, con un enfoque distinto, la misma conclusión de la corrección de sesgo global de anoche: no hay una corrección de sesgo barata que valga la pena para este modelo.

### 2026-09-11 — Modelo de dos etapas (hurdle): mejora sobre XGBoost solo, pero no supera al ensamble

Tercer intento, inspirado directamente en el paper de AES Colombia (Herrera-Mejía et al. 2025) revisado hoy: en vez de un solo XGBoost prediciendo todo el rango de precios, un **clasificador** (XGBoost, `scale_pos_weight` por desbalance) predice si la hora va a estar en "régimen alto" (precio > percentil 80 de train = 491.48 COP/kWh) usando solo variables conocidas de antemano, y esa clasificación enruta a uno de **dos regresores especializados** (uno entrenado solo con horas altas de train, otro solo con horas bajas). `scripts_experimento/hurdle_dos_etapas.py`.

**Resultado**: clasificador con accuracy 94.5% / F1 0.917 (identifica bien el régimen usando info pre-conocida). El modelo completo da MAE 57.94, MAPE 16.38% — **le gana a XGBoost de un solo modelo de forma significativa** (MAE 61.36→57.94, DM p=0.049), y el sesgo por decil mejora en el extremo barato (D1 -20.96% vs. peor en el XGBoost normal), pero **sigue por debajo de ARX+GARCH (55.76) y muy por debajo del ensamble de stacking (44.51, punto 1)**. Conclusión: separar el problema por régimen sí ayuda (confirma la intuición), pero el ensamble de stacking ya captura esa ganancia y más, así que no reemplaza la recomendación del punto 1.

### 2026-09-11 — Markov-Switching (2 regímenes) — empata con ARX+GARCH, no ayuda, y se entiende por qué

Segundo intento de la lista de alternativas para bajar del 15-16% de MAPE: en vez de darle el régimen al modelo como covariable (ONI), dejar que lo infiera él mismo de los datos. `scripts_experimento/markov_switching.py` — `MarkovRegression` (statsmodels) con 2 regímenes, intercepto y varianza que cambian según el régimen inferido, exógenas (hidrología, calendario, festivos, precio rezagado) con coeficiente fijo entre regímenes (mismo criterio de parsimonia que ARX+GARCH, para comparación limpia).

**Resultado: MAE 55.76 (idéntico a ARX+GARCH), MAPE 15.58%, DM test p=0.993 — sin diferencia significativa.** No ayudó. La razón queda clara mirando los parámetros ajustados: el Régimen 1 que el algoritmo EM encontró tiene una varianza 73 veces mayor que el Régimen 0, pero **solo dura ~10 horas en promedio** (persistencia `p[1->0]=0.10` por hora) — es un régimen de picos de volatilidad de corto plazo (tipo ráfaga, similar a lo que ya captura GARCH), no el macro-ciclo de El Niño/La Niña que buscábamos, que opera en escala de meses. A resolución horaria, el EM encuentra primero la estructura de 2 estados más fácil de detectar en los datos, y esa no es la relevante para nuestro problema. Para capturar el régimen correcto probablemente habría que ajustar el modelo sobre la serie agregada diaria (no horaria) o dejar que también las exógenas cambien de coeficiente entre regímenes — no se hizo por costo/beneficio dado que el ensamble de stacking (punto 1) ya dio una mejora mucho mayor y confirmada.

### 2026-09-11 — Ensamble entre familias (stacking): nuevo mejor resultado del proyecto, MAPE 12.05%

Buscando bajar el MAPE de ~15-16% (que llevaba varias sesiones estancado), se probó lo único de tipo "ensamble" que faltaba: combinar modelos de **distintas familias** (no solo semillas/ventanas dentro de N-BEATSx, que ya se había hecho). `scripts_experimento/stacking_ensamble.py` y `stacking_ensamble_cv_2026.py`.

**Metodología**: un meta-modelo NNLS (combinación convexa, pesos ≥ 0, sin intercepto — más robusto que Ridge con pesos libres, que dio resultados inestables con pesos negativos y a veces perdió) combina las predicciones hora por hora de los 5 modelos (Persistencia, XGBoost, ARX+GARCH, N-BEATSx, N-HiTS). Validado con **validación cruzada de 5 pliegues estratificados por día sobre el 100% de Origen 6 (2026)** — cada día se predice con pesos ajustados solo con los otros 4 pliegues, sin fuga de datos.

**Resultado**: MAE 44.51 (vs. 46.82 del mejor individual, N-BEATSx single) — **MAPE 12.05%**, comparado con 15.43% de ARX+GARCH y 16.18% del ensamble de 5 semillas de N-BEATSx (el mejor resultado anterior del proyecto). **Le gana de forma estadísticamente significativa a los 5 modelos individuales** (Diebold-Mariano: N-BEATSx p=0.00002, N-HiTS p=0.011, ARX+GARCH/Persistencia/XGBoost p<0.0001 todos). Pesos promedio entre pliegues: N-BEATSx 0.530, N-HiTS 0.260, Persistencia 0.113, ARX+GARCH 0.093, XGBoost 0.004 (prácticamente descartado por el meta-modelo, información redundante con los demás).

**Nuevo modelo recomendado para OE3**: el ensamble de stacking (NNLS), no un modelo individual. Pendiente: generar bandas de incertidumbre para el ensamble (las actuales son solo de N-BEATSx individual) y recalcular las métricas de decisión (Corr-f/MHD/MPD) sobre sus predicciones.

### 2026-09-10 (noche, cierre) — rMAE extendido a los 6 orígenes, e informe comparativo consolidado

**rMAE por origen y modelo (no solo el holdout único de 2026)** — `scripts_experimento/rmae_walkforward_completo.py`. Nota metodológica: en el MISMO periodo de test, rMAE es solo MAE reescalado por una constante — no cambia el ranking entre modelos (lo dice el propio Lago et al. 2021). Donde sí aporta es **entre orígenes**: el naive-7d varía 30x en dificultad entre regímenes (MAE 7.95 en Origen 2 tranquilo, MAE 242.49 en Origen 4 El Niño fuerte), así que comparar MAE crudo entre orígenes es tan engañoso como comparar MAPE entre meses.

Con rMAE aparece algo que el DM test no mostraba con la misma claridad: **ARX+GARCH pierde contra el naive semanal trivial en 2 de los 3 orígenes de La Niña** (rMAE = 1.036 en Origen 1, 0.990 en Origen 2 — peor que "predecir el mismo precio de hace una semana"), con la variabilidad más alta de los 5 modelos (desv. std. 0.299). **N-BEATSx tiene la ventaja relativa más grande justo en los 3 orígenes de El Niño** (rMAE 0.338/0.268/0.379 en Origen 4/5/6 — el mejor de los 5 modelos en los tres) y el rMAE promedio más bajo de todos junto con Persistencia (0.496). Refuerza la recomendación de N-BEATSx para OE3 con una dimensión que el DM test no capturaba: no solo gana de forma significativa, su ventaja es más grande justo donde más importa (los regímenes volátiles), mientras que ARX+GARCH es inestable entre regímenes pese a su buen MAE puntual en 2026.

**Informe comparativo de modelos consolidado** — `docs/informe_comparativo_modelos.md` (nuevo, deliverable formal de Fase 3, pendiente desde hace días). Reúne en un solo documento lo que hasta ahora vivía disperso en la bitácora: metodología, comparación de las 5 familias de modelo (holdout único y walk-forward de 6 orígenes), Diebold-Mariano, rMAE por origen, bandas de incertidumbre (72h, adaptativas), métricas de decisión (Corr-f/MHD/MPD, con la corrección del 2026-09-10), festivos (única mejora confirmada con significancia estadística), y los intentos fallidos documentados con la misma honestidad que los exitosos (LEAR, 4 hipótesis de El Niño, lógica difusa ONI, análogos de régimen k-NN) — con recomendación final para OE3.

### 2026-09-10 (noche, continuación) — Julio diagnosticado, hipótesis de vacaciones descartada, análogos de régimen (k-NN) fallan, integración de lógica difusa en curso

**Julio 2026, mismo tratamiento que junio** — `scripts_experimento/diagnostico_julio_2026.py`. Historia distinta a junio: no hay racha (1 solo día con MAPE>25%, el 26 de julio, un outlier aislado con sesgo -33.4%), pero **domingo es brutal** (21.38% MAPE vs 11.61% el peor día de semana) y **las horas 8-11am concentran el error** (22-24% MAPE, mucho peor que el resto del día). El sesgo por decil de precio replica el patrón ya conocido (D1=-32.71%, D10=+7.66%). Julio es objetivamente más tranquilo que junio en términos absolutos (MAPE 10.50% vs 19.78%, desviación estándar del error 67.46 vs 99.42) — coherente con el hallazgo de rMAE de la entrada anterior: julio es "fácil" en términos absolutos pero el modelo agrega poco valor relativo porque el naive semanal también es inusualmente bueno ese mes.

**Hipótesis de vacaciones de mitad de año — descartada con test directo** — `scripts_experimento/test_vacaciones_demanda.py`. En vez de inferirlo indirectamente (correlación anomalía de demanda vs error del modelo, ya negativa en los 6 orígenes), se probó la premisa misma: ¿cae la demanda de forma medible en la ventana 15 junio-15 julio, en cada año 2019-2026? Resultado: **0 de 8 años muestran una caída de demanda estadísticamente significativa** (prueba HAC sobre la anomalía promedio vs 30 días; anomalía promedio across años: -0.15%, ruido puro). La hipótesis de vacaciones queda descartada en su raíz — no vale la pena construir `es_vacaciones_escolares` como feature.

**Análogos de régimen vía k-NN fuzzy (Lago et al., citando De Marcos et al. 2020 y Nitka et al. 2021) — falla, y se entiende por qué** — `scripts_experimento/analogos_regimen_knn.py`. Se probó calibrar XGBoost solo con los 12 bloques históricos de 30 días más parecidos (distancia euclidiana en el espacio fuzzy ONI de 7 dimensiones) al régimen justo antes de cada origen, en vez de con todo el histórico cronológico. Resultado: **pierde de forma significativa en los 2 orígenes de El Niño** (Origen 4: MAE 240.10 vs 143.23 cronológico, p<0.0001; Origen 6: MAE 136.05 vs 61.31, p<0.0001), empata en Origen 1. Causa identificada: el análogo usa solo ~8,000 filas (vs 40,000-60,000 del cronológico) — reducir ~80-85% de los datos de entrenamiento le hace más daño a XGBoost que cualquier beneficio de "parecerse al régimen actual", porque el descriptor fuzzy del ONI no captura la tendencia secular del precio (crecimiento de demanda y cambios de mercado 2019-2026) que sí aprovecha ver todo el histórico. Documentado como intento fallido, mismo criterio que las 4 hipótesis de "atacar El Niño" de anoche.

**Lógica difusa del ONI: probada e integrada al pipeline compartido, y luego REVERTIDA** — mismo mecanismo que festivos: 9 columnas nuevas (`fuzzy_nino_debil/moderado/fuerte/muy_fuerte`, `fuzzy_nina_debil/moderado/fuerte/muy_fuerte`, `fuzzy_transicion_rapida`) agregadas en `05_features_compartidas_juan.ipynb`. **Bug encontrado y corregido en el camino**: `fuzzy_nina_fuerte` y `fuzzy_nina_muy_fuerte` son cero constante en todo el histórico 2019-2026 (nunca hubo una Niña con \|ONI\|>=1.5) — rompían la matriz de ARX+GARCH por colinealidad perfecta (`LinAlgError: SVD did not converge`); se excluyeron, quedando 7 columnas fuzzy activas.

**Resultado (notebook 08 y 09 reejecutados con fuzzy)**: mejora mínima en ARX+GARCH (55.76→55.63, dentro del ruido) pero **empeora claramente los 3 modelos de deep learning** — N-BEATSx base 56.11→58.47, N-HiTS base 57.08→60.53, y el ensamble de 5 semillas (mejor resultado de la sesión) 54.39→55.44. Interpretación: "oni" ya era continuo en el pipeline, así que las 7 columnas fuzzy no aportan información nueva, solo más dimensionalidad exógena — una de ellas (`fuzzy_nino_muy_fuerte`) activa en solo ~1,100 de 60,625 horas, casi puro ruido para una red neuronal. **Decisión (con el usuario): revertir la fuzzy del pipeline compartido antes de gastar las varias horas que tomaría reejecutar notebook 10** (6 orígenes completos, la reejecución de 09 solo ya tardó ~2h10min) — la señal de 09 ya era suficientemente clara. Datasets, notebook 05/08/09/10 revertidos a solo-festivos; festivos se mantiene intacto. El prototipo y el experimento completo quedan documentados en `fuzzy_oni_prototipo.py` para referencia futura, mismo criterio que las ideas fallidas de "atacar El Niño".

### 2026-09-10 (noche) — Diagnóstico multi-año, junio 2020-2025, prototipo de lógica difusa ONI, y rMAE (Lago et al. 2021)

**1) Diagnóstico completo replicado en los 6 orígenes (2020-2026)** — `scripts_experimento/diagnostico_completo_anios.py`. De los 5 ángulos probados en 2026, solo 2 se confirman con fuerza en el resto del histórico: (a) el sesgo invertido en los deciles de precio (sobreestima en precios bajos, subestima en los picos) se repite en 5/6 orígenes (excepción: Origen 5/2024); (b) las rachas de días consecutivos malos están ligadas a transiciones de régimen, no a un mes específico — la racha más larga de toda la historia es Origen 4 (2023-10-29 a 11-06, 9 días), durante el arranque del súper Niño, no junio-2026. Domingo/festivo como "días difíciles" NO se sostiene fuera de 2026 (muestra muy chica, 1-3 festivos por ventana de 3 meses). Anomalía de demanda como proxy de vacaciones da correlación débil (|r|<0.17) en los 6 orígenes — resultado negativo consistente. Velocidad de cambio del ONI (pooled, 23 puntos mes-origen) correlaciona 0.254 con el MAPE mensual, más que el nivel del ONI (0.134).

**2) Entrenamiento de junio 2020-2025 (Persistencia, ARX+GARCH, XGBoost, N-BEATSx, N-HiTS) para confirmar si la racha de junio-2026 es estacional** — `scripts_experimento/walkforward_junio_anios.py` (`walkforward_junio_predicciones_crudas.csv`). **No se repite en ningún otro año**: la racha más larga fuera de 2026 es de 2 días (junio 2023); todos los demás junios tienen 0-1 día aislado. Confirma que junio-2026 es un evento atípico del año, no un patrón estacional — consistente con el hallazgo de velocidad de ONI del punto 1. Bonus: la ventaja de los modelos de deep learning sobre la persistencia es dependiente del régimen — ganan con margen en años de transición fuerte (2023, 2024) y pierden o empatan en años tranquilos (2021, 2022, 2025); en 2020 (pandemia) son claramente peores que la persistencia.

**3) Prototipo de lógica difusa sobre el ONI** — `scripts_experimento/fuzzy_oni_prototipo.py` (`fuzzy_oni_preview.csv`). El ONI ya es continuo en el pipeline (no binario), pero se probó codificar el umbral NOAA (débil/moderado/fuerte/muy fuerte) como funciones de membresía triangulares, más una variable `fuzzy_transicion_rapida` (velocidad de cambio en ventana de 30 días). Resultado notable: **junio-julio 2026 tiene `fuzzy_transicion_rapida=1.00`, el máximo de todo el histórico 2019-2026** — la transición de régimen más rápida jamás vista en los datos, más rápida incluso que el arranque del súper Niño 2023. Esto explica por qué junio-2026 tuvo rachas y ningún otro junio las tuvo. **Pendiente de decisión**: solo es un prototipo/vista previa, no está integrado al pipeline compartido (`05`) ni probado en los modelos — requeriría otra ronda de reentrenamiento tipo festivos.

**4) rMAE mensual de N-BEATSx en 2026, siguiendo Lago et al. (2021, Applied Energy 293:116983)** — `scripts_experimento/rmae_mensual_2026.py` (`rmae_mensual_nbeatsx_2026.csv`). El paper argumenta que el MAPE es una métrica poco confiable (se distorsiona con precios bajos) y recomienda rMAE = MAE(modelo)/MAE(naive 7 días). rMAE total 2026 = 0.379 (el modelo comete ~38% del error del naive semanal). La correlación entre el ranking mensual por MAPE y por rMAE es solo 0.432: **por MAPE junio es el peor mes (19.78%), pero por rMAE julio es el peor (0.584 vs 0.500 de junio)** — julio tiene MAPE moderado (10.50%) pero es donde el modelo agrega menos valor relativo sobre el naive semanal, porque el naive-7d es inusualmente bueno ese mes. Julio merece la misma atención que junio, algo que el MAPE por sí solo no mostraba. También se revisó el LEAR/checklist del mismo paper: confirma que el proyecto ya le gana al benchmark publicado del área (ver entrada del 2026-09-10 01:16) y señala dos limitaciones honestas del walk-forward frente a las mejores prácticas del campo — Orígenes 1-5 son ventanas de 3 meses (el paper recomienda mínimo 1 año) y no hay recalibración diaria (se reentrena una vez por origen) — para documentar en el informe comparativo de Fase 3.

**Conclusión de la noche**: la hipótesis de "vacaciones de junio" queda descartada como explicación principal (anomalía de demanda negativa en todos los años); la explicación que sobrevive todas las pruebas es la **velocidad de transición del régimen ONI**, no el mes ni el nivel absoluto de El Niño/La Niña.

### 2026-09-10 (tarde) — Notebook 10 reejecutada con festivos: confirma 2026, matiz en Origen 1 (2020, pandemia)

`10_diebold_mariano_juan.ipynb` reejecutada con festivos integrados en las 5 familias de
modelo (misma edición que 08/09). Comparado contra la corrida anterior (sin festivos):

**Origen 6 (2026, el holdout que realmente importa) — sin cambios de fondo, todo sigue
ganando de forma significativa**: N-BEATSx y N-HiTS le ganan a ARX+GARCH (p≈5e-05) y a la
persistencia (p≈3e-05) en ambos casos. Ningún retroceso aquí.

**Origen 1 (2020-07 a 09, La Niña inicio, plena pandemia) — único origen que empeora**:
N-BEATSx y N-HiTS pasan de "sin diferencia significativa" contra la persistencia a
**perder de forma significativa** (p<1e-10 en ambos, muy fuerte). Hipótesis: los festivos
de 2020 no se comportaron como festivos normales por los confinamientos estrictos (la
gente no salía, el consumo no bajaba igual) — la feature de festivo probablemente
"engañó" al modelo justo en ese origen atípico. Es el único de los 6 orígenes con
confinamiento estricto vigente, coincide con la hipótesis.

**Resto de orígenes (2, 3, 4, 5) — sin cambios relevantes**, mismo patrón de siempre
(N-BEATSx/N-HiTS ganan a ARX+GARCH consistentemente; contra persistencia depende del
origen). Resumen actualizado: N-BEATSx vs ARX+GARCH 6/6 gana; N-BEATSx vs Persistencia
3/6 gana, 1/6 pierde (Origen 1), 2/6 sin diferencia; N-HiTS vs ARX+GARCH 5/6 gana, 1/6
sin diferencia (Origen 1); N-HiTS vs Persistencia igual que antes.

**Veredicto**: festivos se mantiene como mejora neta — el único origen afectado es uno
atípico (pandemia) que no representa el régimen operativo normal, y el holdout que
importa para la recomendación final (2026) mejora sin matices. Vale la pena que quede
anotado como limitación conocida si se defiende el resultado ante los asesores.

### 2026-09-10 — Motor de decisión (OE3): v1, desacople del modelo, y primer swap de fuente

Arrancó el motor de decisión (OE3). Tres piezas nuevas:

- **`src/motor_decision.py`** — lógica reusable, fuera de notebooks. Traduce un pronóstico con bandas
  `[q10,q50,q90]` en una señal comprar/vender/esperar según el rol (generador: {vender, retener,
  esperar}; comercializador: {comprar, evitar_compra, esperar}). Tres métodos de umbral por
  percentil: `fijo` (percentiles del histórico 2019-2025), `rodante` (ventana móvil causal de 30
  días), `banda` (fijo + forzar "esperar" cuando la banda de incertidumbre es ancha = poca
  confianza). Se comparan con un backtest económico simple (precio real promedio en horas de acción
  vs. promedio general) y se elige el ganador por rol/horizonte.
- **`notebooks/12_motor_decision_Rafa.ipynb`** — carga las dos fuentes, corre el backtest, guarda las
  señales finales (`data/processed/resultados/senales_motor_decision_2026.csv`).
- **`dashboard/app.py`** — prototipo Streamlit: selector de rol/horizonte/método/percentiles, gráfica
  precio+banda+señal, KPIs de ventaja económica recalculados en vivo. Muestra el modelo activo por
  horizonte (verde=calibrado) y sugiere en vivo el método ganador del backtest. Verificado sin
  excepciones con `streamlit.testing.v1.AppTest` en las 4 combinaciones rol×horizonte.

**Desacople del modelo de pronóstico.** `data/processed/resultados/fuentes_pronostico.json` es el
único lugar que declara qué archivo/modelo alimenta cada horizonte (`archivo`, `modelo`, `calibrado`,
`cobertura_medida_pct`, ...). `cargar_fuente_pronostico(raiz, horizonte)` lee ese JSON, carga el CSV
y lo valida con `validar_contrato_pronostico()` (columnas obligatorias `fecha_hora, real, q10, q50,
q90`; NaN en los cuantiles = error; cuantiles cruzados = advertencia no bloqueante). Ni el notebook
ni el dashboard tienen rutas de archivo hardcodeadas fuera de ese JSON.

**El desacople se ejercitó de inmediato.** La v1 nació con un *fallback* de XGBoost a 72h (regresión
por cuantiles, cobertura cruda ~73%, sin calibrar) porque N-BEATSx todavía no tenía pipeline a 72h.
Ese pipeline llegó en el commit `3d8d685` (`pronostico_con_bandas_72h_2026_adaptativo.csv`, N-BEATSx
h=72 + CQR adaptativo por tramo, cobertura ~78%). Cambiar la fuente fue editar 5 campos del JSON y
re-ejecutar el notebook -- sin tocar `motor_decision.py` ni `app.py`. El CSV del fallback se borró.
El archivo de N-BEATSx a 72h trae columnas extra (`cutoff`, `paso_horas`, `margen`) que
`cargar_fuente_pronostico` ignora quedándose solo con las 5 del contrato.

Resultado del backtest sobre el holdout 2026 (ambos horizontes ya con N-BEATSx calibrado):

| Horizonte | Rol | Método ganador | Ventaja (COP/kWh) | Frecuencia de acción |
|---|---|---|---|---|
| 24h | generador | banda | 345.6 | 16.9% |
| 24h | comercializador | banda | 281.6 | 14.3% |
| 72h | generador | banda | 299.7 | 19.5% |
| 72h | comercializador | banda | 279.5 | 11.8% |

`banda` gana en los 4 casos: con bandas de incertidumbre bien calibradas (ancho que crece con el
error real), filtrar por confianza mejora la ventaja económica. Con el fallback de XGBoost sin
calibrar, a 72h el método `banda` quedaba fuera del rango de frecuencia válido (bandas
sistemáticamente demasiado anchas) y ganaba `fijo` -- otra confirmación de que la calibración de las
bandas de 72h sí importa para el motor, no es un detalle cosmético.

**Hallazgo de la validación**: las bandas de 24h (N-BEATSx adaptativo) tienen 2 filas de 5208 con
cuantiles cruzados (`q10>q50` o `q50>q90`, ambas el 2026-03-12 de madrugada). Es un borde conocido de
la calibración por cuantiles (no siempre quedan monótonos), no un bug del motor -- advertencia no
bloqueante, documentada en la guía.

**Guía nueva**: `docs/motor_decision_guia.md` -- arquitectura, contrato de datos, cómo correr
notebook/dashboard, y el procedimiento paso a paso ("Cambiar o agregar un modelo"), que ya se validó
en la práctica con el swap de 72h descrito arriba.

**Limitaciones explícitas** (en la guía): (a) el backtest es "precio promedio en horas de acción vs.
promedio general", no una simulación de portafolio con capacidad/contratos/costos de transacción;
(b) el motor decide solo con el nivel de `q50`, todavía no usa Corr-f/MHD/MPD de
`scripts_experimento/metricas_decision.py` como sugiere la nota para Rafael. Los dos son el siguiente
paso, no bloquean usar la v1 para explorar.

### 2026-09-10 01:16 — sesión nocturna: benchmark LEAR, bandas de incertidumbre, ensamble de ventanas, ataque a El Niño y festivos confirmados

Sesión larga (≈3h, en gran parte autónoma mientras Juan dormía) motivada por la lectura de más literatura de pronóstico de precio eléctrico (Lago et al. 2021, el benchmark/revisión de mejores prácticas del área) y por la pregunta de cómo bajar el error específicamente en régimen El Niño. Todo el código vive en `scripts_experimento/` sobre `main` directamente (no en una rama aparte esta vez), sin commitear nada — es reversible por default (nada está en el historial de git todavía). Resultados en `data/processed/resultados/*.csv`.

**1) LEAR y Ensemble LEAR (benchmark estándar del campo, Lago et al. 2021) — el proyecto no lo tenía.**
LEAR = regresión LASSO por hora del día (24 modelos independientes), con recalibración rodante cada 7 días. Se probaron 6 ventanas de calibración (56d, 84d, 365d, 728d, 1456d, completa) y sus ensambles. El mejor ensamble (56d+84d+1456d) dio MAE 74.94 en el holdout 2026 — **peor que todos los modelos actuales del proyecto** (Persistencia 56.31, ARX+GARCH 55.76, N-BEATSx 56.11). Confirma que el proyecto le gana al estándar publicado del área por un margen amplio, con un benchmark corrido de verdad y no solo asumido. Curiosidad: en LEAR las ventanas cortas (56-84 días) le ganaron por mucho a las largas (728d-completa, que dieron 100-116 de MAE) — un modelo lineal no puede "ignorar" el súper Niño 2023-2024 como sí puede hacerlo parcialmente una red no lineal, así que meterlo en la ventana de calibración sesga sus coeficientes lejos del nivel de precio de 2026.

**2) Bandas de incertidumbre para N-BEATSx (`MQLoss`, pendiente del backlog) — hecho, con un problema real encontrado y corregido.**
Entrenado con pérdida multi-cuantil (q10/q50/q90). Cobertura cruda de la banda [q10,q90]: **62.6%**, muy por debajo del 80% nominal — el modelo está sobre-confiado (patrón conocido de MQLoss, no es un bug). Corregido con calibración conforme (CQR): un ajuste estático (calibrado en ene-abr, evaluado en may-ago) solo llegó a 67.6% porque el precio medio se triplica entre esos dos tramos de 2026 (202→594 COP/kWh) — el mismo fenómeno de cambio de régimen de siempre, ahora dentro de un solo año de prueba. La solución que sí funcionó fue una calibración **adaptativa** (margen recalculado cada día con ventana móvil de 30 días de scores de conformidad), que llevó la cobertura a 76.9% y además hace que el ancho de la banda seatan realmente con la volatilidad (36 COP/kWh en febrero tranquilo, 373 en junio disparado). **Contrato final para el motor de decisión (OE3) de Rafael**: `data/processed/resultados/pronostico_con_bandas_2026_adaptativo.csv` (columnas `fecha_hora, real, q50, q10, q90`) — usar esta versión, no la cruda ni la estática.

**3) Variables nuevas del API de XM (gas/costo marginal, disponibilidad, intercambios con Ecuador, mantenimientos) — 12 descargadas, tamizadas por correlación parcial.**
Descargadas: `max_precio_oferta`, `costo_marginal_despacho`, `generacion_ideal`, `demanda_comercial`, `restricciones_aliviadas`, `perdidas_energia`, `importaciones_energia`/`exportaciones_energia` (Ecuador), `precio_escasez`, `precio_escasez_marginal`, `aportes_media_historica`, `compras_arranque_parada`, y `disponibilidad_por_tipo` (hidráulica/térmica, por recurso vía `ListadoRecursos`). Nota: XM no expone una métrica separada de "mantenimientos programados"; `DispoDeclarada` es el proxy más cercano que existe. Tamiz de correlación parcial (descontando `precio_lag24h`, igual método que con la variable H/T de la sesión anterior): 9 de 12 redundantes (correlación parcial <0.05), incluyendo `max_precio_oferta` que resultó ser **fuga de datos pura** (corr. contemporánea 0.9998 — es literalmente el precio marginal). Única candidata fuerte: `costo_marginal_despacho` (parcial 0.178, y sube a 0.199 específicamente en horas de El Niño). Débiles: `compras_arranque_parada` (invisible en promedio, 0.199→ok, -0.102 específicamente en El Niño) y `exportaciones_energia`.

**4) Ensamble de ventanas de calibración en N-BEATSx y N-HiTS (mismo mecanismo de Lago et al., aplicado a los modelos de deep learning) — mejora direccional, no confirmada estadísticamente.**
Igual que con LEAR pero para los dos modelos de deep learning: se entrenó cada uno con 5-6 ventanas (6m/1a/2a/3a/5a/completa) y se promediaron sus pronósticos. Mejor combinación en N-BEATSx (3a+5a+completa): **MAE 54.34** vs 56.11 de la ventana única — pero la prueba de Diebold-Mariano (misma metodología de la notebook `10`) da **p=0.075, no significativo al 5%**. En N-HiTS el mismo mecanismo (1a+2a+5a+completa): MAE 55.85 vs 57.08, con **p=0.375, claramente no significativo**. Veredicto: el ensamble de ventanas ayuda en la dirección correcta en los dos modelos de forma consistente, pero no pasa la vara de rigor que usa el resto del proyecto — se documenta como hallazgo sugestivo, no como mejora confirmada. Nota aparte: la mejor combinación para el **MAE global** (1a+2a+3a+5a+completa, 54.34) no es la misma que minimiza el error **específicamente en El Niño** (las 6 ventanas juntas, incluida la de 6 meses que individualmente es mala, dan El Niño=83.83 vs 85.13 de la otra combinación) — un trade-off menor, documentado en `ventanas_ensambles_por_regimen.csv`.

**5) Ataque directo al error de El Niño — 4 hipótesis probadas en N-BEATSx, las 4 fallaron.**
Motivado por un hallazgo previo de la noche: El Niño es 10.3% del entrenamiento (2019-2025) pero 30.4% del holdout 2026 — el modelo ve poco de lo que más le toca predecir. Se probó, contra un BASE de control (MAE en El Niño 90.79 ± 1.05, 3 semillas):
  - **DUP_NINO** (duplicar el episodio de entrenamiento ago2023-abr2024 como serie extra, para sobre-muestrear sin romper ventanas — la alternativa que se implementó porque `neuralforecast` 3.2.1 no expone ningún parámetro de peso por muestra, verificado por inspección de firma antes de gastar cómputo en la idea original): El Niño 90.45 ± **6.08** — misma media, pero 6x más inestable entre semillas. Descartado.
  - **VARS_NINO** (agregar `costo_marginal_despacho` y `compras_arranque_parada`, las variables que sí pasaron el tamiz): El Niño 92.51 — empeora, a pesar de tener correlación parcial real. Mismo patrón ya visto con la variable H/T de la sesión anterior.
  - **INTERACCION** (`oni × aportes_hidricos_media_7d`, `oni × volumen_embalses_vs_media30d`): El Niño 92.29 — empeora.
  - **TODO** (las tres combinadas): El Niño 92.36 — empeora, y con la mayor inestabilidad (±2.93).
  BASE gana en las 4 comparaciones. El error de El Niño no se resolvió con más información ni con más muestras del propio episodio — sigue abierto como problema.

**6) Festivos colombianos en N-BEATSx — CONFIRMADO con significancia estadística, corrigiendo la lectura de la tarde.**
La prueba de la tarde (3 semillas) había quedado inconclusa. Se extendió a 5 semillas (42, 7, 2024, 13, 99) y se aisló la métrica correcta: el MAE **global** sigue sin diferencia (56.72 vs 56.52, dentro del ruido, porque los festivos son solo 5.5% de las horas), pero el MAE **específicamente en horas festivas** mejora en las 5 de 5 semillas (68.2→58.8, 72.4→58.8, 77.9→72.2, 74.9→65.1, 77.0→58.4). Prueba t pareada sobre las 5 semillas: **t=5.22, p=0.0064** — significativo incluso al 1%. Las features que se probaron (`es_festivo`, `festivo_lag24h/48h/72h/168h`, `mismatch_festivo_24h/168h` — si el rezago cae en un día de tipo distinto al objetivo) se pasan como `futr_exog` porque el calendario de festivos se conoce con años de anticipación, sin riesgo de fuga. Complementario: se verificó por qué la comparación cruda inicial (martes festivo promedio 108 vs martes normal 348) era engañosa — solo hay 2 martes festivos en todo 2019-2026, ambos cayeron en 2020-2021 (los años más baratos de la serie); controlando por el nivel de precio local de esa semana, el efecto real de un festivo entre semana es cotizar ~11% por debajo de su semana, aterrizando casi exactamente en el nivel de un domingo (ratio 0.909 vs 0.896) — consistente con la intuición original de que "un festivo se comporta como domingo".

**Pendiente de decidir con Rafael/asesores**: si las features de festivo se integran formalmente al pipeline compartido (`05_features_compartidas_juan.ipynb`), dado que es la primera mejora de la noche con significancia estadística real. Igual que con la variable H/T, esto tocaría el Alcance del Anexo 1 solo si se considerara "desglose" de algo restringido — no es el caso aquí (festivos es información de calendario, no generación por tipo de recurso), así que no aplica esa restricción.

**7) Combinación festivos + ensamble de ventanas — no suma una mejora global confirmada, pero tampoco rompe la de festivos.**
Última prueba de la noche: entrenar N-BEATSx con las features de festivo (idea 6) en las 5 ventanas ganadoras del ensamble (idea 4: 1a+2a+3a+5a+completa). Resultado: MAE global 53.98 (el número más bajo de toda la noche), MAE en festivos 62.47. Pero el Diebold-Mariano no perdona: contra el oficial (ventana completa sola) da p=0.079 (no significativo, igual de borderline que el ensamble solo sin festivos); contra el ensamble de ventanas sin festivos da p=0.50 — es decir, **la diferencia entre combinar o no combinar es indistinguible de ruido puro**. La mejora específica en festivos sí se mantiene intacta (62.47 vs 74.07 sin festivos, consistente con el hallazgo ya confirmado de la idea 6), lo cual dice que combinar no perjudica el efecto real, solo que no hay evidencia de que sume algo adicional al MAE global más allá de lo que cada pieza ya aportaba por separado.

**Resumen ejecutivo de la noche, para no tener que releer todo**: de 7 ideas probadas (LEAR, bandas de incertidumbre, variables nuevas, ensamble de ventanas, 4 ataques a El Niño, festivos, y la combinación de las dos últimas), **la única con significancia estadística confirmada es festivos** (p=0.0064, mejora consistente en 5/5 semillas). Todo lo demás quedó en alguna de estas tres categorías: (a) confirmado como negativo (LEAR, las 4 ideas de El Niño), (b) direccionalmente positivo pero sin pasar el umbral de significancia (ensamble de ventanas, solo o combinado con festivos), o (c) un deliverable de infraestructura sin comparación de "mejor/peor" aplicable (bandas de incertidumbre calibradas, ahora con cobertura 77% en vez de 62%).

**Archivos de esta sesión** (todos en `scripts_experimento/` y `data/processed/resultados/`, nada commiteado todavía): `modelo_lear.py`, `lear_ensamble.py`, `bandas_incertidumbre_nbeatsx.py`, `calibrar_bandas.py`, `calibrar_bandas_adaptativo.py`, `descargar_variables_nuevas.py`, `descargar_disponibilidad.py`, `tamizar_variables_nuevas.py`, `probar_variables_nuevas.py`, `ventanas_calibracion.py`, `ventanas_calibracion_nhits.py`, `atacar_el_nino.py`, `festivos_en_nbeatsx.py`, `diagnostico_variable_ht.py` (de la sesión anterior), `confusion_festivos.py`.

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

### 2026-09-04 15:05 — Modelo LightGBM (`11`)

- **Nuevo notebook `11_modelo_lightgbm_Rafa.ipynb`**, cuarto modelo de árboles/ensamble evaluado, con el mismo pipeline de datos, las mismas 33 features y el mismo horizonte de 24h que `06` (XGBoost): mismo `dataset_features_2019_2025.csv`/`dataset_features_2026.csv`, mismas columnas excluidas (demanda/generación contemporáneas, versiones crudas de calendario), mismo log-transform del target y misma persistencia de referencia.
- Mismo protocolo de grid search que `06` (validación con 2025, sin tocar 2026): `max_depth` × `learning_rate`, luego refinamiento de `max_depth`. Ganador: `max_depth=4`, `num_leaves=15`, `learning_rate=0.01`.
- **Resultado en el holdout de 2026: MAE 62.16, RMSE 108.06, MAPE 15.90%** — muy cerca de XGBoost (61.15/107.32/15.89%) pero sin superarlo en ninguna métrica, y por detrás de ARX+GARCH y N-BEATSx/N-HiTS. Razón de error ONI alto/bajo: 3.27x (entre la de XGBoost, 3.66x, y la de ARX+GARCH, 3.43x).
- Feature importance nativa (ganancia total) coincide con el top de SHAP/permutation importance de XGBoost: `precio_lag24h` domina de forma clara, seguido de `precio_media_24h` y `precio_media_7d`.
- **Conclusión preliminar**: LightGBM no aporta una mejora sobre XGBoost en este dataset (tamaño moderado, 33 features tabulares) — el crecimiento leaf-wise no se tradujo en ventaja real aquí. No se justifica todavía incluirlo en el walk-forward (`07`) ni en Diebold-Mariano (`10`) a menos que el equipo decida evaluarlo formalmente igual; queda documentado como referencia adicional en la tabla de resultados.
- Nota de entorno: `lightgbm` no estaba en `requirements.txt` — falta agregarlo formalmente (se usó `lightgbm==4.7.0` en esta sesión).

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
