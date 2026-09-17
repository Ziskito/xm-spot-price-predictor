# Informe comparativo de modelos — Predicción del precio de bolsa de energía (XM, Colombia)

**Proyecto de grado, Universidad del Norte.** Autores: Juan David Barceló Barraza (OE1-OE2,
modelos de pronóstico), Rafael Andrés Dede Perdomo (OE3, motor de decisión). Documento formal de
Fase 3 (Anexo 1), consolidando los resultados de OE2 dispersos en la bitácora del `README.md`
hasta el 2026-09-10.

## 1. Objetivo

Comparar, con el mismo criterio de evaluación en todos los casos, las familias de modelo
entrenadas para pronosticar el precio de bolsa horario colombiano, y establecer cuál es el mejor
insumo para el motor de decisión de compra/venta/espera (OE3). El objetivo NO es solo minimizar
el error promedio: como se documenta en la Sección 5, el modelo con mejor MAE no es
automáticamente el mejor insumo para una regla de decisión.

## 2. Datos y metodología

- **Fuente**: 5 series de XM/SIMEM sincronizadas a resolución horaria en un dataset maestro
  (`dataset_maestro_2019_2025.csv`, `dataset_maestro_2026.csv`).
- **Entrenamiento**: 2019-01-01 a 2025-12-31 (60,625 filas). **Prueba (holdout único)**: 2026-01-01
  a 2026-08-05 (5,208 filas).
- **Validación walk-forward**: 6 orígenes temporales, cada uno con su propio corte de
  entrenamiento y ventana de prueba, cubriendo los distintos regímenes ENSO observados en el
  histórico (La Niña de intensidad variable, El Niño fuerte, El Niño 2026). Los orígenes 1-5 son
  ventanas de prueba de **12 meses consecutivos** (jul-2020→jun-2021, …, jul-2024→jun-2025, sin
  solape — el mínimo que recomienda Lago et al. 2021; ampliado desde 3 meses el 2026-09-11); el
  Origen 6 cubre todo 2026 disponible (8 meses).
- **Prueba de significancia**: Diebold-Mariano (HAC/Newey-West, `maxlags=23`), el estándar del
  campo (Diebold & Mariano, 1995; recomendado explícitamente en Lago et al. 2021).
- **Festivos colombianos**: única mejora confirmada con significancia estadística durante la
  sesión del 2026-09-10 (prueba t pareada, 5 semillas de N-BEATSx, p=0.0064) — integrada a las 5
  familias de modelo desde entonces.
- **Limitación restante frente a las mejores prácticas del campo** (Lago et al. 2021): los
  modelos se reentrenan una vez por origen, no se recalibran diariamente como sugiere la práctica
  estándar de EPF. Se intentó recalibración diaria/mensual para 2026 (Sección 7) sin éxito
  concluyente — queda como trabajo futuro, no como descuido.

## 3. Resultados — holdout único (2026)

| # | Modelo | Familia | MAE | RMSE | MAPE | ¿Supera a la persistencia? |
|---|---|---|---|---|---|---|
| — | Persistencia (t-24h) | Baseline | 56.31 | 112.22 | 15.75% | referencia |
| — | Naive estacional (t-168h) | Baseline | 124.53 | 180.60 | 36.75% | No |
| 1 | Prophet | Serie temporal aditiva | 95.84 | 150.22 | 19.93% | No |
| 2 | XGBoost | Gradient boosting | 61.15 | 107.32 | 15.89% | Parcial (solo RMSE) |
| 3 | LightGBM | Gradient boosting | 62.16 | 108.06 | 15.90% | Parcial (solo RMSE) |
| 4 | N-HiTS | Deep learning | 57.08 | 101.35 | 16.36% | Parcial (solo RMSE) |
| 5 | N-BEATSx | Deep learning | 56.11 | 100.92 | 15.93% | Parcial (MAE y RMSE) |
| 5b | **N-BEATSx — ensamble 5 semillas** | Deep learning | **54.39** | **95.95** | 16.18% | Parcial (MAE y RMSE) |
| 6 | ARX+GARCH(1,1) | Econométrico | 55.76 | 109.97 | **15.43%** | **Sí (en las 3 métricas)** |

**Lectura importante: el holdout único engaña.** ARX+GARCH gana en MAE/MAPE puntual, pero la
validación walk-forward (Sección 4) muestra que esa ventaja no es robusta entre regímenes.

## 4. Resultados — walk-forward (6 orígenes, con festivos, orígenes 1-5 de 12 meses completos)

### 4.1 MAE por origen y modelo

| Origen | Régimen | Persistencia | XGBoost | ARX+GARCH | N-BEATSx | N-HiTS |
|---|---|---|---|---|---|---|
| 1 | La Niña (jul-2020 a jun-2021, arranca en pandemia) | 16.80 | 18.25 | **50.88** | 25.79 | **55.38** |
| 2 | La Niña (jul-2021 a jun-2022) | 17.52 | 24.14 | 19.51 | 17.17 | 17.86 |
| 3 | La Niña triple-dip → neutral (jul-2022 a jun-2023) | 39.75 | 59.60 | 49.74 | 40.22 | 37.97 |
| 4 | El Niño 2023-24, episodio completo (jul-2023 a jun-2024) | 67.19 | 133.99 | 80.80 | 66.36 | 68.47 |
| 5 | Neutral / La Niña débil (jul-2024 a jun-2025) | 101.25 | 135.47 | 102.64 | 104.59 | 103.05 |
| 6 | El Niño 2026 (neutral→fuerte) | 56.40 | 61.39 | 55.83 | 46.82 | 46.44 |

**Hallazgo del 2026-09-11, al ampliar de 3 a 12 meses de prueba**: en el Origen 1, N-HiTS pasa de
un MAE competitivo (28.17 con solo 3 meses de prueba) a **55.38** con el año completo — su peor
resultado en todo el proyecto —, y ARX+GARCH de 31.72 a 50.88. N-BEATSx apenas cambia (24.85→25.79).
Investigado mes a mes: el error de ambos crece progresivamente entre jul-2020 y feb-2021 (24→99),
para bajar parcialmente hacia jun-2021 — un deterioro gradual, no un evento puntual. Se descartó
explícitamente la hipótesis de que el error crece con la distancia al corte de entrenamiento
(no se repite en los Orígenes 3-5); el mecanismo exacto queda sin identificar — candidato natural
para profundizar: la dinámica de demanda/precio propia del año de pandemia (jul-2020 a jun-2021).
Esto es precisamente el tipo de problema que una ventana de prueba corta (3 meses) no alcanza a
revelar, y es la razón por la que Lago et al. (2021) recomiendan un mínimo de 1 año.

### 4.2 Diebold-Mariano — cuántos orígenes gana/pierde cada modelo de forma significativa

| Comparación | Gana sig. | Pierde sig. | Sin diferencia |
|---|---|---|---|
| N-BEATSx vs ARX+GARCH | **5/6** | 0/6 | 1/6 |
| N-HiTS vs ARX+GARCH | 4/6 | 1/6 | 1/6 |
| N-BEATSx vs Persistencia | 1/6 | 1/6 | 4/6 |
| N-HiTS vs Persistencia | 1/6 | 1/6 | 4/6 |
| ARX+GARCH vs Persistencia | 0/6 | 4/6 | 2/6 |
| XGBoost vs Persistencia | 0/6 | **6/6** | 0/6 |

**Actualizado el 2026-09-11** (antes, con ventanas de 3 meses, N-BEATSx ganaba a ARX+GARCH en 6/6
y a Persistencia en 3/6): con el año completo, el Origen 1 (la falla de N-HiTS/ARX+GARCH descrita
arriba) deja de ser una victoria significativa de N-BEATSx sobre ARX+GARCH, y N-BEATSx/N-HiTS
dejan de ganarle de forma consistente a la persistencia simple en varios orígenes — matiza pero no
revierte la conclusión: **siguen dominando a ARX+GARCH** y XGBoost pierde contra la persistencia
en absolutamente todos los orígenes, ahora con más evidencia (12 meses en vez de 3).

### 4.3 rMAE por origen (Lago et al. 2021) — la ventaja relativa, no solo la significancia

El naive-7 días (`precio_lag168h`) varía en dificultad entre orígenes — comparar MAE crudo entre
orígenes sin normalizar es tan engañoso como comparar MAPE entre meses de distinta volatilidad.
rMAE = MAE(modelo) / MAE(naive-7d) corrige esto (recalculado 2026-09-11 con los orígenes de 12 meses):

| Origen | Persistencia | XGBoost | ARX+GARCH | N-BEATSx | N-HiTS |
|---|---|---|---|---|---|
| 1 | 0.516 | 0.561 | **1.562** | 0.792 | **1.700** |
| 2 | 0.371 | 0.511 | 0.413 | 0.364 | 0.378 |
| 3 | 0.357 | 0.535 | 0.447 | 0.361 | 0.341 |
| 4 | 0.386 | 0.769 | 0.464 | 0.381 | 0.393 |
| 5 | 0.570 | 0.763 | 0.578 | 0.589 | 0.580 |
| 6 | 0.456 | 0.496 | 0.451 | 0.379 | 0.375 |
| **Media / desv. std.** | **0.443** / 0.087 | 0.606 / 0.126 | 0.653 / **0.449** | 0.478 / 0.177 | 0.628 / **0.532** |

**Con el año completo, el cuadro cambia de forma sustancial respecto a la versión de 3 meses**:
ahora es la **persistencia simple la que tiene el rMAE promedio más bajo y, por lejos, la
variabilidad más chica** (desv. std. 0.087 contra 0.177-0.532 del resto) — el Origen 1 le cuesta
caro a N-HiTS (rMAE 1.700, pierde contra el naive semanal) y a ARX+GARCH (1.562), y ambos arrastran
la desviación estándar del grupo. N-BEATSx sigue siendo el mejor de los modelos "activos" (no
ingenuos) en promedio (0.478) y el más estable de ellos, pero ya no domina tan claramente como con
la muestra corta. Esto no cambia la recomendación para 2026 (Origen 6, sin cambios), pero es una
llamada de atención honesta: **ningún modelo del proyecto es uniformemente mejor que la
persistencia en todos los regímenes** — la ventaja de las redes se concentra en los regímenes de
El Niño (orígenes 4-6, donde sí importa más acertar por el mayor riesgo económico), no es universal.

## 5. Métricas de decisión (Maciejowska, Lipiecki, Uniejewski — Energy Conversion and Management, 2026)

RMSE y MAE correlacionan **<0.20** con la ganancia real de una regla de arbitraje contra el
precio pronosticado; lo que sí correlaciona fuerte (**>0.80**) es qué tan bien el pronóstico
replica la *forma* del día. Implementado en `scripts_experimento/metricas_decision.py`
(`evaluar_modelo()`), corregido el 2026-09-10 tras encontrar dos bugs frente a la fórmula exacta
del paper (ver bitácora).

Corr-f = si el pronóstico ordena las 24 horas del día igual que la realidad (1 = orden perfecto);
MHD = horas de error al ubicar la hora más barata y la más cara (suma de ambas); MPD = COP/kWh que
se pierden por operar en la hora señalada en vez de la óptima. Medido en 2026 para todos los
modelos y ensambles (`metricas_forma_2026.py`):

| Modelo | MAE | Corr-f | MHD (horas) | MPD (COP/kWh) | Amplitud* |
|---|---|---|---|---|---|
| Persistencia | 56.40 | **0.835** | **3.97** | 65.45 | 1.00 |
| XGBoost | 61.39 | 0.798 | 5.03 | 71.01 | 0.79 |
| ARX+GARCH | 55.83 | 0.772 | 8.02 | 81.05 | 0.98 |
| N-BEATSx | 46.82 | 0.775 | 5.70 | 58.82 | 0.81 |
| N-HiTS | 46.44 | 0.779 | 5.73 | 50.91 | 0.74 |
| Ensamble v1 (NNLS) | 44.51 | 0.816 | 4.51 | 53.27 | 0.80 |
| **Ensamble v4 (QRA por franja)** | **42.51** | 0.799 | 5.01 | **47.43** | 0.94 |
| Nivel v4 + forma ajustada (ver abajo) | 43.05 | 0.829 | 4.74 | 46.70 | — |

\*Amplitud: (máx − mín) pronosticado / (máx − mín) real, mediana diaria; < 1 = curva aplanada.

El ensamble v4 es el que **menos plata deja en la mesa** (MPD) y el que menos aplana la curva, pero
la persistencia ordena mejor las horas y ubica mejor el pico y el valle. ARX+GARCH es el peor en
ubicar el momento del día (8 horas de error), pese a su MAE puntual razonable.

**Separar nivel y forma** (`forma_del_dia.py`, `forma_del_dia_v2.py`): como estas tres métricas solo
dependen de cómo se mueve cada hora respecto a la media del día, se puede tomar el nivel del v4 y
ajustar la forma por separado (ensamble LAD sobre las desviaciones respecto a la media, incluyendo
el perfil semanal del precio como candidato). En versión desplegable el Corr-f sube de 0.796 a
0.829 (p=0.001) con +1.6% de MAE; el MHD baja de 4.92 a 4.50 horas pero sin significancia. Si la
regla de OE3 depende del *momento* de actuar, esta variante es la candidata natural (el paper
reporta que el Corr-f correlaciona >0.80 con la ganancia de arbitraje y el MAE <0.20).

## 6. Ensambles entre familias — el mejor resultado del proyecto

La mejora individual de cada modelo se estancó, pero **combinar modelos de familias distintas con
un meta-modelo NNLS** (pesos no negativos, sin intercepto, ajustados por validación cruzada de 5
pliegues por día) sí produjo saltos grandes. La lección que se repitió en cada experimento: lo que
hace ganar al ensamble es la **diversidad entre familias**, no el acierto individual de cada voto
(la persistencia simple tiene MAE muy peor que N-BEATSx y aun así se lleva peso propio).

### 6.1 Horizonte de 24h — ensamble de 5 modelos

Persistencia + XGBoost + ARX+GARCH + N-BEATSx + N-HiTS sobre Origen 6 (2026), 100% fuera de muestra.
La primera versión (NNLS, un solo vector de pesos) dio **MAE 44.51, RMSE 86.36, MAPE 12.05%** y
gana de forma significativa a los 5 modelos individuales (DM p<0.05 en todos, verificado en orden
cronológico). La versión final cambia solo **cómo** se combinan las mismas predicciones:

| Ensamble de 24h | Meta-modelo | MAE | MAPE | Acierto por tercil | MAE desplegable* |
|---|---|---|---|---|---|
| v1 | NNLS, pesos únicos | 44.51 | 12.05% | 90.90% | 46.12 |
| v3 | LAD, pesos por franja de 6h | 43.24 | 11.70% | 91.53% | 45.17 |
| **v4** | **QRA, pesos por franja de 6h** | **42.51** | **11.27%** | **92.01%** | **44.51** |

\*Pesos estimados solo con días anteriores (ventana expansiva), tras 14 días de calentamiento.

- **Franjas horarias** (idea tomada de Uniejewski 2026, que combina por hora del día): la ganancia
  se sostiene con 3, 4 y 6 franjas (DM p<0.0001 en las tres), así que no depende de una partición
  afortunada. Lectura física de los pesos: en la noche la persistencia pesa 41% (el pico nocturno
  se repite día a día), en la madrugada domina N-HiTS, en la tarde ARX+GARCH sube a 33%.
- **QRA** (Nowotarski & Weron 2015) en el cuantil 0.5 agrega un intercepto que absorbe el sesgo de
  nivel común a los votantes (todos sobreestiman en precios bajos). Le gana a LAD por franja
  (p=0.0016) y, en versión desplegable, al ensamble v1 (p=0.0004).
- **Validación de rigor**: la versión desplegable de v1 (46.12) no difiere de su versión por
  validación cruzada en las mismas horas (46.02, p=0.65); el resultado no es un artefacto de usar
  días futuros para fijar los pesos.

Contrato: `data/processed/resultados/stacking_24h_v4_qra_2026.csv`.

### 6.2 Horizonte de 25-72h — ensamble de 9 modelos (formulación directa)

El ensamble de 72h solo fue posible tras notar que la restricción "los modelos no neuronales no
llegan a 25-72h" aplicaba únicamente a la formulación **recursiva**. En formulación **directa** —
un solo modelo que recibe el paso `h` como variable y usa exclusivamente el estado conocido en el
corte `t-h` — árboles y modelos lineales cubren los 72 pasos sin fuga y sin reentrenamientos por
paso. Eso desbloqueó la diversidad de familias que faltaba.

| Tramo | N-BEATSx (línea base) | Mejor individual | Ensamble v2 (NNLS) | **Ensamble v3 (LAD)** | Mejora v3 | DM v3 vs línea base |
|---|---|---|---|---|---|---|
| 1-24h | 58.68 | N-BEATSx 58.68 | 57.42 | **54.68** | −6.8% | p=0.077 (n.s.) |
| 25-48h | 76.20 | Ridge directo 71.12 | 67.07 | **64.32** | **−15.6%** | **p=0.019 (sig.)** |
| 49-72h | 98.56 | Ridge directo 85.52 | 86.67 | **84.50** | **−14.3%** | **p=0.043 (sig.)** |

El meta-modelo final es **LAD** (mínima desviación absoluta, pesos ≥ 0 que suman 1), que optimiza
la misma métrica con la que se evalúa el proyecto y equivale a QRA en el cuantil 0.5 (Nowotarski &
Weron 2015). Le gana al NNLS con significancia en 1-24h (p=0.003) y 25-48h (p=0.0009).
Versión desplegable (pesos solo con cortes anteriores): −4.7% / −11.2% / −12.6% contra N-BEATSx.

> **Corrección metodológica.** Una versión anterior de este informe reportó p=0.023 y p=0.037 para
> el ensamble v2 (NNLS). Esos valores venían de un test DM calculado sobre la serie en el orden de
> los pliegues de la validación cruzada; el estimador HAC asume orden temporal y, al romperlo,
> subestima la varianza. En orden cronológico v2 da p=0.076 y p=0.090 (no significativos). Todos
> los p-valores de este informe están ahora en orden cronológico; el ensamble de 24h no tenía el
> problema (sus p-valores son idénticos en ambos órdenes).

**Hallazgo destacable: a horizontes largos el mejor modelo individual es LINEAL** (Ridge sobre las
regresoras directas), por encima de redes neuronales y árboles. Interpretación física: a 2-3 días
ya no queda información explotable de la dinámica horaria de corto plazo, y lo que domina es el
nivel de fundamentales hidrológicos más la estacionalidad semanal — estructura que un lineal
captura sin sobreajustar y que además extrapola, cosa que los árboles no pueden por construcción.
Consistente con esto, entrenar CatBoost con el doble de datos **empeoró** el error: el cuello de
botella a horizonte largo no es la cantidad de datos.

### 6.3 Bandas de incertidumbre

Calibración conforme **adaptativa** (margen recalculado en ventana móvil de 30 días) en ambos
horizontes. La calibración estática (un solo ajuste) solo llegó a 67-72% de cobertura — el régimen
cambia a mitad de 2026, así que un margen fijo no sirve. Se evalúa con cobertura, ancho y el
Interval Score (Gneiting & Raftery 2007), que penaliza a la vez el ancho y las observaciones que
caen fuera de la banda.

- **24h — QRA** (cuantiles 0.1 y 0.9 sobre los mismos 5 votantes, por franja horaria) + calibración
  causal: cobertura 79.7%, ancho 159.6 COP/kWh, Interval Score 289.8, contra 76.8% / 181.4 / 322.5
  de las bandas de N-BEATSx solo — más cobertura con bandas 12% más angostas (IS p=0.006).
  Contrato: `bandas_24h_qra_causal_2026.csv`.
- **25-72h — ensamble v3**: cobertura 78.4% / 77.5% / 75.0% en 1-24h / 25-48h / 49-72h (objetivo
  80%), ancho creciendo con el horizonte (181 → 228 → 285 COP/kWh). QRA también se probó aquí y no
  mejora el Interval Score de forma significativa.
  Contrato: `pronostico_ensamble_72h_v3_adaptativo.csv`.

Se revisó si la calibración de 72h usaba scores posteriores al corte (el margen de un día objetivo
D usa los 30 días previos a D, y en el tramo 49-72h el corte es D−3). No ocurre con la geometría
actual: con cortes cada 72h a las 23:00 cada tramo cubre un día completo y no existen filas de
D−1 ni D−2 en ese tramo; la versión calculada por corte da resultados idénticos.

### 6.4 rMAE contra persistencia estacional (Lago et al. 2021) en todo el horizonte

0.470 / 0.525 / 0.698 en 1-24h / 25-48h / 49-72h. El modelo le gana al benchmark ingenuo con
holgura en todo el rango; la ventaja se estrecha con el horizonte, como es esperable.

## 7. Intentos que no funcionaron (documentados con el mismo rigor que los exitosos)

| Intento | Resultado | Causa identificada |
|---|---|---|
| LEAR + ensamble (benchmark estándar del campo, Lago et al. 2021) | MAE 74.94 en 2026 — peor que todos los modelos del proyecto | El proyecto ya supera el benchmark publicado del área |
| 4 hipótesis de sobremuestreo de El Niño (duplicar episodio, pesos, etc.) | Ninguna mejoró el error en El Niño de forma confirmada | `neuralforecast` no expone pesos por muestra; duplicar aumenta varianza entre semillas sin mejorar la media |
| Corrección de sesgo global (post-hoc, `real = a + b·pred`) | DM p=0.682, no significativo | El sesgo cambia de signo entre años (2020/2022 subestiman, 2021/2023/2024/2026 sobreestiman) — no hay un sesgo fijo que corregir |
| Corrección por régimen ONI (post-hoc, estratificada) | DM p=0.305, no significativo | Mejora pequeña (91.78→90.74 MAE en El Niño) pero no distinguible de ruido |
| **Lógica difusa del ONI** (membresía débil/moderado/fuerte/muy fuerte + velocidad de transición) | Mejora mínima en ARX+GARCH (55.76→55.63); **empeora N-BEATSx/N-HiTS/ensamble** (56.11→58.47, 57.08→60.53, 54.39→55.44) | `oni` ya era continuo; las 7 columnas fuzzy añaden dimensionalidad exógena sin información nueva — una de ellas activa en solo ~1,100 de 60,625 horas, ruido puro para la red |
| **Análogos de régimen vía k-NN** (calibrar solo con los 12 bloques de 30 días históricamente más parecidos al régimen actual, en vez de todo el cronológico) | Pierde significativamente en los 2 orígenes de El Niño (p<0.0001 en ambos) | Usa solo ~8,000 filas vs 40,000-60,000 cronológicas — perder 80-85% de los datos le hace más daño a XGBoost que cualquier beneficio de similitud de régimen; el descriptor ONI no captura la tendencia secular del precio |
| Markov-Switching de 2 regímenes (intercepto y varianza conmutan) | MAE 55.76, DM vs ARX+GARCH p=0.993 — empate | El régimen detectado dura ~10 horas, no meses: a resolución horaria el EM no supervisado encuentra ráfagas de volatilidad, no el ciclo El Niño/La Niña |
| Corrección adaptativa del punto (sesgo móvil sobre el ensamble) | Empeora significativamente con ventanas de 7d (p=0.010) y 14d (p=0.049); 30d sin diferencia | Confirma que el sesgo cambia de signo a escala **anual**, no semanal — no hay señal que una ventana corta pueda estimar |
| Transformer (TFT) como 6º votante del ensamble de 24h | Abortado tras 2h sin completar la primera fase de validación cruzada | Costo computacional en CPU sin GPU, no resultado negativo — queda pendiente, no descartado |
| XGBoost directo v2 (anclas estacionales de 1/2/3 semanas + objetivo cuantílico) | Peor que v1 en los tres tramos (71.00 vs 65.89 en 1-24h) | El ancla semanal ya estaba capturada por `precio_mismo_hora_disp` y `precio_lag168h`; las anclas extra añaden ruido correlacionado |
| CatBoost directo con el doble de datos de entrenamiento (2.2M filas) | Peor que con 1.1M en los tres tramos (89.45 vs 86.63 en 49-72h) | El cuello de botella a horizonte largo no es cantidad de datos; el muestreo más denso sobrerrepresenta regímenes antiguos ya cubiertos |
| Enriquecer exógenas de N-BEATSx a 72h (9 históricas en vez de 3) | Mejora no significativa: −1.02 y −2.32 MAE en tramos largos, +0.88 en 1-24h (DM p>0.67) | Aporta como votante del ensamble (errores distintos), pero por sí solo no mueve la aguja |
| Fusionar al stacking de 24h los 8 modelos sueltos (CatBoost, hurdle, Markov, RecursiveLS, XGB/CatBoost con Optuna, CatBoost/Ridge directos) | Empeora: 46.02 → 46.45 (p=0.013) | Casi todos son árboles muy correlacionados; 13 pesos se estiman con ruido. Importa la diversidad de familias, no la cantidad de votantes |
| Pesos de ensamble con ventana móvil de 30/60 días (agregación adaptativa de expertos, arXiv:2405.15359) | Peor que la ventana expansiva en 24h (p<0.01); sin ganancia en 72h | Con ~200 días de 2026, 30-60 días no alcanzan para estimar pesos estables |
| Transformación asinh estándar y parametrizada + AVGroll (Uniejewski et al. 2018; Uniejewski 2026) sobre el Ridge directo | asinh estándar empeora 6-16%; AVGroll solo +0.7% | Los picos colombianos son cambios de nivel sostenidos (régimen hidrológico), no picos aislados como en Europa: comprimir la cola alta sesga hacia abajo justo en El Niño. En árboles con pérdida MAE es casi irrelevante por construcción |
| NNLS como meta-modelo del ensamble de 72h (v2) | Superado por LAD (p=0.003 en 1-24h y p=0.0009 en 25-48h) | NNLS minimiza error cuadrático, dominado por los picos; el proyecto se evalúa con MAE |
| Pesos por franja horaria en el ensamble de 72h | Sin diferencia (p>0.6) | ~430 filas por franja y tramo no alcanzan para 9 pesos (en 24h, con 5 pesos y el doble de datos, sí funcionó) |
| QRA para las bandas de 72h | Interval Score sin diferencia significativa contra v3 | Las bandas de v3 ya estaban bien calibradas |
| Hipótesis de vacaciones de mitad de año como explicación de junio-2026 | Descartada — 0 de 8 años (2019-2026) muestra una caída de demanda estadísticamente significativa en la ventana 15 jun-15 jul | La explicación que sí sobrevive todas las pruebas es la velocidad de transición del régimen ONI, no el calendario |

## 8. Diagnóstico de fallas — patrones que sí se sostienen entre años

- **Sesgo invertido por decil de precio**: el modelo sobreestima sistemáticamente en los precios
  bajos y subestima en los picos más caros — confirmado en 5 de 6 orígenes del walk-forward (la
  excepción es, previsiblemente, el Origen 1 — el mismo régimen anómalo de la Sección 4.1 donde
  N-HiTS y ARX+GARCH fallan; ahí ambos sobreestiman en los DOS extremos de precio, un patrón
  distinto al resto).
- **Rachas de días consecutivos malos ligadas a velocidad de transición de régimen, no a un mes
  específico**: la racha más larga de toda la historia es el Origen 4 (2023, 9 días
  consecutivos), durante el arranque del súper Niño — no junio-2026 (4-5 días). La correlación
  pooled entre velocidad de cambio del ONI y MAPE mensual (0.254) es mayor que la del nivel del
  ONI (0.134).
- **Junio y julio de 2026 tienen perfiles de error distintos**: junio muestra rachas de días
  consecutivos malos (coincide con la transición ONI más rápida del histórico, según el
  descriptor fuzzy); julio no tiene rachas pero sí un domingo atípicamente malo (21.4% MAPE) y
  concentra su error en las horas 8-11am.

## 9. Recomendación final para OE3

**Se recomienda un contrato por horizonte, con el especialista de 24h dentro del de 72h:**

| Horizonte | Modelo recomendado | Desempeño (2026, fuera de muestra) | Archivo |
|---|---|---|---|
| **24h (día siguiente)** | Ensamble v4: QRA por franja horaria sobre 5 modelos; bandas QRA + calibración conforme causal | MAE 42.51-42.84 / MAPE ~11.3% · cobertura ~79-80% | `bandas_24h_qra_causal_2026.csv` |
| **1-72h (horizonte completo)** | Contrato unificado: v4 en los pasos 1-24, ensamble LAD de 9 modelos directos (214 cortes diarios) en 25-72 | MAE 42.3 / 64.2 / 74.8 por tramo (CV) · global 60.41 (MAPE 17.35%); versión desplegable: 44.3 / 67.2 / 79.8, global 63.68 (MAPE 17.56%) | `pronostico_unificado_72h_v2_2026.csv` |

**Actualizado el 2026-09-11 (noche)**: el contrato de 72h se reconstruyó con 214 cortes diarios en
vez de 72 no solapados (Sección 4 y bitácora), dando mucha más potencia estadística. El contrato
unificado es legítimo porque los cortes de 72h (diarios, a las 00:00) son ahora el mismo calendario
que los cortes de 24h: para h=1-24 ambos pronostican las mismas horas desde el mismo corte. Frente
al punto de partida de 72h (N-BEATSx solo, MAE global ~78), el contrato unificado reduce el MAE
global en cerca de 22% (versión con validación cruzada) o 18% (versión desplegable, más conservadora).

Razones, en orden de peso:

1. Ambos ensambles ganan con significancia estadística (DM en orden cronológico) a su referencia
   anterior: v4 al ensamble v1 de 24h y a los 5 modelos individuales; v3 a N-BEATSx en 25-48h y
   49-72h. Y ambos sobreviven en versión desplegable (pesos estimados solo con información pasada).
2. El modelo base de las redes (N-BEATSx) gana a ARX+GARCH de forma significativa en 5/6 orígenes
   del walk-forward (12 meses cada uno), y su ventaja relativa (rMAE) es mayor justo en los
   regímenes de El Niño — donde más importa acertar por el mayor riesgo económico de una mala
   decisión. El único origen sin victoria significativa (Origen 1, La Niña 2020-21) coincide con
   una falla real de N-HiTS y ARX+GARCH investigada en la Sección 4.1 — no es ruido.
3. El ensamble de 24h alcanza 92.01% de precisión por tercil (0.14% de errores graves), por encima
   del 84.77% reportado por el estudio de AES Colombia para su mejor planta — la métrica más cercana
   a lo que necesita una regla de compra/venta.
4. Ambos traen bandas de incertidumbre calibradas (adaptativas, 75-80% de cobertura, objetivo 80%)
   que pasan la validación de contrato del motor de decisión (`validar_contrato_pronostico`).

**Integración con OE3**: el motor lee su fuente desde `fuentes_pronostico.json`, que todavía apunta
a N-BEATSx solo en ambos horizontes. Adoptar los contratos nuevos es editar ese JSON — el motor solo
consume `fecha_hora, real, q10, q50, q90`, así que no hay que tocar el notebook 12 ni el dashboard —
y queda pendiente de aprobación del equipo.

**Alternativa simple defendible**: en 49-72h, Ridge directo solo (MAE 85.52) es estadísticamente
indistinguible del ensamble v3 (84.50, p=0.79) y mucho más interpretable. Si la sustentación
prioriza simplicidad sobre robustez, es una opción legítima — pero pierde las bandas calibradas.

**Advertencias explícitas para quien integre el modelo a la regla de decisión**:
- Si la regla depende de *replicar la forma exacta* del día (no solo el nivel), revisar Corr-f
  con cuidado — la persistencia simple gana ahí por construcción.
- No usar corrección de sesgo global ni sobremuestreo de El Niño — ambos se probaron y no
  generalizan.
- Antes de integrar cualquier modelo, correr `evaluar_modelo()` sobre el caso de uso real de la
  regla de decisión, no solo mirar esta tabla.

## 10. Limitaciones conocidas del estudio

- Un solo mercado (Colombia) — no se puede generalizar la superioridad de N-BEATSx a otros
  mercados eléctricos sin repetir el análisis.
- ~~Orígenes 1-5 del walk-forward eran ventanas de 3 meses~~ — **resuelto el 2026-09-11**: ahora
  son ventanas de 12 meses consecutivos sin solape, el mínimo que recomienda Lago et al. (2021).
  Reveló una falla real de N-HiTS/ARX+GARCH en el Origen 1 que la ventana corta ocultaba
  (Sección 4.1) — la ampliación no fue solo un ejercicio de rigor, cambió una conclusión.
- Sin recalibración diaria (estándar del campo) — se reentrena una vez por origen/época. El
  reentrenamiento deliberado de un único punto de corte sí ayuda (N-BEATSx antes de junio-2026,
  MAE 93.59→86.21, p=0.007), pero la recalibración periódica automatizada (mensual, 8 reentrenos
  en 2026) se probó el 2026-09-11 y **empeoró** de forma pareja en los 8 meses (46.8→55.7 de MAE),
  tras corregir dos bugs reales encontrados en el camino — la causa de fondo no se terminó de
  aislar (hipótesis: varianza de convergencia acumulada entre 8 reentrenamientos independientes).
  Queda como pregunta abierta, no como mejora pendiente de aplicar sin más.
- ~~La evaluación del horizonte 25-72h usaba 72 cortes~~ — **resuelto el 2026-09-11**: ahora usa
  214 cortes diarios (`step_size=24`), y la ganancia del ensamble sobre N-BEATSx solo pasó a ser
  significativa en los tres tramos, incluida la versión desplegable (antes no lo era en dos de
  los tres tramos por falta de potencia estadística).
- Los modelos de formulación directa (árboles y lineal) se entrenan con un corte cada 4 horas por
  control de memoria, no por criterio estadístico. Se probó densificar a cada 2 horas y empeoró,
  pero no se exploró el rango intermedio.
- El régimen El Niño es minoritario en el entrenamiento (~10% de 2019-2025) pero ~30% del
  holdout 2026 — la literatura (modelos HMM por régimen) confirma que esto es difícil incluso
  con técnicas dedicadas, no un error metodológico del proyecto.

## Referencias

- Diebold, F.X., Mariano, R.S. (1995). *Comparing predictive accuracy.* JBES 13(3).
- Lago, J., Marcjasz, G., De Schutter, B., Weron, R. (2021). *Forecasting day-ahead electricity
  prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark.*
  Applied Energy 293:116983.
- Maciejowska, K., Lipiecki, A., Uniejewski, B. (2026). *Statistical and economic evaluation of
  forecasts in electricity markets: beyond RMSE and MAE.* Energy Conversion and Management 356,
  arXiv:2511.13616.
- Muñoz-Santiago, N. et al. (2017). Referencia de ARIMA-GARCH para el mercado colombiano.
- Olivares, K. et al. (2023). N-BEATSx. Challu, C. et al. (2023). N-HiTS. (`neuralforecast`, Nixtla)
- Anexo 1 de formulación del proyecto (`docs/ANEXO_1_formulacion_proyecto.md`).
