# Biblioteca de papers del proyecto

Cada entrada tiene la cita completa, el enlace de verificación, y **para qué se usa en el proyecto**
— para que cualquier afirmación del informe se pueda rastrear hasta su fuente.

Los PDF y su texto extraído (`.txt`) están en esta misma carpeta.

---

## Métodos aplicados en el proyecto

### Nasiadka, Nitka & Weron (2022) — selección de ventana por puntos de cambio
> Nasiadka, J., Nitka, W., & Weron, R. (2022). *Calibration window selection based on change-point
> detection for forecasting electricity prices*. Computational Science – ICCS 2022, Londres, Reino
> Unido. Lecture Notes in Computer Science, Springer. arXiv:2204.00872

- Archivo: `Nasiadka_Nitka_Weron_Calibration_Window_ChangePoint.pdf`
- https://arxiv.org/abs/2204.00872
- **Uso:** es el método implementado en `ventana_calibracion_not_24h.py`. Aporta el algoritmo de
  selección de subperíodos (pasos 1-5), la transformación asinh, y los esquemas de promediado de
  ventanas Av(Win) / Av(NOT).
- **Por qué este proyecto puede aportar algo:** los autores cierran diciendo textualmente que si el
  método sirve para modelos más complejos que su autorregresivo — *"e.g., LASSO-estimated AR (LEAR)
  and deep neural networks, is left for future work"* — queda pendiente. Este proyecto tiene un LEAR
  de 24h y contesta esa pregunta.
- **Calibración de expectativas (lo que ellos mismos reportan):** en el mercado alemán, RMSE
  Win(728) 8.2860 → WinH(728) 7.7286 → NOTH(728) 7.5994. O sea que la mayor parte de la ganancia es
  del asinh y no de NOT, y advierten que la parte de NOT *"is not statistically significant if
  considered on its own"*.

### Baranowski, Chen & Fryzlewicz (2019) — el detector NOT
> Baranowski, R., Chen, Y., & Fryzlewicz, P. (2019). *Narrowest-over-threshold detection of multiple
> change-points and change-point-like features*. Journal of the Royal Statistical Society: Series B
> (Statistical Methodology), 81(3), 649-672.

- https://doi.org/10.1111/rssb.12322 · paquete de R: https://cran.r-project.org/package=not
- **Uso:** algoritmo de detección reimplementado en `not_changepoint.py` (no hay puente a R en el
  proyecto). Contraste de razón de verosimilitudes para media y varianza constantes a trozos.
- **Nota:** la reimplementación está validada contra series sintéticas con puntos de cambio
  conocidos (`autoprueba()`, 5/5). El experimento aborta si la autoprueba falla.

### Lago, Marcjasz, De Schutter & Weron (2021) — el benchmark canónico de EPF
> Lago, J., Marcjasz, G., De Schutter, B., & Weron, R. (2021). *Forecasting day-ahead electricity
> prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark*.
> Applied Energy, 293, 116983.

- Archivo: `Lago_Marcjasz_DeSchutter_Weron_2021_EPF_review_benchmark.pdf`
- https://doi.org/10.1016/j.apenergy.2021.116983 · arXiv:2008.08004
- **Uso:** define el LEAR (un LASSO por hora del día) que el proyecto usa como modelo, y fija las
  buenas prácticas de evaluación del área (Diebold-Mariano, protocolo day-ahead).

### Ziel & Weron (2018) — univariante contra multivariante
> Ziel, F., & Weron, R. (2018). *Day-ahead electricity price forecasting with high-dimensional
> structures: Univariate vs. multivariate modeling frameworks*. Energy Economics, 70, 396-420.

- Archivo: `Ziel_Weron_2018_Univariate_vs_Multivariate.pdf`
- https://doi.org/10.1016/j.eneco.2017.12.016
- **Uso:** sustenta el marco multivariante que el proyecto adoptó para entender el salto de
  medianoche: los 24 precios de un día se fijan en UNA subasta, así que la frontera entre días es
  el límite entre dos subastas, no un defecto del modelo.

---

## Mercados hidro-dominados (comparación internacional)

### Kapoor & Wichitaksorn (2023) — Nueva Zelanda, la fuente de la Figura 5a
> Kapoor, G., & Wichitaksorn, N. (2023). *Electricity price forecasting in New Zealand: A
> comparative analysis of statistical and machine learning models with feature selection*. Applied
> Energy, 347, 121446.

- Archivo: `Kapoor_Wichitaksorn_2023_NZ_EPF_AppliedEnergy.pdf`
- https://doi.org/10.1016/j.apenergy.2023.121446 ·
  https://www.sciencedirect.com/science/article/pii/S0306261923008103
- **Uso:** es la fuente de la comparación de MASE de la Figura 5a. **33 modelos** (9 base × 4
  esquemas de selección de variables), 5 regiones.
- **Verificado contra las Tablas 5, 6 y 8:** mejor MASE global 1.2626 (LE-GARCH-t, Central North
  Island), peor 2.5001 (RFE-GARCH, Lower South Island). Ninguno baja de 1.
- **Salvedad obligatoria al citarlo:** transforman los precios con Box-Cox o Yeo-Johnson antes de
  modelar (*"electricity prices... tend to display a skewed distribution... we choose to transform
  them using the Box–Cox or Yeo-Johnson logarithmic transformations"*) y luego escalan min-max. El
  MASE es invariante al escalado lineal pero **no** a la transformación logarítmica, así que su MASE
  y el del proyecto no miden la misma cantidad. Además pronostican precio **diario**, no horario.

### Kapoor, Wichitaksorn & Zhang (2023) — Nueva Zelanda, regímenes
> Kapoor, G., Wichitaksorn, N., & Zhang, W. (2023). *Analyzing and forecasting electricity price
> using regime-switching models: The case of New Zealand market*. Journal of Forecasting, 42(8),
> 2011-2026.

- Archivo: `Kapoor_Wichitaksorn_Zhang_2023_NZ_Regime_Switching_JoF.pdf` · acceso abierto
- https://doi.org/10.1002/for.3004
- **Uso:** respaldo teórico de la línea de regímenes. Markov regime-switching hasta 5 regímenes,
  probabilidades de transición variables, y teoría de valores extremos (EVT-PoT); prueban que el
  EVT-PoT es un caso particular del MRS de 3 regímenes.
- **NO es comparable en exactitud con este proyecto**: sus métricas son sobre precio diario y su
  objetivo es replicar la densidad, no el pronóstico puntual (MAPE de 52% a 299%). **No confundir
  con el paper anterior**: son dos trabajos distintos de autores parcialmente coincidentes, y el de
  la Figura 5 es el de Applied Energy.
- **Hallazgo propio que conecta con el proyecto:** concluyen que más regímenes no mejoran el
  pronóstico pese a la complejidad añadida — consistente con que el Markov-Switching de 2 regímenes
  probado aquí tampoco aportara.

### Huisman & Mahieu (2003) — el modelo de tres regímenes
> Huisman, R., & Mahieu, R. (2003). *Regime jumps in electricity prices*. Energy Economics, 25(5),
> 425-434.

- Archivo: `Huisman_Mahieu_2003_Regime_Jumps_Electricity_Prices.pdf`
- https://doi.org/10.1016/S0140-9883(03)00041-0 · copia gratuita en el repositorio de Erasmus:
  https://repub.eur.nl/pub/73102/
- **Uso:** el modelo de tres regímenes (normal / salto / reversión). Es el origen de la línea de
  regímenes en EPF.

### Das, Schlüter & Schneider (2026) — regímenes con procesos neuronales condicionales
> Das, A., Schlüter, S., & Schneider, L. (2026). *Regime-aware conditional neural processes with
> multi-criteria decision support for operational electricity price forecasting*. Energy Economics,
> 157, 109233. https://doi.org/10.1016/j.eneco.2026.109233 (preprint: arXiv:2508.00040, solo con
> los dos primeros autores; la versión publicada agrega a Lorenz Schneider).

- Archivo: `Das_Schluter_Regime_Aware_Conditional_Neural_Processes.pdf`
- **Uso:** detección de regímenes (DS-HDP-HMM) acoplada a pronóstico de 24h en el mercado alemán.
  Referencia de la línea de "detectar régimen y condicionar el modelo".


### Singh (2027) — compensación de sesgo condicionada al régimen (RA-STBC)
> Singh, G. A. (2027). *Regime-aware deep learning for financial forecasting: an adaptive short-term
> bias compensation framework for the Warsaw Stock Exchange*. Expert Systems With Applications, 332,
> 133584. (Publicado en línea el 8 de julio de 2026.)

- Archivo: `Singh_2027_RA-STBC_Warsaw_ExpertSystems.pdf` · acceso abierto (CC BY)
- https://doi.org/10.1016/j.eswa.2026.133584
- **Aportado por el usuario.** Método extra-energético para corregir el error de un modelo cuando
  hay cambios de régimen. Extiende el STBC de Chang et al. (2024) haciendo que la fuerza de
  corrección λ dependa de un régimen detectado con una mezcla de gaussianas sobre retorno y
  volatilidad (λ = 0.2 calma / 0.5 intermedio / 0.8 crisis).
- **El método, en una línea:** `corregido = predicho + λ · (media móvil de los últimos k=10 errores)`.
  Sin variables explicativas ni modelo aprendido.
- **Cautelas al citarlo.** Su reducción de MSE del 92.27% (λ fijo) convive con exactitud direccional
  del 49.23% —el azar— y rendimiento de cartera de −16.39%. Pronostican **nivel** de un índice
  bursátil, que es casi un paseo aleatorio; corregir hacia los errores recientes acerca el
  pronóstico al último valor observado, o sea que empuja hacia la **persistencia**. Ellos mismos
  concluyen que la aportación es *"in price-level forecasting accuracy"*, no en señal. La Tabla 6
  del paper trae filas duplicadas (Bull ×3, Bear ×2) y el régimen de crisis tiene n=22.
- **Resultado en este proyecto:** probado fielmente en `stbc_regime_aware_24h.py`. La
  autocorrelación del error del ensamble con su media móvil es de solo **+0.127**, la búsqueda en
  validación elige **λ = 0**, y los λ del paper empeoran el resultado (58.93 → 60.42, p=0.029). La
  corrección correlaciona **+0.377** con la dirección de la persistencia, que ya es uno de los seis
  votantes del ensamble: no aporta información nueva.

### Chang et al. (2024) — el STBC original
> Chang, R. I., Wang, C. H., Wei, L. C., & Lu, Y. F. (2024). *LSTM with short-term bias compensation
> to determine trading strategy under black swan events of Taiwan ETF50 stock*. Applied Sciences,
> 14, 8576.

- https://doi.org/10.3390/app14188576 · acceso abierto
- **Uso:** es la fuente del mecanismo STBC (λ fijo). Sin PDF local; se conoce a través de Singh (2027).

---

## Colombia

### Gallón & Barrientos (2021) — enfoque funcional
> Gallón, S., & Barrientos, J. (2021). *Forecasting the Colombian Electricity Spot Price under a
> Functional Approach*. International Journal of Energy Economics and Policy, **11(2)**, 67-74.

- Archivo: `Gallon_Barrientos_2021_Colombia_Functional_Approach.pdf`
- **Corregido en el informe final:** ya cita 11(2), pp. 67-74, como dice el PDF.
- **DOI:** el PDF imprime 10.32479/ijeep.10607, pero en Crossref ese DOI resuelve a otro artículo (sobre petróleo);
  conviene citar también la URL de la revista (verificado el 23-oct-2026).

### Barrientos Marín, Tabares Orozco & Velilla (2018) — redes, ARMA e híbridos
> Barrientos Marín, J., Tabares Orozco, E., & Velilla, E. (2018). *Forecasting Electricity Price in
> Colombia: A Comparison Between Neural Network, ARMA Process and Hybrid Models*. International
> Journal of Energy Economics and Policy, 8(3), 97-106.

- Archivo: `IJEEP_2018_Forecasting_Electricity_Price_Colombia.pdf`
- **Uso:** antecedente colombiano directo. Los tres autores son de la Universidad de Antioquia.

### Ferraro Rodríguez (2025) — correlaciones y ARIMA mensual (maestría, UNAL Medellín)
> Ferraro Rodríguez, C. A. (2025). *Análisis de correlaciones del precio de bolsa de energía eléctrica
> con variables del sector eléctrico y económicas* [Trabajo final de maestría, Magíster en Ingeniería
> Eléctrica]. Universidad Nacional de Colombia, Facultad de Minas, Medellín. Dir.: E. A. Pérez Rojas;
> codir.: N. J. Castrillón Gutiérrez.

- https://repositorio.unal.edu.co/items/05583774-4f00-4fb1-8cfa-ecf426497670 (revisado el 19-oct-2026; PDF de 214
  páginas aportado por Juan, no incluido en el repositorio)
- **Qué hace:** correlaciona el precio promedio *ponderado* diario de bolsa (2010 a abr-2024) con embalse, ofertas
  hidráulicas y térmicas por combustible, Brent, TIE con Ecuador, IPC e IPP. La correlación más alta es con el precio
  de oferta hidráulico ponderado (r = 0,937 diario; 0,958 mensual); los desfases de ±3 días o meses la empeoran.
  Modelo: ARIMA(3,1,0) mensual sobre el precio deflactado con el IPP y sobre la diferencia entre ese precio y la
  oferta hidráulica; entrena 2010-2019 y evalúa 2020 a abr-2024.
- **Uso en el proyecto:** antecedente colombiano. Respalda corregir el predespacho ideal en vez de modelar el precio
  desde cero: el precio lo fija la oferta del recurso marginal, y el predespacho ya trae esas ofertas.
- **Cautelas al citarlo (verificadas en sus anexos de código):**
  - La evaluación es un paso adelante mensual reentrenando con el dato real, y el gráfico de validación corre las
    predicciones un mes hacia atrás (`pred_dates = test.index - pd.DateOffset(months=1)`): como un ARIMA(3,1,0)
    predice casi el último valor, la curva queda casi encima de la real. El "buen ajuste en épocas atípicas" es en
    buena parte ese desfase. No reporta ninguna métrica numérica de error.
  - La ecuación final (5.1) escribe el ARIMA(3,1,0) en niveles, sin la diferenciación, y el código toma como
    "constante" el primer coeficiente AR. El "modelo final" (5.8) predice la oferta hidráulica, no el precio de bolsa.
  - Correlacionar el precio con la oferta hidráulica del mismo día es casi tautológico: la oferta marginal fija el
    precio. Además esa oferta no es pública a tiempo (en este proyecto se comprobó con la OFEI).
  - Las correlaciones con IPC e IPP en niveles son de dos series con tendencia (riesgo de correlación espuria).

### Becerra Pulido (2025) — SARIMAX diario (pregrado, Ingeniería Electrónica, USTA)
> Becerra Pulido, J. D. (2025). *Pronóstico del precio diario de bolsa en el mercado eléctrico colombiano usando
> modelos autorregresivos* [Proyecto de grado, Ingeniería Electrónica]. Universidad Santo Tomás, Bogotá. Dir.:
> G. Garrido Silva; codir.: H. Ramírez Murillo.

- https://repository.usta.edu.co/items/d8dd82e6-6d1c-418d-915d-f035cef594d9 (revisado el 19-oct-2026; PDF aportado
  por Juan, no incluido en el repositorio)
- **Qué hace:** precio promedio diario de 2024 (366 datos, SIMEM) con AR, ARX, SARIMA y SARIMAX; exógenas: demanda,
  participación hidráulica y volumen de embalses. Mejor modelo SARIMAX(0,0,0)(0,1,1,7) con Precio_lag1 y
  Precio_lag7: RMSE 131,78 y MAPE 9,14 % (las unidades dicen COP/MWh pero los valores son COP/kWh).
- **Uso en el proyecto:** el trabajo más comparable (mismo programa, mismo problema). Sus "trabajos futuros"
  (desagregar la generación, híbridos, cambio de régimen, tablero interactivo) ya están en Augur.
- **Cautelas al citarlo:**
  - Usa el precio del día anterior (Precio_lag1), su variable más influyente según su propio análisis OAT, pero XM
    lo publica con unos 3 días de rezago (la TX1 sale hasta el día D+2): no está disponible al pronosticar.
  - Reemplaza por la mediana el 6,28 % de los precios (23 días, los picos) antes de modelar y evalúa sobre esa
    serie suavizada; el error reportado excluye los eventos más difíciles.
  - Elige entre más de 5000 configuraciones por el RMSE del conjunto de prueba (no hay validación aparte).
  - Las conclusiones dicen que la demanda es la variable más influyente; su Tabla 11 muestra que es Precio_lag1.
  - Un solo año y prueba de unos 67 días; la reducción de RMSE de 18,5 % frente al AR no es rastreable en sus tablas.

### García del Villar, Verano Urueta & Ranauro Argumedo (2020) — Uninorte
> García del Villar, V., Verano Urueta, N., & Ranauro Argumedo, R. (2020). *Diseño de un modelo estadístico para la
> predicción de los precios de bolsa de energía en Colombia* [Proyecto final]. Universidad del Norte, Barranquilla.
> Asesor: A. A. Henao Pérez.

- http://hdl.handle.net/10584/9283 · https://repositorio.uninorte.edu.co/items/e0b53330-6f96-47d2-bc99-c71e830e8504
  · https://manglar.uninorte.edu.co/handle/10584/9283
- **El documento completo no es público** (búsqueda del 19-oct-2026): los dos repositorios de Uninorte (el actual y
  Manglar) solo tienen el póster en español e inglés y dos imágenes; no aparece publicado en revistas ni en
  ResearchGate. El póster muestra el precio diario 2010-2020 y un pronóstico de Prophet con banda hasta 2022. Para
  leerlo hay que pedirlo a la biblioteca de Uninorte o al asesor.
- **Qué hace:** metodología de ciencia de datos de IBM; compara series de tiempo, regresión lineal, TBATS, Prophet y
  redes neuronales para el precio promedio del día siguiente. La red neuronal fue la mejor y aun así subestimó el
  precio en 19 % en promedio.
- **Uso en el proyecto:** antecedente de la misma universidad; útil en el estado del arte y para mostrar el avance
  (horario, con información disponible al corte, bandas probabilísticas y motor de decisión).

### Sánchez & Noguera (2024) — MPO ajustado con redes neuronales (Uninorte)
> Sánchez, J., & Noguera, C. (2024). *Estimación del máximo precio ofertado en el mercado de energía mayorista
> colombiano apoyado en IA* [Proyecto final]. Universidad del Norte, Barranquilla. Asesora: A. Arango.

- http://hdl.handle.net/10584/13009 · https://repositorio.uninorte.edu.co/items/b1f5826b-c602-4a23-afb3-2337631be961
- Solo están públicos el resumen, la interfaz (PDF) y una gráfica; no el documento completo.
- **Qué hace:** aplica el Proyecto de Resolución CREG 701 049 de 2024 (ajustar el MPO cuando la planta marginal es
  hidráulica, llevándolo al precio de la última térmica no inflexible despachada) a febrero de 2016 y febrero de 2024
  (El Niño fuerte), estima el MPO ajustado con una red neuronal a partir de aportes y reservas, y lo presenta en un
  dashboard (real, ajustado y estimado, por día).
- **Uso en el proyecto:** antecedente de Uninorte sobre la formación del precio (MPO) y el riesgo regulatorio. La
  asesora es la profesora Adriana Arango Manrique.

---

## Regulación y funcionamiento del mercado (CREG y XM)

### Documento CREG 901 044 (2023) — metodología del precio de bolsa y propuesta de ajuste temporal
> Comisión de Regulación de Energía y Gas (2023). *Metodología para la determinación del precio de bolsa —
> propuesta de ajuste temporal* (Documento CREG 901 044, 18 de diciembre de 2023; soporte del Proyecto de
> Resolución CREG 701 028 de 2023).

- https://gestornormativo.creg.gov.co/gestor/entorno/docs/originales/Proyecto_Resoluci%C3%B3n_CREG_701_028_2023/Documento%20Soporte%20CREG%20901%20044%20de%202023.pdf
- **Uso:** explica la formación del precio de bolsa nacional (PBN) según la CREG 024 de 1995 (modificada por la 011 de
  2010): despacho ideal ex post con demanda real y disponibilidad real, sin red; MPO del último recurso en mérito; más
  el ΔI (CREG 051 de 2009, modificada por la 101 028 de 2022). Datos para el planteamiento del problema: 67 % de la
  capacidad es hidráulica (19.988 MW), tres agentes tienen el 65 %, la hidráulica cubre 70-80 % de la demanda y cae a
  cerca de 50 % con El Niño, índice de Lerner promedio de 0,93 en sep-2023, exposición a bolsa de comercializadores de
  20 % a 100 %.
- **Cautela:** el "PBNA" (precio de bolsa nacional ajustado, 532 $/kWh) es una **propuesta** para liquidar
  transacciones durante El Niño 2023-2024, no el PBN que publica XM, y no hay constancia de que se adoptara. No
  confundirlo con el concepto PBNAL de los archivos de liquidación.

### Proyecto CREG 701 049 de 2024 — ajuste del MPO hidráulico (no adoptado)
> CREG, Proyecto de Resolución 701 049 de 2024 y Documento CREG 901 097 de 2024 (*Metodología para la determinación
> del precio de bolsa: propuestas de ajuste*); Superintendencia de Industria y Comercio, Concepto 183210 de 2024.

- https://gestornormativo.creg.gov.co/gestor/entorno/docs/originales/Proyecto_Resoluci%C3%B3n_CREG_701_049_2024/
- https://gestornormativo.creg.gov.co/gestor/entorno/docs/concepto_superindustria_0183210_2024.htm
- **Qué propone:** si todos los precios horarios del día están por debajo del precio de escasez de activación y el
  MPO de una hora lo fija una planta hidráulica, el ASIC lo ajusta al precio de la última térmica no inflexible
  despachada. Aprobado para consulta el 31-may-2024, publicado el 6-jun-2024, comentarios hasta el 21-jun-2024.
  La SIC cuestionó su justificación.
- **Estado (verificado el 19-oct-2026):** la carpeta de la CREG solo tiene el proyecto, el documento soporte y el
  formato de comentarios; no hay resolución definitiva. Se trata como no adoptado. Es la regla que estudiaron
  Sánchez y Noguera (2024).

### Resoluciones CREG 101 056 y 101 059 de 2024 — precio de oferta ajustado (adoptadas)
> CREG, Resolución 101 056 de 2024 (sesión 1345, 9-oct-2024) y Resolución 101 059 de 2024 (29-oct-2024; Diario
> Oficial 52.925 del 30-oct-2024), que modifican la Resolución CREG 026 de 2014 (Estatuto para Situaciones de Riesgo
> de Desabastecimiento).

- https://gestornormativo.creg.gov.co/gestor/entorno/docs/originales/Resoluci%C3%B3n_CREG_101_056_2024/
- https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_101-59_2024.htm
- **Qué hacen:** en el mecanismo de sostenimiento de la confiabilidad, cuando hay que embalsar agua, las ofertas de
  las hidráulicas comprometidas se llevan al precio de oferta ajustado (POA). La 101 056 lo redefinió como el mayor
  precio térmico del día más su arranque-parada variabilizado (XM: el POA del 30-sep-2024 pasó de 7.233 a 2.497
  COP/kWh) y aplicó desde las transacciones de septiembre de 2024; la 101 059 lo volvió a definir con un
  procedimiento de selección (Anexo 3) y un desempate de hidráulicas por factor de reserva.
- **Para el proyecto:** son cambios de regla dentro del periodo de entrenamiento (sep-oct de 2024). Solo actúan en
  períodos de riesgo.

### Resolución CREG 101 127 de 2026 — medidas transitorias ante El Niño 2026-2027 (vigente y activada)
> CREG, Resolución 101 127 de 2026 y Documento CREG 901 431 de 2026 (*Medidas transitorias para asegurar el
> abastecimiento de energía ante el fenómeno de El Niño 2026-2027*), sesión 1490, 27-ago-2026.

- https://gestornormativo.creg.gov.co/gestor/entorno/docs/originales/Resoluci%C3%B3n_CREG_101_127_2026/
- Activación preventiva anunciada el 9-sep-2026:
  https://creg.gov.co/publicaciones/16044/la-creg-activa-medidas-preventivas-para-proteger-la-confiabilidad-y-el-abastecimiento-de-energia-en-el-pais/
- Seguimiento de XM: https://www.xm.com.co/resoluciones/operacion-y-mercado/resolucion-creg-101-127-de-2026-condicion-del-sistema
- **Qué hace:** suspende el Estatuto de la CREG 026 de 2014 hasta el 31-may-2027 y lo reemplaza por tres componentes:
  (1) curvas ISO-GT que traducen el nivel del embalse en la generación térmica requerida, con disparador según
  embalse, precio de bolsa en punta frente al PEA y térmica promedio de 7 días; (2) despacho térmico mínimo obligado
  (la que no sale en mérito se paga por reconciliación positiva, vía restricciones); (3) referencia de oferta para
  las hidráulicas con embalse: curva de tres puntos según el volumen útil (PES × 1,10 en el mínimo técnico, PES +
  0,01 $/kWh en un X % que fija la CREG, otros costos variables en el nivel de vertimiento), con desviaciones de ±10 %
  reportadas a la SSPD. El precio de bolsa se acota al costo variable de la planta más costosa con OEF.
- **Contexto del documento:** embalse en 79,02 % el 24-ago-2026, 3,3 puntos por debajo de la senda (unos 580 GWh);
  demanda máxima histórica de 262,65 GWh-día el 8-jul-2026, con crecimientos de 6,5 % a 8,8 % anual entre abril y
  julio; NOAA (13-ago-2026) asigna más de 90 % de probabilidad a un El Niño muy fuerte y 69 % a uno histórico en
  oct-dic de 2026.
- **Para el proyecto:** cambia la formación del precio desde septiembre de 2026, fuera de lo que el modelo vio al
  entrenar. Es la explicación regulatoria más directa del mayor error de sep-oct de 2026. El predespacho ideal sí
  incorpora las nuevas ofertas, lo que ayuda a que Augur se adapte.

### Liquidación del MEM: versiones TX1, TX2, TXR y TXF

### Liquidación del MEM: versiones TX1, TX2, TXR y TXF
> CREG, Resolución 157 de 2011 (arts. 21-24), modificada por la Resolución CREG 211 de 2015 (art. 2); compilada en
> los artículos 2.9.2.3.1 a 2.9.2.3.4 de la Resolución Única de Regulación del Sector Eléctrico.
> CREG, Concepto 6762 de 2020 (versiones TXR y TXF).

- https://gestornormativo.creg.gov.co/gestor/entorno/docs/ee_2_9_2_3.htm
- https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_0211_2015.htm
- https://gestornormativo.creg.gov.co/gestor/entorno/docs/concepto_creg_0006762_2020.htm
- **Uso:** calendario de publicación del precio. Primera liquidación (TX1) a más tardar el día D+2 con la información
  disponible; observaciones de los agentes hasta D+3; segunda liquidación (TX2) antes de las 11:00 de D+5;
  observaciones hasta D+6; resumen mensual preliminar (TXR) en los primeros días del mes siguiente; versión final
  (TXF) hacia el día 10, con la facturación; después, notas de ajuste (TX3, TX4…). Sustenta el rezago de unos 3 días
  con que el proyecto usa el precio real y explica por qué no existe una fuente con el precio del día anterior a tiempo.

### Precio de escasez y precio de transacciones en bolsa
> CREG, Resolución 071 de 2006 (Cargo por Confiabilidad); Resolución 140 de 2017 (precio marginal de escasez, precio
> de escasez de activación y ponderado); Resolución 101 066 de 2024 (precios de escasez superior e inferior).

- https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_0140_2017.htm
- https://gestornormativo.creg.gov.co/gestor/entorno/docs/originales/Resoluci%C3%B3n_CREG_101_066_2024/
- **Uso:** cuando el PBN supera el precio de escasez, las compras y ventas en bolsa no se liquidan al PBN sino a un
  precio de transacciones (PTB). Desde la 101 066 de 2024 hay tres tramos (PE1 < PE2 < PE3) con un precio de escasez
  inferior de referencia de 359 $/kWh. Medido con la API de XM (19-oct-2026): en 2026 el PTB (`PrecTransBolsa`) fue
  menor que el PBN en 360 horas (315 en mayo, con 102 COP/kWh de diferencia media; 22 en sep-oct, con unos 600
  COP/kWh). Augur pronostica el PBN; el valor económico del motor en esas horas no es el que pagaría o recibiría un
  agente.

---

## Citadas en el informe pero sin PDF en esta carpeta

Verificadas el 23-oct-2026 contra el PDF y Crossref (los PDF están en Downloads\Referencias_Augur; la lista completa
con su uso está en docs/Referencias_Augur_paper.docx):

- **Herrera-Mejía, L., Hoyos, C. D., La Rotta, D., Mejía, Y., & Castillo, J. (2026).** *Forecasting day-ahead
  hydropower bids in the Colombian electricity market: A two-stage machine learning framework.* Smart Energy, 21,
  100221. https://doi.org/10.1016/j.segy.2025.100221 (no es de 2025: el volumen es de febrero de 2026).
- **Muñoz-Santiago, A., Urquijo-Vanstrahlengs, J., Castro-Otero, A., & Lombana, J. (2017).** *Pronóstico del precio
  de la energía en Colombia utilizando modelos ARIMA con IGARCH.* Revista de Economía del Rosario, 20(1), 127-161.
  (El nombre es Alberto, no «N.»; Crossref registra mal la página.)
- **Dias, M. B. B., Lira, G. R. S., & Freire, V. M. E. (2024).** *Methodology for multi-step forecasting of
  electricity spot prices based on neural networks applied to the Brazilian energy market.* Energies, 17(8), 1864.
- **International Energy Agency (2023)** — Colombia 2023, Energy Policy Review (fuente del 72 % hidroeléctrico del
  informe final).
- **Low Carbon Power (2025)** — matriz de generación de Colombia (70-74 % hidroeléctrico); se citaba en un borrador
  anterior y el informe final ya no la usa.
