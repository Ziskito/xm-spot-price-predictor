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

### Das & Schlüter — regímenes con procesos neuronales condicionales
> Das, A., & Schlüter, S. *Regime-Aware Conditional Neural Processes with Multi-Criteria Decision
> Support for Operational Electricity Price Forecasting*. Universidad de Ulm / Ulm University of
> Applied Sciences, Alemania.

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
- **Corrección pendiente:** el informe cita este trabajo como 11(**1**); el PDF dice claramente
  Vol 11 · Issue 2 · 2021, páginas 67-74.

### Barrientos Marín, Tabares Orozco & Velilla (2018) — redes, ARMA e híbridos
> Barrientos Marín, J., Tabares Orozco, E., & Velilla, E. (2018). *Forecasting Electricity Price in
> Colombia: A Comparison Between Neural Network, ARMA Process and Hybrid Models*. International
> Journal of Energy Economics and Policy, 8(3), 97-106.

- Archivo: `IJEEP_2018_Forecasting_Electricity_Price_Colombia.pdf`
- **Uso:** antecedente colombiano directo. Los tres autores son de la Universidad de Antioquia.

---

## Citadas en el informe pero sin PDF en esta carpeta

Pendientes de incorporar y verificar contra el original:

- **Herrera-Mejía, L. et al. (2025)** — AES Colombia, clasificación de régimen de oferta de plantas
  hidroeléctricas. Smart Energy, Elsevier. *Título exacto sin verificar contra el PDF.*
- **Muñoz-Santiago, N. et al. (2017)** — ARIMA con especificación IGARCH, mercado colombiano.
- **Dias et al. (2024)** — Brasil, comparación contra el DECOMP oficial (Figura 5b).
- **International Energy Agency (2023)** — Colombia 2023, Energy Policy Review.
- **Low Carbon Power (2025)** — matriz de generación de Colombia (fuente del 70-74% hidroeléctrico).
