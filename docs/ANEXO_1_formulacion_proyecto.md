# Anexo 1 — Formulación del Proyecto (versión definitiva, 2026-09)

> Guardado el 2026-09-10 a partir del PDF `Anexo 1_Formulación del Proyecto - 2026_30.pdf` que Juan
> compartió, marcado explícitamente por él como **la versión definitiva** del Anexo 1. Se transcribe
> aquí para tenerlo siempre disponible como referencia, sin depender de que se vuelva a adjuntar en
> cada sesión.
>
> **Recordatorio importante** (también documentado en el README del proyecto): según las
> ACLARACIONES del propio Anexo 1, los proyectos aprobados **no pueden modificarse durante el
> semestre en los ítems 1 (Identificación), 3 (Objetivos) y 6 (Cumplimiento ABET)**. El resto de
> ítems (Delimitación/Alcances, Metodología, Cronograma, etc.) no están bajo esa misma restricción
> formal, aunque cualquier cambio ahí debería igual pasar por los asesores.

## 1. Identificación del proyecto

- **Título:** Diseño de una plataforma de procesamiento y analítica de datos para el pronóstico del
  precio de la energía y el apoyo en la toma de decisiones.
- **Asesores:** José Daniel Soto Ortiz, Daniela María Charris Stand.
- **Estudiantes:** Juan David Barceló Barraza (Cód. 200181896, CC 1044606227), Rafael Andrés Dede
  Perdomo (Cód. 200180841, CC 1043436500).
- **Duración:** 16 semanas.

## 2. Origen del proyecto

**2.1 Planteamiento del problema.** El mercado eléctrico colombiano (administrado por XM) opera con
precio de bolsa horario. Generadores y comercializadores deciden cuándo comprar/vender bajo alta
incertidumbre, influenciada por hidrología (embalses, aportes), El Niño/La Niña, estacionalidad de
demanda, disponibilidad térmica y el dólar. Cerca del 70% de la generación nacional es hidroeléctrica,
así que en eventos secos el precio puede multiplicarse varias veces. Los datos son públicos pero su
presentación convencional (tablas/curvas) no permite reconocer patrones, riesgo ni ventanas de
decisión, en particular para empresas medianas sin herramientas analíticas avanzadas. Es un problema
complejo de ingeniería: múltiples señales horarias acopladas, no lineales, con componentes de alta y
baja frecuencia (diaria/semanal/anual) que deben adquirirse, sincronizarse, depurarse y transformarse
en información interpretable. Ocurre de forma permanente en la operación diaria del mercado spot y
se agudiza en temporadas climáticas extremas.

**2.2 Justificación.** La plataforma no solo pronostica el precio (horizonte 24-72h) sino que lo traduce
en material de apoyo a la decisión, interpretable, trazable y con su incertidumbre asociada. Cuatro
argumentos: (1) los antecedentes colombianos —Agudelo et al. 2015 [1], Muñoz-Santiago et al. 2017
[2]— usan series mensuales o datos previos a la reforma CREG de 2015, ninguno combina PDS con ML
moderno sobre variables horarias; (2) las herramientas comerciales (SIGE, Globalsyde, Enexpa) son para
facturación/contratos de comercializadoras minoristas españolas, no aplican al mercado mayorista
colombiano; (3) no existe un sistema de recomendación accionable — la investigación predice precio
pero no lo traduce en decisiones; (4) el proyecto compara arquitecturas mediante una cadena común de
extracción de características, con explicabilidad (SHAP), cuantificación de incertidumbre y validación.

**2.3 Análisis del cliente.** Empresas generadoras/comercializadoras del mercado spot, en particular
medianas sin herramientas analíticas avanzadas. Buscan: (1) predicciones 24-72h con recomendación
explícita según rol (comprador/vendedor); (2) biblioteca de imágenes de apoyo a la decisión (mapas de
calor) con nivel de confianza y variables dominantes; (3) dashboard accesible sin conocimientos
técnicos.

## 3. Objetivos

**General:** Diseñar, implementar y validar una plataforma de procesamiento de datos que permita el
análisis y pronóstico del precio de la energía en el mercado eléctrico colombiano para la toma de
decisiones.

**Específicos:**
- **OE1.** Adquirir, sincronizar y caracterizar temporal y espectralmente las series históricas de las
  variables del mercado, mediante técnicas de procesamiento de datos, para seleccionar variables
  predictoras.
- **OE2.** Implementar y evaluar diferentes modelos de pronóstico del precio de bolsa, comparando
  precisión frente a persistencia, robustez, costo computacional e interpretabilidad.
- **OE3.** Diseñar e implementar un motor de decisión que traduzca los pronósticos y su
  incertidumbre en señales de compra, venta o espera.
- **OE4.** Validar el funcionamiento del motor para la toma de decisiones.

## 4. Delimitación del proyecto

### 4.1 Alcances

**OE1**
1. Análisis exclusivamente sobre datos públicos disponibles.
2. Pipeline de datos automatizado (API REST de XM y SIMEM).
3. Sincronización mínimo a resolución horaria: precio de bolsa, demanda, generación, volumen de
   embalses, aportes hídricos (entre otras).
4. Datos históricos de enero de 2019 a agosto de 2026.
5. Caracterización temporal y espectral (estadísticas descriptivas, estacionalidad, periodograma) para
   identificar componentes armónicas dominantes.

**OE2**
1. Extracción de características temporales compartidas (rezagos, promedios móviles, armónicas de
   calendario) para al menos 2 modelos.
2. Comparación con horizonte de 24 a 72 horas.
3. Evaluación con MAE, RMSE, MAPE, persistencia y costo computacional.

**OE3**
1. Motor de decisión con reglas basadas en percentiles del precio y su incertidumbre, diferenciando
   rol comprador/vendedor.
2. Material visual de apoyo: semáforo compra/venta/espera, mapa de calor día-hora, bandas de
   incertidumbre y comparación real-vs-pronóstico, con metadatos (unidad, período, confianza, versión
   del modelo).
3. Dashboard que integre pronósticos, recomendaciones e imágenes de decisión.
4. Biblioteca de 4-5 tipos de imágenes con validación funcional.

**OE4**
1. Validación de la estrategia de recomendación mediante backtesting sobre datos históricos.
2. Validación del dashboard/imágenes con muestra de 3-5 usuarios, cuestionario de comprensión.

### 4.2 Limitaciones

**OE1**
1. No se garantiza acceso a datos operativos propietarios de empresas.
2. No se modelarán fenómenos de granularidad intra-horaria.
3. **No se realizará el desglose de generación por tipo de recurso (hidráulica/térmica).**

**OE2**
1. Predicciones de corto plazo no fuera del intervalo 24-72h.
2. No se explorarán modelos distintos a los encontrados en la literatura.

**OE3**
1. El sistema no ejecuta transacciones automáticas ni se integra con los sistemas de despacho de XM.
2. Validación con usuarios: muestra reducida, no estudio formal de usabilidad.
3. No se realizará simulación estructural de nivel 2 (no se simula el mecanismo físico/económico del
   despacho eléctrico colombiano).

**OE4**
1. Validación con usuarios ≤5 personas, no estudio formal de usabilidad.
2. El backtesting se basa en histórico y no garantiza desempeño ante condiciones no observadas
   (cambios regulatorios o hidrológicos extremos fuera del rango histórico).

### 4.3 Entregables

Anexo 1 · Dataset integrado y limpio (ene 2019 - ago 2026) · Notebooks de análisis exploratorio y
modelado · Biblioteca de imágenes de apoyo a la decisión · Módulo de recomendación y resultados de
backtesting · Dashboard · Informe final y repositorio en GitHub · Informe de avance (semana 10) ·
Manual de uso del motor de decisión (semana 16) · Informe final, póster, presentación final, video
(semana 16).

## 5. Recursos requeridos

2 computadores con capacidad de entrenamiento/procesamiento + internet · Entorno Python (editor +
entorno virtual) · Librerías: pandas, numpy, requests, CmdStan, matplotlib, seaborn, Prophet, XGBoost,
SHAP, Scikit-learn, scipy · GitHub · APIs públicas de XM/SIMEM, NOAA CPC, datos IDEAM · Servicio en
la nube · Framework de deep learning (TensorFlow o PyTorch).

## 6. Cumplimiento ABET (SO2 — diseño en ingeniería) — sección bloqueada, no modificar

Decisiones de diseño documentadas: (1) exclusión de variables contemporáneas (demanda, generación)
como predictoras por determinarse en el mismo despacho que el precio — solo versiones rezagadas;
(2) ventana de entrenamiento 2019-2026 justificada por el cambio regulatorio CREG de 2015 con años
de margen para estabilización; (3) sincronización de las 5 variables públicas exige conciliar
resoluciones distintas, verificar faltantes/atípicos, caracterizar en tiempo y espectro; (4) criterio de
selección de modelo: precisión vs. persistencia, costo computacional, interpretabilidad, curva de
aprendizaje, cantidad de parámetros; (5) matriz de alternativas explícita para decidir variable
continua vs. bandera binaria en fenómenos climáticos; (6) detección y corrección repetida de fugas de
información sutiles (horizonte 24-72h).

Prerrequisitos: Competencias Comunicativas I/II · Algoritmos y programación I · Análisis de datos en
ingeniería · Soluciones computacionales · Mediciones e instrumentación · Diseño Electrónico · Señales
y Sistemas · Comunicaciones · Control y agentes inteligentes.

### 6.3 Restricciones reales

- **6.3.1 Económica:** presupuesto limitado, prioriza infraestructura/software de la Uninorte y del
  equipo; código abierto y servicios de bajo costo; nube solo si se requiere mayor capacidad.
- **6.3.2 Ambiental:** el procesamiento masivo y entrenamiento de IA tiene impacto ambiental directo
  (demanda energética/hídrica de centros de datos). Restricción de diseño: **maximizar eficiencia
  computacional** — arquitecturas livianas/optimizadas, evitar reentrenamientos y ejecuciones
  redundantes, favorecer reutilización de datos procesados.
- **6.3.3 Social:** sistema exclusivamente de apoyo a la decisión — no ejecuta transacciones
  automáticas ni sustituye el criterio del usuario. Requiere competencia técnica y acceso tecnológico
  del usuario (no apto para público general sin perfil técnico).
- **6.3.4 Legal/normativa:** uso exclusivo de información pública (XM/SIMEM), amparado en
  Resolución CREG 101-18 de 2022. Respeta Ley 23/1982 y Decisión Andina 351 (propiedad
  intelectual). No ejecuta transacciones ni actúa como agente ante ASIC/XM. Si hay autenticación de
  usuarios: Ley 1581/2012 (datos personales). **La CREG puede hacer intervenciones regulatorias
  transitorias en la formación del precio de bolsa** (p. ej. techos a ofertas en escasez) — obliga a
  adaptar/delimitar la vigencia de los modelos entrenados frente a cambios estructurales.
- **6.3.5 Constructiva:** infraestructura estándar, software open source, arquitectura modular
  desacoplada (adquisición/procesamiento/modelado/incertidumbre/visualización), dashboard
  responsivo.
- **6.3.6 Tiempo:** 16 semanas académicas, cronograma fijo.

### 6.4 Factores SO2 (salud pública, seguridad, bienestar, cultural, social, ambiental, económico)

- **Salud pública:** escaladas del precio (ej. El Niño 2023-2024) afectan presupuestos operativos de
  IPS y otras instituciones — contexto, no depende del proyecto.
- **Seguridad:** cambios en política de datos personales (Ley 1581/2012) o en condiciones de acceso
  de las APIs de XM/SIMEM podrían restringir la disponibilidad de fuentes sin previo aviso.
- **Bienestar:** carga académica simultánea de ambos integrantes, cronograma fijo de 16 semanas sin
  margen institucional adicional.
- **Cultural:** aceptación/resistencia del sector energético colombiano hacia herramientas de
  recomendación basadas en IA podría afectar adopción futura.
- **Social:** empresas medianas objetivo podrían carecer de personal técnico para interpretar
  probabilidades/incertidumbre/SHAP — limita el beneficio social esperado.
- **Ambiental:** eventos climáticos extremos durante el desarrollo podrían generar comportamientos
  atípicos en los datos en tiempo real que compliquen la validación; a la inversa, fallas eléctricas en
  la infraestructura de cómputo podrían interrumpir el trabajo.
- **Económicas:** variaciones COP/USD y cambios unilaterales en precios de proveedores de nube/IA
  (AWS, Claude) fuera del control del proyecto, impactan el presupuesto. Volatilidad macroeconómica,
  fluctuaciones del dólar y costos de combustibles alternativos alteran oferta/demanda del mercado
  eléctrico — pueden inducir picos severos y outliers que distorsionan el entrenamiento.

## 7. Metodología y WBS

- **Fase 1 — Cadena de procesamiento y caracterización** (27 jul - 16 ago, OE1). Trabajo conjunto.
  Dataset maestro horario sincronizado, tratamiento de faltantes/atípicos, variable de pandemia,
  caracterización temporal/espectral, comparación de filtros (media móvil vs. Savitzky-Golay),
  correlación precio-hidrología-demanda.
- **Fase 2 — Extracción de características y modelado** (17 ago - 30 sep, OE2). Trabajo dividido por
  roles. Características comunes (lags, medias móviles) + esquema walk-forward (5 orígenes,
  regímenes hidrológicos distintos); entrenar/comparar familias de modelos (Prophet, XGBoost, deep
  learning), SHAP donde aplique.
- **Fase 3 — Comparación y motor de decisión** (7-20 sep, OE3 parte 1). Trabajo conjunto. Evaluación
  formal vs. persistencia (MAE/RMSE/MAPE/costo), cuantificación de incertidumbre, diseño del motor
  de decisión por umbrales, backtesting (histórico + ene-ago 2026), informe comparativo.
- **Fase 4 — Imágenes de decisión y validación** (17 ago - 4 oct, OE3 parte 2). Trabajo conjunto. 4-5
  imágenes con metadatos, dashboard integrado, validación con 3-5 usuarios.
- **Fase 5 — Cierre** (12 sep - 8 nov). Trabajo conjunto. Ajustes y pruebas de reproducibilidad,
  memoria IEEE + manuales, repositorio ordenado, demo y presentación final.

**Nota de cronograma** (relevante para priorización): Fase 3 (comparación formal + motor de decisión)
corre del 7 al 20 de septiembre de 2026 — es la fase activa en este momento del proyecto.

## 10. Presupuesto (resumen)

| Rubro | Total | Financiación propia | Contrapartida Uninorte |
|---|---|---|---|
| Personal | $32.201.260 | $20.010.344 | $12.190.916 |
| Materiales/equipos/herramientas | $3.160.000 | $3.160.000 | $0 |
| Tecnología | $520.000 | $520.000 | $0 |
| Infraestructura | $928.000 | $0 | $928.000 |
| Transporte y logística | $296.000 | $296.000 | $0 |
| Comunicación y divulgación | $500.000 | $500.000 | $0 |
| **Gran total** | **$37.605.260** | **$24.486.344** | **$13.118.916** |

## 11. Impactos del proyecto (SO4)

- **Económico:** reduce incertidumbre en el análisis de precio, pero no garantiza resultados — debe
  complementarse con criterio del usuario.
- **Social:** facilita acceso a información analítica vía visualizaciones comprensibles; se debe evitar
  dependencia de recomendaciones automáticas, presentando sus limitaciones con transparencia.
- **Ambiental:** mejor planificación de compra/venta puede contribuir indirectamente a gestión más
  eficiente de recursos energéticos; se priorizarán modelos/procesos computacionalmente eficientes
  por el consumo asociado al cómputo/almacenamiento.
- **Global:** base para futuras herramientas de analítica energética en otros mercados, aunque la
  aplicación inicial se limita a las características y datos del mercado eléctrico colombiano.

## Referencias del Anexo 1

1. A. P. Agudelo, J. M. López-Lezama, y E. Velilla, "Predicción del Precio de la Electricidad en la Bolsa
   mediante un Modelo Neuronal No-Lineal Autorregresivo con Entradas Exógenas," *Información
   Tecnológica*, vol. 26, no. 6, pp. 99–108, 2015.
2. A. Muñoz-Santiago, J. Urquijo-Vanstrahlengs, A. Castro-Otero, y J. Lombana, "Pronóstico del precio
   de la energía en Colombia utilizando modelos ARIMA con IGARCH," *Revista de Economía del
   Rosario*, vol. 20, no. 1, pp. 127–161, 2017.
3. T. Chen y C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD '16, San Francisco, CA,
   2016, pp. 785–794.
4. ACIEM y Consejo Profesional Nacional de Ingenierías Eléctrica, Mecánica y Profesiones Afines,
   *Manual de Referencia de Tarifas en Ingeniería*, Cundinamarca, Colombia, 2015.

---

**Aclaraciones del propio documento:** los proyectos aprobados no pueden modificarse durante el
semestre en los ítems 1, 3 y 6. Las propuestas se envían por el enlace del curso en Brightspace con
carta remisoria.
