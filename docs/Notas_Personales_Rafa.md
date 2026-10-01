# Notas personales — Motor de Decisión y Dashboard

> **Esto es para ti, Rafael** — no es el entregable formal para los asesores (para eso están los
> `.docx` en esta misma carpeta). Es tu propio mapa para ubicarte rápido cuando tengas una duda,
> sin releer todo el chat. Está organizado igual que el código: si tienes una duda sobre una parte
> específica, busca el subtítulo que se llama igual.
>
> Última actualización: 2026-09-25.

## Índice rápido

- **Parte 1 — El motor** (`src/motor_decision.py`): los 4 métodos, por qué 25/75, cómo elige el
  ganador, qué respalda cada cosa.
- **Parte 2 — El dashboard** (`dashboard/app.py`): barra lateral, vista Operador, vista Analista.
- **Parte 3 — El notebook de pruebas**.
- **Parte 4 — Resultados que ya tengo** (para no recalcular).
- **Parte 5 — Pendientes reales**.

---

# Parte 1 — El motor de decisión (`src/motor_decision.py`)

## 1.1 El contrato de datos

Todo entra por `cargar_fuente_pronostico(raiz, horizonte)`, que lee
`data/processed/resultados/fuentes_pronostico.json` (dice qué modelo/archivo usar por horizonte) y
valida con `validar_contrato_pronostico()` que el CSV tenga las columnas `fecha_hora, real, q10,
q50, q90`. Si esto cambia de modelo algún día, es editar el JSON — no tocar código.

## 1.2 Método `fijo`

Umbrales = percentiles 25 y 75 del **precio histórico 2019-2025** (`umbrales_fijos()`). Se calculan
una vez y nunca cambian.

**Se rompe cuando el precio actual se aleja mucho de ese histórico** (pasó en 2026 con El Niño): si
todo el mes está por encima del percentil 75 histórico, el método dice "caro" el 100% de las horas
— deja de discriminar nada.

## 1.3 Método `rodante`

Umbrales = percentiles de una ventana móvil de **30 días** sobre el propio `q50` de la serie
(`umbrales_rodantes()`), **causal** (solo mira hacia atrás, nunca el propio instante). Se adapta al
régimen reciente, sin importar el histórico.

**Su debilidad**: no tiene filtro de confianza — puede disparar señal aunque el pronóstico de esa
hora sea muy incierto.

## 1.4 Método `banda`

= `fijo` + un filtro extra: si el ancho de la banda `[q10,q90]` de esa hora está en el percentil 75
o más alto de los anchos históricos (poca confianza del pronóstico), la señal se fuerza a `esperar`.

Fue el método "ganador oficial" hasta el 2026-09-24 — pero hereda el problema de `fijo` (compara
contra 2019-2025), solo que el filtro de confianza lo disimulaba parcialmente.

## 1.5 Método `hibrido` (nuevo, 2026-09-24)

= el umbral de `rodante` (se adapta) + el filtro de confianza de `banda` (ancho de banda). Nunca se
había probado esa combinación antes de esta sesión — surgió directamente de ver que `rodante` era
más estable que `banda` pero le faltaba el filtro de confianza.

```python
if metodo in ("rodante", "hibrido"):
    bajo, alto = umbrales_rodantes(precio, ventana_dias, p_bajo, p_alto)
...
if metodo in ("banda", "hibrido"):
    senal = senal.mask(ancho > umbral_ancho, "esperar")
```

## 1.6 ¿Por qué 25 = "barato" y 75 = "caro"? (y en qué nos basamos para ese margen)

**Respuesta corta: es una convención estadística, no algo que optimizamos nosotros.** El rango
25-75 (rango intercuartílico) es el criterio clásico de Tukey para separar "valores típicos" de
"atípicos" en cualquier distribución — pero ese criterio se diseñó para **detectar outliers**, no
para fijar el umbral de una regla de compra/venta. Son problemas parecidos, no el mismo.

**Lo que sí hicimos**: un barrido probando 6 pares de percentiles (10/90, 15/85, 20/80, 25/75,
30/70, 35/65) con el método `fijo`, midiendo la ventaja económica y la frecuencia de acción de
cada uno, sobre el holdout 2026 completo y partido en dos mitades (ene-abr / may-ago):

**Generador, 24h:**

| Percentiles | Ventaja (año completo) | Frecuencia | Ventaja ene-abr | Ventaja may-ago |
|---|---|---|---|---|
| 10 / 90 | 433.2 COP/kWh | 20% | 736.8 | 197.8 |
| 15 / 85 | 378.1 | 30% | 532.5 | 143.5 |
| 20 / 80 | 348.2 | 30% | 365.9 | 116.7 |
| **25 / 75** | **313.3** | **40%** | 289.4 | 88.6 |
| 30 / 70 | 274.4 | 40% | 199.3 | 73.8 |
| 35 / 65 | 225.8 | 50% | 153.1 | 62.2 |

**Comercializador, 24h:**

| Percentiles | Ventaja (año completo) | Frecuencia |
|---|---|---|
| 10 / 90 | 285.0 COP/kWh | 0% |
| 15 / 85 | 286.4 | 0% |
| 20 / 80 | 283.7 | 10% |
| **25 / 75** | **281.4** | **10%** |
| 30 / 70 | 277.8 | 20% |
| 35 / 65 | 272.7 | 20% |

**Cómo leer esto:**
- Para **generador**, no hay un "pico" real — es una frontera de intercambio pura: percentil más
  extremo (10/90) = mayor ventaja por acción pero mucha menos frecuencia; percentil más angosto
  (35/65) = al revés. El 25/75 cae en un punto razonable de esa frontera (el que mejor balancea
  ventaja × frecuencia entre los 6 probados).
- Para **comercializador**, el percentil elegido **casi no importa** — la ventaja se mueve apenas
  entre 272 y 286 en todo el rango probado.
- Con ambas mitades del año por separado se ve el problema real (sección 4): con **cualquier**
  percentil, la frecuencia del generador se desploma en ene-abr y se dispara en may-ago — el
  percentil exacto no es la causa del problema de fondo.

**Conclusión honesta**: 25/75 no está mal, pero tampoco está "demostrado" como el mejor — es una
convención razonable que además resultó ser un punto decente de la frontera al probarla. Si algún
asesor pregunta "¿por qué 25 y no 20?", la respuesta correcta es esta, no "porque sí".

**Si quiero blindar esto más adelante**: buscar literatura de "detección de picos de precio en
mercados eléctricos" (Christensen, Hurn & Lindsay 2009; Janczura & Weron) — ahí sí definen
"caro"/"barato" con criterios específicos del dominio (media ± k desviaciones estándar), no con un
percentil genérico tipo Tukey.

## 1.7 `evaluar_backtest()` — cómo se mide si una señal "funciona"

Compara el precio real promedio en las horas donde la señal dice actuar (comprar/vender) contra el
precio real promedio de todo el periodo. Positivo = la señal encuentra horas mejores que el
promedio. **No es una simulación de portafolio** — no considera volumen, contratos, ni costos de
transacción. Es la comprobación mínima de que la regla no es puro azar.

## 1.8 `comparar_metodos()` + `elegir_mejor_metodo()` — selección por promedio (la de siempre)

Corre los 4 métodos, marca "válido" el que actúa entre 10%-40% de las horas, elige el de **mayor
ventaja promedio** entre los válidos.

**Su punto ciego**: solo mira el promedio de todo el periodo. No detecta que ese promedio puede
estar sostenido casi enteramente por una sola parte del año.

Desde el 2026-09-24 acepta también `mascara_evaluacion` (opcional): calcula la señal sobre toda la
serie (para que `rodante`/`hibrido` tengan continuidad) pero mide el backtest solo en el rango que
le pases — esto arregló el bug de la vista Analista (sección 2.3).

## 1.9 `comparar_metodos_estable()` + `elegir_mejor_metodo_estable()` — selección por estabilidad (nuevo)

Agregado el 2026-09-24. Parte el periodo en dos mitades y elige el método con mejor resultado en su
**mitad más débil** (criterio maximin, no promedio). Un método solo se marca `estable=True` si:

- su frecuencia es válida (10%-40%) **en las dos mitades**, y
- su ventaja es **≥ 0 en las dos mitades**.

**Resultado de aplicarlo** (reemplazó al criterio de promedio en la vista Operador):

| Horizonte | Rol | Ganador antes (promedio) | Ganador ahora (maximin) |
|---|---|---|---|
| 24h | generador | `banda` | **`hibrido`** |
| 24h | comercializador | `banda` | **`rodante`** |
| 72h | generador | `banda` | **`hibrido`** |
| 72h | comercializador | `banda` | **`rodante`** |

## 1.10 ¿Qué respalda cada decisión? (semáforo rápido)

| Decisión | Estado |
|---|---|
| Percentiles 25/75 | 🔴 Convención (Tukey), sin optimización formal — ver 1.6 |
| Ventana rodante 30 días | 🔴 Sin cita externa; se probó sensibilidad (parte 4): 30 días es más estable entre mitades que 90, aunque 90 gane en promedio simple |
| Percentil 75 de ancho de banda | 🔴 Sin cita, pero la curva es plana (60-90 dan resultados parecidos) |
| Rango de frecuencia válida 10-40% | 🔴 Heurística razonable, sin prueba formal |
| Método `hibrido` | 🟡 Combina 2 piezas ya validadas por separado; la combinación en sí no está en ningún paper |
| Criterio maximin (2 mitades) | 🟡 El criterio general (Wald 1950) sí es formal; el corte en 2 mitades fijas es propio |

**Papers pendientes de leer** (antes de defender esto formalmente):
- Wald (1950), *Statistical Decision Functions* — el criterio maximin.
- Garivier & Moulines (2011), *On UCB Policies for Non-Stationary Bandit Problems* — marco formal
  de "elegir automáticamente la mejor opción cuando el entorno cambia".
- Cesa-Bianchi & Lugosi (2006), *Prediction, Learning, and Games* — alternativa al corte fijo.
- Gama et al. (2014), *A Survey on Concept Drift Adaptation* — el fenómeno de fondo tiene nombre
  formal en la literatura.

*(El detalle completo de esta tabla está en `Informe_Diseno_Motor_Decision_Dashboard.docx`.)*

## 1.11 ¿Se implementaron Corr-f / MHD / MPD? — **No, todavía no**

Quiero que esto quede clarísimo porque es fácil de confundir. Estas 3 métricas **no están
conectadas al motor**:

- Viven en `scripts_experimento/metricas_decision.py`, como herramienta de evaluación de **modelos
  de pronóstico** (no del motor de decisión).
- Se corrieron una sola vez como demo (`metricas_decision_demo.csv`).
- **El motor sigue decidiendo solo con el nivel de `q50`** — nunca mira si el pronóstico acierta la
  forma del día (Corr-f), el momento del pico/valle (MHD), ni el tamaño del movimiento (MPD).

**Por qué no se integraron**: necesitan el precio real del día para calcularse, así que no sirven
como filtro causal para "hoy". Solo se podrían usar (a) para elegir qué modelo alimenta el motor, o
(b) como una señal de confianza retrospectiva/rolling (nueva, no diseñada todavía). Sigue pendiente
— ver Parte 5.

---

# Parte 2 — El dashboard (`dashboard/app.py`)

## 2.1 Barra lateral

Tres controles siempre visibles: **Vista** (Operador/Analista), **Rol** (generador/comercializador),
**Horizonte** (24h/72h). Debajo, un indicador de si el modelo activo está calibrado (verde) o no
(amarillo), con su cobertura medida.

## 2.2 Vista Operador — "¿qué hago ahora?"

Sin controles técnicos — el método lo elige solo `comparacion_estable()` (criterio maximin,
sección 1.9).

- **Selector de día y hora**: un stepper (`st.number_input`, flechas ‹ › — antes era un slider que
  se veía como barra de progreso) **+** una franja de 24 botones clickeables, uno por hora,
  sincronizados entre sí por `st.session_state["hora_operador"]`. Clickear una hora mueve el
  stepper también, y viceversa.
- **Tarjeta grande**: la recomendación (COMPRAR/VENDER/RETENER/EVITAR COMPRA/ESPERAR) con color,
  una frase explicando por qué, una barra que ubica el precio esperado entre "barato" y "caro", y
  un chip de confianza (alta/media/baja) según el ancho de banda de esa hora.
- **Indicadores**: horas para actuar hoy, ventaja histórica de la regla, con qué frecuencia actúa.
- **Nota al final**: dice qué método quedó activo y por qué (ahora menciona el criterio de
  estabilidad, no el de promedio).

## 2.3 Vista Analista — controles finos

- **Percentiles bajo/alto**: slider 5-95, por defecto 25/75 (ver sección 1.6 para el porqué).
- **Método de umbral**: selectbox con las 4 opciones (`fijo`, `rodante`, `banda`, `hibrido`),
  sugiere el ganador del backtest en el rango de fechas elegido.
- **Rango de fechas**: slider. **Desde el 2026-09-24, este slider sí recalcula el método sugerido**
  — antes tenía un bug: el rango solo afectaba el backtest del método ya elegido a mano, nunca cuál
  quedaba "sugerido" (que siempre venía del año completo). Verificado: con el año completo sugiere
  `banda`; filtrando al último mes, sugiere `hibrido`.
- **Tabla comparativa completa**: los 4 métodos con ventaja, frecuencia, y si son "válidos".
- **Gráfica** de precio real + banda + señales, y un expander con los datos crudos filtrados.

---

# Parte 3 — El notebook de pruebas (`notebooks/pruebas_precision_motor_decision_Rafa.ipynb`)

Para probar rápido sin abrir el dashboard completo.

## 3.1 Parámetros (editar y volver a correr)

`HORIZONTE`, `ROL`, `FECHA_INICIO`/`FECHA_FIN`, `INCLUIR_HIBRIDO`, y **`POTENCIA_W`** (potencia de
un cliente de ejemplo, en vatios — hoy 100.000 W / 100 kW, es un placeholder, cambiarlo cuando haya
un cliente real).

## 3.2 Qué calcula y muestra

1. Cobertura y MAE del pronóstico en el rango elegido (esto es del pronóstico, igual para los 4
   métodos).
2. Tabla + gráfica de barras con la ganancia/pérdida **total en pesos** por método
   (`ventaja_cop_kwh × horas_de_acción × potencia_kW`) — no solo la tasa COP/kWh.
3. El eje de la gráfica se ajusta solo al rango real de valores (para que el título nunca se pise
   con las etiquetas, sin importar si los valores son chicos o están en los millones).

**Cuidado técnico si lo edito**: si voy a evaluar un rango de fechas recortado, `rodante`/`hibrido`
necesitan calcularse sobre la serie **completa** primero (continuidad de la ventana móvil) y
filtrarse *después* — nunca cortar el DataFrame antes de llamar a `generar_senales()`.

---

# Parte 4 — Resultados que ya tengo (para no recalcular)

**Último mes disponible (2026-07-07 a 2026-08-05), 24h, generador:**

| Método | Ventaja | Frecuencia | ¿Válida? | Total en pesos (cliente 100 kW) |
|---|---|---|---|---|
| fijo | 0 COP/kWh | 100% | No | 0 COP |
| rodante | +80.5 | 50.6% | No | +2.930.814 COP |
| banda | −16.8 | 53.5% | No | −646.227 COP |
| **hibrido** | **+58.8** | **25.1%** | **Sí** | **+1.064.738 COP** |

**Precisión del pronóstico ese mismo mes** (artifact "Bitácora de Bandas"): cobertura 73.2%
(objetivo 80%), MAE 93.6 COP/kWh (vs. 54.34 del promedio anual — ~72% peor), ancho de banda
promedio 242.4 COP/kWh.

**Diagnóstico**: el resultado negativo de `banda` ese mes es principalmente culpa del **método**
(compara contra 2019-2025, y el precio de jul-ago ya está muy por encima de ese histórico), no del
**modelo de pronóstico** — al usar el propio mes como referencia en vez de 2019-2025, `fijo` también
da positivo con el mismo pronóstico. La cobertura/MAE degradados sí son responsabilidad del modelo,
pero es un problema aparte.

**Ventana rodante**: 90 días gana en el promedio anual, pero es *inestable* entre mitades del año
(mejora en ene-abr, empeora en may-ago). 30 días es más parejo — ejemplo real de que "el promedio
anual más alto" puede ser una métrica engañosa.

---

# Parte 5 — Pendientes reales (no wishlist)

- [ ] Diseñar e implementar la versión "rolling" de Corr-f/MHD/MPD para el motor (sección 1.11).
- [ ] Leer los 4 papers de la sección 1.10 antes de defender el método híbrido/maximin formalmente.
- [ ] La vista Analista todavía no expone el criterio maximin — solo Operador lo usa. Decidir si
      mostrarlo también en Analista para comparar los dos criterios lado a lado.
- [ ] Backtest con simulación de portafolio real (la traducción a pesos del notebook es ilustrativa,
      asume potencia constante, no contratos ni curva de carga real).
- [ ] Confirmar con un cliente real la `POTENCIA_W`, en vez del placeholder de 100 kW.
