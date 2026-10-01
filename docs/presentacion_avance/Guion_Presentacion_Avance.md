# Guion — Presentación de avance (12 min)

> Archivo local, fuera del repositorio. El mismo texto está en las notas del orador de
> `Presentacion_Avance_Barcelo_Dede.pptx` (Vista Moderador en PowerPoint).

## Reparto

| Bloque | Diapositivas | Orador | Tiempo |
|---|---|---|---|
| Portada → Alcances y limitaciones | 1–5 | Rafael | 2:40 |
| Estado del arte → OE2: 72 h y robustez | 6–11 | Juan | 4:40 |
| OE3: motor de decisión → OE3: dashboard | 12–14 | Rafael | 2:30 |
| Dificultades y soluciones → Cronograma | 15–16 | Juan | 1:10 |
| Porcentaje de objetivos → Gracias / ¿Preguntas? | 17–20 | Rafael | 0:55 |

**Total: 11:55** (Rafael 6:05 · Juan 5:50). Quedan unos segundos de margen para los cambios de orador.

## Consejos

- Ensayen con cronómetro: si van largos, recorten en las diapositivas 6 y 13, no en los resultados.
- En los cambios de orador, quien termina presenta al siguiente (ya está escrito al final de las diapositivas 5, 11, 14 y 16).
- Las cifras del motor salen de `informe_avance_oe3.py`; para preguntas sobre el motor, usen `Guion_Preguntas_Asesora_OE3.md`.
- **Antes de presentar:** reemplacen la captura del dashboard de la diapositiva 14 por una del dashboard nuevo (selector de hora con franja de botones y método «híbrido»).

## Guion por diapositiva

### 1. Portada — Rafael (0:20)

Buenos días. Somos Juan David Barceló y Rafael Dede, y vamos a presentar el avance de nuestro proyecto final: una plataforma que pronostica el precio de bolsa de la energía en Colombia y lo traduce en una recomendación para quien compra o vende energía. Nuestros asesores son el profesor José Daniel Soto y la profesora Daniela Charris.

### 2. Índice — Rafael (0:15)

La presentación sigue este orden: primero el problema y los objetivos, luego el estado del arte y el diseño del sistema, y al final el estado actual de cada objetivo, el cronograma y lo que falta.

### 3. Problemática — Rafael (0:45)

El precio de bolsa se fija cada hora, y como cerca del 72 % de la generación colombiana es hidroeléctrica, depende mucho del agua disponible. En la gráfica se ve el precio diario de 2019 a 2026 y los episodios de El Niño: en semanas el precio puede multiplicarse. En 2026, por ejemplo, el precio medio pasó de unos 200 a casi 600 pesos por kilovatio-hora entre el primer y el segundo tramo del año. Los datos de XM son públicos, pero en tablas y curvas no dicen cuándo conviene comprar o vender, sobre todo para empresas medianas sin equipos de analítica. Eso es lo que queremos resolver.

### 4. Objetivos — Rafael (0:40)

El objetivo general es diseñar, implementar y validar una plataforma de procesamiento de datos para analizar y pronosticar el precio de la energía y apoyar la toma de decisiones. Se divide en cuatro objetivos específicos: el primero, adquirir y caracterizar los datos; el segundo, implementar y comparar modelos de pronóstico; el tercero, un motor de decisión que convierta el pronóstico y su incertidumbre en señales de comprar, vender o esperar; y el cuarto, validar ese motor.

### 5. Alcances y limitaciones — Rafael (0:40)

Estos son los alcances y limitaciones tal como quedaron en el Anexo 1. Trabajamos solo con datos públicos, a resolución horaria entre 2019 y agosto de 2026, y comparamos modelos a 24 y 72 horas. El motor usa percentiles del precio y su incertidumbre según el rol, y se valida con backtesting y con 3 a 5 usuarios. No ejecutamos transacciones ni nos integramos con el despacho, no desagregamos la generación por tipo de recurso, y el backtesting solo cubre condiciones que ya ocurrieron. Ahora Juan presenta el estado del arte.

### 6. Estado del arte — Juan (0:50)

En Colombia hay antecedentes con redes NARX, con ARIMA-IGARCH y con series funcionales; este último logra un MAPE de 6,7 %, pero en un periodo calmo, 2000 a 2017, y con un solo modelo. Internacionalmente, Lago y colaboradores definen el benchmark y las buenas prácticas del área, y en Nueva Zelanda, un mercado también hidroeléctrico, ninguno de 33 modelos le gana al pronóstico ingenuo. La brecha es clara: ningún trabajo cuantifica la incertidumbre ni la convierte en una recomendación. Ese es nuestro aporte.

### 7. Datos y variables — Juan (0:40)

Integramos seis fuentes públicas: precio, demanda, generación, embalses y aportes de XM, y el índice ONI de la NOAA para El Niño. Tienen resoluciones distintas, así que todo se lleva a resolución horaria: las diarias por interpolación y el ONI con una rampa que nunca usa datos futuros. El resultado es un dataset de 66.576 horas sin huecos. A partir de él construimos 40 variables, todas con al menos 24 horas de rezago para que el modelo no use información que en la práctica no tendría.

### 8. Diseño del sistema — Juan (0:45)

Este es el diseño del sistema en tres bloques, uno por objetivo. Primero, la cadena de datos: adquisición, sincronización y caracterización. Segundo, el pronóstico: seis modelos de familias distintas combinados en un ensamble, con bandas de incertidumbre. Tercero, la decisión: el motor lee el pronóstico por un contrato de datos, así que funciona con cualquier modelo, y el dashboard muestra la recomendación. Todo está en Python y versionado en GitHub.

### 9. Estado actual · OE1 — Juan (0:40)

El objetivo 1 está completo. El periodograma del precio muestra picos claros en 24, 12 y 168 horas: el ciclo diario, el de medio día y el semanal. Esas componentes definieron las variables del modelo: rezagos de 24 y 168 horas y armónicos de calendario. También comparamos filtros de tendencia, y el de Savitzky-Golay siguió las subidas de precio más rápido y con menos ruido que el promedio móvil.

### 10. OE2: pronóstico a 24 h — Juan (1:00)

En el objetivo 2 comparamos seis modelos contra la persistencia, que repite el precio del día anterior y es la referencia estándar del área. El ensamble de seis modelos logra un MAPE de 10,45 % a 24 horas, frente a 15,81 % de la persistencia: un 27 % menos de error. La prueba de Diebold-Mariano confirma que la diferencia es significativa, con t de 7,76. En la gráfica se ve el pronóstico siguiendo al precio real en 2026, un año que nunca se usó para entrenar.

### 11. OE2: 72 h y robustez — Juan (0:45)

A 72 horas el error crece con la distancia, como es de esperar, pero el ensamble le gana a la persistencia en los tres tramos, con un MAPE global de 16,85 % frente a 21,86 %. Las bandas de incertidumbre cubren alrededor del 78 % de las horas, cerca del 80 % objetivo. Un hallazgo honesto: en los años históricos ningún modelo individual le gana siempre a la persistencia; por eso usamos un ensamble. Ahora Rafael presenta el motor de decisión.

### 12. OE3: motor de decisión — Rafael (1:00)

El motor toma la mediana del pronóstico de cada hora y la compara con dos umbrales: si está por debajo del bajo, el comercializador recibe comprar y el generador retener; si está por encima del alto, evitar compra y vender; si no, esperar. Tenemos cuatro métodos que se diferencian en la referencia: fijo usa el histórico 2019-2025, rodante los últimos 30 días, banda agrega un filtro que espera cuando el pronóstico es muy incierto, e híbrido combina el rodante con ese filtro. La gráfica muestra el híbrido operando una semana de julio: vende en las horas caras, retiene en las baratas y espera cuando la banda es ancha.

### 13. OE3–OE4: selección y resultados — Rafael (0:55)

Al principio elegíamos el método con mejor ventaja promedio del año, y ganaba banda. Pero al partir 2026 en dos mitades vimos que banda casi no actuaba en la primera, solo el 1 % de las horas, porque el precio de 2026 se alejó del histórico. Por eso cambiamos el criterio: ahora gana el método que mejor se sostiene en su peor mitad. Con ese criterio el motor elige híbrido para el generador y rodante para el comercializador, con ventaja positiva en las dos mitades. Para un cliente de ejemplo de 100 kilovatios, en el último mes eso equivale a más 1,06 millones de pesos con híbrido, frente a menos 0,65 millones con banda. Es ilustrativo: no modela contratos ni costos de transacción.

### 14. OE3: dashboard — Rafael (0:35)

El dashboard tiene dos vistas. La vista Operador responde qué hago ahora: la acción para la hora elegida, dónde cae el precio entre los umbrales y el nivel de confianza. La vista Analista deja cambiar el método, los percentiles y el rango de fechas. Ambas se verificaron con pruebas automatizadas en las cuatro combinaciones de rol y horizonte. Juan cuenta las dificultades.

### 15. Dificultades y soluciones — Juan (0:40)

Ninguna dificultad frenó el proyecto, pero varias cambiaron el enfoque. Unir series de distinta resolución sin usar datos futuros exigió reglas explícitas de rezago. Los modelos individuales no superaban a la persistencia en años históricos, y la respuesta fue el ensamble. La demanda del 4 y 5 de agosto llegó mal desde la fuente; lo detectamos al reejecutar todo y reentrenamos. Y en el motor, el mejor método en promedio no era estable, lo que llevó al criterio de estabilidad.

### 16. Cronograma — Juan (0:30)

En el cronograma, a la fecha de corte llevamos 77 % de avance frente a 87 % planeado. Las fases 0 y 1 están completas; el atraso se concentra en la fase 4, imágenes de decisión y validación con usuarios, porque la corrección de datos y los informes consumieron ese tiempo.

### 17. Porcentaje de objetivos — Rafael (0:20)

Por objetivos: OE1 al 100 %, OE2 al 90 %, OE3 al 70 % y OE4 al 35 %, para un avance general de 74 %.

### 18. Próximos pasos — Rafael (0:25)

En las semanas 11 y 12 validamos el dashboard con 3 a 5 usuarios y conectamos el motor al ensamble de seis modelos. En las 13 y 14, la biblioteca de imágenes de decisión y el backtesting histórico del motor. Y en las 15 y 16, el manual del motor, las pruebas de reproducibilidad y los entregables finales.

### 19. Referencias — Rafael (0:05)

Estas son las referencias principales; el listado completo está en el informe.

### 20. Gracias / ¿Preguntas? — Rafael (0:05)

Muchas gracias. Quedamos atentos a sus preguntas.
