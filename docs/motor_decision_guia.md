# Guía del motor de decisión y el dashboard (OE3)

Esta guía es para quien vaya a **usar, mantener o extender** el motor de decisión — no repite el
detalle metodológico que ya está en el README (eso vive en la bitácora del 2026-09-10 y en la nota
para Rafael). Aquí está: qué archivos existen, cómo se conectan, cómo correrlos, y sobre todo **cómo
cambiar qué modelo alimenta cada horizonte sin tocar código**.

## 1. Arquitectura, en un diagrama de texto

```
data/processed/resultados/fuentes_pronostico.json   <- el ÚNICO lugar que dice qué modelo/archivo
        │                                               alimenta cada horizonte (24h, 72h)
        │
        ▼
src/motor_decision.py                                <- lógica del motor (agnóstica al modelo):
    cargar_fuente_pronostico(raiz, horizonte)             - lee el JSON y carga el CSV correspondiente
    validar_contrato_pronostico(df)                       - valida columnas antes de usar un CSV
    generar_senales(df, metodo, rol, ...)                 - aplica 1 de los 3 métodos de umbral
    evaluar_backtest(df, senal, rol)                      - ventaja económica de una señal
    comparar_metodos(df, rol, ...)                        - corre los 3 métodos y compara
    elegir_mejor_metodo(tabla)                             - elige el ganador
        │                              │
        ▼                              ▼
notebooks/12_motor_decision_Rafa.ipynb   dashboard/app.py
  (genera bandas de 72h si hace falta,     (Streamlit + Plotly, lee la misma fuente,
   corre el backtest, guarda señales        recalcula la señal en vivo según los
   finales en un CSV)                       controles que elija el usuario)
```

**La pieza clave es `fuentes_pronostico.json`.** Ni el notebook ni el dashboard tienen una ruta de
archivo de pronóstico escrita a mano — los dos llaman a `cargar_fuente_pronostico(raiz, horizonte)`,
que lee ese JSON. Cambiar el modelo de un horizonte es editar el JSON, no el código.

Estado actual del JSON:

| Horizonte | Modelo | Calibrado | Cobertura medida |
|---|---|---|---|
| 24h | N-BEATSx + calibración conforme adaptativa | Sí | 78.0% |
| 72h | N-BEATSx (h=72, MQLoss) + calibración conforme adaptativa por tramo | Sí | 78.1% |

> El archivo de 72h (`pronostico_con_bandas_72h_2026_adaptativo.csv`) trae columnas extra
> (`cutoff`, `paso_horas`, `margen`) porque es la salida cruda del pipeline multi-horizonte de
> N-BEATSx. `cargar_fuente_pronostico` se queda solo con las 5 del contrato — no hay que pre-procesar
> nada. La serie ya es horaria y continua (ventanas de 72h no solapadas, ene–ago 2026).

## 2. Contrato de datos

Cualquier archivo que se registre en `fuentes_pronostico.json` debe ser un CSV con exactamente estas
columnas (una fila por hora):

| Columna | Significado |
|---|---|
| `fecha_hora` | timestamp de la hora pronosticada |
| `real` | precio de bolsa real observado en esa hora (para poder hacer backtest) |
| `q10` | cuantil 10% del pronóstico |
| `q50` | cuantil 50% / mediana — es el valor que el motor usa como "el precio esperado" |
| `q90` | cuantil 90% del pronóstico |

`validar_contrato_pronostico(df)` (en `src/motor_decision.py`) revisa esto automáticamente cada vez
que se carga un archivo vía `cargar_fuente_pronostico()`:

- **Faltan columnas o hay NaN en q10/q50/q90** → error duro, no se puede usar el archivo.
- **Cuantiles cruzados** (`q10 > q50` o `q50 > q90` en alguna fila) → solo advertencia impresa en
  consola, no bloquea nada. En la práctica aparece como un puñado de filas borde de calibraciones
  adaptativas (2/5208 en las bandas de 24h; las de 72h de N-BEATSx están limpias) — vale la pena
  revisar si ese número crece mucho en un modelo nuevo, pero no justifica parar el pipeline por un
  0.04% de las horas.

## 3. Cómo correr el notebook

```bash
jupyter notebook notebooks/12_motor_decision_Rafa.ipynb
# o: jupyter nbconvert --to notebook --execute --inplace notebooks/12_motor_decision_Rafa.ipynb
```

Qué hace, en orden:
1. Carga las bandas de 24h y 72h vía `cargar_fuente_pronostico(RAIZ, horizonte)`.
2. Corre `comparar_metodos()` para los 3 métodos de umbral, por cada combinación (horizonte, rol).
3. Elige el método ganador de cada combinación con `elegir_mejor_metodo()`.
4. Genera las señales finales y las guarda en
   `data/processed/resultados/senales_motor_decision_2026.csv`.
5. Grafica un ejemplo (últimas 2 semanas, 24h, ambos roles).

Corre en segundos: no entrena nada, solo lee los dos CSV de bandas y aplica los métodos de umbral.
(La primera versión del notebook entrenaba XGBoost por cuantiles para generar unas bandas de 72h
provisionales; eso se eliminó cuando llegó el pipeline calibrado de N-BEATSx a 72h — ver sección 5.)

## 4. Cómo correr el dashboard

```bash
pip install -r requirements.txt   # incluye streamlit y plotly
streamlit run dashboard/app.py
```

Se abre en el navegador (`http://localhost:8501` por defecto). Controles del panel lateral:

- **Rol**: generador o comercializador — cambia qué acción se resalta y cómo se lee la ventaja económica.
- **Horizonte**: 24h o 72h — cambia qué archivo de `fuentes_pronostico.json` se usa. El dashboard
  muestra si el modelo activo está calibrado (verde) o no (amarillo, con su cobertura medida).
- **Método de umbral**: el dashboard sugiere automáticamente el método ganador del backtest
  económico para el rol/horizonte elegidos (recalculado en vivo, no leído de un CSV viejo) — se
  puede cambiar manualmente para comparar. Hay un expander con la tabla comparativa completa.
- **Percentiles bajo/alto** y **rango de fechas**: para explorar sensibilidad sin editar código.

La gráfica principal muestra precio real, banda [q10,q90], y las horas donde la señal dice actuar
(verde = comprar/vender, rojo = evitar\_compra/retener). Los KPIs arriba son la ventaja económica,
frecuencia de acción y horas de acción del `evaluar_backtest()` sobre el período filtrado.

**Nota práctica de caché**: el dashboard usa `st.cache_data` para no releer CSVs en cada interacción.
Si actualizas un archivo de datos (ej. regeneras `pronostico_con_bandas_2026_72h.csv`) con el
dashboard ya corriendo, usa el menú ⋮ → "Clear cache" (o reinicia `streamlit run`) para que lo note.

## 5. Cambiar o agregar un modelo (el caso que más va a pasar)

Este flujo ya se ejecutó una vez en la práctica: la v1 del motor nació con un fallback de XGBoost a
72h, y cuando llegó el pipeline calibrado de N-BEATSx a 72h (commit `3d8d685`) se cambió la fuente
siguiendo exactamente estos pasos, sin tocar `motor_decision.py` ni `app.py`. Para el próximo cambio
(ej. un modelo mejor que N-BEATSx en 24h o 72h), es lo mismo:

Supón que aparece un modelo mejor para un horizonte `H` (`"24h"` o `"72h"`):

1. **Generar el pronóstico** con al menos las 5 columnas del contrato (sección 2) sobre el holdout
   2026 en ese horizonte. Guardarlo en `data/processed/resultados/` con un nombre descriptivo. Puede
   traer columnas extra (como el archivo de 72h de N-BEATSx, con `cutoff`/`paso_horas`) — el loader
   se queda solo con las del contrato.

2. **Validarlo antes de registrarlo**, desde una consola de Python en la raíz del proyecto:
   ```python
   import sys; sys.path.insert(0, "src")
   import pandas as pd
   from motor_decision import validar_contrato_pronostico

   df = pd.read_csv("data/processed/resultados/<archivo_nuevo>.csv", parse_dates=["fecha_hora"])
   validar_contrato_pronostico(df, "el archivo nuevo")
   cobertura = ((df["real"] >= df["q10"]) & (df["real"] <= df["q90"])).mean()
   print(f"Cobertura: {cobertura*100:.1f}%")
   ```

3. **Editar `fuentes_pronostico.json`**, reemplazando la entrada de `H`:
   ```json
   "24h": {
     "archivo": "<archivo_nuevo>.csv",
     "modelo": "<descripción corta del modelo + calibración>",
     "calibrado": true,
     "cobertura_objetivo_pct": 80,
     "cobertura_medida_pct": <el número que salió del paso 2>,
     "generado": "<fecha>",
     "notebook_origen": "<qué notebook/script lo generó>",
     "notas": "<qué reemplaza y por qué>"
   }
   ```

4. **Re-correr `notebooks/12_motor_decision_Rafa.ipynb`** — vuelve a correr el backtest con la nueva
   fuente y regenera `senales_motor_decision_2026.csv`.

5. **Abrir el dashboard** (o limpiar su caché si ya estaba corriendo) — automáticamente muestra el
   nuevo modelo como activo para ese horizonte, sin ningún cambio en `app.py`.

**Caso real ya ejecutado (2026-09-10)**: la v1 tenía un fallback de XGBoost a 72h. Cuando el commit
`3d8d685` agregó `pronostico_con_bandas_72h_2026_adaptativo.csv` (N-BEATSx h=72 + CQR adaptativo), el
cambio fueron esos 5 pasos: se editó la entrada `"72h"` del JSON, se re-ejecutó el notebook `12`
(que además se simplificó: se le quitó la sección que entrenaba XGBoost), se borró el CSV del
fallback, y el dashboard pasó a mostrar N-BEATSx en 72h sin tocarlo. El backtest cambió de resultado
(`banda` pasó a ganar también en 72h) porque las bandas calibradas de N-BEATSx crecen con el error
real y las del fallback no.

**Lo que NO hay que tocar**: `src/motor_decision.py` (agnóstico al modelo por diseño) ni la lógica
de `dashboard/app.py` (lee lo que diga el JSON). Si en algún momento sí hace falta tocar alguno de
los dos, probablemente significa que el nuevo archivo no cumple el contrato de la sección 2 — mejor
arreglar el archivo que el código.

## 6. Cómo decide el motor: los 3 métodos de umbral

Todos parten de dos percentiles del precio (por defecto p25/p75) y comparan `q50` contra ellos:

- **`fijo`**: percentiles calculados una sola vez sobre el precio histórico de entrenamiento
  (2019-2025). Simple, reproducible, fácil de explicar.
- **`rodante`**: percentiles con ventana móvil **causal** de 30 días (usa solo información pasada,
  `shift(1)` antes de la ventana para no incluir la propia hora) — se adapta al régimen reciente.
- **`banda`**: igual que `fijo`, pero fuerza la señal a `esperar` cuando el ancho de la banda
  `[q10,q90]` está en el percentil 75 o más alto de anchos históricos — es decir, cuando el
  pronóstico tiene poca confianza, el motor prefiere no actuar en vez de arriesgarse.

`comparar_metodos()` corre los 3 y calcula, para cada uno, la **ventaja económica**
(`evaluar_backtest()`: precio real promedio en horas de acción vs. promedio general del período) y
la **frecuencia de acción**. Se descartan como "no válidos" los métodos cuya frecuencia cae fuera de
10%-40% (una regla que casi nunca dispara, o que dispara case siempre, no es útil aunque su ventaja
promedio se vea bien) y se elige el de mayor ventaja entre los válidos.

## 7. Señales por rol

| Rol | Señales posibles | Qué significa |
|---|---|---|
| `generador` | `vender` | precio alto — vender a bolsa aprovechando el precio |
| | `retener` | precio bajo — no despachar si se puede evitar |
| | `esperar` | precio en rango medio, o poca confianza (método `banda`) |
| `comercializador` | `comprar` | precio bajo — oportunidad de comprar en bolsa |
| | `evitar_compra` | precio alto — usar contratos bilaterales en vez de bolsa |
| | `esperar` | precio en rango medio, o poca confianza |

## 8. Limitaciones conocidas (léelas antes de confiar en un resultado)

- **El backtest no es una simulación de portafolio.** Es "precio promedio en horas de acción vs.
  promedio general" — no modela capacidad de almacenamiento, contratos ya firmados, costos de
  transacción, ni la secuencia de decisiones a lo largo del tiempo.
- **El motor decide solo con el nivel de `q50`**, no con la forma completa del día. La nota para
  Rafael en el README (sección "cómo evaluar modelos para el motor de decisión") documenta que
  Corr-f/MHD/MPD (`scripts_experimento/metricas_decision.py`) correlacionan mucho mejor con la
  ganancia real de arbitraje que el nivel puntual — integrarlas es el siguiente paso natural, no
  está hecho todavía.
- **Cuantiles cruzados**: ver sección 2. Monitorear si crece en modelos nuevos.

## 9. Troubleshooting

- `ModuleNotFoundError: No module named 'motor_decision'` → falta agregar `src/` al `sys.path`
  (todos los notebooks/scripts de este repo ya tienen ese patrón al inicio, cópialo si escribes uno
  nuevo) o no estás corriendo desde la raíz del proyecto.
- `FileNotFoundError` en `cargar_fuente_pronostico` → revisa que
  `data/processed/resultados/fuentes_pronostico.json` exista y que el campo `"archivo"` de la entrada
  apunte a un CSV que sí está en `data/processed/resultados/`.
- Dashboard no arranca / `ModuleNotFoundError: streamlit` o `plotly` → `pip install -r requirements.txt`.
- El dashboard no refleja un archivo de datos que acabas de regenerar → limpiar caché (sección 4).
