# -*- coding: utf-8 -*-
"""
Limpieza y documentacion de los notebooks de Juan (5-nov-2026), sin volver a ejecutarlos: conservan sus resultados.
- Quita de las salidas las advertencias (stderr) y los errores.
- 01 y 02 (bitacoras de descarga): quita los intentos fallidos y las celdas repetidas, y agrega titulos por seccion.
- A cada notebook le agrega un encabezado con su objetivo, sus entradas y salidas y su estado en el proyecto.
El 03 no se toca aqui: lo arma construir_nb03_caracterizacion.py con los datos actuales.
Uso: python scripts_experimento/limpiar_notebooks_juan.py
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"

ETAPA_1 = ("> **Estado.** Primera etapa del proyecto (diseño del informe de avance: ensamble de seis modelos con el precio "
           "del día anterior). Desde el 14 de octubre de 2026 el modelo en producción es el predespacho de XM corregido "
           "con tres CatBoost (`scripts_experimento/escenario_realista_*.py`; informe final, Solución 3). Este notebook "
           "queda como registro de esa etapa: sus cifras no se recalculan y pueden diferir de las del informe final.")


def md(texto):
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip().splitlines(keepends=True)}


def limpiar_salidas(nb):
    """Quita advertencias (stderr) y errores; devuelve cuantas salidas quito."""
    n = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        antes = len(c.get("outputs", []))
        c["outputs"] = [o for o in c.get("outputs", [])
                        if not (o.get("output_type") == "stream" and o.get("name") == "stderr")
                        and o.get("output_type") != "error"]
        n += antes - len(c["outputs"])
    return n


def empieza(c, texto):
    return "".join(c["source"]).lstrip().startswith(texto)


def armar(nb, plan):
    """plan: lista de (prefijo de la celda de codigo, markdown a poner antes o None, quitar la celda?)."""
    celdas = list(nb["cells"])
    nuevas, usadas, previa = [], set(), None
    for c in celdas:
        if c["cell_type"] == "code" and previa is not None and c["source"] == previa["source"]:
            continue                                   # copia identica de la celda de codigo anterior
        regla = next((r for r in plan if c["cell_type"] == "code" and empieza(c, r[0])), None)
        if regla and regla[2]:
            continue
        if regla and regla[1] and regla[0] not in usadas:
            nuevas.append(md(regla[1]))
            usadas.add(regla[0])
        nuevas.append(c)
        if c["cell_type"] == "code":
            previa = c
    nb["cells"] = nuevas


def procesar(nombre, encabezado, plan=(), final=None, reemplazos=()):
    ruta = NB / nombre
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    quitadas = limpiar_salidas(nb)
    antes = len(nb["cells"])
    armar(nb, list(plan))
    for viejo, nuevo in reemplazos:          # markdown existente que se reescribe
        for c in nb["cells"]:
            if c["cell_type"] == "markdown" and "".join(c["source"]).strip() == viejo:
                c["source"] = nuevo.strip().splitlines(keepends=True)
    primera = nb["cells"][0]
    if primera["cell_type"] == "markdown" and "".join(primera["source"]).lstrip().startswith("# "):
        nb["cells"] = [primera, md(encabezado)] + nb["cells"][1:]        # ya tiene titulo: la nota va debajo
    else:
        nb["cells"] = [md(encabezado)] + nb["cells"]
    nb["cells"] += [md(final)] if final else []
    ruta.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{nombre}: {quitadas} salidas de advertencia/error quitadas; celdas {antes} -> {len(nb['cells'])}")


# ------------------------------------------------------------------------------------------------------------ 01
procesar("01_prueba_conexion_juan.ipynb", """
# 01 · Conexión con la API de XM y primera descarga del precio de bolsa

**Objetivo.** Probar el acceso a los datos públicos de XM con la librería `pydataxm`, resolver sus incompatibilidades
con pandas 3 y descargar el precio de bolsa nacional horario de 2019 a 2024.

**Salida.** `data/precio_bolsa_2019_2024.csv` (el notebook 02 lo extendió a 2025 y lo reemplazó por
`data/precio_bolsa_2019_2025.csv`).

> **No volver a ejecutar sin revisar:** las celdas de descarga llaman a la API de XM y escriben en `data/`. Las salidas
> que se ven son las de la ejecución original.
""", plan=[
    ("# Celda — Parche de compatibilidad", "## 1. Parche de compatibilidad pydataxm–pandas 3\n`pydataxm` 0.3.17 usa "
     "funciones que pandas 3 eliminó; estos parches las reemplazan sin modificar la librería instalada.", False),
    ("# Celda 2 — Primera conexión", None, True),                     # metrica inexistente (PrecioEscasez)
    ("# Celda 3 — Ver todos los indicadores", "## 2. Catálogo de métricas de XM\nEl catálogo dice el nombre exacto "
     "(MetricId) y la entidad de cada variable. El precio de bolsa nacional es `PrecBolsNaci`, entidad `Sistema`.", False),
    ("import sys", None, True),                                         # diagnostico del interprete
    ("# ReadDB es la clase principal", None, True),                     # catalogo repetido
    ("# datetime es libreria estandar", None, True),                    # metrica inexistente (PrecBolsNal)
    ("# Celda 4 — Descarga de prueba", "## 3. Primera descarga y el error de `request_data()`\n`request_data()` "
     "desalinea la columna de fechas al aplanar el JSON de XM. Se revisó el dato crudo y se escribió una función propia "
     "que lo aplana bien.", False),
    ("# Celda — Descarga por bloques (PRUEBA", "## 4. Descarga por bloques\nLa API entrega como máximo 31 días por "
     "llamado, así que el periodo se parte en bloques de 30 días, con reintentos si un bloque falla.", False),
    ("# Celda — Mini-prueba: confirmar que existe dato en enero de 2019", "## 5. Descarga completa 2019-2024", False),
    ("# Celda — Buscar en el catalogo las metricas relacionadas con embalses", None, True),   # se repite en el 02
], final="""
## Resultado
Se descargaron 52.608 horas de precio de bolsa (2019-2024) sin horas faltantes ni duplicadas. Lo aprendido aquí (los
parches de compatibilidad y la descarga propia por bloques) lo reutiliza el notebook 02 para las demás variables.
""")

# ------------------------------------------------------------------------------------------------------------ 02
procesar("02_datos_embalses_hidrologia_juan.ipynb", """
# 02 · Adquisición de hidrología, demanda y generación, y dataset maestro

**Objetivo.** Descargar de XM el volumen útil de los embalses y los aportes hídricos (diarios), la demanda real y la
generación total (horarias); extender el precio a 2025 y 2026, y unir todo en el dataset maestro horario.

**Salidas.** `data/*_2019_2025.csv` y `data/*_2026.csv` de cada variable, y `data/processed/dataset_maestro_*.csv`.
El dataset maestro vigente lo reconstruyó después `scripts_experimento/actualizar_datos_oct2026.py` (14-oct-2026), con
las correcciones de datos; este notebook documenta cómo se armó la primera versión.

> **No volver a ejecutar sin revisar:** las celdas de descarga llaman a la API de XM, escriben en `data/` y una de ellas
> borra los archivos 2019-2024 ya reemplazados. Las salidas que se ven son las de la ejecución original.
""", plan=[
    ("# Celda 1 — Configuracion inicial", "## 1. Configuración\nEl notebook corre en su propio kernel, así que repite "
     "los parches y las funciones del notebook 01.", False),
    ("# Celda 4 — Buscar en el catalogo las metricas relacionadas con embalses", "## 2. Embalses y aportes hídricos "
     "(métricas diarias)\nSon métricas `DailyEntities`: un valor por día. Se descargan con una función propia para ese "
     "formato.", False),
    ("# Celda 11 — Descargar SOLO 2025", "## 3. Extensión a 2025\nSe descarga solo el año que faltaba y se une con lo "
     "ya guardado.", False),
    ("# Celda 16 — Buscar en el catalogo las metricas de demanda", "## 4. Demanda real y generación (métricas horarias)",
     False),
    ("# --- Celda de arranque: imports + ubicar la raíz del proyecto ---", "## 5. Dataset maestro 2019-2025\nLa celda "
     "anterior terminó de guardar la generación y luego el kernel se reinició; desde aquí se recargan los archivos. Las "
     "series diarias se repiten en sus 24 horas y se unen a las horarias.", False),
    ("# --- Mismo diagnóstico, para volumen_util_embalses", None, True),    # diagnostico de bytes
    ("# --- Celda nueva: Descarga enero-agosto 2026 de embalses", "## 6. Enero a agosto de 2026\nVentana de prueba del "
     "Anexo 1. Los valores diarios faltantes se interpolan (no se repiten hacia adelante) y el ONI se une con una rampa "
     "causal, que solo usa meses ya pasados.", False),
], final="""
## Resultado
Seis variables horarias de enero de 2019 a agosto de 2026 en un solo dataset (precio, demanda, generación, volumen de
embalses, aportes y ONI), sin horas faltantes. Es la base del notebook 02b (variables de los modelos) y del 03
(caracterización).
""")
# la celda de arranque quedo repetida (dos copias seguidas): se deja una
ruta = NB / "02_datos_embalses_hidrologia_juan.ipynb"
nb = json.loads(ruta.read_text(encoding="utf-8"))
arranque = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
            and empieza(c, "# --- Celda de arranque: imports + ubicar la raíz del proyecto ---")]
if len(arranque) == 2 and arranque[1] == arranque[0] + 1:
    del nb["cells"][arranque[1]]
    ruta.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("02: celda de arranque repetida quitada")

# ------------------------------------------------------------------------------------------------------------ 02b
procesar("02b_features_compartidas_juan.ipynb", """
# 02b · Variables compartidas de los modelos de la primera etapa

**Objetivo.** Construir, a partir del dataset maestro, las variables que usaron los modelos de la primera etapa (04, 06,
07, 08, 09 y 10): rezagos y medias móviles del precio, armónicos de calendario (el periodograma del notebook 03 hecho
variable), hidrología rezagada y sus derivadas, volatilidad, demanda rezagada y festivos. Todas respetan el horizonte:
ninguna usa información posterior al momento del pronóstico.

**Salidas.** `data/processed/dataset_features_2019_2025.csv` y `dataset_features_2026.csv`. Antes se llamaba
`05_features_compartidas_juan.ipynb`; se renombró porque el número 05 lo tenía también el notebook de XGBoost de Rafael.

""" + ETAPA_1)

# ------------------------------------------------------------------------------------------------------------ 04
procesar("04_modelo_prophet_juan.ipynb", """
# 04 · Modelo Prophet

**Objetivo.** Primera familia de modelo: Prophet (descomposición en tendencia y estacionalidades) con regresores de
hidrología y ONI, ajuste de `changepoint_prior_scale` y diagnóstico por régimen de El Niño.

**Resultado.** Fue el modelo más débil en todas las pruebas: no le ganó a la persistencia y su error se multiplica por
cerca de 5 cuando el ONI es alto. Se excluyó del walk-forward (07).

""" + ETAPA_1, reemplazos=[("SECCION FINAL LUEGO DE VERIFICAR ERRORES", "## Sección final: modelo con los datos "
                           "corregidos\nSe reentrena con la misma configuración (cps = 0,005) después de corregir la "
                           "fuga de información que se encontró en XGBoost.")])

# ------------------------------------------------------------------------------------------------------------ 06-10, 13
procesar("06_modelo_xgboost_juan.ipynb", """
# 06 · Modelo XGBoost

**Objetivo.** Segunda familia de modelo: árboles de gradiente con las variables del 02b. Incluye el ajuste de
hiperparámetros, regresión por cuantiles, interpretabilidad (SHAP, importancia por permutación y dependencia parcial)
y el horizonte de 72 h.

**Dónde se usa.** La interpretabilidad (SHAP y permutación coinciden en las tres variables principales) está en el
informe final, criterio de OE2.

""" + ETAPA_1)
procesar("07_validacion_walkforward_juan.ipynb", ETAPA_1)
procesar("08_modelo_arima_garch_juan.ipynb", ETAPA_1)
procesar("09_modelos_deep_learning_juan.ipynb", ETAPA_1)
procesar("10_diebold_mariano_juan.ipynb", ETAPA_1)
procesar("13_opciones_diseno_dashboard_juan.ipynb", """
> **Estado.** Registro de la comparación de diseños del 6-oct-2026. El diseño vigente (marca Augur en azul marino y
> ámbar, con tema oscuro Pizarra) está en `dashboard/app.py` y evolucionó después de esta comparación.
""")
