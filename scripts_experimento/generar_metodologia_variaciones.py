# -*- coding: utf-8 -*-
"""
Documento para el paper (pedido de Juan, 9-nov-2026): metodologia final de Augur y todas las variaciones que se probaron,
con su resultado, la decision y el script que la reproduce. Salida: docs/Metodologia_y_variaciones_Augur.docx (y .pdf con
Word). Las cifras de las tablas salen de los CSV de resultados; las demas, de la bitacora del README y de
resultados/escenario_realista/mejoras/RESUMEN.md (se indica el script en cada fila).
Formato: tablas sin relleno de color (pedido de Juan).
Uso: python scripts_experimento/generar_metodologia_variaciones.py
"""
import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "data/processed/resultados"
ER = RES / "escenario_realista"
MEJ = ER / "mejoras"
DESTINO = RAIZ / "docs" / "Metodologia_y_variaciones_Augur.docx"
TINTA, GRIS = RGBColor(0x0B, 0x1F, 0x3A), RGBColor(0x55, 0x55, 0x55)


def n(v, d=1):
    s = f"{abs(v):,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("−" if v < 0 else "") + s


doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.59), Cm(27.94)
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin = sec.bottom_margin = Cm(2.0)
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after, st.paragraph_format.line_spacing = Pt(5), 1.1
for nivel, tam in ((1, 15), (2, 12)):
    h = doc.styles[f"Heading {nivel}"]
    h.font.name, h.font.size, h.font.bold, h.font.color.rgb = "Calibri", Pt(tam), True, TINTA
    rf = h.element.rPr.find(qn("w:rFonts"))
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf is not None and rf.get(qn(a)):
            del rf.attrib[qn(a)]
    h.paragraph_format.space_before, h.paragraph_format.keep_with_next = Pt(12 if nivel == 1 else 9), True
ANCHO = sec.page_width - sec.left_margin - sec.right_margin


def runs(par, texto, tam=None, color=None):
    for parte in re.split(r"(\*\*.+?\*\*)", texto):
        if not parte:
            continue
        r = par.add_run(parte[2:-2] if parte.startswith("**") else parte)
        r.bold = parte.startswith("**")
        if tam:
            r.font.size = Pt(tam)
        if color:
            r.font.color.rgb = color
    return par


def P(texto):
    p = runs(doc.add_paragraph(), texto)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def V(texto):
    p = runs(doc.add_paragraph(style="List Bullet"), texto)
    p.paragraph_format.space_after = Pt(2)
    return p


def tabla(enc, filas, anchos, titulo, fuente=None, tam=8.5):
    p = runs(doc.add_paragraph(), f"**{titulo}**", tam=9.5)
    p.paragraph_format.keep_with_next, p.paragraph_format.space_after = True, Pt(2)
    t = doc.add_table(rows=1, cols=len(enc))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    b = OxmlElement("w:tblBorders")
    for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{lado}")
        e.set(qn("w:val"), "single"), e.set(qn("w:sz"), "4"), e.set(qn("w:color"), "8C8C8C")
        b.append(e)
    t._tbl.tblPr.append(b)
    for j, e in enumerate(enc):
        runs(t.rows[0].cells[j].paragraphs[0], f"**{e}**", tam=tam)
    for f in filas:
        cs = t.add_row().cells
        for j, v in enumerate(f):
            runs(cs[j].paragraphs[0], str(v), tam=tam)
    tot = sum(anchos)
    for fila in t.rows:
        for j, c in enumerate(fila.cells):
            c.width = int(ANCHO * anchos[j] / tot)
            for q in c.paragraphs:
                q.paragraph_format.space_after, q.paragraph_format.line_spacing = Pt(1), 1.0
    tr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); tr.append(th)
    if fuente:
        runs(doc.add_paragraph(), fuente, tam=8, color=GRIS)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(0)


def vivo():
    """Todo 2026 hasta el ultimo precio publicado, con el mismo calculo de generar_documento_funcionamiento.vivo()."""
    import numpy as np
    import statsmodels.api as sm
    c = pd.read_csv(RES / "pronostico_realista_bandas_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    c = c.dropna(subset=["real"])
    c = c[c.index <= pd.Timestamp("2026-10-31 23:00")]      # corte fijo, el mismo del informe final (Juan, 9-nov-2026)
    c["pre"] = pd.read_csv(RES / "ensamble_24h_realista_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")["Predespacho"]
    precio = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, usecols=["fecha_hora", "precio_bolsa"], parse_dates=["fecha_hora"])
                        for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]
                       + [pd.read_csv(RAIZ / "data/produccion/serie_xm.csv", usecols=["fecha_hora", "precio_bolsa"], parse_dates=["fecha_hora"])]
                       ).drop_duplicates("fecha_hora", keep="last").set_index("fecha_hora").sort_index()["precio_bolsa"]
    c["pers"] = precio.reindex(c.index - pd.to_timedelta(np.where(c.index.hour == 0, 120, 96), unit="h")).to_numpy()
    e = c["real"] - c["q50"]
    d = (c["pre"] - c["real"]).abs() - e.abs()
    t = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 23}).tvalues[0]
    return {"hasta": c.index.max(), "horas": len(c), "MAE": e.abs().mean(), "MAE_pre": (c["pre"] - c["real"]).abs().mean(),
            "MAE_pers": (c["pers"] - c["real"]).abs().mean(), "t": float(t)}


VIVO = vivo()
MESES = {1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
         9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre"}

# ================================================================================================== portada breve
p = doc.add_paragraph(); r = p.add_run("Augur: metodología y variaciones probadas"); r.bold = True; r.font.size = Pt(20)
r.font.color.rgb = TINTA
runs(doc.add_paragraph(), "Insumo para el artículo · Juan David Barceló Barraza y Rafael Andrés Dede Perdomo · Ingeniería "
                          "Electrónica, Universidad del Norte · 9 de noviembre de 2026", tam=10, color=GRIS)
P("Este documento reúne, en un solo lugar, la metodología final de Augur y **todas las variaciones que se probaron** "
  "para llegar a ella: qué se cambió, cuánto cambió el resultado, si se adoptó y qué script lo reproduce en el "
  "repositorio (github.com/Ziskito/xm-spot-price-predictor). Las cifras están en COP/kWh salvo que se diga otra cosa. "
  "MAE es el error absoluto medio; DM, el estadístico de Diebold-Mariano con varianza HAC (|t| > 1,96 equivale a "
  "p < 0,05).")

# ================================================================================================== 1 metodologia
doc.add_heading("1. Metodología final", 1)
doc.add_heading("1.1 Información disponible al corte", 2)
P("**Notación.** D es el día que se pronostica; D-1 es el día anterior (la víspera), D-4 cuatro días antes, y D+1 y "
  "D+2 los dos días siguientes. Como el precio de bolsa se liquida por hora y la última hora de cada día es las 00:00 "
  "del día siguiente, la «ventana» del día D va de la 01:00 de D a las 00:00 de D+1.")
P("El pronóstico se emite a las 00:00 del día D (el «corte») para la ventana de 01:00 de D a 00:00 de D+1, y solo usa lo publicado "
  "a esa hora (escenario realista): precio, demanda y generación hasta las 23:00 de D-4 (la API de XM publica con unos "
  "tres días de rezago); embalses y aportes de D-2; ONI de NOAA de dos meses antes; el predespacho ideal de D (iMAR, "
  "publicado D-1 hacia las 10:30) y el despacho programado de D (dMAR y dDEM, D-1 hacia las 14:30). Se auditó la hora "
  "de publicación de cada fuente; las ofertas de los generadores (OFEI) se descartaron porque XM agrega los precios el "
  "mes siguiente. Datos: enero de 2019 a agosto de 2026 (66.576 horas) para entrenar y validar, y la operación diaria "
  "desde el 6 de agosto.")
doc.add_heading("1.2 Pronóstico de 24 horas", 2)
V("**Objetivo:** log(precio / predespacho). El modelo no pronostica el precio desde cero: corrige el predespacho de XM.")
V("**Tres CatBoost** con pérdida MAE (1.500 árboles, profundidad 6, tasa 0,05), con conjuntos de variables distintos: "
  "R+P (realistas + predespacho, 49 variables), R+P+F+C (+ forma intradía del predespacho + despacho programado, 61) y "
  "R+P+E (+ error reciente del predespacho, 54). Se reentrenan cada mes con lo publicado hasta 73 h antes del corte y "
  "se promedian.")
V("**Combinador causal (QRA de la mediana)** entre el predespacho y el promedio de los CatBoost, por franjas de 6 h, "
  "reestimado cada día solo con días cuyo precio ya está publicado (margen de 5 días).")
V("**Bandas conformes del 80 %:** q50·exp(±m), con m el cuantil conforme de |ln(real/q50)| de la misma hora en los 60 "
  "días anteriores con precio publicado.")
V("**Detector de picos y desplomes** (HistGradientBoosting, reentrenado cada semana): probabilidad de que el precio "
  "quede ≥ 1,3 o ≤ 0,7 veces la mediana de la ventana.")
V("**72 h:** los pasos 1-24 son el pronóstico de 24 h. Para los pasos 25-72 (D+1 y D+2) todavía no hay predespacho, así "
  "que la referencia es el predespacho de D, con tres CatBoost por día de anticipación (los mismos tres conjuntos de "
  "variables), el mismo combinador y las mismas bandas. Las mejoras del pronóstico de 24 h se aplican también aquí, "
  "porque comparten los datos (incluido el arreglo de la demanda incompleta) y el modelo de los pasos 1-24; las del "
  "motor (ventana elegible y filtro de contrato) se validaron también con el contrato de 72 h.")
doc.add_heading("1.3 Motor de decisión", 2)
V("Umbrales barato y caro = percentiles 25 y 75 del pronóstico en los 30 días anteriores (método «rodante»; el usuario "
  "puede elegir de 7 a 365 días). Otros métodos: fijo (histórico 2019-2025), banda (fijo + «esperar» con banda ancha) "
  "e híbrido (rodante + banda).")
V("Filtro intradía: la señal favorable solo en la mitad del día que el pronóstico pone a favor del rol.")
V("Selección del método por estabilidad: actuar entre el 10 % y el 40 % de las horas en las dos mitades del periodo y "
  "ganar en la peor mitad (maximin).")
V("Filtro de precio de contrato (desde el 9-nov): comprar en bolsa solo por debajo del contrato y vender solo por "
  "encima; por defecto, el MC de XM (costo promedio de los contratos del mercado regulado).")
doc.add_heading("1.4 Protocolo de validación y de adopción", 2)
P("Prueba fuera de muestra en 2026 (15 de enero a 5 de agosto en las comparaciones de variantes; 1 de enero a 5 de "
  "agosto en el contrato publicado), walk-forward en cinco años históricos (15 de julio a 30 de junio, 2020-21 a "
  "2024-25) y la operación en vivo. Una variación solo se adopta si mejora con significancia (DM) y no empeora en los "
  "años históricos; si se eligió mirando 2026, se valida además en 2020-2025. Las referencias (persistencia del último "
  "precio publicado y predespacho) se evalúan en las mismas horas y con la misma información.")

w = pd.read_csv(ER / "walkforward/metricas.csv")
per = ["Origen 1 (2020-21)", "Origen 2 (2021-22)", "Origen 3 (2022-23)", "Origen 4 (2023-24)", "Origen 5 (2024-25)", "2026"]
cab = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2026"]
pw = w.pivot(index="modelo", columns="periodo", values="MAE")[per]
nom = {"Predespacho": "Predespacho de XM solo", "Pre + CatBoost": "Predespacho + CatBoost (Augur)"}
tabla(["Modelo"] + cab, [[nom.get(k, k)] + [n(pw.loc[k, c]) for c in per]
                         for k in ["Persistencia", "XGBoost", "ARX+GARCH", "GARCH-ged", "N-BEATSx", "N-HiTS", "Predespacho",
                                   "Pre + CatBoost"]],
      [4] + [1.15] * 6, "Tabla 1. Resultado principal: MAE con información realista, walk-forward (5 años) y 2026",
      "2026 = 15 de enero a 5 de agosto. Fuente: walkforward/metricas.csv (realista_walkforward_individuales.py). Con todo 2026 y "
      f"la operación hasta el {VIVO['hasta'].day} de {MESES[VIVO['hasta'].month]} ({n(VIVO['horas'], 0)} h, pronóstico publicado): "
      f"Augur {n(VIVO['MAE'])}, predespacho {n(VIVO['MAE_pre'])}, persistencia {n(VIVO['MAE_pers'])} (DM frente al predespacho "
      f"t = {n(VIVO['t'])}).")

# ================================================================================================== 2 variaciones
doc.add_heading("2. Variaciones probadas", 1)
P("Cada tabla agrupa las variaciones de una parte del sistema. «Adoptada» significa que está en la versión publicada.")

doc.add_heading("2.1 Familias de modelos", 2)
tabla(["Variación", "Resultado", "Decisión", "Script"], [
    ["Prophet (fase inicial, con el precio de ayer)", "MAE 92-96 en 2026, el peor modelo", "Descartada", "notebooks 04"],
    ["XGBoost, ARX+GARCH, GARCH-ged (LEAR), N-BEATSx, N-HiTS desde cero", "Sin el precio de ayer, ninguno supera de forma "
     "consistente a la persistencia (Tabla 1)", "Descartadas como modelo principal", "realista_walkforward_individuales.py"],
    ["Ensamble de 6 votantes (diseño del informe de avance)", "Optimista 42,9; realista apenas mejora a la persistencia "
     "(rMAE 0,93)", "Reemplazado", "ensamble_24h_enero.py"],
    ["Ensamble de 7 con el predespacho como votante", "Optimista 31,5 (41,8 con 6 votantes); realista 36,57", "Reemplazado por el "
     "corrector", "ensamble_con_predespacho_24h.py"],
    ["Corregir el predespacho con árboles (CatBoost, LightGBM, XGBoost, Random Forest) o lineal (Lasso)", "Los cuatro de "
     "árboles empatan entre sí; Lasso es peor en 3 de 6 periodos; CatBoost tiene la mejor media", "CatBoost adoptado",
     "realista_benchmark_modelos.py"],
    ["N-BEATSx / N-HiTS con el predespacho como variable futura", "117 → 58, lejos del CatBoost (35)", "Descartada",
     "realista_exp_neuronales.py"],
    ["LEAR realista por hora", "40,8-42,4", "Descartada", "realista_exp_lineal.py"],
    ["Markov-switching (2 regímenes), modelo de dos etapas (hurdle), coeficientes recursivos (RecursiveLS)", "Sin ganancia "
     "útil frente al ensamble (fase optimista, sep-2026)", "Descartadas", "markov_switching.py, hurdle_dos_etapas.py, recursive_ls.py"],
], [5, 5, 2.4, 3.2], "Tabla 2. Familias de modelos")

b = pd.read_csv(ER / "benchmark/metricas.csv")
pb = b.pivot(index="modelo", columns="periodo", values="MAE")[per]
tabla(["Corrector del predespacho"] + cab + ["Media"],
      [[k] + [n(pb.loc[k, c]) for c in per] + [n(pb.loc[k].mean(), 2)]
       for k in ["Predespacho", "Pre + CatBoost", "Pre + LightGBM", "Pre + XGBoost", "Pre + RandomForest", "Pre + Lasso"]],
      [3.6] + [1.05] * 7, "Tabla 3. Benchmark de correctores del predespacho (MAE, mismas variables y presupuesto)",
      "Fuente: benchmark/metricas.csv (realista_benchmark_modelos.py).")

doc.add_heading("2.2 Variantes del CatBoost y del entrenamiento", 2)
tabla(["Variación", "MAE 2026", "Decisión"], [
    ["Predespacho + CatBoost mensual, combinador MAE (versión simple)", "35,06 (margen 5)", "Base"],
    ["Promedio de 3 CatBoost con variables distintas (R+P, R+P+F+C, R+P+E)", "34,71 (p = 0,002); 2020-25: mejora en 3 de 5 "
     "años y empata en 2", "Adoptada"],
    ["3 semillas / reentreno semanal / con despacho programado por tipo de recurso", "34,80 / 35,00 / 35,04 (meseta)",
     "No (sin ganancia; el despacho por tipo choca con el alcance)"],
    ["Pérdida RMSE en vez de MAE", "45,8", "No"],
    ["CatBoost fijo hasta dic-2025 (sin reentrenar)", "37,5", "No: reentrenar es clave"],
    ["Ponderar por antigüedad / profundidad 8 / historia desde 2023", "39,9-41,2 / 39,6 / 38,2", "No"],
    ["Candidato afinado (desde 2021, 3.000 árboles, semanal)", "34,90", "No (elegido mirando 2026; meseta)"],
], [8, 4.2, 4], "Tabla 4. Variantes del CatBoost (15-ene a 5-ago-2026)",
    "Fuente: mejoras/RESUMEN.md, mezcla_metricas.csv, variantes_cat_metricas.csv, hiper_metricas.csv, "
    "realista_validacion_historica_v2.py.")

doc.add_heading("2.3 Variables de entrada", 2)
c = pd.read_csv(MEJ / "caracterizacion_metricas.csv").set_index("variante")
tabla(["Variación", "Resultado", "Decisión"], [
    ["Forma intradía del predespacho (F)", "35,08 sola; aporta dentro del promedio de 3", "Adoptada en R+P+F+C"],
    ["Costo marginal y demanda del despacho programado, dMAR/dDEM (C)", "34,75 sola (no significativo); mejor CatBoost "
     "individual en los 5 años", "Adoptada en R+P+F+C"],
    ["Error reciente del predespacho (E)", "34,88 (p = 0,13)", "Adoptada en R+P+E"],
    ["Delta ex post y precio híbrido como variables", "38,9-40,4 (empeoran)", "No"],
    ["MPO y delta del propio predespacho", "MPO ≈ costo marginal, delta ≈ 0", "No"],
    ["Disponibilidad declarada (OFEI) como margen de reserva", "35,00 → 35,01 (p = 0,96)", "No"],
    ["Precios de oferta (OFEI) y pronóstico oficial de demanda", "Se publican o sobrescriben después del corte (fuga)", "No"],
    [f"Anomalía estacional de embalses (A)", f"{n(c.loc['R+P', 'MAE'], 2)} → {n(c.loc['R+P+A', 'MAE'], 2)} "
     f"(p = {n(c.loc['R+P+A', 'DM_p'], 2)})", "No"],
    ["Aportes acumulados de 180 días (B)", f"→ {n(c.loc['R+P+B', 'MAE'], 2)} (p = {n(c.loc['R+P+B', 'DM_p'], 2)})", "No"],
    ["ONI con 3 y 8 meses más de rezago (C)", f"→ {n(c.loc['R+P+C', 'MAE'], 2)} (p = {n(c.loc['R+P+C', 'DM_p'], 2)})", "No"],
    ["Ablación por grupos (3 semillas)", "Solo el precio pasado (+125,8) y el calendario (+2,9) aportan; ONI, hidrología, "
     "demanda y festivos quedan dentro del ruido", "ENSO e hidrología se usan para leer la confianza, no como predictores"],
    ["Demanda incompleta del último día publicado (XM publica ~26 %)", "CatBoost 36,76 → 37,97 (p < 0,001); el arreglo "
     "(completar con la generación) lo devuelve a 36,76", "Arreglo adoptado (5-nov)"],
], [6.5, 6.2, 3.8], "Tabla 5. Variables (CatBoost R+P, 15-ene a 5-ago-2026)",
    "Fuente: forma_metricas.csv, programado_metricas.csv, errorpre_metricas.csv, disponibilidad_metricas.csv, "
    "caracterizacion_metricas.csv, ablacion_logo_multisemilla.csv, demanda_parcial_metricas.csv.")

doc.add_heading("2.4 Combinación y corrección del pronóstico", 2)
k = pd.read_csv(MEJ / "kalman_metricas.csv")
kal = {r.variante.split(" q=")[0].replace("Kalman ", ""): r.MAE for r in k.itertuples() if "elegido" in r.variante}
tabla(["Variación", "Resultado", "Decisión"], [
    ["Combinador QRA con objetivo MAE, franjas de 6 h, diario", "Referencia (35,02 con un CatBoost)", "Adoptado"],
    ["Objetivo sMAPE en vez de MAE", "36,47 frente a 35,76 (7 votantes)", "No"],
    ["Hora a hora / franjas de 3 h / franjas de 12 h / un grupo", "37,6 / 37,0 / 35,9 / 36,0", "No"],
    ["Olvido exponencial de los días viejos (vida media 30, 60, 120 días)", "36,3-36,4", "No"],
    ["Promedio, mediana o media recortada de los votantes", "62-87", "No"],
    [f"Combinador de Kalman (pesos que se corrigen hora a hora)", f"Nivel {n(kal['nivel'])}, log {n(kal['log'])}, "
     f"robusto {n(kal['robusto'])} frente a 35,0 (DM p < 0,001)", "No"],
    ["Corrección de sesgo con errores ya publicados (global, EWMA, ventanas de 7-30 días)", "Ninguna mejora; las cortas "
     "empeoran (persiguen ruido)", "No"],
    ["Adaptaciones a la CREG 101 127 (sep-2026): más peso a lo reciente; curva de referencia de ofertas hidráulicas",
     "La primera empeora; la segunda mejora en 2026 pero gana en 2 de 5 años históricos y pierde en 2024-25", "No"],
], [6.5, 6.2, 3.8], "Tabla 6. Combinación y corrección",
    "Fuente: combinador_metricas.csv, kalman_metricas.csv (realista_exp_kalman.py), bitácora del README.")

doc.add_heading("2.5 Motor de decisión", 2)
u = pd.read_csv(MEJ / "motor_umbrales_alternativos.csv")
est = u.groupby(["variante", "rol"])["estable"].sum().unstack()
eco = u.pivot_table(index=["variante", "rol"], columns="periodo", values="vs_MC_por_hora")
tabla(["Variación", "Resultado", "Decisión"], [
    ["Métodos fijo, rodante, banda e híbrido", "El criterio de estabilidad elige «rodante» en los cuatro casos de 2026",
     "Rodante"],
    ["Filtro intradía", "Mejora la peor mitad en los cuatro casos de 2026 (por ejemplo, generador a 72 h: de 105 a 133)", "Adoptado"],
    ["Percentiles 10/90 a 35/65", "Sin óptimo interior; 25/75 es el punto intermedio", "25/75"],
    ["Ventana rodante de 7, 14, 30, 60 y 90 días (5 años, 2026 y operación)", f"30 días estable en "
     f"{int(est.loc['R30 (vigente)', 'generador'])}/7 y {int(est.loc['R30 (vigente)', 'comercializador'])}/7 periodos; "
     f"60 y 90 en 3-4/7", "30 días (elegible en el dashboard)"],
    ["Combinar ventanas cortas y largas (7+90, 14+60, 30+365)", "Estables en 2-6 de 7 periodos", "No"],
    ["Ventana cortada en el último cambio de nivel", "Casi nunca se acorta; empata o empeora", "No"],
    ["Umbral por tipo de día, por hora, filtro semanal", "Sin ganancia frente a un umbral más exigente con la misma "
     "frecuencia", "No"],
    ["Histéresis y lógica difusa", "Empeoran o empatan; la señal ya forma bloques", "No"],
    ["Filtro de precio de contrato (MC de XM)", f"Nunca empeora la ganancia frente al contrato; mejora en 8 de 14 casos "
     f"(DM p < 0,01). Comercializador en la operación: {n(eco.loc[('R30 (vigente)', 'comercializador'), 'operación ago-nov'])} "
     f"→ {n(eco.loc[('R30+MC', 'comercializador'), 'operación ago-nov'])} COP/kWh por hora", "Adoptado (9-nov)"],
    ["Ventana y filtro de contrato con el contrato de 72 h (2026 y operación hasta el 31-oct)", "30 días estable en los dos "
     "periodos y los dos roles (60 y 90, no); el filtro nunca empeora frente al contrato (comercializador en la operación: "
     "−46,8 → 0,0 COP/kWh por hora, p < 0,001)", "Igual que a 24 h"],
], [6.2, 6.8, 3.5], "Tabla 7. Motor de decisión",
    "Fuente: motor_umbrales_alternativos.csv (realista_motor_umbrales_alternativos.py), motor_caracterizacion.csv, "
    "motor_fuzzy_histeresis.csv y la bitácora del README.")

doc.add_heading("2.6 Bandas, eventos y 72 h", 2)
tabla(["Variación", "Resultado", "Decisión"], [
    ["Bandas conformes en escala log, por hora, 60 días", "Cobertura 80,6 % (objetivo 80 %), ancho medio 142", "Adoptadas"],
    ["Bandas empíricas con residuos de 30 días (informe de avance)", "Cobertura 81,0 %, ancho 137, con el ensamble anterior",
     "Reemplazadas"],
    ["Detector de eventos realista (solo lo publicado al corte)", "Desplome AUC 0,94; pico 0,97: igual al optimista",
     "Adoptado"],
    ["Detector de desplome para los días 2 y 3", "No supera a la climatología de forma consistente", "No se publica"],
    ["72 h realista (referencia: predespacho de D)", "Pasos 1-24: 33,2 frente a 38,5; 25-48: 50,2 frente a 59,9; "
     "49-72: 73,2 frente a 75,2 (cadena optimista anterior)", "Adoptado"],
], [6.5, 6.5, 3.5], "Tabla 8. Incertidumbre, eventos y horizonte de 72 h",
    "Fuente: eventos_realista.csv, eventos_72h/metricas.csv, pronostico_realista_72h.py.")

doc.add_heading("3. Lecturas para el artículo", 1)
V("Sin el precio de ayer (información realista), ningún modelo que pronostica desde cero supera de forma consistente a "
  "la persistencia; el predespacho de XM sí, y corregirlo con CatBoost lo mejora en todos los años evaluados.")
V("Lo que más importa es la pérdida L1 (MAE) y reentrenar cada mes; el resto de variantes forman una meseta.")
V("El ENSO y la hidrología no mejoran la precisión a 24-72 h, pero sí indican cuándo desconfiar: con El Niño y embalses "
  "bajos el error se triplica (28,5 a 97,1).")
V("Un mejor MAE no garantiza mejores decisiones: lo que mejoró al motor fue el filtro intradía y, frente al contrato, "
  "el filtro de precio de contrato.")
V("Limitaciones: el motor no es una simulación de portafolio ni de almacenamiento (BESS); la CREG 101 127 cambió las "
  "reglas en septiembre de 2026 y duplicó el error de todos los pronósticos, incluido el del predespacho.")

doc.save(str(DESTINO))
print("ok", DESTINO)
