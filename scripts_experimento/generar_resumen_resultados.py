# -*- coding: utf-8 -*-
"""
Resumen de resultados de Augur para mostrar a los asesores (pedido de Juan, 13-nov-2026): lo que se entrega, el
pronostico de 24 y 72 h, la banda, la operacion en vivo, el motor, capturas del dashboard y la caracterizacion.
Cifras: informe_final_assets/cifras.json y cifras_todo2026.json (las mismas del informe final) y los contratos
publicados. Figuras: las del informe final, las del documento de funcionamiento y las capturas del manual (9-nov).
Formato: como los demas documentos (Calibri, tablas sin relleno de color).
Salida: docs/Resumen_resultados_Augur.docx (y .pdf con Word).
Uso: python scripts_experimento/generar_resumen_resultados.py
"""
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402
from PIL import Image  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
RES = RAIZ / "data/processed/resultados"
INF = AQUI / "informe_final_assets"
FUN = AQUI / "funcionamiento_assets"
CAP = RAIZ / "docs/manual/img"
OUT = AQUI / "resumen_assets"
OUT.mkdir(exist_ok=True)
DESTINO = RAIZ / "docs" / "Resumen_resultados_Augur.docx"
C = json.loads((INF / "cifras.json").read_text(encoding="utf-8"))
T26 = json.loads((INF / "cifras_todo2026.json").read_text(encoding="utf-8"))
CC = json.loads((INF / "cifras_caracterizacion.json").read_text(encoding="utf-8"))
TINTA, GRIS = RGBColor(0x0B, 0x1F, 0x3A), RGBColor(0x55, 0x55, 0x55)


def n(v, d=1):
    s = f"{abs(v):,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("−" if v < 0 else "") + s


# ------------------------------------------------------------------------------------------------ figuras propias
def fig_mensual():
    """Error medio por mes de 2026: Augur (pronostico publicado) frente al predespacho de XM, hasta el 31-oct."""
    c = pd.read_csv(RES / "pronostico_realista_bandas_24h_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")
    c = c.dropna(subset=["real"])
    c = c[c.index <= pd.Timestamp("2026-10-31 23:00")]
    c["pre"] = pd.read_csv(RES / "ensamble_24h_realista_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")["Predespacho"]
    c = c.dropna(subset=["pre"])
    mes = (c.index - pd.Timedelta(hours=1)).to_period("M")
    m = pd.DataFrame({"Augur": (c["q50"] - c["real"]).abs(), "pre": (c["pre"] - c["real"]).abs()}).groupby(mes).mean()
    etq = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct"][:len(m)]
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=170)
    x = np.arange(len(m))
    ax.plot(x, m["pre"], color="#94A3B8", lw=2, marker="o", ms=4, label="Predespacho de XM")
    ax.plot(x, m["Augur"], color="#0B1F3A", lw=2.4, marker="o", ms=4, label="Augur")
    ax.axvspan(6.5, len(m) - 0.5, color="#E2E8F0", alpha=.6, lw=0)
    ax.text(6.6, ax.get_ylim()[1] * 0.97, "operación en vivo", fontsize=8.5, color="#475569", va="top")
    for i, v in enumerate(m["Augur"]):
        ax.text(i, v - 5.5, n(v, 0), ha="center", fontsize=8, color="#0B1F3A")
    ax.set_xticks(x, etq); ax.set_ylabel("Error medio (COP/kWh)", fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "mae_mensual.png"); plt.close(fig)
    return OUT / "mae_mensual.png", m


def recorte(nombre, caja, salida):
    im = Image.open(CAP / nombre)
    im.crop(caja).save(OUT / salida)
    return OUT / salida


# ------------------------------------------------------------------------------------------------ documento
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.59), Cm(27.94)
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin = sec.bottom_margin = Cm(1.8)
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after, st.paragraph_format.line_spacing = Pt(5), 1.1
for nivel, tam in ((1, 14), (2, 12)):
    h = doc.styles[f"Heading {nivel}"]
    h.font.name, h.font.size, h.font.bold, h.font.color.rgb = "Calibri", Pt(tam), True, TINTA
    rf = h.element.rPr.find(qn("w:rFonts"))
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf is not None and rf.get(qn(a)):
            del rf.attrib[qn(a)]
    h.paragraph_format.space_before, h.paragraph_format.keep_with_next = Pt(12 if nivel == 1 else 8), True
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


def H(texto, nivel=1):
    doc.add_heading(texto, level=nivel)


def figura(ruta, ancho_cm, pie):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(ruta), width=Cm(ancho_cm))
    q = runs(doc.add_paragraph(), pie, tam=8.5, color=GRIS); q.alignment = WD_ALIGN_PARAGRAPH.CENTER


def tabla(enc, filas, anchos, titulo=None, nota=None, tam=9):
    if titulo:
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
    for i, fila in enumerate(t.rows):
        trpr = fila._tr.get_or_add_trPr()
        cs_ = OxmlElement("w:cantSplit"); cs_.set(qn("w:val"), "true"); trpr.append(cs_)
        for j, c in enumerate(fila.cells):
            c.width = int(ANCHO * anchos[j] / tot)
            for q in c.paragraphs:
                q.paragraph_format.space_after, q.paragraph_format.line_spacing = Pt(1), 1.0
                q.paragraph_format.keep_with_next = i < len(t.rows) - 1     # la tabla no se parte entre paginas
    if nota:
        runs(doc.add_paragraph(), nota, tam=8, color=GRIS)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(0)


K, OP, MO = C["contrato24"], C["operacion"], C["motor"]
T72 = {f["tramo"]: f for f in C["tabla72"]}
rn, rnino = C["regimen_normal_ene_abr"], C["regimen_el_nino_may_ago"]

# ---------------------------------------------------------------- encabezado
p = doc.add_paragraph(); p.add_run().add_picture(str(FUN / "augur_logo.png"), width=Cm(3.0))
runs(doc.add_paragraph(), "**Augur: resumen de resultados**", tam=20, color=TINTA).paragraph_format.space_after = Pt(0)
runs(doc.add_paragraph(), "Pronóstico horario del precio de bolsa en Colombia y motor de decisión · 13 de noviembre de 2026",
     tam=10.5, color=GRIS)
runs(doc.add_paragraph(), "Autores: Juan David Barceló Barraza y Rafael Andrés Dede Perdomo · Asesores: Daniela María "
     "Charris Stand y José Daniel Soto Ortiz · Ingeniería Electrónica, Universidad del Norte", tam=9, color=GRIS)

P(f"**En una línea.** Augur pronostica el precio de bolsa hora a hora con un error medio de {n(K['MAE'])} COP/kWh, "
  f"{n((1 - K['MAE'] / K['MAE_pre']) * 100, 0)} % menos que el predespacho ideal de XM y "
  f"{n((1 - K['MAE'] / K['MAE_pers']) * 100, 0)} % menos que repetir el último precio publicado, y lo convierte en "
  "recomendaciones de comprar, vender o esperar. Usa solo datos públicos y opera en línea desde agosto de 2026.")

H("1. Qué se entrega")
tabla(["Entregable", "Dónde"], [
    ["Dashboard en operación (se actualiza tres veces al día)", "augurpredictor.streamlit.app"],
    ["Video, versión de presentación final (2 min)", "youtu.be/D3X6VozGjP4"],
    ["Video, versión extendida (4 min 32 s)", "youtu.be/_N-HcAQXvl0"],
    ["Informe final (38 páginas), manual de usuario, documento de funcionamiento", "Repositorio, carpeta docs/"],
    ["Código, datos, bitácora", "github.com/Ziskito/xm-spot-price-predictor"]], [3, 2])
P("**Diseño definitivo.** El predespacho ideal de XM corregido por tres modelos CatBoost, usando solo lo publicado a "
  "las 00:00 del día D: precio y demanda hasta D-4, embalses y aportes de D-2, el ONI de dos meses atrás y el "
  "predespacho que XM publica la víspera. La banda de incertidumbre es conforme (calibrada con los errores recientes de "
  "cada hora) y el motor compara el precio esperado con los percentiles de los últimos 30 días.")

# ---------------------------------------------------------------- 24 h
H("2. Pronóstico a 24 horas")
P(f"En las mismas {n(C['horas_comunes'], 0)} horas de 2026 (15-ene a 5-ago) y con solo información pública al corte, el "
  "diseño definitivo tiene el menor error de todos los modelos probados. Los modelos sin predespacho (incluido el ensamble "
  "del informe de avance) quedan cerca de 100 COP/kWh; el predespacho es el dato que más aporta, y los CatBoost corrigen "
  "su error.")
figura(FUN / "comparacion_mae.png", 15.5, "Figura 1. Error medio absoluto (MAE) a 24 h por modelo, 2026.")
tabla(["Medida", "Augur", "Predespacho de XM", "Repetir el último precio"], [
    ["Error medio (MAE, COP/kWh)", n(K["MAE"]), n(K["MAE_pre"]), n(K["MAE_pers"])],
    ["Diferencia de Augur (Diebold-Mariano)", "—", f"t = {n(K['DM_vs_pre'][0])}, p < 0,001",
     f"t = {n(K['DM_vs_pers'][0])}, p < 0,001"],
    ["Ene-abr (régimen normal), MAE", n(rn["MAE"]), n(rn["MAE_pre"]), n(rn["MAE_pers"])],
    ["May-ago (El Niño), MAE", n(rnino["MAE"]), f"{n(rnino['MAE_pre'])} (diferencia no significativa, p = {n(rnino['DM_vs_pre'][1], 2)})", n(rnino["MAE_pers"])]],
    [3.2, 1.4, 2.4, 2.2], titulo=f"Tabla 1. Pronóstico de 24 h, 15-ene a 5-ago-2026 ({n(K['horas'], 0)} horas).",
    nota=f"Error medio con signo: {n(K['sesgo'])} COP/kWh (sin sesgo). MAPE {n(K['MAPE'])} %.")

# ---------------------------------------------------------------- 72 h y banda
H("3. Pronóstico a 72 horas y banda de incertidumbre")
P(f"La banda [q10, q90] promete contener el precio real 8 de cada 10 horas, y lo cumple: a 24 h cubrió el "
  f"**{n(K['cobertura'])} %** de las horas, con un ancho medio de {n(K['ancho'], 0)} COP/kWh. A 72 h el error crece con el "
  "horizonte, pero el tercer día todavía es mejor que repetir el último precio.")
tabla(["Tramo", "MAE Augur", "MAE repetir último precio", "Cobertura de la banda", "Ancho medio de la banda"],
      [[f"Día {i} ({t})", n(T72[t]["MAE"]), n(T72[t]["MAE_pers"]), f"{n(T72[t]['cobertura'])} %", n(T72[t]["ancho"], 0)]
       for i, t in enumerate(("1-24 h", "25-48 h", "49-72 h"), start=1)],
      [1.8, 1.3, 1.8, 1.6, 1.6], titulo="Tabla 2. Pronóstico de 72 h por día (COP/kWh), 2026.")

# ---------------------------------------------------------------- operacion
ruta_m, mm = fig_mensual()
H("4. Operación en vivo")
P(f"Desde el 6 de agosto el sistema publica su pronóstico todos los días. En todo 2026 hasta el 31 de octubre "
  f"({n(T26['horas'], 0)} horas), el error medio es **{n(T26['MAE'])} COP/kWh**, "
  f"{n((1 - T26['MAE'] / T26['MAE_pre']) * 100, 0)} % menor que el del predespacho ({n(T26['MAE_pre'])}; Diebold-Mariano "
  f"t = {n(T26['DM_vs_pre'][0])}, p < 0,001), y la banda cubre el {n(T26['cobertura'])} % de las horas. Septiembre y octubre "
  "fueron más volátiles tras la Resolución CREG 101 127 de 2026, que cambió las reglas de oferta de las hidráulicas; aun "
  "así, Augur se mantiene por debajo del predespacho.")
figura(ruta_m, 15.5, "Figura 2. Error medio por mes de 2026: Augur frente al predespacho de XM (sombreado: operación en vivo).")

# ---------------------------------------------------------------- motor
H("5. Motor de decisión")
P("El motor compara el precio esperado de cada hora con los percentiles 25 y 75 de los últimos 30 días y recomienda "
  "actuar solo en la mitad del día que le corresponde (filtro intradía). Con el criterio de estabilidad (ventaja positiva "
  "y frecuencia de acción entre 10 y 40 % en las dos mitades de 2026) elige el método «rodante» en los cuatro casos.")
tabla(["Resultado", "Valor"], [
    ["Ventaja promedio del método elegido (24 h)", "187,5 a 244,7 COP/kWh sobre el precio medio (394,9)"],
    ["Mejora del filtro intradía en la mitad más débil (generador 24 h)", "de 73,4 a 109,3 COP/kWh"],
    ["Filtro de precio de contrato (MC de XM)", "nunca empeora; mejora de forma significativa en 8 de 14 casos"],
    ["Último mes del periodo, cliente de ejemplo de 100 kW (generador 24 h)", "+3,84 millones de COP con «rodante»"]],
    [3.2, 3.0], titulo="Tabla 3. Resultados del motor de decisión, 2026.",
    nota="Ventaja = diferencia entre el precio real medio en las horas en que el motor recomienda actuar y el del periodo.")
figura(INF / "motor_operacion.png", 15.5, "Figura 3. El motor en operación (13 al 19 de julio de 2026, generador): pronóstico, "
       "banda, precio real y horas de acción.")

# ---------------------------------------------------------------- dashboard
H("6. El dashboard")
P("Tres vistas: **Operador** (qué hacer ahora), **Analista** (backtesting de las reglas con datos reales) y "
  "**Metodología** (desempeño en vivo, cómo funciona y el video). Capturas del 23 de octubre de 2026, rol generador.")
figura(CAP / "op_arriba.png", 16.5, "Figura 4. Vista Operador: recomendación de la hora, nivel de confianza, posibilidad de "
       "pico o desplome y pronóstico del día con la línea de contrato.")
figura(CAP / "op_calor.png", 14.5, "Figura 5. Mapa de calor día-hora con la señal del motor (21 al 23 de octubre).")
figura(CAP / "op_semana.png", 14.5, "Figura 6. La semana alrededor del día elegido: pronóstico, banda, precio real y acciones.")
figura(recorte("an_todo.png", (0, 0, 2160, 1600), "analista_arriba.png"), 15.5,
       "Figura 7. Vista Analista: ventaja, frecuencia de acción y comparación de los cuatro métodos en el rango elegido.")
figura(recorte("met_arriba.png", (0, 0, 2160, 1850), "metodologia_arriba.png"), 15.5,
       "Figura 8. Vista Metodología: video explicativo y desempeño del pronóstico en vivo.")

# ---------------------------------------------------------------- caracterizacion
H("7. Caracterización del precio")
ci, ca, ce = CC["ciclo"], CC["calor"], CC["espectro"]
V(f"**Ciclo diario.** El precio y la demanda tienen su máximo a las {ci['h_max_p']} h; los dos perfiles horarios tienen "
  f"una correlación de {n(ci['corr'], 2)}.")
V(f"**Nivel por año.** El promedio anual va de {n(ca['anual']['2021'], 0)} COP/kWh (2021) a {n(ca['anual']['2024'], 0)} "
  f"(2024); el mes más caro fue octubre de 2024 ({n(ca['max'], 0)}), con el ONI ya neutral y los embalses bajos.")
V(f"**El ciclo diario cambia con el régimen.** Su amplitud promedia {n(ce['nino'], 0)} COP/kWh en El Niño y "
  f"{n(ce['nina'], 0)} en La Niña: en escasez se amplía la diferencia entre el día y la noche, que es lo que aprovecha "
  "el motor.")
figura(INF / "carac_calor_mes.png", 15.5, "Figura 9. Precio promedio por año y mes (COP/kWh), con la escala de color de XM.")
figura(INF / "carac_espectrograma.png", 15.5, "Figura 10. Espectrograma del precio sin tendencia y amplitud del ciclo "
       "diario (franjas: El Niño y La Niña).")

# ---------------------------------------------------------------- validacion y pendientes
H("8. Validación con usuarios y lo que falta")
P("**Formulario corto (clase del 20 de octubre):** 16 respuestas (15 estudiantes y un profesor), nota media 8,5 de 10 y "
  "entre 75 y 88 % de acuerdo con cada afirmación de comprensión y utilidad. Lo que más pidieron: explicar cómo se calculan las "
  "recomendaciones y la banda, y hacer más visibles las unidades y las notas.")
P("**Pendiente para el 20 de noviembre:** respuestas del formulario completo para profesionales del sector (Tabla 21 "
  "del informe), presentación final, póster y resumen para la biblioteca. Sustentación el 24 de noviembre y exposición "
  "en el coliseo el 27.")

pie = sec.footer.paragraphs[0]
pie.alignment = WD_ALIGN_PARAGRAPH.CENTER
runs(pie, "Augur · resumen de resultados · 13-nov-2026", tam=8, color=GRIS)
doc.save(str(DESTINO))
print("Guardado:", DESTINO)
