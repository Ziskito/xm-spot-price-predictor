# -*- coding: utf-8 -*-
"""Genera el informe de Word consolidado del proyecto (version corta, enfocada en estado actual)."""
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ASSETS = Path(r"C:\Users\mgdbj\xm-spot-price-predictor\scripts_experimento\informe_assets")
OUT = Path(r"C:\Users\mgdbj\xm-spot-price-predictor\docs\Informe_Pronostico_Precio_Bolsa_XM.docx")

AZUL = RGBColor(0x18, 0x4f, 0x95)
GRIS = RGBColor(0x52, 0x51, 0x4e)
NEGRO = RGBColor(0x0b, 0x0b, 0x0b)

doc = Document()

# ---------- estilos base ----------
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.font.color.rgb = NEGRO
normal.paragraph_format.space_after = Pt(8)

for lvl, size, color in [(1, 20, AZUL), (2, 15, AZUL), (3, 12.5, GRIS)]:
    st = doc.styles[f"Heading {lvl}"]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.color.rgb = color
    st.font.bold = True
    st.paragraph_format.space_before = Pt(18 if lvl == 1 else 12)
    st.paragraph_format.space_after = Pt(8)


def set_col_widths(table, widths_cm):
    table.autofit = False
    for row in table.rows:
        for cell, w in zip(row.cells, widths_cm):
            cell.width = Cm(w)


def add_table(headers, rows, widths=None, small=False):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.5 if small else 10.5)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "184F95")
        hdr[i]._tc.get_or_add_tcPr().append(shd)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            r = p.add_run(str(val))
            r.font.size = Pt(9.5 if small else 10.5)
    if widths:
        set_col_widths(t, widths)
    doc.add_paragraph()
    return t


def add_pic(name, width=6.3, caption=None):
    doc.add_picture(str(ASSETS / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(caption)
        r.italic = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = GRIS


def p(text="", bold=False, italic=False, size=11, color=None, align=None, space_after=8):
    par = doc.add_paragraph()
    par.paragraph_format.space_after = Pt(space_after)
    if align:
        par.alignment = align
    r = par.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    return par


def bullet(text, level=0):
    par = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    par.add_run(text)
    return par


# =====================================================================
# PORTADA
# =====================================================================
p("", size=8, space_after=0)
p("", size=8, space_after=0)
p("Pronóstico del precio de bolsa de energía eléctrica en Colombia", bold=True, size=26,
  color=AZUL, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
p("Estado actual del sistema de pronóstico: arquitectura, resultados y comparación con la "
  "literatura", size=14, color=GRIS, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)

add_table(["", ""], [
    ["Universidad", "Universidad del Norte"],
    ["Proyecto", "Predicción del precio spot de energía en el mercado XM (Colombia)"],
    ["Autor (OE1-OE2, modelos de pronóstico)", "Juan David Barceló Barraza"],
    ["Colaborador (OE3, motor de decisión)", "Rafael Andrés Dede Perdomo"],
    ["Fecha de este informe", "18 de septiembre de 2026"],
], widths=[6, 10])

p("Resumen ejecutivo", bold=True, size=13, color=AZUL, space_after=6)
p("El sistema pronostica el precio de bolsa horario de Colombia en dos horizontes — 24 horas "
  "(día siguiente) y 72 horas — combinando varios modelos de familias distintas por ensamble. "
  "Este documento describe la arquitectura vigente, sus resultados verificados 100% fuera de "
  "muestra, la comparación contra otros modelos y contra la literatura de mercados eléctricos "
  "hidro-dominados, y el límite cuantificado de mejora con los datos actuales.")
p("Resultados centrales: MAPE 10.74% a 24 horas y MAPE 16.94% a 72 horas (global). Ambos superan "
  "los puntos de referencia disponibles en la literatura de mercados hidro-dominados, en las "
  "métricas que cada estudio reporta.")
doc.add_page_break()

# =====================================================================
# 1. OBJETIVO Y ALCANCE
# =====================================================================
doc.add_heading("1. Objetivo y alcance", level=1)
p("El proyecto pronostica el precio de bolsa horario del mercado eléctrico colombiano, operado "
  "por XM. Colombia tiene una matriz de generación aproximadamente 70-74% hidroeléctrica — 72% "
  "según la IEA (Colombia 2023, Energy Policy Review) y 74% según datos de generación 2025 "
  "(lowcarbonpower.org) —, lo que hace que el precio dependa fuertemente del régimen hidrológico "
  "(embalses, aportes de caudal) y de su modulación por el fenómeno ENSO (El Niño / La Niña). El "
  "período de prueba (2026) coincide con un episodio de El Niño en transición.")
p("Dos productos de pronóstico:")
bullet("24 horas (día siguiente): protocolo day-ahead estándar de la literatura EPF, corte a las "
       "23:00, pronóstico de las 24 horas del día siguiente.")
bullet("72 horas: pronóstico continuo hora a hora hasta 3 días adelante, corte a las 00:00, para "
       "la vista de planeación del motor de decisión (OE3).")
p("Todos los resultados se miden 100% fuera de muestra: entrenamiento en 2019-2025 (60,625 "
  "horas), evaluación en 2026 (enero-agosto), un período que el modelo nunca vio durante el "
  "entrenamiento. La variable de precio de oferta marginal (max_precio_oferta) nunca se usa como "
  "insumo por ser fuga de información (correlación contemporánea 0.9998 con el precio de bolsa), "
  "y las variables de generación se usan solo como agregados de sistema, nunca por tipo de "
  "recurso, según el alcance definido en el Anexo 1 del proyecto.")

# =====================================================================
# 2. METODOLOGÍA
# =====================================================================
doc.add_heading("2. Metodología", level=1)
p("La validación principal usa seis orígenes temporales de tipo walk-forward — cada uno con su "
  "propio corte de entrenamiento (ventana expansiva desde el inicio del histórico) y su propia "
  "ventana de prueba de 12 meses — diseñados para cubrir los regímenes ENSO observados: dos "
  "episodios de La Niña, una transición triple-dip, El Niño fuerte 2023-2024, y El Niño 2026 (el "
  "período de prueba final). Evaluar en un solo corte temporal esconde fallas que solo aparecen "
  "en ciertos regímenes — el walk-forward de 6 orígenes existe para no permitir eso; ya reveló "
  "una falla real de dos modelos que una ventana de prueba corta (3 meses) no alcanzaba a mostrar.")

doc.add_heading("2.1 Métricas usadas y por qué", level=2)
add_table(["Métrica", "Qué mide", "Por qué se usa"], [
    ["MAE", "Error absoluto medio, en COP/kWh", "Unidad física real del error"],
    ["MAPE", "Error porcentual respecto al precio real", "Estándar de la literatura EPF; métrica "
     "principal del proyecto"],
    ["sMAPE", "Error porcentual simétrico (divide por el promedio de real y predicho)",
     "Más robusto que el MAPE cuando el precio real es bajo; métrica del paper de Nueva Zelanda"],
    ["MASE", "MAE del modelo / MAE de un pronóstico ingenuo", "Única métrica válida para "
     "comparar con el paper neozelandés (sus otras métricas están en escala transformada)"],
    ["R² / NRMSE", "Varianza explicada / RMSE normalizado a porcentaje", "Formato de comparación "
     "de los papers de Brasil (Dias et al.) y Noruega"],
    ["TAPI", "Acierto de la dirección del cambio (sube/baja)", "Relevante para decisiones de "
     "compra/venta del motor OE3"],
], widths=[2.8, 6.7, 6.5], small=True)
p("Toda mejora reportada en este documento pasó la prueba de Diebold-Mariano (1995, corrección "
  "de varianza HAC/Newey-West) — el estándar del campo para decidir si una diferencia de error "
  "entre dos modelos es real o puede deberse al azar — y se verificó con al menos 10 particiones "
  "de datos distintas más una versión \"desplegable\" (pesos calculados solo con información "
  "anterior al corte, sin usar días futuros), que es la cifra honesta de producción real.")

doc.add_page_break()

# =====================================================================
# 3. ESTADO ACTUAL DEL SISTEMA
# =====================================================================
doc.add_heading("3. Estado actual del sistema", level=1)
p("Ambos horizontes se resuelven con un ensamble (combinación ponderada) de modelos de familias "
  "distintas, no con un solo modelo — la ganancia viene de combinar modelos cuyos errores no "
  "están correlacionados entre sí, no de que un modelo individual sea el mejor posible.")

doc.add_heading("3.1 Modelo de 24 horas — MAE 41.29, MAPE 10.74%, sMAPE 10.07%", level=2)
p("Combina 6 modelos de familias distintas — persistencia, gradient boosting (XGBoost), "
  "econométrico (ARX+GARCH), deep learning (N-BEATSx, N-HiTS), y un modelo LE-GARCH-t (lineal "
  "con selección LASSO + GARCH + errores de colas pesadas, tomado del modelo ganador de un "
  "estudio sobre el mercado neozelandés) —, con pesos ajustados por franja horaria de 6 horas "
  "usando un combinador de mínima desviación absoluta ponderado por la métrica que se reporta "
  "(MAPE/sMAPE), en vez de por error cuadrático. Esa alineación entre el objetivo del combinador "
  "y la métrica reportada resultó, de forma verificada, en una mejora simultánea de MAE, MAPE y "
  "sMAPE — un efecto de regularización: reduce la influencia de los días de precio muy alto "
  "sobre el ajuste de pesos.")

p("Qué hace cada uno de los 6 modelos, en términos simples:", bold=True, space_after=4)
add_table(["Modelo", "Idea central"], [
    ["Persistencia", "Predice que el precio de una hora será igual al de la misma hora, una "
     "semana atrás. Es la comparación \"ingenua\" de referencia: si un modelo no le gana a esto, "
     "no aporta nada."],
    ["XGBoost (gradient boosting)", "Construye cientos de árboles de decisión pequeños en "
     "secuencia, cada uno corrigiendo los errores del anterior. Bueno para capturar relaciones "
     "no obvias entre variables (p. ej. \"si el embalse está bajo y es diciembre, el precio sube "
     "más de lo normal\") sin que haya que programarlas a mano."],
    ["ARX+GARCH", "Modelo econométrico en dos partes: una parte lineal predice el nivel del "
     "precio a partir de su propio pasado y de variables como demanda o hidrología, y una parte "
     "GARCH modela cómo la volatilidad del error se agrupa en el tiempo (los períodos inestables "
     "tienden a seguir siendo inestables)."],
    ["N-BEATSx / N-HiTS", "Redes neuronales diseñadas específicamente para pronóstico de series "
     "de tiempo (no son redes genéricas adaptadas). Descomponen la señal del precio en "
     "componentes de distinta escala (tendencia, estacionalidad) y aceptan variables externas "
     "como hidrología, ONI y calendario."],
    ["LE-GARCH-t", "El más reciente: lineal con selección automática de variables (LASSO) y "
     "errores de \"colas pesadas\" (distribución t de Student en vez de normal), que resta peso "
     "a los días de precio extremo al momento de ajustar el modelo — tomado del modelo ganador "
     "de un estudio sobre el mercado neozelandés."],
], widths=[4.5, 11.5], small=True)
p("CatBoost, un modelo de dos etapas (\"hurdle\"), Markov-Switching y un filtro recursivo "
  "(RecursiveLS) también se probaron como votantes adicionales, pero se descartaron: sumarlos "
  "empeoraba el resultado del ensamble de forma estadísticamente significativa (documentado en "
  "el informe comparativo de modelos del proyecto).", italic=True, size=9.5, color=GRIS)

add_pic("fig1_evolucion_24h.png", 6.0)

doc.add_heading("3.2 Modelo de 72 horas — MAE 58.84, MAPE 16.94% (global)", level=2)
p("Vota entre un LEAR (LASSO Estimated AutoRegressive, el benchmark canónico del campo — un "
  "modelo lineal con selección de variables por LASSO, uno por cada uno de los 72 pasos de "
  "horizonte), dos redes N-BEATSx y una persistencia estacional. El LEAR resultó ser el mejor "
  "modelo individual del proyecto a horizonte largo, y permitió eliminar tres modelos de árboles "
  "del ensamble sin perder precisión — más simple y más preciso a la vez.")
p("El tramo 1-24h de este horizonte se resuelve reutilizando directamente el ensamble de 24h "
  "descrito arriba, en vez de los modelos generalistas de 72h — se detectó que el canal de 72h "
  "predecía su primer día peor que el modelo dedicado ya existente para ese mismo día (MAPE "
  "13.03% contra 10.54%, con menos información en el segundo caso), sin fuga de datos de por "
  "medio. Esta corrección solo cubre los pasos 1-23; los tramos 25-72h no tienen todavía una "
  "mejora equivalente confirmada.")

p("Qué hace cada modelo, en términos simples:", bold=True, space_after=4)
add_table(["Modelo", "Idea central"], [
    ["LEAR", "El benchmark de referencia del campo: una regresión lineal con selección "
     "automática de variables por LASSO — de decenas de variables candidatas (precio en "
     "distintos momentos del pasado, calendario, festivos), el LASSO elige solo las que de "
     "verdad aportan y descarta el resto, evitando sobreajuste. Se entrena un modelo separado "
     "para cada uno de los 72 pasos del horizonte."],
    ["N-BEATSx / N-BEATSx-exógeno", "La misma red neuronal de pronóstico descrita en la Sección "
     "3.1, entrenada aquí para cubrir el horizonte completo de 72 horas de una sola vez; la "
     "versión \"exógeno\" usa un conjunto más amplio de variables externas."],
    ["Persistencia estacional", "Predice que el precio de una hora futura será igual al de la "
     "misma hora, la semana pasada — captura el patrón semanal del mercado (un lunes se parece a "
     "otro lunes) sin necesitar ningún ajuste."],
    ["\"Puente\" (pasos 1-23)", "No es un modelo nuevo: para las primeras 23 horas del horizonte "
     "de 72h, en vez de usar los cuatro modelos generalistas de arriba, se reutiliza directamente "
     "el ensamble completo de 24 horas de la Sección 3.1 (con sus 6 modelos) como un solo votante "
     "adicional — es el canal más preciso disponible para ese tramo."],
], widths=[4.5, 11.5], small=True)

add_pic("fig3_tramos_puente.png", 6.0)

add_table(["Tramo", "MAE", "MAPE", "sMAPE"], [
    ["1-24h", "41.08", "10.84%", "10.11%"],
    ["25-48h", "62.56", "18.29%", "16.53%"],
    ["49-72h", "72.87", "21.69%", "19.44%"],
    ["Global", "58.84", "16.94%", "15.36%"],
], widths=[4, 4, 4, 4])

doc.add_heading("3.3 Dónde vive el error residual", level=2)
add_pic("fig4_error_por_hora.png", 6.0)
p("Seis horas del día concentran el 37.8% del error total del modelo de 24h: la medianoche (0h), "
  "la media mañana (8-10h) y el inicio de la noche (18-19h). En esas seis horas el sesgo explica "
  "apenas el 8.4% del error — el 91.6% restante es varianza genuina, no un error de modelado "
  "corregible: picos de precio que dependen de qué unidad de generación entra a cubrir la demanda "
  "en ese momento, información de ofertas por unidad e indisponibilidades programadas que hoy está "
  "fuera del alcance de datos del proyecto (Sección 1). La Sección 6 cuantifica el techo que esto "
  "impone.")
p("Nota sobre la medianoche: el precio de bolsa da un salto grande al cambiar de día de despacho "
  "(en 2026, 56.5 COP/kWh de diferencia media entre las 23:00 y las 00:00, frente a 7.4 entre las "
  "00:00 y la 01:00). Ese salto es la razón de que la hora 0 sea una de las más difíciles del día "
  "y de que el origen del pronóstico importe tanto en ese punto.", italic=True, size=9.5,
  color=GRIS)

doc.add_page_break()

# =====================================================================
# 4. COMPARACIÓN CON OTROS MODELOS PROBADOS
# =====================================================================
doc.add_heading("4. Comparación con otros modelos probados internamente", level=1)
add_table(["Modelo", "Familia", "MAE (holdout 2026)", "MAPE"], [
    ["Persistencia (t-24h)", "Línea base", "56.31", "15.75%"],
    ["XGBoost", "Gradient boosting", "61.15", "15.89%"],
    ["N-HiTS", "Deep learning", "57.08", "16.36%"],
    ["N-BEATSx", "Deep learning", "56.11", "15.93%"],
    ["ARX+GARCH(1,1)", "Econométrico", "55.76", "15.43%"],
    ["Ensamble v4 (actual)", "Combinación", "41.29", "10.74%"],
], widths=[4.6, 4, 3.4, 3])
p("El holdout único favorece a ARX+GARCH, pero la validación walk-forward de 6 orígenes (Sección "
  "2) muestra que esa ventaja no es robusta entre regímenes: N-BEATSx le gana con significancia "
  "estadística en 5 de 6 orígenes, y XGBoost pierde contra la persistencia simple en los 6.")
add_pic("fig6_walkforward_regimenes.png", 6.3,
        "Error (MAE) por régimen hidrológico/ENSO en los 6 orígenes de validación. La ventaja de "
        "las redes neuronales es mayor justo en los regímenes de El Niño (orígenes 4-6), donde "
        "más importa acertar por el mayor riesgo económico.")

doc.add_page_break()

# =====================================================================
# 5. COMPARACIÓN CON LA LITERATURA
# =====================================================================
doc.add_heading("5. Comparación con la literatura", level=1)
p("Colombia: Gallón & Barrientos (2021, IJEEP) reportan MAPE 6.72% a 1 día para precio horario "
  "colombiano (2000-2017) — período mucho más tranquilo, sin fundamentales. Da contexto: incluso "
  "un benchmark univariado simple en período calmo da 6-9% de MAPE; este proyecto llega a 10.74% "
  "en 2026, el año más volátil del histórico, con variables causales y validación de 6 regímenes. "
  "Herrera-Mejía et al. (2025, Smart Energy) — AES Colombia — no es comparable en objetivo "
  "(predice oferta por planta, no precio agregado) pero valida de forma independiente que los "
  "modelos fallan igual en transiciones de régimen.")
p("Mercados hidro-dominados comparables (~70%+ hidro): Nueva Zelanda, Brasil y Noruega/Nord Pool. "
  "Ninguno pronostica a resolución horaria como este proyecto (predicen un agregado diario o "
  "semanal, tarea estructuralmente más fácil), así que la comparación válida es \"superamos su "
  "propia métrica normalizada en una tarea más agregada\", nunca una victoria directa sin ese "
  "matiz.")

add_pic("fig5_comparacion_papers.png", 6.5,
        "Comparación contra cada estudio en SU PROPIA métrica reportada (paneles no comparables "
        "entre sí).")

p("El resultado más sólido: MASE < 1 significa \"le gana al pronóstico ingenuo\". Este proyecto "
  "da MASE 0.914; los 33 modelos del paper neozelandés (Kapoor & Wichitaksorn, 2023) tienen todos "
  "MASE > 1.26 — ninguno le gana a su propio ingenuo (el mejor es LE-GARCH-t con 1.2626 en Central "
  "North Island; el peor, RFE-GARCH con 2.5001 en Lower South Island). Los cuatro estudios "
  "internacionales coinciden, además, en un hallazgo que ya se había adoptado en este proyecto por "
  "evidencia propia antes de conocerlos: modelos simples con buena selección de variables (lineales "
  "regularizados, pesos por franja horaria) igualan o superan al deep learning complejo.")

p("Salvedad sobre esta comparación, verificada contra las Tablas 5, 6 y 8 del paper original. "
  "Kapoor & Wichitaksorn transforman los precios antes de modelar (Box-Cox o Yeo-Johnson, seguido "
  "de escalado min-max), porque su serie es muy asimétrica. El MASE es invariante al escalado "
  "lineal — se cancela en el cociente — pero no a la transformación logarítmica. Su MASE está por "
  "tanto medido sobre precio transformado y el de este proyecto sobre COP/kWh crudos: no son "
  "exactamente la misma cantidad. Sumado a que ellos pronostican precio diario y este proyecto "
  "precio horario, la lectura honesta de esta comparación es \"nuestro modelo supera a su propio "
  "ingenuo en una tarea más difícil, cosa que ninguno de los suyos logra en la más fácil\", nunca "
  "una comparación numérica directa.")

doc.add_page_break()

# =====================================================================
# 6. POR QUÉ HAY UN LÍMITE CON LOS DATOS ACTUALES
# =====================================================================
doc.add_heading("6. Por qué el error tiene un límite con los datos actuales", level=1)
add_pic("fig7_techo_8pct.png", 6.0)
p("A 24h, el ensamble está en meseta (agregar más votantes o afinar la granularidad de pesos no "
  "da mejoras significativas), y el techo teórico — acertando perfecto en las 6 horas más "
  "difíciles del día — es 6.66% de MAPE. Bajar más exigiría información nueva (ofertas por unidad "
  "de generación, indisponibilidades programadas), no más modelado sobre las variables ya "
  "disponibles.")
p("A 25-72h se probaron tres vías para extender la mejora del tramo 1-24h (trasladar el votante "
  "de colas pesadas, árboles especializados por tramo, diversidad intra-familia de redes) y las "
  "tres quedaron cerradas con resultado negativo o no concluyente: ahí no existe un canal "
  "\"hermano\" más preciso que trasplantar, y el estado observable en el corte informa cada vez "
  "menos mientras más lejos está el objetivo.")

# =====================================================================
# 7. CONCLUSIONES
# =====================================================================
doc.add_heading("7. Conclusiones y recomendaciones", level=1)
bullet("El sistema actual supera todos los modelos individuales probados internamente y los "
       "puntos de referencia disponibles en la literatura de mercados hidro-dominados, en las "
       "métricas que cada estudio reporta.")
bullet("El error de 24h (MAPE 10.74%) está cerca de su límite teórico (6.66% en el mejor caso "
       "posible) con los datos actuales.")
bullet("El error de 72h se benefició de una corrección estructural grande en su primer día; los "
       "tramos 25-72h son la frontera de trabajo más clara para la siguiente ronda.")
bullet("Pendiente de decisión conjunta con el equipo: regenerar los contratos oficiales del "
       "dashboard de OE3 con las composiciones nuevas, y ajustar el umbral del régimen "
       "hidrológico de la señal de confianza a uno relativo (móvil de 90 días) — 2026 no tiene "
       "ninguna hora en régimen de \"embalse bajo\" bajo el umbral histórico absoluto actual.")

doc.add_page_break()

# =====================================================================
# REFERENCIAS
# =====================================================================
doc.add_heading("Referencias", level=1)
refs = [
 "Albani, V., Marcavillaca, R., Moreira, E. et al. (2025). Short-term forecasting of forward "
 "prices in the Brazilian electricity market with a hybrid stochastic-neural network model. "
 "Energy Economics, 148.",
 "Challu, C., Olivares, K. G., Oreshkin, B. N. et al. (2023). N-HiTS: Neural Hierarchical "
 "Interpolation for Time Series Forecasting. Proceedings of the AAAI Conference on Artificial "
 "Intelligence, 37(6).",
 "Dias, M., Lira, F., Freire, R. (2024). Methodology for Multi-Step Forecasting of Electricity "
 "Spot Prices Based on Neural Networks Applied to the Brazilian Energy Market. Energies, 17(8), "
 "1864.",
 "Diebold, F. X., Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & "
 "Economic Statistics, 13(3).",
 "Gallón, S., Barrientos, J. (2021). Forecasting the Colombian Electricity Spot Price under a "
 "Functional Approach. International Journal of Energy Economics and Policy, 11(2), 67-74.",
 "Herrera-Mejía, L. et al. (2025). Estudio de AES Colombia sobre clasificación de régimen de "
 "oferta de plantas hidroeléctricas individuales bajo riesgo de sequía. Smart Energy, Elsevier, "
 "diciembre 2025. Título exacto pendiente de verificar contra el PDF original.",
 "International Energy Agency (2023). Colombia 2023 — Energy Policy Review. IEA, París.",
 "Kapoor, G., Wichitaksorn, N. (2023). Electricity price forecasting in New Zealand: A "
 "comparative analysis of statistical and machine learning models with feature selection. "
 "Applied Energy, 347.",
 "Lago, J., Marcjasz, G., De Schutter, B., Weron, R. (2021). Forecasting day-ahead electricity "
 "prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. "
 "Applied Energy, 293, 116983.",
 "Low Carbon Power (2025). Colombia Electricity Generation Mix. lowcarbonpower.org.",
 "Muñoz-Santiago, N. et al. (2017). Modelo ARIMA con especificación IGARCH aplicado al mercado "
 "eléctrico colombiano. Referencia interna del proyecto para el modelo ARX+GARCH.",
 "Nowotarski, J., Weron, R. (2015). Computing electricity spot price prediction intervals using "
 "quantile regression and forecast averaging. Computational Statistics, 30.",
 "Olivares, K. G., Challu, C., Marcjasz, G., Weron, R., Dubrawski, A. (2023). Neural basis "
 "expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx. "
 "International Journal of Forecasting, 39(2).",
 "s.a. (2026). Electricity price forecasting across Norway's five bidding zones in the "
 "post-crisis era. arXiv:2604.26634.",
]
for r in sorted(refs):
    par = doc.add_paragraph(style="List Bullet")
    par.paragraph_format.space_after = Pt(6)
    run = par.add_run(r)
    run.font.size = Pt(10)

p("", size=6)
p("Nota: la bitácora completa del proyecto, con el detalle de cada experimento (incluidos los "
  "que no funcionaron), está en README.md del repositorio. Este informe resume el estado actual "
  "y cómo se llegó a él; no repite el detalle iterativo de cada prueba intermedia.", italic=True,
  size=9.5, color=GRIS)

doc.save(str(OUT))
print("Informe corto guardado en", OUT)
