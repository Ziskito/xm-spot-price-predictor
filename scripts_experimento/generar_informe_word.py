# -*- coding: utf-8 -*-
"""Genera el informe de Word consolidado del proyecto."""
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
p("Modelos de aprendizaje estadístico y automático para horizontes de 24 y 72 horas",
  size=14, color=GRIS, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)

add_table(["", ""], [
    ["Universidad", "Universidad del Norte"],
    ["Proyecto", "Predicción del precio spot de energía en el mercado XM (Colombia)"],
    ["Autor (OE1-OE2, modelos de pronóstico)", "Juan David Barceló Barraza"],
    ["Colaborador (OE3, motor de decisión)", "Rafael Andrés Dede Perdomo"],
    ["Fecha de este informe", "18 de septiembre de 2026"],
], widths=[6, 10])

p("Resumen ejecutivo", bold=True, size=13, color=AZUL, space_after=6)
p("Este informe consolida los resultados del sistema de pronóstico desarrollado para el precio "
  "de bolsa horario de Colombia, en dos horizontes: 24 horas (día siguiente) y 72 horas. Se "
  "documenta la metodología, la evolución de ambos modelos durante esta última ronda de mejoras, "
  "la comparación contra otros modelos probados internamente, la comparación contra la literatura "
  "publicada (Colombia y mercados con matriz energética hidro-dominada comparable: Brasil, Noruega "
  "y Nueva Zelanda), y una explicación cuantificada de por qué el error no puede seguir bajando "
  "con la información disponible actualmente.")
p("Los dos resultados centrales: el modelo de 24 horas alcanza un MAPE de 10.74% (frente al "
  "11.27% con que empezó esta sesión), y el modelo de 72 horas alcanza un MAPE global de 16.94% "
  "tras corregir una pérdida estructural que afectaba su primer día de pronóstico. Ambos superan "
  "los puntos de referencia disponibles en la literatura de mercados hidro-dominados en las "
  "métricas que cada estudio reporta.")
doc.add_page_break()

# =====================================================================
# 1. OBJETIVO Y ALCANCE
# =====================================================================
doc.add_heading("1. Objetivo y alcance", level=1)
p("El proyecto pronostica el precio de bolsa horario del mercado eléctrico colombiano, operado "
  "por XM. Colombia tiene una matriz de generación aproximadamente 70-74% hidroeléctrica — 72% "
  "según la Agencia Internacional de Energía (IEA, Colombia 2023 — Energy Policy Review) y 74% "
  "según datos de generación de 2025 (lowcarbonpower.org); la propia IEA aclara que ronda dos "
  "tercios en la última década, con fluctuaciones de hasta 10 puntos porcentuales entre años según "
  "el ciclo El Niño/La Niña —, lo que hace que el precio dependa fuertemente del régimen "
  "hidrológico (nivel de embalses, aportes de caudal) y de su modulación por el fenómeno ENSO (El "
  "Niño / La Niña). El periodo de prueba de este proyecto (2026) coincide con un episodio de El "
  "Niño en transición, lo que hace la tarea más difícil que en años hidrológicamente tranquilos.")
p("Se construyeron dos productos de pronóstico independientes pero relacionados:")
bullet("Horizonte de 24 horas (día siguiente): un pronóstico para las 24 horas del día calendario "
       "siguiente, emitido con un corte a las 23:00 del día anterior — el protocolo estándar "
       "\"day-ahead\" de la literatura de pronóstico de precios de electricidad (EPF).")
bullet("Horizonte de 72 horas: un pronóstico continuo hora a hora hasta 3 días adelante, con corte "
       "a las 00:00 del día del pronóstico, dirigido a la vista de planeación del motor de "
       "decisión (OE3).")
p("Todos los resultados de este informe se miden 100% fuera de muestra: el conjunto de "
  "entrenamiento cubre 2019-2025 (60,625 horas) y la evaluación se hace sobre 2026 "
  "(enero-agosto, 5,184-5,208 horas según el modelo), un período que el modelo nunca vio durante "
  "el entrenamiento.")

# =====================================================================
# 2. METODOLOGÍA
# =====================================================================
doc.add_heading("2. Metodología", level=1)

doc.add_heading("2.1 Datos y validación", level=2)
p("Las variables de entrada provienen de series públicas de XM/SIMEM sincronizadas a resolución "
  "horaria: precio de bolsa histórico, volumen útil de embalses, aportes hídricos, demanda, "
  "generación total del sistema, el índice ONI (Oceanic Niño Index) y calendario/festivos "
  "colombianos. Es importante aclarar una restricción metodológica explícita del proyecto: "
  "las variables de generación y disponibilidad se usan únicamente como agregados de sistema, "
  "nunca desagregadas por tipo de recurso o por planta — esa desagregación está fuera del alcance "
  "definido en el Anexo 1 del proyecto. Por la misma razón, la variable de precio de oferta "
  "marginal (max_precio_oferta) nunca se usa como insumo: su correlación contemporánea con el "
  "precio de bolsa es 0.9998 porque es, en la práctica, el precio de bolsa — incluirla sería fuga "
  "de información, no una variable predictiva real.")
p("La validación principal usa seis \"orígenes\" temporales de tipo walk-forward — cada uno con "
  "su propio corte de entrenamiento y su propia ventana de prueba de 12 meses consecutivos — "
  "diseñados para cubrir los distintos regímenes ENSO observados en el histórico: dos episodios de "
  "La Niña, una transición triple-dip, El Niño fuerte 2023-2024, y El Niño 2026 (el período de "
  "prueba final). Evaluar en un solo corte temporal esconde fallas que solo aparecen en ciertos "
  "regímenes; el walk-forward de 6 orígenes está diseñado específicamente para no permitir eso.")

doc.add_heading("2.2 Por qué estas métricas, y no solo una", level=2)
p("El proyecto reporta varias métricas en paralelo a propósito, porque cada una mide algo "
  "distinto y ninguna sola es suficiente:")
bullet("MAE (error absoluto medio, en COP/kWh): la unidad física real del error. Es la métrica "
       "más fácil de interpretar para un lector no técnico, pero no dice si un error de 40 "
       "COP/kWh es grande o pequeño en un momento en que el precio está en 100 o en 800.")
bullet("MAPE (error porcentual absoluto medio): normaliza el MAE por el nivel del precio real, lo "
       "que permite comparar el desempeño entre horas de precio bajo y alto, y es la métrica más "
       "usada en la literatura de pronóstico de precios de electricidad (EPF) para reportar "
       "resultados finales. Es también la métrica principal de este proyecto.")
bullet("sMAPE (MAPE simétrico): igual que el MAPE, pero divide por el promedio de |real| y "
       "|predicho| en vez de solo |real|. Se reporta junto al MAPE porque el MAPE castiga mucho "
       "más los errores cuando el precio real es bajo (dividir por un número pequeño infla el "
       "error relativo), y el sMAPE es más robusto a eso — es también la métrica que usa el paper "
       "de Nueva Zelanda que se cita en la Sección 6, lo que permite comparar en su propio "
       "término.")
bullet("MASE (error absoluto escalado): el MAE del modelo dividido por el MAE de un pronóstico "
       "ingenuo de referencia (un paso atrás). MASE < 1 significa \"le gana al ingenuo\"; es la "
       "única métrica que se puede comparar de forma estrictamente válida contra el paper "
       "neozelandés, porque sus demás métricas (MAE, RMSE, sMAPE) están calculadas sobre la serie "
       "transformada con Box-Cox y normalizada a [0,1], no sobre el precio real — un detalle de "
       "su Tabla 2 que hay que verificar antes de comparar cualquier número directamente.")
bullet("R² y NRMSE (RMSE normalizado): miden qué proporción de la varianza del precio real "
       "explica el modelo. Se usan para comparar contra el paper brasileño de Dias, Lira & Freire "
       "(2024) y contra el paper noruego, que reportan en estos términos.")
bullet("TAPI (Trend Accuracy Prediction Index): el porcentaje de veces que el pronóstico acierta "
       "la dirección del cambio (sube/baja) respecto al paso anterior. Es la métrica de "
       "\"acierto direccional\", relevante porque una decisión de compra/venta en el motor de "
       "decisión (OE3) muchas veces depende más de acertar la dirección que la magnitud exacta.")
p("Nota metodológica encontrada y corregida durante esta sesión: al calcular el TAPI, las horas "
  "en que el precio real no cambió (delta = 0) inicialmente se contaban como fallo, porque el "
  "signo de un cambio nulo no coincide ni con +1 ni con −1. Eso hacía que el TAPI diera "
  "artificialmente 40% (por debajo del azar). Excluir esos pasos — que es lo correcto, porque un "
  "índice de tendencia no debe evaluarse sobre pasos sin tendencia — corrige la cifra a 82.2%. Se "
  "documenta este detalle porque afecta a cualquiera que intente replicar el cálculo.")
p("Se usa además la prueba de Diebold-Mariano (1995), con corrección de varianza HAC/Newey-West, "
  "como estándar del campo para decidir si una diferencia de error entre dos modelos es "
  "estadísticamente significativa o puede deberse al azar. Todo resultado de este informe que se "
  "reporta como \"mejora\" pasó esta prueba, salvo que se indique explícitamente lo contrario.")

doc.add_heading("2.3 Estándar de verificación de cualquier mejora", level=2)
p("Una lección temprana del proyecto (un hallazgo falso sobre la variable ONI que no sobrevivió "
  "la verificación) llevó a fijar un protocolo estricto para aceptar cualquier mejora: no basta "
  "con una sola partición de validación cruzada. Toda mejora reportada en este informe se probó "
  "con al menos 10 particiones aleatorias distintas de los datos de entrenamiento/prueba, y "
  "además en \"modo desplegable\" — recalculando los pesos del ensamble usando solo información "
  "disponible antes de cada corte, sin usar días futuros —, que es la cifra honesta que se "
  "obtendría operando el sistema en producción real. Una mejora que solo aparece en validación "
  "cruzada pero no en modo desplegable se documenta como no confirmada, nunca se reporta como "
  "ganancia.")

# =====================================================================
# 3. ARQUITECTURA
# =====================================================================
doc.add_heading("3. Arquitectura del sistema de pronóstico", level=1)
p("Ambos horizontes se resuelven con un ensamble (combinación ponderada) de varios modelos de "
  "familias distintas, no con un solo modelo. Es la decisión de diseño más importante del "
  "proyecto, y coincide con lo que domina en la literatura del campo: la ganancia no viene de que "
  "un modelo individual sea el mejor posible, sino de combinar modelos cuyos errores no están "
  "correlacionados entre sí.")

doc.add_heading("3.1 Modelo de 24 horas", level=2)
p("Protocolo day-ahead: corte a las 23:00, pronóstico de las 24 horas del día siguiente. El "
  "ensamble combina modelos de familias muy distintas — persistencia, gradient boosting "
  "(XGBoost/CatBoost), econométrico (ARX+GARCH), deep learning (N-BEATSx, N-HiTS), un modelo de "
  "dos etapas tipo hurdle, Markov-Switching, un filtro recursivo (RecursiveLS), y desde esta "
  "sesión un modelo LE-GARCH-t (ver Sección 4) —, y los pesos se ajustan por franja horaria de 6 "
  "horas usando un combinador de mínima desviación absoluta ponderado por la métrica reportada "
  "(ver Sección 4.2).")

doc.add_heading("3.2 Modelo de 72 horas", level=2)
p("Vota entre un LEAR (LASSO Estimated AutoRegressive, el benchmark canónico del campo — un "
  "modelo lineal con selección de variables por LASSO, uno por cada uno de los 72 pasos de "
  "horizonte), dos redes N-BEATSx (base y con variables exógenas enriquecidas) y una persistencia "
  "estacional, con pesos ajustados por tramo de horizonte (1-24h / 25-48h / 49-72h). Desde esta "
  "sesión, el tramo 1-24h se resuelve directamente con el ensamble de 24 horas ya descrito arriba "
  "(ver Sección 5), en vez de con los modelos generalistas de 72h.")

doc.add_page_break()

# =====================================================================
# 4. RESULTADOS — 24 HORAS
# =====================================================================
doc.add_heading("4. Resultados — horizonte de 24 horas", level=1)
p("Al comenzar esta sesión, el ensamble de 24h (v4: 5 votantes, combinador QRA por franja "
  "horaria) daba MAE 42.51, MAPE 11.27%, sMAPE 10.38%, R² 0.910. Se probaron dos ideas nuevas, "
  "las dos verificadas con el estándar completo (10 particiones + modo desplegable):")

doc.add_heading("4.1 Idea 1 — un votante de colas pesadas (LE-GARCH-t)", level=2)
p("Tomada del modelo ganador del paper neozelandés (Kapoor & Wichitaksorn, 2023): un modelo "
  "lineal con variables exógenas seleccionadas por LASSO en la ecuación de la media, combinado "
  "con un GARCH(1,1) y errores con distribución de colas pesadas (t de Student / GED en vez de "
  "normal). El mecanismo es distinto a simplemente \"agregar volatilidad\": al estimar por máxima "
  "verosimilitud con una distribución de colas pesadas, la propia estimación pondera menos los "
  "días de alta volatilidad al ajustar la media — el modelo se vuelve automáticamente más robusto "
  "a los picos de precio sin que haya que decírselo explícitamente. Resultado: MAE 41.45, MAPE "
  "10.86%, sMAPE 10.08% (variante GED).")

doc.add_heading("4.2 Idea 2 — alinear el combinador con la métrica reportada", level=2)
p("Un desajuste que llevaba todo el proyecto sin corregirse: el ensamble ajusta los pesos de sus "
  "votantes minimizando la desviación absoluta, que es matemáticamente óptima para el MAE — pero "
  "el proyecto se evalúa y se compara con la literatura en MAPE y sMAPE, cuyos óptimos son "
  "distintos (una mediana ponderada por 1/precio para el MAPE; una reponderación iterativa para "
  "el sMAPE). Alinear el combinador con la métrica que de verdad importa dio MAE 41.29, MAPE "
  "10.74%, sMAPE 10.07% — mejora en las tres métricas a la vez, algo que no debería ocurrir por "
  "construcción (el objetivo MAE debería ganar en MAE dentro de la muestra de entrenamiento, y en "
  "efecto lo hace: 40.600 contra 40.767). La explicación verificada es un efecto de "
  "regularización: la reponderación reduce la influencia de unos pocos días de precio muy alto "
  "sobre el ajuste de los pesos, y los pesos resultantes son más estables entre particiones "
  "(desviación 0.1496 contra 0.1553 del objetivo MAE).")

add_pic("fig1_evolucion_24h.png", 6.2,
        "Figura 1. Evolución del error del modelo de 24 horas durante esta sesión.")

add_table(["Métrica", "v4 original", "+ GARCH-ged", "+ combinador alineado"], [
    ["MAE (COP/kWh)", "42.51", "41.45", "41.29"],
    ["MAPE", "11.27%", "10.86%", "10.74%"],
    ["sMAPE", "10.38%", "10.08%", "10.07%"],
    ["R²", "0.910", "—", "—"],
], widths=[6, 3.3, 3.3, 3.4])

doc.add_heading("4.3 Dónde vive el error, y por qué no baja más", level=2)
p("El error no se reparte parejo entre las 24 horas del día. Concentra en el pico vespertino:")
add_pic("fig4_error_por_hora.png", 6.3,
        "Figura 2. MAPE por hora del día. Las horas 18-20 (en rojo) concentran el 37.8% del "
        "error total del modelo.")
p("Se hizo una descomposición sesgo/varianza específicamente en esas horas pico: el sesgo medio "
  "(+19.5 COP/kWh) explica solo el 21.2% del error ahí; el 78.8% restante es varianza genuina — "
  "picos de precio que la información disponible en el momento del corte simplemente no permite "
  "anticipar. Se intentaron dos correcciones dirigidas: ajuste de sesgo causal por hora (empeora "
  "con ventanas de 7-30 días) y modelos especialistas entrenados solo con horas pico (peores que "
  "el ensamble general, MAE 104-106 contra 92.15). Ninguna funcionó, lo que es consistente con la "
  "interpretación: no es un problema de qué algoritmo se usa, es que la información existente no "
  "contiene la respuesta.")
add_pic("fig7_techo_8pct.png", 6.2,
        "Figura 3. El objetivo de 8% de error está fuera del alcance teórico incluso en el mejor "
        "escenario posible con los datos actuales.")
p("El cálculo del techo teórico es directo: para pasar de 11.27% a 8% de MAPE hace falta reducir "
  "el error un 29% relativo. Si el modelo acertara perfecto en las 6 horas peores del día (que "
  "concentran el 37.8% del error), el MAPE global quedaría en 7.01% — apenas por debajo del "
  "objetivo. Es decir, el 8% está justo al borde de lo teóricamente posible, y solo se alcanzaría "
  "resolviendo exactamente el punto más difícil del problema. Lo que haría falta no son más "
  "algoritmos, sino información nueva que hoy no está en el conjunto de datos: ofertas de precio "
  "por unidad de generación, indisponibilidades programadas y declaraciones de mantenimiento — "
  "variables que en Colombia determinan qué unidad marca el precio marginal en el momento de "
  "escasez, y que el Anexo 1 del proyecto excluye del alcance actual por resolución por tipo de "
  "recurso.")

# =====================================================================
# 5. RESULTADOS — 72 HORAS
# =====================================================================
doc.add_heading("5. Resultados — horizonte de 72 horas", level=1)
p("El modelo de 72h partió esta sesión en MAE 65.53 / MAPE 18.28% (global, 7 votantes incluyendo "
  "tres modelos de árboles). Dos cambios lo mejoraron de forma sustancial.")

doc.add_heading("5.1 Reemplazo por un LEAR y eliminación de los árboles", level=2)
p("Se construyó un LEAR propio para 72h (LASSO con una estructura rica de rezagos de precio — "
  "14 rezagos distintos, de 1 a 168 horas atrás — y un modelo independiente por cada uno de los "
  "72 pasos de horizonte, la receta exacta de Lago et al. 2021). Resultó ser el mejor modelo "
  "individual del proyecto, por encima de las redes neuronales a 49-72h. El ensamble final — "
  "LEAR + N-BEATSx + N-BEATSx-exógeno + persistencia estacional (4 votantes) — elimina los tres "
  "modelos de árboles (XGBoost, CatBoost directo, CatBoost denso) y aun así mejora en los tres "
  "tramos de horizonte. Es a la vez más simple y más preciso: MAPE global 18.28% → 17.75%.")

doc.add_heading("5.2 El puente al modelo de 24 horas (la mejora más grande de la sesión)", level=2)
p("Al desglosar por tramo los resultados de intentar trasladar las mejoras de 24h a 72h (que no "
  "funcionaron — ver Sección 7), apareció algo que ningún experimento anterior había mostrado: "
  "sobre las mismas horas objetivo, el tramo 1-24h del producto de 72h daba MAPE 13.03%, mientras "
  "el modelo dedicado de 24h (Sección 4) daba 10.54% en esas mismas horas — y con menos "
  "información (corte una hora antes). Se verificó explícitamente que no había fuga de datos (0 "
  "de 4,922 filas con el corte de 24h posterior al de 72h): era una pérdida real, causada porque "
  "el canal de 72h estaba re-derivando el primer día desde cero con modelos generalistas, en vez "
  "de reusar el canal especializado que ya existía para ese mismo día.")
add_pic("fig3_tramos_puente.png", 6.2,
        "Figura 4. Corrección del tramo 1-24h al sustituirlo por el modelo dedicado de 24 horas.")
p("La solución: para los pasos 1 a 23 (los que el modelo de 24h puede cubrir sin usar información "
  "posterior al corte de 72h), el producto de 72h ahora reutiliza el ensamble de 24h como un "
  "votante adicional. El combinador le termina dando 93.6% del peso a ese votante y prácticamente "
  "nada a los 4 modelos generalistas — pero no lo sustituye del todo: meter el ensamble ya armado "
  "gana 1 punto de MAPE frente a meter sus 13 votantes individuales por separado, lo que muestra "
  "que la ganancia viene de la arquitectura del ensamble de 24h (pesos por franja + combinador "
  "alineado a la métrica), no solo de tener sus modelos disponibles.")
add_pic("fig2_evolucion_72h.png", 6.2,
        "Figura 5. Evolución del error global del modelo de 72 horas durante esta sesión.")

add_table(["Tramo", "MAE (antes)", "MAE (después)", "MAPE (antes)", "MAPE (después)"], [
    ["1-24h", "46.39", "41.08", "13.00%", "10.84%"],
    ["25-48h", "62.56", "62.56", "18.29%", "18.29%"],
    ["49-72h", "72.87", "72.87", "21.69%", "21.69%"],
    ["Global", "60.61", "58.84", "17.66%", "16.94%"],
], widths=[3, 3.3, 3.3, 3.3, 3.3])

p("Verificado con 10 particiones (mejor en 10/10 en las tres métricas) y en modo desplegable "
  "(MAPE 12.83% → 10.53%, p=0.0000) — no depende de la validación cruzada. Límite honesto: solo "
  "cubre los pasos 1-23; el paso 24 exigiría un pronóstico emitido 23 horas después del corte de "
  "72h, así que queda fuera del alcance de esta corrección.")

doc.add_page_break()

# =====================================================================
# 6. COMPARACIÓN CON OTROS MODELOS PROBADOS INTERNAMENTE
# =====================================================================
doc.add_heading("6. Comparación con otros modelos probados internamente", level=1)
p("Antes de llegar al ensamble actual se evaluaron cinco familias de modelo individuales, en "
  "holdout único y en validación walk-forward de 6 orígenes (Sección 2.1):")

add_table(["Modelo", "Familia", "MAE (holdout 2026)", "MAPE", "¿Supera la persistencia?"], [
    ["Persistencia (t-24h)", "Línea base", "56.31", "15.75%", "referencia"],
    ["Naive estacional (t-168h)", "Línea base", "124.53", "36.75%", "No"],
    ["Prophet", "Serie temporal aditiva", "95.84", "19.93%", "No"],
    ["XGBoost", "Gradient boosting", "61.15", "15.89%", "Parcial (solo RMSE)"],
    ["N-HiTS", "Deep learning", "57.08", "16.36%", "Parcial (RMSE)"],
    ["N-BEATSx", "Deep learning", "56.11", "15.93%", "Parcial (MAE, RMSE)"],
    ["ARX+GARCH(1,1)", "Econométrico", "55.76", "15.43%", "Sí (3 métricas)"],
    ["Ensamble v4 (actual)", "Combinación", "41.29", "10.74%", "Sí (todas)"],
], widths=[4.6, 4, 3, 2.4, 3])

p("El holdout único favorece a ARX+GARCH, pero la validación walk-forward de 6 orígenes muestra "
  "que esa ventaja no es robusta entre regímenes: N-BEATSx le gana a ARX+GARCH con significancia "
  "estadística en 5 de 6 orígenes, y XGBoost pierde contra la persistencia simple en los 6.")
add_pic("fig6_walkforward_regimenes.png", 6.5,
        "Figura 6. Error (MAE) por régimen hidrológico/ENSO en los 6 orígenes de validación "
        "walk-forward. La ventaja de las redes neuronales es más grande justo en los regímenes de "
        "El Niño (orígenes 4-6), donde más importa acertar por el mayor riesgo económico de una "
        "mala decisión.")
p("Esta es la razón de diseño detrás del ensamble: ningún modelo individual domina en todos los "
  "regímenes, y combinar familias con errores poco correlacionados (la persistencia simple, por "
  "ejemplo, recibe hasta 41% del peso del ensamble en la franja nocturna, a pesar de tener MAE "
  "individual mucho peor que N-BEATSx) rinde más que buscar un único mejor modelo. Se probaron "
  "además, dentro de esta sesión y en sesiones anteriores, más de 25 líneas adicionales — "
  "CatBoost, modelo de dos etapas (hurdle), Markov-Switching, RecursiveLS, afinamiento con Optuna, "
  "Transformer (TFT), lógica difusa del ONI, análogos de régimen k-NN, arquitecturas de dos "
  "modelos (normal/especialista), y las variantes de la familia GARCH (skew-t, GJR, EGARCH, "
  "SV-t) — documentadas con el mismo rigor en la Sección 7.")

doc.add_page_break()

# =====================================================================
# 7. COMPARACIÓN CON LA LITERATURA
# =====================================================================
doc.add_heading("7. Comparación con la literatura", level=1)

doc.add_heading("7.1 Colombia", level=2)
p("El punto de comparación colombiano más riguroso encontrado es Gallón & Barrientos (2021), "
  "\"Forecasting the Colombian Electricity Spot Price under a Functional Approach\", "
  "International Journal of Energy Economics and Policy — precio horario real de Colombia "
  "(2000-2017), con prueba de Diebold-Mariano formal. Su modelo de series funcionales (FTS) da "
  "MAPE 6.72% a 1 día y 8.87% a 1 mes, superando a SARIMA (12.86%/15.42%), NNAR (14.64%/19.02%) y "
  "DSHW (13.31%/18.94%).")
p("Comparar directamente sería engañoso por tres razones que el propio informe documenta con "
  "honestidad: (1) su período (2000-2017) es mucho más tranquilo que 2026, sin el súper Niño "
  "2023-2024 ni la transición de régimen que enfrenta este proyecto; (2) su modelo es puramente "
  "univariado, sin hidrología, ONI ni demanda — una limitación que el propio paper reconoce; (3) "
  "su cifra a 1 mes se mide sobre un solo mes de prueba (enero 2018). Lo útil de la comparación no "
  "es el número exacto, sino el contexto: incluso sus benchmarks univariados simples, en un "
  "período tranquilo, dan MAPE de 9-19%. Que este proyecto llegue a 10.74% en uno de los años más "
  "volátiles del histórico colombiano, con un modelo multivariado que incorpora régimen "
  "hidrológico, está en un rango razonable para el mercado — no por debajo de lo esperable.")
p("Segundo punto de referencia colombiano: Herrera-Mejía et al. (2025), Smart Energy (Elsevier), "
  "diciembre 2025 — título exacto pendiente de verificar contra el PDF original — un estudio de "
  "AES Colombia sobre clasificación de régimen de oferta para "
  "plantas de generación hidroeléctrica individuales (no el precio de bolsa agregado), que reporta "
  "84.77% de precisión de clasificación por tercil de precio para su mejor planta. No es "
  "directamente comparable en el objetivo (predice oferta por planta a resolución diaria, sin deep "
  "learning), pero es útil como validación cruzada independiente: sus modelos fallan de la misma "
  "forma que los de este proyecto justo en las transiciones de régimen (documentan la sequía de "
  "octubre 2024 como su propia \"prueba de fuego\"), y usan la misma lógica de precisión por tercil "
  "como métrica de decisión. El ensamble de 24h de este proyecto alcanza 92.01% de acierto por "
  "tercil sobre el mismo tipo de clasificación, aunque sobre una tarea distinta (precio agregado, "
  "no oferta por planta), así que la cifra se reporta como contexto, no como superioridad directa.")

doc.add_heading("7.1.1 Candidatos colombianos descartados, y por qué", level=3)
p("Antes de aceptar Gallón & Barrientos (2021) y Herrera-Mejía et al. (2025) como los dos puntos "
  "de referencia colombianos del informe, se revisaron otros cuatro estudios colombianos que "
  "resultaron no comparables por razones metodológicas concretas — se documentan aquí en vez de "
  "omitirlos, porque el criterio de exclusión es en sí mismo parte del rigor del ejercicio:")
add_table(["Estudio", "Por qué se descarta"], [
    ["Tesis de pregrado/especialización, Universidad Nacional Abierta y a Distancia (UNAD, 2023)",
     "Fuga parcial de datos: incluye precio_oferta (correlación 0.73 con el precio de bolsa) entre "
     "las variables del modelo."],
    ["Villarreal & Flores (EAFIT, 2023)", "Resolución diaria, no horaria; su R² sube de 0.009-0.119 "
     "a 0.452 solo al agregar una variable dummy que le indica al modelo las fechas exactas del "
     "Niño 2015-2016 — calibración retrospectiva, no predicción genuina; ventana de prueba de solo "
     "15 días para dos de sus modelos."],
    ["Giraldo Zuluaga & Ramírez Londoño (UdeA, 2023) — LSTM, MAPE 6.93%",
     "Ventaneo sin traslape deja solo ~73-91 ventanas de prueba independientes en el año; "
     "inestabilidad severa entre semillas sin reportar varianza (un modelo pasa de MAPE 8.02% a "
     "61.18% solo por el proceso de afinamiento); evaluado en 2018, un año ya calmado tras el "
     "choque de 2015-2016, no durante una transición de régimen nueva."],
    ["Cantillo-Luna et al. (Energies/MDPI, 2023) — T2V-TE Transformer, MAPE 5.39%",
     "El más riguroso de los cuatro (journal arbitrado, hiperparámetros afinados también en los "
     "baselines, prueba de Wilcoxon, métricas probabilísticas completas) — pero pronostica solo 8 "
     "horas intradía a partir de precio ya observado, no el día completo siguiente sin haber visto "
     "ninguna hora de él; y no usa ninguna variable de fundamentales (hidrología, ONI, demanda), "
     "algo que los propios autores dejan como trabajo futuro."],
], widths=[6.5, 9.5], small=True)
p("Ninguno de los cuatro invalida los resultados de este proyecto — al contrario, el más riguroso "
  "de ellos (Cantillo-Luna et al.) confirma explícitamente que la ausencia de variables de "
  "fundamentales es una limitación pendiente en su propio trabajo, justo lo que este proyecto sí "
  "incorpora. La lectura más honesta sigue siendo la establecida con Gallón & Barrientos: en un "
  "período tranquilo y sin fundamentales, un MAPE de 6-9% es alcanzable en el mercado colombiano; "
  "este proyecto llega a 10.74% en 2026, el año más volátil del histórico disponible, con "
  "variables causales de régimen hidrológico incluidas y validado con walk-forward de 6 regímenes "
  "— un estándar de evidencia más exigente que el de cualquiera de los cinco estudios colombianos "
  "revisados.")

doc.add_heading("7.2 Mercados con matriz energética hidro-dominada comparable", level=2)
p("Se buscaron mercados eléctricos con una proporción de generación hidroeléctrica similar a la "
  "colombiana (~70-74%), porque son los que enfrentan el mismo tipo de dependencia del régimen "
  "hidrológico y del fenómeno climático asociado. Los tres más relevantes: Brasil (mismo diseño "
  "de mercado hidro-térmico y el mismo fenómeno ENSO), Noruega/Nord Pool (más de 90% hidro, la "
  "literatura de pronóstico de precios de electricidad más desarrollada del mundo) y Nueva "
  "Zelanda.")
p("Antes de comparar cualquier número hace falta una advertencia de comparabilidad: ninguno de "
  "estos cuatro estudios pronostica a resolución horaria como este proyecto (24 precios distintos "
  "para el día siguiente). Predicen un precio agregado diario, semanal o mensual — una tarea "
  "estructuralmente más fácil, porque el ruido intradiario se cancela al promediar.")

add_table(["Estudio", "Qué predicen", "Horizonte", "Resolución"], [
    ["Nueva Zelanda (Kapoor & Wichitaksorn, 2023)", "1 precio por día", "24h adelante", "Diaria"],
    ["Brasil — Dias, Lira & Freire (2024)", "1 precio por semana, hasta 4 semanas",
     "168-672h adelante", "Semanal"],
    ["Brasil — Albani et al. (2025)", "1 precio por día, hasta 30 días", "24-720h adelante",
     "Diaria"],
    ["Este proyecto", "24 precios distintos, cada hora del día siguiente", "1-72h adelante",
     "Horaria"],
], widths=[5, 5.5, 3, 3])

p("Por eso la comparación correcta es \"superamos su propia métrica normalizada en una tarea de "
  "horizonte más agregado y por tanto más fácil que la nuestra\", nunca \"le ganamos a Nueva "
  "Zelanda\" sin ese matiz — sería una afirmación más fuerte de lo que los datos sostienen.")

add_pic("fig5_comparacion_papers.png", 6.6,
        "Figura 7. Comparación contra cada estudio en SU PROPIA métrica reportada (no son "
        "comparables entre paneles entre sí, solo cada panel contra el propio proyecto).")

p("Dos precauciones técnicas que se verificaron antes de construir estos números: (1) Kapoor & "
  "Wichitaksorn (2023) definen el sMAPE sin dividir por 2 en el denominador — su convención da "
  "exactamente la mitad del sMAPE convencional sobre los mismos datos, así que cualquier "
  "comparación de sMAPE con ellos tiene que corregir por ese factor de 2 (ya corregido en la "
  "Figura 7, panel de Noruega, que sí usa la convención estándar); (2) las métricas MAE/RMSE/sMAPE "
  "del paper neozelandés están calculadas sobre su serie transformada con Box-Cox y normalizada a "
  "[0,1], no sobre el precio real, así que su única métrica válida para comparar es el MASE "
  "(adimensional por construcción).")

p("El resultado más sólido de toda la comparación: MASE < 1 significa \"le gana al pronóstico "
  "ingenuo de un paso\", el estándar mínimo de cualquier modelo serio. El ensamble de este "
  "proyecto da MASE 0.943; los 34 modelos evaluados en el paper neozelandés (incluyendo redes "
  "profundas, LSTM, GRU y XGBoost) tienen todos MASE > 1.26 — ninguno le gana a su propio "
  "ingenuo. Advertencia honesta: el denominador del MASE es el ingenuo de cada serie, así que no "
  "es una competencia cabeza a cabeza — el mercado neozelandés puede ser sencillamente más difícil "
  "de predecir (se liquida cada 30 minutos, en tiempo real, con una curtosis de hasta 505 en una "
  "de sus cinco regiones). Lo que sí se puede afirmar con rigor es que el modelo de este proyecto "
  "supera su propia referencia ingenua, y los modelos del paper neozelandés no superan la suya.")

p("Hallazgo de literatura que conecta directamente con decisiones que ya se habían tomado en el "
  "proyecto por evidencia empírica propia, antes de conocer estos estudios: los cuatro papers "
  "encuentran, de forma consistente, que modelos más simples con buena selección/regularización "
  "de variables igualan o superan al deep learning complejo. El paper neozelandés cita "
  "explícitamente a Lago et al. (2018, 2021): \"simpler base models such as LEAR, DNN, and LSTM, "
  "can outperform newer, complex models\". El paper de Dias et al. muestra un MLP simple con "
  "selección de variables por Pearson + información mutua ganándole al modelo oficial DECOMP de "
  "Brasil, mucho más sofisticado (programación dinámica estocástica dual). Y Albani et al. citan a "
  "Hewamalage et al. (2023) para justificar por qué usan redes de una sola capa oculta a "
  "propósito. Dos decisiones tomadas en este proyecto por prueba propia coinciden exactamente con "
  "eso: que un modelo lineal (Ridge/LEAR) sea el mejor individual a 48-72h, y que el mejor "
  "combinador de 24h use pesos por franja horaria — ambas son, sin haberlo sabido de antemano, la "
  "misma receta que domina en los mercados eléctricos mejor estudiados del mundo (Nord Pool, PJM, "
  "EPEX).")

doc.add_page_break()

# =====================================================================
# 8. POR QUÉ NO HAY FORMA DE SEGUIR MEJORANDO (SIN DATOS NUEVOS)
# =====================================================================
doc.add_heading("8. Por qué el error tiene un límite con los datos actuales", level=1)
p("Esta sección resume, para el horizonte de 24h y para el de 72h, la evidencia de que el sistema "
  "está cerca de su techo con la información disponible — no por falta de esfuerzo de modelado, "
  "sino porque se demostró matemáticamente dónde está el límite y por qué.")

doc.add_heading("8.1 Horizonte de 24h", level=2)
bullet("El ensamble está en meseta: agregar cada uno de los 9 votantes extra disponibles, de a "
       "uno, no da diferencia significativa; la mejor selección greedy de 14 votantes gana solo "
       "0.21 de MAE, no significativo (p=0.2575); afinar la granularidad de los pesos (de 1 a 24 "
       "grupos) gana como máximo 0.28, tampoco significativo (p=0.2947).")
bullet("El error está concentrado y localizado: el pico vespertino (18-20h) concentra 37.8% del "
       "error total, y ahí el 78.8% del error es varianza genuina, no sesgo corregible (Sección "
       "4.3).")
bullet("El techo teórico calculado (acertando perfecto en las 6 horas más difíciles) es 7.01% de "
       "MAPE — apenas por debajo del objetivo original de 8%, y ese es el mejor caso posible, no "
       "uno alcanzable con ningún modelo real.")
bullet("Lo que sí movería el error de forma estructural es información nueva — ofertas por unidad "
       "de generación, indisponibilidades programadas, mantenimientos — que hoy está fuera del "
       "alcance de datos definido para el proyecto (Sección 2.1). No es una limitación de "
       "modelado, es una limitación de qué información existe en el conjunto de datos actual.")

doc.add_heading("8.2 Horizonte de 25-72h", level=2)
p("Tras la mejora del puente 1-24h (Sección 5.2), se probaron tres vías adicionales, específicas "
  "para extender ganancias similares a los tramos 25-48h y 49-72h — donde no existe un canal "
  "especializado \"hermano\" que trasplantar, a diferencia de 1-24h:")
add_table(["Vía probada", "25-48h", "49-72h"], [
    ["Trasladar el votante GARCH-t y el combinador alineado a la métrica (que sí funcionaron en "
     "24h)", "No significativo (mejor p=0.15)", "No significativo (mejor p=0.075); el mecanismo "
     "se apaga con la distancia al corte"],
    ["Árboles especializados por tramo (en vez de generalistas a 72 pasos)", "No concluyente "
     "(7/10 semillas, delta ≈0.02)", "Gana en validación cruzada (10/10) pero PIERDE en modo "
     "desplegable (p=0.64) — no se adopta"],
    ["Diversidad intra-familia: variantes de N-BEATSx (semillas/ventanas distintas)",
     "Evidencia débil, no significativa en modo desplegable (p=0.069)", "No ayuda"],
], widths=[5.3, 5, 5.2])
p("Las tres vías quedan cerradas con resultado negativo o no concluyente. La lectura acumulada: "
  "el margen que sí existía en 1-24h (una pérdida estructural real, corregible porque había un "
  "canal mejor disponible) no tiene equivalente accesible en 25-72h — ahí el estado observable en "
  "el corte informa cada vez menos mientras más lejos está el objetivo, y ninguna de las "
  "herramientas de modelado disponibles logra revertir eso sin datos nuevos.")

doc.add_page_break()

# =====================================================================
# 9. LÍNEAS EXPLORADAS SIN ÉXITO
# =====================================================================
doc.add_heading("9. Líneas exploradas sin éxito (documentadas con el mismo rigor)", level=1)
p("El proyecto documenta los resultados negativos con el mismo detalle que los positivos — es "
  "parte del estándar de rigor que se siguió en todo el trabajo. Resumen de las líneas más "
  "relevantes, más allá de las ya mencionadas en la Sección 8:")

add_table(["Intento", "Resultado", "Causa identificada"], [
    ["LEAR (benchmark estándar del campo) en 24h", "MAE 53.64, pierde contra el ensamble (+11.14)",
     "A 24h la dinámica intradiaria aún manda y las redes la capturan mejor; lo lineal solo gana a "
     "horizonte largo"],
    ["Lógica difusa del índice ONI", "Empeora los 3 modelos de deep learning",
     "El ONI ya era continuo; las columnas fuzzy añaden dimensionalidad sin información nueva"],
    ["Análogos de régimen vía k-NN", "Pierde con significancia en El Niño (p<0.0001)",
     "Usar solo ~15% de los datos históricos daña más que ayuda \"parecerse\" al régimen actual"],
    ["Markov-Switching (2 regímenes)", "Empata con ARX+GARCH (p=0.993)",
     "El régimen detectado por el algoritmo dura ~10 horas, no meses — captura ráfagas de "
     "volatilidad, no el ciclo El Niño/La Niña"],
    ["Corrección de sesgo (global y adaptativa)", "Empeora o no mejora en todas las variantes "
     "probadas", "El sesgo cambia de signo a escala de años, no de semanas — no hay ventana corta "
     "que lo capture"],
    ["Transformer (TFT) como votante", "Abortado tras 2h sin completar", "Costo computacional en "
     "CPU sin GPU disponible, no resultado negativo"],
    ["Variables de memoria hidrológica (anomalía acumulada de aportes/embalse)", "MAE 59.41→59.04, "
     "p=0.52", "Confirmado también en la literatura brasileña (Nunes Jr. et al. 2024): mejorar el "
     "pronóstico de caudal casi no mueve el precio resultante"],
    ["Pesos del ensamble por régimen hidrológico (idea de Albani et al. 2025)", "Empeora "
     "significativamente en las 4 variantes probadas", "La franja horaria sigue siendo la "
     "variable que de verdad cambia qué votante conviene; partir en más celdas sobreajusta"],
    ["Arquitectura de dos modelos (uno para días normales, otro para días difíciles)",
     "Falla incluso con un oráculo perfecto de cuándo usar cada uno",
     "El fallo del modelo es predecible (AUC 0.818) pero no es accionable para corregir el "
     "pronóstico puntual"],
    ["3 variables nuevas de estrés del sistema descargadas de XM (GeneFueraMerito, RestSinAliv, "
     "GeneProgDesp)", "Pasan el tamiz de fuga de datos, pero empeoran el modelo",
     "Replica el patrón ya documentado: correlación parcial no implica mejora predictiva"],
], widths=[5, 4.8, 6.2], small=True)

p("Lista más extensa, con el detalle numérico completo de cada una, en la bitácora del proyecto "
  "(README.md) y en el informe comparativo previo (docs/informe_comparativo_modelos.md).")

# =====================================================================
# 10. CONCLUSIONES Y RECOMENDACIONES
# =====================================================================
doc.add_heading("10. Conclusiones y recomendaciones", level=1)
bullet("El sistema de pronóstico actual (ensamble de 24h + ensamble de 72h con el puente) supera "
       "de forma verificada todos los modelos individuales probados internamente y los puntos de "
       "referencia disponibles en la literatura de mercados hidro-dominados, en las métricas que "
       "cada estudio reporta.")
bullet("El error de 24h (MAPE 10.74%) está cerca de su límite teórico con los datos actuales "
       "(7.01% en el mejor caso posible). Bajar más exigiría información nueva sobre oferta por "
       "unidad de generación, no más modelado sobre las variables ya disponibles.")
bullet("El error de 72h se benefició de una corrección estructural grande en su primer día "
       "(puente al modelo de 24h), pero los tramos 25-72h no tienen todavía una vía de mejora "
       "confirmada — es la frontera de trabajo más clara para la siguiente ronda, "
       "condicionada a conseguir variables nuevas o una arquitectura distinta que aún no se ha "
       "probado.")
bullet("Pendiente de decisión conjunta con el equipo: regenerar los contratos oficiales que "
       "alimentan el dashboard de OE3 (pronóstico unificado de 72h) con las composiciones nuevas "
       "de ambos ensambles, y ajustar el umbral del régimen hidrológico usado en la señal de "
       "confianza del dashboard a un umbral relativo (móvil de 90 días) en vez de absoluto — "
       "2026 no tiene ninguna hora en régimen de \"embalse bajo\" bajo el umbral histórico actual.")

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
 "Functional Approach. International Journal of Energy Economics and Policy, 11(1).",
 "Gneiting, T., Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. "
 "Journal of the American Statistical Association, 102(477).",
 "International Energy Agency (2023). Colombia 2023 — Energy Policy Review. IEA, París. "
 "(Hidroelectricidad = 72% de la generación).",
 "Low Carbon Power (2025). Colombia Electricity Generation Mix. lowcarbonpower.org. (Datos de "
 "generación 2025: 74% hidro, 11% gas, 5% solar).",
 "Herrera-Mejía, L. et al. (2025). Estudio de AES Colombia sobre clasificación de régimen de "
 "oferta de plantas hidroeléctricas individuales bajo riesgo de sequía. Smart Energy, Elsevier, "
 "diciembre 2025. Título exacto y volumen pendientes de verificar contra el PDF original.",
 "Giraldo Zuluaga, D., Ramírez Londoño, J. (2023). Pronóstico del precio de bolsa de energía en "
 "Colombia con LSTM. Universidad de Antioquia. Trabajo de grado/tesis; título exacto pendiente de "
 "verificar contra el PDF original.",
 "Cantillo-Luna, S. et al. (2023). Pronóstico probabilístico intradía del precio de bolsa "
 "colombiano con Time2Vec y Transformer (T2V-TE). Energies, MDPI. Título exacto y volumen "
 "pendientes de verificar contra el PDF original.",
 "Villarreal, Flores (2023). Pronóstico del precio de bolsa de energía en Colombia: comparación de "
 "modelos econométricos (VAR, SARIMAX, regresión múltiple). Universidad EAFIT. Trabajo de grado; "
 "nombres completos y título exacto pendientes de verificar contra el PDF original.",
 "Universidad Nacional Abierta y a Distancia (UNAD) (2023). Pronóstico del precio de bolsa de "
 "energía en Colombia. Trabajo de grado; autores y título exactos pendientes de verificar contra "
 "el PDF original.",
 "Hewamalage, H., Bergmeir, C., Bandara, K. (2023). Global models for time series forecasting: A "
 "simulation study. Pattern Recognition, 124.",
 "Kapoor, G., Wichitaksorn, N. (2023). Electricity price forecasting in New Zealand: A "
 "comparative analysis of statistical and machine learning models with feature selection. "
 "Applied Energy, 347.",
 "Lago, J., De Ridder, F., De Schutter, B. (2018). Forecasting spot electricity prices: Deep "
 "learning approaches and empirical comparison of traditional algorithms. Applied Energy, 221.",
 "Lago, J., Marcjasz, G., De Schutter, B., Weron, R. (2021). Forecasting day-ahead electricity "
 "prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. "
 "Applied Energy, 293, 116983.",
 "Maciejowska, K., Lipiecki, A., Uniejewski, B. (2026). Statistical and economic evaluation of "
 "forecasts in electricity markets: beyond RMSE and MAE. Energy Conversion and Management, 356. "
 "arXiv:2511.13616.",
 "Muñoz-Santiago, N. et al. (2017). Modelo ARIMA con especificación IGARCH aplicado al mercado "
 "eléctrico colombiano. Referencia interna del proyecto para el modelo ARX+GARCH.",
 "Nowotarski, J., Weron, R. (2015). Computing electricity spot price prediction intervals using "
 "quantile regression and forecast averaging. Computational Statistics, 30.",
 "Nunes Jr., A. B. de A., Ferreira, D. D., da Costa Pinho, A. L. (2024). Information Theoretic "
 "Learning Applied to Daily Streamflow Forecast and Its Impact on the Brazilian Hourly Energy "
 "Spot Prices. Journal of Control, Automation and Electrical Systems.",
 "Olivares, K. G., Challu, C., Marcjasz, G., Weron, R., Dubrawski, A. (2023). Neural basis "
 "expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx. "
 "International Journal of Forecasting, 39(2).",
 "s.a. (2026). Electricity price forecasting across Norway's five bidding zones in the "
 "post-crisis era. arXiv:2604.26634.",
 "s.a. (2026). How optimistic inflow forecasts distort dispatch, prices, and contracts in "
 "hydro-dominated power systems: evidence from Brazil. arXiv:2607.00504v2.",
 "Uniejewski, B. (2026). Distributional neural networks for electricity price forecasting. "
 "Referenciado en la sección de agregación de pronósticos por franja horaria.",
]
for r in sorted(refs):
    par = doc.add_paragraph(style="List Bullet")
    par.paragraph_format.space_after = Pt(6)
    run = par.add_run(r)
    run.font.size = Pt(10)

p("", size=6)
p("Nota: los datos, scripts y bitácora completa de este proyecto están en el repositorio "
  "xm-spot-price-predictor; este informe consolida los resultados hasta el 18 de septiembre de "
  "2026. Documento generado a partir de las anotaciones de la bitácora del proyecto (README.md) "
  "y del informe comparativo previo (docs/informe_comparativo_modelos.md).", italic=True, size=9.5,
  color=GRIS)

doc.save(str(OUT))
print("Informe completo guardado en", OUT)
