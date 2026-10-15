# -*- coding: utf-8 -*-
"""
Informe final (plantilla ABET) a partir del informe ABET del 30-sep (docs/Informe_Avance_ABET_Barcelo_Dede.docx), que se
copia a docs/Informe_Final_ABET_Barcelo_Dede.docx y se edita ubicando cada parrafo y tabla por su texto, como
actualizar_informe_avance_oe3.py: se conservan la plantilla, los estilos, las ecuaciones nativas y el formato.
Las cifras salen de informe_final_assets/cifras.json (informe_final_resultados.py) y las figuras nuevas de
informe_final_figuras.py. Lo que depende del equipo (validacion con usuarios, poster, video) queda resaltado en amarillo.
Uso: python scripts_experimento/generar_informe_final.py
"""
import copy
import json
import re
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.shared import Inches
from docx.text.paragraph import Paragraph

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
from ecuaciones_omml import tabla_ecuacion  # noqa: E402

ORIGEN = RAIZ / "docs" / "Informe_Avance_ABET_Barcelo_Dede.docx"
DESTINO = RAIZ / "docs" / "Informe_Final_ABET_Barcelo_Dede.docx"
FIG = AQUI / "informe_final_assets"
C = json.loads((FIG / "cifras.json").read_text(encoding="utf-8"))
shutil.copyfile(ORIGEN, DESTINO)
doc = Document(str(DESTINO))
body = doc.element.body


# ------------------------------------------------------------------------------------------------ formato de numeros
def n(v, d=1):
    s = f"{abs(v):,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("−" if v < 0 else "") + s


def pc(v, d=1):
    return n(v, d) + " %"


def sg(v, d=1):
    return ("+" if v >= 0 else "") + n(v, d)


# ------------------------------------------------------------------------------------------------ edicion por texto
def texto_de(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def P(prefijo):
    for el in body.iterchildren():
        if el.tag == qn("w:p") and texto_de(el).strip().startswith(prefijo):
            return el
    raise LookupError(f"No encontre el parrafo: {prefijo!r}")


def md(texto):
    """**negrita**, *cursiva* y [[pendiente del equipo, resaltado]] -> segmentos (texto, negrita, cursiva, resaltado)."""
    segs = []
    for parte in re.split(r"(\*\*.+?\*\*|\[\[.+?\]\]|(?<!\*)\*[^*]+?\*(?!\*))", texto):
        if not parte:
            continue
        if parte.startswith("**"):
            segs.append((parte[2:-2], True, False, False))
        elif parte.startswith("[["):
            segs.append((parte[2:-2], False, False, True))
        elif parte.startswith("*"):
            segs.append((parte[1:-1], False, True, False))
        else:
            segs.append((parte, False, False, False))
    return segs


def reescribir(p_el, texto, conservar=0):
    """conservar: numero de runs iniciales que se dejan (la vineta y su tabulacion en las listas)."""
    rpr = None
    runs = p_el.findall(qn("w:r"))
    r0 = runs[conservar] if len(runs) > conservar else None
    if r0 is not None and r0.find(qn("w:rPr")) is not None:
        rpr = copy.deepcopy(r0.find(qn("w:rPr")))
        for tag in ("w:b", "w:bCs", "w:i", "w:iCs", "w:highlight"):
            for x in rpr.findall(qn(tag)):
                rpr.remove(x)
    for ch in list(p_el):
        if ch.tag != qn("w:pPr") and not (ch.tag == qn("w:r") and any(ch is r for r in runs[:conservar])):
            p_el.remove(ch)
    par = Paragraph(p_el, doc._body)
    for t, b, i, h in md(texto):
        run = par.add_run(t)
        if rpr is not None:
            run._r.insert(0, copy.deepcopy(rpr))
        if b:
            run.bold = True
        if i:
            run.italic = True
        if h:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return p_el


def R_(prefijo, texto):
    return reescribir(P(prefijo), texto)


def vineta(prefijo, texto):
    """parrafo de lista (vineta + tabulacion + texto) cuyo texto, despues de la tabulacion, empieza por `prefijo`."""
    for el in body.iterchildren():
        t = Paragraph(el, doc._body).text if el.tag == qn("w:p") else ""      # incluye las tabulaciones (w:tab)
        if "\t" in t and t.split("\t", 1)[1].startswith(prefijo):
            return reescribir(el, texto, conservar=2)
    raise LookupError(f"No encontre la vineta: {prefijo!r}")


def despues(ancla, plantilla, texto):
    nuevo = reescribir(copy.deepcopy(plantilla), texto)
    ancla.addnext(nuevo)
    return nuevo


def eliminar(el):
    el.getparent().remove(el)


def tabla_tras(el):
    x = el.getnext()
    while x is not None and x.tag != qn("w:tbl"):
        x = x.getnext()
    return x


def filas(tbl):
    return tbl.findall(qn("w:tr"))


def celdas(tr):
    return tr.findall(qn("w:tc"))


def celda(tc, texto):
    ps = tc.findall(qn("w:p"))
    for p in ps[1:]:
        tc.remove(p)
    p = ps[0]
    tiene_run = p.find(qn("w:r")) is not None
    if not tiene_run:
        Paragraph(p, doc._body).add_run("")
    reescribir(p, texto)


def fila(tr, valores):
    for tc, v in zip(celdas(tr), valores):
        if v is not None:
            celda(tc, v)


def n_filas(tbl, total):
    """deja la tabla con `total` filas (encabezado incluido), copiando la ultima o quitando del final."""
    trs = filas(tbl)
    while len(trs) < total:
        trs[-1].addnext(copy.deepcopy(trs[-1]))
        trs = filas(tbl)
    for tr in trs[total:]:
        tbl.remove(tr)


def llenar(tbl, matriz, desde=1):
    n_filas(tbl, desde + len(matriz))
    for tr, valores in zip(filas(tbl)[desde:], matriz):
        fila(tr, valores)


def forma(tbl, anchos):
    grid = tbl.find(qn("w:tblGrid"))
    for gc in list(grid):
        grid.remove(gc)
    for w in anchos:
        gc = grid.makeelement(qn("w:gridCol"), {qn("w:w"): str(w)})
        grid.append(gc)
    for tr in filas(tbl):
        tcs = celdas(tr)
        for _ in range(len(anchos) - len(tcs)):
            tr.append(copy.deepcopy(tcs[-1]))
        for tc in celdas(tr)[len(anchos):]:
            tr.remove(tc)
        for tc, w in zip(celdas(tr), anchos):
            tc.find(qn("w:tcPr")).find(qn("w:tcW")).set(qn("w:w"), str(w))


def imagen(cap_prefijo, archivo, ancho_in=None):
    cap = P(cap_prefijo)
    img_p = cap.getprevious()
    assert list(img_p.iter(qn("w:drawing"))), cap_prefijo
    if ancho_in is None:
        ext = next(img_p.iter("{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent"))
        ancho_in = int(ext.get("cx")) / 914400
    for ch in list(img_p):
        if ch.tag != qn("w:pPr"):
            img_p.remove(ch)
    Paragraph(img_p, doc._body).add_run().add_picture(str(FIG / archivo), width=Inches(ancho_in))
    return cap


def quitar_figura(cap_prefijo):
    cap = P(cap_prefijo)
    eliminar(cap.getprevious())
    eliminar(cap)


def renombrar(viejo, nuevo):
    p = P(viejo)
    resto = texto_de(p).strip()[len(viejo):]
    reescribir(p, f"**{nuevo}**{resto}")


# renumeracion previa, de mayor a menor para que los prefijos no se confundan: se quitan las Figuras 6, 8 y 9 del dashboard
# anterior y se abre espacio para la Tabla 16 (validacion historica)
quitar_figura("Figura 6. Selección")
quitar_figura("Figura 8.")
quitar_figura("Figura 9.")
for viejo, nuevo in (("Figura 7.", "Figura 6."), ("Figura 10.", "Figura 7."), ("Figura 11.", "Figura 8."),
                     ("Figura 12.", "Figura 9."), ("Figura 13.", "Figura 10."),
                     ("Tabla 19.", "Tabla 20."), ("Tabla 18.", "Tabla 19."), ("Tabla 17.", "Tabla 18."), ("Tabla 16.", "Tabla 17.")):
    renombrar(viejo, nuevo)

# ================================================================================================== cifras usadas
T24 = {f["modelo"]: f for f in C["tabla24"]}
FINAL = T24["Predespacho + 3 CatBoost (diseño definitivo)"]
PERS = T24["Persistencia (precio publicado más reciente)"]
PRE = T24["Predespacho ideal de XM"]
ENS6 = T24["Ensamble de 6 (diseño del informe de avance)"]
ENS7 = T24["Ensamble de 6 + predespacho"]
K = C["contrato24"]
T72 = {f["tramo"]: f for f in C["tabla72"]}
DIA = C["diario"]
OP = C["operacion"]
MO = C["motor"]
ALT = C["alternativas"]
red = (1 - K["MAE"] / K["MAE_pers"]) * 100
red_pre = (1 - K["MAE"] / K["MAE_pre"]) * 100
rn, rnino = C["regimen_normal_ene_abr"], C["regimen_el_nino_may_ago"]
EV = {(e["evento"], e["metodo"]): e for e in C["eventos"]}
peor_mes_real = max(C["rmae_mensual"].values())


def met(m):
    return "híbrido" if m == "hibrido" else m


# ================================================================================================== portada y seccion 1
R_("Septiembre de 2026", "Noviembre de 2026")
R_("Resumen.", (
    "**Resumen.** El precio de bolsa de la energía eléctrica en Colombia se fija cada hora sobre una matriz mayoritariamente "
    "hidroeléctrica [1] y puede multiplicarse durante semanas en presencia del fenómeno de El Niño. Se diseñó una plataforma que "
    "adquiere y sincroniza los datos públicos del mercado (XM) y el índice climático ONI (NOAA), los caracteriza en tiempo y "
    "frecuencia, pronostica el precio de las siguientes 24 a 72 horas con bandas de incertidumbre y traduce el pronóstico en "
    "recomendaciones de comprar, vender o esperar para generadores y comercializadores, en un dashboard publicado que se "
    "actualiza cada día. Al medir qué datos son públicos en el momento de pronosticar se encontró que XM publica el precio con "
    "unos tres días de rezago, lo que anulaba la ventaja del ensamble de seis modelos del diseño inicial; el diseño definitivo "
    "corrige el predespacho ideal que XM publica la víspera con tres modelos CatBoost reentrenados cada mes. La validación fue "
    "fuera de muestra, en 2026 y en cinco años históricos, con la prueba de Diebold-Mariano, y el motor de decisión se evaluó "
    f"por backtesting. Con solo información pública, el error absoluto medio a 24 h es {n(K['MAE'])} COP/kWh (MAPE "
    f"{pc(K['MAPE'])}), {n(red, 0)} % menor que el de la persistencia con el último precio publicado y {n(red_pre, 0)} % menor "
    f"que el del predespacho solo (p < 0,001); a 72 h es {n(T72['1-72 h']['MAE'])} COP/kWh. Las bandas cubren el "
    f"{pc(K['cobertura'])} de las horas (objetivo 80 %) y el motor obtiene ventaja positiva en las dos mitades del periodo "
    "para los dos roles y los dos horizontes."))
R_("Abstract.", (
    "**Abstract.** The hourly spot price of electricity in Colombia is set on a mostly hydroelectric mix and can multiply for "
    "weeks during El Niño. We designed a platform that acquires and synchronizes public market data (XM) and the ONI climate "
    "index (NOAA), characterizes them in the time and frequency domains, forecasts the price 24 to 72 hours ahead with "
    "uncertainty bands, and turns the forecast into buy, sell or wait recommendations for generators and retailers in a public "
    "dashboard updated every day. Measuring which data are public at forecast time showed that XM publishes the price with "
    "about a three-day lag, which erased the advantage of the initial six-model ensemble; the final design corrects the ideal "
    "pre-dispatch that XM publishes the day before with three CatBoost models retrained every month. It was validated out of "
    "sample in 2026 and in five historical years with the Diebold-Mariano test, and the decision engine was backtested. Using "
    f"only public information, the 24-hour mean absolute error is {K['MAE']:.1f} COP/kWh (MAPE {K['MAPE']:.1f} %), "
    f"{red:.0f} % lower than persistence with the latest published price and {red_pre:.0f} % lower than the pre-dispatch "
    f"alone (p < 0.001); at 72 hours it is {T72['1-72 h']['MAE']:.1f} COP/kWh. The bands cover {K['cobertura']:.1f} % of the "
    "hours (target 80 %) and the engine keeps a positive advantage in both halves of the period for both roles and horizons."))
R_("Presupuesto del Anexo 1 [2]", (
    "Presupuesto del Anexo 1 [2], con honorarios según ACIEM [3] y ejecución estimada al cierre del proyecto (semana 16), cuando "
    "se ejecuta también la divulgación. La publicación del dashboard (Streamlit Community Cloud) y su actualización diaria "
    "(GitHub Actions) son gratuitas para repositorios públicos, así que la operación no agrega costos. Financiación: $24.486.344 "
    "propia y $13.118.916 de la Universidad del Norte (detalle en el Anexo 6)."))
cap = R_("Tabla 1.", "**Tabla 1.** Presupuesto aprobado y ejecución estimada al cierre del proyecto (COP).")
t1 = tabla_tras(cap)
fila(filas(t1)[0], [None, None, "Ejecutado estimado al cierre", None])
for tr in filas(t1)[1:]:
    cs = celdas(tr)
    celda(cs[2], texto_de(cs[1]))
    celda(cs[3], "100,0 %")

# ================================================================================================== 2 introduccion
R_("Con este proyecto se construyó", (
    "Con este proyecto se construyó una plataforma reproducible que adquiere y sincroniza a resolución horaria las series "
    "públicas que caracterizan el MEM colombiano entre enero de 2019 y agosto de 2026, las caracteriza en tiempo y frecuencia "
    "para seleccionar variables predictoras y pronostica el precio con bandas de incertidumbre calibradas. Un motor de decisión "
    "traduce el pronóstico en señales de comprar, vender o esperar según el rol del usuario, visibles en un dashboard publicado "
    "en la web que se actualiza todos los días con los datos nuevos de XM. Durante el diseño se midió qué información es "
    "pública en el momento de pronosticar: el precio de bolsa se publica con unos tres días de rezago, mientras que el "
    "predespacho ideal, que XM calcula con las ofertas de los generadores para el día siguiente, sale la víspera. Esa medición "
    "cambió el diseño: el pronóstico definitivo corrige el predespacho con tres modelos de árboles CatBoost y reduce el error "
    f"absoluto medio (MAE) a 24 h en {n(red, 0)} % frente a la persistencia con el último precio publicado. El trabajo sigue las "
    "cinco fases del Anexo 1 [2] (adquisición y caracterización, modelado, comparación formal y motor de decisión, imágenes de "
    "decisión y validación), con evaluación estrictamente fuera de muestra y la prueba de Diebold-Mariano [7], siguiendo las "
    "buenas prácticas del área de pronóstico de precios de electricidad [6]."))
p = P("El valor práctico de la plataforma")
despues(p, p, (
    "El documento se organiza así: la Sección 3 presenta los objetivos; la 4, la delimitación del proyecto según el Anexo 1; "
    "la 5, el problema y el estado del arte; la 6, el diseño de ingeniería, desde los requerimientos y las alternativas hasta el "
    "diseño definitivo; la 7, las pruebas, sus resultados y su análisis; la 8, el impacto del proyecto, y la 9, las "
    "conclusiones y recomendaciones."))

# ================================================================================================== 3-4 objetivos y delimitacion
R_("Diseñar, implementar y validar una plataforma", (
    "Diseñar, implementar y validar una plataforma de procesamiento de datos que permita el análisis y pronóstico del precio de "
    "la energía en el mercado eléctrico colombiano para la toma de decisiones."))
for k, txt in (("OE1", "Adquirir, sincronizar y caracterizar temporal y espectralmente las series históricas de las variables del "
                       "mercado, mediante técnicas de procesamiento de datos, para seleccionar variables predictoras."),
               ("OE2", "Implementar y evaluar diferentes modelos de pronóstico del precio de bolsa, comparando precisión frente a "
                       "persistencia, robustez, costo computacional e interpretabilidad."),
               ("OE3", "Diseñar e implementar un motor de decisión que traduzca los pronósticos y su incertidumbre en señales de "
                       "compra, venta o espera."),
               ("OE4", "Validar el funcionamiento del motor para la toma de decisiones.")):
    vineta(f"{k}.", f"**{k}.** {txt}")
R_("La Tabla 2 transcribe los alcances", "La Tabla 2 transcribe los alcances del Anexo 1 [2] y la evidencia de su cumplimiento.")
cap = R_("Tabla 2.", "**Tabla 2.** Alcances por objetivo específico (Anexo 1) y evidencia de cumplimiento.")
t2 = tabla_tras(cap)
fila(filas(t2)[0], [None, None, "Cumplimiento"])
for tr, est in zip(filas(t2)[1:], [
        "Cumplido (Secciones 6.8 y 7.3); la adquisición se ejecuta cada día de forma automática",
        "Cumplido: 40 variables compartidas, horizonte de 1 a 72 h, MAE, RMSE, MAPE, persistencia y costo (Tablas 14 a 18)",
        "Cumplido: motor, cuatro tipos de imagen con metadatos y validación funcional, dashboard publicado (Figuras 3 a 6)",
        "Backtest cumplido (Tablas 19 y 20); validación con usuarios: [[completar con los resultados (Tabla 21)]]"]):
    celda(celdas(tr)[2], est)
R_("Entregables del Anexo 1 [2], con su estado", "Entregables del Anexo 1 [2], con su estado al cierre:")
for pre, txt in (
        ("Anexo 1", "Anexo 1 – Formulación del proyecto (entregado)."),
        ("Dataset integrado", "Dataset integrado y limpio (enero de 2019 – agosto de 2026) (entregado, en el repositorio)."),
        ("Notebooks", "Notebooks de análisis exploratorio y de modelado (entregados, en el repositorio)."),
        ("Biblioteca de imágenes", "Biblioteca de imágenes de apoyo a la decisión (entregada en el dashboard: semáforo, mapa de "
                                      "calor día-hora, bandas de incertidumbre y real frente a pronóstico, con metadatos)."),
        ("Módulo de recomendación", "Módulo de recomendación y resultados de backtesting (entregados: src/motor_decision.py y "
                                       "Sección 7.3)."),
        ("Dashboard", "Dashboard (publicado en augurpredictor.streamlit.app, con actualización diaria)."),
        ("Informe final y repositorio", "Informe final y repositorio en GitHub (github.com/Ziskito/xm-spot-price-predictor)."),
        ("Informe de avance", "Informe de avance – Semana 10 (entregado)."),
        ("Manual de uso", "Manual de uso del motor de decisión (entregado: manual de usuario MU-AUG-001, Anexo 4)."),
        ("Informe final – Semana", "Informe final – entrega el 20 de noviembre de 2026 (este documento), con el resumen y el abstract de la Sección 1.2 para la biblioteca (Manglar/DSpace)."),
        ("Póster", "Póster – entrega el 20 de noviembre; exposición abierta al público en el coliseo el 27 de noviembre ([[completar]])."),
        ("Presentación final", "Presentación final – entrega el 20 de noviembre; sustentación ante el jurado el 24 de noviembre ([[completar]])."),
        ("Video", "Video de 2 minutos con el funcionamiento del sistema – entrega el 20 de noviembre ([[completar]]).")):
    vineta(pre, txt)

# ================================================================================================== 5 revision bibliografica
R_("Además, el diseño exigió tomar decisiones", (
    "Además, el diseño exigió tomar decisiones que la literatura no fija para este caso: qué variables construir a partir de las "
    "series públicas (Tabla 11), qué información está realmente disponible en el momento de pronosticar, qué modelos usar y cómo "
    "combinarlos (de más de quince familias evaluadas, el diseño definitivo usa el predespacho de XM y tres CatBoost; Tabla 12), "
    "el rango de años del histórico, la forma de sincronizar series de resolución distinta y la codificación del ONI (Tabla 13). "
    "Cada decisión se tomó formulando alternativas, evaluándolas fuera de muestra y eligiendo con criterios explícitos "
    "(Sección 6.8)."))
t4 = tabla_tras(P("Tabla 4."))
fila(filas(t4)[-1], ["Este proyecto", "Colombia, horaria 2019-2026, solo información pública al corte",
                     "Predespacho de XM corregido con 3 CatBoost; bandas conformes; motor de decisión",
                     f"MAPE {pc(K['MAPE'])} a 24 h; rMAE {n(FINAL['rMAE'], 2)} frente a la persistencia",
                     "Depende de que XM publique el predespacho la víspera"])
p = P("Ninguno de los tres trabajos colombianos")
reescribir(p, texto_de(p).strip() + " Además, ninguno aprovecha la información que el operador del mercado publica antes del día "
                                    "de despacho, como el predespacho ideal de XM [39].")

# ================================================================================================== 6.1 a 6.3
p = P("El diseño resuelve esa ausencia")
despues(p, p, (
    "La disponibilidad de los datos es parte del problema. A las 00:00 del día D, cuando se emite el pronóstico, la API pública "
    "de XM solo tiene el precio, la demanda y la generación hasta las 23:00 de D-4 (unos tres días de rezago), los embalses y "
    "aportes hasta D-2 y el ONI con unos dos meses de rezago. En cambio, el predespacho ideal del día D, que XM calcula con las "
    "ofertas de los generadores, se publica hacia las 10:30 de D-1 en la biblioteca pública de su portal [39]. Un diseño que "
    "suponga conocido el precio del día anterior no se puede operar con fuentes públicas; por eso el diseño definitivo se evaluó "
    "en este escenario realista."))
t5 = tabla_tras(P("Tabla 5."))
for tr in filas(t5)[1:]:
    c0 = texto_de(celdas(tr)[0])
    if c0.startswith("Eficiencia computacional"):
        fila(tr, [None, "Actualización diaria completa (datos, reentrenamiento mensual y pronóstico) < 15 min", None])
    if c0.startswith("Sin fuga"):
        fila(tr, [None, "Solo información publicada al corte: precio y demanda hasta D-4, hidrología de D-2, ONI de M-2 y "
                        "predespacho creado y modificado antes de las 23:00 de D-1",
                  "Medición de la API de XM (Sección 6.1); decisión ABET 1 del Anexo 1"])
n_filas(t5, len(filas(t5)) + 1)
fila(filas(t5)[-1], ["Operación continua", "Pronóstico del día siguiente disponible cada día antes de las 16:00, sin intervención "
                                            "manual", "Uso diario del cliente; horario de publicación de XM"])
t6 = tabla_tras(P("Tabla 6."))
n_filas(t6, len(filas(t6)) + 1)
fila(filas(t6)[-1], ["Resolución CREG 025 de 1995 [41]", "Las ofertas cierran a las 08:00 de D-1 y el predespacho sale después: el "
                                                         "pronóstico se presenta para decisiones posteriores a su publicación "
                                                         "(compras en bolsa, coberturas), no para formular ofertas."])

# ================================================================================================== 6.4 alternativas
R_("Se plantearon tres alternativas", (
    "Se plantearon tres alternativas para el bloque de pronóstico, que determina la calidad de la señal que recibe el motor. Las "
    "tres comparten la cadena de datos de OE1 y el motor de OE3, y se evaluaron con la misma regla: solo la información pública "
    "disponible a las 00:00 del día pronosticado. La Tabla 7 las resume."))
R_("Solución 1:", (
    "**Solución 1: ensamble heterogéneo de seis modelos (diseño del informe de avance).** Combina persistencia, árboles de "
    "gradiente (XGBoost), un modelo econométrico ARX con varianza GARCH [13], dos redes neuronales (N-BEATSx y N-HiTS) y un GARCH "
    "con selección LASSO, con un peso no negativo por modelo y por franja horaria de 6 h [11]. El principio es el de diversificar "
    "un portafolio: si los errores de los modelos no están perfectamente correlacionados, su combinación reduce el error total. "
    f"Con el precio del día anterior su MAE era {n(C['ens6_optimista_mae'])} COP/kWh, 27 % menor que el de la persistencia; "
    "pero ese precio no es público al corte. Con lo publicado, los modelos trabajan con información de hace cuatro días y el "
    f"MAE sube a {n(ENS6['MAE'])} COP/kWh."))
R_("Solución 2:", (
    "**Solución 2: ensamble de seis modelos más el predespacho.** Agrega como séptimo votante el predespacho ideal que XM "
    "publica la víspera, una estimación del precio de cada hora del día siguiente hecha con las ofertas reales. El combinador "
    f"aprende a darle el mayor peso y el MAE baja a {n(ENS7['MAE'])} COP/kWh. Mantiene, sin embargo, el costo de entrenar seis "
    "modelos, incluidas dos redes neuronales, cuya contribución con datos públicos es marginal."))
R_("Solución 3:", (
    "**Solución 3: predespacho corregido con tres CatBoost.** Tres modelos de árboles CatBoost [38] aprenden a corregir el "
    "predespacho: en lugar del precio pronostican el logaritmo de la razón entre el precio y el predespacho, con pérdida de error "
    "absoluto y con conjuntos de variables distintos (las variables públicas y las del predespacho; además, la forma intradía del "
    "predespacho y el despacho programado; o el error reciente del predespacho). Se reentrenan cada mes con lo publicado y su "
    "promedio se combina con el predespacho por regresión cuantílica (QRA). Es la alternativa más liviana, pero depende de que "
    "XM publique el predespacho."))
t7 = tabla_tras(P("Tabla 7."))
llenar(t7, [
    ["**Descripción:** 6 modelos de familias distintas combinados por QRA en franjas de 6 h.",
     "**Descripción:** los 6 modelos y el predespacho de XM combinados por QRA.",
     "**Descripción:** predespacho de XM y promedio de 3 CatBoost combinados por QRA; reentrenamiento mensual."],
    ["**Ventajas:** diversidad de familias; no depende de XM más allá de las series.",
     "**Ventajas:** aprovecha la información del día siguiente; menor error.",
     f"**Ventajas:** el menor error ({n(FINAL['MAE'])} COP/kWh); entrena en segundos por modelo; sin sesgo."],
    ["**Desventajas:** con datos públicos casi no supera a la persistencia.",
     "**Desventajas:** costo de 6 modelos con aporte marginal; dos redes de caja negra.",
     "**Desventajas:** depende de la publicación del predespacho; hereda sus errores de nivel."]])

# ================================================================================================== 6.5 criterios y matriz
t8 = tabla_tras(P("Tabla 8."))
for tr in filas(t8)[1:]:
    c0 = texto_de(celdas(tr)[0])
    if c0.startswith("Precisión"):
        fila(tr, ["Precisión: rMAE frente a la persistencia del día anterior", None, None])
    if c0.startswith("Costo computacional"):
        fila(tr, ["Costo computacional: reentrenamiento del bloque de 24 h (min)", None, None])


def punt_prec(r):
    return 5 if r <= 0.75 else 4 if r <= 0.85 else 3 if r <= 0.95 else 2 if r <= 1 else 1


def punt_rob(r):
    return 5 if r <= 0.85 else 4 if r <= 0.90 else 3 if r <= 0.95 else 2 if r <= 1 else 1


costo = {"S1": (3, "7,2 min"), "S2": (3, "> 7,2 min"), "S3": (4, f"≈ {n(3 * C['costo']['catboost_s'] / 60)} min")}
interp = {"S1": 3, "S2": 3, "S3": 3}
tot = {}
for s in ("S1", "S2", "S3"):
    a = ALT[s]
    tot[s] = (0.35 * punt_prec(a["rMAE_opt"]) + 0.25 * punt_rob(a["peor_mes_opt"]) + 0.15 * costo[s][0] + 0.15 * interp[s]
              + 0.10 * 5)
p = P("Con estos criterios se construyó")
reescribir(p, texto_de(p).strip().replace(
    "La cobertura de las bandas no se puntúa porque las alternativas 2 y 3 usan la misma banda.",
    "La cobertura de las bandas no se puntúa porque las tres alternativas usan el mismo método de calibración conforme."))
R_("Tabla 9.", "**Tabla 9.** Matriz de decisión ponderada (puntaje 1-5; valor medido en 2026 con solo información pública, entre "
               "paréntesis).")
t9 = tabla_tras(P("Tabla 9."))
fila(filas(t9)[0], ["Criterio (peso)", "Sol. 1: Ensamble de 6", "Sol. 2: Ensamble + predespacho", "Sol. 3: Predespacho + CatBoost"])
llenar(t9, [
    ["Precisión (0,35)"] + [f"{punt_prec(ALT[s]['rMAE_opt'])} ({n(ALT[s]['rMAE_opt'], 3)})" for s in ("S1", "S2", "S3")],
    ["Robustez (0,25)"] + [f"{punt_rob(ALT[s]['peor_mes_opt'])} ({n(ALT[s]['peor_mes_opt'], 3)})" for s in ("S1", "S2", "S3")],
    ["Costo computacional (0,15)"] + [f"{costo[s][0]} ({costo[s][1]})" for s in ("S1", "S2", "S3")],
    ["Interpretabilidad (0,15)"] + [str(interp[s]) for s in ("S1", "S2", "S3")],
    ["Costo económico (0,10)", "5 ($0)", "5 ($0)", "5 ($0)"],
    ["**Total ponderado**"] + [f"**{n(tot[s], 2)}**" for s in ("S1", "S2", "S3")]])
R_("Selección.", (
    f"**Selección.** El predespacho corregido con CatBoost obtiene {n(tot['S3'], 2)} frente a {n(tot['S2'], 2)} y "
    f"{n(tot['S1'], 2)}. Con solo información pública, el ensamble de seis modelos pierde frente a la persistencia del día "
    f"anterior (rMAE {n(ALT['S1']['rMAE_opt'], 3)}) y en su peor mes llega a {n(ALT['S1']['peor_mes_opt'], 2)}. Las dos "
    f"alternativas con predespacho cumplen la precisión con holgura; la tercera además tiene el menor error "
    f"({n(FINAL['MAE'])} frente a {n(ENS7['MAE'])} COP/kWh, Diebold-Mariano t = {n(C['dm_definitivo_vs_ens7_realista'][0], 1)}, "
    f"p < 0,001), cumple la robustez en todos los meses de 2026 (peor mes {n(ALT['S3']['peor_mes_opt'], 3)}) y se reentrena en "
    f"{costo['S3'][1]}. Los rMAE de la tabla usan la persistencia del día anterior, más exigente que la que se puede calcular al "
    f"corte con el último precio publicado (frente a esta última, el rMAE del diseño definitivo es {n(FINAL['rMAE'], 2)})."))

# ================================================================================================== 6.7 riesgos
t10 = tabla_tras(P("Tabla 10."))
llenar(t10, [
    ["Técnico", "Fuga de información (usar datos que no son públicos al corte).", "Media", "Alto",
     "Escenario realista medido en la API; predespacho solo si se creó y modificó antes de las 23:00 de D-1; ofertas OFEI "
     "excluidas porque XM agrega los precios después."],
    ["Técnico", "Datos corruptos o revisados por la fuente.", "Media", "Alto",
     "Pruebas de reproducción; la serie de operación se descarga completa cada día para recoger las revisiones de XM; el "
     "dataset oficial no se modifica."],
    ["Operativo", "XM no publica el predespacho de un día o cambia su portal.", "Media", "Alto",
     "Respaldo con el predespacho del día anterior; la falla de la actualización diaria queda visible en GitHub y el dashboard "
     "muestra la fecha de la última actualización."],
    ["Regulatorio", "Intervención transitoria de la CREG en la formación del precio.", "Media", "Alto",
     "Reentrenamiento mensual automático; versión del modelo visible en cada imagen."],
    ["Técnico", "El pronóstico pierde precisión en El Niño.", "Alta", "Medio",
     "Bandas conformes por hora que se ensanchan; detector de picos y desplomes; filtro de confianza en el motor."],
    ["Social/ético", "El usuario toma la recomendación como garantía.", "Baja", "Alto",
     "Mostrar la incertidumbre y advertir que no es asesoría financiera."]])

# ================================================================================================== 6.8 diseno definitivo
R_("Metodología de diseño.", (
    "**Metodología de diseño.** Cada elemento del sistema (variable, modelo, combinador, calibración y regla de decisión) se "
    "definió con el mismo procedimiento: (1) se formula frente a una alternativa o línea base; (2) se evalúa fuera de muestra, "
    "con entrenamiento en 2019-2025 y prueba en 2026, usando solo la información pública disponible al corte; (3) se adopta solo "
    "si mejora con significancia en la prueba de Diebold-Mariano (p < 0,05) y, si la variante se eligió observando 2026, se "
    "valida además en los cinco años históricos; y (4) se documenta en la bitácora del repositorio, incluidos los intentos "
    "fallidos. La Figura 2 muestra el resultado de ese proceso y las Tablas 11 a 13 documentan cada definición."))
imagen("Figura 2.", "diagrama_bloques.png")
R_("Figura 2.", "**Figura 2.** Diagrama de bloques del diseño definitivo y su relación con los objetivos específicos. Fuente: "
                "elaboración propia.")
R_("Definición de variables.", (
    "**Definición de variables.** Todas las variables usan solo información publicada antes del instante de corte (00:00 del "
    "día pronosticado): el precio y la demanda entran con un rezago mínimo de 73 h, la hidrología con el valor diario de D-2, el "
    "ONI con el del mes M-2 y el predespacho con el archivo publicado la víspera. Cada familia se justificó con un resultado de "
    "la caracterización de OE1 o de la evaluación fuera de muestra (Tabla 11)."))
p = P("De las fuentes públicas se descargaron")
reescribir(p, texto_de(p).strip() + (
    " Para el diseño definitivo se descargaron además, de la biblioteca pública del portal de XM [39], 3.192 archivos diarios del "
    "predespacho ideal (enero de 2018 a agosto de 2026) y los del costo marginal y la demanda del despacho programado (dMAR y "
    "dDEM), con su fecha de creación y de modificación para verificar que cada uno estuviera disponible antes del corte."))
R_("Tabla 11.", "**Tabla 11.** Las 65 variables predictoras, agrupadas por familia (y = precio de bolsa; t = hora objetivo; "
                "D = día del corte).")
t11 = tabla_tras(P("Tabla 11."))
llenar(t11, [
    ["Precio rezagado (2)", "yt−96 (yt−120 para la hora 00:00), yt−168", "Picos de 24 h y 168 h del periodograma"],
    ["Medias móviles (3)", "Media del precio publicado en ventanas de 24 h, 7 días y 30 días", "Nivel y tendencia recientes"],
    ["Volatilidad (4)", "Desviación estándar en 24 h y 7 días, rango de 24 h y razón entre ambas desviaciones, con el precio "
                        "publicado", "Heterocedasticidad del precio (colas pesadas)"],
    ["Calendario (6)", "sen/cos(2πh/24), sen/cos(2πd/7), sen/cos(2πda/365,25)", "Armónicos de 24 h, 168 h y anual"],
    ["Hidrología (12)", "Embalses y aportes del día D-2, sus cambios a 1 y 7 días, medias de 7 y 30 días y razón frente a la "
                        "media de 30 días", "Correlación negativa con el precio; los aportes pesan más a 7 días (−0,309)"],
    ["Demanda (4)", "Demanda rezagada 96, 120 y 144 h y su media de 24 h", "Correlación +0,325; la contemporánea se excluye"],
    ["Clima y eventos (2)", "ONI del mes M-2 con rampa causal; indicador de pandemia (25-mar a 31-ago-2020)",
     "Correlación del ONI +0,387"],
    ["Festivos (7)", "Día festivo, festivo hace 1, 2, 3 y 7 días, y desajuste de tipo de día frente a t−24 h y t−168 h",
     "Reducen el error en festivos (p = 0,0064)"],
    ["Predespacho (9)", "Predespacho de la hora objetivo y del día (media, mínimo, máximo, posición relativa, rango), su rezago de "
                        "24 h, su cambio y su logaritmo", f"Estimación del precio de D hecha por XM; MAE {n(PRE['MAE'])} por sí "
                                                         "solo"],
    ["Forma del predespacho y despacho programado (11)", "Pendientes hora a hora, posición en el día, puntaje z y horas del "
                                                         "máximo y del mínimo; costo marginal y demanda programados (dMAR, dDEM)",
     "Señal en las rampas; mejor CatBoost individual en los cinco años históricos"],
    ["Error reciente del predespacho (5)", "Error logarítmico medio del predespacho en las ventanas ya publicadas (D-5, 7 y "
                                           "30 días; por hora)", "Corrige sesgos persistentes del predespacho"]])
R_("Definición de modelos.", (
    "**Definición de modelos.** El diseño definitivo tiene cinco componentes de pronóstico y un detector de eventos. La Tabla 12 "
    "resume su fundamento, configuración y forma de ajuste."))
R_("Tabla 12.", "**Tabla 12.** Componentes del diseño definitivo: fundamento, configuración y ajuste.")
t12 = tabla_tras(P("Tabla 12."))
llenar(t12, [
    ["Predespacho ideal", "Estimación del precio del día siguiente que publica XM [39]",
     "Costo marginal hora a hora de D, publicado hacia las 10:30 de D-1; la hora 00:00 de D+1 usa la de las 23:00 de D",
     "Solo archivos creados y modificados antes de las 23:00 de D-1; si falta, el del día anterior"],
    ["CatBoost R+P", "Árboles con gradiente ordenado [38]", "1.500 árboles, profundidad 6, tasa 0,05, pérdida MAE; objetivo "
                                                          "log(y/predespacho); 49 variables (públicas y del predespacho)",
     "Reentrenado cada mes con lo publicado hasta 73 h antes del corte"],
    ["CatBoost R+P+F+C", "Ídem", "Además, forma del predespacho y despacho programado (60 variables)", "Ídem"],
    ["CatBoost R+P+E", "Ídem", "Además, error reciente del predespacho (54 variables)", "Ídem"],
    ["Combinador", "QRA [11]", "Pesos ≥ 0 e intercepto por franja de 6 h para el predespacho y el promedio de los CatBoost; "
                               "objetivo MAE (Ec. 1)", "Solo días con precio ya publicado (5 días de margen); previa de jul-2024 "
                                                     "a jun-2025 los primeros 14 días"],
    ["Bandas", "Inferencia conforme por grupos [18], [40]", "q10 y q90 = q50·exp(∓m), con m el cuantil conforme de "
                                                            "|log(y/q50)| de la misma hora en los 60 días anteriores ya publicados",
     "Cobertura medida en 2026"],
    ["Pronóstico de 72 h", "Ídem", "Pasos 1-24: el de 24 h; pasos 25-72: predespacho del día del corte como referencia y 3 "
                                   "CatBoost por horizonte", "Margen del combinador de 5 + k días"],
    ["Detector de eventos", "Árboles con gradiente (clasificador)", "P(desplome: y ≤ 0,7 × mediana del día) y P(pico: y ≥ 1,3 × "
                                                                    "mediana) con la forma del predespacho y del precio publicado",
     "Reentrenado cada semana"]])
t_eq = tabla_tras(P("Tabla 12."))
t_eq = tabla_tras(t_eq)
nueva_eq = tabla_ecuacion(r"\mathrm{min}_{w\ge 0,\,b}\sum_{t\in g}{\left|y_t-b-\sum_k{w_k\,\hat{y}_{k,t}}\right|}", 1,
                          ancho_twips=9350, fuente="Calibri")
t_eq.addnext(nueva_eq)
eliminar(t_eq)
R_("La Ecuación 1 es el programa lineal", (
    "La Ecuación 1 es el programa lineal del combinador para cada franja *g* de 6 h: ŷk,t son el predespacho y el promedio de "
    "los CatBoost, y los pesos se estiman con los días cuyo precio ya está publicado. El producto de 72 h usa el mismo esquema, "
    "con el predespacho del día del corte como referencia."))
R_("Modelos evaluados y descartados.", (
    "**Modelos evaluados y descartados.** Con solo la información pública, los seis modelos del diseño inicial pierden su "
    "ventaja (Tabla 14): XGBoost, ARX+GARCH y GARCH-ged quedan apenas por debajo de la persistencia y las dos redes neuronales, "
    "por encima. Antes ya se habían descartado, por no mejorar a la persistencia o al ensamble, CatBoost sobre el precio, un "
    "modelo hurdle, Markov-Switching, mínimos cuadrados recursivos, TFT, LSTM, LightGBM y Prophet. Sobre el predespacho tampoco "
    "mejoraron: XGBoost (MAE de 38 a 46 COP/kWh), el LEAR por hora (40,8), las redes neuronales con el predespacho como variable "
    "futura (58), los precios de oferta por planta (las ofertas OFEI, que XM completa después del corte y serían fuga de "
    "información), un votante especialista en días baratos (35,07 frente a 35,01) ni reducir la amplitud de los picos del "
    "predespacho (de 35,10 a 37,24). La lección de diseño es que la información disponible al corte pesa más que la complejidad "
    "del modelo."))
t13 = tabla_tras(P("Tabla 13."))
filas13 = []
for tr in filas(t13)[1:]:
    c0 = texto_de(celdas(tr)[0])
    if c0.startswith("Rezago mínimo de 24 h"):
        fila(tr, ["Solo información publicada al corte (precio ≥ 73 h, hidrología D-2, ONI M-2)",
                  "Suponer el precio del día anterior (rezago de 24 h)",
                  f"La API de XM publica el precio con unos 3 días de rezago; con el supuesto anterior el ensamble de 6 "
                  f"parecía tener MAE {n(C['ens6_optimista_mae'])}, y con lo publicado tiene {n(ENS6['MAE'])}"])
    elif c0.startswith("Transformación logarítmica"):
        fila(tr, ["Pronosticar log(y/predespacho) con pérdida MAE", "Pronosticar el precio o usar pérdida cuadrática",
                  "Con pérdida cuadrática el MAE sube a 45,8; con MAE el sesgo del pronóstico queda en "
                  f"{sg(FINAL['sesgo'], 1)} COP/kWh"])
    elif c0.startswith("Combinador por franjas"):
        fila(tr, ["Combinador por franjas de 6 h con objetivo MAE y 5 días de margen", "Objetivo sMAPE; 4 días de margen",
                  "Con 4 días de margen el combinador usaba una hora aún no publicada; el objetivo MAE reduce el sesgo"])
    elif c0.startswith("Bandas conformes"):
        fila(tr, ["Bandas conformes en escala log por hora del día (60 días)", "Una sola banda para todas las horas",
                  "Con una banda global la cobertura iba de 54 % (00:00) a 94 % (01:00-04:00); por hora queda entre 77 y 84 %"])
    elif c0.startswith("Walk-forward"):
        fila(tr, [None, None, "Un corte único ocultaba que XGBoost pierde contra la persistencia en los 6 orígenes"])
    elif c0.startswith("Motor elegido por backtest"):
        g = MO["24h|generador"]
        fila(tr, [None, None, f"«fijo» y «banda» ganan el promedio, pero actúan en el {pc(g['metodos']['fijo']['frec_h1'] * 100)} "
                              f"y el {pc(g['metodos']['banda']['frec_h1'] * 100)} de las horas de la primera mitad; el criterio "
                              "estable elige «rodante» en los cuatro casos (Tablas 19 y 20)"])
nuevas13 = [
    ["Predespacho ideal de XM como referencia del pronóstico", "Pronosticar solo con series rezagadas",
     f"MAE {n(ENS6['MAE'])} → {n(FINAL['MAE'])}; validado en 2020-2025 (Tabla 16); con el predespacho de 7 días antes no "
     "mejora (placebo)"],
    ["CatBoost reentrenado cada mes", "Modelo entrenado una sola vez", "Entrenado solo hasta diciembre de 2025: MAE 37,5; "
                                                                       f"mensual: {n(FINAL['MAE'])}"],
    ["Filtro intradía en el motor", "Señales solo por umbrales",
     f"Peor mitad del generador a 24 h: {n(MO['24h|generador']['peor_mitad_sin_filtro'])} → "
     f"{n(min(MO['24h|generador']['metodos']['rodante']['ventaja_h1'], MO['24h|generador']['metodos']['rodante']['ventaja_h2']))} "
     "COP/kWh"],
    ["Dataset oficial congelado y serie de operación aparte", "Actualizar el dataset del proyecto",
     "Las cifras de este informe son reproducibles mientras el dashboard se actualiza cada día"]]
base = len(filas(t13))
n_filas(t13, base + len(nuevas13))
for tr, v in zip(filas(t13)[base:], nuevas13):
    fila(tr, v)
g24 = MO["24h|generador"]
R_("Motor de decisión (cuatro métodos", (
    "**Motor de decisión (cuatro métodos, filtro intradía y selección por estabilidad).** (1) Cada método fija un umbral bajo y "
    "uno alto y compara con ellos la mediana q50 de cada hora: por debajo del bajo el comercializador recibe «comprar» y el "
    "generador «retener»; por encima del alto, «evitar compra» y «vender»; en otro caso, «esperar». Los umbrales por defecto son "
    "los percentiles 25 y 75, el rango intercuartílico convencional para separar valores típicos de extremos [31]. (2) Los "
    "métodos difieren en la referencia: «fijo» usa los percentiles del precio de 2019-2025; «rodante», los de una ventana móvil "
    "causal de 30 días sobre q50, que se adapta al régimen vigente [32]; «banda» es el fijo más un filtro que emite «esperar» "
    "cuando el ancho relativo de la banda supera su percentil 75 en las horas anteriores; e «híbrido» combina el umbral rodante "
    "con ese filtro de confianza. (3) Un filtro intradía conserva la señal favorable (vender o comprar) solo en la mitad del día "
    "que el pronóstico ubica más cara (generador) o más barata (comercializador), y la desfavorable solo en la mitad opuesta: los "
    "umbrales deciden en qué temporada conviene la bolsa y el filtro, a qué hora del día actuar. (4) El método activo se elige "
    "por backtesting: solo son admisibles los que actúan entre 10 % y 40 % de las horas y, entre ellos, gana el de mayor ventaja "
    "en la mitad más débil del periodo (criterio maximin [33]), no el de mayor ventaja promedio."))
R_("La Figura 3 muestra el motor", (
    "La Figura 3 muestra el motor en operación durante una semana de julio de 2026 con el método que elige el criterio de "
    "estabilidad para el generador («rodante»). Cuando la mediana del pronóstico supera el umbral alto en la mitad cara del día, "
    "el motor recomienda vender, y cuando cae bajo el umbral bajo en la mitad barata, retener. Esa semana recomendó vender en 73 "
    "horas y retener en 20, y los picos de la tarde que anticipó el predespacho sí ocurrieron; a diferencia del informe de "
    "avance, cuando el motor usaba las bandas de N-BEATSx, el pronóstico sigue de cerca el precio real."))
imagen("Figura 3.", "motor_operacion.png")
R_("Figura 3.", "**Figura 3.** Motor de decisión en operación, método «rodante» con filtro intradía para el generador (13 al 19 "
                "de julio de 2026): pronóstico mediano, banda [q10, q90], umbrales rodantes y acción recomendada en cada hora.")
R_("Dashboard.", (
    "**Dashboard.** Se publicó en Streamlit Community Cloud (augurpredictor.streamlit.app) con la marca augur. La vista Operador "
    "aplica el método que elige el criterio de estabilidad y muestra, para el día y la hora elegidos, la acción recomendada con "
    "una insignia (favorable, desfavorable o neutral), un medidor que ubica el precio esperado entre los tramos barato, normal y "
    "caro, un nivel de confianza según el ancho relativo de la banda y la probabilidad de pico o desplome; la acompañan la "
    "gráfica del día (con un botón para ampliarla y variables que se ocultan o muestran desde la leyenda), la franja de 24 horas "
    "con el precio y la acción de cada hora, el mapa de calor día-hora, tres indicadores del backtest y la semana de referencia, "
    "cada imagen con sus metadatos (unidad, periodo, confianza y versión del modelo). La vista Analista permite fijar el método, "
    "los percentiles, el rango de fechas y el filtro intradía. La presentación se rediseñó el 15 de octubre con la "
    "retroalimentación del equipo (tarjetas, tema oscuro de alto contraste y diseño adaptable a pantallas angostas) sin cambiar "
    "el motor ni los datos. La interfaz se verificó con pruebas automatizadas de Streamlit en las 16 combinaciones de tema, "
    "vista, rol y horizonte, y en un navegador a varios anchos de pantalla."))
R_("Funcionamiento del prototipo.", (
    "**Funcionamiento del prototipo y operación diaria.** El sistema funciona de principio a fin sin intervención manual. Todos "
    "los días a las 15:40 y a las 18:40, un flujo de GitHub Actions descarga los datos nuevos de XM y NOAA, reentrena los "
    "CatBoost del mes, pronostica la ventana del día siguiente con el predespacho recién publicado y sube los resultados al "
    "repositorio, con lo que el dashboard se actualiza; en el servidor de GitHub la corrida completa tarda unos 9 minutos. Lo ya "
    "pronosticado no se reescribe y el precio real se completa a medida que XM lo publica. El dataset oficial del proyecto (enero "
    "de 2019 a agosto de 2026) no se modifica, de modo que las cifras de este informe son reproducibles. El código completo está "
    "en el Anexo 3, el manual de usuario en el Anexo 4 y los resultados que demuestran el funcionamiento de cada bloque, en la "
    "Sección 7."))
R_("Vista Operador.", (
    "**Vista Operador.** La Figura 4 muestra la vista Operador el 16 de octubre de 2026 a las 08:00 h para el generador, con el "
    "pronóstico emitido la víspera. La tarjeta recomienda «Retener generación»: el medidor ubica el precio esperado, 431 COP/kWh, "
    "en el tramo barato (≤ 503) y lejos del caro (≥ 720), con confianza alta (banda de ±42 COP/kWh, más ancha que el 35 % de las "
    "horas anteriores). El riesgo de desplome a esa hora es de 49 %: el detector anticipa un posible desplome entre las 08:00 y "
    "las 11:00 y un posible pico entre las 16:00 y las 21:00, que la gráfica del día marca con ▼ y ▲. Ese día el generador tiene "
    "6 de 24 horas para despachar y vender, y la regla activa, «rodante», tiene una ventaja histórica de +244 COP/kWh y actúa en "
    "el 22 % de las horas."))
imagen("Figura 4.", "dash_operador.png", 6.3)
R_("Figura 4.", "**Figura 4.** Dashboard publicado, vista Operador (16 de octubre de 2026, 08:00 h, rol generador, horizonte "
                "24 h).")
R_("Rol generador.", (
    "**Mapa de calor día-hora.** El mapa de calor (Figura 5) resume la señal de los 14 días anteriores hora por hora: en la "
    "primera semana de octubre el motor recomendó vender en las horas de la tarde y de la noche y retener al mediodía, y "
    "desde el 11 de octubre, con precios más bajos, retener en buena parte de la mañana y del mediodía. Las fechas de los extremos se escriben o se eligen en un "
    "calendario, y el modo «Precio esperado» colorea las celdas con la escala de color del precio de bolsa que usa XM."))
imagen("Figura 5.", "dash_calor.png", 6.0)
R_("Figura 5.", "**Figura 5.** Mapa de calor día-hora de la señal del motor (3 al 16 de octubre de 2026, rol generador).")
R_("Vista Analista.", (
    "**Vista Analista.** La Figura 6 muestra la vista Analista para el generador a 24 h con todo el periodo publicado (1 de enero "
    "al 17 de octubre de 2026). La tabla compara los cuatro métodos por ventaja y frecuencia de acción en el rango elegido: "
    "«banda» queda sugerido por ventaja (+323 COP/kWh en el 18 % de las horas), pero la vista Operador usa el criterio de "
    "estabilidad, que elige «rodante» porque «banda» y «fijo» casi no actúan en la primera mitad del año (Tabla 20). Debajo, la "
    "tabla «Ver datos filtrados» permite auditar cada recomendación hora a hora y descargarla."))
imagen("Figura 6.", "dash_analista.png", 4.3)
R_("Figura 6.", "**Figura 6.** Dashboard publicado, vista Analista (rol generador, horizonte 24 h, 1 de enero al 17 de octubre de "
                "2026).")
eliminar(P("Rol generador en la vista Analista."))

# ================================================================================================== 7.1 y 7.2
R_("Pregunta técnica.", (
    "**Pregunta técnica.** Con solo la información pública disponible al momento de pronosticar, ¿la plataforma pronostica el "
    "precio de bolsa horario a 24-72 h con un error significativamente menor que la persistencia (rMAE < 1) y con bandas cuya "
    "cobertura esté cerca del 80 %? ¿Y el motor obtiene una ventaja positiva y estable con sus recomendaciones?"))
R_("Hipótesis.", (
    "**Hipótesis.** Para cada hora t, dt = |etpers| − |etmod| es la diferencia entre el error absoluto de la persistencia y el del "
    "modelo. H0: E[dt] = 0 (el modelo no es más preciso). H1: E[dt] > 0 (el modelo tiene menor error). La prueba es de una cola, "
    "porque la especificación exige superar a la referencia; se aplica a 24 h y a 72 h. La persistencia es el precio publicado más "
    "recientemente para la misma hora (hace 96 h), la referencia ingenua que se puede calcular al corte; también se reporta la del "
    "día anterior, que no es pública pero es la referencia usual de la literatura. Para los modelos individuales en cada origen "
    "se usa la prueba de dos colas, para detectar si ganan o pierden."))
R_("Variables.", (
    "**Variables.** La variable dependiente es el error absoluto horario |yt − ŷt| (COP/kWh), resumido en MAE, RMSE, MAPE y sMAPE. "
    "Los factores son: modelo (11 niveles: persistencia, seis modelos del diseño inicial, su ensamble, el predespacho, el ensamble "
    "con predespacho y el diseño definitivo), escenario de información (con el precio del día anterior o solo con lo publicado), "
    "horizonte (3 tramos de 24 h), origen temporal (6 regímenes ENSO) y, en el motor, rol (2) y método (4). El método es una "
    "evaluación fuera de muestra con origen móvil (walk-forward) [19] con comparaciones pareadas hora a hora."))
R_("Instrumento:", (
    "Instrumento: la serie oficial de precio de XM [4], horaria y con hasta cinco decimales en COP/kWh; su error es despreciable "
    "frente a los de pronóstico. Equipo: AMD Ryzen 7 7730U, 15,4 GB de RAM, sin GPU. Repeticiones: "
    f"{n(C['horas_comunes'], 0)} pares hora a hora en las mismas horas de 2026 para comparar modelos ({DIA['dias']} días, del 15 "
    f"de enero al 5 de agosto) y {n(K['horas'], 0)} en el contrato completo, muy por encima de 30; la autocorrelación se corrige "
    "con la varianza de Newey-West [34]. El walk-forward suma seis orígenes de 5.184 a 8.784 horas. Significancia: α = 0,05."))
R_("Invalidación interna controlada.", (
    "**Invalidación interna controlada.** Historia y maduración: seis orígenes con regímenes distintos. Instrumentación: mismo "
    "dataset, periodo y código de métricas para todos; las auditorías de reproducibilidad detectaron y corrigieron la demanda "
    "del 4-5 de agosto de 2026 y, después, la revisión de XM de esos días y la actualización del ONI de NOAA, con las que se "
    "reentrenaron todos los modelos. Regresión estadística: se usa todo 2026, sin elegir periodos extremos. Fuga: solo "
    "información publicada al corte, verificada con la fecha de creación y de modificación de cada archivo, y un placebo con el "
    "predespacho de 7 días antes no mejora el pronóstico. Selección: las variantes del diseño definitivo se eligieron observando "
    "2026, por lo que se validaron además en los cinco orígenes históricos (Tabla 16)."))
R_("El plan se ejecutó en el orden previsto", (
    "El plan se ejecutó en el orden previsto: dataset maestro, características, entrenamiento con 2019-2025 y pronóstico de 2026, "
    "walk-forward, combinación, producto de 72 h, calibración de bandas y backtesting del motor. Hubo tres ajustes: la corrección "
    "de la demanda del 4-5 de agosto, que obligó a repetir el entrenamiento; la medición de la disponibilidad real de los datos, "
    "que obligó a reevaluar todo en el escenario realista y llevó al predespacho; y la actualización del ONI de NOAA y la "
    "revisión de XM del 1-5 de agosto, con las que se reentrenaron los modelos y los orígenes históricos. Ecuaciones de "
    "procesamiento (N horas, yt real, ŷt pronóstico):"))

# ================================================================================================== 7.3 resultados
p = P("OE1 — Datos y caracterización.")
reescribir(p, "**OE1 — Datos y caracterización.**" + texto_de(p).strip()[len("OE1 — Datos y caracterización."):].replace(
    "(Figura 10)", "(Figura 7)") + (
    " Una comparación posterior con un filtro de Butterworth de fase cero, el suavizador de Kalman y wavelets confirmó la "
    "elección: a igual suavizado, Butterworth y Kalman dejan un residuo algo menor (111 frente a 118 COP/kWh), pero llegan "
    "después a la subida (382 y 407 h frente a 348 h)."))
R_("Figura 7.", "**Figura 7.** Periodograma del precio de bolsa sin tendencia. Líneas rojas: períodos de 12, 24 y 168 h.")
R_("OE2 — Pronóstico de 24 h.", (
    f"**OE2 — Pronóstico de 24 h.** La Tabla 14 compara, en las mismas {n(C['horas_comunes'], 0)} horas de 2026 y con solo la "
    "información pública al corte, los modelos individuales, el ensamble del diseño inicial, el predespacho de XM y las dos "
    "alternativas que lo usan."))
R_("Tabla 14.", f"**Tabla 14.** Desempeño a 24 h fuera de muestra con solo información pública (15-ene a 5-ago de 2026, "
                f"n = {n(C['horas_comunes'], 0)} h).")
t14 = tabla_tras(P("Tabla 14."))
orden14 = ["Persistencia (precio publicado más reciente)", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS", "GARCH-ged",
           "Ensamble de 6 (diseño del informe de avance)", "Predespacho ideal de XM", "Ensamble de 6 + predespacho",
           "Promedio de 3 CatBoost", "Predespacho + 3 CatBoost (diseño definitivo)"]
mat = []
for nom in orden14:
    f = T24[nom]
    nm = {"Persistencia (precio publicado más reciente)": "Persistencia (último precio publicado)",
          "Ensamble de 6 (diseño del informe de avance)": "Ensamble de 6 (diseño inicial)",
          "Predespacho ideal de XM": "Predespacho de XM",
          "Predespacho + 3 CatBoost (diseño definitivo)": "**Diseño definitivo**"}.get(nom, nom)
    mat.append([nm, n(f["MAE"], 2), n(f["RMSE"], 2), n(f["MAPE"], 2), n(f["sMAPE"], 2), n(f["rMAE"], 3),
                "—" if f["DM_t"] is None else n(f["DM_t"], 2)])
llenar(t14, mat)
R_("DM (t): Diebold-Mariano frente a la persistencia", (
    f"DM (t): Diebold-Mariano frente a la persistencia (positivo = menor error); el rMAE usa la misma persistencia, el precio "
    f"publicado más reciente de la misma hora. Frente a la persistencia del día anterior, que no es pública al corte (MAE "
    f"{n(C['persistencia_optimista_mae'])}), el diseño definitivo tiene rMAE {n(ALT['S3']['rMAE_opt'], 3)}; con ese precio, el "
    f"ensamble de 6 tenía MAE {n(C['ens6_optimista_mae'])}. El diseño definitivo supera al predespacho solo (t = "
    f"{n(C['dm_definitivo_vs_predespacho'][0], 1)}) y al ensamble de 6 con predespacho (t = "
    f"{n(C['dm_definitivo_vs_ens7_realista'][0], 1)}), ambos con p < 0,001. En el contrato completo (1 de enero al 5 de agosto, "
    f"{n(K['horas'], 0)} h) su MAE es {n(K['MAE'])} COP/kWh y su MAPE {pc(K['MAPE'])}, frente a {n(K['MAE_pers'])} de la "
    f"persistencia y {n(K['MAE_pre'])} del predespacho solo."))
R_("Error diario e intervalos de confianza.", (
    f"**Error diario e intervalos de confianza.** El MAE del diseño definitivo es un promedio de {DIA['dias']} días con "
    f"comportamientos muy distintos, por lo que también se analizó el error de cada día. El MAE diario tiene media "
    f"{n(DIA['media'])} COP/kWh y desviación estándar {n(DIA['desv'])}. Como los días de error alto vienen en rachas "
    f"(autocorrelación de {n(DIA['autocorr'], 2)} entre días consecutivos), el intervalo de confianza del 95 % para la media se "
    f"calculó con bootstrap por bloques de 7 días: [{n(DIA['ic'][0])}; {n(DIA['ic'][1])}] COP/kWh, frente a "
    f"[{n(DIA['ic_pers'][0])}; {n(DIA['ic_pers'][1])}] de la persistencia. En el 80 % de los días el MAE estuvo entre "
    f"{n(DIA['p10'])} y {n(DIA['p90'])} COP/kWh y fue menor que 50 COP/kWh en el {pc(DIA['menor50'])} de los días "
    f"({pc(DIA['menor50_pers'])} para la persistencia). El MAPE diario tiene media {pc(DIA['mape_media'])} con intervalo "
    f"[{pc(DIA['ic_mape'][0])}; {pc(DIA['ic_mape'][1])}]. La diferencia diaria de MAE entre la persistencia y el diseño "
    f"definitivo tiene media {n(DIA['dif_media'])} COP/kWh, con un intervalo [{n(DIA['ic_dif'][0])}; {n(DIA['ic_dif'][1])}] que "
    f"no incluye el cero, y el diseño definitivo tuvo menor error en el {pc(DIA['frac_mejor'])} de los días. El error medio con "
    f"signo es de {sg(DIA['sesgo'])} COP/kWh [{n(DIA['ic_sesgo'][0])}; {n(DIA['ic_sesgo'][1])}]: no hay sesgo significativo. Por "
    f"régimen, entre enero y abril el MAE es {n(rn['MAE'])} frente a {n(rn['MAE_pre'])} del predespacho solo (t = "
    f"{n(rn['DM_vs_pre'][0], 1)}), y en El Niño (mayo a agosto), {n(rnino['MAE'])} frente a {n(rnino['MAE_pre'])}, una "
    f"diferencia no significativa (p = {n(rnino['DM_vs_pre'][1], 2)})."))
R_("Robustez entre regímenes.", (
    "**Robustez entre regímenes.** La Tabla 15 muestra el MAE de los modelos individuales del diseño inicial en los seis orígenes "
    "walk-forward, con el precio del día anterior (O1 a O5: 12 meses cada uno con corte a las 23:00; O6: 2026), y el resultado de "
    "Diebold-Mariano frente a la persistencia."))
t15 = tabla_tras(P("Tabla 15."))
WF, WDM = C["walkforward_mae"], C["walkforward_dm"]
mat = []
for m in ("Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"):
    vals = [n(WF[f"Origen {i}|{m}"], 2) for i in range(1, 7)]
    gp = "referencia" if m == "Persistencia" else f"{WDM[f'{m}|Persistencia']['gana']} / {WDM[f'{m}|Persistencia']['pierde']}"
    mat.append([m] + vals + [gp])
llenar(t15, mat)
cap15 = P("Tabla 15.")
t15 = tabla_tras(cap15)
EH = {e["origen"]: e for e in C["ensamble_historico"]}
VD = {v["origen"]: v for v in C["validacion_definitivo"]}
PH = C["persistencia_realista_historica"]
ultimo = despues(t15, P("Horizonte de 72 h."), (
    "**Validación histórica.** Con el precio del día anterior ningún modelo individual supera de forma consistente a la "
    f"persistencia (Tabla 15): N-BEATSx le gana a ARX+GARCH en {WDM['N-BEATSx|ARX+GARCH']['gana']} de 6 orígenes y frente a la "
    f"persistencia gana en {WDM['N-BEATSx|Persistencia']['gana']} y pierde en {WDM['N-BEATSx|Persistencia']['pierde']}. El "
    "ensamble de seis votantes, en cambio, le gana a la persistencia con significancia en 4 de 5 años (Tabla 16). El diseño "
    "definitivo, con solo información pública, mejora al predespacho solo en los cinco años y reduce el error de la persistencia "
    f"realista entre {n(min(1 - VD[o]['MAE_v2'] / PH[o] for o in VD) * 100, 0)} % y "
    f"{n(max(1 - VD[o]['MAE_v2'] / PH[o] for o in VD) * 100, 0)} %."))
cap16 = despues(ultimo, cap15, "**Tabla 16.** Validación fuera de muestra en los orígenes históricos (MAE en COP/kWh).")
t16 = copy.deepcopy(t15)
cap16.addnext(t16)
forma(t16, [1250, 1300, 1450, 1250, 1250, 1300, 1550])
matriz16 = [["Origen", "Persistencia realista", "Ensamble de 6 (precio del día anterior)", "DM t ensamble",
             "Predespacho solo", "Diseño definitivo", "rMAE definitivo"]]
for i, o in enumerate(("Origen 1", "Origen 2", "Origen 3", "Origen 4", "Origen 5"), start=1):
    e, v = EH[o], VD[o]
    matriz16.append([f"O{i} / {2019 + i}-{str(20 + i)}", n(PH[o]), n(e["MAE_ensamble"], 2), n(e["DM_t_vs_persistencia"], 2),
                     n(v["MAE_predespacho"], 2), n(v["MAE_v2"], 2), n(v["MAE_v2"] / PH[o], 3)])
llenar(t16, matriz16, desde=0)
R_("Horizonte de 72 h.", (
    f"**Horizonte de 72 h.** La Tabla 17 compara el producto de 72 h con la persistencia realista medida desde el mismo corte "
    f"({C['cortes72']} cortes, uno cada 3 días)."))
R_("Tabla 17.", "**Tabla 17.** Pronóstico de 72 h por tramo con solo información pública, prueba 2026.")
t17 = tabla_tras(P("Tabla 17."))
fila(filas(t17)[0], ["Tramo", "MAE", "MAPE", "MAE persistencia", "MAPE persistencia", "rMAE", "DM (t)"])
llenar(t17, [[f["tramo"].replace("1-72 h", "Global 1-72 h"), n(f["MAE"], 2), pc(f["MAPE"], 2), n(f["MAE_pers"], 2),
              pc(f["MAPE_pers"], 2), n(f["rMAE"], 3), n(f["DM_t"], 2)] for f in C["tabla72"]])
R_("Incertidumbre y costo.", (
    "**Incertidumbre y costo.** La Tabla 18 reúne la cobertura de las bandas y el costo computacional del diseño definitivo en el "
    "mismo equipo."))
R_("Tabla 18.", "**Tabla 18.** Cobertura de la banda [q10, q90] (objetivo 80 %) y costo computacional.")
t18 = tabla_tras(P("Tabla 18."))
fila(filas(t18)[0], [None, None, None, "Componente", None])
cs = C["costo"]["catboost_s"]
llenar(t18, [
    ["24 h", pc(K["cobertura"]), n(K["ancho"]), "Un CatBoost (65.304 filas)", f"{n(cs)} s"],
    ["72 h, 1-24 h", pc(T72["1-24 h"]["cobertura"]), n(T72["1-24 h"]["ancho"]), "Reentrenamiento mensual 24 h (3 CatBoost)",
     f"≈ {n(3 * cs / 60)} min"],
    ["72 h, 25-48 h", pc(T72["25-48 h"]["cobertura"]), n(T72["25-48 h"]["ancho"]), "Reentrenamiento mensual 72 h (6 CatBoost)",
     f"≈ {n(6 * cs / 60)} min"],
    ["72 h, 49-72 h", pc(T72["49-72 h"]["cobertura"]), n(T72["49-72 h"]["ancho"]), "Actualización diaria completa (este equipo)",
     "7,0 min"],
    ["", "", "", "Actualización diaria en GitHub Actions", "9 min"],
    ["", "", "", "Ensamble de 6 del diseño inicial (referencia)", "7,2 min"]])
g, cm = MO["24h|generador"], MO["24h|comercializador"]
R_("OE3 y OE4 — Motor de decisión.", (
    "**OE3 y OE4 — Motor de decisión.** La ventaja es la diferencia entre el precio medio en las horas en que el motor recomienda "
    f"actuar y el del periodo (precio medio 2026: {n(MO['precio_medio'])} COP/kWh). Con el criterio de ventaja promedio ganarían "
    f"«fijo» o «banda» en los cuatro casos (de {n(min(MO[c]['metodos'][m]['ventaja_anio'] for c in ('24h|generador', '24h|comercializador', '72h|generador', '72h|comercializador') for m in ('fijo', 'banda')), 0)} "
    f"a {n(max(MO[c]['metodos'][m]['ventaja_anio'] for c in ('24h|generador', '24h|comercializador', '72h|generador', '72h|comercializador') for m in ('fijo', 'banda')), 0)} COP/kWh), "
    "pero al evaluar por separado las dos mitades del periodo (corte: 19 de abril) ese resultado no se sostiene: para el "
    f"generador a 24 h, «fijo» actúa en el {pc(g['metodos']['fijo']['frec_h1'] * 100)} de las horas de la primera mitad y en el "
    f"{pc(g['metodos']['fijo']['frec_h2'] * 100)} de la segunda, y para el comercializador, en el "
    f"{pc(cm['metodos']['fijo']['frec_h1'] * 100)} y en el {pc(cm['metodos']['fijo']['frec_h2'] * 100)} (Tabla 20, Figura 8). La "
    "causa es la referencia fija: el precio de 2026 se alejó del rango 2019-2025, de modo que casi todas las horas quedan por "
    "encima o por debajo de los percentiles históricos. Con el criterio de estabilidad y el filtro intradía, el motor elige "
    "«rodante» en los cuatro casos, con ventaja positiva y frecuencia válida en las dos mitades (Tabla 19). Su ventaja promedio es "
    f"menor ({n(min(MO[c]['metodos']['rodante']['ventaja_anio'] for c in ('24h|generador', '24h|comercializador', '72h|generador', '72h|comercializador')))} a "
    f"{n(max(MO[c]['metodos']['rodante']['ventaja_anio'] for c in ('24h|generador', '24h|comercializador', '72h|generador', '72h|comercializador')))} COP/kWh), "
    "pero se mantiene en todo el año. Con un pronóstico mejor la ventaja de las reglas de umbral casi no cambia, porque premia "
    "sobre todo actuar en los meses caros; el filtro intradía es el que mejora la mitad más débil del generador a 24 h, de "
    f"{n(g['peor_mitad_sin_filtro'])} a {n(min(g['metodos']['rodante']['ventaja_h1'], g['metodos']['rodante']['ventaja_h2']))} "
    "COP/kWh."))
R_("Tabla 19.", "**Tabla 19.** Backtesting del motor de decisión sobre 2026 (pronóstico con solo información pública): método "
                "elegido por el criterio de estabilidad.")
t19 = tabla_tras(P("Tabla 19."))
mat = []
for h, rol in (("24h", "generador"), ("24h", "comercializador"), ("72h", "generador"), ("72h", "comercializador")):
    c = MO[f"{h}|{rol}"]
    d = c["metodos"][c["metodo_estable"]]
    mat.append([h, rol, met(c["metodo_estable"]), f"{n(d['ventaja_h1'])} / {n(d['ventaja_h2'])}",
                f"{pc(d['frec_h1'] * 100)} / {pc(d['frec_h2'] * 100)}"])
llenar(t19, mat)
R_("Tabla 20.", "**Tabla 20.** Métodos del motor a 24 h: ventaja del año completo frente a cada mitad del periodo.")
t20 = tabla_tras(P("Tabla 20."))
mat = []
for rol in ("generador", "comercializador"):
    c = MO[f"24h|{rol}"]
    for m in ("fijo", "rodante", "banda", "hibrido"):
        d = c["metodos"][m]
        mat.append([rol, met(m), n(d["ventaja_anio"]), pc(d["frec_anio"] * 100),
                    f"{n(d['ventaja_h1'])} / {'sin acción' if d['ventaja_h2'] is None else n(d['ventaja_h2'])}",
                    f"{pc(d['frec_h1'] * 100)} / {pc(d['frec_h2'] * 100)}", "sí" if d["estable"] else "no"])
llenar(t20, mat)
imagen("Figura 8.", "frecuencia_mitades.png")
R_("Figura 8.", "**Figura 8.** Horas con acción del generador a 24 h en cada mitad del periodo 2026, por método (filtro "
                 "intradía activo). Franja verde: rango válido de 10 a 40 %.")
B = {r["par"]: r for r in MO["barrido"]["pares"]}
R_("Sensibilidad de los umbrales.", (
    f"**Sensibilidad de los umbrales.** Se barrieron seis pares de percentiles (10/90 a 35/65) con el método elegido para el "
    f"generador a 24 h (Figura 9). No aparece un óptimo interior: los pares más extremos dan más ventaja por hora de acción pero "
    f"actúan menos ({n(B['10/90']['ventaja'])} COP/kWh en el {pc(B['10/90']['frec'])} de las horas con 10/90, frente a "
    f"{n(B['35/65']['ventaja'])} en el {pc(B['35/65']['frec'])} con 35/65), y el par 25/75 queda en un punto intermedio "
    f"({n(B['25/75']['ventaja'])} COP/kWh, {pc(B['25/75']['frec'])}). En el barrido de la bitácora también se probaron ventanas "
    "rodantes de 7 a 90 días, percentiles del filtro de ancho de 60 a 90, histéresis y lógica difusa: ninguna configuración "
    "ajustada en una mitad de 2026 mejoró a la actual en la otra, por lo que se conservaron los valores por defecto."))
imagen("Figura 9.", "barrido_percentiles.png")
R_("Figura 9.", f"**Figura 9.** Ventaja frente a frecuencia de acción para seis pares de percentiles (método "
                 f"{met(MO['barrido']['metodo'])}, generador, 24 h).")
UM = MO["ultimo_mes"]
um = UM["metodos"]
R_("Último mes y traducción a pesos.", (
    f"**Último mes y traducción a pesos.** Entre el 6 de julio y el 5 de agosto de 2026 la banda cubrió el {pc(UM['cobertura'])} "
    f"de las horas y el MAE fue {n(UM['mae'])} COP/kWh. Para el generador a 24 h, los cuatro métodos dieron ventaja positiva: "
    f"«rodante» {sg(um['rodante']['ventaja'])} COP/kWh en el {pc(um['rodante']['frec'])} de las horas, «híbrido» "
    f"{sg(um['hibrido']['ventaja'])} en el {pc(um['hibrido']['frec'])}, «banda» {sg(um['banda']['ventaja'])} en el "
    f"{pc(um['banda']['frec'])} y «fijo» {sg(um['fijo']['ventaja'])} en el {pc(um['fijo']['frec'])}. Para un cliente de ejemplo "
    f"de 100 kW que opere a potencia constante en las horas de acción, esto equivale a {sg(um['rodante']['cop'] / 1e6, 2)} "
    f"millones de COP con «rodante», {sg(um['hibrido']['cop'] / 1e6, 2)} con «híbrido», {sg(um['banda']['cop'] / 1e6, 2)} con "
    f"«banda» y {sg(um['fijo']['cop'] / 1e6, 2)} con «fijo» (Figura 10); en el informe de avance, con el pronóstico de "
    "N-BEATSx, «banda» perdía 0,65 millones en el mismo mes. La traducción a pesos es ilustrativa y no sustituye una evaluación "
    "económica completa [8]."))
imagen("Figura 10.", "ultimo_mes_pesos.png")
cap10 = R_("Figura 10.", "**Figura 10.** Ganancia total por método en el último mes del periodo (generador, 24 h, cliente de "
                         "ejemplo de 100 kW).")
plant = P("Último mes y traducción a pesos.")
x = despues(cap10, plant, (
    f"**Operación posterior al periodo del proyecto.** Desde el 6 de agosto el sistema opera con los datos que XM publica cada "
    f"día. Hasta el 11 de octubre ({n(OP['horas'], 0)} horas con precio publicado), el MAE fue {n(OP['MAE'])} COP/kWh (MAPE "
    f"{pc(OP['MAPE'])}), frente a {n(OP['MAE_pre'])} del predespacho solo; por mes, {n(OP['mensual']['2026-08'])} en agosto, "
    f"{n(OP['mensual']['2026-09'])} en septiembre y {n(OP['mensual']['2026-10'])} en lo corrido de octubre, meses más volátiles. "
    f"La banda cubrió el {pc(OP['cobertura'])} de las horas, por debajo del objetivo, porque se calibra con los 60 días "
    "anteriores y el precio se volvió más volátil en septiembre y octubre."))
x = despues(x, plant, (
    f"**Eventos.** El detector de picos y desplomes, con solo información pública, alcanza un AUC de "
    f"{n(EV[('desplome', 'clasificador realista')]['AUC'], 3)} para desplomes y {n(EV[('pico', 'clasificador realista')]['AUC'], 3)} "
    f"para picos, frente a {n(EV[('desplome', 'predespacho solo (0/1)')]['AUC'], 3)} y "
    f"{n(EV[('pico', 'predespacho solo (0/1)')]['AUC'], 3)} de usar el predespacho solo como alarma."))
x = despues(x, plant, (
    "**OE4 — Validación con usuarios.** El dashboard se evaluó con [[N]] usuarios mediante un cuestionario anónimo en Microsoft "
    "Forms: perfil (relación con el sector, familiaridad con el mercado y dispositivo), once afirmaciones sobre la comprensión y "
    "la utilidad de cada imagen de decisión con una escala de acuerdo de cinco niveles, una nota global de 1 a 10 y tres "
    "preguntas abiertas (lo más útil, lo que cambiarían y lo que agregarían). [[Completar con los resultados: Tabla 21 y "
    "comentarios principales.]]"))
cap21 = despues(x, P("Tabla 20."), "**Tabla 21.** Resultados del cuestionario de validación con usuarios ([[completar]]).")
t21 = copy.deepcopy(tabla_tras(P("Tabla 20.")))
cap21.addnext(t21)
forma(t21, [5800, 1800, 1800])
afirm = ["La recomendación del día (comprar, vender o esperar) se identifica rápido.",
         "Entendí qué significa la banda de incertidumbre del pronóstico.",
         "El mapa de calor día-hora ayuda a ubicar las horas convenientes.",
         "La gráfica de precio real frente al pronosticado permite juzgar si confiar en el pronóstico.",
         "Las alertas de pico y desplome del precio son claras.",
         "La información de cada gráfica (unidad, periodo, confianza, modelo) es suficiente.",
         "La navegación (día, hora, horizonte, rol) es intuitiva.",
         "La vista Analista es útil para comparar los métodos del motor.",
         "El diseño visual es claro y profesional.",
         "El manual de usuario resolvió mis dudas.",
         "Usaría esta herramienta como apoyo para decidir cuándo comprar o vender energía.",
         "Nota global (1 a 10)"]
llenar(t21, [["Afirmación", "Promedio (1-5)", "De acuerdo o muy de acuerdo"]] + [[a, "[[—]]", "[[—]]"] for a in afirm], desde=0)
for tr in filas(t21)[1:]:
    Paragraph(celdas(tr)[0].find(qn("w:p")), doc._body).alignment = 0

# ================================================================================================== 7.4 analisis
R_("Comparación con los criterios.", (
    f"**Comparación con los criterios.** El diseño definitivo cumple la precisión con holgura: rMAE de {n(FINAL['rMAE'], 2)} "
    f"frente a la persistencia realista y de {n(ALT['S3']['rMAE_opt'], 2)} frente a la del día anterior a 24 h, y de "
    f"{n(T72['1-72 h']['rMAE'], 2)} frente a la persistencia realista a 72 h. Cumple la robustez mensual (rMAE < 1 en los ocho "
    f"meses de 2026; peor mes {n(ALT['S3']['peor_mes_opt'], 2)} frente a la persistencia del día anterior) y el costo "
    f"(actualización diaria completa en 7 min, menos de 15). La cobertura de la banda ({pc(K['cobertura'])} a 24 h y "
    f"{pc(T72['1-72 h']['cobertura'])} a 72 h) queda dentro de 80 ± 5 puntos en el periodo del proyecto; en la operación de "
    f"agosto a octubre bajó al {pc(OP['cobertura'])}."))
R_("Prueba de hipótesis.", (
    f"**Prueba de hipótesis.** Diebold-Mariano del diseño definitivo frente a la persistencia realista: t = "
    f"{n(K['DM_vs_pers'][0], 2)} a 24 h y t = {n(T72['1-72 h']['DM_t'], 2)} a 72 h, con p unilateral < 0,001. Se rechaza H0 en "
    f"ambos casos. Frente al predespacho solo, t = {n(K['DM_vs_pre'][0], 2)} (p < 0,001): la corrección con CatBoost aporta "
    "información más allá de la que publica XM, sobre todo en temporada normal; en El Niño la diferencia no es significativa."))
R_("Lectura honesta de la robustez.", (
    "**Lectura honesta de la robustez.** Con el precio del día anterior, ningún modelo individual supera de forma consistente a "
    "la persistencia en los orígenes históricos, pero el ensamble de seis votantes sí lo hace en 4 de 5 años. Las redes "
    "neuronales resultaron muy sensibles a cambios pequeños en los datos: al corregir el ONI histórico en ±0,1 en cuatro "
    "trimestres, su error cambió hasta 3 COP/kWh y seis de las 36 conclusiones de Diebold-Mariano se invirtieron, mientras "
    "XGBoost y ARX+GARCH casi no cambiaron. El hallazgo más importante, sin embargo, es de información y no de modelado: con solo "
    f"lo publicado al corte el ensamble de 6 pierde casi toda su ventaja (rMAE {n(ENS6['rMAE'], 2)}) y el pronóstico solo es útil "
    "con el predespacho. Las variantes del diseño definitivo se eligieron observando 2026, pero la validación en los cinco años "
    "históricos lo confirma (Tabla 16). Persisten dos limitaciones: los desfases de nivel de días completos que comete el "
    "predespacho (por ejemplo, del 11 al 18 de junio de 2026) no se pueden anticipar con información pública, y el modelo hereda "
    "la amplitud de los picos que el predespacho anticipa y no ocurren."))
R_("Validación externa (Tabla 4).", (
    f"**Validación externa (Tabla 4).** El MAPE ({pc(K['MAPE'])}) es mayor que el del antecedente colombiano horario [15] (6,7 %), "
    "que evaluó 2000-2017 sin los eventos extremos de 2023-2026 y con un modelo univariado que supone conocido el precio del día "
    "anterior. Frente a Nueva Zelanda [16], el error queda por debajo del de la referencia ingenua en todos los años evaluados, "
    "mientras ninguno de sus 33 modelos lo logra; se lee como «superar al propio ingenuo en una tarea más difícil», no como "
    "comparación numérica directa. La referencia comercial más cercana es la del propio operador: el predespacho ideal de XM, "
    "que el diseño definitivo mejora en 2026 y en los cinco años históricos."))
p = P("Interpretabilidad (criterio de OE2).")
reescribir(p, texto_de(p).strip().replace("Interpretabilidad (criterio de OE2).", "**Interpretabilidad (criterio de OE2).**", 1) + (
    " En el diseño definitivo, el combinador da en promedio un peso de 0,60 al predespacho y de 0,37 al promedio de los "
    "CatBoost, de modo que cada pronóstico se puede explicar como el predespacho de XM corregido según el comportamiento "
    "reciente del mercado."))
despues(p, p, "**Validación con usuarios.** [[Completar: comparar el promedio de cada afirmación con un criterio de aceptación "
              "(por ejemplo, 4 de 5 o más en las afirmaciones de comprensión), reportar la nota global y resumir los comentarios "
              "abiertos.]]")

# ================================================================================================== 8 impacto
p = P("Empresas medianas sin equipos de analítica")
reescribir(p, texto_de(p).strip() + " El dashboard es público y no requiere instalación.")
R_("Reentrenar el ensamble de 24 h toma", (
    f"Reentrenar cada mes los tres CatBoost de 24 h toma cerca de {n(3 * cs / 60)} minutos y la actualización diaria completa "
    "unos 7 minutos en un portátil sin GPU (unas milésimas de kWh con un consumo de unos 25 W) o 9 minutos en un servidor "
    "compartido de GitHub, despreciable frente a entrenar redes profundas en centros de datos. Una mejor planeación de compra y "
    "venta puede favorecer un uso más eficiente del recurso hídrico en épocas secas."))
R_("Solo se usa información pública de XM/SIMEM", (
    "Solo se usa información pública de XM/SIMEM [21], incluido el predespacho ideal que XM publica sin restricción en su portal "
    "[39]; no se recolectan datos personales [22] y no se ejecutan transacciones. El pronóstico con predespacho se presenta para "
    "decisiones posteriores a su publicación (10:30 de D-1), no para formular ofertas, que cierran a las 08:00 [41]. Ante "
    "intervenciones transitorias de la CREG en la formación del precio, los modelos se reentrenan cada mes de forma automática y "
    "cada imagen indica la versión del modelo."))
cr = MO["24h|comercializador"]["metodos"]["rodante"]["ventaja_anio"]
gr = MO["24h|generador"]["metodos"]["rodante"]["ventaja_anio"]
R_("El costo del proyecto es de", (
    "El costo del proyecto es de $37.605.260 (Tabla 1), casi todo en horas de personal, y su operación no agrega costos. En el "
    "backtesting de 2026 a 24 h, con el método que elige el criterio de estabilidad («rodante»), las horas en que el motor "
    f"recomienda vender tuvieron un precio medio {n(gr)} COP/kWh por encima del promedio, y las de comprar, {n(cr)} COP/kWh por "
    f"debajo. Para un cliente de ejemplo de 100 kW, en el último mes evaluado esto equivale a {sg(um['rodante']['cop'] / 1e6, 2)} "
    "millones de COP. No es una ganancia garantizada: no modela contratos, capacidad ni costos de transacción."))

# ================================================================================================== 9 conclusiones
R_("El primer objetivo específico", (
    "El primer objetivo específico se cumplió. Se sincronizaron a resolución horaria las cinco variables del mercado y el ONI "
    "entre enero de 2019 y agosto de 2026, en 66.576 horas sin huecos, y la cadena de adquisición quedó automatizada: descarga "
    "cada día los datos nuevos de XM y NOAA. La caracterización identificó la componente de 24 h como dominante, con armónicos "
    "de 12 y 168 h, y esos resultados definieron las variables predictoras; el filtro de Savitzky-Golay detecta los cambios de "
    "régimen 376 h antes que el promedio móvil, una elección que se confirmó frente a Butterworth, Kalman y wavelets."))
R_("El segundo objetivo", (
    "El segundo objetivo se cumplió. Se compararon más de quince familias de modelos frente a la persistencia, con métricas de "
    "precisión, robustez en seis regímenes, costo computacional e interpretabilidad. Medir la información realmente pública al "
    "corte cambió el diseño: el ensamble de seis modelos, que reducía el error en 27 % con el precio del día anterior, pierde casi "
    "toda su ventaja con lo publicado, y el diseño definitivo corrige el predespacho de XM con tres CatBoost. Con solo "
    f"información pública, su error absoluto medio a 24 h es {n(K['MAE'])} COP/kWh (MAPE {pc(K['MAPE'])}), {n(red, 0)} % menor "
    "que el de la persistencia y significativamente menor que el del predespacho solo, y a 72 h es "
    f"{n(T72['1-72 h']['MAE'])} COP/kWh; la ventaja se confirmó en los cinco años históricos."))
R_("El tercer objetivo", (
    "El tercer objetivo se cumplió. El motor traduce el pronóstico y su incertidumbre en señales de comprar, vender o esperar para "
    "dos roles y dos horizontes, con cuatro métodos por percentil, un filtro intradía y selección por estabilidad. La biblioteca "
    "de imágenes del Anexo 1 (semáforo, mapa de calor día-hora, bandas de incertidumbre y real frente a pronóstico, con "
    "metadatos) está integrada en un dashboard publicado en la web, verificado con pruebas automatizadas, que se actualiza cada "
    "día."))
R_("El cuarto objetivo", (
    "El cuarto objetivo se cumplió en el backtesting: en 2026 el criterio de estabilidad elige «rodante» en los cuatro casos de "
    "rol y horizonte, con ventaja positiva y frecuencia de acción válida en ambas mitades del año, y en los cinco orígenes "
    "históricos el método elegido también tiene ventaja positiva en las dos mitades en todos los casos. [[Completar con la "
    "validación con usuarios: número de usuarios y resultado principal.]] Con los cuatro objetivos cumplidos se cumple el "
    "objetivo general: la plataforma procesa los datos públicos del mercado, pronostica el precio con su incertidumbre y apoya "
    "la toma de decisiones de principio a fin, en operación diaria."))
R_("Como recomendaciones para la etapa final", (
    "Como recomendaciones para una fase siguiente: (1) evaluar el pronóstico y el motor durante más meses de operación, ya que "
    f"la cobertura de la banda bajó al {pc(OP['cobertura'])} entre agosto y octubre de 2026, y considerar una ventana de "
    "calibración más corta en periodos volátiles; (2) ofrecer un pronóstico sin predespacho para los generadores, que deben "
    "ofertar antes de las 08:00 de D-1; (3) estudiar los desfases de nivel diarios del predespacho, que hoy no se anticipan con "
    "información pública; (4) incorporar al motor métricas de decisión basadas en la forma del día [8] y evaluar el criterio de "
    "estabilidad con más de dos subperiodos o con una formulación de bandits no estacionarios [37]; y (5) ampliar la validación "
    "con usuarios a un estudio formal de usabilidad."))

# ================================================================================================== 10 bibliografia y anexos
ultima = P("[37]")
for ref in (
        "[38]\tL. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush y A. Gulin, \"CatBoost: unbiased boosting with categorical "
        "features,\" en Adv. Neural Inf. Process. Syst. 31 (NeurIPS 2018), Montreal, Canadá, 2018.",
        "[39]\tXM S.A. E.S.P., \"Informes de despacho: predespacho ideal y despacho programado,\" biblioteca pública del portal "
        "xm.com.co. Consultado: oct. 2026.",
        "[40]\tV. Vovk, A. Gammerman y G. Shafer, Algorithmic Learning in a Random World. Nueva York, NY, EE. UU.: Springer, 2005.",
        "[41]\tComisión de Regulación de Energía y Gas (CREG), \"Resolución CREG 025 de 1995, por la cual se establece el Código de "
        "Redes,\" Bogotá, Colombia, 1995."):
    ultima = despues(ultima, ultima, ref)
R_("Los anexos 1, 3 y 5 están en el repositorio", (
    "Los anexos 1, 3 y 5 están en el repositorio del proyecto (github.com/Ziskito/xm-spot-price-predictor): formulación en docs/, "
    "código fuente en notebooks/, src/, dashboard/ y scripts_experimento/, y datos y predicciones en data/. El anexo 2 corresponde "
    "a docs/tabla_riesgos_proyecto.xlsx; el anexo 4, al manual de usuario (docs/Manual_Usuario_augur.pdf); el anexo 7, a la "
    "bitácora del README y al historial de commits; y el anexo 8, al video del proyecto ([[completar enlace]]). El dashboard está "
    "publicado en augurpredictor.streamlit.app."))

doc.save(str(DESTINO))
print("Guardado:", DESTINO)
