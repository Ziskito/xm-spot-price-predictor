# -*- coding: utf-8 -*-
"""
Rellena docs/TEMPLATE_INFORME_AVANCE_PF.docx (plantilla oficial del informe de avance, semana 10).

No toca la plantilla ni el informe ABET anterior (generar_informe_avance.py). La bibliografia se lee de
ese script para tener una sola fuente. Las cifras son las mismas, ya verificadas, del informe ABET.
"""
import ast
import copy
import json
import re
from datetime import date
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from ecuaciones_omml import tabla_ecuacion

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
PLANTILLA = RAIZ / "docs" / "TEMPLATE_INFORME_AVANCE_PF.docx"
SALIDA = RAIZ / "docs" / "Informe_Avance_PF_Barcelo_Dede.docx"
FIG = RAIZ / "scripts_experimento" / "informe_avance_assets"
FECHA_CORTE = date(2026, 9, 26)
FECHA_TXT = "26/09/2026"
INICIO_SEMESTRE = date(2026, 7, 27)
JUAN = "Juan David Barceló Barraza"
RAFA = "Rafael Andrés Dede Perdomo"

src = (RAIZ / "scripts_experimento" / "generar_informe_avance.py").read_text(encoding="utf-8")
BIB = ast.literal_eval(re.search(r"^BIB = (\{.*?^\})", src, re.S | re.M).group(1))
BIB.update({
    "kapoor2023b": 'G. Kapoor, N. Wichitaksorn y W. Zhang, "Analyzing and forecasting electricity price using regime-switching models: The case of New Zealand market," *J. Forecast.*, vol. 42, no. 8, pp. 2011–2026, 2023.',
    "nasiadka2022": 'J. Nasiadka, W. Nitka y R. Weron, "Calibration window selection based on change-point detection for forecasting electricity prices," en *Computational Science – ICCS 2022*, Londres, Reino Unido, LNCS. Springer, 2022, arXiv: 2204.00872.',
    "baranowski2019": 'R. Baranowski, Y. Chen y P. Fryzlewicz, "Narrowest-over-threshold detection of multiple change points and change-point-like features," *J. R. Stat. Soc. Ser. B*, vol. 81, no. 3, pp. 649–672, 2019, doi: 10.1111/rssb.12322.',
    "chang2024": 'R. I. Chang, C. H. Wang, L. C. Wei y Y. F. Lu, "LSTM with short-term bias compensation to determine trading strategy under black swan events of Taiwan ETF50 stock," *Appl. Sci.*, vol. 14, art. 8576, 2024.',
    "singh2027": 'G. A. Singh, "Regime-aware deep learning for financial forecasting: An adaptive short-term bias compensation framework for the Warsaw Stock Exchange," *Expert Syst. Appl.*, vol. 332, art. 133584, 2027.',
})
ED = json.loads((RAIZ / "data/processed/resultados/informe_avance/error_diario_2026.json").read_text(encoding="utf-8"))


def dec(x, d=2):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


ENCONTRADAS = ["huisman2003", "kapoor2023b", "nasiadka2022", "baranowski2019", "chang2024", "singh2027"]
ORDEN = []


def cita(claves):
    nums = []
    for k in claves.split(","):
        k = k.strip()
        assert k in BIB, k
        assert k not in ENCONTRADAS, k
        if k not in ORDEN:
            ORDEN.append(k)
        nums.append(ORDEN.index(k) + 1)
    return ", ".join(f"[{n}]" for n in nums)


def expandir(texto):
    return re.sub(r"\{c:([^}]+)\}", lambda m: cita(m.group(1)), texto)


# numeracion fija de tablas, figuras y ecuaciones: el texto las cita por nombre y el script verifica el orden
T = {k: i for i, k in enumerate(["objetivos", "fuentes", "variables", "walkforward", "72h", "desempeno",
                                 "motor_elegido", "motor_mitades", "fases", "gantt"], 1)}
F = {k: i for i, k in enumerate(["bloques", "serie", "periodograma", "filtros", "walkforward", "pronostico",
                                 "motor", "dashboard", "mitades", "ganancia"], 1)}
E = {k: i for i, k in enumerate(["periodograma", "savgol", "pearson", "arx", "qra", "conforme", "metricas",
                                 "dm", "motor", "maximin"], 1)}

# ------------------------------------------------------------------------------------------------
# utilidades XML (Arial, como la plantilla)
# ------------------------------------------------------------------------------------------------
doc = docx.Document(str(PLANTILLA))
BODY = doc.element.body


def _rpr(b=False, i=False, sz=None, color=None, vert=None):
    rpr = OxmlElement("w:rPr")
    fn = OxmlElement("w:rFonts")
    for a in ("ascii", "hAnsi", "cs"):
        fn.set(qn(f"w:{a}"), "Arial")
    rpr.append(fn)
    if b:
        rpr.append(OxmlElement("w:b"))
    if i:
        rpr.append(OxmlElement("w:i"))
    if color:
        c = OxmlElement("w:color"); c.set(qn("w:val"), color); rpr.append(c)
    if sz:
        for tag in ("w:sz", "w:szCs"):
            s = OxmlElement(tag); s.set(qn("w:val"), str(int(sz * 2))); rpr.append(s)
    if vert:
        v = OxmlElement("w:vertAlign"); v.set(qn("w:val"), vert); rpr.append(v)
    return rpr


def _run(texto, **kw):
    r = OxmlElement("w:r")
    r.append(_rpr(**kw))
    t = OxmlElement("w:t"); t.set(qn("xml:space"), "preserve"); t.text = texto
    r.append(t)
    return r


def _runs(texto, sz=None, color=None, b_all=False, i_all=False, _exp=True):
    """markup: **negrita**, *cursiva*, ~sub~, ^sup^"""
    if _exp:
        texto = expandir(texto)
    out = []
    for tok in re.split(r"(\*\*.+?\*\*|\*[^*]+?\*)", texto):
        if not tok:
            continue
        if tok.startswith("**"):
            out += _runs(tok[2:-2], sz, color, True, i_all, False)
        elif tok.startswith("*") and len(tok) > 1:
            out += _runs(tok[1:-1], sz, color, b_all, True, False)
        else:
            for t2 in re.split(r"(~[^~]+?~|\^[^^]+?\^)", tok):
                if not t2:
                    continue
                if t2.startswith("~") and len(t2) > 2:
                    out.append(_run(t2[1:-1], vert="subscript", b=b_all, i=i_all, sz=sz, color=color))
                elif t2.startswith("^") and len(t2) > 2:
                    out.append(_run(t2[1:-1], vert="superscript", b=b_all, i=i_all, sz=sz, color=color))
                else:
                    out.append(_run(t2, b=b_all, i=i_all, sz=sz, color=color))
    return out


def _par(texto="", jc="both", after=120, before=0, ind_left=None, hanging=None, first=None, sz=None,
         color=None, b_all=False, i_all=False, keep_next=False, line=240, tabs=None):
    p = OxmlElement("w:p")
    ppr = OxmlElement("w:pPr")
    if keep_next:
        ppr.append(OxmlElement("w:keepNext"))
    if tabs:
        tb = OxmlElement("w:tabs")
        for val, pos in tabs:
            t = OxmlElement("w:tab"); t.set(qn("w:val"), val); t.set(qn("w:pos"), str(pos)); tb.append(t)
        ppr.append(tb)
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:before"), str(before)); sp.set(qn("w:after"), str(after))
    sp.set(qn("w:line"), str(line)); sp.set(qn("w:lineRule"), "auto")
    ppr.append(sp)
    if ind_left is not None or first is not None:
        ind = OxmlElement("w:ind")
        if ind_left is not None:
            ind.set(qn("w:left"), str(ind_left))
        if hanging:
            ind.set(qn("w:hanging"), str(hanging))
        if first:
            ind.set(qn("w:firstLine"), str(first))
        ppr.append(ind)
    j = OxmlElement("w:jc"); j.set(qn("w:val"), jc); ppr.append(j)
    p.append(ppr)
    for r in _runs(texto, sz=sz, color=color, b_all=b_all, i_all=i_all):
        p.append(r)
    return p


def _clonar(proto, texto):
    """copia un parrafo de la plantilla (conserva estilo, numeracion y formato) y cambia su texto."""
    p = copy.deepcopy(proto)
    for e in list(p):
        if e.tag != qn("w:pPr"):
            p.remove(e)
    ppr = p.find(qn("w:pPr"))
    if ppr.find(qn("w:keepNext")) is None:
        ppr.insert(1 if ppr.find(qn("w:pStyle")) is not None else 0, OxmlElement("w:keepNext"))
    base = ppr.find(qn("w:rPr"))
    for tok in re.split(r"(\*\*.+?\*\*)", expandir(texto)):
        if not tok:
            continue
        r = OxmlElement("w:r")
        rpr = copy.deepcopy(base) if base is not None else OxmlElement("w:rPr")
        if tok.startswith("**"):
            tok = tok[2:-2]
            if rpr.find(qn("w:b")) is None:
                rpr.insert(1, OxmlElement("w:b"))
        r.append(rpr)
        t = OxmlElement("w:t"); t.set(qn("xml:space"), "preserve"); t.text = tok
        r.append(t)
        p.append(r)
    return p


def _sombra(tcpr, color):
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color); tcpr.append(shd)


def _campo(p, instr, rpr_base):
    for kind, texto in (("begin", None), ("instr", instr), ("separate", None), ("text", "1"), ("end", None)):
        r = OxmlElement("w:r"); r.append(copy.deepcopy(rpr_base))
        if kind == "instr":
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr; r.append(it)
        elif kind == "text":
            t = OxmlElement("w:t"); t.text = texto; r.append(t)
        else:
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), kind); r.append(fc)
        p.append(r)


class Cursor:
    nt = nf = ne = 0

    def __init__(self, el):
        self.el = el

    def add(self, el):
        self.el.addnext(el)
        self.el = el
        return el

    def p(self, texto, **kw):
        return self.add(_par(texto, **kw))

    def clon(self, proto, texto):
        return self.add(_clonar(proto, texto))

    def viñetas(self, items, after=60):
        for it in items:
            self.add(_par("•\t" + it, ind_left=567, hanging=283, after=after, tabs=[("left", 567)]))

    def dato(self, etiqueta, valor, after=0):
        return self.add(_par(f"**{etiqueta}** {valor}", jc="left", after=after))

    def cierre_actividad(self, tiempo, peso=None, avance=None, after=240):
        txt = f"**Tiempo:** {tiempo}."
        if peso is not None:
            txt += f"\t**Peso en el objetivo:** {peso} %."
        if avance is not None:
            txt += f"\t**Avance de la actividad:** {avance} %."
        return self.add(_par(txt, jc="left", after=after, tabs=[("left", 2700), ("left", 5900)]))

    def ecuacion(self, clave, latex):
        Cursor.ne += 1
        assert E[clave] == self.ne, (clave, self.ne)
        self.add(tabla_ecuacion(latex, self.ne, ancho_twips=9400, fuente="Arial"))
        return self.p("", after=0, sz=4)

    def titulo_tabla(self, clave, texto):
        Cursor.nt += 1
        assert T[clave] == self.nt, (clave, self.nt)
        return self.p(f"Tabla {self.nt}. {texto}", jc="center", after=60, before=120, sz=10, i_all=True,
                      keep_next=True)

    def figura(self, clave, archivo, texto, ancho_cm=15.0):
        Cursor.nf += 1
        assert F[clave] == self.nf, (clave, self.nf)
        tmp = doc.add_paragraph()
        tmp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        tmp.paragraph_format.keep_with_next = True
        tmp.paragraph_format.space_before = Pt(6)
        tmp.paragraph_format.space_after = Pt(2)
        tmp.add_run().add_picture(str(FIG / archivo), width=Cm(ancho_cm))
        self.add(tmp._p)
        return self.p(f"Figura {self.nf}. {texto}", jc="center", after=200, sz=10, i_all=True)

    def tabla(self, filas, anchos_cm, sz=9, alinear=None, negrita_col0=False, sombras=None, encabezado=True):
        """bordes simples y encabezado en negrita sobre gris claro, como la Tabla 1 de la plantilla."""
        ncol = len(filas[0])
        t = doc.add_table(rows=len(filas), cols=ncol)
        tbl = t._tbl
        tblpr = tbl.tblPr
        for e in list(tblpr):
            tblpr.remove(e)
        tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), "5000"); tw.set(qn("w:type"), "pct"); tblpr.append(tw)
        jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "center"); tblpr.append(jc)
        bd = OxmlElement("w:tblBorders")
        for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{lado}")
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "auto")
            bd.append(e)
        tblpr.append(bd)
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblpr.append(lay)
        total = sum(anchos_cm)
        tw_dxa = [int(9400 * a / total) for a in anchos_cm]
        for gc, w in zip(tbl.tblGrid.findall(qn("w:gridCol")), tw_dxa):
            gc.set(qn("w:w"), str(w))
        for ri, fila in enumerate(filas):
            row = t.rows[ri]
            trpr = row._tr.get_or_add_trPr()
            enc = ri == 0 and encabezado
            trpr.append(OxmlElement("w:tblHeader" if enc else "w:cantSplit"))
            for ci, valor in enumerate(fila):
                cell = row.cells[ci]
                tcpr = cell._tc.get_or_add_tcPr()
                for e in list(tcpr):
                    tcpr.remove(e)
                w = OxmlElement("w:tcW"); w.set(qn("w:w"), str(tw_dxa[ci])); w.set(qn("w:type"), "dxa"); tcpr.append(w)
                color = "D9D9D9" if enc else (sombras(ri, ci) if sombras else None)
                if color:
                    _sombra(tcpr, color)
                mar = OxmlElement("w:tcMar")
                for lado, v in (("top", 30), ("left", 60), ("bottom", 30), ("right", 60)):
                    e = OxmlElement(f"w:{lado}"); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); mar.append(e)
                tcpr.append(mar)
                va = OxmlElement("w:vAlign"); va.set(qn("w:val"), "center"); tcpr.append(va)
                p = cell.paragraphs[0]._p
                for e in list(p):
                    p.remove(e)
                for li, linea in enumerate(str(valor).split("\n")):
                    if li > 0:
                        p = OxmlElement("w:p"); cell._tc.append(p)
                    ppr = OxmlElement("w:pPr")
                    sp = OxmlElement("w:spacing"); sp.set(qn("w:after"), "0"); sp.set(qn("w:before"), "0")
                    sp.set(qn("w:line"), "240"); sp.set(qn("w:lineRule"), "auto"); ppr.append(sp)
                    al = "center" if enc else (alinear[ci] if alinear else "left")
                    j = OxmlElement("w:jc"); j.set(qn("w:val"), al); ppr.append(j)
                    ppr.append(_rpr(sz=sz))
                    p.append(ppr)
                    for r in _runs(linea, sz=sz, b_all=enc or (negrita_col0 and ci == 0)):
                        p.append(r)
        if len(filas) <= 12:
            for tr in tbl.findall(qn("w:tr"))[:-1]:
                for pp in tr.iter(qn("w:p")):
                    pp.find(qn("w:pPr")).insert(0, OxmlElement("w:keepNext"))
        self.add(tbl)
        return self.p("", after=60, sz=6)


def parrafo(texto, estilo=None):
    for p in doc.paragraphs:
        if p.text.strip().startswith(texto) and (estilo is None or p.style.name == estilo):
            return p
    raise KeyError(texto)


# ------------------------------------------------------------------------------------------------
# 0) prototipos de la plantilla y limpieza de los marcadores de posicion
# ------------------------------------------------------------------------------------------------
H1_CUANT = parrafo("CUANTIFICACIÓN DEL PORCENTAJE", "Heading 1")._p
H1_EXPL = parrafo("EXPLICACIÓN DETALLADA", "Heading 1")._p
H1_CRONO = parrafo("CRONOGRAMA CON", "Heading 1")._p
H1_DIFIC = parrafo("DIFICULTADES ENCONTRADAS", "Heading 1")._p
H1_PLAN = parrafo("METODOLOGIA Y ACTIVIDADES A REALIZAR", "Heading 1")._p
H1_REFS = parrafo("REFERENCIAS EN EL DOCUMENTO", "Heading 1")._p
H1_ENC = parrafo("ENCONTRADAS HASTA EL MOMENTO", "Heading 1")._p
P_H2 = copy.deepcopy(parrafo("2.1 Actividades realizadas", "Heading 2")._p)
P_ACT = copy.deepcopy(parrafo("Actividad", "Title")._p)
P_SEMANA = copy.deepcopy(parrafo("SEMANA 11 - 12")._p)
T_OBJ = doc.tables[0]._tbl
CAP_T1 = parrafo("Tabla 1. Cuantificación")._p

conservar = {H1_CUANT, H1_EXPL, H1_CRONO, H1_DIFIC, H1_PLAN, H1_REFS, H1_ENC, T_OBJ, CAP_T1,
             parrafo("REFERENCIAS", "Heading 1")._p}
elems = list(BODY)
i0 = elems.index(H1_CUANT)
for e in elems[i0:]:
    if e.tag in (qn("w:p"), qn("w:tbl")) and e not in conservar:
        BODY.remove(e)
for pp in list(BODY):
    for b in pp.iter(qn("w:lastRenderedPageBreak")):
        b.getparent().remove(b)

# portada
titulo = parrafo("Título del proyecto final", "Title")._p
for r in titulo.findall(qn("w:r"))[1:]:
    titulo.remove(r)
titulo.find(qn("w:r")).find(qn("w:t")).text = ("Diseño de una plataforma de procesamiento y analítica de datos "
                                              "para el pronóstico del precio de la energía y el apoyo en la toma "
                                              "de decisiones")
ases = parrafo("Asesor (es):")._p
c = Cursor(ases)
c.p("José Daniel Soto Ortiz", jc="center", after=0)
c.p("Daniela María Charris Stand", jc="center", after=360)
c.p("**Estudiantes:**", jc="center", after=120)
c.p(f"{JUAN} — Código 200181896", jc="center", after=0)
c.p(f"{RAFA} — Código 200180841", jc="center", after=0)
fecha_p = parrafo("Día/ mes / año")._p
for r in fecha_p.findall(qn("w:r"))[1:]:
    fecha_p.remove(r)
fecha_p.find(qn("w:r")).find(qn("w:t")).text = FECHA_TXT
# se retiran los parrafos vacios que empujan la portada (compensan las lineas de nombres agregadas)
cont = list(BODY)
for e in cont[cont.index(titulo) + 1: cont.index(ases)][-4:] + cont[cont.index(ases) + 6: cont.index(fecha_p)][:3]:
    if e.tag == qn("w:p") and not "".join(t.text or "" for t in e.iter(qn("w:t"))).strip():
        BODY.remove(e)

# encabezado: fecha, estudiantes y paginacion automatica
hdr = doc.sections[0].header
for tc in hdr._element.iter(qn("w:tc")):
    texto = "".join(t.text or "" for t in tc.iter(qn("w:t")))
    ps = tc.findall(qn("w:p"))
    if texto.startswith("Fecha"):
        ps[0].append(_run(FECHA_TXT))
    elif texto.startswith("Estudiantes"):
        ps[-1].append(_run("J. D. Barceló Barraza", sz=8))
        ps[-1].append(OxmlElement("w:r"))
        ps[-1][-1].append(OxmlElement("w:br"))
        ps[-1].append(_run("R. A. Dede Perdomo", sz=8))
    elif "ginas" in texto:
        p = ps[0]
        base = copy.deepcopy(p.findall(qn("w:r"))[1].find(qn("w:rPr")))
        for r in p.findall(qn("w:r")):
            p.remove(r)
        r = OxmlElement("w:r"); r.append(copy.deepcopy(base))
        t = OxmlElement("w:t"); t.set(qn("xml:space"), "preserve"); t.text = "Página "; r.append(t); p.append(r)
        _campo(p, " PAGE ", base)
        r = OxmlElement("w:r"); r.append(copy.deepcopy(base))
        t = OxmlElement("w:t"); t.set(qn("xml:space"), "preserve"); t.text = " de "; r.append(t); p.append(r)
        _campo(p, " NUMPAGES ", base)

# tabla de contenido (campo TOC; Word lo actualiza al exportar)
toc_t = parrafo("TABLA DE CONTENIDO", "Title")._p
nxt = toc_t.getnext()
while nxt is not None and nxt is not H1_CUANT:
    siguiente = nxt.getnext()
    if nxt.tag == qn("w:p"):
        BODY.remove(nxt)
    nxt = siguiente
p_toc = _par("", jc="left", after=0)
_campo(p_toc, ' TOC \\o "1-2" \\h \\z \\u ', _rpr(sz=10))
toc_t.addnext(p_toc)
docx.text.paragraph.Paragraph(H1_CUANT, None).paragraph_format.page_break_before = True
for nombre, negrita in (("toc 1", True), ("toc 2", False)):
    try:
        st = doc.styles[nombre]
    except KeyError:
        continue
    st.font.all_caps = False
    st.font.small_caps = False
    st.font.bold = negrita

# ================================================================================================
# 1. CUANTIFICACION
# ================================================================================================
AV = {"OE1": 100, "OE2": 90, "OE3": 70, "OE4": 35}
AV_G = sum(AV.values()) / 4
OBJ = {
    "General": "Diseñar, implementar y validar una plataforma de procesamiento de datos que permita el análisis y "
               "pronóstico del precio de la energía en el mercado eléctrico colombiano para la toma de decisiones.",
    "OE1": "Adquirir, sincronizar y caracterizar temporal y espectralmente las series históricas de las variables "
           "del mercado, mediante técnicas de procesamiento de datos, para seleccionar variables predictoras.",
    "OE2": "Implementar y evaluar diferentes modelos de pronóstico del precio de bolsa, comparando precisión frente "
           "a persistencia, robustez, costo computacional e interpretabilidad.",
    "OE3": "Diseñar e implementar un motor de decisión que traduzca los pronósticos y su incertidumbre en señales de "
           "compra, venta o espera.",
    "OE4": "Validar el funcionamiento del motor para la toma de decisiones.",
}
c = Cursor(H1_CUANT)
c.p("Los objetivos se transcriben del Anexo 1 {c:anexo1}. Los cuatro objetivos específicos son necesarios para "
    "cumplir el general, por lo que cada uno pesa 25 %. El avance de cada objetivo específico no es una "
    "apreciación: es la suma ponderada del avance de sus actividades, detalladas en la Sección 2, "
    "A~OE~ = Σ~i~ p~i~ · a~i~, donde p~i~ es el peso de la actividad i en el objetivo (Σ~i~ p~i~ = 100 %) y a~i~ su "
    f"avance. El avance del objetivo general es 0,25 · (100 + 90 + 70 + 35) = {AV_G:.2f} % ≈ {round(AV_G)} %."
    .replace("73.75", "73,75"), before=120)
BODY.remove(CAP_T1)
c.titulo_tabla("objetivos", f"Cuantificación del porcentaje de logro de los objetivos (corte: {FECHA_TXT}).")
BODY.remove(T_OBJ)
c.add(T_OBJ)
tbl = docx.table.Table(T_OBJ, doc)
nueva = copy.deepcopy(tbl.rows[-1]._tr)
T_OBJ.append(nueva)
tbl = docx.table.Table(T_OBJ, doc)


def celda(cel, texto, b=False, jc="left"):
    p = cel.paragraphs[0]._p
    for e in list(p):
        if e.tag != qn("w:pPr"):
            p.remove(e)
    for extra in cel._tc.findall(qn("w:p"))[1:]:
        cel._tc.remove(extra)
    ppr = p.find(qn("w:pPr"))
    if ppr is None:
        ppr = OxmlElement("w:pPr"); p.insert(0, ppr)
    for e in ppr.findall(qn("w:jc")):
        ppr.remove(e)
    j = OxmlElement("w:jc"); j.set(qn("w:val"), jc); ppr.append(j)
    for r in _runs(texto, sz=9, b_all=b):
        p.append(r)


for fila in tbl.rows:
    fila._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
etiquetas = [("General", "General", f"{round(AV_G)} %", "100 %")] + [
    (f"Específico {k}", f"OE{k}", f"{AV[f'OE{k}']} %", "25 %") for k in range(1, 5)]
for fila, (lab, clave, avance, peso) in zip(tbl.rows[1:], etiquetas):
    celda(fila.cells[0], lab, b=True)
    celda(fila.cells[1], OBJ[clave])
    celda(fila.cells[2], avance, jc="center")
    celda(fila.cells[3], peso, jc="center")
c.p("", after=60, sz=6)

# ================================================================================================
# 2. METODOLOGIA Y ACTIVIDADES REALIZADAS
# ================================================================================================
c = Cursor(H1_EXPL)
c.p("El proyecto sigue las cinco fases de la metodología del Anexo 1 {c:anexo1}: cadena de procesamiento y "
    "caracterización (OE1), extracción de características y modelado (OE2), comparación formal y motor de decisión "
    "(OE3), imágenes de decisión y validación (OE3 y OE4) y cierre. Todo elemento del sistema (variable, modelo, "
    "combinador, calibración o regla de decisión) se definió con el mismo procedimiento de ingeniería: (1) se "
    "formula frente a una alternativa o línea base explícita; (2) se evalúa estrictamente fuera de muestra, con "
    "entrenamiento en 2019-2025, ajuste de hiperparámetros contra 2025 y prueba en enero-agosto de 2026 "
    "(5.184 horas nunca vistas); (3) se adopta solo si la mejora es estadísticamente significativa en la prueba de "
    "Diebold-Mariano (p < 0,05) {c:diebold1995} y, cuando el procedimiento tiene aleatoriedad, si se repite en 10 "
    "particiones distintas; y (4) se documenta en la bitácora del repositorio, incluidos los intentos que no "
    "mejoraron. Este protocolo sigue las buenas prácticas del área de pronóstico de precios de electricidad "
    "{c:lago2021}, {c:weron2014} y evita dos errores "
    "comunes en pronóstico de precios: la fuga de información futura y la selección del modelo con el mismo "
    "periodo con el que se reporta. La Figura 1 muestra la arquitectura resultante y su relación con los objetivos.",
    before=120)
c.figura("bloques", "f_diagrama_bloques.png", "Diagrama de bloques de la plataforma y su relación con los objetivos "
         "específicos. Fuente: elaboración propia.", ancho_cm=12.5)
c.p("En cada actividad, el **tiempo** es la duración asignada en el cronograma del Anexo 1 (la Sección 3 compara lo "
    "planeado con lo ejecutado) o, para las actividades que no figuran en él, la duración estimada de ejecución; el "
    "**peso** es su participación en el objetivo y el **avance**, su grado de terminación a la fecha de corte.")

def h2(k):
    global c
    c.add(_clonar(P_H2, f"2.{k} Actividades realizadas que contribuyen al logro del objetivo específico {k}:"))
    c.p(f"*OE{k}. {OBJ[f'OE{k}']}* **Avance: {AV[f'OE{k}']} %.**", after=160, before=60)


def actividad(titulo):
    c.add(_clonar(P_ACT, titulo))


# ---------------------------------------------------------------- OE1
h2(1)
actividad("Adquisición automatizada y sincronización de las series en un dataset maestro horario")
c.p("Se implementó una función propia de descarga contra la API REST pública de XM {c:xm}, porque la librería "
    "*pydataxm* resultó incompatible con la versión de pandas del entorno. La función consulta por lotes las "
    "métricas horarias de precio de bolsa, demanda real y generación total, y las diarias de volumen útil de "
    "embalses y aportes hídricos; un segundo lector toma el Índice Oceánico del Niño (ONI) publicado por NOAA "
    "{c:noaa}. Las seis series se alinearon sobre una malla horaria única en hora legal de Colombia (UTC−5, sin "
    "horario de verano), con marcas de tiempo ISO 8601 {c:iso8601} para unir sin ambigüedad fuentes de resolución "
    "horaria, diaria y mensual. El resultado es un dataset maestro de **66.576 horas continuas** (1 de enero de 2019 "
    "a 5 de agosto de 2026), sin huecos ni duplicados, verificado con una prueba de continuidad del índice temporal "
    "(diferencia constante de 1 h entre registros consecutivos). La Tabla 2 resume las fuentes. La Figura 2 muestra "
    "el promedio diario del precio (cada punto resume las 24 horas de un día) con los episodios de El Niño y La "
    "Niña marcados según el ONI; se usa para ver de un vistazo la relación entre el nivel del precio y el clima, que "
    "es la "
    "señal principal.")
c.titulo_tabla("fuentes", "Fuentes de datos, resolución original y tratamiento para llevarlas a resolución horaria.")
c.tabla([
    ["Variable", "Fuente", "Resolución original", "Unidad", "Paso a resolución horaria"],
    ["Precio de bolsa", "XM", "Horaria", "COP/kWh", "Directo"],
    ["Demanda real del sistema", "XM", "Horaria", "kWh", "Directo"],
    ["Generación total del sistema", "XM", "Horaria", "kWh", "Directo"],
    ["Volumen útil de embalses", "XM", "Diaria", "kWh (energía)", "Interpolación lineal"],
    ["Aportes hídricos", "XM", "Diaria", "kWh (energía)", "Interpolación lineal"],
    ["ONI (anomalía de temperatura del Pacífico)", "NOAA CPC", "Mensual", "°C", "Rampa causal (sin datos futuros)"],
], [4.6, 1.8, 2.6, 2.4, 4.4], sz=8.5, alinear=["left", "center", "center", "center", "left"])
c.figura("serie", "f_serie_oni.png", "Precio de bolsa promedio diario 2019-2026 y episodios ENSO (naranja: El Niño, "
         "ONI ≥ 0,5; azul: La Niña, ONI ≤ −0,5). Fuente: elaboración propia con datos de XM y NOAA.", ancho_cm=14.0)
c.dato("Responsables:", JUAN + ".")
c.dato("Herramientas:", "Python 3, requests y pandas; API REST de XM (servapibi.xm.com.co); datos ONI de NOAA CPC; "
       "Jupyter (notebooks 01 y 02); Git y GitHub.")
c.cierre_actividad("7 días", 30, 100)

actividad("Tratamiento de faltantes y atípicos, y construcción de variables indicadoras")
c.p("Las series diarias se llevaron a resolución horaria por **interpolación lineal**. La alternativa de escalón "
    "(repetir el valor diario las 24 horas) se comparó en una ablación 2×2: la interpolación mejoró el horizonte de "
    "72 h sin afectar el de 24 h, y es segura frente a fuga de información porque la variable entra al modelo "
    "rezagada al menos 24 h. Los valores extremos del precio **no se eliminaron**: la distribución tiene una cola "
    "larga real (mediana 242,28 COP/kWh, percentil 99 de 1.449,73 y máximo de 2.675,65) que contiene la información "
    "de los regímenes secos; en lugar de recortarla, los modelos que lo requieren trabajan con el logaritmo del "
    "precio, dado que la media (336,7) supera ampliamente a la mediana. Se construyeron dos variables de contexto: "
    "un **indicador de pandemia** (25 de marzo a 31 de agosto de 2020), que conserva un año de datos en vez de "
    "excluir 2020, y el **ONI como variable continua** con una rampa causal que solo usa valores ya publicados; "
    "interpolar el ONI con el mes siguiente habría introducido información futura. Durante la auditoría de "
    "reproducibilidad del 24 de septiembre se detectó que la demanda del 4-5 de agosto de 2026 llegaba de la fuente "
    "entre 70 % y 86 % por debajo de su valor real; se corrigió y se reentrenó toda la cadena (Sección 4).")
c.dato("Responsables:", JUAN + ".")
c.dato("Herramientas:", "Python 3, pandas y numpy; notebooks 02 y 03; scripts de auditoría de datos del repositorio.")
c.cierre_actividad("14 días", 20, 100)

actividad("Caracterización temporal y espectral del precio de bolsa")
c.p("Tras retirar la tendencia, se estimó el periodograma de la serie horaria {c:oppenheim2010} (Ecuación "
    f"{E['periodograma']}), con frecuencia de muestreo f~s~ = 1 muestra/h, de modo que la frecuencia de Nyquist es "
    "0,5 ciclos/h y el período en horas es 1/f~k~:")
c.ecuacion("periodograma", [r"P(f_k)=\frac{1}{N}\left|\sum_{n=0}^{N-1}{x[n]\,e^{-j2\pi kn/N}}\right|^2",
                            r"f_k=\frac{k\,f_s}{N}"])
c.p("La componente de **24 h** es la dominante del espectro: su pico concentra por sí solo el 6,8 % de la potencia "
    "entre 2 y 2.000 h. Le siguen el armónico de 12 h, una componente semanal de 168 h y ciclos de 20 a 42 días "
    "asociados a la hidrología (Figura 3). Este resultado es una decisión de diseño, no solo un diagnóstico: fija "
    "los rezagos de 24 y 168 h y los armónicos de calendario que usan todos los modelos (Actividad 2.1).")
c.figura("periodograma", "f_periodograma.png", "Periodograma del precio de bolsa sin tendencia, enero de 2019 a "
         "agosto de 2026. Líneas rojas: períodos de 12, 24 y 168 h.", ancho_cm=13.0)
c.dato("Responsables:", JUAN + ".")
c.dato("Herramientas:", "Python 3, numpy, scipy.signal (periodograma) y matplotlib; notebook 03.")
c.cierre_actividad("7 días", 25, 100)

actividad("Comparación de filtros de tendencia: promedio móvil frente a Savitzky-Golay")
c.p("Para seguir el nivel del precio durante los cambios de régimen se compararon un promedio móvil de 30 días y "
    "un filtro de Savitzky-Golay {c:savitzky1964}, que ajusta por mínimos cuadrados un polinomio de orden 3 en cada "
    "ventana de 2M + 1 = 721 h. Ese ajuste equivale a un filtro FIR de coeficientes fijos c~m~:")
c.ecuacion("savgol", [r"\hat{y}[n]=\sum_{m=-M}^{M}{c_m\,x[n+m]}", r"M=360"])
c.p("Las métricas fueron el tiempo que tarda cada filtro en alcanzar el 90 % de la subida real durante El Niño "
    "2023-2024 y la desviación estándar del residuo. Savitzky-Golay alcanzó el 90 % de la subida **376 h antes** "
    "que el promedio móvil (348 h frente a 724 h) y dejó menos ruido residual (117,70 frente a 138,03 COP/kWh), "
    "porque conserva los puntos de giro que el promedio móvil aplana (Figura 4). Como su ventana es centrada, se usa "
    "para caracterizar la serie y no como variable predictora, lo que evitaría usar datos futuros.")
c.figura("filtros", "f_filtros.png", "Filtros de tendencia sobre el precio horario en el tramo de El Niño "
         "2023-2024.", ancho_cm=13.0)
c.dato("Responsables:", JUAN + ".")
c.dato("Herramientas:", "Python 3, scipy.signal (savgol_filter), pandas y matplotlib; notebook 03.")
c.cierre_actividad("7 días", 15, 100)

actividad("Análisis de correlación entre precio, hidrología y demanda")
c.p(f"Se calculó el coeficiente de Pearson (Ecuación {E['pearson']}) entre el precio y cada variable, con distintos "
    "rezagos y ventanas de promedio:")
c.ecuacion("pearson", r"r_{xy}=\frac{\sum_t{(x_t-\bar{x})(y_t-\bar{y})}}{\sqrt{\sum_t{(x_t-\bar{x})^2}\,\sum_t{(y_t-\bar{y})^2}}}")
c.p("La correlación con el precio fue positiva para el ONI (0,387), la demanda (0,325) y la generación (0,323), y "
    "negativa para los aportes (−0,286) y los embalses (−0,164); los aportes pesan más promediados a 7 días "
    "(−0,309). El análisis también mostró que el índice climático no basta: el mayor pico diario del periodo "
    "(2.499 COP/kWh, 30 de septiembre de 2024) ocurrió con ONI neutro, cuando el volumen útil de los embalses era "
    "el más bajo de todos los septiembres de 2019 a 2025. Por eso la hidrología acumulada entró al modelo con 12 "
    "variables. Dos variables se **excluyeron** por fuga de información: la demanda y la generación "
    "contemporáneas, que se fijan en el mismo despacho que el precio, y el precio de oferta marginal, cuya "
    "correlación con el precio es 0,9998.")
c.dato("Responsables:", JUAN + ".")
c.dato("Herramientas:", "Python 3, pandas, scipy.stats y seaborn; notebook 03.")
c.cierre_actividad("7 días", 10, 100)

# ---------------------------------------------------------------- OE2
h2(2)
actividad("Extracción de características compartidas")
c.p("Se construyó un único conjunto de 40 variables, compartido por todos los modelos para que las comparaciones "
    "sean justas. La regla de diseño es que **toda variable se desplaza al menos 24 h** antes de calcularse, de modo "
    "que en el instante de corte solo se usa información ya publicada; una variable con rezago de 1 h que inflaba la "
    "precisión de forma irreal se eliminó por esta razón. Cada familia se justifica con un resultado de OE1 "
    "(Tabla 3). Los festivos se incorporaron después de probarlos: reducen el error de N-BEATSx en esos días de "
    "74,1 a 62,6 COP/kWh (p = 0,0064).")
c.p("De las fuentes públicas se descargaron, para enero de 2019 a agosto de 2026, 66.576 registros horarios de "
    "cada una de las tres series horarias de XM (precio de bolsa, demanda real y generación total), 2.776 registros "
    "diarios de cada una de las dos series hidrológicas (volumen útil de embalses y aportes hídricos) y los 90 valores "
    "trimestrales publicados del ONI (de DJF 2019 a MJJ 2026). Cada familia de variables sale de una sola de esas "
    "fuentes: las de precio rezagado, medias móviles y volatilidad (9 variables), de la serie horaria de precio; la "
    "de demanda (4), de la demanda horaria; la de hidrología (12), seis de los embalses y seis de los aportes; el "
    "ONI, de la serie de NOAA; y el indicador de pandemia y las 13 variables de calendario y festivos, de la marca "
    "de tiempo y del calendario oficial de festivos de Colombia. La generación total se usó en la caracterización, "
    "pero no como predictora: su valor en la hora objetivo se fija en el mismo despacho que el precio y su valor "
    "rezagado repite la información de la demanda, con la que tiene una correlación de 0,996.")
c.titulo_tabla("variables", "Las 40 variables predictoras por familia (y = precio de bolsa; t = hora objetivo).")
c.tabla([
    ["Familia (n.º)", "Definición", "Justificación (OE1)"],
    ["Precio rezagado (2)", "y~t−24~, y~t−168~", "Picos de 24 h y 168 h del periodograma"],
    ["Medias móviles (3)", "Media de y~t−24~ en 24 h, 7 días y 30 días", "Nivel y tendencia recientes"],
    ["Volatilidad (4)", "Desviación estándar en 24 h y 7 días, rango de 24 h y razón entre desviaciones",
     "Heterocedasticidad del precio"],
    ["Calendario (6)", "sen/cos(2πh/24), sen/cos(2πd/7), sen/cos(2πd~a~/365,25)",
     "Armónicos de 24 h, 168 h y anual; continuidad 23 h → 0 h"],
    ["Hidrología (12)", "Embalses y aportes rezagados 24 h, cambios a 1 y 7 días, medias de 7 y 30 días y razón "
     "frente a la media de 30 días", "Correlación negativa con el precio"],
    ["Demanda (4)", "Demanda rezagada 24, 48 y 72 h y su media de 24 h", "Correlación +0,325"],
    ["Clima y eventos (2)", "ONI con rampa causal; indicador de pandemia", "Correlación del ONI +0,387"],
    ["Festivos (7)", "Festivo; festivo hace 1, 2, 3 y 7 días; desajuste de tipo de día frente a t−24 h y t−168 h",
     "Error en festivos 74,1 → 62,6 (p = 0,0064)"],
], [3.2, 7.4, 5.2], sz=8.5, negrita_col0=True)
c.dato("Responsables:", JUAN + ".")
c.dato("Herramientas:", "Python 3, pandas, numpy y holidays; notebook 05 (características compartidas).")
c.cierre_actividad("7 días", 15, 100)

actividad("Modelos base y estadísticos de la literatura")
c.p("La referencia de todo el proyecto es la **persistencia**, ŷ~t~ = y~t−24~ {c:lago2021}. Sobre ella se "
    "implementaron modelos estadísticos respaldados por la literatura. El **ARX+GARCH**, adaptación del ARIMA-IGARCH "
    "propuesto para Colombia {c:munoz2017}, modela la media del logaritmo del precio con 15 regresoras "
    "(hidrología, ONI, pandemia, calendario y festivos) y la varianza condicional con un GARCH(1,1) "
    "{c:bollerslev1986} (Ecuación " f"{E['arx']}):")
c.ecuacion("arx", [[r"\ln(y_t)=\beta_0+\boldsymbol{\beta}^{\mathsf{T}}\mathbf{x}_t+\phi\ln(y_{t-24})+\varepsilon_t"],
                   [r"\varepsilon_t=\sigma_t z_t", r"\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2"]])
c.p("Se validaron 27 configuraciones contra 2025; se mantuvo el GARCH(1,1) porque el ganador de la validación, un "
    "GJR(2,1), tenía persistencia α + β = 1,02, es decir, varianza explosiva. El **GARCH-ged**, variante del mejor "
    "modelo de Nueva Zelanda, otro mercado hidrodominado {c:kapoor2023}, combina una regresión LASSO "
    "{c:tibshirani1996} por hora con varianza GARCH de errores con distribución generalizada {c:nelson1991}. "
    "También se probaron el **LEAR** del benchmark de Lago et al. {c:lago2021}, con 24 regresiones LASSO (una por "
    "hora), y **Prophet** {c:taylor2018}, que fue el más débil de todas las pruebas (rMAE ≈ 1,70).")
c.dato("Responsables:", JUAN + ".")
c.dato("Herramientas:", "Python 3, statsmodels, arch (GARCH), scikit-learn (LASSO), Prophet con CmdStan; notebooks "
       "04 y 08.")
c.cierre_actividad("14 días", 15, 100)

actividad("Modelos de aprendizaje automático y profundo")
c.p("Se entrenaron modelos de familias distintas, para que sus errores no estén correlacionados. **XGBoost** "
    "{c:chen2016} (500 árboles, profundidad 3, tasa 0,01, submuestreo 0,8, objetivo log y) y **LightGBM** "
    "exploran relaciones no lineales entre las 40 variables. **N-BEATSx** {c:olivares2023} y **N-HiTS** "
    "{c:challu2023} son redes profundas de expansión de bases con variables exógenas, con contexto de 168 h "
    "(elegido entre 1, 2 y 3 semanas contra 2025) y horizonte de 24 h. Otros modelos se evaluaron y "
    "**descartaron** con rMAE frente a la persistencia mayor o igual que 1: CatBoost (1,016), hurdle de dos etapas "
    "(1,028), Markov-Switching de 2 regímenes (0,990), mínimos cuadrados recursivos (1,052), TFT (1,020), LSTM "
    "(1,173) y LightGBM (≈ 1,10). En total se evaluaron 15 familias; la lección de diseño es que importa la "
    "diversidad de familias y no la cantidad de modelos: agregar ocho votantes más empeoró el ensamble de MAE 46,02 "
    "a 46,45 (p = 0,013).")
c.dato("Responsables:", f"{RAFA} (primera versión de XGBoost y LightGBM, notebooks 05 y 11) y {JUAN} (XGBoost "
       "final, redes profundas y modelos descartados, notebooks 06 y 09).")
c.dato("Herramientas:", "Python 3, xgboost, lightgbm, catboost, neuralforecast (PyTorch), Optuna; portátil AMD "
       "Ryzen 7 7730U, 15,4 GB de RAM, sin GPU.")
c.cierre_actividad("14 días", 20, 100)

actividad("Validación walk-forward en seis regímenes climáticos")
c.p("Un único corte de prueba puede esconder que un modelo solo funciona en un tipo de año. Por eso se aplicó una "
    "validación de origen móvil {c:tashman2000} con **seis orígenes** (el Anexo 1 planeaba cinco): O1 a O5 son "
    "años completos con regímenes ENSO distintos (La Niña, transición, El Niño fuerte y neutro) y O6 es 2026. En "
    "cada origen el modelo se reentrena solo con datos anteriores y se compara con la persistencia mediante "
    "Diebold-Mariano de dos colas (Tabla 4, Figura 5).")
c.titulo_tabla("walkforward", "MAE (COP/kWh) por origen walk-forward y resultado de Diebold-Mariano frente a la "
               "persistencia (gana / pierde, α = 0,05).")
c.tabla([
    ["Modelo", "O1\n2020-21", "O2\n2021-22", "O3\n2022-23", "O4\n2023-24", "O5\n2024-25", "O6\n2026", "Gana / pierde"],
    ["Persistencia", "16,80", "17,52", "39,75", "67,19", "101,25", "56,40", "referencia"],
    ["XGBoost", "18,25", "24,14", "59,60", "133,99", "135,47", "61,39", "0 / 6"],
    ["ARX+GARCH", "50,88", "19,51", "49,74", "80,80", "102,64", "55,83", "0 / 4"],
    ["N-BEATSx", "25,79", "17,17", "40,22", "66,36", "104,59", "45,15", "1 / 1"],
    ["N-HiTS", "55,38", "17,86", "37,97", "68,47", "103,05", "46,42", "1 / 1"],
], [2.8, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 2.2], sz=8.5,
    alinear=["left"] + ["center"] * 7)
c.figura("walkforward", "f_walkforward.png", "MAE por origen temporal y régimen ENSO de los modelos individuales.",
         ancho_cm=13.5)
c.p("La lectura es exigente: en los orígenes históricos ningún modelo individual supera de forma consistente a la "
    "persistencia, y XGBoost pierde en los seis. Esto motivó el ensamble (Actividad 2.5). Esta actividad tiene un "
    "**avance de 50 %**: la evaluación de los modelos individuales está completa, pero la del ensamble de seis "
    "votantes, que es el diseño adoptado, solo existe para 2026; extenderla a O1-O5 es la tarea pendiente de OE2 "
    "(Sección 5).")
c.dato("Responsables:", JUAN + ".")
c.dato("Herramientas:", "Python 3, pandas, xgboost, arch, neuralforecast; notebooks 07 y 10.")
c.cierre_actividad("24 días", 20, 50)

actividad("Ensamble de seis familias, extensión a 72 h y bandas de incertidumbre")
c.p("Los seis votantes (persistencia, XGBoost, ARX+GARCH, N-BEATSx, N-HiTS y GARCH-ged) se combinan por "
    "regresión cuantílica (QRA) {c:nowotarski2015}. Para cada franja de 6 horas g se resuelve el programa lineal "
    f"de la Ecuación {E['qra']}, con pesos no negativos y un objetivo alineado con el sMAPE, que es la métrica "
    "reportada (los pesos ω~t~ se actualizan iterativamente):")
c.ecuacion("qra", [r"\mathrm{min}_{w\ge 0,\,b}\sum_{t\in g}{\omega_t\left|y_t-b-\sum_k{w_k\,\hat{y}_{k,t}}\right|}",
                   r"\omega_t=\frac{2}{|y_t|+|\hat{y}_t|}"])
c.p("Ambas decisiones se validaron: los pesos por franja reducen el MAE de 44,38 a 43,24 (p = 0,0006) y el "
    "objetivo sMAPE baja el MAPE de 10,66 % a 10,47 % en 10 de 10 particiones. El producto de 72 h reutiliza el "
    "ensamble en las primeras 23 horas y combina, en el resto, un LEAR de 72 modelos LASSO, dos N-BEATSx y una "
    "persistencia semanal (Tabla 5). La incertidumbre se expresa con la banda [q~10~, q~90~] de N-BEATSx, "
    "calibrada por **inferencia conforme adaptativa** {c:romano2019}, {c:gibbs2021}: con la puntuación de no "
    "conformidad E~i~, cada día d se ensancha la banda con el cuantil empírico Q̂~d~ de los 30 días anteriores "
    f"(Ecuación {E['conforme']}, con 1 − α = 0,8):")
c.ecuacion("conforme", [r"E_i=\mathrm{max}\left(\hat{q}_{10,i}-y_i,\;y_i-\hat{q}_{90,i}\right)",
                        r"C_t=\left[\hat{q}_{10,t}-\hat{Q}_d,\;\hat{q}_{90,t}+\hat{Q}_d\right]"])
c.p("Aquí Q̂~d~ es el cuantil de nivel ⌈(n + 1)(1 − α)⌉/n de las n puntuaciones E~i~ de los 30 días anteriores al "
    "día d, de modo que el margen crece cuando el modelo empieza a fallar más. La banda cruda cubría solo el 62,6 % de las horas y la calibración estática entre 67 % y 72 %, porque el "
    "precio de 2026 cambió de régimen a mitad del año; la versión adaptativa cubre el 78,0 % a 24 h (objetivo "
    "80 %), con un ancho medio de 167,6 COP/kWh. La Figura 6 muestra el pronóstico y su banda en dos semanas de "
    "julio de 2026.")
c.titulo_tabla("72h", "Pronóstico de 72 h por tramo frente a la persistencia medida desde el mismo corte (prueba "
               "2026) y cobertura de la banda.")
c.tabla([
    ["Tramo", "MAE ensamble", "MAPE ensamble", "MAE persistencia", "MAPE persistencia", "rMAE", "DM (t)",
     "Cobertura [q~10~, q~90~]"],
    ["1-24 h", "40,83", "10,57 %", "55,67", "15,82 %", "0,733", "6,77", "78,9 %"],
    ["25-48 h", "62,56", "18,29 %", "79,84", "22,78 %", "0,784", "5,80", "77,0 %"],
    ["49-72 h", "72,87", "21,69 %", "94,10", "26,98 %", "0,774", "6,19", "78,5 %"],
    ["**Global 1-72 h**", "**58,75**", "**16,85 %**", "**76,54**", "**21,86 %**", "**0,768**", "**10,51**", "**78,1 %**"],
], [2.4, 1.9, 1.9, 2.1, 2.1, 1.5, 1.5, 2.2], sz=8.5, alinear=["left"] + ["center"] * 7)
c.figura("pronostico", "f_real_vs_pronostico.png", "Pronóstico de 24 h del ensamble frente al precio real, con la "
         "banda [q~10~, q~90~] calibrada, 6 al 19 de julio de 2026.", ancho_cm=14.0)
c.dato("Responsables:", JUAN + ".")
c.dato("Herramientas:", "Python 3, scipy.optimize (programación lineal), numpy y pandas; scripts de combinación y "
       "calibración del repositorio.")
c.cierre_actividad("7 días", 15, 100)

actividad("Comparación formal: precisión, significancia, costo e interpretabilidad")
c.p("Todos los modelos se compararon sobre las mismas 5.184 horas de 2026 con las métricas de la Ecuación "
    f"{E['metricas']}, donde el rMAE usa la persistencia como referencia y el MASE escala por el error ingenuo de "
    "entrenamiento {c:hyndman2006}:")
c.ecuacion("metricas", [[r"\mathrm{MAE}=\frac{1}{N}\sum_t{|y_t-\hat{y}_t|}",
                         r"\mathrm{MAPE}=\frac{100\,\%}{N}\sum_t{\frac{|y_t-\hat{y}_t|}{y_t}}"],
                        [r"\mathrm{rMAE}=\frac{\mathrm{MAE}_{\mathrm{modelo}}}{\mathrm{MAE}_{\mathrm{pers}}}"]])
c.p("La significancia se evalúa con la prueba de Diebold-Mariano {c:diebold1995} sobre la diferencia de errores "
    "absolutos, con varianza de Newey-West {c:newey1987} para corregir la autocorrelación de los errores horarios "
    f"(Ecuación {E['dm']}). Para el ensamble la hipótesis es de una cola, H~0~: E[d~t~] = 0 frente a H~1~: "
    "E[d~t~] > 0, porque la especificación exige superar a la referencia:")
c.ecuacion("dm", [r"DM=\frac{\bar{d}}{\sqrt{\hat{V}_{\mathrm{HAC}}(\bar{d})}}", r"d_t=|e_t^{\mathrm{ref}}|-|e_t^{\mathrm{modelo}}|"])
c.titulo_tabla("desempeno", "Desempeño a 24 h fuera de muestra (1 de enero a 5 de agosto de 2026, n = 5.184 h).")
c.tabla([
    ["Modelo", "MAE\n(COP/kWh)", "RMSE\n(COP/kWh)", "MAPE (%)", "sMAPE (%)", "rMAE", "DM (t)"],
    ["Persistencia", "56,40", "112,34", "15,81", "14,77", "1,000", "—"],
    ["XGBoost", "61,39", "107,64", "15,97", "15,26", "1,089", "−2,99"],
    ["ARX+GARCH", "55,83", "110,08", "15,48", "14,56", "0,990", "1,41"],
    ["N-BEATSx", "45,15", "89,06", "11,78", "10,85", "0,800", "5,13"],
    ["N-HiTS", "46,42", "90,54", "12,48", "11,49", "0,823", "4,16"],
    ["GARCH-ged", "45,50", "94,44", "12,32", "11,18", "0,807", "5,53"],
    ["**Ensamble (6 votantes)**", "**40,97**", "**85,48**", "**10,45**", "**9,80**", "**0,727**", "**7,76**"],
], [3.8, 2.0, 2.0, 1.8, 1.8, 1.6, 1.6], sz=8.5, alinear=["left"] + ["center"] * 6)
c.p("El ensamble reduce el error absoluto medio en **27 %** frente a la persistencia (40,97 frente a 56,40 "
    "COP/kWh) y el MAPE de 15,81 % a 10,45 %, y se "
    "rechaza H~0~ con p unilateral < 0,001 a 24 h (t = 7,76), en la versión desplegable, cuyos pesos solo usan "
    "días anteriores (t = 7,02), y a 72 h (t = 10,51). Reentrenar el bloque de 24 h toma 7,2 minutos en el portátil "
    "sin GPU, dentro del límite de 15 minutos fijado como requerimiento. Para la interpretabilidad, SHAP "
    "{c:lundberg2017} y la importancia por permutación sobre XGBoost coinciden en las tres variables más "
    "importantes (precio rezagado 24 h, media de 24 h y precio rezagado 168 h; correlación de Spearman entre "
    "rankings de 0,811), las mismas que la caracterización de OE1 señalaba como dominantes.")
c.p("**Error diario e intervalos de confianza.** El MAE de 40,97 COP/kWh es un promedio de "
    f"{ED['n_dias']} días con comportamientos muy distintos, por lo que también se analizó el error de cada día. "
    f"El MAE diario del ensamble tiene media {dec(ED['mae_ensamble']['media'])} COP/kWh y desviación estándar "
    f"{dec(ED['mae_ensamble']['de'])}. Como los días de error alto vienen en rachas (autocorrelación de "
    f"{dec(ED['acf1_mae_diario_ensamble'])} entre días consecutivos), el intervalo de confianza del 95 % para la "
    f"media se calculó con bootstrap por bloques de {ED['bloque_dias']} días: "
    f"[{dec(ED['mae_ensamble']['ic95_bloques'][0], 1)}; {dec(ED['mae_ensamble']['ic95_bloques'][1], 1)}] "
    f"COP/kWh, frente a [{dec(ED['mae_persistencia']['ic95_bloques'][0], 1)}; "
    f"{dec(ED['mae_persistencia']['ic95_bloques'][1], 1)}] de la persistencia. En el 80 % de los días el MAE "
    f"del ensamble estuvo entre {dec(ED['mae_ensamble']['p10'], 1)} y {dec(ED['mae_ensamble']['p90'], 1)} "
    f"COP/kWh; fue menor que 50 COP/kWh en el {dec(ED['pct_dias_bajo_umbral']['50']['ensamble'], 1)} % de "
    f"los días ({dec(ED['pct_dias_bajo_umbral']['50']['persistencia'], 1)} % para la persistencia) y menor "
    f"que 100 en el {dec(ED['pct_dias_bajo_umbral']['100']['ensamble'], 1)} % "
    f"({dec(ED['pct_dias_bajo_umbral']['100']['persistencia'], 1)} %). En términos relativos, el MAPE diario "
    f"tiene media {dec(ED['mape_ensamble']['media'])} % con intervalo "
    f"[{dec(ED['mape_ensamble']['ic95_bloques'][0], 1)} %; {dec(ED['mape_ensamble']['ic95_bloques'][1], 1)} %], "
    f"y en el 90 % de los días queda por debajo de {dec(ED['mape_ensamble']['p90'], 1)} %. La diferencia "
    f"diaria de MAE entre la persistencia y el ensamble tiene media "
    f"{dec(ED['diferencia_per_menos_ens']['media'], 1)} COP/kWh, con un intervalo "
    f"[{dec(ED['diferencia_per_menos_ens']['ic95_bloques'][0], 1)}; "
    f"{dec(ED['diferencia_per_menos_ens']['ic95_bloques'][1], 1)}] que no incluye el cero, y el ensamble tuvo "
    f"menor error en el {dec(ED['pct_dias_ensamble_mejor'], 1)} % de los días, un resultado coherente con la "
    "prueba de Diebold-Mariano. Por último, el error medio con signo es de "
    f"+{dec(ED['sesgo_ensamble']['media'], 1)} COP/kWh "
    f"[{dec(ED['sesgo_ensamble']['ic95_bloques'][0], 1)}; {dec(ED['sesgo_ensamble']['ic95_bloques'][1], 1)}]: "
    "el ensamble subestima levemente el precio, en cerca del 1 % del precio medio de 2026.")
c.dato("Responsables:", JUAN + ".")
c.dato("Herramientas:", "Python 3, statsmodels (varianza HAC), shap, scipy.stats; notebook 10 y scripts de "
       "evaluación.")
c.cierre_actividad("14 días", 15, 100)

# ---------------------------------------------------------------- OE3
h2(3)
actividad("Diseño del motor de decisión por umbrales")
c.p("El motor traduce el pronóstico probabilístico de cada hora en una acción discreta según el rol del usuario. "
    f"Cada método fija un umbral bajo u~L~ y uno alto u~H~ y compara con ellos la mediana q̂~50~ (Ecuación "
    f"{E['motor']}); los métodos «banda» e «híbrido» emiten además «esperar» cuando el ancho de la banda supera su "
    "percentil 75 histórico w~75~, es decir, cuando la confianza es baja:")
c.ecuacion("motor", [[r"a_t=C\ \ \mathrm{si}\ \ \hat{q}_{50,t}<u_L"], [r"a_t=V\ \ \mathrm{si}\ \ \hat{q}_{50,t}>u_H"],
                     [r"a_t=E\ \ \mathrm{en\ otro\ caso}"]])
c.p("donde C es «comprar» para el comercializador o «retener» para el generador, V es «evitar compra» o «vender», "
    "y E es «esperar».", after=120)
c.p("Los umbrales por defecto son los percentiles 25 y 75, el rango intercuartílico convencional para separar "
    "valores típicos de extremos {c:tukey1977}. Los cuatro métodos difieren en la referencia: «fijo» usa los "
    "percentiles del precio de entrenamiento 2019-2025; «rodante», los de una ventana móvil causal de 30 días "
    "sobre q̂~50~, que se adapta al régimen vigente {c:gama2014}; «banda» es el fijo más el filtro de confianza; e "
    "«híbrido» combina el umbral rodante con ese filtro. El método activo se elige por backtesting con un criterio "
    f"de **estabilidad** (Ecuación {E['maximin']}): entre los métodos cuya frecuencia de acción f está entre 10 % y "
    "40 % en ambas mitades del periodo, gana el de mayor ventaja V en su mitad más débil (criterio maximin "
    "{c:wald1950}), no el de mayor promedio:")
c.ecuacion("maximin", [[r"m^{*}=\mathrm{arg\,max}_{m\in M_{\mathrm{adm}}}\ \mathrm{min}\left(V_m^{(1)},V_m^{(2)}\right)"],
                       [r"M_{\mathrm{adm}}=\left\{m:\ 10\,\%\le f_m^{(h)}\le 40\,\%,\ h=1,2\right\}"]])
c.p("La Figura 7 muestra el motor en operación durante una semana de julio de 2026 con el método que elige el "
    "criterio de estabilidad para el generador («híbrido»). Cuando la mediana del pronóstico supera el umbral alto, "
    "el motor recomienda vender, y cuando cae bajo el umbral bajo, retener; si en esas horas la banda es más ancha "
    "que su percentil 75, la confianza es baja y el motor espera. El 13 de julio, por ejemplo, recomendó vender en "
    "horas en que el precio real cayó, un error que proviene del pronóstico y que la banda no alcanzó a señalar.")
c.figura("motor", "f_motor_hibrido.png", "Motor de decisión en operación, método «híbrido» para el generador (13 al "
         "19 de julio de 2026): pronóstico mediano, banda [q~10~, q~90~], umbrales rodantes y acción recomendada en "
         "cada hora.", ancho_cm=13.0)
c.dato("Responsables:", RAFA + ".")
c.dato("Herramientas:", "Python 3, pandas y numpy; módulo src/motor_decision.py y notebook 12.")
c.cierre_actividad("7 días", 30, 100)

actividad("Contrato de datos y desacople entre el motor y el modelo de pronóstico")
c.p("El motor no importa ningún modelo: lee el pronóstico a través de un contrato de datos "
    "(fuentes_pronostico.json) con la fecha y hora, el valor real y los cuantiles q~10~, q~50~ y q~90~ por "
    "horizonte. Esta decisión de arquitectura, orientada a la mantenibilidad de ISO/IEC 25010 {c:iso25010}, permite "
    "cambiar el modelo sin tocar el motor; se verificó al cambiar la fuente de 24 h a las bandas de N-BEATSx de "
    "72 h sin modificar el código del motor.")
c.dato("Responsables:", RAFA + ".")
c.dato("Herramientas:", "Python 3, JSON; src/motor_decision.py; Git y GitHub.")
c.cierre_actividad("3 días", 10, 100)

actividad("Diseño de la interfaz del dashboard")
c.p("Se diseñaron primero bocetos editables de tres vistas (Operador, Analista y Sistema) y luego la interfaz en "
    "Streamlit. La vista **Operador** está pensada para un usuario sin formación en probabilidad: aplica el "
    "método elegido por el criterio de estabilidad y muestra, para la hora seleccionada, la acción recomendada, la "
    "posición del precio esperado entre los umbrales y un nivel de confianza según el ancho de la banda; la hora se "
    "elige con un selector o haciendo clic en la franja de 24 horas del día. La vista **Analista** permite fijar a "
    "mano el método y los percentiles, y recalcula el método sugerido para el rango de fechas elegido.")
c.dato("Responsables:", RAFA + ".")
c.dato("Herramientas:", "Streamlit, Python 3, matplotlib; bocetos en dashboard/mockups.")
c.cierre_actividad("28 días", 15, 100)

actividad("Integración del dashboard con datos y modelos reales")
c.p("El dashboard corre localmente con los pronósticos y bandas de 2026 (Figura 8), y ambas vistas se verificaron "
    "sin errores con pruebas automatizadas de Streamlit en las cuatro combinaciones de rol (generador y "
    "comercializador) y horizonte (24 y 72 h). La actividad está al **50 %**: el motor todavía consume las bandas "
    "de N-BEATSx y no el ensamble de seis votantes, y su nivel de confianza aún no considera que el error sube de 41 "
    "a 52 COP/kWh cuando el pronóstico se lanza a las 23:00 en lugar de a las 00:00.")
c.figura("dashboard", "f_dashboard_operador.png", "Prototipo del dashboard en funcionamiento, vista Operador (5 de "
         "agosto de 2026, 08:00 h, rol generador).", ancho_cm=9.0)
c.dato("Responsables:", RAFA + ".")
c.dato("Herramientas:", "Streamlit y su módulo de pruebas automatizadas (AppTest), Python 3; dashboard/app.py.")
c.cierre_actividad("14 días", 20, 50)

actividad("Biblioteca de imágenes de apoyo a la decisión")
c.p("El Anexo 1 exige una biblioteca de 4 a 5 tipos de imágenes con sus metadatos (unidad, periodo, confianza y "
    "versión del modelo). Hoy existen prototipos de tres tipos dentro del dashboard y del análisis (semáforo de "
    "acción por hora, banda de incertidumbre y comparación real frente a pronóstico, como la Figura 6). Faltan el "
    "mapa de calor día-hora, los metadatos normalizados y la validación funcional, por lo que el avance es de 20 % y "
    "el grueso se ejecuta en las semanas 13-14 (Sección 5).")
c.dato("Responsables:", f"{RAFA} y {JUAN}.")
c.dato("Herramientas:", "Python 3, matplotlib, Streamlit.")
c.cierre_actividad("35 días", 25, 20)

# ---------------------------------------------------------------- OE4
h2(4)
actividad("Backtesting del motor de decisión sobre 2026")
c.p("Se evaluó el motor sobre las 5.184 horas de 2026 con los pronósticos fuera de muestra. La **ventaja** de un "
    "método es la diferencia entre el precio medio del periodo y el de las horas en que recomienda actuar, con el "
    "signo del rol (para el generador, cuánto más caro vende; para el comercializador, cuánto más barato compra); "
    "el precio medio de 2026 fue 397,9 COP/kWh. Con el criterio de ventaja promedio, «banda» gana en los cuatro "
    "casos (345,6, 281,6, 299,7 y 279,5 COP/kWh). Sin embargo, al evaluar por separado las dos mitades del periodo "
    "(corte: 19 de abril), ese resultado no se sostiene: para el generador a 24 h, «banda» actúa en el 1,2 % de las "
    "horas de la primera mitad y en el 32,6 % de la segunda, y para el comercializador no actúa en ninguna hora de "
    "la segunda mitad (Tabla 8, Figura 9). La causa es la referencia fija: el precio de 2026 se alejó del rango "
    "2019-2025. Con el criterio de estabilidad el motor elige «híbrido» para el generador y «rodante» para el "
    "comercializador en ambos horizontes (Tabla 7), con una ventaja promedio menor (149,9 a 169,6 COP/kWh) pero "
    "sostenida en todo el año.")
c.titulo_tabla("motor_elegido", "Backtesting del motor sobre 2026: método elegido por el criterio de estabilidad.")
c.tabla([
    ["Horizonte", "Rol", "Método", "Ventaja 1.ª / 2.ª mitad (COP/kWh)", "Frecuencia de acción 1.ª / 2.ª mitad"],
    ["24 h", "Generador", "híbrido", "78,1 / 183,5", "16,6 % / 22,3 %"],
    ["24 h", "Comercializador", "rodante", "82,7 / 248,2", "15,1 % / 14,1 %"],
    ["72 h", "Generador", "híbrido", "73,3 / 146,0", "19,1 % / 27,0 %"],
    ["72 h", "Comercializador", "rodante", "71,9 / 219,6", "17,5 % / 14,6 %"],
], [2.0, 3.2, 2.2, 4.2, 4.4], sz=8.5, alinear=["center", "left", "left", "center", "center"])
c.titulo_tabla("motor_mitades", "Métodos del motor a 24 h: ventaja del año completo frente a cada mitad del "
               "periodo.")
c.tabla([
    ["Rol", "Método", "Ventaja año", "Frecuencia año", "Ventaja 1.ª / 2.ª mitad", "Frecuencia 1.ª / 2.ª mitad",
     "¿Estable?"],
    ["Generador", "fijo", "313,3", "37,6 %", "305,7 / 125,0", "2,6 % / 72,5 %", "no"],
    ["Generador", "rodante", "244,1", "29,4 %", "110,8 / 201,2", "18,6 % / 40,1 %", "no"],
    ["Generador", "banda", "345,6", "16,9 %", "185,7 / 162,5", "1,2 % / 32,6 %", "no"],
    ["Generador", "híbrido", "166,8", "19,5 %", "78,1 / 183,5", "16,6 % / 22,3 %", "**sí**"],
    ["Comercializador", "fijo", "281,4", "14,4 %", "85,9 / 426,2", "28,7 % / 0,1 %", "no"],
    ["Comercializador", "rodante", "169,6", "14,6 %", "82,7 / 248,2", "15,1 % / 14,1 %", "**sí**"],
    ["Comercializador", "banda", "281,6", "14,3 %", "85,9 / sin acción", "28,7 % / 0,0 %", "no"],
    ["Comercializador", "híbrido", "225,0", "10,8 %", "82,7 / 293,9", "15,1 % / 6,4 %", "no"],
], [2.8, 1.8, 1.8, 1.9, 2.8, 3.0, 1.7], sz=8.5, alinear=["left", "left"] + ["center"] * 5)
c.figura("mitades", "f_motor_mitades.png", "Horas con acción del generador a 24 h en cada mitad del periodo 2026, "
         "por método. Franja verde: rango válido de 10 a 40 %.", ancho_cm=12.5)
c.p("Se barrieron además seis pares de percentiles (10/90 a 35/65) sin encontrar un óptimo interior: los pares "
    "extremos dan más ventaja por hora pero actúan menos (433,2 COP/kWh en el 20,8 % de las horas con 10/90, frente "
    "a 225,8 en el 49,1 % con 35/65), y 25/75 queda en un punto intermedio. En el último mes evaluado (7 de julio a "
    "5 de agosto de 2026), para un cliente de ejemplo de 100 kW que opere a potencia constante en las horas de "
    "acción, el método elegido para el generador («híbrido») habría significado +1,06 millones de COP, frente a "
    "−0,65 millones con «banda» (Figura 10). La traducción a pesos es ilustrativa y no reemplaza una evaluación "
    "económica completa {c:maciejowska2025}.")
c.figura("ganancia", "f_ganancia_ultimo_mes.png", "Ganancia o pérdida total por método en el último mes "
         "(generador, 24 h, cliente de ejemplo de 100 kW).", ancho_cm=12.5)
c.p("Las demás actividades de OE4 no se han ejecutado y forman parte del plan de la Sección 5: backtesting sobre "
    "los orígenes históricos (peso 15 %), validación con 3 a 5 usuarios (40 %) y documentación de los hallazgos "
    "(10 %). Por eso el avance de OE4 es 35 %.")
c.dato("Responsables:", RAFA + ".")
c.dato("Herramientas:", "Python 3, pandas, matplotlib; notebook 12 y src/motor_decision.py.")
c.cierre_actividad("7 días", 35, 100)

# ================================================================================================
# 3. CRONOGRAMA
# ================================================================================================
FASES = [
    ("Fase 0 — Formulación", [
        ("Definir problema, objetivo y justificación", "07-27", "07-28", 100),
        ("Delimitación del proyecto", "07-27", "08-18", 100),
        ("Elaboración del cronograma", "07-27", "08-18", 100),
        ("Ruta crítica del proyecto", "07-27", "08-18", 100),
        ("Análisis de riesgo", "08-11", "08-18", 100)]),
    ("Fase 1 — Procesamiento y caracterización (OE1)", [
        ("Sincronización de fuentes en dataset maestro horario", "07-27", "08-02", 100),
        ("Tratamiento de datos faltantes y atípicos", "07-27", "08-09", 100),
        ("Variables indicadoras de pandemia y ONI", "08-03", "08-09", 100),
        ("Documentación del preprocesamiento", "08-04", "08-09", 100),
        ("Caracterización temporal", "08-10", "08-16", 100),
        ("Caracterización espectral (periodograma)", "08-10", "08-16", 100),
        ("Comparación de filtros (media móvil y Savitzky-Golay)", "08-10", "08-16", 100),
        ("Correlaciones precio-hidrología-demanda", "08-10", "08-16", 100)]),
    ("Fase 2 — Características y modelado (OE2)", [
        ("Extracción de características compartidas", "08-17", "08-23", 100),
        ("Documento técnico de características", "08-17", "08-23", 100),
        ("Modelos avanzados y de aprendizaje automático", "08-24", "09-06", 100),
        ("Modelos base de la literatura", "08-24", "09-06", 100),
        ("Validación walk-forward por regímenes", "09-07", "09-30", 50)]),
    ("Fase 3 — Comparación y motor de decisión (OE3)", [
        ("Comparación formal frente a persistencia", "09-07", "09-13", 100),
        ("Cuantificación de incertidumbre", "09-07", "09-13", 100),
        ("Diseño del motor de decisión por umbrales", "09-07", "09-13", 100),
        ("Backtesting (histórico y ene-ago 2026)", "09-14", "09-20", 70),
        ("Informe comparativo de modelos", "09-14", "09-20", 100)]),
    ("Fase 4 — Imágenes de decisión y validación (OE3-OE4)", [
        ("Diseño de la interfaz del dashboard", "08-17", "09-13", 100),
        ("Diseño visual de la biblioteca de imágenes", "08-17", "09-06", 40),
        ("Integración del dashboard con datos y modelos", "09-14", "09-27", 50),
        ("Integración de la biblioteca con datos reales", "09-14", "09-27", 0),
        ("Validación con usuarios (3-5 personas)", "09-29", "10-04", 0)]),
    ("Fase 5 — Cierre", [
        ("Ajustes finales y pruebas de reproducibilidad", "09-12", "09-18", 50),
        ("Memoria IEEE, manual técnico y de usuario", "10-12", "11-01", 20),
        ("Repositorio reproducible ordenado", "10-12", "10-18", 50),
        ("Demo final", "11-02", "11-08", 0)]),
]


def fecha(mmdd):
    m, d = map(int, mmdd.split("-"))
    return date(2026, m, d)


def semana(d):
    return (d - INICIO_SEMESTRE).days // 7 + 1


def planeado(ini, fin):
    tot = (fin - ini).days + 1
    return min(max((FECHA_CORTE - ini).days + 1, 0), tot) / tot * 100


resumen = []
for nombre, tareas in FASES:
    dur = [(fecha(b) - fecha(a)).days + 1 for _, a, b, _ in tareas]
    logro = sum(d * x for d, (_, _, _, x) in zip(dur, tareas)) / sum(dur)
    plan = sum(d * planeado(fecha(a), fecha(b)) for d, (_, a, b, _) in zip(dur, tareas)) / sum(dur)
    ini = min(fecha(a) for _, a, _, _ in tareas)
    fin = max(fecha(b) for _, _, b, _ in tareas)
    resumen.append((nombre, ini, fin, plan, logro, sum(dur)))
dur_tot = sum(r[5] for r in resumen)
plan_tot = sum(r[3] * r[5] for r in resumen) / dur_tot
logro_tot = sum(r[4] * r[5] for r in resumen) / dur_tot
MESES = ["", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def fmt(d):
    return f"{d.day} {MESES[d.month]}"


def num(x, d=0):
    return f"{x:.{d}f}".replace(".", ",")


c = Cursor(H1_CRONO)
S_HOY = semana(FECHA_CORTE)
c.p("La base es el cronograma del Anexo 1 {c:anexo1}: 36 tareas en seis fases, con la semana 1 iniciando el 27 de "
    f"julio de 2026. A la fecha de corte ({FECHA_TXT}, semana {S_HOY}) se cuantificó el logro de cada tarea y se "
    "comparó con lo que el cronograma preveía para esa fecha. El porcentaje **planeado** de una tarea es la fracción "
    "de su duración ya transcurrida; el de una fase, el promedio de sus tareas ponderado por duración, igual que el "
    "**logrado** (Tabla 9). La Tabla 10 detalla cada tarea sobre las 16 semanas.", before=120)
c.titulo_tabla("fases", f"Logro por etapa frente a lo planeado a la fecha de corte ({FECHA_TXT}).")
filas = [["Etapa", "Periodo planeado", "% planeado a la fecha", "% logrado", "Diferencia"]]
for nombre, ini, fin, plan, logro, _ in resumen:
    filas.append([nombre, f"{fmt(ini)} – {fmt(fin)}", f"{num(plan)} %", f"{num(logro)} %",
                  f"{'+' if logro - plan >= 0.5 else ''}{num(logro - plan)} p.p.".replace("-", "−")])
filas.append(["**Proyecto completo (ponderado por duración)**", f"{fmt(INICIO_SEMESTRE)} – 8 nov",
              f"**{num(plan_tot)} %**", f"**{num(logro_tot)} %**",
              f"**{'+' if logro_tot - plan_tot >= 0.5 else ''}{num(logro_tot - plan_tot)} p.p.**".replace("-", "−")])
c.tabla(filas, [6.2, 3.2, 2.4, 2.0, 2.2], sz=8.5, alinear=["left", "center", "center", "center", "center"])

# diagrama de Gantt como tabla
AZUL_F, AZUL_T, VERDE, HOY = "2E75B6", "BDD7EE", "A9D08E", "FFE699"
cab = ["Actividad / semana"] + [str(k) for k in range(1, 17)] + ["% logro"]
filas, tipos = [cab], [None]
for nombre, tareas in FASES:
    r = next(x for x in resumen if x[0] == nombre)
    filas.append([f"**{nombre}**"] + [""] * 16 + [f"**{num(r[4])} %**"])
    tipos.append(("fase", semana(r[1]), semana(r[2]), r[4]))
    for tn, a, b, x in tareas:
        filas.append([tn] + [""] * 16 + [f"{x} %"])
        tipos.append(("tarea", semana(fecha(a)), semana(fecha(b)), x))


def sombra_gantt(ri, ci):
    tipo = tipos[ri]
    if tipo is None or ci == 0 or ci == 17:
        return None
    k, s0, s1, x = tipo
    if not s0 <= ci <= s1:
        return HOY if ci == S_HOY else None
    if k == "fase":
        return AZUL_F
    return VERDE if x == 100 else AZUL_T


c.titulo_tabla("gantt", "Cronograma del Anexo 1 con el logro de cada tarea. Barras verdes: tareas terminadas; "
               "azul claro: en curso o pendientes; azul oscuro: duración de la fase; columna amarilla: semana "
               f"actual ({S_HOY}). La semana 1 inicia el 27 de julio de 2026.")
c.tabla(filas, [5.4] + [0.52] * 16 + [1.25], sz=7, sombras=sombra_gantt,
        alinear=["left"] + ["center"] * 17)
tabla_gantt = c.el.getprevious()
for e in tabla_gantt.findall(qn("w:tr"))[0].findall(qn("w:tc"))[S_HOY].iter(qn("w:shd")):
    e.set(qn("w:fill"), HOY)
for tc in tabla_gantt.iter(qn("w:tc")):
    for lado in tc.iter(qn("w:left")):
        lado.set(qn("w:w"), "20")
    for lado in tc.iter(qn("w:right")):
        lado.set(qn("w:w"), "20")

c.p(f"El proyecto va **{num(abs(logro_tot - plan_tot))} puntos porcentuales "
    f"{'por encima' if logro_tot >= plan_tot else 'por debajo'}** de lo planeado a la fecha. Las fases 0, 1 y 3 "
    "están prácticamente completas y la Fase 2 superó lo previsto en alcance (seis orígenes en lugar de cinco, 15 "
    "familias de modelos y un ensamble que no estaba en el plan), aunque la validación walk-forward del ensamble "
    "sigue pendiente. El atraso se concentra en la Fase 4: la biblioteca de imágenes y la integración del motor con "
    "el ensamble se postergaron para priorizar la corrección de la demanda de agosto de 2026, que obligó a "
    "reentrenar la cadena completa, y la validación con usuarios (prevista del 29 de septiembre al 4 de octubre) se "
    "reprogramó para las semanas 11 y 12. La Fase 5 va adelantada en limpieza del repositorio y en pruebas de "
    "reproducibilidad de los notebooks 01 a 03. Este porcentaje mide el cronograma ponderando cada tarea por su "
    "duración, por eso difiere del 74 % de la Tabla 1, que pondera por objetivos y actividades.")

# ================================================================================================
# 4. DIFICULTADES
# ================================================================================================
c = Cursor(H1_DIFIC)
c.p("Ninguna dificultad ha impedido el avance del proyecto, pero varias obligaron a cambiar el enfoque. Se "
    "resumen con la solución adoptada:", before=120)
for tit, txt in [
    ("Incompatibilidad de la librería de descarga.", "La librería pydataxm no funcionaba con la versión de pandas "
     "del entorno. Se escribió una función propia contra la API REST de XM, que además permite descargar por lotes "
     "y reintentar."),
    ("Series de resolución distinta y riesgo de fuga de información.", "Unir datos horarios, diarios y mensuales "
     "sin usar información futura exigió reglas explícitas: rezago mínimo de 24 h, rampa causal para el ONI y "
     "exclusión de las variables que se fijan en el mismo despacho que el precio. Una auditoría sistemática del 18 "
     "de septiembre encontró una fuga residual en el ONI, que se cuantificó y resultó sin efecto en los resultados."),
    ("Los modelos individuales no superan a la persistencia en los años históricos.", "El walk-forward mostró que "
     "XGBoost pierde en los seis orígenes y que ningún modelo gana de forma consistente. La respuesta fue un "
     "ensamble de familias diversas con pesos por franja horaria, que sí supera a la persistencia con "
     "significancia, y dejar explícito que falta evaluarlo en los orígenes históricos."),
    ("Protocolo del corte de pronóstico.", "Se detectó que el corte de las 00:00 cae dentro del día de despacho "
     "que se pronostica, cuando ya se conoce la primera hora. Con el lanzamiento a las 23:00 el MAE del ensamble "
     "sube de 41 a 52 COP/kWh, aún mejor que la persistencia (56,40). Ambas cifras se reportan y el motor deberá "
     "ajustar su confianza según la hora de lanzamiento."),
    ("Error en los datos de la fuente.", "La demanda del 4-5 de agosto de 2026 llegó de la fuente entre 70 % y 86 % "
     "por debajo de su valor real. Se detectó al reejecutar los notebooks en una copia aislada; se corrigió, se "
     "reentrenó todo y el MAPE del ensamble pasó de 10,76 % a 10,49 % en 10 de 10 particiones."),
    ("El mejor método del motor en promedio no era estable.", "«banda» ganaba la ventaja promedio, pero casi no "
     "actuaba en la primera mitad de 2026. Se cambió el criterio de selección a uno de estabilidad (maximin entre "
     "mitades), que elige «híbrido» y «rodante»."),
    ("Límite de información en los cambios bruscos de precio dentro del día.", "Cuatro métodos de familias "
     "distintas para mejorar esas rampas (calibración de amplitud, análisis funcional, compensación de sesgo y "
     "selección de ventana por puntos de cambio; ver 6.2) no las mejoraron, e incluso conociendo la demanda real "
     "el error solo baja entre 3 % y 5 %. Lo que las mueve son las ofertas por planta, que el Anexo 1 deja fuera del "
     "alcance; se documentó como límite del proyecto."),
    ("Atraso en OE3 y OE4.", "La corrección de datos y los informes consumieron tiempo de la Fase 4. Se "
     "reprogramaron la validación con usuarios y la biblioteca de imágenes (Sección 5) y se priorizó su cierre "
     "sobre nuevas pruebas de modelos."),
]:
    c.p(f"**{tit}** {txt}", ind_left=284, hanging=284, after=100)

# ================================================================================================
# 5. PLAN SEMANAS 11 A 16
# ================================================================================================
c = Cursor(H1_PLAN)
c.p("Las actividades pendientes se ordenaron según su efecto sobre los objetivos y la ruta crítica: primero lo que "
    "falta para cerrar OE3 y OE4, que son los objetivos con menor avance, y después el cierre del proyecto.",
    before=120)
PLAN = [
    ("SEMANA 11 - 12 (5 al 18 de octubre):", [
        ("Validación del dashboard con 3 a 5 usuarios (OE4)",
         "Sesiones con usuarios del mercado o afines en los roles de generador y comercializador. Cada usuario "
         "resuelve tareas en la vista Operador y responde un cuestionario de comprensión sobre la acción "
         "recomendada, el nivel de confianza y la lectura de la banda de incertidumbre. Se registran el porcentaje "
         "de respuestas correctas y los errores de interpretación.",
         f"{JUAN} y {RAFA}.", "Streamlit, cuestionario en línea, guion de pruebas.", "10 días"),
        ("Conexión del motor al ensamble de seis votantes (OE3)",
         "Regenerar el contrato de datos con el ensamble de 24 h y el producto de 72 h, y ajustar el nivel de "
         "confianza del motor según la hora de lanzamiento del pronóstico.",
         f"{RAFA} (motor) y {JUAN} (contratos del ensamble).", "Python 3, src/motor_decision.py, JSON.", "7 días"),
        ("Evaluación del ensamble completo en los orígenes históricos 1 a 5 (OE2)",
         "Reentrenar los seis votantes y el combinador en cada origen del walk-forward y aplicar Diebold-Mariano "
         "frente a la persistencia, para cerrar la validación por regímenes.",
         JUAN + ".", "Python 3, neuralforecast, xgboost, arch, scipy.", "10 días"),
    ]),
    ("SEMANA 13 - 14 (19 de octubre al 1 de noviembre):", [
        ("Biblioteca de 4 a 5 tipos de imágenes de decisión (OE3)",
         "Completar el mapa de calor día-hora y normalizar los metadatos (unidad, periodo, confianza y versión del "
         "modelo) de las cinco imágenes; integrarlas al dashboard y verificarlas funcionalmente.",
         f"{RAFA} y {JUAN}.", "Python 3, matplotlib, Streamlit.", "10 días"),
        ("Backtesting histórico del motor y umbrales relativos (OE4 y OE3)",
         "Repetir el backtesting en los orígenes 1 a 5 y usar umbrales relativos móviles de 90 días para el "
         "régimen hidrológico, ya que en 2026 ninguna hora cae bajo el umbral absoluto de «embalse bajo».",
         RAFA + ".", "Python 3, pandas, notebook 12.", "7 días"),
        ("Documentación de los hallazgos de la validación (OE4)",
         "Analizar los resultados de las sesiones con usuarios y aplicar los ajustes de interfaz que resulten.",
         f"{JUAN} y {RAFA}.", "Streamlit, hoja de cálculo.", "4 días"),
        ("Memoria en formato IEEE, manual técnico y manual de usuario (Fase 5)",
         "Consolidar la documentación técnica acumulada en la bitácora y en este informe.",
         f"{JUAN} y {RAFA}.", "Word, Python (generación de tablas y figuras), GitHub.", "14 días"),
    ]),
    ("SEMANA 15 - 16 (2 al 15 de noviembre):", [
        ("Pruebas de reproducibilidad y repositorio ordenado (Fase 5)",
         "Reejecutar todos los notebooks en una copia aislada del proyecto y dejar el repositorio con instrucciones "
         "de instalación y ejecución.",
         JUAN + ".", "Python 3, Jupyter, Git y GitHub.", "5 días"),
        ("Manual de uso del motor de decisión (entregable de la semana 16)",
         "Documentar cómo interpretar cada acción, el nivel de confianza y las limitaciones del motor.",
         RAFA + ".", "Word, capturas del dashboard.", "5 días"),
        ("Demo, informe final, póster, presentación y video (semana 16)",
         "Preparar los entregables finales del Anexo 1 y la demostración del sistema completo.",
         f"{JUAN} y {RAFA}.", "Word, PowerPoint, Streamlit, editor de video.", "10 días"),
    ]),
]
for semana_txt, acts in PLAN:
    c.clon(P_SEMANA, semana_txt)
    for tit, desc, resp, herr, tiempo in acts:
        actividad(tit)
        c.p(desc, before=40)
        c.dato("Responsables:", resp)
        c.dato("Herramientas:", herr)
        c.cierre_actividad(tiempo)

# ================================================================================================
# 6. REFERENCIAS
# ================================================================================================
assert not [k for k in ENCONTRADAS if k in ORDEN]
c = Cursor(H1_REFS)
c.p("", after=0, sz=6)
for k, clave in enumerate(ORDEN, 1):
    c.p(f"[{k}]\t" + BIB[clave], jc="left", ind_left=567, hanging=567, after=80, sz=10, tabs=[("left", 567)])
c = Cursor(H1_ENC)
c.p("Literatura revisada durante el proyecto que no se cita en este informe; varias de estas ideas se probaron en "
    "experimentos que no se adoptaron (Sección 4).", before=120, after=100)
n0 = len(ORDEN)
for k, clave in enumerate(ENCONTRADAS, n0 + 1):
    c.p(f"[{k}]\t" + BIB[clave], jc="left", ind_left=567, hanging=567, after=80, sz=10, tabs=[("left", 567)])

for p_ in doc.paragraphs:
    if p_.style.name in ("Heading 1", "Heading 2") and p_.text.strip():
        p_.paragraph_format.keep_with_next = True
doc.save(str(SALIDA))
print("Guardado:", SALIDA)
print("Referencias citadas:", len(ORDEN), "| encontradas:", len(ENCONTRADAS))
print("Tablas:", len(T), "Figuras:", len(F), "Ecuaciones:", len(E))
print(f"Planeado a la fecha {num(plan_tot, 1)} % | logrado {num(logro_tot, 1)} %")
for r in resumen:
    print(f"  {r[0]}: plan {num(r[3], 1)} % | logro {num(r[4], 1)} %")
