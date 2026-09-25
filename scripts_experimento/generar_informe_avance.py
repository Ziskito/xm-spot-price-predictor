# -*- coding: utf-8 -*-
"""
Rellena docs/PLANTILLA_FINAL_ABET (1).docx con el informe de avance (sustentacion 2026-09-25).

No toca la plantilla: escribe un documento nuevo en docs/. Respeta estilos, numeracion de titulos,
margenes y tablas de la plantilla; solo retira los recuadros de "PREGUNTAS OBLIGATORIAS" y las
lineas de "Extension maxima sugerida" (igual que los informes de ejemplo del semestre anterior).
Todas las cifras salen de data/processed/resultados/informe_avance/ (informe_avance_resultados.py).
"""
import copy
import json
import re
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
PLANTILLA = RAIZ / "docs" / "PLANTILLA_FINAL_ABET (1).docx"
SALIDA = RAIZ / "docs" / "Informe_Avance_ABET_Barcelo_Dede.docx"
FIG = RAIZ / "scripts_experimento" / "informe_avance_assets"
C = json.loads((RAIZ / "data/processed/resultados/informe_avance/claves_informe.json").read_text(encoding="utf-8"))
COSTO = json.loads((RAIZ / "data/processed/resultados/informe_avance/costo_computacional.json").read_text(encoding="utf-8"))

# ------------------------------------------------------------------------------------------------
# bibliografia (IEEE). Se numera por orden de primera cita en el texto.
# ------------------------------------------------------------------------------------------------
BIB = {
    "anexo1": 'J. D. Barceló Barraza y R. A. Dede Perdomo, "Anexo 1 — Formulación del proyecto: Diseño de una plataforma de procesamiento y analítica de datos para el pronóstico del precio de la energía y el apoyo en la toma de decisiones," Depto. de Ingeniería Eléctrica y Electrónica, Universidad del Norte, Barranquilla, Colombia, 2026.',
    "iea": 'International Energy Agency, "Colombia 2023 — Energy Policy Review," IEA, París, Francia, 2023.',
    "xm": 'XM S.A. E.S.P., "API de datos del Sistema de Información del Mercado de Energía Mayorista," servapibi.xm.com.co. Consultado: ago. 2026.',
    "noaa": 'NOAA Climate Prediction Center, "Cold & Warm Episodes by Season — Oceanic Niño Index (ONI)," cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni. Consultado: 6-ago-2026.',
    "weron2014": 'R. Weron, "Electricity price forecasting: A review of the state-of-the-art with a look into the future," *Int. J. Forecast.*, vol. 30, no. 4, pp. 1030–1081, 2014.',
    "huisman2003": 'R. Huisman y R. Mahieu, "Regime jumps in electricity prices," *Energy Econ.*, vol. 25, no. 5, pp. 425–434, 2003, doi: 10.1016/S0140-9883(03)00041-0.',
    "agudelo2015": 'A. P. Agudelo, J. M. López-Lezama y E. Velilla, "Predicción del precio de la electricidad en la bolsa mediante un modelo neuronal no-lineal autorregresivo con entradas exógenas," *Inf. Tecnol.*, vol. 26, no. 6, pp. 99–108, 2015, doi: 10.4067/S0718-07642015000600012.',
    "munoz2017": 'A. Muñoz-Santiago, J. Urquijo-Vanstrahlengs, A. Castro-Otero y J. Lombana, "Pronóstico del precio de la energía en Colombia utilizando modelos ARIMA con IGARCH," *Rev. Econ. Rosario*, vol. 20, no. 1, pp. 127–161, 2017, doi: 10.12804/revistas.urosario.edu.co/economia/a.6152.',
    "barrientos2018": 'J. Barrientos Marín, E. Tabares Orozco y E. Velilla, "Forecasting electricity price in Colombia: A comparison between neural network, ARMA process and hybrid models," *Int. J. Energy Econ. Policy*, vol. 8, no. 3, pp. 97–106, 2018.',
    "gallon2021": 'S. Gallón y J. Barrientos, "Forecasting the Colombian electricity spot price under a functional approach," *Int. J. Energy Econ. Policy*, vol. 11, no. 2, pp. 67–74, 2021, doi: 10.32479/ijeep.10607.',
    "lago2021": 'J. Lago, G. Marcjasz, B. De Schutter y R. Weron, "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark," *Appl. Energy*, vol. 293, art. 116983, 2021, doi: 10.1016/j.apenergy.2021.116983.',
    "kapoor2023": 'G. Kapoor y N. Wichitaksorn, "Electricity price forecasting in New Zealand: A comparative analysis of statistical and machine learning models with feature selection," *Appl. Energy*, vol. 347, art. 121446, 2023, doi: 10.1016/j.apenergy.2023.121446.',
    "ziel2018": 'F. Ziel y R. Weron, "Day-ahead electricity price forecasting with high-dimensional structures: Univariate vs. multivariate modeling frameworks," *Energy Econ.*, vol. 70, pp. 396–420, 2018, doi: 10.1016/j.eneco.2017.12.016.',
    "olivares2023": 'K. G. Olivares, C. Challu, G. Marcjasz, R. Weron y A. Dubrawski, "Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx," *Int. J. Forecast.*, vol. 39, no. 2, pp. 884–900, 2023, doi: 10.1016/j.ijforecast.2022.03.001.',
    "challu2023": 'C. Challu, K. G. Olivares, B. N. Oreshkin, F. Garza Ramírez, M. Mergenthaler Canseco y A. Dubrawski, "NHITS: Neural hierarchical interpolation for time series forecasting," en *Proc. AAAI Conf. Artif. Intell.*, vol. 37, no. 6, 2023, pp. 6989–6997, doi: 10.1609/aaai.v37i6.25854.',
    "chen2016": 'T. Chen y C. Guestrin, "XGBoost: A scalable tree boosting system," en *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.*, San Francisco, CA, EE. UU., 2016, pp. 785–794.',
    "bollerslev1986": 'T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity," *J. Econometrics*, vol. 31, no. 3, pp. 307–327, 1986.',
    "nelson1991": 'D. B. Nelson, "Conditional heteroskedasticity in asset returns: A new approach," *Econometrica*, vol. 59, no. 2, pp. 347–370, 1991.',
    "tibshirani1996": 'R. Tibshirani, "Regression shrinkage and selection via the lasso," *J. R. Stat. Soc. Ser. B*, vol. 58, no. 1, pp. 267–288, 1996.',
    "nowotarski2015": 'J. Nowotarski y R. Weron, "Computing electricity spot price prediction intervals using quantile regression and forecast averaging," *Comput. Stat.*, vol. 30, no. 3, pp. 791–803, 2015, doi: 10.1007/s00180-014-0523-0.',
    "romano2019": 'Y. Romano, E. Patterson y E. Candès, "Conformalized quantile regression," en *Adv. Neural Inf. Process. Syst. 32 (NeurIPS 2019)*, Vancouver, Canadá, 2019.',
    "gibbs2021": 'I. Gibbs y E. Candès, "Adaptive conformal inference under distribution shift," en *Adv. Neural Inf. Process. Syst. 34 (NeurIPS 2021)*, 2021.',
    "diebold1995": 'F. X. Diebold y R. S. Mariano, "Comparing predictive accuracy," *J. Bus. Econ. Stat.*, vol. 13, no. 3, pp. 253–263, 1995.',
    "newey1987": 'W. K. Newey y K. D. West, "A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix," *Econometrica*, vol. 55, no. 3, pp. 703–708, 1987, doi: 10.2307/1913610.',
    "hyndman2006": 'R. J. Hyndman y A. B. Koehler, "Another look at measures of forecast accuracy," *Int. J. Forecast.*, vol. 22, no. 4, pp. 679–688, 2006, doi: 10.1016/j.ijforecast.2006.03.001.',
    "tashman2000": 'L. J. Tashman, "Out-of-sample tests of forecasting accuracy: An analysis and review," *Int. J. Forecast.*, vol. 16, no. 4, pp. 437–450, 2000, doi: 10.1016/S0169-2070(00)00065-0.',
    "savitzky1964": 'A. Savitzky y M. J. E. Golay, "Smoothing and differentiation of data by simplified least squares procedures," *Anal. Chem.*, vol. 36, no. 8, pp. 1627–1639, 1964, doi: 10.1021/ac60214a047.',
    "oppenheim2010": 'A. V. Oppenheim y R. W. Schafer, *Discrete-Time Signal Processing*, 3.ª ed. Upper Saddle River, NJ, EE. UU.: Pearson, 2010.',
    "lundberg2017": 'S. M. Lundberg y S.-I. Lee, "A unified approach to interpreting model predictions," en *Adv. Neural Inf. Process. Syst. 30 (NIPS 2017)*, Long Beach, CA, EE. UU., 2017.',
    "taylor2018": 'S. J. Taylor y B. Letham, "Forecasting at scale," *Am. Stat.*, vol. 72, no. 1, pp. 37–45, 2018.',
    "maciejowska2025": 'K. Maciejowska, A. Lipiecki y B. Uniejewski, "Statistical and economic evaluation of forecasts in electricity markets: Beyond RMSE and MAE," *Energy Convers. Manag.*, vol. 356, art. 121408, 2026, doi: 10.1016/j.enconman.2026.121408.',
    "iso8601": 'ISO 8601-1:2019, *Date and time — Representations for information interchange — Part 1: Basic rules*, International Organization for Standardization, Ginebra, Suiza, 2019.',
    "iso25010": 'ISO/IEC 25010:2011, *Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — System and software quality models*, ISO/IEC, Ginebra, Suiza, 2011.',
    "ley1581": 'Congreso de la República de Colombia, "Ley Estatutaria 1581 de 2012, por la cual se dictan disposiciones generales para la protección de datos personales," Bogotá, Colombia, oct. 2012.',
    "creg2022": 'Comisión de Regulación de Energía y Gas (CREG), "Resolución CREG 101 018 de 2022," Bogotá, Colombia, 2022.',
    "ley23": 'Congreso de la República de Colombia, "Ley 23 de 1982, sobre derechos de autor," Bogotá, Colombia, ene. 1982.',
    "aciem2015": 'ACIEM y Consejo Profesional Nacional de Ingenierías Eléctrica, Mecánica y Profesiones Afines, *Manual de Referencia de Tarifas en Ingeniería*. Cundinamarca, Colombia, 2015.',
    "pedregosa2011": 'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.',
}
ORDEN = []


def cita(claves):
    nums = []
    for k in claves.split(","):
        k = k.strip()
        assert k in BIB, k
        if k not in ORDEN:
            ORDEN.append(k)
        nums.append(ORDEN.index(k) + 1)
    return ", ".join(f"[{n}]" for n in nums)


def expandir(texto):
    return re.sub(r"\{c:([^}]+)\}", lambda m: cita(m.group(1)), texto)


def f(x, d=2):
    """numero con coma decimal y punto de miles (convencion colombiana)."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def pct(x, d=2):
    return f(x, d) + " %"


# ------------------------------------------------------------------------------------------------
# utilidades XML
# ------------------------------------------------------------------------------------------------
doc = docx.Document(str(PLANTILLA))
BODY = doc.element.body


def _run(texto, b=False, i=False, sz=None, color=None, sub=False, sup=False):
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    if b:
        rpr.append(OxmlElement("w:b"))
    if i:
        rpr.append(OxmlElement("w:i"))
    if color:
        c = OxmlElement("w:color"); c.set(qn("w:val"), color); rpr.append(c)
    if sz:
        s = OxmlElement("w:sz"); s.set(qn("w:val"), str(int(sz * 2))); rpr.append(s)
        s2 = OxmlElement("w:szCs"); s2.set(qn("w:val"), str(int(sz * 2))); rpr.append(s2)
    if sub or sup:
        v = OxmlElement("w:vertAlign"); v.set(qn("w:val"), "subscript" if sub else "superscript"); rpr.append(v)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = texto
    r.append(t)
    return r


def _runs(texto, sz=None, color=None, b_all=False, i_all=False, _exp=True):
    """markup: **negrita**, *cursiva*, ~sub~, ^sup^ (admite ~ y ^ dentro de ** y *)"""
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
                    out.append(_run(t2[1:-1], sub=True, b=b_all, i=i_all, sz=sz, color=color))
                elif t2.startswith("^") and len(t2) > 2:
                    out.append(_run(t2[1:-1], sup=True, b=b_all, i=i_all, sz=sz, color=color))
                else:
                    out.append(_run(t2, b=b_all, i=i_all, sz=sz, color=color))
    return out


def _par(texto="", jc="both", after=100, before=0, ind_left=None, hanging=None, sz=None, color=None,
         b_all=False, i_all=False, keep_next=False, style=None, line=None):
    p = OxmlElement("w:p")
    ppr = OxmlElement("w:pPr")
    if style:
        ps = OxmlElement("w:pStyle"); ps.set(qn("w:val"), style); ppr.append(ps)
    if keep_next:
        ppr.append(OxmlElement("w:keepNext"))
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:after"), str(after)); sp.set(qn("w:before"), str(before))
    if line:
        sp.set(qn("w:line"), str(line)); sp.set(qn("w:lineRule"), "auto")
    ppr.append(sp)
    if ind_left is not None:
        ind = OxmlElement("w:ind"); ind.set(qn("w:left"), str(ind_left))
        if hanging:
            ind.set(qn("w:hanging"), str(hanging))
        ppr.append(ind)
    j = OxmlElement("w:jc"); j.set(qn("w:val"), jc); ppr.append(j)
    p.append(ppr)
    for r in _runs(texto, sz=sz, color=color, b_all=b_all, i_all=i_all):
        p.append(r)
    return p


class Cursor:
    def __init__(self, elemento):
        self.el = elemento

    def add(self, el):
        self.el.addnext(el)
        self.el = el
        return el

    # ---- bloques de texto ----
    def p(self, texto, **kw):
        return self.add(_par(texto, **kw))

    def viñetas(self, items, after=60):
        for it in items:
            self.add(_par("•\t" + it, ind_left=426, hanging=284, after=after))

    def numerada(self, items, after=60):
        for k, it in enumerate(items, 1):
            self.add(_par(f"{k}.\t" + it, ind_left=426, hanging=284, after=after))

    def ecuacion(self, texto, num):
        p = _par("", jc="left", after=100, before=40)
        ppr = p.find(qn("w:pPr"))
        tabs = OxmlElement("w:tabs")
        for val, pos in (("center", 4680), ("right", 9350)):
            t = OxmlElement("w:tab"); t.set(qn("w:val"), val); t.set(qn("w:pos"), str(pos)); tabs.append(t)
        ppr.insert(0, tabs)
        r = OxmlElement("w:r"); r.append(OxmlElement("w:tab")); p.append(r)
        for rr in _runs(texto, i_all=True):
            p.append(rr)
        r = OxmlElement("w:r"); r.append(OxmlElement("w:tab")); p.append(r)
        p.append(_run(f"({num})"))
        return self.add(p)

    # ---- tablas y figuras ----
    def titulo_tabla(self, n, texto):
        return self.p(f"**Tabla {n}.** {texto}", jc="center", after=60, before=120, sz=10, keep_next=True)

    def figura(self, n, archivo, texto, ancho_cm=16.0):
        tmp = doc.add_paragraph()
        tmp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        tmp.paragraph_format.keep_with_next = True
        tmp.paragraph_format.space_before = Pt(6)
        tmp.add_run().add_picture(str(FIG / archivo), width=Cm(ancho_cm))
        self.add(tmp._p)
        return self.p(f"**Figura {n}.** {texto}", jc="center", after=160, sz=10)

    def tabla(self, filas, anchos_cm, encabezado=True, sz=9, alinear=None, negrita_col0=False, espacio=True):
        """tabla con el mismo estilo de las tablas de contenido de la plantilla:
        bordes simples, encabezado gris oscuro 2E2E2E con texto blanco en negrita."""
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
        look = OxmlElement("w:tblLook"); look.set(qn("w:val"), "04A0"); tblpr.append(look)
        total = sum(anchos_cm)
        tw_dxa = [int(9350 * a / total) for a in anchos_cm]
        grid = tbl.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), tw_dxa):
            gc.set(qn("w:w"), str(w))
        for ri, fila in enumerate(filas):
            row = t.rows[ri]
            if ri == 0 and encabezado:
                trpr = row._tr.get_or_add_trPr()
                trpr.append(OxmlElement("w:tblHeader"))
            else:
                trpr = row._tr.get_or_add_trPr()
                cs = OxmlElement("w:cantSplit"); trpr.append(cs)
            for ci, valor in enumerate(fila):
                cell = row.cells[ci]
                tcpr = cell._tc.get_or_add_tcPr()
                for e in list(tcpr):
                    tcpr.remove(e)
                w = OxmlElement("w:tcW"); w.set(qn("w:w"), str(tw_dxa[ci])); w.set(qn("w:type"), "dxa"); tcpr.append(w)
                enc = ri == 0 and encabezado
                if enc:
                    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto")
                    shd.set(qn("w:fill"), "2E2E2E"); tcpr.append(shd)
                mar = OxmlElement("w:tcMar")
                for lado, v in (("top", 50), ("left", 80), ("bottom", 50), ("right", 80)):
                    e = OxmlElement(f"w:{lado}"); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); mar.append(e)
                tcpr.append(mar)
                va = OxmlElement("w:vAlign"); va.set(qn("w:val"), "center"); tcpr.append(va)
                p = cell.paragraphs[0]._p
                for e in list(p):
                    if e.tag != qn("w:pPr"):
                        p.remove(e)
                lineas = str(valor).split("\n")
                for li, linea in enumerate(lineas):
                    if li > 0:
                        p = copy.deepcopy(p)
                        for e in list(p):
                            if e.tag != qn("w:pPr"):
                                p.remove(e)
                        cell._tc.append(p)
                    ppr = p.find(qn("w:pPr"))
                    if ppr is None:
                        ppr = OxmlElement("w:pPr"); p.insert(0, ppr)
                    for e in list(ppr):
                        ppr.remove(e)
                    sp = OxmlElement("w:spacing"); sp.set(qn("w:after"), "0"); sp.set(qn("w:before"), "0"); ppr.append(sp)
                    al = "center" if enc else (alinear[ci] if alinear else "left")
                    j = OxmlElement("w:jc"); j.set(qn("w:val"), al); ppr.append(j)
                    for r in _runs(linea, sz=sz, color="FFFFFF" if enc else None,
                                   b_all=enc or (negrita_col0 and ci == 0)):
                        p.append(r)
        if len(filas) <= 7:
            for tr in tbl.findall(qn("w:tr"))[:-1]:
                for pp in tr.iter(qn("w:p")):
                    ppr = pp.find(qn("w:pPr"))
                    ppr.insert(0, OxmlElement("w:keepNext"))
        self.add(tbl)
        return self.p("", after=60) if espacio else tbl


def buscar_parrafo(texto, estilo=None, desde=0):
    for i, p in enumerate(doc.paragraphs):
        if i < desde:
            continue
        if p.text.strip() == texto and (estilo is None or p.style.name == estilo):
            return p
    raise KeyError(texto)


def encabezado(texto, nivel):
    return buscar_parrafo(texto, f"Heading {nivel}")


# ------------------------------------------------------------------------------------------------
# 1) limpiar guias de la plantilla
# ------------------------------------------------------------------------------------------------
tablas = list(doc.tables)
t_normas, t_alt, t_crit = tablas[14], tablas[15], tablas[16]
for t in tablas:
    txt = t.rows[0].cells[0].text
    if "PREGUNTAS OBLIGATORIAS" not in txt and not txt.strip().startswith("EVIDENCIA ABET"):
        continue
    if t in (t_alt, t_crit):
        t._tbl.remove(t.rows[0]._tr)          # se conserva la tabla, se quita solo la fila de guia
    else:
        t._tbl.getparent().remove(t._tbl)
for p in list(doc.paragraphs):
    if p.text.strip().startswith("Extensión máxima sugerida"):
        p._p.getparent().remove(p._p)

# ------------------------------------------------------------------------------------------------
# 2) portada
# ------------------------------------------------------------------------------------------------
TITULO = ("Diseño de una plataforma de procesamiento y analítica de datos para el pronóstico del precio "
          "de la energía y el apoyo en la toma de decisiones")


def reemplazar(p, nuevo):
    runs = p.runs
    runs[0].text = nuevo
    for r in runs[1:]:
        r.text = ""


reemplazar(buscar_parrafo("[ Título del Proyecto — máximo 15 palabras ]"), TITULO)
reemplazar(buscar_parrafo("[Nombre completo estudiante 1] — [CC / Código]"),
           "Juan David Barceló Barraza — CC 1044606227 / Código 200181896")
reemplazar(buscar_parrafo("[Nombre completo estudiante 2] — [CC / Código]"),
           "Rafael Andrés Dede Perdomo — CC 1043436500 / Código 200180841")
reemplazar(buscar_parrafo("[Nombre del/los asesor(es)]"), "José Daniel Soto Ortiz · Daniela María Charris Stand")
reemplazar(buscar_parrafo("Programa de Ingeniería [Eléctrica / Electrónica]"), "Programa de Ingeniería Electrónica")
fecha = buscar_parrafo("[Mes] de [Año]")
reemplazar(fecha, "Septiembre de 2026")
# el titulo aprobado ocupa tres lineas: se retira una linea vacia de la portada para que el salto de
# pagina no caiga solo en una hoja nueva (evita una pagina en blanco despues de la portada)
p_tit = buscar_parrafo(TITULO)._p
sig = p_tit.getnext()
if sig is not None and sig.tag == qn("w:p") and not "".join(sig.itertext()).strip() and not sig.findall(".//" + qn("w:br")):
    sig.getparent().remove(sig)

# ------------------------------------------------------------------------------------------------
# cifras
# ------------------------------------------------------------------------------------------------
H = {r["modelo"]: r for r in C["oe2_holdout"]}
ENS, PER = H["Ensamble (6 votantes)"], H["Persistencia"]
T72 = {r["tramo"]: r for r in C["oe2_72h"]}
G72 = T72["Global 1-72 h"]
DES = C["oe2_desplegable"]
LIT = C["oe2_literatura_24h"]
BAN = C["oe2_bandas"]
WF = C["oe2_wf_mae"]
WFR = {r["modelo"]: r for r in C["oe2_wf_resumen"]}
BT = C["oe3_backtest"]
D1 = C["oe1_desc"]
COR = C["oe1_corr"]
AV = {"OE1": 100, "OE2": 90, "OE3": 70, "OE4": 35}
AV_G = round(sum(AV.values()) / 4)
t_nn = COSTO["N-BEATSx+N-HiTS_s"]


def mae15(m):
    return sum(WF[m][f"Origen {k}"] for k in range(1, 6)) / 5
t_total = sum(COSTO[k] for k in ["Persistencia_s", "XGBoost_s", "ARX+GARCH_s", "GARCH-ged_s", "Combinador_sMAPE_s"]) + t_nn

# ================================================================================================
# 1. INFORMACION PRINCIPAL
# ================================================================================================
cur = Cursor(encabezado("Título del proyecto", 2)._p)
cur.p(f"**{TITULO}.**")

cur = Cursor(encabezado("Resumen ejecutivo", 2)._p)
cur.p("**Resumen.** El precio de bolsa en Colombia se fija cada hora sobre una matriz cercana a 72 % "
      "hidroeléctrica {c:iea} y puede multiplicarse en semanas durante El Niño. Se diseñó una plataforma en "
      "Python que adquiere y sincroniza los datos públicos de XM y NOAA, los caracteriza en tiempo y frecuencia, "
      "pronostica a 24-72 h con un ensamble de seis modelos combinados por regresión cuantílica, calibra bandas de "
      "incertidumbre por inferencia conforme y alimenta un motor de decisión con dashboard por rol. Se validó fuera "
      "de muestra en enero-agosto de 2026 (5.184 horas), con walk-forward en seis regímenes climáticos y la prueba "
      f"de Diebold-Mariano. A 24 h el MAPE es {pct(ENS['MAPE'])} frente a {pct(PER['MAPE'])} de la persistencia (27 % "
      f"menos error, p < 0,001); a 72 h, {pct(G72['MAPE'])} frente a {pct(G72['MAPE_persistencia'])}. Avance global "
      f"estimado: {AV_G} %.")
cur.p("**Abstract.** Colombia's hourly electricity spot price rests on a roughly 72 % hydroelectric mix and can "
      "multiply within weeks during El Niño. We designed a Python platform that acquires and synchronizes public XM and NOAA data, characterizes them in time "
      "and frequency, forecasts 24-72 h ahead with an ensemble of six models combined by quantile regression, "
      "calibrates uncertainty bands with conformal inference, and feeds a role-based decision engine and dashboard. "
      "It was validated out of sample on January-August 2026 (5,184 hours) with walk-forward evaluation across six "
      f"climate regimes and the Diebold-Mariano test. The 24-hour MAPE is {ENS['MAPE']:.2f} % versus "
      f"{PER['MAPE']:.2f} % for persistence (27 % lower error, p < 0.001); at 72 hours, {G72['MAPE']:.2f} % versus "
      f"{G72['MAPE_persistencia']:.2f} %. Estimated overall progress: {AV_G} %.")

cur = Cursor(encabezado("Costo total del proyecto", 2)._p)
fr = 9 / 16
rubros = [("Personal", 32201260, fr), ("Materiales, equipos y herramientas", 3160000, fr), ("Tecnología", 520000, fr),
          ("Infraestructura", 928000, fr), ("Transporte y logística", 296000, fr), ("Comunicación y divulgación", 500000, 0.0)]
tot_p = sum(r[1] for r in rubros)
tot_e = sum(round(r[1] * r[2]) for r in rubros)
cur.p("Presupuesto del Anexo 1 {c:anexo1} (honorarios según ACIEM {c:aciem2015}) con la ejecución estimada a la "
      "semana 9 de 16 por prorrateo; la divulgación se ejecuta en la semana 16. Financiación: $24.486.344 propia y "
      "$13.118.916 de la Universidad del Norte. Detalle en el Anexo 6.")
cur.titulo_tabla(1, "Presupuesto aprobado y ejecución estimada a la semana 9 (COP).")
filas = [["Rubro", "Aprobado (Anexo 1)", "Ejecutado estimado", "% ejecutado"]]
for nom, v, k in rubros:
    filas.append([nom, "$" + f(v, 0), "$" + f(round(v * k), 0), pct(k * 100, 1)])
filas.append(["**Total**", "**$" + f(tot_p, 0) + "**", "**$" + f(tot_e, 0) + "**", "**" + pct(tot_e / tot_p * 100, 1) + "**"])
cur.tabla(filas, [6.2, 3.4, 3.4, 2.4], alinear=["left", "right", "right", "right"], sz=8.5, espacio=False)

# ================================================================================================
# 2. INTRODUCCION
# ================================================================================================
cur = Cursor(encabezado("Introducción", 1)._p)
cur.p("El mercado mayorista de energía colombiano, administrado por XM, fija un precio de bolsa para cada hora del "
      "día. Como cerca de 72 % de la generación es hidroeléctrica {c:iea}, ese precio responde a la hidrología "
      "(nivel de embalses y aportes de los ríos) y a su modulación por el fenómeno El Niño-Oscilación del Sur: "
      "entre 2023 y 2024 el promedio diario osciló entre 124 y 2.499 COP/kWh. Los agentes que compran o "
      "venden en bolsa deben decidir hora a hora bajo esa incertidumbre, y la literatura sobre pronóstico de "
      "precios eléctricos muestra que se trata de series con saltos, estacionalidad múltiple y cambios de régimen "
      "{c:weron2014}.")
cur.p("Con este proyecto se construyó, a la fecha, una plataforma reproducible que (i) adquiere y sincroniza a "
      "resolución horaria las series públicas del mercado entre enero de 2019 y agosto de 2026; (ii) las "
      "caracteriza en tiempo y frecuencia para seleccionar variables predictoras; (iii) compara seis familias de "
      "modelos y los combina en un ensamble que reduce el error en 27 % frente a la persistencia con significancia "
      "estadística; y (iv) traduce el pronóstico y su incertidumbre en señales de comprar, vender o esperar "
      "según el rol del usuario, visibles en un dashboard.")
cur.p("Metodológicamente, el trabajo sigue las cinco fases del Anexo 1 {c:anexo1}: cadena de procesamiento y "
      "caracterización (OE1), extracción de características y modelado (OE2), comparación formal y motor de "
      "decisión (OE3) e imágenes de decisión y validación (OE3 y OE4). Todas las decisiones se tomaron con "
      "evaluación estrictamente fuera de muestra y con la prueba de Diebold-Mariano {c:diebold1995}, siguiendo las "
      "buenas prácticas del área {c:lago2021}.")
cur.p("El documento se organiza así. La Sección 3 presenta los objetivos y su porcentaje de avance; la 4, los "
      "alcances, limitaciones y entregables; la 5, el planteamiento del problema y la comparación con el estado "
      "del arte; la 6, el diseño de ingeniería (requerimientos, normas, alternativas, matriz de decisión, riesgos "
      "y diseño definitivo); la 7, el plan experimental y sus resultados; la 8, los impactos; la 9, las "
      "conclusiones por objetivo; y las secciones 10 y 11, la bibliografía y los anexos.")

# ================================================================================================
# 3. OBJETIVOS
# ================================================================================================
cur = Cursor(encabezado("Objetivo general", 2)._p)
cur.p("Diseñar, implementar y validar una plataforma de procesamiento de datos que permita el análisis y "
      f"pronóstico del precio de la energía en el mercado eléctrico colombiano para la toma de decisiones "
      f"**(avance: {AV_G} %)**.")
cur.p("El objetivo es verificable en las Conclusiones con tres criterios medibles: que la plataforma procese el "
      "rango histórico completo sin huecos, que su pronóstico supere a la persistencia con significancia "
      "estadística (α = 0,05) y que el motor de decisión sea validado mediante backtesting y con usuarios. Los "
      "objetivos están transcritos del Anexo 1 {c:anexo1}, cuyo numeral 3 no puede modificarse durante el "
      "semestre; el título, del numeral 1, tampoco.")
cur = Cursor(encabezado("Objetivos específicos", 2)._p)
objetivos = [
    ("OE1", "Adquirir, sincronizar y caracterizar temporal y espectralmente las series históricas de las variables "
            "del mercado, mediante técnicas de procesamiento de datos, para seleccionar variables predictoras."),
    ("OE2", "Implementar y evaluar diferentes modelos de pronóstico del precio de bolsa, comparando precisión frente "
            "a persistencia, robustez, costo computacional e interpretabilidad."),
    ("OE3", "Diseñar e implementar un motor de decisión que traduzca los pronósticos y su incertidumbre en señales de "
            "compra, venta o espera."),
    ("OE4", "Validar el funcionamiento del motor para la toma de decisiones."),
]
cur.viñetas([f"**{k}.** {t} **({AV[k]} %)**" for k, t in objetivos])


# ================================================================================================
# 4. DELIMITACION
# ================================================================================================
cur = Cursor(encabezado("Alcances", 2)._p)
cur.p("La Tabla 2 resume lo que cubre el proyecto según el Anexo 1 {c:anexo1} y el estado de cada alcance.")
cur.titulo_tabla(2, "Alcances por objetivo específico y estado a la fecha.")
cur.tabla([
    ["Objetivo", "Alcance", "Estado"],
    ["OE1", "Datos públicos (XM/SIMEM, NOAA); adquisición automatizada por API REST; sincronización horaria de precio, demanda, generación, embalses y aportes; histórico ene-2019 a ago-2026; caracterización temporal y espectral.", "Cumplido (66.576 h)"],
    ["OE2", "Características compartidas (rezagos, promedios móviles, armónicas de calendario) para al menos 2 modelos; horizonte de 24 a 72 h; evaluación con MAE, RMSE, MAPE, persistencia y costo computacional.", "Cumplido (40 variables, 6 modelos)"],
    ["OE3", "Motor por percentiles del precio y su incertidumbre según rol; material visual con metadatos; dashboard integrado; biblioteca de 4-5 tipos de imágenes.", "Motor y dashboard funcionales; biblioteca pendiente"],
    ["OE4", "Backtesting de la estrategia de recomendación; validación del dashboard con 3-5 usuarios.", "Backtest hecho; usuarios pendiente"],
], [1.6, 10.2, 3.8], sz=8.5, alinear=["center", "left", "left"], negrita_col0=True)

cur = Cursor(encabezado("Limitaciones", 2)._p)
cur.p("La Tabla 3 lista lo que el proyecto no cubre y por qué se excluyó.")
cur.titulo_tabla(3, "Limitaciones y motivo de su exclusión.")
cur.tabla([
    ["Objetivo", "No se cubre", "Motivo"],
    ["OE1", "Datos propietarios; fenómenos intra-horarios; desglose de generación por tipo de recurso.", "Acceso a datos; el precio se fija por hora; alcance acordado en el Anexo 1 (solo agregados del sistema)."],
    ["OE2", "Horizontes fuera de 24-72 h; modelos sin respaldo en la literatura.", "Horizonte útil para decisiones de bolsa; exigencia de respaldo bibliográfico."],
    ["OE3", "Transacciones automáticas o integración con el despacho; simulación estructural del despacho.", "Regulación y responsabilidad; tiempo y alcance técnico del semestre."],
    ["OE4", "Estudio formal de usabilidad (> 5 personas); garantía fuera del rango histórico.", "Tiempo; el backtesting solo cubre condiciones ya ocurridas."],
], [1.6, 7.0, 7.0], sz=8.5, alinear=["center", "left", "left"], negrita_col0=True)

cur = Cursor(encabezado("Entregables", 2)._p)
cur.p("Según el Anexo 1 {c:anexo1}: dataset integrado 2019-2026, notebooks de análisis y modelado, repositorio en "
      "GitHub e informe de avance (**entregados**); módulo de recomendación con backtesting y dashboard (**versión 1 "
      "funcional**); biblioteca de imágenes de apoyo (**en curso**); manual de uso del motor (**borrador técnico**); e "
      "informe final, póster, presentación y video de la semana 16 (**pendientes**).")

# ================================================================================================
# 5. REVISION BIBLIOGRAFICA
# ================================================================================================
cur = Cursor(encabezado("Planteamiento del problema", 2)._p)
cur.p("El precio de bolsa colombiano es una señal horaria con saltos y cambios de régimen. En los datos del "
      f"proyecto (enero 2019 - agosto 2026, {f(C['oe1_filas'], 0)} horas descargadas de la API de XM {{c:xm}}) la "
      f"mediana es {f(D1['50%'])} COP/kWh, pero el percentil 99 llega a {f(D1['99%'])} y el máximo a {f(D1['max'])} "
      "COP/kWh: una cola larga que un promedio no describe. La Figura 1 muestra que los tramos de precio alto "
      "coinciden en buena parte con episodios de El Niño definidos por el índice ONI de NOAA {c:noaa} (2023-2024 y "
      "2026). Sin embargo, el mayor pico, un promedio diario de 2.499 COP/kWh el 30 de septiembre de 2024, ocurrió "
      "con ONI neutro, cuando el volumen útil de los embalses era el más bajo de todos los septiembres de 2019 a "
      "2025: el precio depende de la hidrología acumulada, no solo del índice climático.")
cur.figura(1, "f_serie_oni.png", "Precio de bolsa promedio diario 2019-2026 y episodios ENSO. Fuente: elaboración "
           "propia con datos de XM y NOAA.", ancho_cm=12)
cur.p("El problema impacta a generadores y comercializadores del mercado mayorista, en especial a empresas "
      "medianas sin herramientas analíticas: comprar en bolsa en la hora equivocada de un evento seco puede costar "
      "varias veces el precio típico. Ocurre todos los días y se agudiza en temporadas climáticas extremas {c:anexo1}. "
      "Es un problema complejo de ingeniería (ABET) porque tiene **requerimientos en conflicto** (precisión frente a "
      "costo computacional e interpretabilidad), **partes interesadas opuestas** (el generador quiere vender caro y el "
      "comercializador comprar barato), **múltiples disciplinas** (señales, estadística, aprendizaje automático, "
      "mercado eléctrico, interfaces) y **no tiene solución única**: ningún modelo gana en todos los regímenes "
      "(Sección 7).")

cur = Cursor(encabezado("Descripción de avances tecnológicos", 2)._p)
cur.p("El pronóstico de precios eléctricos (EPF) evolucionó de modelos econométricos univariados hacia modelos "
      "multivariados con selección de variables y redes profundas especializadas {c:weron2014,lago2021}. La Tabla 4 "
      "compara críticamente los trabajos más cercanos.")
cur.titulo_tabla(4, "Comparación crítica de trabajos previos frente a este proyecto.")
cur.tabla([
    ["Trabajo", "Datos y resolución", "Metodología", "Calidad de resultado", "Limitación frente a este proyecto"],
    ["Agudelo et al. 2015 {c:agudelo2015}", "Colombia, anterior a la reforma CREG de 2015", "Red neuronal NARX con entradas exógenas", "Pronóstico puntual sin incertidumbre", "Datos previos a 2015; sin decisión"],
    ["Muñoz-Santiago et al. 2017 {c:munoz2017}", "Colombia, serie agregada", "ARIMA con IGARCH", "Modela la persistencia de la volatilidad", "Un solo modelo; sin variables horarias"],
    ["Barrientos Marín et al. 2018 {c:barrientos2018}", "Colombia, mensual (185 obs.)", "ARMAX, NARX e híbrido; prueba DM", "Sin diferencia significativa entre modelos", "Resolución mensual, horizonte 12 meses"],
    ["Gallón y Barrientos 2021 {c:gallon2021}", "Colombia, horaria 2000-2017", "Series de tiempo funcionales (FPCA)", "MAPE promedio 6,7 % a 1 día", "Periodo calmo; univariado; sin decisión"],
    ["Lago et al. 2021 {c:lago2021}", "5 mercados europeos, horaria", "LEAR y redes profundas; benchmark abierto", "Define buenas prácticas y DM", "Sin mercados hidro-dominados"],
    ["Kapoor y Wichitaksorn 2023 {c:kapoor2023}", "Nueva Zelanda (hidro), diaria", "33 modelos con selección de variables", "Ningún modelo con MASE < 1 (mejor 1,26)", "Resolución diaria; sin recomendación"],
    ["Este proyecto", "Colombia, horaria 2019-2026", "Ensamble de 6 familias, bandas conformes y motor de decisión", f"MAPE {pct(ENS['MAPE'])} a 24 h; MASE {f(LIT['MASE_naive24h'], 3)}", "Validación con usuarios pendiente"],
], [3.2, 3.0, 3.4, 3.2, 3.6], sz=8)
cur.p("La brecha es concreta: los trabajos colombianos usan series mensuales, datos anteriores a la reforma de "
      "2015 o un solo modelo univariado, y ninguno cuantifica la incertidumbre ni la traduce en una recomendación. "
      "Los referentes internacionales fijan el método de evaluación {c:lago2021,diebold1995} y muestran que modelos "
      "simples con buena selección de variables compiten con el aprendizaje profundo {c:ziel2018,kapoor2023}, y la "
      "literatura reciente advierte que el MAE no basta para medir el valor de un pronóstico en decisiones de "
      "mercado {c:maciejowska2025}. No existe una plataforma para el mercado colombiano horario que combine "
      "procesamiento de señales, un ensamble validado por regímenes y un motor de decisión por rol: esa es la brecha "
      "que cierra este proyecto.")

# ================================================================================================
# 6. DISENO
# ================================================================================================
cur = Cursor(encabezado("Problema", 2)._p)
cur.p("Desde el diseño, el problema se traduce en un sistema de procesamiento de señales y decisión con tres "
      "bloques acoplados: (1) una cadena de adquisición y acondicionamiento que convierta cinco fuentes de "
      "resolución distinta (horaria, diaria y mensual) en una señal multivariada horaria continua, sin huecos ni "
      "fugas de información; (2) un estimador que, con la información disponible en el instante de corte, produzca "
      "la mediana y los cuantiles del precio para las siguientes 1 a 72 horas; y (3) una lógica que convierta ese "
      "pronóstico probabilístico en una acción discreta por rol.")

cur = Cursor(encabezado("Análisis de requerimientos del cliente, restricciones y factores", 2)._p)
cur.p("**Restricciones** {c:anexo1}: *económica* (software abierto y computadores del equipo), *ambiental* "
      "(maximizar la eficiencia computacional), *social* (el sistema apoya la decisión, no la sustituye), *legal* "
      "(solo información pública; posibles intervenciones de la CREG), *constructiva* (arquitectura modular) y *de "
      "tiempo* (16 semanas). **Factores**: salud pública (las escaladas de precio afectan a instituciones como las "
      "IPS), seguridad (acceso a las APIs), cultural (resistencia a recomendaciones basadas en IA), social (usuarios "
      "sin formación en probabilidad) y económico (tasa de cambio, precios de nube).")
cur.p("**Requerimientos del cliente.** Generadores y comercializadores medianos piden pronósticos de 24 a 72 h con "
      "una recomendación según su rol, un nivel de confianza visible y un dashboard usable sin conocimientos técnicos. "
      "La Tabla 5 muestra su traducción a especificaciones medibles y la evidencia de cada criterio.")
cur.titulo_tabla(5, "Traducción de requerimientos y restricciones en especificaciones técnicas medibles.")
cur.tabla([
    ["Requerimiento / restricción", "Especificación técnica medible", "Evidencia del criterio"],
    ["Pronóstico útil para decidir", "rMAE < 1 frente a la persistencia, con p < 0,05 en Diebold-Mariano", "Buenas prácticas EPF {c:lago2021,diebold1995}"],
    ["Horizonte de decisión", "Pronóstico horario de 1 a 72 h, evaluado por tramos de 24 h", "Alcance OE2 del Anexo 1"],
    ["Nivel de confianza visible", "Banda [q10, q90] con cobertura empírica de 80 % ± 5 puntos", "Inferencia conforme {c:romano2019}"],
    ["Robustez ante El Niño", "rMAE < 1 en todos los meses de prueba y en varios regímenes ENSO", "Validación de origen móvil {c:tashman2000}"],
    ["Eficiencia computacional", "Reentrenamiento del ensamble de 24 h < 15 min en un portátil sin GPU", "Restricción 6.3.2 del Anexo 1"],
    ["Recomendación por rol", "Señales comprar/vender/esperar; actuar entre 10 % y 40 % de las horas", "Guía del motor de decisión"],
    ["Sin fuga de información", "Solo variables disponibles en el corte (rezago ≥ 24 h)", "Decisión ABET 1 del Anexo 1"],
    ["Usable sin formación técnica", "Una acción por hora con semáforo; comprensión verificada con 3-5 usuarios", "Alcance OE4 del Anexo 1"],
], [4.0, 7.0, 4.6], sz=8.5)

cur = Cursor(encabezado("Estándares, normas y regulaciones", 2)._p)
cur.p("La Tabla 6 relaciona cada norma con la decisión de diseño que motivó.")
cur.titulo_tabla(6, "Normas y regulaciones aplicadas en el diseño.")
normas = [
    ("ISO 8601-1:2019 {c:iso8601}", "Marcas de tiempo AAAA-MM-DD hh:mm:ss en hora legal de Colombia (UTC−5, sin horario "
     "de verano), para unir sin ambigüedad fuentes horarias, diarias y mensuales."),
    ("Resolución CREG 101 018 de 2022 {c:creg2022}", "Uso exclusivo de información pública de XM/SIMEM; el sistema no "
     "actúa como agente ni ejecuta transacciones, y documenta la fecha de corte de cada modelo."),
    ("Ley 1581 de 2012 {c:ley1581}", "El dashboard no recolecta ni almacena datos personales."),
    ("ISO/IEC 25010:2011 {c:iso25010}", "Mantenibilidad: el motor lee el pronóstico por un contrato de datos "
     "(fuentes_pronostico.json) y es independiente del modelo."),
]
for fila, (n, dcs) in zip(t_normas.rows[1:], normas):
    for celda, txt in zip(fila.cells, (n, dcs)):
        p = celda.paragraphs[0]._p
        for r in _runs(txt, sz=9):
            p.append(r)
for fila in t_normas.rows[1:]:
    for celda in fila.cells:
        mar = celda._tc.tcPr.find(qn("w:tcMar"))
        if mar is not None:
            for lado in ("top", "bottom"):
                e = mar.find(qn(f"w:{lado}"))
                if e is not None:
                    e.set(qn("w:w"), "60")
cur = Cursor(t_normas._tbl)
cur.add(_par("", after=40))
cur.p("También se respeta la Ley 23 de 1982 (derechos de autor): toda fuente se cita y solo se usan librerías "
      "de código abierto.")

cur = Cursor(encabezado("Alternativas de solución", 2)._p)
cur.p("Se plantearon tres alternativas para el bloque de pronóstico, que determina la calidad de la señal que recibe "
      "el motor. Las tres comparten la cadena de datos de OE1 y el motor de OE3 (Tabla 7).")
alt = {
    "Descripción": [
        "Modelo econométrico único ARX+GARCH(1,1) {c:bollerslev1986,munoz2017}: regresión log-lineal del precio sobre "
        "hidrología, ONI y calendario, con varianza condicional que da la banda.",
        "Modelo único de aprendizaje profundo N-BEATSx {c:olivares2023} con variables exógenas, contexto de 168 h y "
        "cuantiles calibrados por inferencia conforme.",
        "Ensamble de seis modelos de familias distintas combinados por QRA ponderado por sMAPE en franjas de 6 h "
        "{c:nowotarski2015}, con puente a 72 h y bandas conformes."],
    "Ventajas": [
        "Entrena en segundos; coeficientes interpretables; modela la volatilidad agrupada.",
        "Captura patrones no lineales e intradía; multi-horizonte nativo.",
        "Combina errores poco correlacionados: menor error y mayor estabilidad mes a mes; pesos transparentes."],
    "Desventajas": [
        "Supuesto lineal; en 2026 empata con la persistencia y en 4 de 6 orígenes es peor.",
        "Caja negra; minutos de entrenamiento; en orígenes con corte a las 23:00 empata con la persistencia.",
        "Mayor costo y más piezas; su robustez histórica solo se mide en 2026."],
}
for fila in t_alt.rows[1:]:
    etiqueta = fila.cells[0].text.strip()
    unicas = []
    for c_ in fila.cells:
        if not unicas or c_._tc is not unicas[-1]._tc:
            unicas.append(c_)
    for k, celda in enumerate(unicas):
        p = celda.paragraphs[0]._p
        for e in list(p):
            if e.tag != qn("w:pPr"):
                p.remove(e)
        for r in _runs(f"**{etiqueta}:** " + alt[etiqueta][k], sz=8.5):
            p.append(r)
t_alt._tbl.addprevious(_par("**Tabla 7.** Alternativas de solución para el bloque de pronóstico.", jc="center",
                            after=60, before=120, sz=10, keep_next=True))

cur = Cursor(encabezado("Evaluación y selección de la alternativa óptima", 2)._p)
cur.p("Se definieron cinco criterios técnicos más el costo (Tabla 8), con rangos derivados de la Tabla 5.")
crit = [
    ("Precisión: rMAE frente a persistencia", "< 1; ideal ≤ 0,85", "{c:lago2021,hyndman2006}"),
    ("Robustez: peor rMAE mensual en 2026", "< 1 en todos los meses", "{c:tashman2000}"),
    ("Calibración: cobertura de [q10, q90] (%)", "80 ± 5", "{c:romano2019}"),
    ("Costo computacional: reentrenamiento del bloque de 24 h (min)", "< 15", "Anexo 1, 6.3.2"),
    ("Interpretabilidad (escala 1-5)", "≥ 3", "{c:lundberg2017}"),
    ("Costo económico adicional (COP)", "0", "Anexo 1, 6.3.1"),
]
filas_crit = list(t_crit.rows[1:])
while len(filas_crit) < len(crit):
    nueva = copy.deepcopy(filas_crit[-1]._tr)
    filas_crit[-1]._tr.addnext(nueva)
    filas_crit = list(t_crit.rows[1:])
for fila, valores in zip(filas_crit, crit):
    unicas = []
    for c_ in fila.cells:
        if not unicas or c_._tc is not unicas[-1]._tc:
            unicas.append(c_)
    for celda, txt in zip(unicas, valores):
        p = celda.paragraphs[0]._p
        for e in list(p):
            if e.tag != qn("w:pPr"):
                p.remove(e)
        for r in _runs(txt, sz=9):
            p.append(r)
        mar = celda._tc.tcPr.find(qn("w:tcMar"))
        if mar is not None:
            for lado in ("top", "bottom"):
                e = mar.find(qn(f"w:{lado}"))
                if e is not None:
                    e.set(qn("w:w"), "60")
t_crit._tbl.addprevious(_par("**Tabla 8.** Criterios de ingeniería para evaluar las alternativas.", jc="center",
                             after=60, before=120, sz=10, keep_next=True))
cur = Cursor(t_crit._tbl)
cur.add(_par("", after=40))
cur.p("**Método de decisión:** matriz ponderada con puntajes de 1 a 5 y reglas fijadas antes de ver el resultado: "
      "precisión y robustez 5 si rMAE ≤ 0,75 (robustez ≤ 0,85), 4 si ≤ 0,85 (≤ 0,90), 3 si ≤ 0,95, 2 si ≤ 1,00 y 1 si "
      "> 1; costo 5 si < 1 min, 4 si < 5 min, 3 si < 10 min; interpretabilidad 5 con coeficientes, 3 con pesos por "
      "modelo y 2 si es caja negra. La cobertura no se puntúa porque las alternativas 2 y 3 comparten la misma banda.")

cur = Cursor(buscar_parrafo("Matriz de decisión")._p)
nbx_s = t_nn / 2
cur.titulo_tabla(9, "Matriz de decisión ponderada (puntaje 1-5; valor medido en 2026 entre paréntesis).")
cur.tabla([
    ["Criterio (peso)", "Sol. 1: ARX+GARCH", "Sol. 2: N-BEATSx", "Sol. 3: Ensamble"],
    ["Precisión (0,35)", f"2 ({f(H['ARX+GARCH']['rMAE'], 3)})", f"4 ({f(H['N-BEATSx']['rMAE'], 3)})", f"5 ({f(ENS['rMAE'], 3)})"],
    ["Robustez (0,25)", "1 (1,009)", "3 (0,933)", "5 (0,830)"],
    ["Costo computacional (0,15)", f"5 ({f(COSTO['ARX+GARCH_s'], 1)} s)", f"4 (≈{f(nbx_s / 60, 1)} min)", f"3 ({f(t_total / 60, 1)} min)"],
    ["Interpretabilidad (0,15)", "5", "2", "3"],
    ["Costo económico (0,10)", "5 ($0)", "5 ($0)", "5 ($0)"],
    ["**Total ponderado**", "**2,95**", "**3,55**", "**4,40**"],
], [5.0, 3.6, 3.6, 3.6], alinear=["left", "center", "center", "center"], sz=8.5)
cur.p("**Selección.** El ensamble obtiene 4,40 frente a 3,55 y 2,95. Es la única alternativa que cumple con holgura "
      f"la precisión (rMAE {f(ENS['rMAE'], 3)}) y la robustez (rMAE < 1 en los ocho meses de 2026, peor mes 0,830), con "
      f"{f(t_total / 60, 1)} min de reentrenamiento del bloque de 24 h. Su menor interpretabilidad se compensa mostrando "
      f"los pesos del combinador y la banda de incertidumbre. El tiempo de N-BEATSx se estimó como la mitad del medido "
      f"para las dos redes ({f(t_nn / 60, 1)} min).")

cur = Cursor(encabezado("Análisis de riesgo del diseño definitivo", 2)._p)
cur.p("La Tabla 10 resume los riesgos principales del diseño final; el análisis completo está en el Anexo 2.")
cur.titulo_tabla(10, "Matriz de riesgo del diseño definitivo y plan de mitigación.")
cur.tabla([
    ["Tipo", "Riesgo", "Prob.", "Impacto", "Mitigación"],
    ["Técnico", "Fuga de información (variables no disponibles en el corte).", "Media", "Alto", "Rezago ≥ 24 h verificado por variable; precio de oferta marginal excluido."],
    ["Técnico", "Datos corruptos en la fuente o el procesamiento.", "Media", "Alto", "Pruebas en copia aislada; se detectó y corrigió la demanda del 4-5 de ago. de 2026."],
    ["Regulatorio", "Intervención transitoria de la CREG en la formación del precio.", "Media", "Alto", "Fecha de corte documentada; el ensamble de 24 h se reentrena en < 8 min."],
    ["Técnico", "Ningún modelo mejora de forma significativa en El Niño.", "Alta", "Medio", "Método «banda»: el motor espera cuando la incertidumbre es alta."],
    ["Social/ético", "El usuario toma la recomendación como garantía.", "Baja", "Alto", "Mostrar la incertidumbre y advertir que no es asesoría financiera."],
    ["Administrativo", "Retraso de OE3/OE4.", "Media", "Alto", "Priorizar su cierre sobre nuevas pruebas de modelos."],
], [2.3, 4.5, 1.3, 1.5, 6.0], sz=8.5)

# ------------------------------------------------------------------------------------------------
# 6.8 DISENO DEFINITIVO: metodologia, variables, modelos y decisiones
# ------------------------------------------------------------------------------------------------
cur = Cursor(encabezado("Diseño definitivo", 2)._p)
cur.p("**Metodología de diseño.** Cada elemento del sistema (variable, modelo, combinador, calibración y regla de "
      "decisión) se definió con el mismo procedimiento: (1) se formula frente a una alternativa o línea base; (2) se "
      "evalúa fuera de muestra, con entrenamiento en 2019-2025, ajuste de hiperparámetros contra 2025 y prueba en "
      "2026; (3) se adopta solo si mejora con significancia en la prueba de Diebold-Mariano (p < 0,05) y, cuando hay "
      "aleatoriedad, en las 10 particiones evaluadas; y (4) se documenta en la bitácora del repositorio, incluidos los "
      "intentos fallidos. La Figura 2 muestra el resultado de ese proceso y las Tablas 11 a 13 documentan cada "
      "definición.")
cur.figura(2, "f_diagrama_bloques.png", "Diagrama de bloques de la plataforma y su relación con los objetivos "
           "específicos. Fuente: elaboración propia.", ancho_cm=10.5)
cur.p("**Definición de variables.** Todas las variables se construyen desplazando primero la serie al menos 24 h, "
      "de modo que en el instante de corte solo se usa información ya publicada. Cada familia se justificó con un "
      "resultado de la caracterización de OE1 (Tabla 11).")
cur.titulo_tabla(11, "Las 40 variables predictoras, agrupadas por familia (y = precio de bolsa; t = hora objetivo).")
cur.tabla([
    ["Familia (n.º)", "Definición", "Justificación (OE1)"],
    ["Precio rezagado (2)", "y~t−24~, y~t−168~", "Picos de 24 h y 168 h del periodograma"],
    ["Medias móviles (3)", "Media de y~t−24~ en ventanas de 24 h, 7 días y 30 días", "Nivel y tendencia recientes"],
    ["Volatilidad (4)", "Desviación estándar en 24 h y 7 días, rango de 24 h y razón entre ambas desviaciones", "Heterocedasticidad del precio (colas pesadas)"],
    ["Calendario (6)", "sen/cos(2πh/24), sen/cos(2πd/7), sen/cos(2πd~a~/365,25)", "Armónicos de 24 h, 168 h y anual; continuidad del ciclo (23 h → 0 h)"],
    ["Hidrología (12)", "Embalses y aportes rezagados 24 h, sus cambios a 1 y 7 días, medias de 7 y 30 días y razón frente a la media de 30 días", "Correlación negativa con el precio; los aportes pesan más a 7 días (−0,309)"],
    ["Demanda (4)", "Demanda rezagada 24, 48 y 72 h y su media de 24 h", "Correlación +0,325; la contemporánea se excluye"],
    ["Clima y eventos (2)", "ONI con rampa causal; indicador de pandemia (25-mar a 31-ago-2020)", "Correlación del ONI +0,387"],
    ["Festivos (7)", "Día festivo, festivo hace 1, 2, 3 y 7 días, y desajuste de tipo de día frente a t−24 h y t−168 h", "Reducen el error de N-BEATSx en festivos de 74,1 a 62,6 COP/kWh (p = 0,0064)"],
], [3.0, 7.4, 5.2], sz=8.5, negrita_col0=True)
cur.p("**Definición de modelos.** Los seis votantes se eligieron de la literatura del área y por pertenecer a "
      "familias distintas, para que sus errores no estén correlacionados. La Tabla 12 resume su fundamento, "
      "configuración y forma de ajuste.")
cur.titulo_tabla(12, "Modelos del ensamble: fundamento, configuración y ajuste.")
cur.tabla([
    ["Modelo", "Fundamento", "Configuración", "Cómo se ajustó"],
    ["Persistencia", "Referencia ingenua {c:lago2021}", "ŷ~t~ = y~t−24~", "Sin parámetros"],
    ["XGBoost", "Árboles con gradiente {c:chen2016}", "500 árboles, profundidad 3, tasa 0,01, submuestreo 0,8; objetivo log(y); 40 variables", "Búsqueda en rejilla validada contra 2025"],
    ["ARX+GARCH", "ARIMA-IGARCH colombiano {c:munoz2017,bollerslev1986}", "Media log-lineal con 15 regresoras (hidrología, ONI, pandemia, calendario, festivos) + log y~t−24~; varianza GARCH(1,1)", "27 configuraciones validadas contra 2025; se mantuvo GARCH(1,1) porque el ganador de validación (GJR(2,1)) tenía persistencia 1,02 (explosiva)"],
    ["N-BEATSx", "Expansión de bases con exógenas {c:olivares2023}", "Contexto de 168 h, horizonte 24 h, 1.000 pasos; exógenas: embalses, aportes, demanda rezagada, ONI, calendario, festivos", "Contexto elegido entre 1, 2 y 3 semanas contra 2025"],
    ["N-HiTS", "Interpolación jerárquica {c:challu2023}", "Mismo contexto, horizonte y exógenas que N-BEATSx", "Igual que N-BEATSx"],
    ["GARCH-ged", "LE-GARCH-t, mejor modelo en Nueva Zelanda {c:kapoor2023,nelson1991}", "Regresión LASSO {c:tibshirani1996} por paso horario + varianza GARCH con errores de distribución generalizada (GED)", "Elegido entre 10 variantes GARCH como sexto votante: MAE del ensamble 41,40 (p = 0,002)"],
    ["Combinador", "QRA {c:nowotarski2015}", "Pesos ≥ 0 e intercepto por franja de 6 h; objetivo alineado con el sMAPE (Ec. 1)", "Validación cruzada de 5 pliegues por días; 10 particiones"],
    ["Bandas", "Regresión cuantílica conforme {c:romano2019}", "Cuantiles q10/q90 de N-BEATSx ensanchados por un margen conforme de 30 días móviles", "Cobertura medida en 2026"],
], [2.2, 3.3, 5.6, 4.5], sz=8)
cur.ecuacion("min~w≥0,b~ Σ~t∈g~ ω~t~ · | y~t~ − b − Σ~k~ w~k~ · ŷ~k,t~ |,   ω~t~ = 2 / ( |y~t~| + |ŷ~t~| )", 1)
cur.p("La Ecuación 1 es el programa lineal del combinador para cada franja *g*: ŷ~k,t~ es el pronóstico del modelo "
      "*k* y los pesos ω~t~ se actualizan iterativamente para minimizar el sMAPE. El producto de 72 h reutiliza el "
      "ensamble de 24 h en los pasos 1-23 y combina, en el resto, un LEAR (72 modelos LASSO, uno por paso), dos "
      "N-BEATSx y una persistencia semanal.")
cur.p("**Modelos evaluados y descartados.** También se probaron, sobre las mismas horas de 2026, modelos de la "
      "literatura que no entraron al diseño (entre paréntesis, rMAE frente a la persistencia): CatBoost (1,016), "
      "modelo hurdle de dos etapas (1,028), Markov-Switching de 2 regímenes (0,990), mínimos cuadrados recursivos "
      "(1,052), TFT (1,020), LSTM (1,173), LightGBM (≈ 1,10, igual a XGBoost) y Prophet (≈ 1,70, el más "
      "débil en todas las pruebas; estos dos últimos, con una versión anterior del pipeline). El LEAR de 24 h "
      "(0,940) no mejoró el ensamble de 24 h, pero sí se usa a 72 h. Como votantes adicionales tampoco sirvieron: "
      "agregar ocho de ellos empeoró el ensamble de MAE 46,02 a 46,45 (p = 0,013) y TFT o LSTM no dieron diferencia "
      "significativa (p = 0,89 y 0,38). La lección de diseño es que importa la diversidad de familias, no la cantidad "
      "de votantes.")
cur.p("**Decisiones de diseño.** La Tabla 13 documenta las decisiones principales, incluidas las seis que el Anexo "
      "1 exige registrar (sección ABET): cada una con la alternativa descartada y la evidencia que la sustenta.")
cur.titulo_tabla(13, "Decisiones de diseño, alternativa descartada y evidencia.")
cur.tabla([
    ["Decisión adoptada", "Alternativa descartada", "Evidencia"],
    ["Ventana de datos ene-2019 a ago-2026", "Incluir 2015-2018", "Cambio estructural de la regulación CREG de 2015, con margen de estabilización; rango aprobado en el Anexo 1"],
    ["Excluir demanda y generación contemporáneas y el precio de oferta marginal", "Usarlas como predictoras", "Se fijan en el mismo despacho que el precio; el precio de oferta marginal correlaciona 0,9998 con el precio (fuga pura)"],
    ["Rezago mínimo de 24 h en todas las variables", "Rezagos de 1 h", "Una variable con rezago de 1 h inflaba la precisión de forma irreal y se eliminó del pipeline"],
    ["Hidrología diaria a horaria por interpolación lineal", "Escalón (repetir el valor diario)", "Ablación 2×2: mejora el horizonte de 72 h sin afectar 24 h; segura porque la variable entra rezagada 24 h"],
    ["ONI mensual con rampa causal", "Escalón mensual o interpolación con el mes siguiente", "La interpolación usaría el dato del mes siguiente (fuga); la rampa mejoró la estabilidad entre regímenes en la ablación"],
    ["ONI como variable continua", "Bandera binaria El Niño o categorías difusas NOAA", "Las categorías difusas empeoraron N-BEATSx (56,11 → 58,47) y N-HiTS (57,08 → 60,53); la bandera binaria pierde la intensidad del evento (no se probó)"],
    ["Pandemia como bandera binaria", "Excluir 2020", "Evento administrativo de inicio y fin definidos; conserva un año de datos"],
    ["Transformación logarítmica del precio (XGBoost, ARX)", "Precio en COP/kWh", "Distribución asimétrica: media 336,7 frente a mediana 242,3 COP/kWh"],
    ["Walk-forward de 6 orígenes con regímenes ENSO distintos", "Un solo corte de prueba", "Un corte único ocultaba que XGBoost pierde contra la persistencia en los 6 orígenes"],
    ["Combinador por franjas de 6 h con objetivo sMAPE", "Pesos globales u objetivo MAE", "Por franja: MAE 43,24 frente a 44,38 global (p = 0,0006); objetivo sMAPE: MAPE 10,66 → 10,47 % en 10/10 particiones"],
    ["Bandas conformes adaptativas (30 días)", "Bandas crudas o calibración estática", "Cobertura 62,6 % cruda y 67-72 % estática, frente a 78 % adaptativa"],
    ["Motor elegido por backtest con frecuencia de acción del 10 al 40 %", "Un umbral fijo único", "El método «banda» gana en los 4 casos (Tabla 18)"],
], [4.6, 3.8, 7.2], sz=8)
cur.p("**Motor de decisión (método «banda»).** (1) Se calculan los percentiles p25 y p75 del precio de entrenamiento "
      "y el percentil 75 del ancho de banda histórico; (2) si el ancho q90 − q10 de una hora supera ese percentil, se "
      "emite «esperar»; (3) si no, la mediana q50 se compara con p25 y p75: por debajo de p25 el comercializador "
      "recibe «comprar» y el generador «retener»; por encima de p75, «evitar compra» y «vender»; en otro caso, "
      "«esperar».")
cur.p("**Funcionamiento del prototipo.** El pipeline se ejecutó de principio a fin: los notebooks 01 a 03 se "
      "reejecutaron sin errores (01 y 02 en una copia aislada del proyecto), el ensamble se regeneró con los datos "
      "corregidos y el dashboard corre localmente con los datos de 2026 (Figura 3). El código completo está en el "
      "Anexo 3 y los resultados que demuestran el funcionamiento de cada bloque, en la Sección 7.")
cur.figura(3, "f_dashboard_operador.png", "Prototipo del dashboard en funcionamiento, vista Operador (5 de "
           "agosto de 2026, 08:00 h, rol generador).", ancho_cm=9.5)

# ================================================================================================
# 7. PRUEBAS
# ================================================================================================
cur = Cursor(encabezado("Planeación del diseño del experimento", 2)._p)
cur.p("**Pregunta técnica.** ¿La plataforma pronostica el precio de bolsa horario a 24-72 h con un error "
      "significativamente menor que la persistencia (rMAE < 1) y con bandas cuya cobertura esté cerca del 80 %?")
cur.p("**Hipótesis.** Para cada hora *t*, d~t~ = |e~t~^pers^| − |e~t~^ens^| es la diferencia entre el error absoluto "
      "de la persistencia y el del ensamble. **H~0~:** E[d~t~] = 0 (el ensamble no es más preciso). **H~1~:** "
      "E[d~t~] > 0 (el ensamble tiene menor error). La prueba es **de una cola**, porque la especificación exige "
      "superar a la referencia; se aplica a 24 h y a 72 h. Para los modelos individuales en cada origen se usa la de "
      "dos colas, para detectar si ganan o pierden.")
cur.p("**Variables.** La variable dependiente es el error absoluto horario |y~t~ − ŷ~t~| (COP/kWh), resumido en MAE, "
      "RMSE, MAPE y sMAPE. Los factores son: modelo (7 niveles: seis votantes y ensamble), horizonte (3 tramos de "
      "24 h), origen temporal (6 regímenes ENSO), pesos del combinador (validación cruzada con 10 particiones o "
      "versión desplegable causal) y hora de lanzamiento (00:00 o 23:00). El método es una **evaluación fuera de "
      "muestra con origen móvil** (walk-forward) {c:tashman2000} con comparaciones pareadas hora a hora.")

cur = Cursor(encabezado("Instrumentación y validación", 3)._p)
cur.p("**Instrumento:** la serie oficial de precio de XM {c:xm}, horaria y con hasta cinco decimales en COP/kWh; su "
      "error es despreciable frente a los de pronóstico. Equipo: AMD Ryzen 7 7730U, 15,4 GB de RAM, sin GPU. "
      "**Repeticiones:** 5.184 pares hora a hora en 2026 (216 días), muy por encima de 30; la autocorrelación se "
      "corrige con la varianza de Newey-West {c:newey1987}. El walk-forward suma seis orígenes de 5.184 a 8.784 horas, "
      "y la combinación se repitió con 10 particiones. **Significancia:** α = 0,05.")
cur.p("**Invalidación interna controlada.** *Historia y maduración*: seis orígenes con regímenes distintos. "
      "*Instrumentación*: mismo dataset, periodo y código de métricas para todos; una auditoría de reproducibilidad "
      "detectó que la demanda del 4-5 de agosto de 2026 estaba entre 70 % y 86 % por debajo del valor real, y se "
      "corrigió y reentrenó. *Regresión estadística*: se usa todo 2026, sin elegir periodos extremos. *Fuga*: sin "
      "variables contemporáneas; hiperparámetros ajustados contra 2025. La composición del ensamble (incluido "
      "GARCH-ged) sí se eligió observando 2026; por eso se reporta la versión desplegable como cifra de operación.")
cur = Cursor(encabezado("Marco normativo de las pruebas", 3)._p)
cur.p("No existe una norma IEC o IEEE para evaluar pronósticos de precio de electricidad. Se sigue el protocolo del "
      "benchmark abierto de Lago et al. {c:lago2021}, referente de la comunidad EPF: evaluación fuera de muestra, "
      "referencia ingenua, error relativo y prueba de Diebold-Mariano {c:diebold1995} con varianza HAC {c:newey1987}; "
      "las métricas adimensionales siguen a Hyndman y Koehler {c:hyndman2006} y las fechas, la ISO 8601 {c:iso8601}.")

cur = Cursor(encabezado("Ejecución del plan experimental", 2)._p)
cur.p("El plan se ejecutó en el orden previsto: dataset maestro, características, entrenamiento con 2019-2025 y "
      "pronóstico de 2026, walk-forward, combinación, producto de 72 h, calibración de bandas y backtesting del "
      "motor. Hubo dos ajustes: la corrección de la demanda (que obligó a repetir el entrenamiento y la combinación) "
      "y el análisis de la hora de lanzamiento, al detectar que las redes usan un corte a las 00:00, más favorable "
      "que el de las 23:00. Ecuaciones de procesamiento (N horas, y~t~ real, ŷ~t~ pronóstico):")
cur.ecuacion("MAE = (1/N) Σ~t~ | y~t~ − ŷ~t~ |,     RMSE = √[ (1/N) Σ~t~ ( y~t~ − ŷ~t~ )² ]", 2)
cur.ecuacion("MAPE = (100 %/N) Σ~t~ | y~t~ − ŷ~t~ | / y~t~,     sMAPE = (100 %/N) Σ~t~ | y~t~ − ŷ~t~ | / [ ( |y~t~| + |ŷ~t~| ) / 2 ]", 3)
cur.ecuacion("rMAE = MAE~modelo~ / MAE~pers~,     MASE = MAE~modelo~ / [ (1/(T−24)) Σ~t~ | y~t~ − y~t−24~ | ]~entrenamiento~", 4)
cur.ecuacion("DM = d̄ / √( V̂~HAC~(d̄) ),     d~t~ = | e~t~^ref^ | − | e~t~^modelo^ |", 5)
cur.ecuacion("P(f~k~) = (1/N) | Σ~n=0~^N−1^ x[n] · e^−j2πkn/N^ |²,     f~s~ = 1 muestra/h", 6)
cur.p("La Ecuación 6 es el periodograma de la serie sin tendencia {c:oppenheim2010}; con f~s~ = 1 muestra/h, el "
      "período en horas es 1/f. El filtro de Savitzky-Golay {c:savitzky1964} ajusta en cada ventana de 721 h un "
      "polinomio de orden 3 por mínimos cuadrados. La cobertura es el porcentaje de horas con q10 ≤ y~t~ ≤ q90 y la "
      "ventaja del motor, la diferencia entre el precio medio en las horas de acción y el del periodo.")

cur = Cursor(encabezado("Resultados", 2)._p)
cur.p(f"**OE1 — Datos y caracterización.** El dataset tiene {f(C['oe1_filas'], 0)} horas continuas (1-ene-2019 a "
      "5-ago-2026), sin huecos ni duplicados. En el periodograma (Figura 4) la componente de 24 h es la dominante: "
      f"su pico concentra por sí solo el {pct(C['oe1_picos'][0][1], 1)} de la potencia entre 2 y 2.000 h, el mayor del "
      "espectro; le siguen el armónico de 12 h, una componente semanal de 168 h y ciclos de 20 a 42 días asociados "
      "a la hidrología. En la comparación de filtros de tendencia sobre El Niño 2023-2024, Savitzky-Golay alcanzó el "
      "90 % de la subida real 376 h antes que el promedio móvil (348 h frente a 724 h) y dejó menos ruido residual "
      "(desviación estándar de 117,70 frente a 138,03 COP/kWh). La correlación con el precio fue positiva para el "
      f"ONI ({f(COR['oni'], 3)}), la demanda ({f(COR['demanda'], 3)}) y la generación ({f(COR['generacion'], 3)}), y "
      f"negativa para aportes ({f(COR['aportes_hidricos'], 3)}) y embalses ({f(COR['volumen_embalses'], 3)}).")
cur.figura(4, "f_periodograma.png", "Periodograma del precio de bolsa sin tendencia. Líneas rojas: períodos de "
           "12, 24 y 168 h.", ancho_cm=11.5)
cur.p("**OE2 — Pronóstico de 24 h.** La Tabla 14 resume el desempeño sobre las 5.184 horas de prueba de 2026.")
cur.titulo_tabla(14, "Desempeño a 24 h fuera de muestra (1-ene a 5-ago de 2026, n = 5.184 h).")
filas = [["Modelo", "MAE\n(COP/kWh)", "RMSE\n(COP/kWh)", "MAPE\n(%)", "sMAPE\n(%)", "rMAE", "DM (t)"]]
for m in ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS", "GARCH-ged", "Ensamble (6 votantes)"]:
    r = H[m]
    dmt = "—" if m == "Persistencia" else f(r["DM_t"], 2)
    nombre = f"**{m}**" if m.startswith("Ensamble") else m
    filas.append([nombre, f(r["MAE"]), f(r["RMSE"]), f(r["MAPE"]), f(r["sMAPE"]), f(r["rMAE"], 3), dmt])
cur.tabla(filas, [4.4, 2.0, 2.0, 1.7, 1.7, 1.6, 1.6], alinear=["left"] + ["center"] * 6, sz=8.5)
cur.p("DM (t): Diebold-Mariano frente a la persistencia (positivo = menor error). Con 10 particiones del combinador: "
      f"MAE 41,10 ± 0,08, MAPE 10,49 % ± 0,02. Versión desplegable (pesos solo con días anteriores): MAE "
      f"{f(DES['MAE'])} y MAPE {pct(DES['MAPE'])}, frente a {f(DES['MAE_persistencia_mismas_horas'])} y "
      f"{pct(DES['MAPE_persistencia_mismas_horas'])} de la persistencia en las mismas horas.", sz=9.5)
cur.p("**Robustez entre regímenes.** La Tabla 15 muestra el MAE de los modelos individuales en los seis orígenes "
      "walk-forward (O1 a O5: 12 meses cada uno con corte a las 23:00; O6: 2026).")
orig = ["Origen 1", "Origen 2", "Origen 3", "Origen 4", "Origen 5", "Origen 6"]
cur.titulo_tabla(15, "MAE (COP/kWh) por origen walk-forward y resultado de Diebold-Mariano frente a la persistencia.")
filas = [["Modelo", "O1\n2020-21", "O2\n2021-22", "O3\n2022-23", "O4\n2023-24", "O5\n2024-25", "O6\n2026", "Gana / pierde"]]
for m in ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"]:
    gp = "referencia" if m == "Persistencia" else f"{WFR[m]['gana_sig']} / {WFR[m]['pierde_sig']}"
    filas.append([m] + [f(WF[m][o]) for o in orig] + [gp])
cur.tabla(filas, [3.0, 1.55, 1.55, 1.55, 1.55, 1.55, 1.55, 2.3], alinear=["left"] + ["center"] * 7, sz=8.5)
cur.p("**Horizonte de 72 h.** La Tabla 16 compara el producto de 72 h con la persistencia medida desde el mismo corte.")
cur.titulo_tabla(16, "Pronóstico de 72 h por tramo, prueba 2026.")
filas = [["Tramo", "MAE ensamble", "MAPE ensamble", "MAE persistencia", "MAPE persistencia", "rMAE", "DM (t)"]]
for k in ["1-24 h", "25-48 h", "49-72 h", "Global 1-72 h"]:
    r = T72[k]
    filas.append([k, f(r["MAE"]), pct(r["MAPE"]), f(r["MAE_persistencia"]), pct(r["MAPE_persistencia"]),
                  f(r["rMAE"], 3), f(r["DM_t"], 2)])
cur.tabla(filas, [3.0, 2.1, 2.1, 2.3, 2.3, 1.6, 1.6], alinear=["left"] + ["center"] * 6, sz=8.5)
cur.p("**Incertidumbre y costo.** La Tabla 17 reúne la cobertura de las bandas y el tiempo de cada modelo del "
      "bloque de 24 h en el mismo equipo (el canal de 72 h no se cronometró).")
cur.titulo_tabla(17, "Cobertura de la banda [q10, q90] (objetivo 80 %) y costo computacional.")
cur.tabla([
    ["Banda", "Cobertura (%)", "Ancho (COP/kWh)", "Modelo", "Tiempo"],
    ["24 h", f(BAN["24h"], 1), f(BAN["24h_ancho"], 1), "XGBoost", f"{f(COSTO['XGBoost_s'], 2)} s"],
    ["72 h, 1-24 h", f(BAN["72h_1-24"], 1), f(BAN["72h_1-24_ancho"], 1), "ARX+GARCH", f"{f(COSTO['ARX+GARCH_s'], 2)} s"],
    ["72 h, 25-48 h", f(BAN["72h_25-48"], 1), f(BAN["72h_25-48_ancho"], 1), "GARCH-ged", f"{f(COSTO['GARCH-ged_s'], 1)} s"],
    ["72 h, 49-72 h", f(BAN["72h_49-72"], 1), f(BAN["72h_49-72_ancho"], 1), "N-BEATSx y N-HiTS", f"{f(t_nn / 60, 1)} min"],
    ["", "", "", "Combinador QRA", f"{f(COSTO['Combinador_sMAPE_s'], 1)} s"],
    ["", "", "", "**Total bloque 24 h**", f"**{f(t_total / 60, 1)} min**"],
], [2.8, 2.4, 2.8, 4.2, 2.4], alinear=["left", "center", "center", "left", "center"], sz=8.5)
cur.p("**OE3 y OE4 — Motor de decisión.** La Tabla 18 muestra el método elegido para cada horizonte y rol "
      f"(precio medio 2026: {f(C['oe3_precio_medio_2026'], 1)} COP/kWh). La ventaja es la diferencia entre el precio "
      "medio en las horas en que el motor recomienda actuar y el del periodo. En los cuatro casos gana el método "
      "«banda»; los métodos fijo y rodante dan ventajas menores (156,8 a 313,3 COP/kWh), y el fijo a 72 h para el "
      "generador actúa en el 41 % de las horas, fuera del rango válido.")
cur.titulo_tabla(18, "Backtesting del motor de decisión sobre 2026 (método elegido).")
filas = [["Horizonte", "Rol", "Método", "Ventaja (COP/kWh)", "Frecuencia de acción"]]
for r in BT:
    if r["ganador"]:
        filas.append([r["horizonte"], r["rol"], r["metodo"], f(r["ventaja_cop_kwh"], 1), pct(r["frecuencia_accion"] * 100, 1)])
cur.tabla(filas, [2.2, 3.6, 2.4, 3.4, 3.4], sz=8.5, alinear=["center", "left", "left", "center", "center"])

cur = Cursor(encabezado("Análisis e interpretación de resultados", 2)._p)
cur.p("**Comparación con los criterios.** El ensamble cumple la precisión con holgura: rMAE de "
      f"{f(ENS['rMAE'], 3)} a 24 h y {f(G72['rMAE'], 3)} a 72 h, por debajo de 1 y del ideal de 0,85; cumple la "
      f"robustez mensual (rMAE < 1 en los ocho meses de 2026) y el costo ({f(t_total / 60, 1)} min < 15 min). La "
      f"cobertura ({f(BAN['24h'], 1)} % a 24 h y {f(BAN['72h'], 1)} % a 72 h) queda dentro de 80 ± 5 puntos, algo por "
      "debajo del nominal.")
cur.p(f"**Prueba de hipótesis.** Diebold-Mariano del ensamble frente a la persistencia: t = {f(ENS['DM_t'], 2)} a "
      f"24 h, t = {f(DES['DM_vs_persistencia'][0], 2)} en la versión desplegable y t = {f(G72['DM_t'], 2)} a 72 h, todos "
      "con p unilateral < 0,001. **Se rechaza H~0~ en los tres casos**: el ensamble es significativamente más preciso "
      "que la persistencia.")
ph = C["oe2_ens_corte23"]
cur.p("**Lectura honesta de la robustez.** En los orígenes históricos 1 a 5 ningún modelo individual supera de "
      f"forma consistente a la persistencia: N-BEATSx no le gana en ninguno, pierde en el O1 y empata en los demás "
      f"(MAE medio {f(mae15('N-BEATSx'))} frente a {f(mae15('Persistencia'))}), y XGBoost pierde en los seis. El "
      "ensamble completo solo puede evaluarse en 2026; extenderlo a los orígenes 1-5 es trabajo pendiente. Además, las "
      "redes usan un corte a las 00:00, cuando ya se conoce la primera hora del día de despacho; con el pronóstico "
      f"lanzado a las 23:00 el ensamble sube a MAE {f(ph['MAE'])} (MAPE {pct(ph['MAPE'])}; cálculo previo a la "
      f"corrección de la demanda, que afecta 48 de 5.184 horas), aún mejor que la persistencia ({f(PER['MAE'])}).")
cur.p("**Validación externa (Tabla 4).** El MAPE es mayor que el del antecedente colombiano horario {c:gallon2021} "
      "(10,45 % frente a 6,7 %), pero ese trabajo evaluó 2000-2017, sin los eventos extremos de 2023-2026 y con un "
      "modelo univariado. Frente a Nueva Zelanda {c:kapoor2023}, el MASE es menor que 1 mientras ninguno de sus 33 "
      "modelos baja de 1,26; como ellos pronostican precio diario transformado con Box-Cox, se lee como «superar al "
      "propio ingenuo en una tarea más difícil», no como comparación numérica directa.")
cur.p("**Evidencia por objetivo.** OE1: 66.576 horas sin huecos y armónicos de 24, 12 y 168 h que definieron las "
      "variables (Figura 4, Tabla 11). OE2: seis modelos comparados en precisión, robustez, costo e "
      "interpretabilidad (Tablas 14 a 17); SHAP {c:lundberg2017} y permutación sobre XGBoost coinciden en las tres "
      "variables más importantes (precio rezagado 24 h, media de 24 h y precio rezagado 168 h; Spearman 0,811). OE3: "
      "motor con tres métodos, dos roles y dos horizontes (Tabla 18, Figura 3). OE4: backtesting hecho; falta la "
      "validación con usuarios.")

# ================================================================================================
# 8. IMPACTOS
# ================================================================================================
cur = Cursor(encabezado("Impacto social", 2)._p)
cur.p("Empresas medianas sin equipos de analítica acceden a un pronóstico con su incertidumbre y a una "
      "recomendación en lenguaje simple. Para evitar la dependencia de la recomendación, el dashboard muestra siempre "
      "el nivel de confianza y aclara que no es asesoría financiera {c:anexo1}.")
cur = Cursor(encabezado("Impacto ambiental", 2)._p)
cur.p(f"Reentrenar el ensamble de 24 h toma {f(t_total / 60, 1)} minutos en un portátil sin GPU (unas milésimas de "
      "kWh con un consumo de unos 25 W), despreciable frente a entrenar en centros de datos. Una mejor planeación de "
      "compra y venta puede favorecer un uso más eficiente del recurso hídrico en épocas secas.")
cur = Cursor(encabezado("Impacto legal y regulatorio", 2)._p)
cur.p("Solo se usa información pública de XM/SIMEM {c:creg2022}, no se recolectan datos personales {c:ley1581} y no se "
      "ejecutan transacciones. Ante intervenciones transitorias de la CREG en la formación del precio, cada modelo "
      "documenta su fecha de corte y el pipeline permite reentrenar rápidamente.")
cur = Cursor(encabezado("Impacto económico y financiero", 2)._p)
cur.p(f"El costo del proyecto es de ${f(tot_p, 0)} (Tabla 1), casi todo en horas de personal. En el backtesting de "
      "2026, las horas en que el motor recomienda vender tuvieron un precio medio 345,6 COP/kWh por encima del "
      "promedio, y las de comprar, 281,6 COP/kWh por debajo. No es una ganancia garantizada: no modela contratos, "
      "capacidad ni costos de transacción.")

# ================================================================================================
# 9. CONCLUSIONES
# ================================================================================================
cur = Cursor(encabezado("Conclusiones y recomendaciones", 1)._p)
cur.p("**OE1 (100 %): cumplido.** Se sincronizaron a resolución horaria las cinco variables del mercado y el ONI "
      "entre enero de 2019 y agosto de 2026 (66.576 horas, sin huecos). La caracterización identificó la componente "
      "de 24 h como dominante, con armónicos de 12 y 168 h, que definieron las 40 variables predictoras; el filtro "
      "de Savitzky-Golay detecta cambios de régimen 376 h antes que el promedio móvil.")
cur.p("**OE2 (90 %): cumplido en lo técnico, en consolidación.** Se compararon seis modelos de familias distintas "
      "y nueve descartados. El ensamble reduce el MAE en 27 % frente a la persistencia a 24 h "
      f"({f(ENS['MAE'])} frente a {f(PER['MAE'])} COP/kWh) y en 23 % a 72 h, con significancia estadística. Falta "
      "validar el ensamble completo en los orígenes históricos 1 a 5.")
cur.p("**OE3 (70 %): en ejecución.** El motor traduce las bandas en señales de comprar, vender o esperar para dos "
      "roles y dos horizontes, y el dashboard funciona. Faltan la biblioteca de imágenes y conectar el motor al "
      "ensamble de seis votantes (hoy usa las bandas de N-BEATSx).")
cur.p("**OE4 (35 %): en ejecución.** El backtesting eligió el método «banda» en los cuatro casos, con ventajas de "
      "280 a 346 COP/kWh. Falta la validación con 3 a 5 usuarios, planeada en la Fase 4.")
cur.p(f"**Objetivo general ({AV_G} %).** La plataforma procesa, pronostica y recomienda de principio a fin con "
      "resultados verificados estadísticamente; se cumplirá al cerrar OE3 y OE4.")
cur.p("**Recomendaciones:**", before=100)
cur.numerada([
    "Validar el dashboard con 3-5 usuarios mediante un cuestionario de comprensión (OE4).",
    "Conectar el motor al ensamble de seis votantes y ajustar su confianza según la hora de lanzamiento (el error "
    "sube de 41 a 52 COP/kWh entre el corte de las 00:00 y el de las 23:00).",
    "Usar umbrales relativos móviles de 90 días para el régimen hidrológico: en 2026 ninguna hora cae bajo el "
    "umbral absoluto de «embalse bajo».",
    "Evaluar el ensamble completo en los orígenes 1 a 5 y completar la biblioteca de imágenes (OE3).",
    "En una fase posterior, incorporar métricas de decisión basadas en la forma del día {c:maciejowska2025} e "
    "información de ofertas por planta, hoy fuera del alcance, que explica buena parte del error residual.",
])

# ================================================================================================
# 10. BIBLIOGRAFIA
# ================================================================================================
cur = Cursor(encabezado("Bibliografía (formato IEEE)", 1)._p)
for k, clave in enumerate(ORDEN, 1):
    cur.p(f"[{k}]\t" + BIB[clave], jc="left", ind_left=567, hanging=567, after=80, sz=10)

# ================================================================================================
# 11. ANEXOS: ubicacion de cada anexo
# ================================================================================================
cur = Cursor(buscar_parrafo("Anexo 8 — Fotografías y/o video del proyecto")._p)
cur.p("Los anexos 1, 3 y 5 están en el repositorio del proyecto (github.com/Ziskito/xm-spot-price-predictor): "
      "formulación en docs/, código fuente en notebooks/, src/, dashboard/ y scripts_experimento/, y datos y "
      "predicciones en data/. El anexo 2 corresponde a docs/tabla_riesgos_proyecto.xlsx y el anexo 7, a la bitácora "
      "del README y al historial de commits.", before=160)

for p_ in doc.paragraphs:
    if (p_.style.name in ("Heading 1", "Heading 2", "Heading 3") and p_.text.strip()) or p_.text.strip() == "Matriz de decisión":
        p_.paragraph_format.keep_with_next = True
doc.save(str(SALIDA))
print("Guardado:", SALIDA)
print("Referencias citadas:", len(ORDEN))
sin_citar = [k for k in BIB if k not in ORDEN]
print("Sin citar:", sin_citar)
