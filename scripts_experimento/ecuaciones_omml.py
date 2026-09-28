# -*- coding: utf-8 -*-
"""Convierte LaTeX en ecuaciones nativas de Word (OMML) con la hoja MML2OMML.XSL que trae Office."""
from pathlib import Path

from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from latex2mathml.converter import convert
from lxml import etree

XSL = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
_A_OMML = etree.XSLT(etree.parse(str(XSL)))


def omml(latex):
    """elemento m:oMath listo para insertar en un w:p."""
    mathml = etree.fromstring(convert(latex, display="block").encode("utf-8"))
    return parse_xml(etree.tostring(_A_OMML(mathml).getroot()))


def _p(jc):
    p = OxmlElement("w:p")
    ppr = OxmlElement("w:pPr")
    sp = OxmlElement("w:spacing"); sp.set(qn("w:before"), "0"); sp.set(qn("w:after"), "0"); ppr.append(sp)
    j = OxmlElement("w:jc"); j.set(qn("w:val"), jc); ppr.append(j)
    p.append(ppr)
    return p


def tabla_ecuacion(latex, numero, ancho_twips=9400, fuente="Arial"):
    """tabla sin bordes de 3 columnas: la ecuación en modo presentación al centro y su número a la derecha.
    Una lista de LaTeX pone varias expresiones en la misma línea, separadas por comas; una lista que contiene
    listas reparte las expresiones en varias líneas."""
    anchos = [ancho_twips // 10, ancho_twips * 8 // 10, ancho_twips // 10]
    tbl = OxmlElement("w:tbl")
    tblpr = OxmlElement("w:tblPr")
    tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), str(ancho_twips)); tw.set(qn("w:type"), "dxa"); tblpr.append(tw)
    jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "center"); tblpr.append(jc)
    bd = OxmlElement("w:tblBorders")
    for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{lado}"); e.set(qn("w:val"), "nil"); bd.append(e)
    tblpr.append(bd)
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblpr.append(lay)
    tbl.append(tblpr)
    grid = OxmlElement("w:tblGrid")
    for a in anchos:
        g = OxmlElement("w:gridCol"); g.set(qn("w:w"), str(a)); grid.append(g)
    tbl.append(grid)
    tr = OxmlElement("w:tr")
    trpr = OxmlElement("w:trPr"); trpr.append(OxmlElement("w:cantSplit")); tr.append(trpr)
    if isinstance(latex, str):
        lineas = [[latex]]
    elif all(isinstance(x, str) for x in latex):
        lineas = [list(latex)]
    else:
        lineas = [[x] if isinstance(x, str) else list(x) for x in latex]
    for ci, a in enumerate(anchos):
        tc = OxmlElement("w:tc")
        tcpr = OxmlElement("w:tcPr")
        w = OxmlElement("w:tcW"); w.set(qn("w:w"), str(a)); w.set(qn("w:type"), "dxa"); tcpr.append(w)
        va = OxmlElement("w:vAlign"); va.set(qn("w:val"), "center"); tcpr.append(va)
        tc.append(tcpr)
        if ci == 1:
            for partes in lineas[:-1]:
                p = _p("center")
                para = OxmlElement("m:oMathPara")
                for k, parte in enumerate(partes):
                    om = omml(parte)
                    if k:
                        r = OxmlElement("m:r"); t = OxmlElement("m:t"); t.set(qn("xml:space"), "preserve")
                        t.text = ",  "; r.append(t)
                        para[-1].append(r)
                        for hijo in list(om):
                            para[-1].append(hijo)
                    else:
                        para.append(om)
                p.append(para)
                tc.append(p)
            p = _p("center")
            para = OxmlElement("m:oMathPara")
            for partes in lineas[-1:]:
                for k, parte in enumerate(partes):
                    om = omml(parte)
                    if k:
                        r = OxmlElement("m:r"); t = OxmlElement("m:t"); t.set(qn("xml:space"), "preserve")
                        t.text = ",\u2003\u2003"; r.append(t)
                        para[-1].append(r)
                        for hijo in list(om):
                            para[-1].append(hijo)
                    else:
                        para.append(om)
            p.append(para)
        elif ci == 2:
            p = _p("right")
            r = OxmlElement("w:r")
            rpr = OxmlElement("w:rPr")
            fn = OxmlElement("w:rFonts")
            for at in ("ascii", "hAnsi", "cs"):
                fn.set(qn(f"w:{at}"), fuente)
            rpr.append(fn); r.append(rpr)
            t = OxmlElement("w:t"); t.text = f"({numero})"; r.append(t); p.append(r)
        else:
            p = _p("left")
        tc.append(p)
        tr.append(tc)
    tbl.append(tr)
    return tbl
