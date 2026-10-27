# -*- coding: utf-8 -*-
"""
Manual de usuario de Augur en PDF (docs/Manual_Usuario_augur.pdf), 14-oct-2026; rev. 1.4 (23-oct) con exportacion y reportes.
Las figuras export_jpg y reporte_dia_p1 (seccion 4.9) son archivos exportados desde la app y guardados en docs/manual/img/.

Formato de manual tecnico (documento MU-AUG-001): Times New Roman, blanco y negro, secciones numeradas, figuras y
tablas numeradas, globos numerados sobre las capturas, procedimientos por pasos, avisos NOTA / PRECAUCION / ADVERTENCIA,
tabla de sintoma-causa-accion y pie con "Pagina X de Y".
1) Levanta el dashboard en local (streamlit, puerto 8599) y toma capturas reales con Edge sin ventana (CDP).
2) Recorta las capturas y les dibuja los globos numerados.
3) Arma el HTML (docs/manual/manual.html) con las cifras vigentes del modelo (fuentes_pronostico.json y contrato de 24 h).
4) Lo imprime a PDF con Edge (CDP Page.printToPDF, carta, encabezado y pie) en dos pasadas: la segunda llena el indice
   con la pagina de cada titulo.
Uso: generar_manual_usuario.py [--sin-capturas]   (reusa las capturas ya tomadas en docs/manual/img/)
"""
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "data" / "processed" / "resultados"
DOCS = RAIZ / "docs"
MAN = DOCS / "manual"
IMG = MAN / "img"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PUERTO = 8599
PDF = DOCS / "Manual_Usuario_augur.pdf"

# rev. 1.5: el dia y la hora van fijos en la URL (?dia=&hora=), los mismos de las figuras del informe final, para que
# las capturas sean reproducibles; Metodologia, mas alta por el video que va arriba
DIA_CAPTURA = "dia=2026-10-23&hora=8"
CAPTURAS = {  # nombre: (parametros de la URL, alto de la ventana)
    "operador_generador": (f"vista=operador&tema=claro&{DIA_CAPTURA}", 3600),
    "operador_oscuro": (f"vista=operador&tema=oscuro&{DIA_CAPTURA}", 1100),
    "analista": ("vista=analista&tema=claro", 2000),
    "metodologia": ("vista=metodologia&tema=claro", 2400),
}
ESCALA = 1.5

# posicion de cada elemento en la pagina (px CSS): con ella se calculan los recortes y los globos, asi que siguen
# alineados aunque cambie el diseno (rev. 1.3, 19-oct-2026; antes eran coordenadas fijas)
JS_CAJAS = r"""(() => {
  const r = e => { if (!e) return null; const b = e.getBoundingClientRect(); return [b.left, b.top + window.scrollY, b.width, b.height]; };
  const q = s => r(document.querySelector(s));
  const lab = (tid, txt) => r([...document.querySelectorAll('[data-testid="' + tid + '"]')]
                               .find(e => (e.innerText || '').includes(txt)));
  const sub = document.querySelectorAll('.hero-d .sub-card');
  return {sidebar: q('[data-testid="stSidebar"]'), oscuro: q('.st-key-modo_oscuro'), vista: q('.st-key-vista_sel'),
    rol: lab('stSelectbox', 'Rol'), horizonte: lab('stSelectbox', 'Horizonte'), modelo: q('.modelo'),
    dia: q('.st-key-dia_operador'), hora: q('.st-key-hora_operador'), ant: q('.st-key-dia_ant'), sig: q('.st-key-dia_sig'),
    ult: q('.st-key-dia_ult'), hero: q('.hero-d'), medidor: q('.hd-svg'), conf: r(sub[0]), riesgo: r(sub[1]),
    graf_dia: q('.st-key-sec_dia_principal'), centrar: q('.st-key-centrar_dia'), ampliar: q('.st-key-ampliar_dia'),
    franja: q('.st-key-sec_franja'), calor: q('.st-key-sec_calor'), indicadores: q('.tarjetas'), semana: q('.st-key-sec_semana'),
    metodos: q('.st-key-sec_metodos'), backtest: q('.st-key-sec_backtest'), datos: lab('stExpander', 'Ver datos'),
    percentiles: lab('stSlider', 'Percentiles'), rango: lab('stSlider', 'Rango de fechas'), intradia: q('.st-key-filtro_intradia_w'),
    metodo: lab('stSelectbox', 'Método'), desempeno: q('.st-key-sec_desempeno'), metodologia: q('.st-key-sec_metodo'),
    exp_dia: q('.st-key-sec_dia_principal .exp-bar'), meta_dia: q('.meta'), rep_dia: q('.rep-dia'), rep_completo: q('.rep-completo')};
})()"""


def captura_cdp(url, salida, ancho, alto, espera=30):
    """Captura con Edge sin ventana controlado por el protocolo de depuracion (CDP): a diferencia de
    --screenshot, espera de verdad a que Streamlit termine de dibujar (llega por WebSocket)."""
    import urllib.request
    from websockets.sync.client import connect
    perfil = MAN / "_perfil_edge"
    proc = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", "--remote-debugging-port=9333", f"--user-data-dir={perfil}",
                             f"--window-size={ancho},{alto}", "--hide-scrollbars", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9333/json").read())
                break
            except Exception:
                time.sleep(0.5)
        ws_url = [t for t in tabs if t.get("type") == "page"][0]["webSocketDebuggerUrl"]
        with connect(ws_url, max_size=80_000_000) as ws:
            n = [0]

            def cmd(method, **params):
                n[0] += 1
                ws.send(json.dumps({"id": n[0], "method": method, "params": params}))
                while True:
                    m = json.loads(ws.recv())
                    if m.get("id") == n[0]:
                        return m.get("result", {})
            cmd("Emulation.setDeviceMetricsOverride", width=ancho, height=alto, deviceScaleFactor=1.5, mobile=False)
            cmd("Page.enable")
            cmd("Page.navigate", url=url)
            time.sleep(espera)
            r = cmd("Page.captureScreenshot", format="png", captureBeyondViewport=False)
            salida.write_bytes(base64.b64decode(r["data"]))
            # se mide despues de la captura y se repite si la pagina aun se estaba redibujando (elementos sin caja)
            for _ in range(5):
                cajas = cmd("Runtime.evaluate", expression=JS_CAJAS, returnByValue=True)["result"]["value"]
                if cajas.get("sidebar") and (cajas.get("hero") or cajas.get("metodos") or cajas.get("desempeno")):
                    break
                time.sleep(3)
            salida.with_name(salida.stem + "_cajas.json").write_text(json.dumps(cajas), encoding="utf-8")
    finally:
        proc.terminate()
        time.sleep(2)


def capturar():
    IMG.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(RAIZ / "dashboard" / "app.py"), "--server.port", str(PUERTO),
                             "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=RAIZ,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(15)
        for nombre, (q, alto) in CAPTURAS.items():
            out = IMG / f"{nombre}.png"
            captura_cdp(f"http://localhost:{PUERTO}/?{q}", out, 1440, alto)
            print("  captura", nombre, out.exists() and out.stat().st_size, flush=True)
    finally:
        proc.terminate()


def cajas(captura):
    """Cajas (x0, y0, x1, y1) en pixeles de la captura (escala 1,5) de cada elemento medido."""
    datos = json.loads((IMG / f"{captura}_cajas.json").read_text(encoding="utf-8"))
    return {k: tuple(round(v * ESCALA) for v in (b[0], b[1], b[0] + b[2], b[1] + b[3])) for k, b in datos.items() if b}


def recortes():
    o, a, m = cajas("operador_generador"), cajas("analista"), cajas("metodologia")
    ancho = 2160
    arriba = max(o["hero"][3], o["graf_dia"][3]) + 24
    caja = lambda b, pad=12: (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
    return {"op_arriba": ("operador_generador", (0, 0, ancho, arriba)),
            "op_tarjeta": ("operador_generador", caja(o["hero"])),
            "op_grafico_dia": ("operador_generador", caja(o["graf_dia"])),
            "op_franja": ("operador_generador", caja(o["franja"])),
            "op_calor": ("operador_generador", caja(o["calor"])),
            "op_indicadores": ("operador_generador", caja(o["indicadores"])),
            "op_semana": ("operador_generador", caja(o["semana"])),
            "op_oscuro": ("operador_oscuro", (0, 0, ancho, max(cajas("operador_oscuro")["hero"][3], cajas("operador_oscuro")["graf_dia"][3]) + 24)),
            "an_todo": ("analista", (0, 0, ancho, max(a["datos"][3], a["metodo"][3]) + 30)),
            "met_arriba": ("metodologia", (0, 0, ancho, m["desempeno"][3] + 24))}


def recortar():
    from PIL import Image
    for nombre, (fuente, caja) in recortes().items():
        ruta = IMG / f"{fuente}.png"
        if ruta.exists():
            Image.open(ruta).crop(caja).save(IMG / f"{nombre}.png")


def img(nombre, ancho="100%", pie=""):
    ruta = IMG / f"{nombre}.png"
    if not ruta.exists():
        return f'<div class="falta">[captura {nombre} no disponible]</div>'
    b64 = base64.b64encode(ruta.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" style="width:{ancho}">'
            + (f"<figcaption>{pie}</figcaption>" if pie else "") + "</figure>")


DOC = "MU-AUG-001"
REV = "1.5"
FECHA = "27 de octubre de 2026"

# globos numerados (coordenadas en pixeles del recorte, calculadas con las cajas de cada elemento). Los de la barra
# lateral (x < 200) van en un margen blanco a la izquierda de la figura, con linea guia hasta la etiqueta; los demas,
# en la esquina superior izquierda del elemento
def globos():
    o, a = cajas("operador_generador"), cajas("analista")
    lateral = lambda n, b: (n, 60, (b[1] + b[3]) // 2) if b else None
    esquina = lambda n, b, dx=-6, dy=-6: (n, max(b[0] + dx, 230), max(b[1] + dy, 26)) if b else None
    op = [lateral(1, o.get("oscuro")), lateral(2, o.get("vista")), lateral(3, o.get("rol")), lateral(4, o.get("horizonte")),
          lateral(5, o.get("modelo")), esquina(6, o.get("dia")), esquina(7, o.get("hora")), esquina(8, o.get("ant")),
          esquina(9, o.get("ult")), esquina(10, o.get("hero"), 18, 18), esquina(11, o.get("medidor"), 0, 10),
          esquina(12, o.get("conf")), esquina(13, o.get("riesgo")), esquina(14, o.get("graf_dia"), 18, 18),
          esquina(15, o.get("centrar"), -10, 4), esquina(16, o.get("ampliar"), -10, 4), esquina(17, o.get("exp_dia"), -10, 4)]
    an = [esquina(1, a.get("indicadores")), esquina(2, a.get("metodos")), esquina(3, a.get("backtest")),
          esquina(4, a.get("datos")), lateral(5, a.get("percentiles")), lateral(6, a.get("rango")), lateral(7, a.get("intradia")),
          lateral(8, a.get("metodo"))]
    return {"op_arriba": [g for g in op if g], "an_todo": [g for g in an if g]}


MARGEN = 90


def anotar():
    """Globos negros numerados con linea guia (estilo manual de equipo) sobre las capturas."""
    from PIL import Image, ImageDraw, ImageFont
    fuente = ImageFont.truetype("C:/Windows/Fonts/timesbd.ttf", 30)
    for nombre, lista in globos().items():
        ruta = IMG / f"{nombre}.png"
        if not ruta.exists():
            continue
        org = Image.open(ruta).convert("RGB")
        im = Image.new("RGB", (org.width + MARGEN, org.height), "white")
        im.paste(org, (MARGEN, 0))
        dib = ImageDraw.Draw(im)
        r = 22
        for num, x, y in lista:
            if x < 200:                       # barra lateral: globo en el margen y linea guia
                cx = MARGEN // 2 - 4
                dib.line((cx + r, y, MARGEN + max(x - 30, 8), y), fill="#0B1F3A", width=3)
            else:
                cx = MARGEN + x
            dib.ellipse((cx - r, y - r, cx + r, y + r), fill="#0B1F3A", outline="white", width=3)
            dib.text((cx, y + 1), str(num), fill="white", font=fuente, anchor="mm")
        im.save(IMG / f"{nombre}_globos.png")


def _mae72(conf):
    c = pd.read_csv(RES / conf["72h"]["archivo"])
    return {f"{a}-{b}": float(np.abs(c.q50 - c.real)[c.paso_horas.between(a, b)].mean()) for a, b in ((1, 24), (25, 48), (49, 72))}


def cifras():
    conf = json.loads((RES / "fuentes_pronostico.json").read_text(encoding="utf-8"))
    c24 = pd.read_csv(RES / conf["24h"]["archivo"], parse_dates=["fecha_hora"]).dropna(subset=["real"])
    y, q = c24["real"], c24["q50"]
    return {"modelo24": conf["24h"]["modelo"], "modelo72": conf["72h"]["modelo"],
            "cob24": conf["24h"].get("cobertura_medida_pct"), "cob72": conf["72h"].get("cobertura_medida_pct"),
            "mae24": float(np.abs(q - y).mean()), "mape24": float((np.abs(q - y) / y).mean() * 100),
            "ancho24": float((c24["q90"] - c24["q10"]).mean()),
            "mae72": _mae72(conf),
            "desde": c24["fecha_hora"].min(), "hasta": c24["fecha_hora"].max(), "generado": conf["24h"].get("generado")}


def fig(nombre, num, pie, ancho="100%"):
    ruta = IMG / f"{nombre}.png"
    if not ruta.exists():
        return f'<p class="falta">[Figura {num} no disponible]</p>'
    b64 = base64.b64encode(ruta.read_bytes()).decode()
    return (f'<div class="figura"><img src="data:image/png;base64,{b64}" style="width:{ancho}">'
            f'<div class="pie"><b>Figura {num}.</b> {pie}</div></div>')


def lado(figura, contenido, ancho_fig=58):
    """Figura a la izquierda y su tabla o explicacion a la derecha (formato horizontal)."""
    return (f'<div class="lado"><div class="lado-fig" style="width:{ancho_fig}%">{figura}</div>'
            f'<div class="lado-txt" style="width:{100 - ancho_fig}%">{contenido}</div></div>')


def aviso(tipo, texto):
    clase = {"Precaución": "precaucion", "Advertencia": "advertencia"}.get(tipo, "nota")
    return f'<table class="aviso {clase}"><tr><td class="tipo">{tipo}</td><td>{texto}</td></tr></table>'


# identidad de marca de Augur (dashboard/app.py, TEMAS["claro"]): tinta, acento ambar, gris del logo, IBM Plex Sans en la marca
TINTA, ACENTO, TENUE, GRIS = "#0B1F3A", "#D97706", "#94A3B8", "#5B6B82"
PAPEL = "#F3EBDB"          # color papel crema (pergamino)


def logo(tam=48, tinta=TINTA, acento=ACENTO, tenue=TENUE):
    """Logo de Augur: tres aves que suben y crecen (gris, tinta, ambar)."""
    def ave(x, y, k, color, grosor):
        return (f'<path d="M{x - 6 * k} {y} C{x - 4 * k} {y - 4 * k} {x - 1.5 * k} {y - 4 * k} {x} {y + k} '
                f'C{x + 1.5 * k} {y - 4 * k} {x + 4 * k} {y - 4 * k} {x + 6 * k} {y}" stroke="{color}" stroke-width="{grosor}"/>')
    return (f'<svg width="{tam}" height="{tam}" viewBox="0 0 48 48" fill="none" stroke-linecap="round" stroke-linejoin="round">'
            + ave(10, 37, .8, tenue, 2.8) + ave(23, 26, 1.05, tinta, 3.4) + ave(36, 13, 1.3, acento, 3.8) + "</svg>")


def marca(tam, tam_nombre, sub=""):
    return (f'<div class="marca">{logo(tam)}<div><div class="marca-nom" style="font-size:{tam_nombre}pt">Augur</div>'
            + (f'<div class="marca-sub">{sub}</div>' if sub else "") + "</div></div>")


CSS = """
@page { size: letter landscape; margin: 16mm 18mm 16mm 18mm; }
html { background: #F3EBDB; }  /* mismo tono que PAPEL */
body { font-family: 'Times New Roman', Times, serif; font-size: 11.5pt; line-height: 1.35; color: #111; margin: 0; text-align: justify; }
h1 { font-size: 14pt; font-weight: bold; text-transform: uppercase; color: #0B1F3A; margin: 0 0 10pt; padding-bottom: 3pt;
     border-bottom: 1.5pt solid #D97706; break-after: avoid; margin-top: 14pt; }
h1.salto, h2.salto { page-break-before: always; margin-top: 0; }
h2 { font-size: 12pt; font-weight: bold; color: #0B1F3A; margin: 13pt 0 5pt; break-after: avoid; page-break-after: avoid; }
h3 { font-size: 11.5pt; font-weight: bold; font-style: italic; color: #0B1F3A; margin: 10pt 0 4pt; break-after: avoid; page-break-after: avoid; }
p { margin: 0 0 6pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 4pt; font-size: 10.5pt; page-break-inside: avoid; text-align: left; }
th, td { border: 0.75pt solid #94A3B8; padding: 3pt 5pt; vertical-align: top; background: transparent; }
th { background: #D97706; color: #fff; font-weight: bold; border-color: #D97706; }
.tit-tabla { font-size: 10.5pt; margin: 8pt 0 2pt; text-align: center; color: #0B1F3A; break-after: avoid; page-break-after: avoid; }
.figura { text-align: center; margin: 8pt 0 10pt; page-break-inside: avoid; }
.figura img { border: 0.75pt solid #94A3B8; }
.pie { font-size: 10.5pt; margin-top: 3pt; text-align: center; }
.pie b, .tit-tabla b { color: #0B1F3A; }
table.aviso { margin: 8pt 0; border: 1.2pt solid #94A3B8; }
table.aviso td { border: none; padding: 5pt 7pt; }
table.aviso td.tipo { width: 28mm; font-weight: bold; text-transform: uppercase; text-align: center; vertical-align: middle; color: #fff; background: #5B6B82; }
table.aviso.precaucion { border-color: #D97706; } table.aviso.precaucion td.tipo { background: #D97706; }
table.aviso.advertencia { border-color: #B45309; } table.aviso.advertencia td.tipo { background: #B45309; }
ul { margin: 2pt 0 6pt; padding-left: 7mm; } li { margin-bottom: 2pt; }
code { font-family: 'Courier New', monospace; font-size: 10pt; }
.marca { display: flex; align-items: center; gap: 10px; }
.marca-nom { font-family: 'IBM Plex Sans', 'Segoe UI', Arial, sans-serif; font-weight: 700; color: #0B1F3A; letter-spacing: .3px; line-height: 1; }
.marca-sub { font-family: 'IBM Plex Sans', 'Segoe UI', Arial, sans-serif; font-size: 8.5pt; font-weight: 600; letter-spacing: 1.6pt;
             text-transform: uppercase; color: #5B6B82; margin-top: 4pt; }
.portada { text-align: left; }
.portada .cab { border-top: 3pt solid #0B1F3A; border-bottom: 1pt solid #D97706; padding: 6pt 0; font-size: 11pt; letter-spacing: 1pt;
                text-transform: uppercase; color: #0B1F3A; }
.portada .marca { margin-top: 0; }
.portada .tipo { font-size: 30pt; font-weight: bold; color: #0B1F3A; margin: 14mm 0 3mm; letter-spacing: 1pt; }
.portada .barra { width: 36mm; height: 3pt; background: #D97706; margin: 0 0 5mm; }
.portada .desc { font-size: 13pt; margin-bottom: 0; color: #2B3D57; }
.portada table td { font-size: 11pt; }
.portada table td:first-child { width: 45mm; font-weight: bold; color: #0B1F3A; background: rgba(217,119,6,.12); }
.portada .nota-pie { margin-top: 10mm; font-size: 10pt; border-top: 1pt solid #D97706; padding-top: 5pt; color: #2B3D57; }
.indice table, .indice td { border: none; font-size: 11.5pt; padding: 1.5pt 0; }
.indice td.n { width: 12mm; } .indice td.p { width: 12mm; text-align: right; }
.indice tr.sub td { font-size: 11pt; } .indice tr.sub td.t { padding-left: 8mm; }
.indice tr.cap td { font-weight: bold; color: #0B1F3A; padding-top: 5pt; }
.falta { border: 0.75pt dashed #000; padding: 14pt; text-align: center; }
.lado { display: flex; gap: 7mm; align-items: flex-start; page-break-inside: avoid; margin: 4pt 0 8pt; }
.lado-fig .figura { margin-top: 0; } .lado-txt .tit-tabla { margin-top: 0; text-align: left; }
table.compacta { width: auto; font-size: 8.8pt; margin: 2pt 0 4pt; }
table.compacta th, table.compacta td { padding: 1.5pt 5pt; white-space: nowrap; }
.nota-lado { font-size: 9.5pt; text-align: left; }
.dos { display: flex; gap: 12mm; } .dos > div { flex: 1; }
.indice { column-count: 2; column-gap: 14mm; }
.portada { display: flex; gap: 16mm; align-items: center; height: 170mm; }
.portada .izq { width: 55%; } .portada .der { width: 45%; }
.anexo table { font-size: 9.8pt; margin: 2pt 0; } .anexo th, .anexo td { padding: 2pt 5pt; }
.anexo table.aviso { margin: 5pt 0; } .anexo table.aviso td { padding: 3pt 6pt; font-size: 9.8pt; }
"""

INDICE = [
    ("", "CONTROL DE REVISIONES", 1),
    ("1", "INTRODUCCIÓN", 1),
    ("2", "ACCESO", 1), ("2.1", "Acceso web", 2), ("2.2", "Parámetros de la dirección", 2),
    ("3", "DESCRIPCIÓN DE LA INTERFAZ", 1), ("3.1", "Vista Operador", 2), ("3.2", "Vista Analista", 2), ("3.3", "Vista Metodología", 2), ("3.4", "Tema oscuro", 2),
    ("4", "FUNCIONES DE LA VISTA OPERADOR", 1), ("4.1", "Rol y horizonte", 2), ("4.2", "Selección de día y hora", 2),
    ("4.3", "Recomendación horaria", 2), ("4.4", "Pronóstico del día", 2), ("4.5", "Franja horaria", 2),
    ("4.6", "Mapa de calor día-hora", 2), ("4.7", "Indicadores", 2), ("4.8", "Semana de referencia", 2),
    ("4.9", "Exportar gráficas y reportes", 2),
    ("5", "FUNCIONES DE LA VISTA ANALISTA", 1), ("5.1", "Parámetros del umbral", 2), ("5.2", "Rango de evaluación", 2),
    ("5.3", "Filtro intradía", 2), ("5.4", "Método de umbral", 2), ("5.5", "Resultados y comparación de métodos", 2),
    ("5.6", "Datos filtrados", 2),
    ("6", "REFERENCIA DE SEÑALES Y SÍMBOLOS", 1),
    ("7", "SOLUCIÓN DE PROBLEMAS", 1),
    ("", "ANEXO A. MODELO Y DATOS", 1),
]


def indice_html(paginas):
    filas = []
    for num, tit, nivel in INDICE:
        pag = paginas.get(tit, "")
        clase = "cap" if nivel == 1 else "sub"
        filas.append(f'<tr class="{clase}"><td class="n">{num}</td><td class="t">{tit}</td><td class="p">{pag}</td></tr>')
    return '<table style="page-break-inside:auto">' + "".join(filas) + "</table>"


def html(c, paginas):
    f = lambda t: pd.Timestamp(t).strftime("%d/%m/%Y")
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>{DOC} Manual de usuario de Augur</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@600;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>

<div class="portada">
 <div class="izq">
  {marca(96, 44, "Motor de decisión · precio de bolsa de energía")}
  <div class="tipo">MANUAL DE USUARIO</div>
  <div class="barra"></div>
  <div class="desc">Tablero de pronóstico del precio de bolsa y motor de decisión horaria</div>
 </div>
 <div class="der">
  <table>
    <tr><td>Documento</td><td>{DOC}</td></tr>
    <tr><td>Revisión</td><td>{REV}</td></tr>
    <tr><td>Fecha de emisión</td><td>{FECHA}</td></tr>
    <tr><td>Versión del pronóstico</td><td>{c['generado']}</td></tr>
    <tr><td>Dirección web</td><td>augurpredictor.streamlit.app</td></tr>
    <tr><td>Elaborado por</td><td>Juan Barceló Barraza y Rafael Dede</td></tr>
    <tr><td>Clasificación</td><td>Uso académico</td></tr>
  </table>
  <div class="nota-pie">Prototipo académico. Las recomendaciones del sistema no constituyen asesoría financiera.</div>
 </div>
</div>

<h1 class="salto">Control de revisiones</h1>
<div class="tit-tabla"><b>Tabla 1.</b> Historial de revisiones</div>
<table><tr><th style="width:14mm">Rev.</th><th style="width:28mm">Fecha</th><th>Descripción</th><th style="width:44mm">Elaboró</th></tr>
<tr><td>1.0</td><td>14/10/2026</td><td>Emisión inicial. Pronóstico de 24 h del escenario realista y detector de eventos realista.</td><td>J. Barceló, R. Dede</td></tr>
<tr><td>1.1</td><td>14/10/2026</td><td>Actualización diaria automática y pronóstico del día siguiente; versión del modelo en la barra lateral; fechas editables en el mapa de calor; la hora se conserva al cambiar de vista.</td><td>J. Barceló, R. Dede</td></tr>
<tr><td>1.2</td><td>15/10/2026</td><td>Interfaz rediseñada: tarjeta de recomendación con medidor, gráfica del día con botón Ampliar y variables que se ocultan o muestran desde la leyenda, franja horaria con precio y acción, tema oscuro de alto contraste y diseño adaptable a pantallas angostas.</td><td>J. Barceló, R. Dede</td></tr>
<tr><td>1.3</td><td>19/10/2026</td><td>Nombre oficial con mayúscula (Augur). Vista Metodología. Tema oscuro Pizarra. Controles de día reordenados con «Ir al último». Gráficas con acercamiento por Ctrl + rueda o pellizco, desplazamiento con dos dedos, botón Centrar, clic en una hora de la gráfica del día y carga de días vecinos. Números y fechas en formato colombiano. Leyenda propia del mapa de calor. Actualización diaria también al mediodía.</td><td>J. Barceló, R. Dede</td></tr>
<tr><td>1.4</td><td>23/10/2026</td><td>Picos y desplomes se presentan como <i>posibilidad</i>, no como certeza; en el horizonte de 72 h se estima la posibilidad de pico de los días 2 y 3. Exportación de cada gráfica y de la franja horaria en JPG y PDF; reporte del día y reporte completo en PDF. El mapa de calor abre con 3 días. Horas de la franja en formato 00:00. Unidades (COP/kWh) en umbrales, franja y tablas, y notas más visibles.</td><td>J. Barceló, R. Dede</td></tr>
<tr><td>1.5</td><td>27/10/2026</td><td>Video explicativo al inicio de la vista Metodología y validación de todo 2026. En la semana de referencia y en el backtest, las horas de acción de los días sin precio real se marcan sobre el pronóstico. Colores de ▼ y ▲ según el rol. Una etiqueta por día en el mapa de calor. Tabla de datos con coma decimal. Enlaces con día y hora (<code>dia</code>, <code>hora</code>).</td><td>J. Barceló, R. Dede</td></tr></table>
<h2 class="salto">Contenido</h2>
<div class="indice">{indice_html(paginas)}</div>

<h1 class="salto">1. Introducción</h1>
<p>Augur es un tablero web que presenta el pronóstico horario del precio de bolsa de energía en Colombia, con su banda de
incertidumbre del 80 %, y una recomendación de acción por hora generada por el motor de decisión. Este manual describe
las funciones del tablero y su uso.</p>
<div class="tit-tabla"><b>Tabla 2.</b> Roles y señales</div>
<table><tr><th>Rol</th><th>Acción favorable</th><th>Acción desfavorable</th><th>Neutra</th></tr>
<tr><td>Generador</td><td>Vender</td><td>Retener</td><td>Esperar</td></tr>
<tr><td>Comercializador</td><td>Comprar</td><td>Evitar compra</td><td>Esperar</td></tr></table>
<p>El tablero tiene tres vistas: <b>Operador</b>, para la decisión del día; <b>Analista</b>, para revisar y ajustar las reglas del
motor sobre datos históricos; y <b>Metodología</b>, que muestra el desempeño en vivo del pronóstico y explica cómo funciona.</p>

<h1>2. Acceso</h1>
<h2>2.1 Acceso web</h2>
<p>El tablero está publicado en Streamlit Community Cloud en <code>https://augurpredictor.streamlit.app</code> y no requiere
autenticación. La primera carga del día puede demorar hasta un minuto mientras se activa el servidor.</p>
<h2>2.2 Parámetros de la dirección</h2>
<div class="tit-tabla" style="text-align:left"><b>Tabla 3.</b> Parámetros de la dirección web</div>
<table class="compacta"><tr><th>Parámetro</th><th>Efecto</th></tr>
<tr><td><code>vista=operador | analista | metodologia</code></td><td>Vista inicial</td></tr>
<tr><td><code>tema=claro | oscuro</code></td><td>Tema inicial</td></tr>
<tr><td><code>dia=AAAA-MM-DD</code>, <code>hora=0 a 23</code></td><td>Día y hora iniciales de la vista Operador</td></tr>
<tr><td><code>?vista=analista&amp;tema=oscuro</code></td><td>Ejemplo combinado</td></tr></table>

<h1 class="salto">3. Descripción de la interfaz</h1>
<h2>3.1 Vista Operador</h2>
{lado(fig("op_arriba_globos", 1, "Vista Operador (rol generador, horizonte 24 h)"), '''<div class="tit-tabla"><b>Tabla 4.</b> Elementos de la vista Operador</div>
<table class="compacta"><tr><th>N.º</th><th>Elemento</th><th>Sección</th></tr>
<tr><td>1</td><td>Modo oscuro</td><td>—</td></tr>
<tr><td>2</td><td>Vista</td><td>—</td></tr>
<tr><td>3</td><td>Rol</td><td>4.1</td></tr>
<tr><td>4</td><td>Horizonte de pronóstico</td><td>4.1</td></tr>
<tr><td>5</td><td>Modelo activo</td><td>4.1</td></tr>
<tr><td>6</td><td>Día operativo</td><td>4.2</td></tr>
<tr><td>7</td><td>Hora del día</td><td>4.2</td></tr>
<tr><td>8</td><td>Anterior / Siguiente</td><td>4.2</td></tr>
<tr><td>9</td><td>Ir al último</td><td>4.2</td></tr>
<tr><td>10</td><td>Recomendación</td><td>4.3</td></tr>
<tr><td>11</td><td>Medidor de precio</td><td>4.3</td></tr>
<tr><td>12</td><td>Confianza</td><td>4.3</td></tr>
<tr><td>13</td><td>Posibilidad de salto</td><td>4.3</td></tr>
<tr><td>14</td><td>Pronóstico del día</td><td>4.4</td></tr>
<tr><td>15</td><td>Centrar</td><td>4.4</td></tr>
<tr><td>16</td><td>Ampliar</td><td>4.4</td></tr>
<tr><td>17</td><td>Exportar (JPG / PDF)</td><td>4.9</td></tr></table>
<p class="nota-lado">Debajo se ubican el botón del reporte del día (4.9), la franja horaria (4.5), el mapa de calor (4.6), los indicadores (4.7), la semana de referencia (4.8) y el botón del reporte completo (4.9).</p>''', 69)}
<h2 style="page-break-before:always">3.2 Vista Analista</h2>
{lado(fig("an_todo_globos", 2, "Vista Analista"), '''<div class="tit-tabla"><b>Tabla 5.</b> Elementos de la vista Analista</div>
<table class="compacta"><tr><th>N.º</th><th>Elemento</th><th>Sección</th></tr>
<tr><td>1</td><td>Tarjetas de resultado</td><td>5.5</td></tr>
<tr><td>2</td><td>Comparación de métodos</td><td>5.5</td></tr>
<tr><td>3</td><td>Gráfica de backtest</td><td>5.5</td></tr>
<tr><td>4</td><td>Ver datos filtrados</td><td>5.6</td></tr>
<tr><td>5</td><td>Percentiles del umbral</td><td>5.1</td></tr>
<tr><td>6</td><td>Rango de fechas</td><td>5.2</td></tr>
<tr><td>7</td><td>Filtro intradía</td><td>5.3</td></tr>
<tr><td>8</td><td>Método de umbral</td><td>5.4</td></tr></table>''', 56)}
<h2 style="page-break-before:always">3.3 Vista Metodología</h2>
{lado(fig("met_arriba", 3, "Vista Metodología: video y desempeño en vivo"), '''<p>Presenta, sin controles adicionales, cómo le ha ido al
pronóstico de 24 h y cómo funciona. Arriba, <b>Cómo funciona Augur (4 min)</b> reproduce el video explicativo de la plataforma.
<b>Desempeño en vivo</b> compara el pronóstico con el precio real que XM ya publicó (últimos
30 días o todo 2026): error del pronóstico y del predespacho ideal de XM, mejora frente a ese predespacho, porcentaje de horas en que
el precio real cayó dentro de la banda y acierto de las recomendaciones, con la gráfica del error de cada día.</p>
<p>Debajo, <b>Cómo funciona</b> resume el flujo (datos públicos, corte a las 00:00, pronóstico, incertidumbre y motor), cómo decide
el motor, la validación del informe final (con una línea de todo 2026 que se actualiza cada día) y las limitaciones, y <b>Fórmulas y validación</b> presenta las ecuaciones del pronóstico, la
banda, el motor y las métricas.</p>''', 44)}
<h2 style="page-break-before:always">3.4 Tema oscuro</h2>
{lado(fig("op_oscuro", 4, "Vista Operador con el tema oscuro"), '''<p>El interruptor <b>Modo oscuro</b> (1) cambia a un tema de fondo
azul pizarra y alto contraste, pensado para salas de control, proyección y uso nocturno. Las señales conservan sus colores (verde, naranja y gris)
y todas las funciones son las mismas que en el tema claro. El tema también se puede fijar desde la dirección web (Tabla 3).</p>
<p>En pantallas de menos de 1200 px de ancho (tabletas o ventanas pequeñas), la tarjeta de recomendación y la gráfica del día se
muestran una debajo de la otra.</p>''', 66)}

<h1 class="salto">4. Funciones de la vista Operador</h1>
<h2>4.1 Rol y horizonte</h2>
<p><b>Rol</b> (3) define cómo se traduce el pronóstico en acciones (Tabla 2). <b>Horizonte de pronóstico</b> (4) selecciona el
pronóstico de 24 h, emitido a las 00:00 para la 01:00 a 00:00 siguientes, o el de 72 h. El recuadro <b>Modelo activo</b> (5) indica la
versión del pronóstico y la cobertura medida de su banda; debajo se indica la fecha de la última actualización, hasta dónde llega el precio real
publicado por XM y hasta dónde llega el pronóstico. Con el horizonte de 72 h, el primer día trae la posibilidad de pico y de desplome, y el
segundo y el tercero solo la de pico, que es menos confiable que la del primer día.</p>
<h2>4.2 Selección de día y hora</h2>
<p><b>Día operativo</b> (6) solo admite días con las 24 horas pronosticadas. <b>Hora del día</b> (7) fija la hora de la
recomendación; también se selecciona con un clic en la franja horaria (4.5) o en la gráfica del día (4.4). Los botones <b>Anterior</b> y
<b>Siguiente</b> (8) recorren los días sin abrir el calendario, e <b>Ir al último</b> (9), resaltado en ámbar, salta al último día con
pronóstico. El día y la hora seleccionados se conservan al cambiar de vista. Los días más recientes aún no tienen precio
real: XM lo publica con unos tres días de rezago.</p>
<h2>4.3 Recomendación horaria</h2>
{lado(fig("op_tarjeta", 5, "Tarjeta de recomendación"), '''<p>La tarjeta (10) muestra la acción para el día, la hora y el rol seleccionados, con una insignia que la califica como favorable,
desfavorable o neutral. El medidor (11) ubica el precio esperado (aguja y valor en COP/kWh) frente a los tramos barato, normal y caro
del método activo; a su lado se indican los umbrales de cada tramo y la acción que le corresponde según el rol. La tarjeta toma el
alto de la gráfica del día. <b>Confianza</b> (12) clasifica el ancho relativo de la banda frente a las horas
anteriores en alta, media o baja. <b>Posibilidad de salto en esta hora</b> (13) presenta la probabilidad estimada de desplome (▼) y de pico (▲)
de la hora; se resalta desde 30 % y se listan las horas del día que superan ese umbral. Es una probabilidad, no una certeza: el evento puede no ocurrir.</p>'''
+ aviso("Nota", "El método de umbral de la vista Operador es el más estable del backtest para el rol y el horizonte elegidos (criterio maximin entre mitades del periodo). Si ningún método cumple la condición de estabilidad, la tarjeta muestra el aviso ⚠."), 48)}
<h2>4.4 Pronóstico del día</h2>
{lado(fig("op_grafico_dia", 6, "Pronóstico del día"), '''<p>La gráfica (14) presenta la mediana del pronóstico, la banda del 80 %, los umbrales barato y caro y la hora seleccionada (línea
vertical), de 00:00 a 23:00, en COP/kWh. En días con precio publicado incluye el precio real. Los marcadores ▼ y ▲ señalan las horas con
posibilidad de desplome o de pico. Un clic en un punto del día elegido selecciona esa hora.</p>
<p>Ctrl + rueda del ratón o el pellizco del panel táctil acercan y alejan; arrastrar o deslizar dos dedos a los lados mueve la gráfica en
el tiempo, y aparecen los días vecinos (si se llega al borde de lo cargado, se cargan más días). <b>Centrar</b> (15) devuelve la gráfica al
día elegido. Las <b>variables</b> de la leyenda funcionan como botones: un clic oculta o muestra la variable y un doble clic deja solo
esa. <b>Ampliar</b> (16) abre la gráfica en una ventana grande.</p>''', 40)}
<h2>4.5 Franja horaria</h2>
{fig("op_franja", 7, "Franja horaria")}
<p>Resume las 24 horas del día en dos filas. Cada casilla indica la hora (00:00 a 23:00), el precio esperado en COP/kWh y la acción abreviada
(Tabla 9), con el color de la señal: verde, acción favorable; naranja, acción desfavorable; gris, esperar. Los símbolos ▼ y ▲ sobre una
casilla señalan posibilidad de desplome o de pico. Un clic en una casilla selecciona esa hora para la recomendación; la casilla
seleccionada se resalta con un borde.</p>
<h2>4.6 Mapa de calor día-hora</h2>
{lado(fig("op_calor", 8, "Mapa de calor día-hora"), '''<p>Muestra la señal o el precio esperado de varios días a la vez; abre con los 3 días que terminan en el día elegido. <b>Días</b> y <b>Horas</b> acotan el periodo y las horas mostradas
(extremos juntos para un solo día u hora). Las fechas de los extremos de <b>Días</b> se escriben directamente o se eligen en el
calendario, y la barra se ajusta; <b>Colorear por</b> alterna entre <b>Señal del motor</b> y <b>Precio esperado</b>; con la señal
del motor, la leyenda bajo el mapa indica el color de cada acción. El recuadro marca el día y la hora seleccionados.</p>''', 60)}
<h2>4.7 Indicadores</h2>
{lado(fig("op_indicadores", 9, "Indicadores"), '''<div class="tit-tabla"><b>Tabla 6.</b> Indicadores de la vista Operador</div>
<table><tr><th style="width:38mm">Indicador</th><th>Contenido</th></tr>
<tr><td>Horas para actuar hoy</td><td>Horas del día con acción favorable; barras de los últimos 14 días.</td></tr>
<tr><td>Ventaja histórica de la regla</td><td>Diferencia entre el precio medio de las horas de acción y el promedio del periodo en el backtest de 2026 (COP/kWh); serie mensual.</td></tr>
<tr><td>Con qué frecuencia actúa</td><td>Porcentaje de horas con acción favorable; serie mensual. Rango útil: 10 % a 40 %. La barra inferior reparte el periodo entre acción favorable, desfavorable y esperar.</td></tr></table>''', 55)}
<h2>4.8 Semana de referencia</h2>
{lado(fig("op_semana", 10, "Semana de referencia"), '''<p>Compara el precio real con el pronóstico en siete días alrededor del día seleccionado (si este es de los últimos, la semana se
corre hacia atrás), con las horas de acción marcadas: sobre el precio real si XM ya lo publicó y, si no, sobre el pronóstico. Admite los mismos gestos de la gráfica del día (4.4), la barra de herramientas
de la gráfica al pasar el cursor y la consulta de valores; las variables se ocultan o muestran desde la leyenda.</p>''', 62)}
<h2 style="page-break-before:always">4.9 Exportar gráficas y reportes</h2>
{lado(fig("export_jpg", 11, "Gráfica del día exportada en JPG"), '''<p>Sobre cada gráfica y sobre la franja horaria aparece la barra
<b>Exportar: JPG · PDF</b> (17). El archivo no es una foto de la pantalla: la gráfica se vuelve a dibujar en fondo blanco, con los
colores del tema claro (también desde el tema oscuro), la leyenda pequeña arriba a la derecha y el eje horizontal con las 24 horas
cuando se ve un solo día. Arriba lleva el logo de Augur, el título y la fecha de exportación; abajo, la unidad, el periodo, la confianza y
la versión del modelo. El JPG sirve para presentaciones y el PDF es una hoja horizontal lista para imprimir.</p>
<p>Si la gráfica se acercó o se movió, se exporta el encuadre que se ve en ese momento. Mientras se genera el archivo, el botón dice
«Generando…».</p>''', 60)}
{lado(fig("reporte_dia_p1", 12, "Primera página del reporte del día"), '''<p><b>Reporte del día (PDF)</b>, bajo la gráfica del día,
arma un documento de tres páginas con el día, la hora y el rol elegidos. <b>Descargar reporte completo (PDF)</b>, al final de la vista
Operador, agrega el mapa de calor (con el rango y el color elegidos), los indicadores y la semana.</p>
<div class="tit-tabla" style="text-align:left"><b>Tabla 7.</b> Contenido de los reportes en PDF</div>
<table class="compacta"><tr><th>Sección</th><th>Del día</th><th>Completo</th></tr>
<tr><td>Recomendación y cifras clave</td><td>Sí</td><td>Sí</td></tr>
<tr><td>Pronóstico del día (figura)</td><td>Sí</td><td>Sí</td></tr>
<tr><td>Confianza y posibilidad de saltos</td><td>Sí</td><td>Sí</td></tr>
<tr><td>Franja y tabla de las 24 horas</td><td>Sí</td><td>Sí</td></tr>
<tr><td>Mapa de calor</td><td>No</td><td>Sí</td></tr>
<tr><td>Indicadores del periodo</td><td>No</td><td>Sí</td></tr>
<tr><td>Semana de referencia</td><td>No</td><td>Sí</td></tr>
<tr><td>Notas metodológicas</td><td>Sí</td><td>Sí</td></tr></table>
<p>Cada reporte lleva código de documento, fecha de emisión, figuras y tablas numeradas y número de página.</p>''', 42)}
{aviso("Nota", "Los archivos se generan en el navegador y se guardan en la carpeta de descargas. El primero de la sesión puede tardar unos segundos más, mientras se carga el componente de PDF.")}

<h1 class="salto">5. Funciones de la vista Analista</h1>
<p>Los controles de esta vista se ubican en la barra lateral y solo afectan a la vista Analista; la recomendación de la vista Operador
no cambia.</p>
{lado(fig("an_controles_globos", 13, "Controles de la vista Analista"), '''<h2 style="margin-top:0">5.1 Parámetros del umbral</h2>
<p><b>Percentiles del umbral</b> (5) define los percentiles bajo y alto que separan los precios baratos y caros (25 y 75 por defecto).</p>
<h2>5.2 Rango de evaluación</h2>
<p><b>Rango de fechas</b> (6) delimita el periodo sobre el que se calculan los resultados y se sugiere el método.</p>
<h2>5.3 Filtro intradía</h2>
<p><b>Filtro intradía</b> (7) conserva la acción favorable solo en la mitad del día que el pronóstico ubica más cara (generador) o más
barata (comercializador). Activo por defecto.</p>
<h2>5.4 Método de umbral</h2>
<div class="tit-tabla"><b>Tabla 8.</b> Métodos de umbral</div>
<table><tr><th style="width:22mm">Método</th><th>Umbrales</th></tr>
<tr><td>fijo</td><td>Percentiles del precio histórico 2019-2025.</td></tr>
<tr><td>rodante</td><td>Percentiles del pronóstico de los 30 días anteriores.</td></tr>
<tr><td>banda</td><td>Fijo; espera cuando la banda es más ancha que el percentil 75 de las horas anteriores.</td></tr>
<tr><td>híbrido</td><td>Rodante, con el filtro de banda ancha.</td></tr></table>
<p>El selector (8) inicia en el método sugerido para el rango elegido.</p>''', 30)}
<h2>5.5 Resultados y comparación de métodos</h2>
<p>Las tarjetas (1) muestran la ventaja económica, la frecuencia de acción y las horas de acción del método seleccionado. La tabla (2)
compara los cuatro métodos e indica si su frecuencia está en el rango de 10 % a 40 %. La gráfica (3) muestra el precio real, el
pronóstico, la banda y las señales del método en el rango elegido. Abre en los últimos 30 días; los botones <b>7 días</b>,
<b>30 días</b>, <b>90 días</b> y <b>Todo</b> cambian el encuadre sin cambiar las cifras. Admite los gestos de la gráfica del día (4.4) y sus
variables se ocultan o muestran desde la leyenda.</p>
<h2>5.6 Datos filtrados</h2>
<p><b>Ver datos filtrados</b> (4) despliega la tabla horaria del rango: precio real, cuantiles del pronóstico, señal y probabilidades de
evento, con las cifras en COP/kWh con un decimal y las probabilidades en porcentaje. El selector <b>Mostrar</b> deja todas las horas o solo las de una señal (acción favorable, desfavorable o esperar). La tabla
admite búsqueda, ordenamiento y descarga en CSV.</p>

<h1>6. Referencia de señales y símbolos</h1>
<div class="tit-tabla"><b>Tabla 9.</b> Señales y símbolos</div>
<table><tr><th style="width:44mm">Elemento</th><th>Significado</th></tr>
<tr><td>Verde</td><td>Vender (generador) o comprar (comercializador).</td></tr>
<tr><td>Naranja</td><td>Retener (generador) o evitar compra en bolsa (comercializador).</td></tr>
<tr><td>Gris</td><td>Esperar.</td></tr>
<tr><td>VEN / COM / RET / EVI / ESP</td><td>Vender, comprar, retener, evitar compra y esperar (abreviaturas de la franja horaria).</td></tr>
<tr><td>Favorable / desfavorable / neutral</td><td>Insignia de la tarjeta de recomendación.</td></tr>
<tr><td>▼</td><td>Posibilidad de desplome de 30 % o más (precio ≤ 0,7 × mediana del día). Naranja para el generador; verde para el comercializador.</td></tr>
<tr><td>▲</td><td>Posibilidad de pico de 30 % o más (precio ≥ 1,3 × mediana del día). Verde para el generador; naranja para el comercializador.</td></tr>
<tr><td>⚠</td><td>Ningún método fue estable en el backtest.</td></tr>
<tr><td>Confianza alta / media / baja</td><td>Ancho relativo de la banda frente a las horas anteriores.</td></tr>
<tr><td>Línea vertical (gráficas)</td><td>Hora seleccionada.</td></tr>
<tr><td>Ampliar</td><td>Abre la gráfica del día en una ventana grande.</td></tr>
<tr><td>Centrar</td><td>Devuelve la gráfica del día al día elegido.</td></tr>
<tr><td>Ctrl + rueda / pellizco</td><td>Acerca o aleja la gráfica; doble clic restablece el encuadre.</td></tr>
<tr><td>Exportar: JPG · PDF</td><td>Descarga la gráfica o la franja (4.9).</td></tr>
<tr><td>Reporte del día / completo</td><td>Descarga el reporte en PDF (4.9).</td></tr></table>

<h1>7. Solución de problemas</h1>
<div class="tit-tabla"><b>Tabla 10.</b> Síntoma, causa y acción</div>
<table><tr><th style="width:46mm">Síntoma</th><th style="width:56mm">Causa probable</th><th>Acción</th></tr>
<tr><td>Página en blanco o en carga.</td><td>Servidor en activación.</td><td>Esperar y recargar.</td></tr>
<tr><td>«No pude cargar la fuente de pronóstico».</td><td>Falta el archivo indicado en <code>fuentes_pronostico.json</code>.</td><td>Restaurar el repositorio completo.</td></tr>
<tr><td>Día no seleccionable.</td><td>Día sin las 24 horas pronosticadas.</td><td>Seleccionar otro día.</td></tr>
<tr><td>Sin precio real en los últimos días.</td><td>XM publica el precio con unos 3 días de rezago.</td><td>Ninguna; se completa en las actualizaciones siguientes.</td></tr>
<tr><td>El pronóstico no llega a mañana.</td><td>XM aún no publica el predespacho del día siguiente o la actualización no ha corrido.</td><td>Consultar después del mediodía.</td></tr>
<tr><td>Sin marcadores ▼ en el segundo o tercer día (72 h).</td><td>A 72 h solo se estima la posibilidad de pico de esos días.</td><td>Ninguna; es lo esperado.</td></tr>
<tr><td>La descarga no inicia.</td><td>El navegador bloquea las descargas del sitio o el archivo aún se genera.</td><td>Esperar a que el botón deje de decir «Generando…» y permitir las descargas.</td></tr>
<tr><td>Una variable no aparece en la gráfica.</td><td>Se ocultó con un clic en la leyenda.</td><td>Clic de nuevo en su nombre o doble clic sobre la leyenda.</td></tr>
<tr><td>Aviso ⚠ en la recomendación.</td><td>Ningún método estable para el rol y horizonte.</td><td>Revisar la comparación de métodos (5.5).</td></tr></table>

<h1 class="salto">Anexo A. Modelo y datos</h1>
<div class="anexo">
<div class="tit-tabla"><b>Tabla A.1.</b> Pronóstico vigente</div>
<table><tr><th style="width:56mm">Característica</th><th>Valor</th></tr>
<tr><td>Modelo (24 h)</td><td>{c['modelo24']}</td></tr>
<tr><td>Error absoluto medio (24 h)</td><td>{c['mae24']:.1f} COP/kWh (MAPE {c['mape24']:.1f} %), {f(c['desde'])} a {f(c['hasta'])}</td></tr>
<tr><td>Cobertura de la banda (24 h)</td><td>{c['cob24']} % (objetivo 80 %); ancho medio {c['ancho24']:.0f} COP/kWh</td></tr>
<tr><td>Modelo (72 h)</td><td>{c['modelo72']}</td></tr>
<tr><td>Error absoluto medio (72 h)</td><td>Pasos 1-24: {c['mae72']['1-24']:.1f}; pasos 25-48: {c['mae72']['25-48']:.1f}; pasos 49-72: {c['mae72']['49-72']:.1f} COP/kWh</td></tr>
<tr><td>Cobertura de la banda (72 h)</td><td>{c['cob72']} % (objetivo 80 %)</td></tr></table>
<div class="tit-tabla"><b>Tabla A.2.</b> Datos disponibles al emitir el pronóstico del día D (00:00)</div>
<table><tr><th>Dato</th><th style="width:38mm">Fuente</th><th>Disponible hasta</th></tr>
<tr><td>Precio de bolsa, demanda, generación</td><td>XM</td><td>23:00 de D-4</td></tr>
<tr><td>Embalses y aportes hídricos</td><td>XM</td><td>D-2 (diario)</td></tr>
<tr><td>ONI</td><td>NOAA</td><td>Rezago de unos 2 meses</td></tr>
<tr><td>Disponibilidad declarada</td><td>XM</td><td>D-1, 08:10</td></tr>
<tr><td>Predespacho ideal</td><td>XM</td><td>D-1, 10:30</td></tr>
<tr><td>Despacho programado</td><td>XM</td><td>D-1, 14:30</td></tr></table>
{aviso("Nota", "El pronóstico se actualiza todos los días hacia las 12:00, las 15:40 y las 18:40 (hora de Colombia): se descargan los datos nuevos de XM y NOAA, se reentrenan los modelos del mes con todo lo publicado y se pronostica hasta la ventana del día siguiente. Lo ya pronosticado para un día iniciado no se modifica; el precio real se completa a medida que XM lo publica.")}
{aviso("Advertencia", "El cierre de ofertas es a las 08:00 de D-1 (Resolución CREG 025 de 1995) y el predespacho se publica después. El pronóstico de 24 h aplica a decisiones posteriores a la publicación del predespacho, no a la oferta de los generadores.")}
{aviso("Nota", "El horizonte de 72 h supone el precio conocido hasta el corte (escenario optimista); su desempeño con fuentes públicas sería menor.") if "optimista" in c["modelo72"] else aviso("Nota", "En el horizonte de 72 h no hay predespacho para los días D+1 y D+2: los pasos 25 a 72 parten del predespacho del día D y su error crece con el horizonte (Tabla A.1).")}
</div>
</body></html>"""


PIE = ('<div style="width:100%;text-align:center;font-family:Times New Roman,serif;font-size:9pt;color:#5B4A2F">'
       '<span class="pageNumber"></span></div>')
CABEZA = '<div></div>'


def imprimir(html_ruta, pdf_ruta):
    """Imprime el HTML a PDF con Edge sin ventana (CDP Page.printToPDF), con encabezado y pie de pagina."""
    import urllib.request
    from websockets.sync.client import connect
    perfil = MAN / "_perfil_edge"
    proc = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", "--remote-debugging-port=9334", f"--user-data-dir={perfil}",
                             "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9334/json").read())
                break
            except Exception:
                time.sleep(0.5)
        ws_url = [t for t in tabs if t.get("type") == "page"][0]["webSocketDebuggerUrl"]
        with connect(ws_url, max_size=200_000_000) as ws:
            n = [0]

            def cmd(method, **params):
                n[0] += 1
                ws.send(json.dumps({"id": n[0], "method": method, "params": params}))
                while True:
                    m = json.loads(ws.recv())
                    if m.get("id") == n[0]:
                        return m.get("result", {})
            cmd("Page.enable")
            cmd("Page.navigate", url=html_ruta.as_uri())
            time.sleep(4)
            r = cmd("Page.printToPDF", displayHeaderFooter=True, headerTemplate=CABEZA, footerTemplate=PIE,
                    printBackground=True, preferCSSPageSize=True)
            pdf_ruta.write_bytes(base64.b64decode(r["data"]))
            # hoja completa color papel (Chrome no pinta los margenes de pagina): se pone debajo de cada pagina
            hoja = MAN / "_papel.html"
            hoja.write_text('<!doctype html><style>@page{size:letter landscape;margin:0}html,body{margin:0;height:100%;background:'
                            + PAPEL + '}</style><body></body>', encoding="utf-8")
            cmd("Page.navigate", url=hoja.as_uri())
            time.sleep(1.5)
            r = cmd("Page.printToPDF", printBackground=True, preferCSSPageSize=True)
            (MAN / "_papel.pdf").write_bytes(base64.b64decode(r["data"]))
    finally:
        proc.terminate()
        time.sleep(2)
    papel(pdf_ruta)


def papel(pdf_ruta):
    from pypdf import PdfReader, PdfWriter
    fondo = PdfReader(str(MAN / "_papel.pdf")).pages[0]
    lector, escritor = PdfReader(str(pdf_ruta)), PdfWriter()
    for pag in lector.pages:
        pag.merge_page(fondo, over=False)          # el papel queda debajo del contenido
        escritor.add_page(pag)
    with open(pdf_ruta, "wb") as fh:
        escritor.write(fh)


def paginas_de(pdf_ruta):
    """Pagina en que empieza cada titulo del indice (segunda pasada)."""
    import re
    from pypdf import PdfReader
    textos = [re.sub(r"\s+", " ", p.extract_text() or "").upper() for p in PdfReader(str(pdf_ruta)).pages]
    paginas = {}
    # el cuerpo empieza en la Introduccion (el indice puede ocupar varias paginas y repite los titulos)
    inicio = next(i for i, t in enumerate(textos) if "1. INTRODUCCIÓN" in t)
    for num, tit, nivel in INDICE:
        clave = re.sub(r"\s+", " ", (f"{num}. {tit}" if nivel == 1 and num else f"{num} {tit}" if num else tit)).upper()
        for i, t in enumerate(textos[inicio:], start=inicio + 1):
            if clave in t:
                paginas[tit] = i
                break
        if tit == "CONTROL DE REVISIONES":
            paginas[tit] = 2
    return paginas


if __name__ == "__main__":
    MAN.mkdir(parents=True, exist_ok=True)
    if "--sin-capturas" not in sys.argv:
        capturar()
    recortar()
    anotar()
    from PIL import Image
    a = cajas("analista")
    Image.open(IMG / "an_todo_globos.png").crop((0, a["percentiles"][1] - 80, a["sidebar"][2] + MARGEN,
                                                a["metodo"][3] + 30)).save(IMG / "an_controles_globos.png")
    c = cifras()
    ruta = MAN / "manual.html"
    ruta.write_text(html(c, {}), encoding="utf-8")
    imprimir(ruta, PDF)
    pags = paginas_de(PDF)
    faltan = [t for _, t, _ in INDICE if t not in pags]
    ruta.write_text(html(c, pags), encoding="utf-8")
    imprimir(ruta, PDF)
    print("PDF:", PDF, PDF.stat().st_size, "| titulos sin pagina en el indice:", faltan)
