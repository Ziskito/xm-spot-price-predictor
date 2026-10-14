# -*- coding: utf-8 -*-
"""
Manual de usuario de augur en PDF (docs/Manual_Usuario_augur.pdf), 14-oct-2026.

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

CAPTURAS = {  # nombre: (parametros de la URL, alto de la ventana)
    "operador_generador": ("vista=operador&tema=claro", 2600),
    "operador_oscuro": ("vista=operador&tema=oscuro", 1100),
    "analista": ("vista=analista&tema=claro", 2000),
}


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


# recortes en pixeles de la captura (escala 1,5): (captura, (x0, y0, x1, y1))
RECORTES = {
    "sb": ("operador_generador", (0, 0, 450, 1290)),
    "op_arriba": ("operador_generador", (0, 0, 2160, 1034)),
    "op_tarjeta": ("operador_generador", (560, 400, 1370, 930)),
    "op_grafico_dia": ("operador_generador", (1400, 405, 2050, 1030)),
    "op_franja": ("operador_generador", (546, 1053, 2160, 1297)),
    "op_calor": ("operador_generador", (546, 1297, 2160, 2305)),
    "op_indicadores": ("operador_generador", (546, 2330, 2160, 2720)),
    "op_semana": ("operador_generador", (546, 2730, 2160, 3360)),
    "an_todo": ("analista", (0, 0, 2160, 1905)),
}


def recortar():
    from PIL import Image
    for nombre, (fuente, caja) in RECORTES.items():
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
REV = "1.0"
FECHA = "14 de octubre de 2026"

# globos numerados (coordenadas en pixeles del recorte). Los de la barra lateral (x < 200) van en un margen blanco a la
# izquierda de la figura, con linea guia hasta la etiqueta; los demas, en espacios libres junto al elemento
GLOBOS = {
    "op_arriba": [(1, 51, 224), (2, 47, 332), (3, 47, 618), (4, 47, 745), (5, 55, 985), (6, 522, 355), (7, 1200, 296),
                  (8, 1880, 300), (9, 522, 468), (10, 522, 669), (11, 522, 760), (12, 522, 854), (13, 1388, 470)],
    "an_todo": [(1, 556, 307), (2, 556, 588), (3, 556, 967), (4, 556, 1794), (5, 33, 1497), (6, 33, 1624), (7, 33, 1750),
                (8, 33, 1822)],
}
MARGEN = 90


def anotar():
    """Globos negros numerados con linea guia (estilo manual de equipo) sobre las capturas."""
    from PIL import Image, ImageDraw, ImageFont
    fuente = ImageFont.truetype("C:/Windows/Fonts/timesbd.ttf", 30)
    for nombre, globos in GLOBOS.items():
        ruta = IMG / f"{nombre}.png"
        if not ruta.exists():
            continue
        org = Image.open(ruta).convert("RGB")
        im = Image.new("RGB", (org.width + MARGEN, org.height), "white")
        im.paste(org, (MARGEN, 0))
        dib = ImageDraw.Draw(im)
        r = 22
        for num, x, y in globos:
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
    c24 = pd.read_csv(RES / conf["24h"]["archivo"], parse_dates=["fecha_hora"])
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


def aviso(tipo, texto):
    clase = {"Precaución": "precaucion", "Advertencia": "advertencia"}.get(tipo, "nota")
    return f'<table class="aviso {clase}"><tr><td class="tipo">{tipo}</td><td>{texto}</td></tr></table>'


# identidad de marca de augur (dashboard/app.py, TEMAS["claro"]): tinta, acento ambar, gris del logo, IBM Plex Sans en la marca
TINTA, ACENTO, TENUE, GRIS = "#0B1F3A", "#D97706", "#94A3B8", "#5B6B82"


def logo(tam=48, tinta=TINTA, acento=ACENTO, tenue=TENUE):
    """Logo de augur: tres aves que suben y crecen (gris, tinta, ambar)."""
    def ave(x, y, k, color, grosor):
        return (f'<path d="M{x - 6 * k} {y} C{x - 4 * k} {y - 4 * k} {x - 1.5 * k} {y - 4 * k} {x} {y + k} '
                f'C{x + 1.5 * k} {y - 4 * k} {x + 4 * k} {y - 4 * k} {x + 6 * k} {y}" stroke="{color}" stroke-width="{grosor}"/>')
    return (f'<svg width="{tam}" height="{tam}" viewBox="0 0 48 48" fill="none" stroke-linecap="round" stroke-linejoin="round">'
            + ave(10, 37, .8, tenue, 2.8) + ave(23, 26, 1.05, tinta, 3.4) + ave(36, 13, 1.3, acento, 3.8) + "</svg>")


def marca(tam, tam_nombre, sub=""):
    return (f'<div class="marca">{logo(tam)}<div><div class="marca-nom" style="font-size:{tam_nombre}pt">augur</div>'
            + (f'<div class="marca-sub">{sub}</div>' if sub else "") + "</div></div>")


CSS = """
@page { size: letter; margin: 24mm 20mm 22mm 25mm; }
body { font-family: 'Times New Roman', Times, serif; font-size: 11.5pt; line-height: 1.35; color: #111; margin: 0; text-align: justify; }
h1 { font-size: 14pt; font-weight: bold; text-transform: uppercase; color: #0B1F3A; margin: 0 0 10pt; padding-bottom: 3pt;
     border-bottom: 1.5pt solid #D97706; page-break-before: always; break-after: avoid; }
h2 { font-size: 12pt; font-weight: bold; color: #0B1F3A; margin: 13pt 0 5pt; break-after: avoid; page-break-after: avoid; }
h3 { font-size: 11.5pt; font-weight: bold; font-style: italic; color: #0B1F3A; margin: 10pt 0 4pt; break-after: avoid; page-break-after: avoid; }
p { margin: 0 0 6pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 4pt; font-size: 10.5pt; page-break-inside: avoid; text-align: left; }
th, td { border: 0.75pt solid #8A96A8; padding: 3pt 5pt; vertical-align: top; }
th { background: #0B1F3A; color: #fff; font-weight: bold; border-color: #0B1F3A; }
.tit-tabla { font-size: 10.5pt; margin: 8pt 0 2pt; text-align: center; color: #0B1F3A; break-after: avoid; page-break-after: avoid; }
.figura { text-align: center; margin: 8pt 0 10pt; page-break-inside: avoid; }
.figura img { border: 0.75pt solid #8A96A8; }
.pie { font-size: 10.5pt; margin-top: 3pt; text-align: center; }
.pie b, .tit-tabla b { color: #0B1F3A; }
table.aviso { margin: 8pt 0; border: 1.2pt solid #0B1F3A; }
table.aviso td { border: none; padding: 5pt 7pt; }
table.aviso td.tipo { width: 28mm; font-weight: bold; text-transform: uppercase; text-align: center; vertical-align: middle; color: #fff; background: #0B1F3A; }
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
.portada .marca { margin-top: 38mm; }
.portada .tipo { font-size: 30pt; font-weight: bold; color: #0B1F3A; margin: 14mm 0 3mm; letter-spacing: 1pt; }
.portada .barra { width: 36mm; height: 3pt; background: #D97706; margin: 0 0 5mm; }
.portada .desc { font-size: 13pt; margin-bottom: 24mm; color: #2B3D57; }
.portada table td { font-size: 11pt; }
.portada table td:first-child { width: 45mm; font-weight: bold; color: #0B1F3A; background: #EEF1F6; }
.portada .nota-pie { margin-top: 22mm; font-size: 10pt; border-top: 1pt solid #D97706; padding-top: 5pt; color: #2B3D57; }
.indice table, .indice td { border: none; font-size: 11.5pt; padding: 1.5pt 0; }
.indice td.n { width: 12mm; } .indice td.p { width: 12mm; text-align: right; }
.indice tr.sub td { font-size: 11pt; } .indice tr.sub td.t { padding-left: 8mm; }
.indice tr.cap td { font-weight: bold; color: #0B1F3A; padding-top: 5pt; }
.falta { border: 0.75pt dashed #000; padding: 14pt; text-align: center; }
"""

INDICE = [
    ("", "CONTROL DE REVISIONES", 1),
    ("1", "INTRODUCCIÓN", 1),
    ("2", "ACCESO", 1), ("2.1", "Acceso web", 2), ("2.2", "Ejecución local", 2), ("2.3", "Parámetros de la dirección", 2),
    ("3", "DESCRIPCIÓN DE LA INTERFAZ", 1), ("3.1", "Vista Operador", 2), ("3.2", "Vista Analista", 2),
    ("4", "FUNCIONES DE LA VISTA OPERADOR", 1), ("4.1", "Rol y horizonte", 2), ("4.2", "Selección de día y hora", 2),
    ("4.3", "Recomendación horaria", 2), ("4.4", "Pronóstico del día", 2), ("4.5", "Franja horaria", 2),
    ("4.6", "Mapa de calor día-hora", 2), ("4.7", "Indicadores", 2), ("4.8", "Semana de referencia", 2),
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
    return '<table>' + "".join(filas) + "</table>"


def html(c, paginas):
    f = lambda t: pd.Timestamp(t).strftime("%d/%m/%Y")
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>{DOC} Manual de usuario de augur</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@600;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>

<div class="portada">
  <div class="cab">Proyecto de grado · Ingeniería Electrónica</div>
  {marca(96, 44, "Motor de decisión · precio de bolsa de energía")}
  <div class="tipo">MANUAL DE USUARIO</div>
  <div class="barra"></div>
  <div class="desc">Tablero de pronóstico del precio de bolsa y motor de decisión horaria</div>
  <table>
    <tr><td>Documento</td><td>{DOC}</td></tr>
    <tr><td>Revisión</td><td>{REV}</td></tr>
    <tr><td>Fecha de emisión</td><td>{FECHA}</td></tr>
    <tr><td>Versión del pronóstico</td><td>{c['generado']}</td></tr>
    <tr><td>Elaborado por</td><td>Juan Barceló Barraza y Rafael Dede</td></tr>
    <tr><td>Clasificación</td><td>Uso académico</td></tr>
  </table>
  <div class="nota-pie">Prototipo académico. Las recomendaciones del sistema no constituyen asesoría financiera.</div>
</div>

<h1>Control de revisiones</h1>
<div class="tit-tabla"><b>Tabla 1.</b> Historial de revisiones</div>
<table><tr><th style="width:14mm">Rev.</th><th style="width:28mm">Fecha</th><th>Descripción</th><th style="width:44mm">Elaboró</th></tr>
<tr><td>1.0</td><td>14/10/2026</td><td>Emisión inicial. Pronóstico de 24 h del escenario realista y detector de eventos realista.</td><td>J. Barceló, R. Dede</td></tr></table>
<h2 style="margin-top:18pt">Contenido</h2>
<div class="indice">{indice_html(paginas)}</div>

<h1>1. Introducción</h1>
<p>augur es un tablero web que presenta el pronóstico horario del precio de bolsa de energía en Colombia, con su banda de
incertidumbre del 80 %, y una recomendación de acción por hora generada por el motor de decisión. Este manual describe
las funciones del tablero y su uso.</p>
<div class="tit-tabla"><b>Tabla 2.</b> Roles y señales</div>
<table><tr><th>Rol</th><th>Acción favorable</th><th>Acción desfavorable</th><th>Neutra</th></tr>
<tr><td>Generador</td><td>Vender</td><td>Retener</td><td>Esperar</td></tr>
<tr><td>Comercializador</td><td>Comprar</td><td>Evitar compra</td><td>Esperar</td></tr></table>
<p>El tablero tiene dos vistas: <b>Operador</b>, para la decisión del día, y <b>Analista</b>, para revisar y ajustar las reglas del
motor sobre datos históricos.</p>

<h1>2. Acceso</h1>
<h2>2.1 Acceso web</h2>
<p>El tablero está publicado en Streamlit Community Cloud y no requiere autenticación. La primera carga del día puede demorar hasta un
minuto mientras se activa el servidor.</p>
<h2>2.2 Ejecución local</h2>
<p>Requiere Python 3.12. Desde la raíz del repositorio <code>github.com/Ziskito/xm-spot-price-predictor</code>:</p>
<p><code>pip install -r dashboard/requirements.txt</code><br><code>streamlit run dashboard/app.py</code></p>
<p>El tablero queda disponible en <code>http://localhost:8501</code>.</p>
<h2>2.3 Parámetros de la dirección</h2>
<div class="tit-tabla"><b>Tabla 3.</b> Parámetros de la dirección web</div>
<table><tr><th style="width:56mm">Parámetro</th><th>Efecto</th></tr>
<tr><td><code>vista=operador | analista</code></td><td>Vista inicial.</td></tr>
<tr><td><code>tema=claro | oscuro</code></td><td>Tema inicial.</td></tr></table>
<p>Ejemplo: <code>?vista=analista&amp;tema=oscuro</code>.</p>

<h1>3. Descripción de la interfaz</h1>
<h2>3.1 Vista Operador</h2>
{fig("op_arriba_globos", 1, "Vista Operador (rol generador, horizonte 24 h)")}
<div class="tit-tabla"><b>Tabla 4.</b> Elementos de la vista Operador</div>
<table><tr><th style="width:10mm">N.º</th><th style="width:50mm">Elemento</th><th>Sección</th></tr>
<tr><td>1</td><td>Modo oscuro</td><td>Cambia el tema de la interfaz.</td></tr>
<tr><td>2</td><td>Vista</td><td>Operador o Analista.</td></tr>
<tr><td>3</td><td>Rol</td><td>4.1</td></tr>
<tr><td>4</td><td>Horizonte de pronóstico</td><td>4.1</td></tr>
<tr><td>5</td><td>Modelo activo</td><td>4.1</td></tr>
<tr><td>6</td><td>Día operativo</td><td>4.2</td></tr>
<tr><td>7</td><td>Anterior / Siguiente / Último</td><td>4.2</td></tr>
<tr><td>8</td><td>Hora del día</td><td>4.2</td></tr>
<tr><td>9</td><td>Recomendación</td><td>4.3</td></tr>
<tr><td>10</td><td>Escala de precio</td><td>4.3</td></tr>
<tr><td>11</td><td>Confianza</td><td>4.3</td></tr>
<tr><td>12</td><td>Riesgo de salto</td><td>4.3</td></tr>
<tr><td>13</td><td>Pronóstico del día</td><td>4.4</td></tr></table>
<p>Debajo se ubican la franja horaria (4.5), el mapa de calor (4.6), los indicadores (4.7) y la semana de referencia (4.8).</p>
<h2>3.2 Vista Analista</h2>
{fig("an_todo_globos", 2, "Vista Analista")}
<div class="tit-tabla"><b>Tabla 5.</b> Elementos de la vista Analista</div>
<table><tr><th style="width:10mm">N.º</th><th style="width:50mm">Elemento</th><th>Sección</th></tr>
<tr><td>1</td><td>Tarjetas de resultado</td><td>5.5</td></tr>
<tr><td>2</td><td>Comparación de métodos</td><td>5.5</td></tr>
<tr><td>3</td><td>Gráfica de backtest</td><td>5.5</td></tr>
<tr><td>4</td><td>Ver datos filtrados</td><td>5.6</td></tr>
<tr><td>5</td><td>Percentiles del umbral</td><td>5.1</td></tr>
<tr><td>6</td><td>Rango de fechas</td><td>5.2</td></tr>
<tr><td>7</td><td>Filtro intradía</td><td>5.3</td></tr>
<tr><td>8</td><td>Método de umbral</td><td>5.4</td></tr></table>

<h1>4. Funciones de la vista Operador</h1>
<h2>4.1 Rol y horizonte</h2>
<p><b>Rol</b> (3) define cómo se traduce el pronóstico en acciones (Tabla 2). <b>Horizonte de pronóstico</b> (4) selecciona el
pronóstico de 24 h, emitido a las 00:00 para la 01:00 a 00:00 siguientes, o el de 72 h. El recuadro <b>Modelo activo</b> (5) identifica el
modelo en uso y la cobertura medida de su banda. Las probabilidades de picos y desplomes solo están disponibles con el horizonte de 24 h.</p>
<h2>4.2 Selección de día y hora</h2>
<p><b>Día operativo</b> (6) solo admite días con las 24 horas pronosticadas. Los botones <b>Anterior</b>, <b>Siguiente</b> y <b>Último</b> (7)
recorren los días sin abrir el calendario. <b>Hora del día</b> (8) fija la hora de la recomendación; también se selecciona con un clic en la
franja horaria (4.5). El día seleccionado se conserva al cambiar de vista.</p>
<h2>4.3 Recomendación horaria</h2>
<p>La tarjeta (9) muestra la acción para el día, la hora y el rol seleccionados. La escala de precio (10) ubica el precio esperado frente a los
umbrales barato y caro del método activo. <b>Confianza</b> (11) clasifica el ancho relativo de la banda frente a las horas anteriores en
alta, media o baja. <b>Riesgo de salto</b> (12) presenta la probabilidad de desplome (▼) y de pico (▲) de la hora; se resalta desde 30 % y se
listan las horas del día que superan ese umbral.</p>
{aviso("Nota", "El método de umbral de la vista Operador es el más estable del backtest para el rol y el horizonte elegidos (criterio maximin entre mitades del periodo). Si ningún método cumple la condición de estabilidad, la tarjeta muestra el aviso ⚠.")}
<h2>4.4 Pronóstico del día</h2>
<p>La gráfica (13) presenta la mediana del pronóstico, la banda del 80 %, los umbrales barato y caro y la hora seleccionada. En días con
precio publicado incluye el precio real. Los marcadores ▼ y ▲ señalan las horas con riesgo de salto.</p>
<h2>4.5 Franja horaria</h2>
{fig("op_franja", 3, "Franja horaria")}
<p>Resume la señal de las 24 horas del día: verde, acción favorable; naranja, acción desfavorable; gris, esperar. Un clic en una hora la
selecciona como hora de la recomendación.</p>
<h2>4.6 Mapa de calor día-hora</h2>
{fig("op_calor", 4, "Mapa de calor día-hora")}
<p>Muestra la señal o el precio esperado de varios días a la vez. <b>Días</b> y <b>Horas</b> acotan el periodo y las horas mostradas
(extremos juntos para un solo día u hora); <b>Colorear por</b> alterna entre <b>Señal del motor</b> y <b>Precio esperado</b>. El
recuadro marca el día y la hora seleccionados.</p>
<h2>4.7 Indicadores</h2>
{fig("op_indicadores", 5, "Indicadores")}
<div class="tit-tabla"><b>Tabla 6.</b> Indicadores de la vista Operador</div>
<table><tr><th style="width:50mm">Indicador</th><th>Contenido</th></tr>
<tr><td>Horas para actuar hoy</td><td>Horas del día con acción favorable; barras de los últimos 14 días.</td></tr>
<tr><td>Ventaja histórica de la regla</td><td>Diferencia entre el precio medio de las horas de acción y el promedio del periodo en el backtest de 2026 (COP/kWh); serie mensual.</td></tr>
<tr><td>Con qué frecuencia actúa</td><td>Porcentaje de horas con acción favorable; serie mensual. Rango útil: 10 % a 40 %.</td></tr></table>
<h2>4.8 Semana de referencia</h2>
{fig("op_semana", 6, "Semana de referencia")}
<p>Compara el precio real con el pronóstico desde tres días antes hasta cuatro días después del día seleccionado, con las horas de
acción marcadas. Permite ampliar, desplazar y consultar valores sobre la gráfica.</p>

<h1>5. Funciones de la vista Analista</h1>
<p>Los controles de esta vista se ubican en la barra lateral y solo afectan a la vista Analista; la recomendación de la vista Operador
no cambia.</p>
<h2>5.1 Parámetros del umbral</h2>
<p><b>Percentiles del umbral</b> (5) define los percentiles bajo y alto que separan los precios baratos y caros (25 y 75 por defecto).</p>
<h2>5.2 Rango de evaluación</h2>
<p><b>Rango de fechas</b> (6) delimita el periodo sobre el que se calculan los resultados y se sugiere el método.</p>
<h2>5.3 Filtro intradía</h2>
<p><b>Filtro intradía</b> (7) conserva la acción favorable solo en la mitad del día que el pronóstico ubica más cara (generador) o más
barata (comercializador). Activo por defecto.</p>
<h2>5.4 Método de umbral</h2>
<div class="tit-tabla"><b>Tabla 7.</b> Métodos de umbral</div>
<table><tr><th style="width:26mm">Método</th><th>Umbrales</th></tr>
<tr><td>fijo</td><td>Percentiles del precio histórico 2019-2025.</td></tr>
<tr><td>rodante</td><td>Percentiles del pronóstico de los 30 días anteriores.</td></tr>
<tr><td>banda</td><td>Fijo; espera cuando la banda es más ancha que el percentil 75 de las horas anteriores.</td></tr>
<tr><td>híbrido</td><td>Rodante, con el filtro de banda ancha.</td></tr></table>
<p>El selector (8) inicia en el método sugerido para el rango elegido.</p>
<h2>5.5 Resultados y comparación de métodos</h2>
<p>Las tarjetas (1) muestran la ventaja económica, la frecuencia de acción y las horas de acción del método seleccionado. La tabla (2)
compara los cuatro métodos e indica si su frecuencia está en el rango de 10 % a 40 %. La gráfica (3) muestra el precio real, el
pronóstico, la banda y las señales del método en el rango elegido.</p>
<h2>5.6 Datos filtrados</h2>
<p><b>Ver datos filtrados</b> (4) despliega la tabla horaria del rango: precio real, cuantiles del pronóstico, señal y probabilidades de
evento. La tabla admite ordenamiento y descarga en CSV.</p>

<h1>6. Referencia de señales y símbolos</h1>
<div class="tit-tabla"><b>Tabla 8.</b> Señales y símbolos</div>
<table><tr><th style="width:44mm">Elemento</th><th>Significado</th></tr>
<tr><td>Verde</td><td>Vender (generador) o comprar (comercializador).</td></tr>
<tr><td>Naranja</td><td>Retener (generador) o evitar compra en bolsa (comercializador).</td></tr>
<tr><td>Gris</td><td>Esperar.</td></tr>
<tr><td>▼</td><td>Probabilidad de desplome ≥ 30 % (precio ≤ 0,7 × mediana del día).</td></tr>
<tr><td>▲</td><td>Probabilidad de pico ≥ 30 % (precio ≥ 1,3 × mediana del día).</td></tr>
<tr><td>⚠</td><td>Ningún método fue estable en el backtest.</td></tr>
<tr><td>Confianza alta / media / baja</td><td>Ancho relativo de la banda frente a las horas anteriores.</td></tr>
<tr><td>Línea vertical (gráficas)</td><td>Hora seleccionada.</td></tr></table>

<h1>7. Solución de problemas</h1>
<div class="tit-tabla"><b>Tabla 9.</b> Síntoma, causa y acción</div>
<table><tr><th style="width:46mm">Síntoma</th><th style="width:56mm">Causa probable</th><th>Acción</th></tr>
<tr><td>Página en blanco o en carga.</td><td>Servidor en activación.</td><td>Esperar y recargar.</td></tr>
<tr><td>«No pude cargar la fuente de pronóstico».</td><td>Falta el archivo indicado en <code>fuentes_pronostico.json</code>.</td><td>Restaurar el repositorio completo.</td></tr>
<tr><td>Día no seleccionable.</td><td>Día sin las 24 horas pronosticadas.</td><td>Seleccionar otro día.</td></tr>
<tr><td>Sin marcadores ▼ / ▲.</td><td>Horizonte de 72 h.</td><td>Seleccionar 24 h.</td></tr>
<tr><td>Aviso ⚠ en la recomendación.</td><td>Ningún método estable para el rol y horizonte.</td><td>Revisar la comparación de métodos (5.5).</td></tr></table>

<h1>Anexo A. Modelo y datos</h1>
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
{aviso("Advertencia", "El cierre de ofertas es a las 08:00 de D-1 (Resolución CREG 025 de 1995) y el predespacho se publica después. El pronóstico de 24 h aplica a decisiones posteriores a la publicación del predespacho, no a la oferta de los generadores.")}
{aviso("Nota", "El horizonte de 72 h supone el precio conocido hasta el corte (escenario optimista); su desempeño con fuentes públicas sería menor.") if "optimista" in c["modelo72"] else aviso("Nota", "En el horizonte de 72 h no hay predespacho para los días D+1 y D+2: los pasos 25 a 72 parten del predespacho del día D y su error crece con el horizonte (Tabla A.1).")}
</body></html>"""


PIE = ('<div style="width:100%;font-family:Times New Roman,serif;font-size:8.5pt;color:#2B3D57;margin:0 20mm 0 25mm;'
       'border-top:1pt solid #D97706;padding-top:3pt;display:flex;justify-content:space-between">'
       f'<span>{DOC} · Rev. {REV} · Manual de usuario de augur</span>'
       '<span>Página <span class="pageNumber"></span> de <span class="totalPages"></span></span></div>')
CABEZA = ('<div style="width:100%;margin:0 20mm 0 25mm;border-bottom:0.75pt solid #0B1F3A;padding-bottom:3pt;display:flex;'
          'align-items:center;justify-content:space-between;font-family:Times New Roman,serif;font-size:8.5pt;color:#2B3D57">'
          f'<span style="display:flex;align-items:center;gap:5px">{logo(16)}<b style="font-family:Arial,sans-serif;font-size:10pt;color:#0B1F3A">augur</b></span>'
          '<span>Manual de usuario · Proyecto de grado · Ingeniería Electrónica</span></div>')


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
    finally:
        proc.terminate()
        time.sleep(2)


def paginas_de(pdf_ruta):
    """Pagina en que empieza cada titulo del indice (segunda pasada)."""
    import re
    from pypdf import PdfReader
    textos = [re.sub(r"\s+", " ", p.extract_text() or "").upper() for p in PdfReader(str(pdf_ruta)).pages]
    paginas = {}
    inicio = next(i for i, t in enumerate(textos) if "CONTENIDO" in t) + 1    # despues de la pagina del indice
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
    c = cifras()
    ruta = MAN / "manual.html"
    ruta.write_text(html(c, {}), encoding="utf-8")
    imprimir(ruta, PDF)
    pags = paginas_de(PDF)
    faltan = [t for _, t, _ in INDICE if t not in pags]
    ruta.write_text(html(c, pags), encoding="utf-8")
    imprimir(ruta, PDF)
    print("PDF:", PDF, PDF.stat().st_size, "| titulos sin pagina en el indice:", faltan)
