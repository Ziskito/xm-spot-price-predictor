# -*- coding: utf-8 -*-
"""
Dashboard del motor de decision (OE3).

Corre con:  streamlit run dashboard/app.py

Dos vistas, elegibles en la barra lateral:

  * "Operador" (por defecto): responde una sola pregunta -- "que hago ahora".
    Una tarjeta grande con la accion recomendada para la hora elegida, el
    pronostico del dia, la franja hora por hora, la confianza del pronostico
    y 3 indicadores. Sin perillas: el metodo de umbral lo elige el backtest.

  * "Analista": los controles finos (metodo manual, percentiles, rango de
    fechas, tabla comparativa de metodos).

Tema claro u oscuro (negro puro) con el interruptor de la barra lateral. La vista y el tema
iniciales se pueden fijar por enlace: ?vista=analista&tema=oscuro.

Los datos entran SIEMPRE por cargar_fuente_pronostico() (src/motor_decision.py),
es decir segun data/processed/resultados/fuentes_pronostico.json -- si ese
archivo cambia (ej. 72h pasa a otro modelo), este dashboard lo refleja sin
tocar una sola linea aqui. Ver docs/motor_decision_guia.md.
"""
import copy
import hmac
import json
import sys
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def encontrar_raiz_proyecto():
    actual = Path(__file__).resolve().parent
    for carpeta in [actual, *actual.parents]:
        if (carpeta / "src" / "motor_decision.py").exists():
            return carpeta
    raise FileNotFoundError(f"No encontre src/motor_decision.py subiendo desde {actual}")


RAIZ = encontrar_raiz_proyecto()
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))


@st.cache_resource
def _version_cargada():
    return {"v": None}


def _version_datos():
    """Huella de los archivos de pronostico: cambia con cada actualizacion diaria (GitHub Actions)."""
    res = RAIZ / "data" / "processed" / "resultados"
    return tuple(f.stat().st_mtime_ns if f.exists() else 0 for f in
                 (res / "fuentes_pronostico.json", res / "produccion" / "estado.json"))


# st.cache_data no sabe que los CSV cambiaron: sin esto, la app en linea seguiria mostrando los datos del
# momento en que arranco aunque el workflow diario suba datos nuevos
if _version_cargada()["v"] != _version_datos():
    st.cache_data.clear()
    _version_cargada()["v"] = _version_datos()

from motor_decision import (  # noqa: E402
    generar_senales, evaluar_backtest, comparar_metodos, elegir_mejor_metodo,
    comparar_metodos_estable, elegir_mejor_metodo_estable,
    cargar_fuente_pronostico, umbrales_fijos, umbrales_rodantes,
    horas_sin_umbral, precio_previo_ventana, ancho_relativo,
)

st.set_page_config(
    page_title="Augur · Motor de decisión",
    page_icon=str(Path(__file__).parent / "assets" / "augur_icono.png"),
    layout="wide",
)

# --------------------------------------------------------------------------
# Sistema de diseno: tokens por tema + hoja de estilo
# --------------------------------------------------------------------------
# Opcion B "Energia" del notebook 13 (azul marino institucional, ambar de acento). "Esperar" usa la
# clave "ambar" por compatibilidad, pero es gris: asi no choca con el acento y en el mapa de calor solo
# resaltan las horas en que conviene actuar (tambien es la que mejor se distingue con daltonismo).
TEMAS = {
    "claro": {
        "verde": "#047857", "verde_bg": "#D1FAE5",
        "rojo": "#C2410C", "rojo_bg": "#FFEDD5",
        "ambar": "#475569", "ambar_bg": "#E2E8F0",
        "tinta": "#0B1F3A", "texto": "#2B3D57", "gris": "#5B6B82",
        "linea": "#DDE3EC", "panel": "#FFFFFF", "fondo": "#F4F6FA", "lateral": "#FFFFFF",
        "neutro_bg": "#E9EDF3", "control": "#FFFFFF", "control_borde": "#C9D2DF",
        "banda": "rgba(30,58,138,0.12)", "q50": "#1E3A8A",
        "acento": "#D97706", "acento_texto": "#0B1F3A", "acento_sombra": "rgba(217,119,6,.35)",
        "calor_verde": "#6CCB9F", "calor_ambar": "#E6ECF3", "calor_rojo": "#F6AE79",
        "scroll": "#94A3B8", "scroll_hover": "#5B6B82", "sombra": "0 1px 2px rgba(11,31,58,.07)",
        "logo_tenue": "#94A3B8",
        "ultimo_bg": "#FEF3C7", "ultimo_txt": "#92400E", "sel_bg": "#0B1F3A", "sel_txt": "#FFFFFF",
        "franja_verde": "#A3F0BE", "franja_rojo": "#FEC98F", "franja_gris": "#DFE5EE",
        "franja_txt_verde": "#065F46", "franja_txt_rojo": "#9A3412", "franja_txt_gris": "#334155",
    },
}


def _rgba(hexa, a):
    h = hexa.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _mezcla(hexa, fondo, a):
    """Color opaco equivalente a `hexa` con opacidad `a` sobre `fondo` (para medir contraste y para la franja)."""
    c, f = (tuple(int(x.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4)) for x in (hexa, fondo))
    return "#" + "".join(f"{round(a * u + (1 - a) * v):02X}" for u, v in zip(c, f))


def _paleta_oscura(fondo, panel, lateral, linea, neutro, borde, tinta, texto, gris, verde, rojo, q50, acento,
                   calor_neutro, tenue, txt_verde, txt_rojo):
    return {"verde": verde, "verde_bg": _rgba(verde, .15), "rojo": rojo, "rojo_bg": _rgba(rojo, .16),
            "ambar": texto, "ambar_bg": _rgba(gris, .16), "tinta": tinta, "texto": texto, "gris": gris,
            "linea": linea, "panel": panel, "fondo": fondo, "lateral": lateral, "neutro_bg": neutro,
            "control": panel, "control_borde": borde, "banda": _rgba(q50, .18), "q50": q50,
            "acento": acento, "acento_texto": fondo, "acento_sombra": _rgba(acento, .30),
            "calor_verde": verde, "calor_ambar": calor_neutro, "calor_rojo": rojo,
            "scroll": borde, "scroll_hover": gris, "sombra": "0 1px 3px rgba(0,0,0,.35)", "logo_tenue": tenue,
            "ultimo_bg": _rgba(acento, .14), "ultimo_txt": acento, "sel_bg": neutro, "sel_txt": tinta,
            "franja_verde": _mezcla(verde, panel, .28), "franja_rojo": _mezcla(rojo, panel, .28),
            "franja_gris": neutro, "franja_txt_verde": txt_verde, "franja_txt_rojo": txt_rojo, "franja_txt_gris": texto}


# modo oscuro "Pizarra" (elegido por Juan el 16-oct-2026, en reemplazo del negro puro)
TEMAS["oscuro"] = _paleta_oscura("#0F172A", "#1E293B", "#162033", "#334155", "#273449", "#475569", "#F8FAFC",
                                 "#CBD5E1", "#94A3B8", "#34D399", "#FB923C", "#93C5FD", "#F59E0B", "#334155",
                                 "#64748B", "#A7F3D0", "#FED7AA")


def es(v, dec=0, signo=False):
    """Numero con formato colombiano: punto de miles y coma decimal (1.273; 446,2; +3,1)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{v:{'+' if signo else ''},.{dec}f}".replace(",", "§").replace(".", ",").replace("§", ".")


FUENTE = "'IBM Plex Sans', 'Segoe UI', sans-serif"  # cargada desde Google Fonts en .streamlit/config.toml

CSS_BASE = Template("""
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@600;700&display=swap');
.stApp { background: $fondo; color: $tinta; }
[data-testid="stHeader"] { background: transparent; }
/* boton para abrir/cerrar la barra lateral: color de acento para que se vea a la primera */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button, [data-testid="stHeader"] button {
  background: $acento !important; border: none !important; border-radius: 10px !important;
  color: $acento_texto !important; opacity: 1 !important; padding: 6px 12px 6px 8px !important;
  width: auto !important; height: auto !important;
  box-shadow: 0 2px 10px $acento_sombra !important; }
[data-testid="stExpandSidebarButton"]:hover, [data-testid="stHeader"] button:hover {
  filter: brightness(1.12); }
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapseButton"] button *,
[data-testid="stHeader"] button * { color: $acento_texto !important; fill: $acento_texto !important; opacity: 1 !important; }
[data-testid="stExpandSidebarButton"]::after { content: "Menú"; font-weight: 700; font-size: .9rem;
  margin-left: 6px; color: $acento_texto; }
[data-testid="stNumberInputContainer"], [data-testid="stNumberInputContainer"] input {
  background: $control !important; border-color: $control_borde !important;
  color: $tinta !important; -webkit-text-fill-color: $tinta !important; }
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { color: $tinta !important; }
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], #MainMenu, footer { display: none; }
.block-container { padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1240px; }
h1, h2, h3, h4, h5 { color: $tinta; }
.stApp p, .stApp li, .stApp label { color: $texto; }
.stMarkdown strong, .stMarkdown b { color: $tinta; }
hr { border-color: $linea !important; }

[data-testid="stSidebar"] { background: $lateral; border-right: 1px solid $linea; }

/* barras de desplazamiento visibles en ambos temas (Chrome/Edge y Firefox) */
* { scrollbar-width: thin; scrollbar-color: $scroll transparent; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: $scroll; border-radius: 8px; border: 2px solid transparent;
                            background-clip: content-box; }
::-webkit-scrollbar-thumb:hover { background: $scroll_hover; background-clip: content-box; }
[data-testid="stSidebar"] * { color: $texto; }
[data-testid="stSidebar"] h3 { color: $tinta; }
[data-testid="stWidgetLabel"] p { color: $tinta !important; font-weight: 600; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: $gris !important; opacity: 1 !important; }
/* titulo principal como h1 (accesibilidad) sin el estilo ni el ancla que Streamlit le pone a los encabezados */
.cab-card h1.tit { margin: 0; padding: 0; font-family: inherit; letter-spacing: normal; }
.cab-card h1.tit a, .cab-card [data-testid="stHeaderActionElements"] { display: none !important; }
[data-testid="stElementContainer"]:has(> [data-testid="stHtml"] script):not(:has(div, p, span)) { display: none; }
[class*="st-key-oculto_"] { display: none !important; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden;
  clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
.ley-calor { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: .78rem; color: $texto; margin: -4px 0 8px 4px; }
.ley-calor i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px;
  vertical-align: -2px; border: 1px solid $linea; }

/* controles de formulario (Streamlit 1.6x usa componentes react-aria) */
.stSelectbox [role="group"], [data-testid="stDateInputField"] {
  background: $control !important; border-color: $control_borde !important; }
.stSelectbox input, [data-testid="stDateInputField"] span {
  color: $tinta !important; -webkit-text-fill-color: $tinta !important; }
.stSelectbox button svg, [data-testid="stDateInputField"] svg { fill: $gris; color: $gris; }
.react-aria-Popover, [role="listbox"], [role="dialog"] {
  background: $control !important; color: $tinta !important; border-color: $control_borde !important; }
[role="option"], [role="dialog"] button, [role="dialog"] td, [role="dialog"] th,
[role="dialog"] span, [role="gridcell"] { color: $tinta !important; }
[role="option"][data-focused], [role="option"][data-hovered] { background: $neutro_bg !important; }
[data-testid="stSliderTickBar"] p, [data-testid="stSliderThumbValue"] p { color: $gris !important; }
/* deslizadores de rango: cuando los dos extremos quedan encimados (por ejemplo, un solo dia), el de
   encima no deja mover el otro. Cada extremo se agarra solo por su mitad exterior: la izquierda mueve
   el inicio y la derecha el final, asi siempre se pueden separar */
[data-testid="stSlider"] div:has(> div[data-rac] + div[data-rac]) > div[data-rac] { pointer-events: none; }
[data-testid="stSlider"] div:has(> div[data-rac] + div[data-rac]) > div[data-rac]::before {
  content: ""; position: absolute; top: -10px; bottom: -10px; pointer-events: auto; }
[data-testid="stSlider"] div:has(> div[data-rac] + div[data-rac]) > div[data-rac]:has(+ div[data-rac])::before {
  left: -12px; right: 50%; }
[data-testid="stSlider"] div:has(> div[data-rac] + div[data-rac]) > div[data-rac] + div[data-rac]::before {
  left: 50%; right: -12px; }
/* mapa de calor: las fechas de los extremos de la barra de dias se editan directamente (campos de fecha con el
   aspecto de las etiquetas de la barra); las etiquetas propias de la barra se ocultan */
.st-key-dias_calor [data-testid="stSliderThumbValue"], .st-key-dias_calor [data-testid="stSliderTickBar"] { display: none; }
.st-key-dias_calor, .st-key-dias_calor [data-testid="stHorizontalBlock"] { gap: 0 !important; }
.st-key-dias_calor [data-testid="stDateInputField"] {
  background: transparent !important; border: none !important; box-shadow: none !important; min-height: 0 !important;
  height: 22px; padding: 0 !important; }
.st-key-dias_calor [data-testid="stDateInputField"] [role="group"] { padding: 0 !important; }
.st-key-dias_calor [data-testid="stHorizontalBlock"] { margin-top: 13px; margin-bottom: -13px; position: relative; z-index: 1; }
.st-key-dias_calor [data-testid="stDateInputField"] span { color: $gris !important; -webkit-text-fill-color: $gris !important;
  font-size: .875rem; cursor: text; }
.st-key-dias_calor [data-testid="stDateInputField"]:hover span { color: $tinta !important; -webkit-text-fill-color: $tinta !important; }
.st-key-dias_calor [data-testid="stDateInputField"] button { display: none; }
.st-key-dias_fin [data-testid="stDateInputField"] [role="group"] { justify-content: flex-end; }
.etq-dias { font-size: .875rem; font-weight: 600; color: $tinta; margin-bottom: 2px; }
/* botones de dia anterior / siguiente / ultimo */
.st-key-dia_ant button, .st-key-dia_sig button, .st-key-dia_ult button {
  background: $control !important; color: $tinta !important; border: 1px solid $control_borde !important;
  border-radius: 10px !important; min-height: 2.5rem; padding: 0 8px !important; }
.st-key-dia_ant button p, .st-key-dia_sig button p, .st-key-dia_ult button p { color: $tinta !important; font-weight: 600; font-size: .86rem; }
.st-key-dia_ant button:hover:enabled, .st-key-dia_sig button:hover:enabled, .st-key-dia_ult button:hover:enabled {
  border-color: $acento !important; }
.st-key-dia_ant button:disabled, .st-key-dia_sig button:disabled, .st-key-dia_ult button:disabled { opacity: .45; }
.st-key-dia_ult { margin-left: 14px; }
.st-key-dia_ult button { background: $ultimo_bg !important; border-color: $acento !important; }
.st-key-dia_ult button p, .st-key-dia_ult button span { color: $ultimo_txt !important; }
.st-key-dia_ult button:hover:enabled { filter: brightness(.96); }
/* calendario del selector de fecha: Streamlit lo dibuja con el fondo del tema base (claro), lo que en
   modo oscuro dejaba los numeros claros sobre blanco */
[data-testid="stDateInputCalendar"] { background: $control !important; border: 1px solid $control_borde;
  border-radius: 10px; box-shadow: 0 8px 24px rgba(0,0,0,.25); }
[data-testid="stDateInputCalendar"] * { color: $tinta; }
[data-testid="stDateInputCalendar"] svg { fill: $tinta; }
[data-testid="stDateInputCalendar"] td[aria-disabled="true"], [data-testid="stDateInputCalendar"] td[aria-disabled="true"] * {
  color: $gris !important; opacity: .55; }
[data-testid="stDateInputCalendar"] td[aria-selected="true"] *, [data-testid="stDateInputCalendar"] [aria-selected="true"] {
  color: $acento_texto !important; }
[data-testid="stDateInputCalendar"] select, [data-testid="stDateInputCalendar"] button { background: transparent; }
[data-testid="stExpander"] details { background: $panel; border: 1px solid $linea; border-radius: 10px; }
[data-testid="stExpander"] summary p { color: $tinta !important; }
[data-testid="stTooltipIcon"] svg { stroke: $gris; }

.titulo-app { font-size: 1.55rem; font-weight: 800; color: $tinta; margin-bottom: .1rem; }
/* marca Augur (diseño 08 "Bandada en tendencia", dashboard/mockups/augur.html) */
.marca { display: flex; align-items: center; gap: 10px; }
.marca svg { flex: none; }
.marca-nom { font-family: 'Lexend', 'IBM Plex Sans', sans-serif; font-weight: 700; color: $tinta; line-height: 1; letter-spacing: -.01em; }
.marca-sub { font-size: .62rem; letter-spacing: .13em; text-transform: uppercase; color: $gris; font-weight: 600; margin-top: 5px; }
.marca-lateral { margin: 0 0 14px; }
.marca-lateral .marca-nom { font-size: 1.7rem; }
.marca-cab { margin-bottom: 0; flex: none; }
.marca-cab .marca-nom { font-size: 1.9rem; }
.marca-cab .marca-sub { font-size: .74rem; }
.sub-app { color: $gris; margin-bottom: 1.1rem; font-size: .95rem; }
.seccion { font-weight: 700; color: $tinta; margin: 10px 0 4px; }

.hero { border-radius: 12px; padding: 20px 22px; border: 1px solid $linea;
        background: $panel; margin-bottom: 16px; box-shadow: $sombra; min-height: 340px; }
.hero-verde { border-top: 5px solid $verde; }
.hero-rojo  { border-top: 5px solid $rojo; }
.hero-ambar { border-top: 5px solid $ambar; }
.hero-tag { font-size: .76rem; letter-spacing: .05em; text-transform: uppercase; color: $gris; }
.hero-accion { font-size: 2.05rem; font-weight: 800; margin: .25rem 0 .45rem; line-height: 1.1; }
.hero-verde .hero-accion { color: $verde; }
.hero-rojo  .hero-accion { color: $rojo; }
.hero-ambar .hero-accion { color: $ambar; }
.hero-frase { font-size: 1rem; color: $texto; max-width: 62ch; }

.gauge-track { position: relative; height: 14px; border-radius: 6px;
               margin: 22px 0 7px; border: 1px solid $linea; }
.gauge-marker { position: absolute; top: -6px; width: 3px; height: 24px;
                background: $tinta; border-radius: 2px; transform: translateX(-1.5px); }
.gauge-labels { display: flex; justify-content: space-between; font-size: .76rem; color: $gris; }
.gauge-labels b { color: $tinta; }

.chip { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .78rem; font-weight: 700; }
.chip-alta  { background: $verde_bg; color: $verde; }
.chip-media { background: $ambar_bg; color: $ambar; }
.chip-baja  { background: $rojo_bg; color: $rojo; }
.chip-nota { color: $gris; font-size: .8rem; margin-left: 8px; }

.tarjetas { display: flex; gap: 12px; margin: 6px 0 12px; flex-wrap: wrap; }
.stat { flex: 1 1 220px; background: $panel; border: 1px solid $linea;
        border-radius: 10px; padding: 14px 16px; box-shadow: $sombra; }
.stat-l { font-size: .8rem; color: $gris; }
.stat-fila { display: flex; align-items: flex-end; justify-content: space-between; gap: 10px; }
.stat-v { font-size: 1.5rem; font-weight: 750; color: $tinta; margin: 2px 0; white-space: nowrap; }
.stat-s { font-size: .78rem; color: $gris; }

.leyenda { font-size: .82rem; color: $gris; margin-top: 4px; }
.leyenda b { color: $tinta; }
.meta { font-size: .74rem; color: $gris; margin: -2px 0 18px; line-height: 1.45; }
.meta b { color: $texto; font-weight: 600; }
[data-testid="stRadio"] label p { color: $texto; }

[data-testid="stPlotlyChart"] { background: $panel; border: 1px solid $linea; border-radius: 10px;
                                padding: 6px 6px 2px; box-shadow: $sombra; }

.modelo { background: $verde_bg; border-radius: 8px; padding: 12px 14px; font-size: .86rem; }
.modelo b { color: $verde !important; }
.modelo span { color: $texto !important; }

.tabla-met { width: 100%; border-collapse: separate; border-spacing: 0; background: $panel;
             border: 1px solid $linea; border-radius: 10px; overflow: hidden; font-size: .9rem; }
.tabla-met th { text-align: left; color: $gris; font-weight: 600; padding: 10px 14px;
                border-bottom: 1px solid $linea; font-size: .8rem; }
.tabla-met th, .tabla-met td { border-left: none !important; border-right: none !important; border-top: none !important; }
.tabla-met td { padding: 10px 14px; color: $tinta; border-bottom: 1px solid $linea; }
.tabla-met tr:last-child td { border-bottom: none; }
.tabla-met tr.activo td { background: $neutro_bg; font-weight: 700; }
.etq { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: .72rem;
       font-weight: 700; margin-left: 6px; }
.etq-si { background: $verde_bg; color: $verde; }
.etq-no { background: $rojo_bg; color: $rojo; }
.etq-sug { background: $neutro_bg; color: $tinta; }
/* APROBADO: vistas como navegacion en la barra lateral, la activa en una tarjeta oscura */
[data-testid="stSidebar"] .st-key-vista_sel [role="radiogroup"] { gap: 4px; width: 100%; }
[data-testid="stSidebar"] .st-key-vista_sel label { width: 100%; padding: 8px 12px; border-radius: 8px; margin: 0; }
[data-testid="stSidebar"] .st-key-vista_sel label:has(input:checked) { background: $sel_bg; box-shadow: inset 3px 0 0 $acento; }
[data-testid="stSidebar"] .st-key-vista_sel label:has(input:checked) * { color: $sel_txt !important; }

/* ===== v2: estilo "Control Room" (Stitch) ===== */
.stApp { font-variant-numeric: tabular-nums; }
.block-container { max-width: 1320px; }
.cab-card { background: $panel; border: 1px solid $linea; border-radius: 12px; padding: 16px 22px; box-shadow: $sombra;
            display: flex; align-items: center; gap: 22px; margin-bottom: 16px; }
.cab-card .tit { font-size: 1.3rem; font-weight: 700; color: $tinta; line-height: 1.25; }
.cab-card .bajada { font-size: .9rem; color: $gris; margin-top: 4px; max-width: 80ch; }
.cab-card .der { margin-left: auto; text-align: right; font-size: .78rem; color: $gris; white-space: nowrap; }
.cab-card .der b { color: $tinta; }
.pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .7rem; font-weight: 700;
        letter-spacing: .06em; text-transform: uppercase; background: $neutro_bg; color: $tinta; }
.st-key-sec_franja, .st-key-sec_calor, .st-key-sec_semana, .st-key-sec_datos, .st-key-sec_metodos, .st-key-sec_backtest,
.st-key-sec_desempeno, .st-key-sec_metodo, .st-key-sec_formulas, .st-key-sec_acceso {
  background: $panel; border: 1px solid $linea; border-radius: 12px; padding: 16px 18px 6px; box-shadow: $sombra;
  margin-bottom: 14px; }
.st-key-sec_acceso { margin-top: 12vh; padding: 22px 24px 14px; }
.st-key-sec_acceso [data-baseweb="input"], .st-key-sec_acceso [data-baseweb="base-input"],
.st-key-sec_acceso input { background: $control !important; color: $tinta !important; }
.st-key-sec_acceso [data-baseweb="input"] { border: 1px solid $control_borde !important; }
.st-key-sec_acceso [data-testid="stTextInputRootElement"], .st-key-sec_acceso [data-testid="stTextInputRootElement"] *:not(svg):not(path),
.st-key-sec_acceso [data-baseweb="input"] > div, .st-key-sec_acceso [data-baseweb="input"] button {
  background: $control !important; border-color: $control_borde !important; }
.st-key-sec_acceso [data-baseweb="input"] button svg { fill: $gris !important; color: $gris !important; }
.st-key-sec_acceso input::placeholder { color: $gris !important; opacity: 1; }
.st-key-sec_acceso button[kind="primaryFormSubmit"] { background: $acento !important; border-color: $acento !important; }
.st-key-sec_acceso button[kind="primaryFormSubmit"] * { color: $acento_texto !important; font-weight: 700; }
[class*="st-key-form_"] { border: 1px solid $linea; border-radius: 10px; padding: 10px 14px 4px; margin-bottom: 4px; }
[class*="st-key-form_"] .t { font-size: .68rem; letter-spacing: .08em; text-transform: uppercase; color: $gris;
  font-weight: 700; }
[class*="st-key-form_"] p, [class*="st-key-form_"] li { font-size: .82rem; color: $texto; }
.st-key-sec_formulas .katex { color: $tinta; font-size: 1.05em; }
[data-testid="stPlotlyChart"] { border: none !important; box-shadow: none !important; padding: 0 !important;
  border-radius: 0 !important; background: transparent !important; }
.sec-tit { font-size: 1.05rem; font-weight: 700; color: $tinta; }
.sec-sub { font-size: .82rem; color: $gris; margin: 2px 0 8px; }
.meta { background: $fondo; border: 1px solid $linea; border-radius: 8px; padding: 8px 12px; margin: 4px 0 16px; }
.hero { border-radius: 12px; min-height: 0; }
.badge { display: inline-block; padding: 3px 9px; border-radius: 4px; font-size: .7rem; font-weight: 700;
         letter-spacing: .07em; text-transform: uppercase; vertical-align: middle; }
.badge-verde { background: $verde_bg; color: $verde; border: 1px solid $verde; }
.badge-rojo { background: $rojo_bg; color: $rojo; border: 1px solid $rojo; }
.badge-ambar { background: $ambar_bg; color: $ambar; border: 1px solid $ambar; }
.hero-cab { display: flex; align-items: center; gap: 12px; margin: .35rem 0 .45rem; }
.hero-cab .hero-accion { margin: 0; }
.escala-tit { font-size: .68rem; letter-spacing: .08em; text-transform: uppercase; color: $gris; font-weight: 700;
              margin-top: 18px; display: flex; justify-content: space-between; align-items: baseline; }
.escala-tit .v { font-size: 1.25rem; letter-spacing: 0; text-transform: none; color: $tinta; font-weight: 700; }
.gauge-track { height: 12px; border-radius: 4px; margin: 26px 0 6px; border: none; }
.gauge-marker { top: -7px; height: 26px; width: 3px; }
.gauge-bocadillo { position: absolute; top: -26px; transform: translateX(-50%); background: $tinta; color: $panel;
                   font-size: .7rem; font-weight: 700; padding: 1px 6px; border-radius: 4px; white-space: nowrap; }
.escala3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; font-size: .72rem; color: $gris; line-height: 1.35; }
.escala3 div:nth-child(2) { text-align: center; }
.escala3 div:nth-child(3) { text-align: right; }
.escala3 b { color: $tinta; }
.sub-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
.sub-card { border: 1px solid $linea; border-radius: 8px; padding: 10px 12px; background: $fondo; font-size: .8rem; color: $texto; }
.sub-card .t { font-size: .66rem; letter-spacing: .08em; text-transform: uppercase; color: $gris; font-weight: 700; margin-bottom: 4px; }
.sub-card .g { font-size: .95rem; font-weight: 700; color: $tinta; }
.stat-l { text-transform: uppercase; letter-spacing: .07em; font-size: .7rem; font-weight: 700; }
.distrib { display: flex; height: 10px; border-radius: 5px; overflow: hidden; margin: 10px 0 6px; }
.distrib-ley { font-size: .74rem; color: $gris; }
.distrib-ley i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin: 0 4px 0 8px; }
.st-key-sec_franja { padding: 12px 16px 8px; }
.st-key-sec_franja [data-testid="stVerticalBlock"] { gap: .6rem; }
.st-key-sec_franja .leyenda { font-size: .72rem; line-height: 1.45; }
.st-key-sec_franja .sec-sub { margin-bottom: 4px; }
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-sec_dia_"]), [class*="st-key-sec_dia_"] { flex: 1 1 auto; }
[class*="st-key-sec_dia_"] { background: $panel; border: 1px solid $linea; border-radius: 12px; padding: 14px 16px 4px;
  box-shadow: $sombra; }
[class*="st-key-sec_dia_"] [data-testid="stPlotlyChart"], .st-key-sec_semana [data-testid="stPlotlyChart"] {
  border: none; box-shadow: none; padding: 0; }
[class*="st-key-sec_dia_"] [data-testid="stVerticalBlock"], .st-key-sec_semana [data-testid="stVerticalBlock"] { gap: .45rem; }
.pista-ley { font-size: .74rem; color: $gris; }
/* pantalla completa de Streamlit y ventana Ampliar: sin el borde ni el relleno de tarjeta, que la hacian desbordar */
[data-testid="stPlotlyChart"]:has(button[aria-label="Close fullscreen"]), .st-key-graf_ampliada [data-testid="stPlotlyChart"] {
  padding: 0 !important; border: none !important; box-shadow: none !important; border-radius: 0 !important; }
.st-key-graf_ampliada { overflow: hidden; }
[class*="st-key-ampliar_"] { display: flex; justify-content: flex-end; }
[class*="st-key-ampliar_"] button p { font-size: .8rem; }
/* pantallas angostas (tablet, ventana pequena): tarjeta y grafica del dia una debajo de la otra, asi la
   leyenda cabe en dos columnas y la tarjeta no se aprieta */
@media (max-width: 1200px) {
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .hero-d) { flex-wrap: wrap; }
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .hero-d) > [data-testid="stColumn"] {
    width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }
}
@media (max-width: 900px) {
  .cab-card { flex-wrap: wrap; gap: 12px 22px; }
  .cab-card > div:nth-child(2) { flex: 1 1 100%; }
  .cab-card .der { margin-left: 0; text-align: left; }
}
/* vista Metodologia */
.st-key-sec_desempeno .tarjetas { margin: 6px 0 4px; }
.flujo { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 6px 0 14px; }
.paso { border: 1px solid $linea; border-top: 3px solid $acento; border-radius: 10px; padding: 10px 12px; background: $fondo; }
.paso .n { font-size: .74rem; font-weight: 700; letter-spacing: .04em; color: $tinta; margin-bottom: 4px; }
.paso .d { font-size: .78rem; color: $texto; line-height: 1.45; }
.fichas-met { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }
.ficha-met { border: 1px solid $linea; border-radius: 10px; padding: 10px 14px; background: $panel; font-size: .82rem;
             color: $texto; line-height: 1.5; }
.ficha-met .t { font-size: .68rem; letter-spacing: .08em; text-transform: uppercase; color: $gris; font-weight: 700; }
.ficha-met ul { margin: 4px 0 0; padding-left: 18px; } .ficha-met li { margin-bottom: 3px; }
.ficha-met a { color: $acento; font-weight: 600; }
@media (max-width: 1100px) { .flujo { grid-template-columns: 1fr 1fr; } .fichas-met { grid-template-columns: 1fr; } }
/* tarjeta de recomendacion con medidor */
.hero-d { padding: 14px 18px 16px; }
/* la tarjeta se estira hasta el alto de la grafica del dia y reparte el espacio sobrante */
[data-testid="stElementContainer"]:has(.hero-d) { flex: 1 1 auto; }
[data-testid="stElementContainer"]:has(.hero-d) .stMarkdown,
[data-testid="stElementContainer"]:has(.hero-d) .stMarkdown > div,
[data-testid="stElementContainer"]:has(.hero-d) [data-testid="stMarkdownContainer"] { height: 100%; align-items: stretch; }
.hero-d { height: 100%; box-sizing: border-box; margin-bottom: 0; display: flex; flex-direction: column;
          justify-content: space-evenly; gap: 6px; }
.hero-d .hero-tag { font-size: .68rem; }
.hero-d .hero-cab { margin: 0 0 .1rem; }
.hero-d .badge { font-size: .66rem; padding: 2px 8px; }
.hero-d .hero-accion { font-size: clamp(1.2rem, 1.75vw, 1.6rem); margin: .15rem 0 .35rem; }
.hero-d .hero-frase { font-size: .95rem; line-height: 1.5; }
.hero-d .sub-cards { margin-top: 10px; gap: 8px; }
.hero-d .sub-card { padding: 9px 12px; font-size: .76rem; line-height: 1.5; }
.hero-d .sub-card .t { font-size: .6rem; margin-bottom: 2px; }
.hero-d .sub-card .g { font-size: .92rem; }
.hero-d .chip { font-size: .68rem; padding: 2px 8px; }
.hero-d .hd-fila { display: grid; grid-template-columns: 190px 1fr; gap: 18px; align-items: center; margin-top: 4px; }
.hd-svg { width: 100%; display: block; }
.hd-et { font-size: 16px; font-weight: 700; fill: $texto; }
.hd-v { text-align: center; font-size: 1.6rem; font-weight: 800; color: $tinta; margin-top: -2px; line-height: 1.1; }
.hd-u { text-align: center; font-size: .7rem; color: $gris; }
.hd-ley { font-size: .78rem; color: $gris; margin-top: 8px; line-height: 1.65; }
.hd-ley i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 6px; }
.hd-ley b { color: $tinta; }
""")

CSS_OSCURO_EXTRA = """
:root, .stApp { color-scheme: dark; }
[data-testid="stDataFrame"] { filter: invert(0.92) hue-rotate(180deg); }
"""

# --------------------------------------------------------------------------
# Semantica de cada senal (texto para humanos, no jerga de percentiles)
# --------------------------------------------------------------------------
SIGNIFICADO = {
    "comprar":       ("COMPRAR EN BOLSA",       "verde"),
    "vender":        ("DESPACHAR Y VENDER",     "verde"),
    "evitar_compra": ("EVITAR LA BOLSA",        "rojo"),
    "retener":       ("RETENER GENERACIÓN",     "rojo"),
    "esperar":       ("ESPERAR",                "ambar"),
}
FRASE = {
    "comprar": "El precio esperado está barato. Es buena hora para cubrir consumo comprando en bolsa.",
    "vender": "El precio esperado está alto. Es buena hora para despachar y vender energía.",
    "evitar_compra": "El precio esperado está alto. Conviene cubrirse con contratos y no comprar en bolsa ahora.",
    "retener": "El precio esperado está bajo. Si puedes, evita despachar y guarda generación para más tarde.",
    "esperar": "El precio esperado está en su rango normal. No hay una ventaja clara en actuar ahora.",
}
MESES = ["", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


# --------------------------------------------------------------------------
# Carga de datos (cacheada)
# --------------------------------------------------------------------------
# st.cache_data solo se invalida cuando cambia el codigo de la funcion cacheada, no las constantes que usa (el 20-oct
# la pagina publicada seguia mostrando "Pico" en la leyenda tras cambiar NOMBRE_CORTO). Cambiar esta version al tocar
# textos, nombres o estilos que quedan dentro de datos o figuras en cache: el primer usuario tras el despliegue las limpia.
VERSION_CACHE = "2026-10-20"


@st.cache_resource
def _limpiar_cache_al_cambiar_version(version):
    st.cache_data.clear()
    return version


_limpiar_cache_al_cambiar_version(VERSION_CACHE)


@st.cache_data
def cargar_bandas(horizonte):
    try:
        return cargar_fuente_pronostico(RAIZ, horizonte)
    except (FileNotFoundError, KeyError, ValueError) as e:
        return None, {"error": str(e)}


@st.cache_data
def cargar_precio_historico():
    path = RAIZ / "data" / "processed" / "dataset_maestro_2019_2025.csv"
    return pd.read_csv(path, usecols=["precio_bolsa"])["precio_bolsa"].values


@st.cache_data
def cargar_precio_fechado():
    """Precio real 2019-2026 indexado por hora: calienta la ventana de 30 dias de "rodante"/"hibrido"
    con los precios anteriores al inicio del pronostico (si no, el primer mes sale todo "esperar")."""
    partes = [pd.read_csv(RAIZ / "data" / "processed" / f, usecols=["fecha_hora", "precio_bolsa"],
                          parse_dates=["fecha_hora"])
              for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]
    return pd.concat(partes).drop_duplicates("fecha_hora").set_index("fecha_hora")["precio_bolsa"].sort_index()


@st.cache_data
def comparacion(horizonte, rol, p_bajo, p_alto, inicio=None, fin=None, intradia=True):
    """Corre el backtest de los 4 metodos y devuelve (tabla, metodo_sugerido, es_valido).

    Si se dan inicio/fin (fechas), la senal se calcula sobre TODA la serie
    (continuidad para "rodante"/"hibrido") pero el backtest -- y por lo tanto
    que metodo queda como "sugerido" -- se mide solo en ese rango."""
    bandas, _ = cargar_bandas(horizonte)
    mascara = None
    if inicio is not None and fin is not None:
        mascara = (bandas["fecha_hora"].dt.date >= inicio) & (bandas["fecha_hora"].dt.date <= fin)
    tabla = comparar_metodos(bandas, rol, cargar_precio_historico(), p_bajo=p_bajo, p_alto=p_alto,
                             mascara_evaluacion=mascara, precio_previo=cargar_precio_fechado(), intradia=intradia)
    metodo, es_valido = elegir_mejor_metodo(tabla)
    return tabla, metodo, es_valido


@st.cache_data
def comparacion_estable(horizonte, rol, intradia=True):
    """Elige el metodo que mejor se sostiene en su mitad mas debil del periodo
    (criterio maximin, ver comparar_metodos_estable() en motor_decision.py)."""
    bandas, _ = cargar_bandas(horizonte)
    tabla = comparar_metodos_estable(bandas, rol, cargar_precio_historico(), precio_previo=cargar_precio_fechado(),
                                     intradia=intradia)
    metodo, es_estable = elegir_mejor_metodo_estable(tabla)
    return tabla, metodo, es_estable


UMBRAL_ALERTA = 0.30   # probabilidad desde la que se marca una hora con riesgo de desplome o pico


@st.cache_data
def cargar_probabilidades_eventos():
    """P(desplome) y P(pico) por hora del pronostico de 24 h: desplome = precio <= 0,7 x mediana del dia;
    pico = precio >= 1,3 x mediana. El archivo lo indica la clave "eventos" del horizonte de 24 h en
    fuentes_pronostico.json (desde el 14-oct-2026, el detector realista de realista_eventos.py); sin esa clave,
    el de eventos_escenarios_24h.py. None si no existe el archivo."""
    res = RAIZ / "data" / "processed" / "resultados"
    try:
        nombre = json.loads((res / "fuentes_pronostico.json").read_text(encoding="utf-8"))["24h"].get("eventos")
    except Exception:
        nombre = None
    ruta = res / (nombre or "probabilidades_eventos_24h_2026.csv")
    if not ruta.exists():
        return None
    return pd.read_csv(ruta, parse_dates=["fecha_hora"]).drop_duplicates("fecha_hora").set_index("fecha_hora")


@st.cache_data
def serie_con_senal(horizonte, rol, metodo, p_bajo=25, p_alto=75, intradia=True):
    bandas, _ = cargar_bandas(horizonte)
    previo = cargar_precio_fechado()
    senal = generar_senales(bandas, metodo, rol, cargar_precio_historico(), p_bajo=p_bajo, p_alto=p_alto,
                            precio_previo=previo, intradia=intradia)
    # horas en que el metodo aun no tiene umbral: su "esperar" es por falta de datos, no una decision
    out = bandas.assign(senal=senal, sin_umbral=horas_sin_umbral(bandas, metodo, precio_previo=previo))
    prob = cargar_probabilidades_eventos()
    out["p_desplome"], out["p_pico"] = np.nan, np.nan
    validas = np.ones(len(out), dtype=bool) if horizonte == "24h" else pasos_1_24(horizonte, out["fecha_hora"])
    if prob is not None:
        # En 72 h (Juan, 19-oct: no salian los picos ni los desplomes) el detector de 24 h solo vale en los pasos
        # 1-24, que son el mismo pronostico de 24 h con el mismo corte.
        for col in ("p_desplome", "p_pico"):
            out.loc[validas, col] = prob[col].reindex(out.loc[validas, "fecha_hora"]).to_numpy()
    # Pasos 25-72: detector por horizonte (realista_eventos_72h.py). Validado contra la climatologia (20-oct-2026), solo
    # el de pico se publica; el de desplome no la supera de forma consistente en los años historicos.
    out["solo_pico"] = False
    if horizonte == "72h":
        p72 = cargar_probabilidades_eventos_72h()
        if p72 is not None:
            resto = ~validas
            out.loc[resto, "p_pico"] = p72["p_pico"].reindex(out.loc[resto, "fecha_hora"]).to_numpy()
            out.loc[resto, "solo_pico"] = True
    return out


@st.cache_data
def cargar_probabilidades_eventos_72h():
    """P(pico) de los pasos 25-72 del contrato de 72 h (realista_eventos_72h.py --operacion). None si no existe."""
    ruta = RAIZ / "data" / "processed" / "resultados" / "probabilidades_eventos_72h_realista_2026.csv"
    if not ruta.exists():
        return None
    return pd.read_csv(ruta, parse_dates=["fecha_hora"]).drop_duplicates("fecha_hora").set_index("fecha_hora")


@st.cache_data
def pasos_1_24(horizonte, fechas):
    """True en las horas del horizonte de 72 h que son los pasos 1-24 de su corte (mismo pronostico y corte que 24 h)."""
    try:
        res = RAIZ / "data" / "processed" / "resultados"
        nombre = json.loads((res / "fuentes_pronostico.json").read_text(encoding="utf-8"))[horizonte]["archivo"]
        paso = pd.read_csv(res / nombre, usecols=["fecha_hora", "paso_horas"], parse_dates=["fecha_hora"]
                           ).drop_duplicates("fecha_hora").set_index("fecha_hora")["paso_horas"]
        return (paso.reindex(fechas) <= 24).fillna(False).to_numpy()
    except (FileNotFoundError, KeyError, ValueError):
        return np.zeros(len(fechas), dtype=bool)


@st.cache_data
def resumen_mensual(horizonte, rol, metodo, intradia=True):
    """Ventaja y frecuencia de accion del metodo, mes a mes (para las mini graficas)."""
    df = serie_con_senal(horizonte, rol, metodo, intradia=intradia)
    filas = []
    for (anio, mes), g in df.groupby([df["fecha_hora"].dt.year, df["fecha_hora"].dt.month]):
        bt = evaluar_backtest(g, g["senal"], rol)
        filas.append({"mes": f"{MESES[mes]} {anio}", "ventaja": bt["ventaja_cop_kwh"],
                      "frecuencia": bt["frecuencia_accion"] * 100})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Utilidades de presentacion
# --------------------------------------------------------------------------
def nivel_confianza(ancho_actual, anchos_historicos):
    """Confianza = que tan estrecha es la banda [q10,q90] de esta hora frente
    a las horas de referencia (en el operador: ancho relativo al precio, horas anteriores).
    Banda ancha -> el modelo esta menos seguro -> confianza baja."""
    pct_mas_anchas_que = float((anchos_historicos < ancho_actual).mean() * 100)
    if pct_mas_anchas_que >= 75:
        return "baja", pct_mas_anchas_que
    if pct_mas_anchas_que >= 40:
        return "media", pct_mas_anchas_que
    return "alta", pct_mas_anchas_que


def pos_en_barra(valor, lo, hi):
    return float(np.clip((valor - lo) / (hi - lo), 0, 1) * 100)


def umbrales_serie(bandas, metodo, precio_hist, p_bajo, p_alto):
    """Umbrales (bajo, alto) para cada fila de bandas: series moviles para rodante/hibrido, constantes si no."""
    if metodo in ("rodante", "hibrido"):
        return umbrales_rodantes(bandas["q50"], 30, p_bajo, p_alto,
                                 previo=precio_previo_ventana(bandas, cargar_precio_fechado(), 30))
    u = umbrales_fijos(precio_hist, p_bajo, p_alto)
    return (pd.Series(u["bajo"], index=bandas.index), pd.Series(u["alto"], index=bandas.index))


def _set_hora(h):
    st.session_state["hora_operador"] = st.session_state["_copia_hora_operador"] = h


def _mover_dia(paso, dias):
    """Botones de dia anterior / siguiente / ultimo: salta al dia disponible mas cercano en esa direccion."""
    actual = st.session_state.get("dia_operador", dias[-1])
    if paso == "ultimo":
        nuevo = dias[-1]
    elif paso < 0:
        nuevo = max((d for d in dias if d < actual), default=dias[0])
    else:
        nuevo = min((d for d in dias if d > actual), default=dias[-1])
    st.session_state["dia_operador"] = st.session_state["_copia_dia_operador"] = nuevo


ETQ_CORTA = {"vender": "VEN", "comprar": "COM", "retener": "RET", "evitar_compra": "EVI", "esperar": "ESP"}


def franja_botones(dia_df, hora_sel, tk):
    """Franja horaria clickeable: cada hora es un boton coloreado segun su senal;
    comparte estado con el deslizador de hora (session_state["hora_operador"])."""
    # la abreviatura (VEN/RET/ESP) va en el color de su senal; en claro, tintes mas fuertes que los de la
    # tarjeta para que las horas de accion se distingan de un vistazo (Juan, 15-oct)
    # colores de la franja por tema, con contraste WCAG >= 4,5 para la hora y la etiqueta (revision del 16-oct)
    color_etq = {"verde": tk["franja_txt_verde"], "rojo": tk["franja_txt_rojo"], "ambar": tk["franja_txt_gris"]}
    color_bg = {"verde": tk["franja_verde"], "rojo": tk["franja_rojo"], "ambar": tk["franja_gris"]}
    reglas = []
    for _, r in dia_df.iterrows():
        h = int(r["fecha_hora"].hour)
        color = SIGNIFICADO[r["senal"]][1]
        borde = f"2px solid {tk['tinta']}" if h == hora_sel else "1px solid transparent"
        # :hover/:focus/:active se pisan porque el boton de Streamlit trae su propio
        # estado (borde de color, sombra) que rompe el look de celda plana.
        reglas.append(
            f'.st-key-hora_c_{h} button, .st-key-hora_c_{h} button:hover, '
            f'.st-key-hora_c_{h} button:focus, .st-key-hora_c_{h} button:active {{ '
            f'background:{color_bg[color]} !important; color:{tk["tinta"]} !important; '
            f'border:{borde} !important; box-shadow:none !important; '
            f'border-radius:8px !important; font-weight:600 !important; '
            f'font-size:.66rem !important; padding:4px 0 !important; min-height:0 !important; '
            f'flex-direction:column !important; gap:0 !important; }}'
            f'.st-key-hora_c_{h} button p {{ color:{tk["franja_txt_gris"]} !important; white-space:pre; line-height:1.25; '
            f'font-size:.7rem !important; text-align:center; margin:0 !important; }}'
            f'.st-key-hora_c_{h} button p::after {{ content:"\\A {r["q50"]:.0f}"; '
            f'white-space:pre; color:{tk["tinta"]}; font-weight:700; font-size:.86rem; }}'
            f'.st-key-hora_c_{h} button::after {{ content:"{ETQ_CORTA[r["senal"]]}"; display:block; '
            f'color:{color_etq[color]}; font-weight:700; font-size:.72rem; letter-spacing:.04em; line-height:1.25; }}'
        )
        # marca de riesgo de salto: triangulo hacia abajo (desplome) o hacia arriba (pico)
        pd_, pp_ = r.get("p_desplome", np.nan), r.get("p_pico", np.nan)
        marca = ("▼", tk["rojo"]) if pd_ >= UMBRAL_ALERTA and pd_ >= pp_ else \
                (("▲", tk["verde"]) if pp_ >= UMBRAL_ALERTA else None)
        if marca:
            reglas.append(f'.st-key-hora_c_{h} {{ position:relative; }}'
                          f'.st-key-hora_c_{h}::after {{ content:"{marca[0]}"; position:absolute; top:-9px; '
                          f'left:50%; transform:translateX(-50%); font-size:.6rem; color:{marca[1]}; '
                          f'pointer-events:none; }}')
    st.markdown(f"<style>{''.join(reglas)}</style>", unsafe_allow_html=True)
    resumen = "; ".join(f'{int(r["fecha_hora"].hour):02d}:00, {es(r["q50"])} COP/kWh, {SIGNIFICADO[r["senal"]][0].lower()}'
                        for _, r in dia_df.iterrows())
    st.markdown(f'<div class="sr-only">Pronóstico por hora: {resumen}.</div>', unsafe_allow_html=True)

    filas_dia = list(dia_df.iterrows())
    for mitad in (filas_dia[:12], filas_dia[12:]):
        cols = st.columns(12, gap="small")
        for col, (_, r) in zip(cols, mitad):
            h = int(r["fecha_hora"].hour)
            with col:
                with st.container(key=f"hora_c_{h}"):
                    st.button(f"{h:02d}", key=f"hora_btn_{h}", on_click=_set_hora, args=(h,),
                              use_container_width=True)


def mini_linea(valores, color, linea, ancho=130, alto=38):
    """Mini grafica de linea en SVG (con area sombreada y linea de cero si cruza)."""
    v = np.asarray(valores, dtype=float)
    ok = ~np.isnan(v)
    if ok.sum() < 2:
        return ""
    lo, hi = np.nanmin(v), np.nanmax(v)
    rango = (hi - lo) or 1.0
    xs = np.linspace(3, ancho - 4, len(v))
    ys = alto - 4 - (v - lo) / rango * (alto - 8)
    pts = [(x, y) for x, y, o in zip(xs, ys, ok) if o]
    trazo = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{trazo} {pts[-1][0]:.1f},{alto} {pts[0][0]:.1f},{alto}"
    cero = ""
    if lo < 0 < hi:
        y0 = alto - 4 - (0 - lo) / rango * (alto - 8)
        cero = f'<line x1="0" x2="{ancho}" y1="{y0:.1f}" y2="{y0:.1f}" stroke="{linea}" stroke-dasharray="3 3"/>'
    ux, uy = pts[-1]
    return (f'<svg width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}" aria-hidden="true">{cero}'
            f'<polygon points="{area}" fill="{color}" fill-opacity="0.14"/>'
            f'<polyline points="{trazo}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>'
            f'<circle cx="{ux:.1f}" cy="{uy:.1f}" r="3" fill="{color}"/></svg>')


def mini_barras(valores, color, ancho=130, alto=38):
    """Mini grafica de barras en SVG; la ultima barra (el dia elegido) va resaltada."""
    v = np.asarray(valores, dtype=float)
    if len(v) == 0:
        return ""
    tope = max(v.max(), 1.0)
    paso = ancho / len(v)
    barras = []
    for i, x in enumerate(v):
        h = max(2.0, x / tope * (alto - 4))
        op = "1" if i == len(v) - 1 else "0.42"
        barras.append(f'<rect x="{i * paso + 1:.1f}" y="{alto - h:.1f}" width="{paso - 2:.1f}" '
                      f'height="{h:.1f}" rx="1.5" fill="{color}" fill-opacity="{op}"/>')
    return f'<svg width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}" aria-hidden="true">{"".join(barras)}</svg>'


def tarjetas_html(items):
    piezas = "".join(
        f'<div class="stat"><div class="stat-l">{l}</div>'
        f'<div class="stat-fila"><div class="stat-v">{v}</div>{g}</div>'
        f'<div class="stat-s">{s}</div></div>'
        for l, v, g, s in items
    )
    return f'<div class="tarjetas">{piezas}</div>'


# fechas de los ejes en formato colombiano (Plotly no trae los meses en espanol: van en numero)
FECHAS_EJE = [dict(dtickrange=[None, 86400000], value="%H:%M<br>%d/%m"),
              dict(dtickrange=[86400000, "M1"], value="%d/%m"),
              dict(dtickrange=["M1", None], value="%m/%Y")]


# zoom de las graficas (Juan, 16-oct): Ctrl + rueda o pellizco del touchpad acerca y aleja (el navegador manda el
# pellizco como rueda con Ctrl); la rueda sola sigue desplazando la pagina; arrastrar mueve la grafica; doble clic
# vuelve al encuadre inicial. El zoom de rueda de Plotly temblaba con el touchpad (estira la imagen y luego la
# redibuja en cada evento), asi que zoom_con_ctrl() lo hace aparte: suma los eventos y redibuja una vez por cuadro.
CONF_GRAF = {"displayModeBar": False, "scrollZoom": False, "doubleClick": "reset"}
# la semana y el backtest conservan la barra de herramientas de Plotly al pasar el cursor (sin el logo)
CONF_GRAF_BARRA = {**CONF_GRAF, "displayModeBar": "hover", "displaylogo": False}


def zoom_con_ctrl():
    st.html("""<script>
    const w = window;
    if (!w.__augurZoom) {
      w.__augurZoom = true;
      // zoom horizontal con Ctrl + rueda / pellizco, anclado en el cursor; un relayout por cuadro de animacion
      // (y desplazamiento lateral con dos dedos en el touchpad, sin Ctrl, con el cursor sobre la grafica)
      let g = null, acum = 0, acumX = 0, ancla = 0.5, pendiente = false;
      const aplicar = () => {
        pendiente = false;
        if (!g || !g._fullLayout || (!acum && !acumX)) return;
        const xa = g._fullLayout.xaxis, f = Math.exp(acum * 0.0022);
        let a = xa.r2l(xa.range[0]), b = xa.r2l(xa.range[1]);
        const corr = acumX / xa._length * (b - a);
        a += corr; b += corr;
        acum = 0; acumX = 0;
        const c = a + (b - a) * ancla;
        const x0 = c - (c - a) * f, x1 = c + (b - c) * f;
        w.Plotly.relayout(g, Object.assign({"xaxis.range[0]": xa.l2r(x0), "xaxis.range[1]": xa.l2r(x1)}, rangoY(g, x0, x1)))
          .then(() => pedirMas(g));
      };
      // eje Y ajustado a lo que se ve (asi al alejarse no se cortan los picos de los dias vecinos)
      const rangoY = (gd, x0, x1) => {
        const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
        if (ya.type === "category" || xa.type !== "date") return {};
        let lo = Infinity, hi = -Infinity;
        // puntos ya calculados por Plotly (sirven igual con lista de fechas o con inicio + paso)
        for (const cd of (gd.calcdata || [])) {
          const t = cd[0] && cd[0].trace;
          if (!t || !t.visible || t.visible === "legendonly" || t.type !== "scatter") continue;
          for (const q of cd) {
            if (q.x >= x0 && q.x <= x1 && isFinite(q.y)) { if (q.y < lo) lo = q.y; if (q.y > hi) hi = q.y; }
          }
        }
        if (!isFinite(lo)) return {};
        const pad = (hi - lo) * 0.08 || 10;
        return {"yaxis.range[0]": lo - pad, "yaxis.range[1]": hi + pad};
      };
      // al acercarse al borde de los datos cargados, pedir 30 dias mas (boton oculto de Streamlit)
      const pedirMas = (gd) => {
        const cont = gd.closest(".st-key-sec_dia_principal") ? "mas_dia" : gd.closest(".st-key-sec_semana") ? "mas_sem" : null;
        if (!cont || !gd.calcdata || (gd.__pidio && Date.now() - gd.__pidio < 4000)) return;
        let dmin = Infinity, dmax = -Infinity;
        for (const cd of gd.calcdata) for (const q of cd) if (isFinite(q.x) && isFinite(q.y)) {
          if (q.x < dmin) dmin = q.x; if (q.x > dmax) dmax = q.x; }
        const xa = gd._fullLayout.xaxis, a = xa.r2l(xa.range[0]), b = xa.r2l(xa.range[1]), margen = 12 * 3600000;
        const m = (gd.layout && gd.layout.meta) || {};
        const lado = a < dmin + margen && m.atras ? "atras" : b > dmax - margen && m.adelante ? "adelante" : null;
        if (!lado) return;
        const btn = document.querySelector(".st-key-" + cont + "_" + lado + " button");
        if (btn) {
          gd.__pidio = Date.now();
          // se guarda el encuadre para devolverlo cuando lleguen los datos nuevos (la pagina redibuja la grafica)
          w.__augurVista = w.__augurVista || {};
          w.__augurVista[cont] = {a: xa.range[0], b: xa.range[1], dmin: dmin, dmax: dmax, t: Date.now()};
          btn.click();
        }
      };
      setInterval(() => {
        for (const [cont, v] of Object.entries(w.__augurVista || {})) {
          if (Date.now() - v.t > 20000) { delete w.__augurVista[cont]; continue; }
          const gd = document.querySelector(".st-key-" + (cont === "mas_dia" ? "sec_dia_principal" : "sec_semana") + " .js-plotly-plot");
          if (!gd || !gd.calcdata) continue;
          let mn = Infinity, mx = -Infinity;
          for (const cd of gd.calcdata) for (const q of cd) if (isFinite(q.x) && isFinite(q.y)) {
            if (q.x < mn) mn = q.x; if (q.x > mx) mx = q.x; }
          if (mn === v.dmin && mx === v.dmax) continue;      // todavia no llegan los datos nuevos
          delete w.__augurVista[cont];
          const xa = gd._fullLayout.xaxis;
          w.Plotly.relayout(gd, Object.assign({"xaxis.range[0]": v.a, "xaxis.range[1]": v.b},
                                              rangoY(gd, xa.r2l(v.a), xa.r2l(v.b))));
        }
      }, 150);
      w.__augurPedirMas = pedirMas;
      // al arrastrar (pan) tambien se reajusta el eje Y
      setInterval(() => {
        document.querySelectorAll(".js-plotly-plot").forEach((gd) => {
          if (gd.__augurPan || !gd.on) return;
          gd.__augurPan = true;
          gd.on("plotly_relayout", (ev) => {
            if (!ev || ev["xaxis.range[0]"] === undefined || ev["yaxis.range[0]"] !== undefined) return;
            const xa = gd._fullLayout.xaxis, y = rangoY(gd, xa.r2l(xa.range[0]), xa.r2l(xa.range[1]));
            if (y["yaxis.range[0]"] !== undefined) w.Plotly.relayout(gd, y);
            pedirMas(gd);
          });
        });
      }, 700);
      w.addEventListener("wheel", (e) => {
        const gd = e.target.closest && e.target.closest(".js-plotly-plot");
        if (!gd) return;
        e.stopPropagation();                 // la rueda sola desplaza la pagina, no la grafica
        if (!w.Plotly || !gd._fullLayout) return;
        const lateral = !e.ctrlKey && Math.abs(e.deltaX) > Math.abs(e.deltaY);
        if (!e.ctrlKey && !lateral) return;  // desplazamiento vertical: la pagina
        if (gd._fullLayout.xaxis.type !== "date" && lateral) return;
        e.preventDefault();                  // evita el zoom de toda la pagina o el "atras" del navegador
        const xa = gd._fullLayout.xaxis, r = gd.getBoundingClientRect(), k = e.deltaMode === 1 ? 16 : 1;
        if (gd !== g) { g = gd; acum = 0; acumX = 0; }
        if (lateral) acumX += e.deltaX * k;
        else {
          ancla = Math.min(1, Math.max(0, (e.clientX - r.left - xa._offset) / xa._length));
          acum += e.deltaY * k;
        }
        if (!pendiente) { pendiente = true; requestAnimationFrame(aplicar); }
      }, {capture: true, passive: false});
      // clic en "Pronostico del dia": oprime el boton de esa hora en la franja (mismo efecto que hacer clic alli)
      setInterval(() => {
        const g = document.querySelector(".st-key-sec_dia_principal .js-plotly-plot");
        if (!g || g.__augurClic || !g.on) return;
        g.__augurClic = true;
        g.on("plotly_click", (ev) => {
          const p = ev && ev.points && ev.points[0];
          if (!p) return;
          // la x llega como texto o como milisegundos (fechas sin zona horaria: se leen en UTC)
          const fx = typeof p.x === "number" ? new Date(p.x).toISOString() : String(p.x).replace(" ", "T");
          const dsel = g.layout && g.layout.meta && g.layout.meta.dia;
          if (dsel && fx.slice(0, 10) !== dsel) return;   // solo horas del dia elegido
          const h = parseInt(fx.slice(11, 13), 10);
          const b = document.querySelector(".st-key-hora_btn_" + h + " button");
          if (b) b.click();
        });
      }, 700);
    }
    </script>""", unsafe_allow_javascript=True, width="content")


def estilo_figura(fig, tk, alto, titulo, margen_leyenda=64, fechas=True):
    # La leyenda se ancla al borde inferior del lienzo (no del area de datos) y se le
    # reserva margen propio, para que no se monte sobre las etiquetas del eje X.
    fig.update_layout(
        height=alto, title=dict(text=titulo, x=0.01, font=dict(size=15, color=tk["tinta"])),
        paper_bgcolor=tk["panel"], plot_bgcolor=tk["panel"], font=dict(color=tk["gris"], family=FUENTE),
        legend=dict(orientation="h", yref="container", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                    font=dict(color=tk["texto"], size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=46, l=10, r=10, b=margen_leyenda), hovermode="x unified", dragmode="pan",
        hoverlabel=dict(bgcolor=tk["panel"], bordercolor=tk["linea"], font=dict(color=tk["tinta"])),
        yaxis_title="COP/kWh", separators=",.",   # coma decimal y punto de miles en ejes y etiquetas
    )
    if fechas:
        fig.update_xaxes(tickformatstops=FECHAS_EJE, hoverformat="%d/%m/%Y %H:%M")
    for eje in (fig.update_xaxes, fig.update_yaxes):
        eje(automargin=True, gridcolor=tk["linea"], zerolinecolor=tk["linea"], linecolor=tk["linea"],
            tickfont=dict(color=tk["gris"]), title_font=dict(color=tk["gris"]))
    return fig


def trazar_banda(fig, df, tk):
    fig.add_trace(go.Scatter(x=df["fecha_hora"], y=df["q90"], line=dict(width=0), legendgroup="banda",
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["fecha_hora"], y=df["q10"], line=dict(width=0), fill="tonexty",
                             fillcolor=tk["banda"], name="Banda [q10, q90]", legendgroup="banda", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["fecha_hora"], y=df["q50"], name="Pronóstico (q50)",
                             line=dict(color=tk["q50"], width=1.6, dash="dot"),
                             hovertemplate="%{y:,.0f} COP/kWh"))
    fig.add_trace(go.Scatter(x=df["fecha_hora"], y=df["real"], name="Precio real",
                             line=dict(color=tk["tinta"], width=1.5),
                             hovertemplate="%{y:,.0f} COP/kWh"))


def grafico_precio(df, tk, titulo, alto=340, marca_x=None):
    """Precio real, pronostico y banda, con las horas de accion marcadas."""
    fig = go.Figure()
    trazar_banda(fig, df, tk)
    for senal, (etiqueta, color) in SIGNIFICADO.items():
        if senal == "esperar":
            continue
        sub = df[df["senal"] == senal]
        if len(sub):
            fig.add_trace(go.Scatter(
                x=sub["fecha_hora"], y=sub["real"], mode="markers", name=etiqueta.capitalize(),
                marker=dict(color=tk[color], size=7), hoverinfo="skip",
            ))
    if marca_x is not None:
        fig.add_vline(x=marca_x, line_dash="dot", line_color=tk["gris"])
    return estilo_figura(fig, tk, alto, titulo)


def vista_reciente(fig, df, tk, dias=30):
    """Periodos largos hora a hora se vuelven una mancha: la grafica abre en los ultimos `dias` y trae botones
    para ver 7 dias, 30, 90 o todo el rango (los datos y las cifras de arriba no cambian, solo el encuadre)."""
    ini, fin = df["fecha_hora"].min(), df["fecha_hora"].max()
    if (fin - ini).days <= dias + 15:
        return fig
    fig.update_xaxes(range=[fin - pd.Timedelta(days=dias), fin], rangeselector=dict(
        buttons=[dict(count=7, step="day", stepmode="backward", label="7 días"),
                 dict(count=30, step="day", stepmode="backward", label="30 días"),
                 dict(count=90, step="day", stepmode="backward", label="90 días"),
                 dict(step="all", label="Todo")],
        x=0, y=1.02, yanchor="bottom", bgcolor=tk["neutro_bg"], activecolor=tk["control_borde"],
        bordercolor=tk["linea"], font=dict(color=tk["tinta"], size=11)))
    fig.update_layout(margin_t=40)
    return fig


NOMBRE_CORTO = {"Banda [q10, q90]": "Banda", "Pronóstico (q50)": "Pronóstico",
                "Posibilidad de desplome": "Pos. desplome", "Posibilidad de pico": "Pos. pico"}


def valores_y(tr):
    """y de una traza como arreglo de numeros. Al copiar una figura (go.Figure(fig), p. ej. la grafica ampliada) Plotly
    guarda los arreglos en su formato binario {"dtype": "f4", "bdata": base64}; sin decodificarlo, pandas fallaba con
    "could not convert string to float: 'f4'" (20-oct-2026)."""
    y = tr.y
    if isinstance(y, dict) and "bdata" in y:
        import base64
        y = np.frombuffer(base64.b64decode(y["bdata"]), dtype=np.dtype(y["dtype"]))
    return pd.to_numeric(pd.Series(y), errors="coerce")


def leyenda_en_columnas(fig, ancho=72, columnas_min=2):
    """Entradas de la leyenda de igual ancho en pixeles, asi quedan en columnas alineadas aunque ocupen varias filas.
    Las series vacias (p. ej. el precio real de un dia que XM aun no publica) salen de la leyenda."""
    for tr in fig.data:
        tr.name = NOMBRE_CORTO.get(tr.name, tr.name)
        if tr.name and tr.y is not None and valores_y(tr).isna().all():
            tr.showlegend = False
    n = sum(1 for tr in fig.data if tr.showlegend is not False and tr.name)
    filas = -(-n // columnas_min)
    fig.update_layout(legend=dict(entrywidth=ancho, entrywidthmode="pixels", itemwidth=30, xref="container", x=0.02,
                                  font_size=10.5),
                      margin_b=34 + 22 * filas)
    return fig


@st.dialog("Pronóstico del día", width="large")
def ampliar_grafica(fig):
    leyenda_en_columnas(fig, ancho=110)
    fig.update_layout(height=500, margin_b=56)
    st.markdown('<div class="pista-ley">&#128161; Variables de la gráfica: clic para ocultar o mostrar &middot; doble clic para ver solo una &middot; '
                'arrastra sobre la gráfica para acercar y haz doble clic en ella para volver.</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch", theme=None, config=CONF_GRAF, key="graf_ampliada")


def pista_variables(cont, extra=""):
    cont.markdown('<div class="pista-ley">&#128161; Variables: clic para ocultar o mostrar &middot; doble clic para ver '
                  f'solo una &middot; Ctrl + rueda o pellizco para acercar &middot; arrastra o desliza dos dedos a los lados para moverla{extra}.</div>', unsafe_allow_html=True)


def sin_titulo(fig):
    # el titulo va en la tarjeta; uirevision: lo que se oculta con la leyenda se conserva al cambiar de hora o de dia
    fig.update_layout(title_text="", margin_t=12, uirevision="fija")
    return fig


# ejes de fecha de la grafica del dia: solo la hora de cerca, y la fecha al alejarse (Juan, 16-oct)
HORAS_EJE = [dict(dtickrange=[None, 3 * 3600000], value="%H:%M"),
             dict(dtickrange=[3 * 3600000, 86400000], value="%H:%M<br>%d/%m"),
             dict(dtickrange=[86400000, None], value="%d/%m")]


def fechas_compactas(fig):
    """Series horarias sin lista de fechas: inicio (x0, en texto) + paso de 1 hora (dx), y precios en float32. La
    grafica con 4 meses hora a hora pesa poco. Las fechas van en texto a proposito: en milisegundos Plotly las
    corria segun la zona horaria del navegador (5 h en Colombia; Juan, 16-oct)."""
    for tr in fig.data:
        if tr.x is None or len(tr.x) < 48 or tr.y is None:
            continue                                   # marcadores sueltos: se dejan como estan
        serie = pd.Series(pd.to_numeric(pd.Series(tr.y), errors="coerce").to_numpy(),
                          index=pd.to_datetime(pd.Series(tr.x)).to_numpy())
        serie = serie[~serie.index.duplicated()].sort_index()
        serie = serie.reindex(pd.date_range(serie.index.min(), serie.index.max(), freq="h"))
        tr.x, tr.x0, tr.dx = None, f"{serie.index[0]:%Y-%m-%d %H:%M:%S}", 3600000
        tr.y = serie.to_numpy("float32")
    fig.update_xaxes(type="date")
    return fig


def grafico_dia(v, bajo_s, alto_s, tk):
    """Pronostico de varios dias (el bloque cargado) con los umbrales del motor; el encuadre en el dia elegido y la
    linea de la hora se ponen despues (encuadrar), para que la base se pueda guardar en cache."""
    fig = go.Figure()
    trazar_banda(fig, v, tk)
    fig.add_trace(go.Scatter(x=v["fecha_hora"], y=alto_s.reindex(v.index), name="Umbral caro",
                             line=dict(color=tk["rojo"], width=1, dash="dash"),
                             hovertemplate="%{y:,.0f} COP/kWh"))
    fig.add_trace(go.Scatter(x=v["fecha_hora"], y=bajo_s.reindex(v.index), name="Umbral barato",
                             line=dict(color=tk["verde"], width=1, dash="dash"),
                             hovertemplate="%{y:,.0f} COP/kWh"))
    for col, simbolo, color, nombre in (("p_desplome", "triangle-down", tk["rojo"], "Posibilidad de desplome"),
                                        ("p_pico", "triangle-up", tk["verde"], "Posibilidad de pico")):
        if col in v and (v[col] >= UMBRAL_ALERTA).any():
            m = v[v[col] >= UMBRAL_ALERTA]
            fig.add_trace(go.Scatter(x=m["fecha_hora"], y=m["q50"], mode="markers", name=nombre,
                                     marker=dict(symbol=simbolo, size=11, color=color, line=dict(width=1, color=tk["panel"])),
                                     customdata=m[col] * 100, hovertemplate=nombre + ": %{customdata:.0f} %<extra></extra>"))
    fig = estilo_figura(fig, tk, 340, "Pronóstico del día", margen_leyenda=96, fechas=False)
    fig.update_xaxes(tickformatstops=HORAS_EJE, hoverformat="%d/%m/%Y %H:%M")
    return fechas_compactas(fig)


# ---- graficas en cache (Juan, 16-oct: los botones de dia y hora tardaban ~3 s porque cada clic reconstruia las
# graficas). Los datos se cargan en bloques fijos de 120 dias, que no cambian al pasar de un dia al siguiente; en cada
# clic solo se cambian el encuadre, la linea de la hora y el recuadro del mapa de calor.
def bloque_dias(dia, clave=None):
    """Bloque de ~30 dias alrededor del dia (fijo para 10 dias seguidos, asi sirve la cache). Si la persona arrastra
    la grafica hasta el borde, el script de la pagina oprime un boton oculto que suma 30 dias en esa direccion
    (cargar_mas); al cambiar de dia vuelve al bloque corto, que es el que se dibuja rapido."""
    k = (pd.Timestamp(dia) - pd.Timestamp("2019-01-01")).days // 10
    ini = pd.Timestamp("2019-01-01") + pd.Timedelta(days=10 * k - 10)
    atras, adelante = 0, 0
    if clave and st.session_state.get(clave, (None,))[0] == ini:
        atras, adelante = st.session_state[clave][1:]
    return ini - pd.Timedelta(days=atras), ini + pd.Timedelta(days=30 + adelante)


def cargar_mas(clave, dia, lado):
    k = (pd.Timestamp(dia) - pd.Timestamp("2019-01-01")).days // 10
    ini = pd.Timestamp("2019-01-01") + pd.Timedelta(days=10 * k - 10)
    _, atras, adelante = st.session_state.get(clave, (None, 0, 0)) if st.session_state.get(clave, (None,))[0] == ini \
        else (None, 0, 0)
    st.session_state[clave] = (ini, atras + 30 * (lado < 0), adelante + 30 * (lado > 0))


def botones_cargar_mas(clave, dia):
    """Botones ocultos que oprime el script de la pagina al llegar al borde de los datos cargados."""
    with st.container(key=f"oculto_{clave}"):
        st.button("Cargar días anteriores", key=f"{clave}_atras", on_click=cargar_mas, args=(clave, dia, -1))
        st.button("Cargar días siguientes", key=f"{clave}_adelante", on_click=cargar_mas, args=(clave, dia, 1))


@st.cache_data(max_entries=24, show_spinner=False)
def base_dia(horizonte, rol, metodo, intradia, tema, b_ini, b_fin):
    df = serie_con_senal(horizonte, rol, metodo, intradia=intradia)
    bajo_s, alto_s = umbrales_serie(df, metodo, cargar_precio_historico(), 25, 75)
    v = df[(df["fecha_hora"] >= b_ini) & (df["fecha_hora"] < b_fin)]
    return leyenda_en_columnas(sin_titulo(grafico_dia(v, bajo_s, alto_s, TEMAS[tema]))).to_dict()


@st.cache_data(max_entries=24, show_spinner=False)
def base_semana(horizonte, rol, metodo, intradia, tema, b_ini, b_fin):
    df = serie_con_senal(horizonte, rol, metodo, intradia=intradia)
    v = df[(df["fecha_hora"] >= b_ini) & (df["fecha_hora"] < b_fin)]
    return fechas_compactas(sin_titulo(grafico_precio(v, TEMAS[tema], "La semana alrededor del día elegido"))).to_dict()


@st.cache_data(max_entries=24, show_spinner=False)
def base_calor(horizonte, rol, metodo, intradia, tema, dias, horas, modo):
    df = serie_con_senal(horizonte, rol, metodo, intradia=intradia)
    return grafico_calor(df, list(dias), list(horas), None, None, modo, TEMAS[tema]).to_dict()


def encuadrar(base, x=None, y=None, rev=None, formas=(), meta=None):
    """Copia la base en cache y le pone encuadre, revision de ejes (para conservar o reiniciar el zoom) y formas."""
    fig = {"data": base["data"], "layout": copy.deepcopy(base["layout"])}
    lay = fig["layout"]
    for eje, rango in (("xaxis", x), ("yaxis", y)):
        lay.setdefault(eje, {})
        if rango is not None:
            lay[eje]["range"] = [str(r) if isinstance(r, pd.Timestamp) else r for r in rango]
        if rev is not None:
            lay[eje]["uirevision"] = rev
    lay["shapes"] = list(lay.get("shapes", [])) + list(formas)
    if meta is not None:
        lay["meta"] = meta
    return fig


def meta_bloque(df, bloque, dia=None):
    """Lo que el script de la pagina necesita saber: el dia elegido (clic) y si hay datos fuera del bloque cargado."""
    return {"dia": f"{pd.Timestamp(dia):%Y-%m-%d}" if dia is not None else "",
            "atras": bool(df["fecha_hora"].min() < bloque[0]), "adelante": bool(df["fecha_hora"].max() >= bloque[1])}


def linea_vertical(x, color, punteada=False):
    return {"type": "line", "xref": "x", "yref": "y domain", "x0": str(x), "x1": str(x), "y0": 0, "y1": 1,
            "line": {"color": color, "width": 1, **({"dash": "dot"} if punteada else {})}}


def rango_y(*series):
    vals = pd.concat([pd.Series(x) for x in series]).dropna()
    if not len(vals):
        return None
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * .08 or 10
    return [lo - pad, hi + pad]


def metadatos_html(periodo, contenido):
    """Pie de cada imagen de decision con los metadatos que exige el Anexo 1 (unidad, periodo,
    confianza y version del modelo)."""
    cob = es(meta.get("cobertura_medida_pct", "n/d"), 1)
    return (f'<div class="meta"><b>{contenido}</b> &middot; Unidad: COP/kWh &middot; Periodo: {periodo} '
            f'&middot; Confianza: banda [q10, q90], cobertura medida {cob} % &middot; Modelo de {horizonte}, '
            f'versión {meta.get("generado", "n/d")}</div>')


CODIGO_SENAL = {"evitar_compra": 0, "retener": 0, "esperar": 1, "comprar": 2, "vender": 2}


# escala de color del mapa de calor del Precio de Bolsa Nacional en Sinergox (XM): azul en el minimo del periodo,
# crema en el valor medio y naranja-rojo en el maximo
ESCALA_XM = [[0.0, "#2A7FC0"], [0.22, "#8BBCD6"], [0.45, "#E4ECDF"], [0.5, "#F4EED6"], [0.62, "#F6CFA4"],
             [0.8, "#EF8A45"], [1.0, "#E3511B"]]


def grafico_calor(df, dias, horas, dia_sel, hora_sel, modo, tk):
    """Mapa de calor dia-hora: filas = dias, columnas = horas; color = senal del motor o precio esperado."""
    sub = df[df["fecha_hora"].dt.date.isin(dias) & df["fecha_hora"].dt.hour.isin(horas)].copy()
    sub["d"] = sub["fecha_hora"].dt.strftime("%d/%m")
    sub["h"] = sub["fecha_hora"].dt.hour
    orden = [pd.Timestamp(x).strftime("%d/%m") for x in dias]
    piv = lambda col: sub.pivot(index="d", columns="h", values=col).reindex(index=orden, columns=horas)
    # siempre queda una linea entre dias y entre horas para que cada celda se distinga
    hueco_y, hueco_x = (2 if len(dias) <= 45 else 1), (2 if len(horas) <= 30 else 1)
    etiqueta = {k: v[0].capitalize() for k, v in SIGNIFICADO.items()}
    sub["etq"] = sub["senal"].map(etiqueta)
    # XM publica el precio real con ~3 dias de rezago: los dias recientes y futuros aun no lo tienen
    sub["real_txt"] = sub["real"].map(lambda v: "aún sin publicar" if pd.isna(v) else es(v))
    datos = np.dstack([piv("q50").to_numpy(), piv("real_txt").to_numpy(), piv("q10").to_numpy(),
                       piv("q90").to_numpy(), piv("etq").to_numpy()])
    plantilla = ("%{y} · %{x}:00 h<br>%{customdata[4]}<br>Pronóstico: %{customdata[0]:,.0f} COP/kWh"
                 "<br>Banda: %{customdata[2]:,.0f} – %{customdata[3]:,.0f}<br>Real: %{customdata[1]}"
                 "<extra></extra>")
    if modo == "Señal del motor":
        z = piv("senal").apply(lambda c: c.map(CODIGO_SENAL)).to_numpy(dtype=float)
        escala = [[0, tk["calor_rojo"]], [1 / 3, tk["calor_rojo"]], [1 / 3, tk["calor_ambar"]],
                  [2 / 3, tk["calor_ambar"]], [2 / 3, tk["calor_verde"]], [1, tk["calor_verde"]]]
        fig = go.Figure(go.Heatmap(z=z, x=horas, y=orden, zmin=0, zmax=2, colorscale=escala,
                                   showscale=False, xgap=hueco_x, ygap=hueco_y, customdata=datos,
                                   hovertemplate=plantilla, opacity=0.85))
    else:
        fig = go.Figure(go.Heatmap(z=piv("q50").to_numpy(dtype=float), x=horas, y=orden,
                                   colorscale=ESCALA_XM, xgap=hueco_x, ygap=hueco_y, customdata=datos,
                                   hovertemplate=plantilla,
                                   colorbar=dict(title=dict(text="COP/kWh", font=dict(color=tk["gris"])),
                                                 tickfont=dict(color=tk["gris"]), thickness=12)))
    if modo == "Señal del motor" and "sin_umbral" in sub and sub["sin_umbral"].any():
        # se pintan aparte para no confundirlas con un "esperar" decidido por el motor
        su = sub.pivot(index="d", columns="h", values="sin_umbral").reindex(index=orden, columns=horas)
        fig.add_trace(go.Heatmap(z=np.where(su.fillna(False).to_numpy(bool), 1.0, np.nan), x=horas, y=orden,
                                 colorscale=[[0, tk["linea"]], [1, tk["linea"]]], showscale=False,
                                 xgap=hueco_x, ygap=hueco_y, customdata=datos,
                                 hovertemplate="%{y} · %{x}:00 h<br>Sin umbral todavía (faltan datos previos)"
                                               "<br>Pronóstico: %{customdata[0]:,.0f} COP/kWh<extra></extra>"))
    if dia_sel in dias and hora_sel in horas:
        y0 = orden.index(pd.Timestamp(dia_sel).strftime("%d/%m"))
        fig.add_shape(type="rect", x0=hora_sel - 0.5, x1=hora_sel + 0.5, y0=y0 - 0.5, y1=y0 + 0.5,
                      line=dict(color=tk["tinta"], width=2.5))
    # hasta 1100 px; con muchos dias la figura crece para que cada fila mida al menos 9 px
    alto = int(np.clip(150 + 24 * len(dias), 250, max(1100, 150 + 9 * len(dias))))
    fig = estilo_figura(fig, tk, alto, f"Mapa de calor día-hora · {modo.lower()}", margen_leyenda=60, fechas=False)
    fig.update_layout(hovermode="closest", yaxis_title=None, xaxis_title="Hora del día")
    fig.update_xaxes(dtick=1 if len(horas) <= 12 else 2, showgrid=False, range=[horas[0] - 0.5, horas[-1] + 0.5])
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category",
                     nticks=min(len(dias), 40))
    return fig


def tabla_metodos_html(tabla, metodo_activo, metodo_sugerido):
    filas = []
    for _, r in tabla.iterrows():
        m = r["metodo"]
        ventaja = "—" if pd.isna(r["ventaja_cop_kwh"]) else es(r['ventaja_cop_kwh'], 1, True)
        etq_val = '<span class="etq etq-si">válido</span>' if r["valido"] else '<span class="etq etq-no">fuera de 10-40 %</span>'
        sug = '<span class="etq etq-sug">sugerido</span>' if m == metodo_sugerido else ""
        clase = ' class="activo"' if m == metodo_activo else ""
        filas.append(f"<tr{clase}><td>{m}{sug}</td><td>{ventaja}</td>"
                     f"<td>{es(r['frecuencia_accion'] * 100, 1)} %</td><td>{es(r['horas_accion'])}</td>"
                     f"<td>{etq_val}</td></tr>")
    return ('<table class="tabla-met"><thead><tr><th>Método</th><th>Ventaja (COP/kWh)</th>'
            '<th>Frecuencia de acción</th><th>Horas de acción</th><th>Rango 10-40 %</th></tr></thead>'
            f'<tbody>{"".join(filas)}</tbody></table>')


# --------------------------------------------------------------------------
# Barra lateral y tema
# --------------------------------------------------------------------------
def logo_svg(tk, tam):
    """Logo de Augur: tres aves que suben y crecen (gris, tinta, ambar), el augurio favorable como tendencia."""
    def ave(x, y, k, color, grosor):
        return (f'<path d="M{x - 6 * k} {y} C{x - 4 * k} {y - 4 * k} {x - 1.5 * k} {y - 4 * k} {x} {y + k} '
                f'C{x + 1.5 * k} {y - 4 * k} {x + 4 * k} {y - 4 * k} {x + 6 * k} {y}" stroke="{color}" '
                f'stroke-width="{grosor}"/>')
    return (f'<svg width="{tam}" height="{tam}" viewBox="0 0 48 48" fill="none" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">'
            + ave(10, 37, .8, tk["logo_tenue"], 2.8) + ave(23, 26, 1.05, tk["tinta"], 3.4)
            + ave(36, 13, 1.3, tk["acento"], 3.8) + '</svg>')


def marca_html(tk, tam, bajada, clase):
    return (f'<div class="marca {clase}">{logo_svg(tk, tam)}<div><div class="marca-nom">Augur</div>'
            f'<div class="marca-sub">{bajada}</div></div></div>')


if "modo_oscuro" not in st.session_state:
    st.session_state["modo_oscuro"] = st.query_params.get("tema") == "oscuro"


def clave_acceso():
    """Clave del acceso (secreto `clave_acceso` en Streamlit Cloud: Settings > Secrets). Sin secreto la app queda
    abierta, como antes; para cerrarla o cambiar quién entra basta con poner o cambiar la clave, sin tocar el código."""
    try:
        return str(st.secrets.get("clave_acceso", "") or "")
    except Exception:          # sin archivo de secretos (p. ej. local)
        return ""


def puerta_acceso():
    clave = clave_acceso()
    if not clave or st.session_state.get("acceso_ok") == clave:
        return
    tk_p = TEMAS["oscuro" if st.session_state["modo_oscuro"] else "claro"]
    st.markdown(f"<style>{CSS_BASE.substitute(tk_p)}{CSS_OSCURO_EXTRA if st.session_state['modo_oscuro'] else ''}"
                "[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'] { display: none; }</style>",
                unsafe_allow_html=True)
    _, centro, _ = st.columns([1, 1.1, 1])
    with centro.container(key="sec_acceso"):
        st.markdown(marca_html(tk_p, 46, "Motor de decisión", "marca-cab") +
                    '<div class="sec-tit" style="margin-top:14px">Acceso restringido</div><div class="sec-sub">'
                    'El dashboard está en evaluación. Ingresa la clave que te compartió el equipo del proyecto.</div>',
                    unsafe_allow_html=True)
        with st.form("form_acceso", border=False):
            intento = st.text_input("Clave", type="password", placeholder="Clave de acceso")
            entrar = st.form_submit_button("Entrar", type="primary", width="stretch")
        if entrar:
            if hmac.compare_digest(intento.strip(), clave):
                st.session_state["acceso_ok"] = clave
                st.rerun()
            st.error("Clave incorrecta.")
    st.stop()


puerta_acceso()
zoom_con_ctrl()

with st.sidebar:
    oscuro = st.session_state["modo_oscuro"]
    tk = TEMAS["oscuro" if oscuro else "claro"]
    st.markdown(marca_html(tk, 38, "Motor de decisión", "marca-lateral"), unsafe_allow_html=True)
    oscuro = st.toggle(":material/dark_mode: Modo oscuro", key="modo_oscuro")
    st.query_params["tema"] = "oscuro" if oscuro else "claro"
    TEMA = "oscuro" if oscuro else "claro"
    tk = TEMAS[TEMA]
    st.markdown(f"<style>{CSS_BASE.substitute(tk)}{CSS_OSCURO_EXTRA if oscuro else ''}</style>",
                unsafe_allow_html=True)

    vistas = ["Operador", "Analista", "Metodología"]
    # la URL solo fija la vista inicial; despues manda session_state. Si el indice se recalculara desde la
    # URL en cada ejecucion, Streamlit veria un control nuevo y el primer clic para volver se perderia
    if "vista_sel" not in st.session_state:
        st.session_state["vista_sel"] = {"analista": "Analista", "metodologia": "Metodología"}.get(
            str(st.query_params.get("vista", "")).lower().replace("í", "i"), "Operador")
    vista = st.radio("Vista", vistas, key="vista_sel",
                     captions=["Qué hago ahora", "Controles y backtest", "Desempeño y cómo funciona"])
    st.query_params["vista"] = vista.lower().replace("í", "i")
    st.divider()
    rol = st.selectbox(
        "Rol", ["generador", "comercializador"],
        help=("Generador: vende energía, busca precios altos. "
              "Comercializador: compra energía, busca precios bajos."),
    )
    horizonte = st.selectbox("Horizonte de pronóstico", ["24h", "72h"])

bandas, meta = cargar_bandas(horizonte)
if bandas is None:
    st.error(
        f"No pude cargar la fuente de pronóstico para {horizonte}: {meta.get('error')}\n\n"
        "Revisa `data/processed/resultados/fuentes_pronostico.json` y corre "
        "`notebooks/12_motor_decision_Rafa.ipynb` si hace falta regenerar el CSV."
    )
    st.stop()

precio_hist = cargar_precio_historico()
# filtro intradia del motor: activo por defecto; se cambia desde la vista Analista y se recuerda en la sesion
INTRADIA = st.session_state.get("_filtro_intradia", True)


def _guardar_intradia():
    st.session_state["_filtro_intradia"] = st.session_state["filtro_intradia_w"]


with st.sidebar:
    st.divider()
    estado = "Modelo activo" if meta.get("calibrado") else "Modelo activo (sin calibrar)"
    st.markdown(
        f'<div class="modelo"><b>{estado}</b><br><span>Pronóstico de {horizonte}, versión '
        f'{meta.get("generado", "n/d")}</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Cobertura medida de la banda: {es(meta.get('cobertura_medida_pct', '-'), 1)} %")
    # actualizacion diaria (scripts_experimento/actualizar_produccion.py, GitHub Actions)
    try:
        est = json.loads((RAIZ / "data/processed/resultados/produccion/estado.json").read_text(encoding="utf-8"))
        st.caption(f"Actualizado {pd.Timestamp(est['actualizado']):%d/%m/%Y %H:%M} · precio real de XM hasta el "
                   f"{pd.Timestamp(est['precio_real_hasta']):%d/%m %H:%M} · pronóstico hasta el "
                   f"{pd.Timestamp(est['pronostico_24h_hasta']):%d/%m %H:%M}")
    except (OSError, ValueError, KeyError):
        pass

st.markdown(
    f'<div class="cab-card">{marca_html(tk, 46, "Motor de decisión", "marca-cab")}'
    f'<div><h1 class="tit">Motor de decisión &middot; precio de bolsa de energía '
    f'<span class="pill">Colombia &middot; SIN</span></h1>'
    f'<div class="bajada">Pronóstico hora a hora del precio de bolsa de energía en Colombia y recomendación de cuándo '
    f'comprar, vender o esperar. Herramienta de apoyo a la decisión; no es asesoría financiera.</div></div>'
    f'<div class="der">Pronóstico de {horizonte}<br>versión <b>{meta.get("generado", "n/d")}</b></div></div>',
    unsafe_allow_html=True,
)


# ==========================================================================
# VISTA OPERADOR
# ==========================================================================
def vista_operador():
    tabla, metodo, es_valido = comparacion_estable(horizonte, rol, INTRADIA)
    df = serie_con_senal(horizonte, rol, metodo, intradia=INTRADIA)

    # Solo se ofrecen dias con las 24 horas: segun el modelo, el pronostico puede cubrir 01:00 a 00:00
    # del dia siguiente y dejar el primer o el ultimo dia con una sola hora.
    horas_por_dia = df["fecha_hora"].dt.date.value_counts()
    completos = sorted(horas_por_dia[horas_por_dia >= 24].index) or sorted(horas_por_dia.index)
    dia_min, dia_max = completos[0], completos[-1]

    # el dia se guarda en session_state (con copia, porque Streamlit borra el estado de un control que no
    # se dibuja, p. ej. al pasar por la vista Analista) y se acota al rango del horizonte activo
    elegido = st.session_state.get("dia_operador", st.session_state.get("_copia_dia_operador"))
    if elegido is None:
        st.session_state["dia_operador"] = dia_max
    elif not dia_min <= elegido <= dia_max:
        st.session_state["dia_operador"] = min(max(elegido, dia_min), dia_max)
    elif "dia_operador" not in st.session_state:
        st.session_state["dia_operador"] = elegido

    # orden pedido por Juan (16-oct): dia y hora juntos; "Ultimo" separado y de otro color, porque se confundia
    # con "Siguiente"
    c1, c2, c_ant, c_sig, c_ult = st.columns([1.5, 1.05, .62, .62, .78], vertical_alignment="bottom")
    with c1:
        dia = st.date_input("Día operativo", min_value=dia_min, max_value=dia_max, key="dia_operador",
                            format="DD/MM/YYYY")
        st.session_state["_copia_dia_operador"] = dia
    with c_ant:
        st.button("Anterior", icon=":material/chevron_left:", key="dia_ant", width="stretch",
                  on_click=_mover_dia, args=(-1, completos), disabled=dia <= dia_min)
    with c_sig:
        st.button("Siguiente", icon=":material/chevron_right:", key="dia_sig", width="stretch",
                  on_click=_mover_dia, args=(1, completos), disabled=dia >= dia_max)
    with c_ult:
        st.button("Ir al último", icon=":material/last_page:", key="dia_ult", width="stretch",
                  on_click=_mover_dia, args=("ultimo", completos), disabled=dia >= dia_max)
    with c2:
        # como el dia: Streamlit borra el estado del control al pasar por la vista Analista, asi que se guarda una copia
        if "hora_operador" not in st.session_state:
            st.session_state["hora_operador"] = st.session_state.get("_copia_hora_operador", 8)
        hora = st.number_input("Hora del día", 0, 23, step=1, key="hora_operador")
        st.session_state["_copia_hora_operador"] = hora

    dia_df = df[df["fecha_hora"].dt.date == dia]
    if dia_df.empty:
        st.warning("No hay pronóstico para ese día.")
        return

    objetivo = dia_df[dia_df["fecha_hora"].dt.hour == hora]
    fila = objetivo.iloc[0] if len(objetivo) else dia_df.iloc[0]
    idx = fila.name
    senal = fila["senal"]
    q50 = float(fila["q50"])

    # confianza: ancho de la banda RELATIVO al precio esperado, frente a las horas ANTERIORES
    # (sin mirar el futuro; el mismo criterio del filtro de banda ancha del motor)
    rel = ancho_relativo(df)
    previas = rel[df["fecha_hora"] < fila["fecha_hora"]].to_numpy()
    ancho_t0 = float(fila["q90"] - fila["q10"])
    conf, conf_pct = nivel_confianza(float(rel.loc[idx]), previas if len(previas) else rel.to_numpy())

    bajo_s, alto_s = umbrales_serie(df, metodo, precio_hist, 25, 75)
    bajo_val, alto_val = float(bajo_s.loc[idx]), float(alto_s.loc[idx])

    # Clasificacion "cruda" del precio (antes del filtro de banda ancha), para
    # explicar el caso en que 'banda'/'hibrido' fuerzan esperar pese a un precio extremo.
    if np.isnan(bajo_val) or np.isnan(alto_val):
        crudo = "normal"
    elif q50 <= bajo_val:
        crudo = "bajo"
    elif q50 >= alto_val:
        crudo = "alto"
    else:
        crudo = "normal"

    titulo, color = SIGNIFICADO[senal]
    if senal == "esperar" and metodo in ("banda", "hibrido") and crudo != "normal":
        frase = (
            f"El precio esperado está {crudo}, pero el pronóstico de esta hora es muy incierto "
            "(banda ancha). El motor prefiere esperar antes que arriesgarse con una señal poco confiable."
        )
    else:
        frase = FRASE[senal]
    lo, hi = np.nanpercentile(precio_hist, [2, 98])

    if np.isnan(bajo_val) or np.isnan(alto_val):
        gauge_html = (
            '<div class="stat-s" style="margin-top:14px">Umbrales móviles aún sin definir '
            '(los métodos rodante e híbrido necesitan 30 días de historia antes de operar).</div>'
        )
    else:
        pb, pa, pq = (pos_en_barra(x, lo, hi) for x in (bajo_val, alto_val, q50))
        # tramos: barato | normal | caro, con el color de la accion que le corresponde a cada rol
        c_bar, c_car = ((tk["verde"], tk["rojo"]) if rol == "comercializador" else (tk["rojo"], tk["verde"]))
        grad = (f"linear-gradient(90deg,{c_bar} 0%,{c_bar} {pb:.1f}%,"
                f"{tk['neutro_bg']} {pb:.1f}%,{tk['neutro_bg']} {pa:.1f}%,"
                f"{c_car} {pa:.1f}%,{c_car} 100%)")
        a_bar, a_car = (("comprar en bolsa", "evitar la bolsa") if rol == "comercializador"
                        else ("retener generación", "despachar y vender"))
        gauge_html = f"""
  <div class="escala-tit"><span>Escala de precio</span><span>esperado ahora <span class="v">{es(q50, 1)} COP/kWh</span></span></div>
  <div class="gauge-track" style="background:{grad};opacity:.85">
    <div class="gauge-marker" style="left:{pq:.1f}%"></div>
    <div class="gauge-bocadillo" style="left:{pq:.1f}%">{q50:.0f}</div>
  </div>
  <div class="escala3">
    <div><b>Barato &le; {bajo_val:.0f}</b><br>{a_bar}</div>
    <div><b>{bajo_val:.0f} &ndash; {alto_val:.0f}</b><br>esperar</div>
    <div><b>Caro &ge; {alto_val:.0f}</b><br>{a_car}</div>
  </div>"""

    # riesgo de salto de la hora (desplome / pico) segun el detector de eventos
    riesgo_html = ""
    p_d, p_p = float(fila.get("p_desplome", np.nan)), float(fila.get("p_pico", np.nan))
    if not (np.isnan(p_d) and np.isnan(p_p)):
        def chip_riesgo(nombre, p, clase):
            estilo = f"chip {clase}" if p >= UMBRAL_ALERTA else "chip"
            fondo = "" if p >= UMBRAL_ALERTA else f' style="background:{tk["neutro_bg"]};color:{tk["texto"]}"'
            return f'<span class="{estilo}"{fondo}>{nombre} {p * 100:.0f} %</span>'
        horas_d = dia_df.loc[dia_df["p_desplome"] >= UMBRAL_ALERTA, "fecha_hora"].dt.hour.tolist()
        horas_p = dia_df.loc[dia_df["p_pico"] >= UMBRAL_ALERTA, "fecha_hora"].dt.hour.tolist()
        resumen = []
        if horas_d:
            resumen.append(f"posibilidad de desplome a las {', '.join(f'{h:02d}' for h in horas_d)} h")
        if horas_p:
            resumen.append(f"posibilidad de pico a las {', '.join(f'{h:02d}' for h in horas_p)} h")
        chips = ("" if np.isnan(p_d) else chip_riesgo("&#9660; Posibilidad de desplome", p_d, "chip-baja") + " ") + \
            ("" if np.isnan(p_p) else chip_riesgo("&#9650; Posibilidad de pico", p_p, "chip-alta"))
        nota = ("Es una probabilidad estimada, no una certeza: puede no ocurrir."
                + (" Para el segundo y el tercer día del pronóstico solo se estima la de pico, y es menos confiable "
                   "que la del primer día." if bool(fila.get("solo_pico", False)) else ""))
        riesgo_html = (f'<div class="sub-card"><div class="t">Posibilidad de salto en esta hora</div>{chips}'
                       f'<div style="margin-top:6px">{"Este día: " + "; ".join(resumen) if resumen else "Sin alertas este día."}</div>'
                       f'<div style="margin-top:4px;opacity:.75;font-size:.8rem">{nota}</div></div>')

    # si ninguna regla fue estable en el backtest, se avisa en la tarjeta (no solo al final de la pagina)
    aviso_estable = ("" if es_valido else
                     f'<div class="chip-nota" style="margin:10px 0 0">&#9888; Ninguna regla fue estable en el backtest; '
                     f'se usa <b>{metodo}</b>, la menos inestable. Tómala con cautela (detalle al final de la página).</div>')

    # ---- Recomendacion + pronostico del dia ----
    favorable = {"verde": "favorable", "rojo": "desfavorable", "ambar": "neutral"}[color]
    c_bar, c_car = (tk["verde"], tk["rojo"]) if rol == "comercializador" else (tk["rojo"], tk["verde"])
    a_bar, a_car = (("comprar en bolsa", "evitar la bolsa") if rol == "comercializador"
                    else ("retener generación", "despachar y vender"))
    umbral_ok = not (np.isnan(bajo_val) or np.isnan(alto_val))
    tag = (f'<div class="hero-tag">Recomendaci&oacute;n &middot; {dia:%d/%m/%Y} &middot; {hora:02d}:00 h '
           f'&middot; rol {rol}</div>')
    badge = f'<span class="badge badge-{color}">{favorable}</span>'
    conf_card = (f'<div class="sub-card"><div class="t">Nivel de confianza</div><div class="g">Confianza {conf}</div>'
                 f'banda de &plusmn;{ancho_t0 / 2:.0f} COP/kWh, más ancha que el {conf_pct:.0f} % de las horas '
                 f'anteriores{aviso_estable}</div>')
    tramos = ([(f"Barato &le; {bajo_val:.0f}", a_bar, c_bar),
               (f"{bajo_val:.0f} &ndash; {alto_val:.0f}", "esperar", tk["ambar"]),
               (f"Caro &ge; {alto_val:.0f}", a_car, c_car)] if umbral_ok else [])

    def medidor():
        if not umbral_ok:
            return gauge_html
        cx, cy, r = 130, 132, 96

        def punto(f, rr=r):
            ang = np.pi * (1 - min(max(f, 0.0), 1.0))
            return cx + rr * np.cos(ang), cy - rr * np.sin(ang)

        def arco(f1, f2, col):
            (x1, y1), (x2, y2) = punto(f1), punto(f2)
            return (f'<path d="M{x1:.1f},{y1:.1f} A{r},{r} 0 0 1 {x2:.1f},{y2:.1f}" stroke="{col}" '
                    f'stroke-width="20" fill="none"/>')
        fb, fa, fq = pb / 100, pa / 100, pq / 100
        xa, ya = punto(fq, r - 28)
        (lbx, lby), (lax, lay) = punto(fb, r + 21), punto(fa, r + 21)
        return ('<svg viewBox="0 0 260 150" class="hd-svg" aria-hidden="true">'
                + arco(0, fb, c_bar) + arco(fb, fa, tk["control_borde"]) + arco(fa, 1, c_car)
                + f'<line x1="{cx}" y1="{cy}" x2="{xa:.1f}" y2="{ya:.1f}" stroke="{tk["tinta"]}" stroke-width="4" '
                  'stroke-linecap="round"/>'
                + f'<circle cx="{cx}" cy="{cy}" r="7" fill="{tk["tinta"]}"/>'
                + f'<text x="{lbx:.1f}" y="{lby:.1f}" class="hd-et" text-anchor="middle">{bajo_val:.0f}</text>'
                + f'<text x="{lax:.1f}" y="{lay:.1f}" class="hd-et" text-anchor="middle">{alto_val:.0f}</text>'
                + '</svg>')

    ley = "".join(f'<div><i style="background:{c}"></i><b>{a}</b> &middot; {b}</div>' for a, b, c in tramos)
    tarjeta = (f'<div class="hero hero-{color} hero-d">{tag}<div class="hd-fila"><div>{medidor()}'
               f'<div class="hd-v">{es(q50, 1)}</div><div class="hd-u">COP/kWh esperado</div></div>'
               f'<div><div class="hero-cab">{badge}</div><div class="hero-accion">{titulo}</div>'
               f'<div class="hero-frase">{frase}</div><div class="hd-ley">{ley}</div></div></div>'
               f'<div class="sub-cards">{conf_card}{riesgo_html}</div></div>')
    col_hero, col_dia = st.columns([5, 4], gap="medium")
    with col_hero:
        st.markdown(tarjeta, unsafe_allow_html=True)
    with col_dia:
        sec_d = st.container(key="sec_dia_principal")
        c_tit, c_cen, c_amp = sec_d.columns([2.2, 1, 1], vertical_alignment="center")
        c_tit.markdown('<div class="sec-tit">Pronóstico del día</div>', unsafe_allow_html=True)
        pista_variables(sec_d, " &middot; clic en un punto para analizar esa hora")
        # bloque de 120 dias ya cargado en el navegador: al mover la grafica los vecinos aparecen sin recargar
        inicio_dia = pd.Timestamp(dia)
        # el encuadre se conserva mientras se mira el mismo dia; "Centrar" (o cambiar de dia) lo devuelve al dia elegido.
        # La leyenda conserva lo ocultado (uirevision del layout); los ejes llevan su propia revision.
        c_cen.button("Centrar", icon=":material/center_focus_strong:", key="centrar_dia", type="tertiary",
                     on_click=lambda: st.session_state.update(rev_dia=st.session_state.get("rev_dia", 0) + 1))
        bloque_d = bloque_dias(dia, "mas_dia")
        fig_dia = encuadrar(
            base_dia(horizonte, rol, metodo, INTRADIA, TEMA, *bloque_d),
            x=[inicio_dia, inicio_dia + pd.Timedelta(hours=23)],     # 00:00 a 23:00 exactas (Juan, 19-oct)
            y=rango_y(dia_df["q10"], dia_df["q90"], dia_df["real"], alto_s.reindex(dia_df.index),
                      bajo_s.reindex(dia_df.index)),
            rev=f"{dia}-{st.session_state.get('rev_dia', 0)}", meta=meta_bloque(df, bloque_d, dia),
            formas=[linea_vertical(inicio_dia + pd.Timedelta(hours=int(hora)), tk["gris"])])
        if c_amp.button("Ampliar", icon=":material/open_in_full:", key="ampliar_dia", type="tertiary"):
            ampliar_grafica(go.Figure(fig_dia))
        # clic en la grafica = elegir esa hora (lo resuelve zoom_con_ctrl(): oprime el boton de la hora en la franja)
        sec_d.plotly_chart(fig_dia, width="stretch", theme=None, config=CONF_GRAF, key="graf_dia")
        botones_cargar_mas("mas_dia", dia)
    st.markdown(metadatos_html(f"{dia:%d/%m/%Y}, 00:00 a 23:00 h",
                               "Semáforo y pronóstico del día con su banda de incertidumbre"),
                unsafe_allow_html=True)

    # ---- El dia hora por hora ----
    sec = st.container(key="sec_franja")
    sec.markdown('<div class="sec-tit">El día, hora por hora</div><div class="sec-sub">Pronóstico de cada hora y la '
                 'acción del motor; haz clic en una hora para verla arriba.</div>', unsafe_allow_html=True)
    with sec:
        franja_botones(dia_df, hora, tk)
    if rol == "comercializador":
        leyenda = ("<b>Verde</b> = hora para comprar en bolsa &nbsp;&middot;&nbsp; "
                   "<b>naranja</b> = hora para cubrirse con contratos &nbsp;&middot;&nbsp; "
                   "<b>gris</b> = esperar. Recuadro con borde = la hora seleccionada &mdash; "
                   "haz clic en cualquier hora para elegirla directamente.")
    else:
        leyenda = ("<b>Verde</b> = hora para despachar y vender &nbsp;&middot;&nbsp; "
                   "<b>naranja</b> = hora para retener generación &nbsp;&middot;&nbsp; "
                   "<b>gris</b> = esperar. Recuadro con borde = la hora seleccionada &mdash; "
                   "haz clic en cualquier hora para elegirla directamente.")
    # la hora 00:00 del dia es la ultima de la ventana anterior: se decide con la mayoria de horas, no con cualquiera
    solo_pico = dia_df["solo_pico"].mean() > 0.5
    if dia_df["p_desplome"].notna().any() and not solo_pico:
        leyenda += (f" &#9660; / &#9650; sobre una hora = posibilidad de desplome / pico de {UMBRAL_ALERTA * 100:.0f} % o más "
                    "(precio 30 % por debajo / por encima de la mediana del día).")
    elif dia_df["p_pico"].notna().any():
        leyenda += (f" &#9650; sobre una hora = posibilidad de pico de {UMBRAL_ALERTA * 100:.0f} % o más (precio 30 % por "
                    "encima de la mediana del día). Para el segundo y el tercer día del pronóstico de 72 h solo se estima "
                    "la posibilidad de pico, menos confiable que la del primer día; la de desplome no superó a la "
                    "frecuencia histórica en la validación.")
    elif horizonte == "72h":
        leyenda += " Para este día no hay estimación de pico ni de desplome."
    sec.markdown(f'<div class="leyenda" style="margin-bottom:10px">{leyenda}</div>', unsafe_allow_html=True)

    # ---- Mapa de calor dia-hora: rango de dias (de 1 dia al ultimo disponible) y de horas a eleccion ----
    sec_calor = st.container(key="sec_calor")
    sec_calor.markdown('<div class="sec-tit">Mapa de calor día-hora</div><div class="sec-sub">La señal o el precio '
                       'esperado de varios días a la vez; el recuadro marca el día y la hora elegidos.</div>',
                       unsafe_allow_html=True)
    clave_dias = f"rango_dias_calor_{horizonte}"  # por horizonte: cada uno tiene su propio rango de fechas
    # Streamlit borra el estado de un control que no se dibuja (el del otro horizonte), asi que se guarda
    # una copia aparte. El estado solo se escribe para restaurarla o para inicializar: escribirlo en cada
    # ejecucion pisaria lo que elige el usuario.
    def valido(v):
        v = v if isinstance(v, (tuple, list)) else (v,)
        return all(d is not None and completos[0] <= d <= completos[-1] for d in v)
    if not valido(st.session_state.get(clave_dias)):
        copia = st.session_state.get(f"_copia_{clave_dias}")
        if copia is not None and valido(copia):
            st.session_state[clave_dias] = copia
        else:
            previos = [d for d in completos if d <= dia]
            st.session_state[clave_dias] = (previos[-14:][0], previos[-1])
    # las fechas de los extremos de la barra se pueden escribir: si la barra cambio, las fechas la siguen; si el
    # usuario escribe una fecha, la barra toma el nuevo rango (si quedan al reves, se ordenan)
    clave_ini, clave_fin = f"ini_{clave_dias}", f"fin_{clave_dias}"
    actual = tuple(st.session_state[clave_dias])
    if (st.session_state.get(f"_ultimo_{clave_dias}") != actual or clave_ini not in st.session_state
            or clave_fin not in st.session_state):
        st.session_state[clave_ini], st.session_state[clave_fin] = actual
        st.session_state[f"_ultimo_{clave_dias}"] = actual

    def _fechas_a_barra():
        a, b = st.session_state.get(clave_ini), st.session_state.get(clave_fin)
        if a is None or b is None:
            return
        nuevo = (min(a, b), max(a, b))
        st.session_state[clave_dias] = st.session_state[f"_copia_{clave_dias}"] = nuevo
        st.session_state[f"_ultimo_{clave_dias}"] = None          # las fechas se reescriben ya ordenadas
    c_dias, c_horas, c_modo = sec_calor.columns([5, 3, 2])
    with c_dias, st.container(key="dias_calor"):
        st.markdown('<div class="etq-dias">Días</div>', unsafe_allow_html=True)
        f_ini, _, f_fin = st.columns([1, 1, 1])
        with f_ini:
            st.date_input("Primer día", min_value=completos[0], max_value=completos[-1], key=clave_ini,
                          format="DD/MM/YYYY", on_change=_fechas_a_barra, label_visibility="collapsed")
        with f_fin, st.container(key="dias_fin"):
            st.date_input("Último día", min_value=completos[0], max_value=completos[-1], key=clave_fin,
                          format="DD/MM/YYYY", on_change=_fechas_a_barra, label_visibility="collapsed")
        rango_dias = st.slider("Días", min_value=completos[0], max_value=completos[-1], key=clave_dias,
                               step=pd.Timedelta(days=1).to_pytimedelta(), format="DD/MM/YYYY",
                               label_visibility="collapsed")
        d_ini, d_fin = rango_dias if isinstance(rango_dias, (tuple, list)) else (rango_dias, rango_dias)
        st.session_state[f"_copia_{clave_dias}"] = (d_ini, d_fin)
    with c_horas:
        rango_horas = st.slider("Horas", 0, 23, (0, 23), key="rango_horas_calor",
                                help="Para ver una sola hora, junta los dos extremos.")
        h_ini, h_fin = rango_horas if isinstance(rango_horas, (tuple, list)) else (rango_horas, rango_horas)
    with c_modo:
        modo_calor = st.radio("Colorear por", ["Señal del motor", "Precio esperado"], key="modo_calor")
    dias_calor = [d for d in completos if d_ini <= d <= d_fin]
    horas_calor = list(range(h_ini, h_fin + 1))
    fig_calor = base_calor(horizonte, rol, metodo, INTRADIA, TEMA, tuple(dias_calor), tuple(horas_calor), modo_calor)
    if dia in dias_calor and hora in horas_calor:     # recuadro del dia y la hora elegidos
        y0 = dias_calor.index(dia)
        fig_calor = encuadrar(fig_calor, formas=[{"type": "rect", "x0": hora - .5, "x1": hora + .5, "y0": y0 - .5,
                                                  "y1": y0 + .5, "line": {"color": tk["tinta"], "width": 2.5}}])
    sec_calor.plotly_chart(fig_calor, width="stretch", theme=None, config=CONF_GRAF)
    if modo_calor == "Señal del motor":
        favor, contra = ("Comprar en bolsa", "Evitar la bolsa") if rol == "comercializador" else             ("Despachar y vender", "Retener generación")
        items = [(tk["calor_verde"], favor), (tk["calor_rojo"], contra), (tk["calor_ambar"], "Esperar")]
        if "sin_umbral" in df and df["sin_umbral"].any():
            items.append((tk["linea"], "Sin umbral todavía"))
        sec_calor.markdown('<div class="ley-calor">' + "".join(f'<span><i style="background:{c}"></i>{t}</span>'
                                                              for c, t in items) + "</div>", unsafe_allow_html=True)
    n_d, n_h = len(dias_calor), len(horas_calor)
    st.markdown(metadatos_html(f"{d_ini:%d/%m/%Y} a {d_fin:%d/%m/%Y}, {h_ini:02d}:00 a {h_fin:02d}:00 h "
                               f"({n_d} día{'s' if n_d > 1 else ''} × {n_h} hora{'s' if n_h > 1 else ''})",
                               "Mapa de calor día-hora; recuadro = día y hora elegidos"), unsafe_allow_html=True)

    # ---- Indicadores del dia y del backtest, con mini graficas ----
    accion = "comprar" if rol == "comercializador" else "vender"
    por_dia = (df.assign(d=df["fecha_hora"].dt.date, a=df["senal"].eq(accion))
                 .groupby("d")["a"].sum())
    ultimos = por_dia.loc[:dia].tail(14)
    n_accion_dia = int(por_dia.get(dia, 0))
    bt = evaluar_backtest(df, df["senal"], rol)
    ventaja = bt["ventaja_cop_kwh"]
    mensual = resumen_mensual(horizonte, rol, metodo, INTRADIA)
    verbo = "comprado más barato" if rol == "comercializador" else "vendido más caro"
    # reparto de las senales en todo el periodo (favorable / desfavorable / esperar)
    favor, contra = ("comprar", "evitar_compra") if rol == "comercializador" else ("vender", "retener")
    reparto = df["senal"].value_counts(normalize=True)
    pf, pc_, pe = (float(reparto.get(x, 0)) * 100 for x in (favor, contra, "esperar"))
    distrib_html = (f'<div class="distrib"><div style="width:{pf:.1f}%;background:{tk["verde"]}"></div>'
                    f'<div style="width:{pc_:.1f}%;background:{tk["rojo"]}"></div>'
                    f'<div style="width:{pe:.1f}%;background:{tk["neutro_bg"]}"></div></div>'
                    f'<div class="distrib-ley"><i style="background:{tk["verde"]}"></i>{favor.replace("_", " ")} {pf:.0f} %'
                    f'<i style="background:{tk["rojo"]}"></i>{SIGNIFICADO[contra][0].capitalize().split()[0].lower()} {pc_:.0f} %'
                    f'<i style="background:{tk["neutro_bg"]}"></i>esperar {pe:.0f} %</div>')
    st.markdown(tarjetas_html([
        ("Horas para actuar hoy", f"{n_accion_dia} de 24",
         mini_barras(ultimos.values, tk["verde"]),
         "cuadros verdes de la franja; barras: últimos 14 días"),
        ("Ventaja histórica de la regla",
         "&mdash;" if np.isnan(ventaja) else f"{es(ventaja, 0, True)} COP/kWh",
         mini_linea(mensual["ventaja"], tk["q50"], tk["linea"]),
         f"habrías {verbo} que el promedio en el backtest 2026; línea: mes a mes"),
        ("Con qué frecuencia actúa",
         f"{es(bt['frecuencia_accion'] * 100)} % del tiempo",
         mini_linea(mensual["frecuencia"], tk["ambar"], tk["linea"]),
         f"{es(bt['horas_accion'])} horas de acción en todo el periodo; línea: mes a mes" + distrib_html),
    ]), unsafe_allow_html=True)

    # ---- Contexto: la semana alrededor del dia elegido ----
    # siempre 7 dias: si el dia elegido esta al final de la serie, la semana se corre hacia atras (antes quedaba
    # recortada); se cargan +-60 dias para que al mover la grafica aparezcan los vecinos sin recargar
    ultimo = df["fecha_hora"].max().normalize() + pd.Timedelta(days=1)
    fin = min(pd.Timestamp(dia) + pd.Timedelta(days=4), ultimo)
    ini = fin - pd.Timedelta(days=7)
    ventana = df[(df["fecha_hora"] >= ini) & (df["fecha_hora"] < fin)]
    marca = pd.Timestamp(dia) + pd.Timedelta(hours=int(hora))
    sec_sem = st.container(key="sec_semana")
    sec_sem.markdown('<div class="sec-tit">La semana alrededor del día elegido</div>', unsafe_allow_html=True)
    pista_variables(sec_sem)
    bloque_s = bloque_dias(ini + pd.Timedelta(days=3), "mas_sem")
    fig_sem = encuadrar(base_semana(horizonte, rol, metodo, INTRADIA, TEMA, *bloque_s), meta=meta_bloque(df, bloque_s),
                        x=[ini, fin - pd.Timedelta(hours=1)], y=rango_y(ventana["q10"], ventana["q90"], ventana["real"]),
                        rev=f"{dia}", formas=[linea_vertical(marca, tk["gris"], punteada=True)])
    sec_sem.plotly_chart(fig_sem, width="stretch", theme=None, config=CONF_GRAF_BARRA)
    botones_cargar_mas("mas_sem", ini + pd.Timedelta(days=3))
    # periodo realmente graficado (al final de la serie la semana puede quedar recortada)
    v_ini = ventana["fecha_hora"].min() if len(ventana) else ini
    v_fin = ventana["fecha_hora"].max() if len(ventana) else fin - pd.Timedelta(hours=1)
    st.markdown(metadatos_html(f"{v_ini:%d/%m/%Y} a {v_fin:%d/%m/%Y}",
                               "Comparación real frente a pronóstico con las horas de acción"),
                unsafe_allow_html=True)

    # ---- Nota sobre la regla activa ----
    if es_valido:
        st.caption(
            f"Regla activa: **{metodo}**. La elige el motor automáticamente porque fue la más "
            "estable en el backtest 2026: mantuvo ventaja económica positiva y una frecuencia de "
            "acción razonable (10-40%) en las dos mitades del periodo, no solo en el promedio del "
            "año completo. "
            + ("Con filtro intradía: cada señal se conserva solo en la mitad del día que le corresponde "
               "según el pronóstico (vender o comprar en las horas más caras o baratas del día). " if INTRADIA else "")
            + "Para ver la comparación o forzar otro método, entra a la vista Analista."
        )
    else:
        st.warning(
            f"Regla activa: **{metodo}** (la más estable disponible, pero ninguna se sostuvo con "
            "ventaja positiva y frecuencia razonable en las DOS mitades del periodo para este "
            "rol/horizonte). Revisa la vista Analista antes de confiar en la señal."
        )


# ==========================================================================
# VISTA ANALISTA
# ==========================================================================
def vista_analista():
    with st.sidebar:
        st.divider()
        st.subheader("Controles de análisis")
        p_bajo, p_alto = st.slider("Percentiles del umbral (bajo / alto)", 5, 95, (25, 75), step=5)

        # El rango de fechas se pide ANTES de comparar metodos para que "el metodo
        # sugerido" responda al rango que el usuario filtra.
        rango = bandas["fecha_hora"].dt.date
        fecha_min, fecha_max = rango.min(), rango.max()
        inicio, fin = st.slider("Rango de fechas", min_value=fecha_min, max_value=fecha_max,
                                value=(fecha_min, fecha_max))

        st.toggle("Filtro intradía", key="filtro_intradia_w", value=INTRADIA, on_change=_guardar_intradia,
                  help=("Conserva cada señal solo en la mitad del día que le corresponde según el pronóstico: "
                        "vender o comprar en las horas más caras o más baratas del día, retener o evitar la bolsa "
                        "en las opuestas. Los umbrales deciden en qué temporada conviene la bolsa; el filtro decide "
                        "a qué hora del día actuar."))
        tabla_metodos, metodo_sugerido, es_valido = comparacion(horizonte, rol, p_bajo, p_alto, inicio, fin, INTRADIA)
        etiqueta = (f"{metodo_sugerido} (sugerido para este rango)" if es_valido
                    else f"{metodo_sugerido} (sugerido, sin método 100% válido en este rango)")
        opciones = ["fijo", "rodante", "banda", "hibrido"]
        metodo = st.selectbox(
            "Método de umbral", opciones, index=opciones.index(metodo_sugerido),
            help=("**fijo**: percentiles del precio histórico 2019-2025.\n\n"
                  "**rodante**: percentiles con ventana móvil de los 30 días anteriores.\n\n"
                  "**banda**: percentiles fijos, y «esperar» cuando la banda de incertidumbre es ancha.\n\n"
                  "**hibrido**: umbral rodante con el mismo «esperar» del método banda.\n\n"
                  f"Sugerido por el backtest económico en el rango de fechas elegido: {etiqueta}."),
        )

    base = serie_con_senal(horizonte, rol, metodo, p_bajo, p_alto, intradia=INTRADIA)
    base = base[(base["fecha_hora"].dt.date >= inicio) & (base["fecha_hora"].dt.date <= fin)]
    resultado = evaluar_backtest(base, base["senal"], rol)

    accion_principal = "comprar" if rol == "comercializador" else "vender"
    ventaja = resultado["ventaja_cop_kwh"]
    st.markdown(tarjetas_html([
        (f"Ventaja económica ({accion_principal})",
         "&mdash;" if np.isnan(ventaja) else f"{es(ventaja, 1, True)} COP/kWh", "",
         "precio medio en horas de acción frente al promedio del periodo filtrado"),
        ("Frecuencia de acción", f"{es(resultado['frecuencia_accion'] * 100, 1)} %", "",
         "rango útil: 10 % a 40 % de las horas"),
        ("Horas de acción", es(resultado['horas_accion']), "",
         f"del {inicio:%d/%m/%Y} al {fin:%d/%m/%Y}"),
    ]), unsafe_allow_html=True)

    sec_m = st.container(key="sec_metodos")
    sec_m.markdown('<div class="sec-tit">Comparación de métodos en el rango elegido</div><div class="sec-sub">Ventaja y '
                   'frecuencia de acción de cada método; la vista Operador usa el más estable entre las dos mitades del '
                   'periodo, no el de mayor ventaja.</div>', unsafe_allow_html=True)
    sec_m.markdown(tabla_metodos_html(tabla_metodos, metodo, metodo_sugerido), unsafe_allow_html=True)
    sec_m.markdown("")

    sec_bt = st.container(key="sec_backtest")
    sec_bt.markdown(f'<div class="sec-tit">Precio real, pronóstico y señales &mdash; método {metodo}</div>',
                    unsafe_allow_html=True)
    pista_variables(sec_bt)
    sec_bt.plotly_chart(
        fechas_compactas(vista_reciente(sin_titulo(grafico_precio(
            base, tk, f"Precio real, pronóstico y señales — método {metodo}", alto=460)), base, tk)
            .update_xaxes(uirevision=f"{inicio}-{fin}").update_yaxes(uirevision=f"{inicio}-{fin}")),
        width="stretch", theme=None, config=CONF_GRAF_BARRA,
    )
    st.markdown(metadatos_html(f"{inicio:%d/%m/%Y} a {fin:%d/%m/%Y}",
                               f"Backtest del método {metodo} sobre real frente a pronóstico"),
                unsafe_allow_html=True)

    st.caption("La tabla de abajo trae, hora a hora y solo para el rango de fechas elegido, los datos con los que se "
               "calculan las cifras y la gráfica de esta vista: precio real (real), pronóstico (q50) y su banda "
               "(q10 a q90), la señal del método (senal), si el método aún no tenía umbral (sin_umbral) y, en 24 h, "
               "las probabilidades de desplome y pico. Se puede ordenar por columna y descargar en CSV.")
    with st.expander("Ver datos filtrados"):
        favor, contra = ("comprar", "evitar_compra") if rol == "comercializador" else ("vender", "retener")
        filtro = st.segmented_control("Mostrar", ["Todas", f"Solo {favor}", f"Solo {contra.replace('_', ' ')}", "Solo esperar"],
                                      default="Todas", key="filtro_datos")
        vista_df = base
        if filtro == f"Solo {favor}":
            vista_df = base[base["senal"] == favor]
        elif filtro and filtro.startswith("Solo") and filtro != "Solo esperar":
            vista_df = base[base["senal"] == contra]
        elif filtro == "Solo esperar":
            vista_df = base[base["senal"] == "esperar"]
        st.dataframe(vista_df, width="stretch", hide_index=True,
                     column_config={"fecha_hora": st.column_config.DatetimeColumn("Fecha y hora", format="DD/MM/YYYY HH:mm"),
                                    "real": st.column_config.NumberColumn("Real (COP/kWh)", format="%.1f"),
                                    "q10": st.column_config.NumberColumn("q10", format="%.1f"),
                                    "q50": st.column_config.NumberColumn("q50", format="%.1f"),
                                    "q90": st.column_config.NumberColumn("q90", format="%.1f"),
                                    "senal": "Señal", "sin_umbral": "Sin umbral",
                                    "p_desplome": st.column_config.NumberColumn("P(desplome)", format="percent"),
                                    "p_pico": st.column_config.NumberColumn("P(pico)", format="percent")})
        st.caption(f"{es(len(vista_df))} horas. Usa la lupa de la tabla para buscar y el botón de descarga para bajarla en CSV.")


# ==========================================================================
# VISTA METODOLOGIA: desempeno en vivo y como funciona
# ==========================================================================
@st.cache_data
def cargar_predespacho():
    """Predespacho ideal de XM (costo marginal del iMAR), la referencia que publica XM la vispera."""
    ruta = RAIZ / "data" / "processed" / "predespacho_ideal_xm.csv"
    p = pd.read_csv(ruta, parse_dates=["fecha_hora"], usecols=["fecha_hora", "costo_marginal"])
    return p.set_index("fecha_hora")["costo_marginal"]


def grafico_error_diario(d, tk):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["pre"], name="Predespacho de XM", mode="lines+markers",
                             line=dict(color=tk["gris"], width=1.5, dash="dash"), marker=dict(size=4),
                             hovertemplate="%{y:,.0f} COP/kWh"))
    fig.add_trace(go.Scatter(x=d.index, y=d["modelo"], name="Augur", mode="lines+markers",
                             line=dict(color=tk["q50"], width=2.2), marker=dict(size=5),
                             hovertemplate="%{y:,.0f} COP/kWh"))
    fig = estilo_figura(fig, tk, 320, "", margen_leyenda=56, fechas=False)
    fig.update_layout(title_text="", margin_t=12, uirevision="fija", yaxis_title="Error medio del día (COP/kWh)")
    fig.update_xaxes(tickformat="%d/%m", hoverformat="%d/%m/%Y")
    return fig


def vista_metodologia():
    # ---------------------------------------------------------------- desempeno en vivo (pronostico de 24 h)
    sec = st.container(key="sec_desempeno")
    sec.markdown('<div class="sec-tit">Desempeño en vivo</div><div class="sec-sub">Qué tan bien le ha ido al pronóstico '
                 'de 24 h y a las recomendaciones, comparados con el precio real que XM publica unos tres días después. '
                 'Se recalcula solo con cada publicación de XM.</div>', unsafe_allow_html=True)
    periodo = sec.segmented_control("Periodo", ["Últimos 30 días", "Todo 2026"], default="Últimos 30 días",
                                    key="periodo_desempeno", label_visibility="collapsed") or "Últimos 30 días"
    _, metodo, _ = comparacion_estable("24h", rol, INTRADIA)
    df = serie_con_senal("24h", rol, metodo, intradia=INTRADIA)
    df = df[df["real"].notna()].copy()
    df["pre"] = cargar_predespacho().reindex(df["fecha_hora"]).to_numpy()
    if periodo == "Últimos 30 días":
        df = df[df["fecha_hora"] > df["fecha_hora"].max() - pd.Timedelta(days=30)]
    df = df[df["pre"].notna()]
    e_m, e_p = (df["q50"] - df["real"]).abs(), (df["pre"] - df["real"]).abs()
    mejora = (1 - e_m.mean() / e_p.mean()) * 100
    cob = ((df["real"] >= df["q10"]) & (df["real"] <= df["q90"])).mean() * 100
    # acierto: la accion favorable acierta si el precio real de esa hora quedo del lado bueno de la mediana real
    # de su dia (por encima para vender, por debajo para comprar); la desfavorable, al reves. Al azar seria ~50 %
    mediana = df.groupby(df["fecha_hora"].dt.normalize())["real"].transform("median")
    favor, contra = ("vender", "retener") if rol == "generador" else ("comprar", "evitar_compra")
    alto = df["real"] > mediana
    ok_f = (alto if rol == "generador" else ~alto)[df["senal"] == favor]
    ok_c = (~alto if rol == "generador" else alto)[df["senal"] == contra]
    pct = lambda s: f"{es(s.mean() * 100)} %" if len(s) else "s/d"
    diario = pd.DataFrame({"modelo": e_m.to_numpy(), "pre": e_p.to_numpy()},
                          index=df["fecha_hora"].dt.normalize().to_numpy()).groupby(level=0).mean()
    sec.markdown(tarjetas_html([
        ("Error del pronóstico", f"{es(e_m.mean(), 1)} COP/kWh",
         mini_linea(diario["modelo"].tail(30), tk["q50"], tk["linea"]),
         f"MAPE {es((e_m / df['real']).mean() * 100, 1)} %; el predespacho de XM: {es(e_p.mean(), 1)} COP/kWh"),
        ("Frente al predespacho de XM", f"{es(mejora, 1, True)} %", "",
         f"menos error; Augur estuvo más cerca del precio real en el {(e_m < e_p).mean() * 100:.0f} % de las horas"),
        ("Precio real dentro de la banda", f"{cob:.0f} %", "",
         "objetivo 80 %: la banda [q10, q90] debería contener el precio real 8 de cada 10 horas"),
        (f"Acierto de las recomendaciones", f"{pct(ok_f)} · {pct(ok_c)}", "",
         f"{favor.replace('_', ' ')} ({len(ok_f)} h) · {SIGNIFICADO[contra][0].lower()} ({len(ok_c)} h): el precio real "
         f"quedó del lado correcto de la mediana del día; al azar sería cerca de 50 %"),
    ]), unsafe_allow_html=True)
    sec.markdown('<div class="sec-tit" style="font-size:.95rem;margin-top:10px">Error de cada día: Augur frente al '
                 'predespacho de XM</div>', unsafe_allow_html=True)
    pista_variables(sec)
    sec.plotly_chart(grafico_error_diario(diario, tk), width="stretch", theme=None,
                     config=CONF_GRAF, key="graf_error_diario")
    st.markdown(metadatos_html(f"{df['fecha_hora'].min():%d/%m/%Y} a {df['fecha_hora'].max():%d/%m/%Y} "
                               f"({es(len(df))} horas con precio real)",
                               f"Desempeño del pronóstico de 24 h y del método «{metodo}» para el {rol}"),
                unsafe_allow_html=True)

    # ---------------------------------------------------------------- como funciona
    sec2 = st.container(key="sec_metodo")
    sec2.markdown('<div class="sec-tit">Cómo funciona</div><div class="sec-sub">De los datos públicos a la '
                  'recomendación de cada hora. Todo el proceso corre solo cada día.</div>', unsafe_allow_html=True)
    pasos = [
        ("1 · Datos públicos", "XM: precio de bolsa, demanda, generación, embalses, aportes y predespacho ideal. "
                               "NOAA: índice ONI de El Niño."),
        ("2 · Corte a las 00:00", "Solo se usa lo publicado antes de pronosticar: el precio real llega con unos 3 días "
                                  "de rezago; el predespacho del día, la víspera hacia las 10:30."),
        ("3 · Pronóstico", "El predespacho ideal de XM, corregido por tres modelos CatBoost que se reentrenan cada "
                           "mes y se combinan por franjas de 6 horas."),
        ("4 · Incertidumbre", "Una banda [q10, q90] por hora, calibrada con los errores recientes para contener el "
                              "precio real el 80 % de las veces."),
        ("5 · Motor de decisión", "Compara el precio esperado con los umbrales barato y caro y entrega la acción "
                                  "de cada hora según el rol."),
    ]
    sec2.markdown('<div class="flujo">' + "".join(f'<div class="paso"><div class="n">{a}</div><div class="d">{b}</div></div>'
                                                  for a, b in pasos) + "</div>", unsafe_allow_html=True)
    fichas = [
        ("Cómo decide el motor",
         "<ul><li>Umbral barato y caro: percentiles 25 y 75 del precio (fijos sobre 2019-2025 o móviles sobre los "
         "últimos 30 días, según el método).</li>"
         "<li>Filtro de confianza: si la banda de la hora es más ancha que el 75 % de las horas anteriores, espera.</li>"
         "<li>Filtro intradía: vender solo en la mitad más cara del día; retener, en la más barata.</li>"
         f"<li>De los cuatro métodos se usa el más estable en el backtest; hoy, «{metodo}» para el {rol}.</li></ul>"),
        ("Validación del informe final",
         "<ul><li>Fuera de muestra, en 4.848 horas de 2026 y en cinco años históricos.</li>"
         "<li>24 h: error medio de 33,7 COP/kWh (MAPE 10 %), 68 % menor que la persistencia y 11 % menor que el "
         "predespacho (prueba de Diebold-Mariano, p &lt; 0,001).</li>"
         "<li>72 h: 52,2 COP/kWh. Cobertura de la banda: 80,6 %.</li></ul>"),
        ("Limitaciones",
         "<ul><li>Es apoyo a la decisión, no asesoría financiera.</li>"
         "<li>El pronóstico de 24 h sale después de las 10:30 de la víspera: no sirve para la oferta de las 08:00.</li>"
         "<li>En periodos muy volátiles (como sep-oct de 2026) el error crece y la banda cubre menos.</li>"
         "<li>XM revisa el precio después de publicarlo (versiones TX1, TX2, TXR y TXF de la liquidación): los días "
         "recientes llevan la versión preliminar, que en algunas horas cambia más de 20 COP/kWh.</li>"
         "<li>Pronostica el precio de bolsa nacional, no el precio de transacciones en bolsa (PTB), que se liquida "
         "con el tope del precio de escasez.</li>"
         "<li>No considera contratos ni la exposición real a bolsa de cada empresa.</li></ul>"),
    ]
    sec2.markdown('<div class="fichas-met">' + "".join(f'<div class="ficha-met"><div class="t">{a}</div>{b}</div>'
                                                       for a, b in fichas) + "</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------- formulas y validacion
    # mismas ecuaciones que pronostico_realista_24h.py, realista_motor.bandas y src/motor_decision.py
    sec3 = st.container(key="sec_formulas")
    sec3.markdown('<div class="sec-tit">Fórmulas y validación</div><div class="sec-sub">Las ecuaciones que corren cada '
                  'día, en el orden del flujo. <i>y</i> es el precio real de bolsa, <i>P</i> el predespacho ideal de XM y '
                  '<i>h</i> la hora pronosticada.</div>', unsafe_allow_html=True)
    bloques = [
        ("1 · Corrección del predespacho",
         [r"r_h = \ln\frac{y_h}{P_h} \qquad \hat{r}^{(k)}_h = f_k(\mathbf{x}_h), \quad k = 1, 2, 3",
          r"\bar{y}_h = P_h \exp\!\Big(\tfrac{1}{3}\textstyle\sum_{k} \hat{r}^{(k)}_h\Big)"],
         "Cada CatBoost (1.500 árboles, profundidad 6) aprende cuánto se equivoca el predespacho, en escala "
         "logarítmica, minimizando el error absoluto. Los tres usan variables distintas: base (precio y demanda con "
         "rezago de al menos 73 h, hidrología, ONI, calendario y predespacho), + forma intradía del predespacho y "
         "despacho programado, y + error reciente del predespacho. Se reentrenan cada mes."),
        ("2 · Combinación por franjas (QRA)",
         [r"\hat{y}_h = w_1 P_h + w_2\, \bar{y}_h + b",
          r"(w_1, w_2, b) = \arg\min_{w \ge 0,\, b} \sum_{t \in \mathcal{T}} \left| y_t - w_1 P_t - w_2 \bar{y}_t - b \right|"],
         "Regresión cuantílica de la mediana, resuelta como programa lineal para cada franja de 6 h. "
         "𝒯 son las horas de esa franja con precio real ya publicado (días hasta D-5). Los pesos se recalculan cada "
         "día: cuando el predespacho acierta, pesa más; cuando falla de forma sistemática, pesa más la corrección."),
        ("3 · Banda de incertidumbre (conformal)",
         [r"s_t = \left| \ln\frac{y_t}{\hat{y}_t} \right| \qquad m_h = \hat{Q}_{\lceil (n+1)\,0.8 \rceil / n}\big(\{s_t\}\big)",
          r"q_{10} = \hat{y}_h\, e^{-m_h} \qquad q_{90} = \hat{y}_h\, e^{+m_h}"],
         "Los errores relativos de la misma hora en los últimos 60 días con precio publicado fijan el ancho; "
         "el cuantil con corrección (n+1) de la predicción conformal da una cobertura nominal del 80 %. "
         "En escala log la banda nunca baja de cero y crece con el nivel del precio."),
        ("4 · Motor de decisión",
         [r"U_b = \mathrm{P}_{25}(\text{precio}) \qquad U_a = \mathrm{P}_{75}(\text{precio})",
          r"\text{acci\'on}_h = \begin{cases} \text{favorable} & \hat{y}_h \text{ del lado bueno de } U \\ "
          r"\text{desfavorable} & \hat{y}_h \text{ del lado malo de } U \\ \text{esperar} & \text{si no, o si } "
          r"\dfrac{q_{90}-q_{10}}{\hat{y}_h} > \mathrm{P}_{75}(\text{anchos previos}) \end{cases}"],
         "Para el comercializador lo favorable es comprar si ŷ ≤ U<sub>b</sub>; para el generador, vender si "
         "ŷ ≥ U<sub>a</sub>. Los umbrales son fijos (2019-2025) o móviles (30 días) según el método. El filtro "
         "intradía deja la acción favorable solo en la mitad del día que el pronóstico pone a su favor."),
    ]
    c1, c2 = sec3.columns(2, gap="medium")
    for i, (tit, ecs, txt) in enumerate(bloques):
        caja = (c1 if i % 2 == 0 else c2).container(key=f"form_{i}")
        caja.markdown(f'<div class="t">{tit}</div>', unsafe_allow_html=True)
        for e in ecs:
            caja.latex(e)
        caja.markdown(txt, unsafe_allow_html=True)

    sec3.markdown('<div class="sec-tit" style="font-size:.95rem;margin-top:10px">Cómo se validó</div>',
                  unsafe_allow_html=True)
    v1, v2 = sec3.columns(2, gap="medium")
    val = v1.container(key="form_val")
    val.markdown('<div class="t">Métricas</div>', unsafe_allow_html=True)
    val.latex(r"\mathrm{MAE} = \frac{1}{N}\sum_{t} |y_t - \hat{y}_t| \qquad "
              r"\mathrm{MAPE} = \frac{100}{N}\sum_{t} \frac{|y_t - \hat{y}_t|}{y_t}")
    val.latex(r"\text{Cobertura} = \frac{1}{N}\sum_{t} \mathbb{1}\left[q_{10,t} \le y_t \le q_{90,t}\right]")
    val.latex(r"d_t = |e^{\,\mathrm{pre}}_t| - |e^{\,\mathrm{Augur}}_t| \qquad "
              r"\mathrm{DM} = \frac{\bar{d}}{\sqrt{\hat{\sigma}^2_{\mathrm{HAC}}/N}} \sim \mathcal{N}(0,1)")
    val.markdown("La prueba de Diebold-Mariano dice si la diferencia de error frente al predespacho es real o "
                 "suerte; la varianza HAC corrige que los errores de horas seguidas están correlacionados.")
    pro = v2.container(key="form_prot")
    pro.markdown('<div class="t">Protocolo</div>', unsafe_allow_html=True)
    pro.markdown(
        "- **Sin mirar el futuro.** Cada día se pronostica solo con lo que XM y NOAA ya habían publicado a las 00:00: "
        "precio, demanda y generación hasta D-4, embalses y aportes de D-2, ONI de dos meses antes y el predespacho "
        "de la víspera.\n"
        "- **Fuera de muestra y en orden.** Ventana creciente: el modelo de cada mes se entrena solo con meses "
        "anteriores, y el combinador y la banda solo con días ya publicados (margen de 5 días).\n"
        "- **Contra quién se compara.** Persistencia (el último precio publicado) y el predespacho ideal de XM.\n"
        "- **Dónde.** Todo 2026 (24 h: MAE 33,7 COP/kWh; DM p < 0,001) y cinco años históricos, 2020-2025, "
        "donde supera al predespacho en todos.\n"
        "- **El motor.** Backtest hora a hora con el precio real; el método se elige por estabilidad entre las dos "
        "mitades del periodo, no por el mejor resultado.")


if vista == "Operador":
    vista_operador()
elif vista == "Metodología":
    vista_metodologia()
else:
    vista_analista()
