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

Tema claro u oscuro con el interruptor de la barra lateral. La vista y el tema
iniciales se pueden fijar por enlace: ?vista=analista&tema=oscuro.

Los datos entran SIEMPRE por cargar_fuente_pronostico() (src/motor_decision.py),
es decir segun data/processed/resultados/fuentes_pronostico.json -- si ese
archivo cambia (ej. 72h pasa a otro modelo), este dashboard lo refleja sin
tocar una sola linea aqui. Ver docs/motor_decision_guia.md.
"""
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

from motor_decision import (  # noqa: E402
    generar_senales, evaluar_backtest, comparar_metodos, elegir_mejor_metodo,
    comparar_metodos_estable, elegir_mejor_metodo_estable,
    cargar_fuente_pronostico, umbrales_fijos, umbrales_rodantes,
    horas_sin_umbral, precio_previo_ventana, ancho_relativo,
)

st.set_page_config(
    page_title="augur · Motor de decisión",
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
        "acento": "#D97706", "acento_texto": "#FFFFFF", "acento_sombra": "rgba(217,119,6,.35)",
        "calor_verde": "#059669", "calor_ambar": "#CBD5E1", "calor_rojo": "#EA580C",
        "scroll": "#94A3B8", "scroll_hover": "#5B6B82", "sombra": "0 1px 2px rgba(11,31,58,.07)",
        "logo_tenue": "#94A3B8",
    },
    "oscuro": {
        "verde": "#34D399", "verde_bg": "rgba(52,211,153,0.15)",
        "rojo": "#FB923C", "rojo_bg": "rgba(251,146,60,0.16)",
        "ambar": "#94A3B8", "ambar_bg": "rgba(148,163,184,0.16)",
        "tinta": "#EAF0F8", "texto": "#BCC8DA", "gris": "#8193AD",
        "linea": "#1D3355", "panel": "#0D1F38", "fondo": "#071426", "lateral": "#0A1A30",
        "neutro_bg": "#17304F", "control": "#10243F", "control_borde": "#26406A",
        "banda": "rgba(251,191,36,0.16)", "q50": "#FBBF24",
        "acento": "#FBBF24", "acento_texto": "#0B1F3A", "acento_sombra": "rgba(251,191,36,.30)",
        "calor_verde": "#10B981", "calor_ambar": "#334155", "calor_rojo": "#F97316",
        "scroll": "#3A5478", "scroll_hover": "#8193AD", "sombra": "0 1px 3px rgba(0,0,0,.35)",
        "logo_tenue": "#64748B",
    },
}
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
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: $gris !important; }

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
/* botones de dia anterior / siguiente / ultimo */
.st-key-dia_ant button, .st-key-dia_sig button, .st-key-dia_ult button {
  background: $control !important; color: $tinta !important; border: 1px solid $control_borde !important;
  border-radius: 10px !important; min-height: 2.5rem; padding: 0 8px !important; }
.st-key-dia_ant button p, .st-key-dia_sig button p, .st-key-dia_ult button p { color: $tinta !important; font-weight: 600; font-size: .86rem; }
.st-key-dia_ant button:hover:enabled, .st-key-dia_sig button:hover:enabled, .st-key-dia_ult button:hover:enabled {
  border-color: $acento !important; }
.st-key-dia_ant button:disabled, .st-key-dia_sig button:disabled, .st-key-dia_ult button:disabled { opacity: .45; }
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
/* marca augur (diseño 08 "Bandada en tendencia", dashboard/mockups/augur.html) */
.marca { display: flex; align-items: center; gap: 10px; }
.marca svg { flex: none; }
.marca-nom { font-family: 'Lexend', 'IBM Plex Sans', sans-serif; font-weight: 700; color: $tinta; line-height: 1; letter-spacing: -.01em; }
.marca-sub { font-size: .62rem; letter-spacing: .13em; text-transform: uppercase; color: $gris; font-weight: 600; margin-top: 5px; }
.marca-lateral { margin: 0 0 14px; }
.marca-lateral .marca-nom { font-size: 1.7rem; }
.marca-cab { margin-bottom: .55rem; }
.marca-cab .marca-nom { font-size: 2.1rem; }
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
    """P(desplome) y P(pico) por hora del pronostico de 24 h (scripts_experimento/eventos_escenarios_24h.py):
    desplome = precio <= 0,7 x mediana del dia; pico = precio >= 1,3 x mediana. None si no existe el archivo."""
    ruta = RAIZ / "data" / "processed" / "resultados" / "probabilidades_eventos_24h_2026.csv"
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
    if horizonte == "24h" and prob is not None:   # el detector de eventos se entreno con la ventana de 24 h
        out["p_desplome"] = prob["p_desplome"].reindex(out["fecha_hora"]).to_numpy()
        out["p_pico"] = prob["p_pico"].reindex(out["fecha_hora"]).to_numpy()
    else:
        out["p_desplome"], out["p_pico"] = np.nan, np.nan
    return out


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
    st.session_state["hora_operador"] = h


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


def franja_botones(dia_df, hora_sel, tk):
    """Franja horaria clickeable: cada hora es un boton coloreado segun su senal;
    comparte estado con el deslizador de hora (session_state["hora_operador"])."""
    color_bg = {"verde": tk["verde_bg"], "rojo": tk["rojo_bg"], "ambar": tk["ambar_bg"]}
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
            f'border-radius:6px !important; font-weight:700 !important; '
            f'font-size:.72rem !important; padding:9px 0 !important; min-height:0 !important; }}'
            f'.st-key-hora_c_{h} button p {{ color:{tk["tinta"]} !important; white-space:nowrap; '
            f'font-size:.72rem !important; }}'
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

    cols = st.columns(24, gap="small")
    for col, (_, r) in zip(cols, dia_df.iterrows()):
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
    return (f'<svg width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">{cero}'
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
    return f'<svg width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">{"".join(barras)}</svg>'


def tarjetas_html(items):
    piezas = "".join(
        f'<div class="stat"><div class="stat-l">{l}</div>'
        f'<div class="stat-fila"><div class="stat-v">{v}</div>{g}</div>'
        f'<div class="stat-s">{s}</div></div>'
        for l, v, g, s in items
    )
    return f'<div class="tarjetas">{piezas}</div>'


def estilo_figura(fig, tk, alto, titulo, margen_leyenda=64):
    # La leyenda se ancla al borde inferior del lienzo (no del area de datos) y se le
    # reserva margen propio, para que no se monte sobre las etiquetas del eje X.
    fig.update_layout(
        height=alto, title=dict(text=titulo, x=0.01, font=dict(size=15, color=tk["tinta"])),
        paper_bgcolor=tk["panel"], plot_bgcolor=tk["panel"], font=dict(color=tk["gris"], family=FUENTE),
        legend=dict(orientation="h", yref="container", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                    font=dict(color=tk["texto"], size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=46, l=10, r=10, b=margen_leyenda), hovermode="x unified",
        hoverlabel=dict(bgcolor=tk["panel"], bordercolor=tk["linea"], font=dict(color=tk["tinta"])),
        yaxis_title="COP/kWh",
    )
    for eje in (fig.update_xaxes, fig.update_yaxes):
        eje(automargin=True, gridcolor=tk["linea"], zerolinecolor=tk["linea"], linecolor=tk["linea"],
            tickfont=dict(color=tk["gris"]), title_font=dict(color=tk["gris"]))
    return fig


def trazar_banda(fig, df, tk):
    fig.add_trace(go.Scatter(x=df["fecha_hora"], y=df["q90"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["fecha_hora"], y=df["q10"], line=dict(width=0), fill="tonexty",
                             fillcolor=tk["banda"], name="Banda [q10, q90]", hoverinfo="skip"))
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


def grafico_dia(dia_df, bajo_s, alto_s, hora, tk):
    """Pronostico del dia elegido con los umbrales del motor y la hora seleccionada."""
    fig = go.Figure()
    trazar_banda(fig, dia_df, tk)
    fig.add_trace(go.Scatter(x=dia_df["fecha_hora"], y=alto_s.loc[dia_df.index], name="Umbral caro",
                             line=dict(color=tk["rojo"], width=1, dash="dash"),
                             hovertemplate="%{y:,.0f} COP/kWh"))
    fig.add_trace(go.Scatter(x=dia_df["fecha_hora"], y=bajo_s.loc[dia_df.index], name="Umbral barato",
                             line=dict(color=tk["verde"], width=1, dash="dash"),
                             hovertemplate="%{y:,.0f} COP/kWh"))
    for col, simbolo, color, nombre in (("p_desplome", "triangle-down", tk["rojo"], "Riesgo de desplome"),
                                        ("p_pico", "triangle-up", tk["verde"], "Riesgo de pico")):
        if col in dia_df and (dia_df[col] >= UMBRAL_ALERTA).any():
            m = dia_df[dia_df[col] >= UMBRAL_ALERTA]
            fig.add_trace(go.Scatter(x=m["fecha_hora"], y=m["q50"], mode="markers", name=nombre,
                                     marker=dict(symbol=simbolo, size=11, color=color, line=dict(width=1, color=tk["panel"])),
                                     customdata=m[col] * 100, hovertemplate=nombre + ": %{customdata:.0f} %<extra></extra>"))
    marca = dia_df["fecha_hora"].iloc[0].normalize() + pd.Timedelta(hours=int(hora))
    fig.add_vline(x=marca, line_color=tk["gris"], line_width=1)
    fig = estilo_figura(fig, tk, 340, "Pronóstico del día", margen_leyenda=96)
    fig.update_xaxes(tickformat="%H:%M")
    return fig


def metadatos_html(periodo, contenido):
    """Pie de cada imagen de decision con los metadatos que exige el Anexo 1 (unidad, periodo,
    confianza y version del modelo)."""
    cob = meta.get("cobertura_medida_pct", "n/d")
    return (f'<div class="meta"><b>{contenido}</b> &middot; Unidad: COP/kWh &middot; Periodo: {periodo} '
            f'&middot; Confianza: banda [q10, q90], cobertura medida {cob} % (objetivo '
            f'{meta.get("cobertura_objetivo_pct", 80)} %) &middot; Modelo ({horizonte}): '
            f'{meta.get("modelo", "n/d")}, versión {meta.get("generado", "n/d")}</div>')


CODIGO_SENAL = {"evitar_compra": 0, "retener": 0, "esperar": 1, "comprar": 2, "vender": 2}


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
    datos = np.dstack([piv("q50").to_numpy(), piv("real").to_numpy(), piv("q10").to_numpy(),
                       piv("q90").to_numpy(), piv("etq").to_numpy()])
    plantilla = ("%{y} · %{x}:00 h<br>%{customdata[4]}<br>Pronóstico: %{customdata[0]:,.0f} COP/kWh"
                 "<br>Banda: %{customdata[2]:,.0f} – %{customdata[3]:,.0f}<br>Real: %{customdata[1]:,.0f}"
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
                                   colorscale="YlOrRd", xgap=hueco_x, ygap=hueco_y, customdata=datos,
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
    fig = estilo_figura(fig, tk, alto, f"Mapa de calor día-hora · {modo.lower()}", margen_leyenda=60)
    fig.update_layout(hovermode="closest", yaxis_title=None, xaxis_title="Hora del día")
    fig.update_xaxes(dtick=1 if len(horas) <= 12 else 2, showgrid=False, range=[horas[0] - 0.5, horas[-1] + 0.5])
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category",
                     nticks=min(len(dias), 40))
    return fig


def tabla_metodos_html(tabla, metodo_activo, metodo_sugerido):
    filas = []
    for _, r in tabla.iterrows():
        m = r["metodo"]
        ventaja = "—" if pd.isna(r["ventaja_cop_kwh"]) else f"{r['ventaja_cop_kwh']:+,.1f}"
        etq_val = '<span class="etq etq-si">válido</span>' if r["valido"] else '<span class="etq etq-no">fuera de 10-40 %</span>'
        sug = '<span class="etq etq-sug">sugerido</span>' if m == metodo_sugerido else ""
        clase = ' class="activo"' if m == metodo_activo else ""
        filas.append(f"<tr{clase}><td>{m}{sug}</td><td>{ventaja}</td>"
                     f"<td>{r['frecuencia_accion'] * 100:.1f} %</td><td>{int(r['horas_accion']):,}</td>"
                     f"<td>{etq_val}</td></tr>")
    return ('<table class="tabla-met"><thead><tr><th>Método</th><th>Ventaja (COP/kWh)</th>'
            '<th>Frecuencia de acción</th><th>Horas de acción</th><th>Rango 10-40 %</th></tr></thead>'
            f'<tbody>{"".join(filas)}</tbody></table>')


# --------------------------------------------------------------------------
# Barra lateral y tema
# --------------------------------------------------------------------------
def logo_svg(tk, tam):
    """Logo de augur: tres aves que suben y crecen (gris, tinta, ambar), el augurio favorable como tendencia."""
    def ave(x, y, k, color, grosor):
        return (f'<path d="M{x - 6 * k} {y} C{x - 4 * k} {y - 4 * k} {x - 1.5 * k} {y - 4 * k} {x} {y + k} '
                f'C{x + 1.5 * k} {y - 4 * k} {x + 4 * k} {y - 4 * k} {x + 6 * k} {y}" stroke="{color}" '
                f'stroke-width="{grosor}"/>')
    return (f'<svg width="{tam}" height="{tam}" viewBox="0 0 48 48" fill="none" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-label="augur">'
            + ave(10, 37, .8, tk["logo_tenue"], 2.8) + ave(23, 26, 1.05, tk["tinta"], 3.4)
            + ave(36, 13, 1.3, tk["acento"], 3.8) + '</svg>')


def marca_html(tk, tam, bajada, clase):
    return (f'<div class="marca {clase}">{logo_svg(tk, tam)}<div><div class="marca-nom">augur</div>'
            f'<div class="marca-sub">{bajada}</div></div></div>')


if "modo_oscuro" not in st.session_state:
    st.session_state["modo_oscuro"] = st.query_params.get("tema") == "oscuro"

with st.sidebar:
    oscuro = st.session_state["modo_oscuro"]
    tk = TEMAS["oscuro" if oscuro else "claro"]
    st.markdown(marca_html(tk, 38, "Motor de decisión", "marca-lateral"), unsafe_allow_html=True)
    oscuro = st.toggle(":material/dark_mode: Modo oscuro", key="modo_oscuro")
    st.query_params["tema"] = "oscuro" if oscuro else "claro"
    tk = TEMAS["oscuro" if oscuro else "claro"]
    st.markdown(f"<style>{CSS_BASE.substitute(tk)}{CSS_OSCURO_EXTRA if oscuro else ''}</style>",
                unsafe_allow_html=True)

    vistas = ["Operador", "Analista"]
    # la URL solo fija la vista inicial; despues manda session_state. Si el indice se recalculara desde la
    # URL en cada ejecucion, Streamlit veria un control nuevo y el primer clic para volver se perderia
    if "vista_sel" not in st.session_state:
        st.session_state["vista_sel"] = "Analista" if st.query_params.get("vista") == "analista" else "Operador"
    vista = st.radio("Vista", vistas, key="vista_sel", captions=["Qué hago ahora", "Controles y backtest"])
    st.query_params["vista"] = vista.lower()
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
        f'<div class="modelo"><b>{estado}</b><br><span>{meta.get("modelo", "n/d")}</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Cobertura medida: {meta.get('cobertura_medida_pct', '-')}% "
        f"(objetivo {meta.get('cobertura_objetivo_pct', 80)}%)"
    )

st.markdown(marca_html(tk, 54, "Motor de decisión &middot; precio de bolsa de energía", "marca-cab"),
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-app">Qué conviene hacer con la energía en cada hora, según el pronóstico con '
    'bandas de incertidumbre. Prototipo OE3 &middot; no es asesoría financiera ni una simulación de portafolio.</div>',
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

    c1, c_ant, c_sig, c_ult, c2 = st.columns([2.1, .62, .62, .55, 1.4], vertical_alignment="bottom")
    with c1:
        dia = st.date_input("Día operativo", min_value=dia_min, max_value=dia_max, key="dia_operador")
        st.session_state["_copia_dia_operador"] = dia
    with c_ant:
        st.button("Anterior", icon=":material/chevron_left:", key="dia_ant", width="stretch",
                  on_click=_mover_dia, args=(-1, completos), disabled=dia <= dia_min, help="Día anterior")
    with c_sig:
        st.button("Siguiente", icon=":material/chevron_right:", key="dia_sig", width="stretch",
                  on_click=_mover_dia, args=(1, completos), disabled=dia >= dia_max, help="Día siguiente")
    with c_ult:
        st.button("Último", icon=":material/last_page:", key="dia_ult", width="stretch",
                  on_click=_mover_dia, args=("ultimo", completos), disabled=dia >= dia_max,
                  help="Último día con pronóstico")
    with c2:
        if "hora_operador" not in st.session_state:
            st.session_state["hora_operador"] = 8
        hora = st.number_input("Hora del día", 0, 23, step=1, key="hora_operador")

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
        grad = (f"linear-gradient(90deg,{tk['verde_bg']} 0%,{tk['verde_bg']} {pb:.1f}%,"
                f"{tk['neutro_bg']} {pb:.1f}%,{tk['neutro_bg']} {pa:.1f}%,"
                f"{tk['rojo_bg']} {pa:.1f}%,{tk['rojo_bg']} 100%)")
        gauge_html = f"""
  <div class="gauge-track" style="background:{grad}">
    <div class="gauge-marker" style="left:{pq:.1f}%"></div>
  </div>
  <div class="gauge-labels">
    <span>barato &le; {bajo_val:.0f}</span>
    <span>esperado ahora: <b>{q50:.0f} COP/kWh</b></span>
    <span>caro &ge; {alto_val:.0f}</span>
  </div>"""

    # riesgo de salto de la hora (desplome / pico) segun el detector de eventos
    riesgo_html = ""
    if not np.isnan(fila.get("p_desplome", np.nan)):
        p_d, p_p = float(fila["p_desplome"]), float(fila["p_pico"])
        def chip_riesgo(nombre, p, clase):
            estilo = f"chip {clase}" if p >= UMBRAL_ALERTA else "chip"
            fondo = "" if p >= UMBRAL_ALERTA else f' style="background:{tk["neutro_bg"]};color:{tk["texto"]}"'
            return f'<span class="{estilo}"{fondo}>{nombre} {p * 100:.0f} %</span>'
        horas_d = dia_df.loc[dia_df["p_desplome"] >= UMBRAL_ALERTA, "fecha_hora"].dt.hour.tolist()
        horas_p = dia_df.loc[dia_df["p_pico"] >= UMBRAL_ALERTA, "fecha_hora"].dt.hour.tolist()
        resumen = []
        if horas_d:
            resumen.append(f"posible desplome a las {', '.join(f'{h:02d}' for h in horas_d)} h")
        if horas_p:
            resumen.append(f"posible pico a las {', '.join(f'{h:02d}' for h in horas_p)} h")
        riesgo_html = (f'<div style="margin-top:10px">{chip_riesgo("&#9660; Desplome", p_d, "chip-baja")} '
                       f'{chip_riesgo("&#9650; Pico", p_p, "chip-alta")}'
                       f'<span class="chip-nota">riesgo de salto en esta hora'
                       f'{" &middot; hoy: " + "; ".join(resumen) if resumen else ""}</span></div>')

    # si ninguna regla fue estable en el backtest, se avisa en la tarjeta (no solo al final de la pagina)
    aviso_estable = ("" if es_valido else
                     f'<div class="chip-nota" style="margin:10px 0 0">&#9888; Ninguna regla fue estable en el backtest; '
                     f'se usa <b>{metodo}</b>, la menos inestable. Tómala con cautela (detalle al final de la página).</div>')

    # ---- Recomendacion + pronostico del dia ----
    col_hero, col_dia = st.columns([5, 4], gap="medium")
    with col_hero:
        st.markdown(f"""
<div class="hero hero-{color}">
  <div class="hero-tag">Recomendaci&oacute;n &middot; {dia:%d/%m/%Y} &middot; {hora:02d}:00 h &middot; rol {rol}</div>
  <div class="hero-accion">{titulo}</div>
  <div class="hero-frase">{frase}</div>
  {gauge_html}
  <div style="margin-top:14px">
    <span class="chip chip-{conf}">Confianza {conf}</span>
    <span class="chip-nota">banda de &plusmn;{ancho_t0/2:.0f} COP/kWh &mdash;
      relativo al precio, más ancha que el {conf_pct:.0f}% de las horas anteriores</span>{aviso_estable}
  </div>
  {riesgo_html}
</div>
""", unsafe_allow_html=True)
    with col_dia:
        st.plotly_chart(grafico_dia(dia_df, bajo_s, alto_s, hora, tk), width="stretch", theme=None,
                        config={"displayModeBar": False})
    st.markdown(metadatos_html(f"{dia:%d/%m/%Y}, 00:00 a 23:00 h",
                               "Semáforo y pronóstico del día con su banda de incertidumbre"),
                unsafe_allow_html=True)

    # ---- El dia hora por hora ----
    st.markdown('<div class="seccion">El día, hora por hora</div>', unsafe_allow_html=True)
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
    if dia_df["p_desplome"].notna().any():
        leyenda += (f" &#9660; / &#9650; sobre una hora = riesgo de desplome / pico de {UMBRAL_ALERTA * 100:.0f} % o más "
                    "(precio 30 % por debajo / por encima de la mediana del día).")
    st.markdown(f'<div class="leyenda">{leyenda}</div>', unsafe_allow_html=True)

    # ---- Mapa de calor dia-hora: rango de dias (de 1 dia al ultimo disponible) y de horas a eleccion ----
    st.markdown('<div class="seccion">Mapa de calor día-hora</div>', unsafe_allow_html=True)
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
    c_dias, c_horas, c_modo = st.columns([5, 3, 2])
    with c_dias:
        rango_dias = st.slider("Días", min_value=completos[0], max_value=completos[-1], key=clave_dias,
                               step=pd.Timedelta(days=1).to_pytimedelta(), format="DD/MM/YYYY",
                               help="Del primer día disponible al más reciente. Para ver un solo día, "
                                    "junta los dos extremos.")
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
    st.plotly_chart(grafico_calor(df, dias_calor, horas_calor, dia, hora, modo_calor, tk), width="stretch",
                    theme=None, config={"displayModeBar": False})
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
    st.markdown(tarjetas_html([
        ("Horas para actuar hoy", f"{n_accion_dia} de 24",
         mini_barras(ultimos.values, tk["verde"]),
         "cuadros verdes de la franja; barras: últimos 14 días"),
        ("Ventaja histórica de la regla",
         "&mdash;" if np.isnan(ventaja) else f"{ventaja:+.0f} COP/kWh",
         mini_linea(mensual["ventaja"], tk["q50"], tk["linea"]),
         f"habrías {verbo} que el promedio en el backtest 2026; línea: mes a mes"),
        ("Con qué frecuencia actúa",
         f"{bt['frecuencia_accion']*100:.0f}% del tiempo",
         mini_linea(mensual["frecuencia"], tk["ambar"], tk["linea"]),
         f"{bt['horas_accion']} horas de acción en todo el periodo; línea: mes a mes"),
    ]), unsafe_allow_html=True)

    # ---- Contexto: la semana alrededor del dia elegido ----
    ini = pd.Timestamp(dia) - pd.Timedelta(days=3)
    fin = pd.Timestamp(dia) + pd.Timedelta(days=4)
    ventana = df[(df["fecha_hora"] >= ini) & (df["fecha_hora"] < fin)]
    marca = pd.Timestamp(dia) + pd.Timedelta(hours=int(hora))
    st.plotly_chart(
        grafico_precio(ventana, tk, "La semana alrededor del día elegido", marca_x=marca),
        width="stretch", theme=None, config={"displaylogo": False},
    )
    st.markdown(metadatos_html(f"{ini:%d/%m/%Y} a {fin - pd.Timedelta(hours=1):%d/%m/%Y}",
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
            help=("fijo: percentiles del precio histórico 2019-2025. "
                  "rodante: percentiles con ventana móvil causal de 30 días. "
                  "banda: percentiles fijos + 'esperar' forzado cuando la banda de incertidumbre es ancha. "
                  "hibrido: umbral rodante + el mismo filtro de 'esperar' de banda.\n\n"
                  f"Sugerido por el backtest económico en el rango de fechas elegido: {etiqueta}."),
        )

    base = serie_con_senal(horizonte, rol, metodo, p_bajo, p_alto, intradia=INTRADIA)
    base = base[(base["fecha_hora"].dt.date >= inicio) & (base["fecha_hora"].dt.date <= fin)]
    resultado = evaluar_backtest(base, base["senal"], rol)

    accion_principal = "comprar" if rol == "comercializador" else "vender"
    ventaja = resultado["ventaja_cop_kwh"]
    st.markdown(tarjetas_html([
        (f"Ventaja económica ({accion_principal})",
         "&mdash;" if np.isnan(ventaja) else f"{ventaja:+.1f} COP/kWh", "",
         "precio medio en horas de acción frente al promedio del periodo filtrado"),
        ("Frecuencia de acción", f"{resultado['frecuencia_accion']*100:.1f}%", "",
         "rango útil: 10 % a 40 % de las horas"),
        ("Horas de acción", f"{resultado['horas_accion']:,}", "",
         f"del {inicio:%d/%m/%Y} al {fin:%d/%m/%Y}"),
    ]), unsafe_allow_html=True)

    st.markdown('<div class="seccion">Comparación de métodos en el rango elegido</div>', unsafe_allow_html=True)
    st.markdown(tabla_metodos_html(tabla_metodos, metodo, metodo_sugerido), unsafe_allow_html=True)
    st.markdown("")

    st.plotly_chart(
        grafico_precio(base, tk, f"Precio real, pronóstico y señales — método {metodo}", alto=460),
        width="stretch", theme=None, config={"displaylogo": False},
    )
    st.markdown(metadatos_html(f"{inicio:%d/%m/%Y} a {fin:%d/%m/%Y}",
                               f"Backtest del método {metodo} sobre real frente a pronóstico"),
                unsafe_allow_html=True)

    with st.expander("Ver datos filtrados"):
        st.dataframe(base, width="stretch")


if vista == "Operador":
    vista_operador()
else:
    vista_analista()
