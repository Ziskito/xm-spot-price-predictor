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
)

st.set_page_config(
    page_title="Motor de decision - precio de bolsa",
    page_icon="⚡",
    layout="wide",
)

# --------------------------------------------------------------------------
# Sistema de diseno: tokens por tema + hoja de estilo
# --------------------------------------------------------------------------
TEMAS = {
    "claro": {
        "verde": "#15803D", "verde_bg": "#DCFCE7",
        "rojo": "#B91C1C", "rojo_bg": "#FEE2E2",
        "ambar": "#B45309", "ambar_bg": "#FEF3C7",
        "tinta": "#0F172A", "texto": "#334155", "gris": "#64748B",
        "linea": "#E2E8F0", "panel": "#FFFFFF", "fondo": "#F1F5F9", "lateral": "#FFFFFF",
        "neutro_bg": "#E8EDF3", "control": "#FFFFFF", "control_borde": "#CBD5E1",
        "banda": "rgba(37,99,235,0.13)", "q50": "#2563EB", "acento": "#2563EB", "scroll": "#94A3B8", "scroll_hover": "#64748B", "sombra": "0 1px 2px rgba(15,23,42,.06)",
    },
    "oscuro": {
        "verde": "#34D399", "verde_bg": "rgba(52,211,153,0.16)",
        "rojo": "#F87171", "rojo_bg": "rgba(248,113,113,0.16)",
        "ambar": "#FBBF24", "ambar_bg": "rgba(251,191,36,0.16)",
        "tinta": "#E6EAF2", "texto": "#C3CBD9", "gris": "#8D99AE",
        "linea": "#24304A", "panel": "#111A2E", "fondo": "#0A1120", "lateral": "#0D1527",
        "neutro_bg": "#1C2740", "control": "#16213A", "control_borde": "#2C3A58",
        "banda": "rgba(96,165,250,0.20)", "q50": "#60A5FA", "acento": "#3B82F6", "scroll": "#5B6B88", "scroll_hover": "#8D99AE", "sombra": "0 1px 3px rgba(0,0,0,.35)",
    },
}

CSS_BASE = Template("""
.stApp { background: $fondo; color: $tinta; }
[data-testid="stHeader"] { background: transparent; }
/* boton para abrir/cerrar la barra lateral: color de acento para que se vea a la primera */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button, [data-testid="stHeader"] button {
  background: $acento !important; border: none !important; border-radius: 10px !important;
  color: #FFFFFF !important; opacity: 1 !important; padding: 6px 12px 6px 8px !important;
  width: auto !important; height: auto !important;
  box-shadow: 0 2px 10px rgba(37,99,235,.45) !important; }
[data-testid="stExpandSidebarButton"]:hover, [data-testid="stHeader"] button:hover {
  filter: brightness(1.12); }
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapseButton"] button *,
[data-testid="stHeader"] button * { color: #FFFFFF !important; fill: #FFFFFF !important; opacity: 1 !important; }
[data-testid="stExpandSidebarButton"]::after { content: "Menú"; font-weight: 700; font-size: .9rem;
  margin-left: 6px; color: #FFFFFF; }
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
[data-testid="stExpander"] details { background: $panel; border: 1px solid $linea; border-radius: 14px; }
[data-testid="stExpander"] summary p { color: $tinta !important; }
[data-testid="stTooltipIcon"] svg { stroke: $gris; }

.titulo-app { font-size: 1.55rem; font-weight: 800; color: $tinta; margin-bottom: .1rem; }
.sub-app { color: $gris; margin-bottom: 1.1rem; font-size: .95rem; }
.seccion { font-weight: 700; color: $tinta; margin: 10px 0 4px; }

.hero { border-radius: 18px; padding: 20px 22px; border: 1px solid $linea;
        background: $panel; margin-bottom: 16px; box-shadow: $sombra; min-height: 340px; }
.hero-verde { border-left: 7px solid $verde; }
.hero-rojo  { border-left: 7px solid $rojo; }
.hero-ambar { border-left: 7px solid $ambar; }
.hero-tag { font-size: .76rem; letter-spacing: .05em; text-transform: uppercase; color: $gris; }
.hero-accion { font-size: 2.05rem; font-weight: 800; margin: .25rem 0 .45rem; line-height: 1.1; }
.hero-verde .hero-accion { color: $verde; }
.hero-rojo  .hero-accion { color: $rojo; }
.hero-ambar .hero-accion { color: $ambar; }
.hero-frase { font-size: 1rem; color: $texto; max-width: 62ch; }

.gauge-track { position: relative; height: 14px; border-radius: 8px;
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
        border-radius: 14px; padding: 14px 16px; box-shadow: $sombra; }
.stat-l { font-size: .8rem; color: $gris; }
.stat-fila { display: flex; align-items: flex-end; justify-content: space-between; gap: 10px; }
.stat-v { font-size: 1.5rem; font-weight: 750; color: $tinta; margin: 2px 0; white-space: nowrap; }
.stat-s { font-size: .78rem; color: $gris; }

.leyenda { font-size: .82rem; color: $gris; margin-top: 4px; }
.leyenda b { color: $tinta; }

[data-testid="stPlotlyChart"] { background: $panel; border: 1px solid $linea; border-radius: 14px;
                                padding: 6px 6px 2px; box-shadow: $sombra; }

.modelo { background: $verde_bg; border-radius: 12px; padding: 12px 14px; font-size: .86rem; }
.modelo b { color: $verde !important; }
.modelo span { color: $texto !important; }

.tabla-met { width: 100%; border-collapse: separate; border-spacing: 0; background: $panel;
             border: 1px solid $linea; border-radius: 14px; overflow: hidden; font-size: .9rem; }
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
def comparacion(horizonte, rol, p_bajo, p_alto, inicio=None, fin=None):
    """Corre el backtest de los 4 metodos y devuelve (tabla, metodo_sugerido, es_valido).

    Si se dan inicio/fin (fechas), la senal se calcula sobre TODA la serie
    (continuidad para "rodante"/"hibrido") pero el backtest -- y por lo tanto
    que metodo queda como "sugerido" -- se mide solo en ese rango."""
    bandas, _ = cargar_bandas(horizonte)
    mascara = None
    if inicio is not None and fin is not None:
        mascara = (bandas["fecha_hora"].dt.date >= inicio) & (bandas["fecha_hora"].dt.date <= fin)
    tabla = comparar_metodos(bandas, rol, cargar_precio_historico(),
                             p_bajo=p_bajo, p_alto=p_alto, mascara_evaluacion=mascara)
    metodo, es_valido = elegir_mejor_metodo(tabla)
    return tabla, metodo, es_valido


@st.cache_data
def comparacion_estable(horizonte, rol):
    """Elige el metodo que mejor se sostiene en su mitad mas debil del periodo
    (criterio maximin, ver comparar_metodos_estable() en motor_decision.py)."""
    bandas, _ = cargar_bandas(horizonte)
    tabla = comparar_metodos_estable(bandas, rol, cargar_precio_historico())
    metodo, es_estable = elegir_mejor_metodo_estable(tabla)
    return tabla, metodo, es_estable


@st.cache_data
def serie_con_senal(horizonte, rol, metodo, p_bajo=25, p_alto=75):
    bandas, _ = cargar_bandas(horizonte)
    senal = generar_senales(bandas, metodo, rol, cargar_precio_historico(), p_bajo=p_bajo, p_alto=p_alto)
    return bandas.assign(senal=senal)


@st.cache_data
def resumen_mensual(horizonte, rol, metodo):
    """Ventaja y frecuencia de accion del metodo, mes a mes (para las mini graficas)."""
    df = serie_con_senal(horizonte, rol, metodo)
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
    al resto de horas. Banda ancha -> el modelo esta menos seguro -> confianza baja."""
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
        return umbrales_rodantes(bandas["q50"], 30, p_bajo, p_alto)
    u = umbrales_fijos(precio_hist, p_bajo, p_alto)
    return (pd.Series(u["bajo"], index=bandas.index), pd.Series(u["alto"], index=bandas.index))


def _set_hora(h):
    st.session_state["hora_operador"] = h


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
        paper_bgcolor=tk["panel"], plot_bgcolor=tk["panel"], font=dict(color=tk["gris"]),
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
    marca = dia_df["fecha_hora"].iloc[0].normalize() + pd.Timedelta(hours=int(hora))
    fig.add_vline(x=marca, line_color=tk["gris"], line_width=1)
    fig = estilo_figura(fig, tk, 340, "Pronóstico del día", margen_leyenda=96)
    fig.update_xaxes(tickformat="%H:%M")
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
if "modo_oscuro" not in st.session_state:
    st.session_state["modo_oscuro"] = st.query_params.get("tema") == "oscuro"

with st.sidebar:
    st.markdown("### ⚡ Motor de decisión")
    oscuro = st.toggle(":material/dark_mode: Modo oscuro", key="modo_oscuro")
    st.query_params["tema"] = "oscuro" if oscuro else "claro"
    tk = TEMAS["oscuro" if oscuro else "claro"]
    st.markdown(f"<style>{CSS_BASE.substitute(tk)}{CSS_OSCURO_EXTRA if oscuro else ''}</style>",
                unsafe_allow_html=True)

    vistas = ["Operador", "Analista"]
    inicial = 1 if st.query_params.get("vista") == "analista" else 0
    vista = st.radio("Vista", vistas, index=inicial, captions=["Qué hago ahora", "Controles y backtest"])
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

st.markdown('<div class="titulo-app">Motor de decisión &middot; precio de bolsa de energía</div>',
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
    tabla, metodo, es_valido = comparacion_estable(horizonte, rol)
    df = serie_con_senal(horizonte, rol, metodo)

    # Solo se ofrecen dias con las 24 horas: segun el modelo, el pronostico puede cubrir 01:00 a 00:00
    # del dia siguiente y dejar el primer o el ultimo dia con una sola hora.
    horas_por_dia = df["fecha_hora"].dt.date.value_counts()
    completos = sorted(horas_por_dia[horas_por_dia >= 24].index) or sorted(horas_por_dia.index)
    dia_min, dia_max = completos[0], completos[-1]

    c1, c2 = st.columns([2, 1])
    with c1:
        dia = st.date_input("Día operativo", value=dia_max, min_value=dia_min, max_value=dia_max)
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

    anchos = (df["q90"] - df["q10"]).values
    ancho_t0 = float(fila["q90"] - fila["q10"])
    conf, conf_pct = nivel_confianza(ancho_t0, anchos)

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
      más ancha que el {conf_pct:.0f}% de las horas típicas</span>
  </div>
</div>
""", unsafe_allow_html=True)
    with col_dia:
        st.plotly_chart(grafico_dia(dia_df, bajo_s, alto_s, hora, tk), width="stretch", theme=None,
                        config={"displayModeBar": False})

    # ---- El dia hora por hora ----
    st.markdown('<div class="seccion">El día, hora por hora</div>', unsafe_allow_html=True)
    franja_botones(dia_df, hora, tk)
    if rol == "comercializador":
        leyenda = ("<b>Verde</b> = hora para comprar en bolsa &nbsp;&middot;&nbsp; "
                   "<b>rojo</b> = hora para cubrirse con contratos &nbsp;&middot;&nbsp; "
                   "<b>ámbar</b> = esperar. Recuadro con borde = la hora seleccionada &mdash; "
                   "haz clic en cualquier hora para elegirla directamente.")
    else:
        leyenda = ("<b>Verde</b> = hora para despachar y vender &nbsp;&middot;&nbsp; "
                   "<b>rojo</b> = hora para retener generación &nbsp;&middot;&nbsp; "
                   "<b>ámbar</b> = esperar. Recuadro con borde = la hora seleccionada &mdash; "
                   "haz clic en cualquier hora para elegirla directamente.")
    st.markdown(f'<div class="leyenda">{leyenda}</div>', unsafe_allow_html=True)

    # ---- Indicadores del dia y del backtest, con mini graficas ----
    accion = "comprar" if rol == "comercializador" else "vender"
    por_dia = (df.assign(d=df["fecha_hora"].dt.date, a=df["senal"].eq(accion))
                 .groupby("d")["a"].sum())
    ultimos = por_dia.loc[:dia].tail(14)
    n_accion_dia = int(por_dia.get(dia, 0))
    bt = evaluar_backtest(df, df["senal"], rol)
    ventaja = bt["ventaja_cop_kwh"]
    mensual = resumen_mensual(horizonte, rol, metodo)
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

    # ---- Nota sobre la regla activa ----
    if es_valido:
        st.caption(
            f"Regla activa: **{metodo}**. La elige el motor automáticamente porque fue la más "
            "estable en el backtest 2026: mantuvo ventaja económica positiva y una frecuencia de "
            "acción razonable (10-40%) en las dos mitades del periodo, no solo en el promedio del "
            "año completo. Para ver la comparación o forzar otro método, entra a la vista Analista."
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

        tabla_metodos, metodo_sugerido, es_valido = comparacion(horizonte, rol, p_bajo, p_alto, inicio, fin)
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

    base = serie_con_senal(horizonte, rol, metodo, p_bajo, p_alto)
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

    with st.expander("Ver datos filtrados"):
        st.dataframe(base, width="stretch")


if vista == "Operador":
    vista_operador()
else:
    vista_analista()
