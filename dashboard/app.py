# -*- coding: utf-8 -*-
"""
Dashboard del motor de decision (OE3).

Corre con:  streamlit run dashboard/app.py

Dos vistas, elegibles en la barra lateral:

  * "Operador" (por defecto): responde una sola pregunta -- "que hago ahora".
    Una tarjeta grande con la accion recomendada para la hora elegida, la
    franja del dia hora por hora, la confianza del pronostico y 3 indicadores.
    Sin perillas: el metodo de umbral lo elige solo el backtest economico.

  * "Analista": los controles finos (metodo manual, percentiles, rango de
    fechas, tabla comparativa de metodos). Es el dashboard de exploracion
    de siempre, para nosotros los que disenamos la regla.

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


def encontrar_raiz_proyecto(marcador="requirements.txt"):
    actual = Path(__file__).resolve().parent
    for carpeta in [actual, *actual.parents]:
        if (carpeta / marcador).exists():
            return carpeta
    raise FileNotFoundError(f"No encontre '{marcador}' subiendo desde {actual}")


RAIZ = encontrar_raiz_proyecto()
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from motor_decision import (  # noqa: E402
    generar_senales, evaluar_backtest, comparar_metodos, elegir_mejor_metodo,
    cargar_fuente_pronostico, umbrales_fijos, umbrales_rodantes,
)

st.set_page_config(
    page_title="Motor de decision - precio de bolsa",
    page_icon="⚡",
    layout="wide",
)

# --------------------------------------------------------------------------
# Sistema de diseno: tokens + hoja de estilo
# --------------------------------------------------------------------------
# Estos tokens son el puente con Figma: mismos nombres, mismos valores. Si el
# diseno cambia en Figma, se editan aca y toda la interfaz se mueve con ellos.
T = {
    "verde": "#15803D", "verde_bg": "#DCFCE7",
    "rojo": "#B91C1C", "rojo_bg": "#FEE2E2",
    "ambar": "#B45309", "ambar_bg": "#FEF3C7",
    "tinta": "#0F172A", "gris": "#64748B",
    "linea": "#E2E8F0", "panel": "#FFFFFF", "fondo": "#F1F5F9",
    "neutro_bg": "#EEF2F7",
}

CSS = Template("""
.stApp { background: $fondo; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1080px; }
#MainMenu, footer { visibility: hidden; }
h1, h2, h3, h4 { color: $tinta; }

.titulo-app { font-size: 1.55rem; font-weight: 800; color: $tinta; margin-bottom: .1rem; }
.sub-app { color: $gris; margin-bottom: 1.1rem; font-size: .95rem; }

.hero { border-radius: 18px; padding: 22px 24px; border: 1px solid $linea;
        background: $panel; margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(15,23,42,.05); }
.hero-verde { border-left: 7px solid $verde; }
.hero-rojo  { border-left: 7px solid $rojo; }
.hero-ambar { border-left: 7px solid $ambar; }
.hero-tag { font-size: .78rem; letter-spacing: .05em; text-transform: uppercase; color: $gris; }
.hero-accion { font-size: 2.05rem; font-weight: 800; margin: .2rem 0 .4rem; line-height: 1.1; }
.hero-verde .hero-accion { color: $verde; }
.hero-rojo  .hero-accion { color: $rojo; }
.hero-ambar .hero-accion { color: $ambar; }
.hero-frase { font-size: 1.02rem; color: #334155; max-width: 62ch; }

.gauge-track { position: relative; height: 14px; border-radius: 8px;
               margin: 20px 0 7px; border: 1px solid $linea; }
.gauge-marker { position: absolute; top: -6px; width: 3px; height: 24px;
                background: $tinta; border-radius: 2px; transform: translateX(-1.5px); }
.gauge-labels { display: flex; justify-content: space-between; font-size: .76rem; color: $gris; }

.chip { display: inline-block; padding: 3px 11px; border-radius: 999px;
        font-size: .78rem; font-weight: 700; }
.chip-alta  { background: $verde_bg; color: $verde; }
.chip-media { background: $ambar_bg; color: $ambar; }
.chip-baja  { background: $rojo_bg; color: $rojo; }
.chip-nota { color: $gris; font-size: .8rem; margin-left: 8px; }

.franja { display: flex; gap: 3px; margin: 8px 0 4px; flex-wrap: wrap; }
.celda { flex: 1 1 28px; min-width: 28px; text-align: center; font-size: .72rem;
         padding: 9px 0; border-radius: 6px; color: #1E293B; }
.celda-verde { background: $verde_bg; }
.celda-rojo  { background: $rojo_bg; }
.celda-ambar { background: $ambar_bg; }
.celda-sel { outline: 2px solid $tinta; font-weight: 800; }

.tarjetas { display: flex; gap: 12px; margin: 4px 0 8px; flex-wrap: wrap; }
.stat { flex: 1 1 180px; background: $panel; border: 1px solid $linea;
        border-radius: 14px; padding: 14px 16px; }
.stat-l { font-size: .8rem; color: $gris; }
.stat-v { font-size: 1.5rem; font-weight: 750; color: $tinta; margin: 2px 0; }
.stat-s { font-size: .78rem; color: $gris; }

.leyenda { font-size: .82rem; color: $gris; margin-top: 4px; }
.leyenda b { color: #334155; }
""").substitute(T)

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Semantica de cada senal (texto para humanos, no jerga de percentiles)
# --------------------------------------------------------------------------
# (titulo grande, color/tono)
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
PLOT_COLOR = {"verde": T["verde"], "rojo": T["rojo"], "ambar": T["ambar"]}


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
def comparacion(horizonte, rol, p_bajo, p_alto):
    """Corre el backtest de los 3 metodos y devuelve (tabla, metodo_sugerido, es_valido)."""
    bandas, _ = cargar_bandas(horizonte)
    tabla = comparar_metodos(bandas, rol, cargar_precio_historico(), p_bajo=p_bajo, p_alto=p_alto)
    metodo, es_valido = elegir_mejor_metodo(tabla)
    return tabla, metodo, es_valido


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


def umbrales_para_hora(bandas, metodo, precio_hist, idx, p_bajo, p_alto):
    """Devuelve (bajo, alto) efectivos en la fila `idx` de bandas para el metodo dado."""
    if metodo == "rodante":
        bajo_s, alto_s = umbrales_rodantes(bandas["q50"], 30, p_bajo, p_alto)
        return float(bajo_s.get(idx, np.nan)), float(alto_s.get(idx, np.nan))
    u = umbrales_fijos(precio_hist, p_bajo, p_alto)
    return u["bajo"], u["alto"]


def franja_html(dia_df, rol, hora_sel):
    celdas = []
    for _, r in dia_df.iterrows():
        h = int(r["fecha_hora"].hour)
        color = SIGNIFICADO[r["senal"]][1]
        sel = " celda-sel" if h == hora_sel else ""
        tip = f"{h:02d}:00 h - {SIGNIFICADO[r['senal']][0].lower()} (precio esperado {r['q50']:.0f} COP/kWh)"
        celdas.append(f'<div class="celda celda-{color}{sel}" title="{tip}">{h:02d}</div>')
    return '<div class="franja">' + "".join(celdas) + "</div>"


def tarjetas_html(items):
    piezas = "".join(
        f'<div class="stat"><div class="stat-l">{l}</div>'
        f'<div class="stat-v">{v}</div><div class="stat-s">{s}</div></div>'
        for l, v, s in items
    )
    return f'<div class="tarjetas">{piezas}</div>'


def grafico_precio(df, rol, titulo, alto=340, marca_x=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["fecha_hora"], y=df["q90"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["fecha_hora"], y=df["q10"], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(37,99,235,0.12)", name="Banda [q10,q90]", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["fecha_hora"], y=df["real"],
                             line=dict(color=T["tinta"], width=1.4), name="Precio real"))
    for senal, (etiqueta, color) in SIGNIFICADO.items():
        if senal == "esperar":
            continue
        sub = df[df["senal"] == senal]
        if len(sub):
            fig.add_trace(go.Scatter(
                x=sub["fecha_hora"], y=sub["real"], mode="markers",
                marker=dict(color=PLOT_COLOR[color], size=7), name=etiqueta.capitalize(),
            ))
    if marca_x is not None:
        fig.add_vline(x=marca_x, line_dash="dot", line_color=T["gris"])
    fig.update_layout(
        height=alto, title=titulo, xaxis_title=None, yaxis_title="COP/kWh",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=54, l=10, r=10, b=10),
    )
    fig.update_xaxes(gridcolor=T["linea"])
    fig.update_yaxes(gridcolor=T["linea"])
    return fig


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Motor de decisión")
    vista = st.radio(
        "Vista", ["Operador", "Analista"],
        captions=["Qué hago ahora", "Controles y backtest"],
    )
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
    if meta.get("calibrado"):
        st.success(f"Modelo activo: {meta.get('modelo', 'n/d')}")
    else:
        st.warning(f"Modelo activo (sin calibrar): {meta.get('modelo', 'n/d')}")
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
    tabla, metodo, es_valido = comparacion(horizonte, rol, 25, 75)

    fechas = bandas["fecha_hora"]
    dia_min, dia_max = fechas.dt.date.min(), fechas.dt.date.max()

    c1, c2 = st.columns([2, 1])
    with c1:
        dia = st.date_input("Día operativo", value=dia_max,
                            min_value=dia_min, max_value=dia_max)
    with c2:
        hora = st.slider("Hora del día", 0, 23, 8)

    # Señal para toda la serie con el método elegido por el backtest
    senal_full = generar_senales(bandas, metodo, rol, precio_hist, p_bajo=25, p_alto=75)
    df = bandas.assign(senal=senal_full)

    dia_df = df[df["fecha_hora"].dt.date == dia]
    if dia_df.empty:
        st.warning("No hay pronóstico para ese día.")
        return

    objetivo = dia_df[dia_df["fecha_hora"].dt.hour == hora]
    fila = objetivo.iloc[0] if len(objetivo) else dia_df.iloc[0]
    idx = fila.name
    senal = fila["senal"]
    q50 = float(fila["q50"])

    # Confianza a partir del ancho de la banda de esa hora
    anchos = (bandas["q90"] - bandas["q10"]).values
    ancho_t0 = float(fila["q90"] - fila["q10"])
    conf, conf_pct = nivel_confianza(ancho_t0, anchos)

    bajo_val, alto_val = umbrales_para_hora(bandas, metodo, precio_hist, idx, 25, 75)

    # Clasificacion "cruda" del precio (antes del filtro de banda ancha), para
    # poder explicar el caso en que 'banda' fuerza esperar pese a un precio extremo.
    if np.isnan(bajo_val) or np.isnan(alto_val):
        crudo = "normal"
    elif q50 <= bajo_val:
        crudo = "bajo"
    elif q50 >= alto_val:
        crudo = "alto"
    else:
        crudo = "normal"

    # ---- Tarjeta grande ----
    titulo, color = SIGNIFICADO[senal]
    if senal == "esperar" and metodo == "banda" and crudo != "normal":
        frase = (
            f"El precio esperado está {crudo}, pero el pronóstico de esta hora es muy incierto "
            "(banda ancha). El motor prefiere esperar antes que arriesgarse con una señal poco confiable."
        )
    else:
        frase = FRASE[senal]
    lo, hi = np.nanpercentile(precio_hist, [2, 98])

    if np.isnan(bajo_val) or np.isnan(alto_val):
        gauge_html = (
            '<div class="stat-s" style="margin-top:14px">Umbrales moviles aun sin definir '
            '(el metodo rodante necesita 30 dias de historia antes de operar).</div>'
        )
    else:
        pb = pos_en_barra(bajo_val, lo, hi)
        pa = pos_en_barra(alto_val, lo, hi)
        pq = pos_en_barra(q50, lo, hi)
        grad = (f"linear-gradient(90deg,{T['verde_bg']} 0%,{T['verde_bg']} {pb:.1f}%,"
                f"{T['neutro_bg']} {pb:.1f}%,{T['neutro_bg']} {pa:.1f}%,"
                f"{T['rojo_bg']} {pa:.1f}%,{T['rojo_bg']} 100%)")
        gauge_html = f"""
  <div class="gauge-track" style="background:{grad}">
    <div class="gauge-marker" style="left:{pq:.1f}%"></div>
  </div>
  <div class="gauge-labels">
    <span>barato &le; {bajo_val:.0f}</span>
    <span>esperado ahora: <b>{q50:.0f} COP/kWh</b></span>
    <span>caro &ge; {alto_val:.0f}</span>
  </div>"""

    st.markdown(f"""
<div class="hero hero-{color}">
  <div class="hero-tag">Recomendaci&oacute;n &middot; {dia:%d/%m/%Y} &middot; {hora:02d}:00 h &middot; rol {rol}</div>
  <div class="hero-accion">{titulo}</div>
  <div class="hero-frase">{frase}</div>
  {gauge_html}
  <div style="margin-top:14px">
    <span class="chip chip-{conf}">Confianza {conf}</span>
    <span class="chip-nota">banda de &plusmn;{ancho_t0/2:.0f} COP/kWh &mdash;
      mas ancha que el {conf_pct:.0f}% de las horas tipicas</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ---- El dia hora por hora ----
    st.markdown("**El día, hora por hora**")
    st.markdown(franja_html(dia_df, rol, hora), unsafe_allow_html=True)
    if rol == "comercializador":
        leyenda = ("<b>Verde</b> = hora para comprar en bolsa &nbsp;&middot;&nbsp; "
                   "<b>rojo</b> = hora para cubrirse con contratos &nbsp;&middot;&nbsp; "
                   "<b>ámbar</b> = esperar. Recuadro negro = la hora seleccionada.")
    else:
        leyenda = ("<b>Verde</b> = hora para despachar y vender &nbsp;&middot;&nbsp; "
                   "<b>rojo</b> = hora para retener generación &nbsp;&middot;&nbsp; "
                   "<b>ámbar</b> = esperar. Recuadro negro = la hora seleccionada.")
    st.markdown(f'<div class="leyenda">{leyenda}</div>', unsafe_allow_html=True)

    # ---- Indicadores del dia y del backtest ----
    n_accion_dia = int(dia_df["senal"].isin(["comprar", "vender"]).sum())
    bt = evaluar_backtest(df, df["senal"], rol)
    ventaja = bt["ventaja_cop_kwh"]
    verbo = "comprado más barato" if rol == "comercializador" else "vendido más caro"
    st.markdown(tarjetas_html([
        ("Horas para actuar hoy", f"{n_accion_dia} de 24",
         "cuadros verdes en la franja de arriba"),
        ("Ventaja histórica de la regla",
         "&mdash;" if np.isnan(ventaja) else f"{ventaja:+.0f} COP/kWh",
         f"habrías {verbo} que el promedio, en el backtest 2026"),
        ("Con qué frecuencia actúa",
         f"{bt['frecuencia_accion']*100:.0f}% del tiempo",
         f"{bt['horas_accion']} horas de acción en todo el periodo"),
    ]), unsafe_allow_html=True)

    # ---- Contexto: la semana alrededor del dia elegido ----
    ini = pd.Timestamp(dia) - pd.Timedelta(days=3)
    fin = pd.Timestamp(dia) + pd.Timedelta(days=4)
    ventana = df[(df["fecha_hora"] >= ini) & (df["fecha_hora"] < fin)]
    marca = pd.Timestamp(dia) + pd.Timedelta(hours=int(hora))
    st.plotly_chart(
        grafico_precio(ventana, rol, "La semana alrededor del día elegido", marca_x=marca),
        width='stretch',
    )

    # ---- Nota sobre la regla activa ----
    if es_valido:
        st.caption(
            f"Regla activa: **{metodo}**. La elige el motor automáticamente porque dio la mejor "
            "ventaja económica en el backtest 2026 entre los métodos que actúan una fracción "
            "razonable del tiempo (10-40%). Para ver la comparación o forzar otro método, entra a la vista Analista."
        )
    else:
        st.warning(
            f"Regla activa: **{metodo}** (elegida por mayor ventaja, pero ningún método cayó en el "
            "rango de frecuencia 10-40% para este rol/horizonte). Revisa la vista Analista antes de confiar en la señal."
        )


# ==========================================================================
# VISTA ANALISTA
# ==========================================================================
def vista_analista():
    with st.sidebar:
        st.divider()
        st.subheader("Controles de análisis")
        p_bajo, p_alto = st.slider("Percentiles del umbral (bajo / alto)", 5, 95, (25, 75), step=5)

        tabla_metodos, metodo_sugerido, es_valido = comparacion(horizonte, rol, p_bajo, p_alto)
        etiqueta = (f"{metodo_sugerido} (sugerido)" if es_valido
                    else f"{metodo_sugerido} (sugerido, sin método 100% válido)")
        opciones = ["fijo", "rodante", "banda"]
        metodo = st.selectbox(
            "Método de umbral", opciones, index=opciones.index(metodo_sugerido),
            help=("fijo: percentiles del precio histórico 2019-2025. "
                  "rodante: percentiles con ventana móvil causal de 30 días. "
                  "banda: percentiles fijos + 'esperar' forzado cuando la banda de incertidumbre es ancha.\n\n"
                  f"Sugerido por el backtest económico: {etiqueta}."),
        )

        rango = bandas["fecha_hora"].dt.date
        fecha_min, fecha_max = rango.min(), rango.max()
        inicio, fin = st.slider("Rango de fechas", min_value=fecha_min, max_value=fecha_max,
                                value=(fecha_min, fecha_max))

    senal = generar_senales(bandas, metodo, rol, precio_hist, p_bajo=p_bajo, p_alto=p_alto)
    base = bandas.assign(senal=senal)
    base = base[(base["fecha_hora"].dt.date >= inicio) & (base["fecha_hora"].dt.date <= fin)]
    resultado = evaluar_backtest(base, base["senal"], rol)

    accion_principal = "comprar" if rol == "comercializador" else "vender"
    ventaja = resultado["ventaja_cop_kwh"]
    col1, col2, col3 = st.columns(3)
    col1.metric(
        f"Ventaja económica ({accion_principal})",
        f"{ventaja:.1f} COP/kWh" if not np.isnan(ventaja) else "-",
        help="Precio real promedio en horas de acción vs. promedio general del periodo filtrado.",
    )
    col2.metric("Frecuencia de acción", f"{resultado['frecuencia_accion']*100:.1f}%")
    col3.metric("Horas de acción", f"{resultado['horas_accion']}")

    with st.expander("Ver comparación completa de métodos", expanded=True):
        st.dataframe(tabla_metodos, width='stretch')

    st.plotly_chart(
        grafico_precio(base, rol, f"Precio real, banda y señales — método {metodo}", alto=460),
        width='stretch',
    )

    with st.expander("Ver datos filtrados"):
        st.dataframe(base, width='stretch')


if vista == "Operador":
    vista_operador()
else:
    vista_analista()
