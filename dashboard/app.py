# -*- coding: utf-8 -*-
"""
Prototipo de dashboard para el motor de decision (OE3).

Corre con: streamlit run dashboard/app.py

Carga las bandas de pronostico a traves de cargar_fuente_pronostico() (src/motor_decision.py), es
decir, SIEMPRE segun lo que diga data/processed/resultados/fuentes_pronostico.json -- si ese archivo
cambia (ej. 72h pasa de XGBoost a N-BEATSx), este dashboard lo refleja sin tocar una sola linea de
codigo aqui. Ver docs/motor_decision_guia.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def encontrar_raiz_proyecto(marcador="requirements.txt"):
    actual = Path(__file__).resolve().parent
    for carpeta in [actual, *actual.parents]:
        if (carpeta / marcador).exists():
            return carpeta
    raise FileNotFoundError(f"No encontré '{marcador}' subiendo desde {actual}")


RAIZ = encontrar_raiz_proyecto()
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from motor_decision import (  # noqa: E402
    generar_senales, evaluar_backtest, comparar_metodos, elegir_mejor_metodo,
    cargar_fuente_pronostico,
)

st.set_page_config(page_title="Motor de decisión — precio de bolsa", layout="wide")

COLORES_SEÑAL = {
    "comprar": "#2ca02c", "evitar_compra": "#d62728",
    "vender": "#2ca02c", "retener": "#d62728",
    "esperar": "#7f7f7f",
}


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


st.title("Motor de decisión — precio de bolsa de energía")
st.caption(
    "Prototipo OE3. Explora la señal comprar/vender/esperar sobre el holdout 2026, "
    "por rol, horizonte y método de umbral. No es asesoría financiera ni una simulación de portafolio."
)

precio_historico_train = cargar_precio_historico()

# --- Controles ---
with st.sidebar:
    st.header("Configuración")
    rol = st.radio("Rol", ["generador", "comercializador"], help=(
        "Generador: vende energía, busca precios altos. "
        "Comercializador: compra energía, busca precios bajos."
    ))
    horizonte = st.radio("Horizonte de pronóstico", ["24h", "72h"])

bandas, meta = cargar_bandas(horizonte)

if bandas is None:
    st.error(
        f"No pude cargar la fuente de pronóstico para {horizonte}: {meta.get('error')}\n\n"
        "Revisa `data/processed/resultados/fuentes_pronostico.json` y corre "
        "`notebooks/12_motor_decision_Rafa.ipynb` si hace falta regenerar el CSV."
    )
    st.stop()

with st.sidebar:
    if meta.get("calibrado"):
        st.success(f"Modelo activo: {meta['modelo']}")
    else:
        st.warning(f"Modelo activo (sin calibrar): {meta['modelo']}")
    st.caption(f"Cobertura medida: {meta.get('cobertura_medida_pct', '—')}% (objetivo {meta.get('cobertura_objetivo_pct', 80)}%)")

    # --- Método sugerido, calculado en vivo con el backtest económico ---
    tabla_metodos = comparar_metodos(bandas, rol, precio_historico_train)
    metodo_sugerido, es_valido = elegir_mejor_metodo(tabla_metodos)
    etiqueta_sugerido = f"{metodo_sugerido} (sugerido)" if es_valido else f"{metodo_sugerido} (sugerido, sin método 100% válido)"

    opciones_metodo = ["fijo", "rodante", "banda"]
    metodo = st.selectbox(
        "Método de umbral", opciones_metodo, index=opciones_metodo.index(metodo_sugerido),
        help=(
            "fijo: percentiles del precio histórico 2019-2025. "
            "rodante: percentiles con ventana móvil causal de 30 días. "
            "banda: percentiles fijos + 'esperar' forzado cuando la banda de incertidumbre es ancha.\n\n"
            f"Sugerido por el backtest económico para este rol/horizonte: {etiqueta_sugerido}."
        ),
    )
    with st.expander("Ver comparación completa de métodos"):
        st.dataframe(tabla_metodos, width="stretch")

    p_bajo, p_alto = st.slider("Percentiles del umbral (bajo / alto)", 5, 95, (25, 75), step=5)

    rango_fechas = bandas["fecha_hora"].dt.date
    fecha_min, fecha_max = rango_fechas.min(), rango_fechas.max()
    inicio, fin = st.slider(
        "Rango de fechas", min_value=fecha_min, max_value=fecha_max,
        value=(fecha_min, fecha_max),
    )

# --- Recalcular la señal en vivo con los controles elegidos ---
senal = generar_senales(bandas, metodo, rol, precio_historico_train, p_bajo=p_bajo, p_alto=p_alto)
base = bandas.assign(senal=senal)
base = base[(base["fecha_hora"].dt.date >= inicio) & (base["fecha_hora"].dt.date <= fin)]

resultado = evaluar_backtest(base, base["senal"], rol)

# --- KPIs ---
col1, col2, col3 = st.columns(3)
accion_principal = "comprar" if rol == "comercializador" else "vender"
ventaja = resultado["ventaja_cop_kwh"]
col1.metric(
    f"Ventaja económica ({accion_principal})",
    f"{ventaja:.1f} COP/kWh" if not np.isnan(ventaja) else "—",
    help="Precio real promedio en horas de acción vs. promedio general del período filtrado.",
)
col2.metric("Frecuencia de acción", f"{resultado['frecuencia_accion']*100:.1f}%")
col3.metric("Horas de acción", f"{resultado['horas_accion']}")

# --- Gráfica principal ---
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=base["fecha_hora"], y=base["q90"], line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=base["fecha_hora"], y=base["q10"], line=dict(width=0), fill="tonexty",
    fillcolor="rgba(31,119,180,0.15)", name="Banda [q10,q90]", hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=base["fecha_hora"], y=base["real"], line=dict(color="black", width=1), name="Precio real",
))
for valor_senal, color in COLORES_SEÑAL.items():
    subset = base[base["senal"] == valor_senal]
    if len(subset) and valor_senal != "esperar":
        fig.add_trace(go.Scatter(
            x=subset["fecha_hora"], y=subset["real"], mode="markers",
            marker=dict(color=color, size=6), name=valor_senal,
        ))

fig.update_layout(
    height=500, xaxis_title="Fecha", yaxis_title="COP/kWh",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=40),
)
st.plotly_chart(fig, width="stretch")

with st.expander("Ver datos filtrados"):
    st.dataframe(base, width="stretch")
