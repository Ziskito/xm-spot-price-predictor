# -*- coding: utf-8 -*-
"""
Figuras del informe final (ABET), 15-oct-2026: diagrama de bloques del diseno definitivo, motor en operacion con el
pronostico realista (mismo codigo y parametros del dashboard) y capturas actuales del dashboard (las del manual de
usuario, generar_manual_usuario.py). Salida: scripts_experimento/informe_final_assets/
"""
import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "src"))
from motor_decision import generar_senales, precio_previo_ventana, umbrales_rodantes  # noqa: E402

OUT = AQUI / "informe_final_assets"
OUT.mkdir(exist_ok=True)
AZUL, NARANJA, VERDE, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#898781"

# ---------------------------------------------------------------- 1) diagrama de bloques del diseno definitivo
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Calibri", "Arial"], "font.size": 10})
fig, ax = plt.subplots(figsize=(8.4, 6.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def caja(x, y, w, h, titulo, texto, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=color + "22", ec=color, lw=1.4))
    ax.text(x + w / 2, y + h - 3.2, titulo, ha="center", va="top", fontsize=10.5, fontweight="bold", color="#1b1b1b")
    ax.text(x + w / 2, y + h - 8.2, texto, ha="center", va="top", fontsize=8.4, color="#333333", linespacing=1.25)


def flecha(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.3))


AZ, NA, VE = "#2a78d6", "#eb6834", "#1baf7a"
ax.text(2, 98, "OE1 · Procesamiento de datos", fontsize=9.5, color=AZ, fontweight="bold")
caja(2, 73, 30, 22, "Adquisición", "API de métricas de XM (precio,\ndemanda, generación, embalses,\naportes) · portal XM (predespacho,\ndMAR, dDEM) · ONI de NOAA", AZ)
caja(35, 73, 30, 22, "Sincronización", "Dataset horario oficial\nene-2019 a ago-2026 (66.576 h)\n+ serie de operación que se\nactualiza cada día", AZ)
caja(68, 73, 30, 22, "Caracterización", "Estadística, estacionalidad,\nperiodograma, filtros de\ntendencia, correlación cruzada", AZ)
flecha(32.8, 84, 34.2, 84)
flecha(65.8, 84, 67.2, 84)
ax.text(2, 68, "OE2 · Pronóstico", fontsize=9.5, color=NA, fontweight="bold")
caja(2, 41, 30, 24, "Información al corte", "Solo lo publicado a las 00:00:\nprecio y demanda hasta D-4,\nhidrología D-2, ONI M-2,\npredespacho de D (sale en D-1)", NA)
caja(35, 41, 30, 24, "Modelos", "Predespacho ideal de XM\n+ 3 CatBoost (pérdida MAE,\nlog precio/predespacho)\nreentrenados cada mes", NA)
caja(68, 41, 30, 24, "Combinación y bandas", "QRA por franja de 6 h\n(objetivo MAE) · bandas\nconformes por hora · 72 h ·\nprobabilidad de pico/desplome", NA)
flecha(83, 72.2, 17, 65.8)
flecha(32.8, 53, 34.2, 53)
flecha(65.8, 53, 67.2, 53)
ax.text(2, 36, "OE3 / OE4 · Decisión y validación", fontsize=9.5, color=VE, fontweight="bold")
caja(2, 10, 30, 23, "Contrato de datos", "fuentes_pronostico.json\nfecha_hora, real, q10,\nq50, q90 por horizonte", VE)
caja(35, 10, 30, 23, "Motor de decisión", "4 métodos por percentil,\nfiltro intradía y selección\npor estabilidad; comprar ·\nvender · esperar", VE)
caja(68, 10, 30, 23, "Dashboard", "augurpredictor.streamlit.app\nOperador y Analista ·\nbacktest · actualización diaria\n(GitHub Actions)", VE)
flecha(83, 40.2, 17, 33.8)
flecha(32.8, 21.5, 34.2, 21.5)
flecha(65.8, 21.5, 67.2, 21.5)
ax.text(50, 3, "Python de código abierto, versionado en GitHub; solo datos públicos disponibles antes del corte.",
        ha="center", fontsize=8.6, color="#555555", style="italic")
fig.savefig(OUT / "diagrama_bloques.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 2) motor en operacion (contrato realista de 24 h)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"], "font.size": 11,
                     "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})
RES = RAIZ / "data" / "processed" / "resultados"
df = pd.read_csv(RES / "pronostico_realista_bandas_24h_2026.csv", parse_dates=["fecha_hora"])
df = df[df["fecha_hora"] <= "2026-08-05 00:00"].reset_index(drop=True)[["fecha_hora", "real", "q10", "q50", "q90"]]
hist = pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", usecols=["precio_bolsa"])["precio_bolsa"].values
previo = pd.concat([pd.read_csv(RAIZ / "data/processed" / f, usecols=["fecha_hora", "precio_bolsa"], parse_dates=["fecha_hora"])
                    for f in ("dataset_maestro_2019_2025.csv", "dataset_maestro_2026.csv")]).drop_duplicates("fecha_hora")
previo = previo.set_index("fecha_hora")["precio_bolsa"].sort_index()
senal = generar_senales(df, "rodante", "generador", hist, precio_previo=previo, intradia=True)
bajo, alto = umbrales_rodantes(df["q50"], 30, 25, 75, previo=precio_previo_ventana(df, previo, 30))
ini = pd.Timestamp("2026-07-13")
sem = ((df["fecha_hora"] >= ini) & (df["fecha_hora"] < ini + pd.Timedelta(days=7))).to_numpy()
d = df[sem]
fig, ax = plt.subplots(figsize=(8.2, 3.8))
ax.fill_between(d.fecha_hora, d.q10, d.q90, color=AZUL, alpha=0.13, lw=0, label="Banda [q10, q90]")
ax.plot(d.fecha_hora, d.real, color="#111111", lw=1.0, label="Precio real")
ax.plot(d.fecha_hora, d.q50, color=AZUL, lw=1.2, label="Pronóstico (mediana q50)")
ax.plot(d.fecha_hora, alto[sem], color=GRIS, lw=1.0, ls="--", label="Umbrales rodantes (p25 y p75)")
ax.plot(d.fecha_hora, bajo[sem], color=GRIS, lw=1.0, ls="--")
for tipo, color, marca, nombre in (("vender", VERDE, "^", "Vender"), ("retener", NARANJA, "v", "Retener")):
    m = (senal[sem] == tipo).to_numpy()
    ax.scatter(d.fecha_hora[m], d.q50[m], color=color, marker=marca, s=22, zorder=3, label=nombre)
ax.set_ylabel("Precio de bolsa (COP/kWh)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
ax.set_title("Motor de decisión, método «rodante» con filtro intradía, rol generador (13 al 19 de julio de 2026)", fontsize=11)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8.5)
fig.savefig(OUT / "motor_operacion.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("acciones en la semana:", senal[sem].value_counts().to_dict())

# ---------------------------------------------------------------- 3) capturas actuales del dashboard (manual de usuario)
IMG = RAIZ / "docs" / "manual" / "img"
for origen, destino in (("op_arriba.png", "dash_operador.png"), ("op_calor.png", "dash_calor.png"), ("an_todo.png", "dash_analista.png"),
                        ("op_semana.png", "dash_semana.png")):
    shutil.copyfile(IMG / origen, OUT / destino)
print("ok")
