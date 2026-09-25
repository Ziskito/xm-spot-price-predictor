# -*- coding: utf-8 -*-
"""Diagrama de bloques del sistema para el informe de avance."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(r"C:\Users\mgdbj\xm-spot-price-predictor\scripts_experimento\informe_avance_assets")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Calibri", "Arial"], "font.size": 10})

fig, ax = plt.subplots(figsize=(8.4, 6.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def caja(x, y, w, h, titulo, texto, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=color + "22", ec=color, lw=1.4))
    ax.text(x + w / 2, y + h - 3.2, titulo, ha="center", va="top", fontsize=10.5, fontweight="bold", color="#1b1b1b")
    ax.text(x + w / 2, y + h - 8.2, texto, ha="center", va="top", fontsize=8.6, color="#333333", linespacing=1.25)


def flecha(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.3))


AZ, NA, VE, VI = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
ax.text(2, 98, "OE1 · Procesamiento de datos", fontsize=9.5, color=AZ, fontweight="bold")
caja(2, 74, 30, 21, "Adquisición", "API REST XM / SIMEM\n(precio, demanda, generación,\nembalses, aportes) + ONI NOAA", AZ)
caja(35, 74, 30, 21, "Sincronización", "Dataset maestro horario\nene-2019 a ago-2026\n66 576 h, sin huecos", AZ)
caja(68, 74, 30, 21, "Caracterización", "Estadística, estacionalidad,\nperiodograma, filtros SG/MA,\ncorrelación cruzada", AZ)
flecha(32.8, 84.5, 34.2, 84.5)
flecha(65.8, 84.5, 67.2, 84.5)

ax.text(2, 69, "OE2 · Pronóstico", fontsize=9.5, color=NA, fontweight="bold")
caja(2, 43, 30, 23, "Características", "33 variables sin fuga:\nrezagos ≥ 24 h, medias móviles,\narmónicos de calendario,\nhidrología y ONI rezagados", NA)
caja(35, 43, 30, 23, "6 modelos", "Persistencia · XGBoost ·\nARX+GARCH · N-BEATSx ·\nN-HiTS · GARCH-ged", NA)
caja(68, 43, 30, 23, "Ensamble y bandas", "QRA ponderado sMAPE por\nfranja de 6 h (24 h) + puente\na 72 h; bandas CQR\nadaptativas (q10, q50, q90)", NA)
flecha(83, 73.2, 17, 66.8)
flecha(32.8, 54.5, 34.2, 54.5)
flecha(65.8, 54.5, 67.2, 54.5)

ax.text(2, 38, "OE3 / OE4 · Decisión y validación", fontsize=9.5, color=VE, fontweight="bold")
caja(2, 12, 30, 23, "Contrato de datos", "fuentes_pronostico.json\nfecha_hora, real, q10,\nq50, q90 por horizonte", VE)
caja(35, 12, 30, 23, "Motor de decisión", "Umbrales por percentil\n(fijo, rodante, banda),\nrol generador/comercializador\ncomprar · vender · esperar", VE)
caja(68, 12, 30, 23, "Dashboard", "Streamlit · vistas Operador\ny Analista · backtest 2026 ·\nvalidación con usuarios\n(pendiente, OE4)", VE)
flecha(83, 42.2, 17, 35.8)
flecha(32.8, 23.5, 34.2, 23.5)
flecha(65.8, 23.5, 67.2, 23.5)
ax.text(50, 4, "Todo el pipeline en Python (código abierto), versionado en GitHub; datos exclusivamente públicos.",
        ha="center", fontsize=8.6, color="#555555", style="italic")
fig.savefig(OUT / "f_diagrama_bloques.png", dpi=220, bbox_inches="tight", facecolor="white")
print("ok")
