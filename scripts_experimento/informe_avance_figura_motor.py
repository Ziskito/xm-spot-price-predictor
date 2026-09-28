# -*- coding: utf-8 -*-
"""Figura del motor de decisión en operación: una semana de 2026 con el pronóstico, su banda, los umbrales
rodantes y las acciones del método «híbrido» para el generador (el que elige el criterio de estabilidad).

El «híbrido» se reconstruye con src/motor_decision.py: umbral rodante + el mismo filtro de ancho de «banda».
Reproduce exactamente la Tabla 19 del informe (generador: 166,8 COP/kWh, 19,5 % de las horas).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
sys.path.insert(0, str(RAIZ / "src"))
from motor_decision import cargar_fuente_pronostico, evaluar_backtest, generar_senales, umbrales_rodantes  # noqa: E402

FIG = RAIZ / "scripts_experimento" / "informe_avance_assets"
AZUL, NARANJA, VERDE, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
                     "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})

df, _ = cargar_fuente_pronostico(RAIZ, "24h")
df = df.reset_index(drop=True)
hist = pd.read_csv(RAIZ / "data/processed/dataset_maestro_2019_2025.csv", usecols=["precio_bolsa"])["precio_bolsa"].values
senal = generar_senales(df, "rodante", "generador", hist)
ancho = df["q90"] - df["q10"]
filtro = ancho > np.nanpercentile(ancho, 75)
senal = senal.mask(filtro, "esperar")
bt = evaluar_backtest(df, senal, "generador")
assert abs(bt["ventaja_cop_kwh"] - 166.8) < 0.05 and abs(bt["frecuencia_accion"] - 0.195) < 0.001, bt
bajo, alto = umbrales_rodantes(df["q50"], 30, 25, 75)

ini = pd.Timestamp("2026-07-13")
sem = (df["fecha_hora"] >= ini) & (df["fecha_hora"] < ini + pd.Timedelta(days=7))
d = df[sem]
fig, ax = plt.subplots(figsize=(8.2, 3.8))
ax.fill_between(d.fecha_hora, d.q10, d.q90, color=AZUL, alpha=0.13, lw=0, label="Banda [q10, q90]")
ax.plot(d.fecha_hora, d.real, color="#111111", lw=1.0, label="Precio real")
ax.plot(d.fecha_hora, d.q50, color=AZUL, lw=1.2, label="Pronóstico (mediana q50)")
ax.plot(d.fecha_hora, alto[sem], color=GRIS, lw=1.0, ls="--", label="Umbrales rodantes (p25 y p75)")
ax.plot(d.fecha_hora, bajo[sem], color=GRIS, lw=1.0, ls="--")
for tipo, color, marca, nombre in (("vender", VERDE, "^", "Vender"), ("retener", NARANJA, "v", "Retener")):
    m = senal[sem] == tipo
    ax.scatter(d.fecha_hora[m], d.q50[m], color=color, marker=marca, s=22, zorder=3, label=nombre)
m = filtro[sem] & ((d.q50 > alto[sem]) | (d.q50 < bajo[sem]))
ax.scatter(d.fecha_hora[m], d.q50[m], facecolors="none", edgecolors=GRIS, marker="o", s=26, zorder=3,
           label="Esperar por banda ancha")
ax.set_ylabel("Precio de bolsa (COP/kWh)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
ax.set_title("Motor de decisión, método «híbrido», rol generador (13 al 19 de julio de 2026)")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8.5)
fig.savefig(FIG / "f_motor_hibrido.png", bbox_inches="tight", facecolor="white")
print("ok", bt, "acciones en la semana:", senal[sem].value_counts().to_dict())
