# -*- coding: utf-8 -*-
"""Genera los graficos del informe de Word, con la paleta validada del proyecto."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

RES = Path(r"C:\Users\mgdbj\xm-spot-price-predictor\data\processed\resultados")
OUT = Path(r"C:\Users\mgdbj\xm-spot-price-predictor\scripts_experimento\informe_assets")
OUT.mkdir(exist_ok=True)

# ---- paleta validada (dataviz skill) ----
BLUE = "#2a78d6"      # nuestro modelo
ORANGE = "#eb6834"    # comparacion / referencia externa
AQUA = "#1baf7a"       # mejora / bueno
YELLOW = "#eda100"
VIOLET = "#4a3aa7"
RED = "#e34948"        # peor / antes
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 11,
})


def limpiar_ejes(ax, quitar_y=True):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(not quitar_y)
    ax.spines["bottom"].set_color(BASE)
    if quitar_y:
        ax.set_yticks([])
    ax.tick_params(length=0)


def etiquetas_barras(ax, barras, fmt="{:.2f}", color=INK, dy=0.02, fontsize=10, fontweight="normal"):
    ymax = max(b.get_height() for b in barras)
    for b in barras:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + ymax * dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize, color=color, fontweight=fontweight)


# =====================================================================
# 1) Evolucion del modelo de 24h a lo largo de la sesion
# =====================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.2))
etapas = ["v4 original\n(5 votantes, QRA)", "+ GARCH-ged\n(colas pesadas)", "+ combinador\nalineado a sMAPE"]
mape = [11.27, 10.86, 10.74]
colores = [MUTED, ORANGE, BLUE]
barras = ax.bar(etapas, mape, color=colores, width=0.55)
etiquetas_barras(ax, barras, "{:.2f}%", fontweight="bold")
ax.set_ylim(0, 13.2)
ax.set_ylabel("MAPE (%)")
ax.set_title("Modelo de 24 horas: evolución del error en esta sesión", fontsize=13, color=INK, pad=14)
limpiar_ejes(ax)
ax.axhline(10.0, color=AQUA, linestyle=(0, (4, 3)), linewidth=1.2)
ax.text(2.55, 10.1, "meta 10%", color=AQUA, fontsize=9, va="bottom", ha="right")
plt.tight_layout()
plt.savefig(OUT / "fig1_evolucion_24h.png", dpi=200)
plt.close()

# =====================================================================
# 2) Evolucion del modelo de 72h (global)
# =====================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.2))
etapas = ["7 votantes\n(vigente original)", "LEAR + 3 (elimina\narboles)", "+ puente al\nensamble de 24h"]
mape = [18.28, 17.75, 16.94]
colores = [MUTED, ORANGE, BLUE]
barras = ax.bar(etapas, mape, color=colores, width=0.55)
etiquetas_barras(ax, barras, "{:.2f}%", fontweight="bold")
ax.set_ylim(0, 21)
ax.set_ylabel("MAPE global (%), horas 1-72")
ax.set_title("Modelo de 72 horas: evolución del error global en esta sesión", fontsize=13, color=INK, pad=14)
limpiar_ejes(ax)
plt.tight_layout()
plt.savefig(OUT / "fig2_evolucion_72h.png", dpi=200)
plt.close()

# =====================================================================
# 3) Error por tramo de horizonte, antes/despues del puente
# =====================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.4))
tramos = ["1-24h", "25-48h", "49-72h"]
antes = [13.00, 18.29, 21.69]
despues = [10.84, 18.29, 21.69]
x = np.arange(len(tramos))
w = 0.32
b1 = ax.bar(x - w / 2, antes, w, label="Antes del puente", color=RED)
b2 = ax.bar(x + w / 2, despues, w, label="Después del puente", color=BLUE)
etiquetas_barras(ax, b1, "{:.1f}%", fontsize=9)
etiquetas_barras(ax, b2, "{:.1f}%", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(tramos)
ax.set_ylabel("MAPE (%)")
ax.set_title("El error crece con la distancia al corte — y el tramo 1-24h\nse corrige con el puente al modelo de 24h", fontsize=12.5, color=INK, pad=14)
limpiar_ejes(ax)
ax.legend(frameon=False, loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "fig3_tramos_puente.png", dpi=200)
plt.close()

# =====================================================================
# 4) Error por hora del dia (concentracion en el pico vespertino)
# =====================================================================
d = pd.read_csv(RES / "diagnostico_error_24h_por_hora.csv")
PEORES_6 = set(d.nlargest(6, "aporte_%")["hora"])  # las 6 horas que mas aportan al error (por MAPE)
horas_txt = ", ".join(str(h) for h in sorted(PEORES_6))
fig, ax = plt.subplots(figsize=(9, 4.4))
colores = [RED if h in PEORES_6 else BLUE for h in d["hora"]]
barras = ax.bar(d["hora"], d["MAPE"], color=colores, width=0.72)
ax.axhline(d["MAPE"].mean(), color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
ax.text(23.3, d["MAPE"].mean() + 0.3, f"media diaria {d['MAPE'].mean():.1f}%", color=MUTED,
        fontsize=9, ha="right")
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hora del día")
ax.set_ylabel("MAPE (%)")
aporte_total = d.nlargest(6, "aporte_%")["aporte_%"].sum()
ax.set_title(f"El error no se reparte parejo: las 6 horas en rojo ({horas_txt})\nconcentran el {aporte_total:.1f}% de todo el error del modelo de 24h", fontsize=12.5, color=INK, pad=14)
limpiar_ejes(ax)
plt.tight_layout()
plt.savefig(OUT / "fig4_error_por_hora.png", dpi=200)
plt.close()

# =====================================================================
# 5) Comparacion con la literatura — panel de 4 metricas (cada paper con la suya)
# =====================================================================
fig, axs = plt.subplots(2, 2, figsize=(10.5, 8.6))

# 5a) MASE vs Nueva Zelanda (Kapoor & Wichitaksorn 2023)
ax = axs[0, 0]
nombres = ["Nuestro\nensamble 24h", "Mejores modelos\nNZ (LE-GARCH-t,\nLEAR)", "Peores modelos\nNZ"]
# Verificado contra las Tablas 5 y 6 de Kapoor & Wichitaksorn (2023, Applied Energy 347):
# mejor MASE global 1.2626 (LE-GARCH-t, CNI), peor 2.5001 (RFE-GARCH, LSI). Son 33 modelos.
vals = [0.914, 1.30, 2.50]
colores = [BLUE, ORANGE, MUTED]
barras = ax.bar(nombres, vals, color=colores, width=0.55)
etiquetas_barras(ax, barras, "{:.2f}", fontweight="bold")
ax.axhline(1.0, color=INK2, linestyle=(0, (3, 2)), linewidth=1)
ax.text(2.45, 1.03, "= ingenuo", color=INK2, fontsize=8.5, ha="right")
ax.set_ylabel("MASE (menor = mejor;\n<1 le gana al ingenuo)")
ax.set_title("Nueva Zelanda — Kapoor &\nWichitaksorn (2023)", fontsize=11.5)
limpiar_ejes(ax)

# 5b) MAPE/NRMSE vs Brasil (Dias et al. 2024) y DECOMP oficial
ax = axs[0, 1]
grupos = ["Nuestro\nensamble 24h", "MLP\n(Dias et al.\n2024)", "DECOMP\n(oficial Brasil)"]
mape_b = [10.72, 14.65, 19.44]
nrmse_b = [21.58, 24.70, 53.89]
x = np.arange(len(grupos))
w = 0.32
b1 = ax.bar(x - w / 2, mape_b, w, label="MAPE", color=BLUE)
b2 = ax.bar(x + w / 2, nrmse_b, w, label="NRMSE", color=VIOLET)
etiquetas_barras(ax, b1, "{:.1f}", fontsize=8.5)
etiquetas_barras(ax, b2, "{:.1f}", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels(grupos, fontsize=9.5)
ax.set_ylabel("% (menor = mejor)")
ax.set_title("Brasil — Dias, Lira & Freire\n(2024) y modelo oficial", fontsize=11.5)
ax.legend(frameon=False, fontsize=9, loc="upper left")
limpiar_ejes(ax)

# 5c) sMAPE / R2 vs Noruega
ax = axs[1, 0]
grupos = ["Nuestro\nensamble 24h", "LightGBM Noruega\n(mejor zona)", "LightGBM Noruega\n(peor zona)"]
smape_n = [10.04, 12.07, 25.32]
colores = [BLUE, AQUA, MUTED]
barras = ax.bar(grupos, smape_n, color=colores, width=0.55)
etiquetas_barras(ax, barras, "{:.2f}%", fontweight="bold")
ax.set_ylabel("sMAPE (%, menor = mejor)")
ax.set_title("Noruega, 5 zonas Nord Pool\n(arXiv 2604.26634)", fontsize=11.5)
limpiar_ejes(ax)

# 5d) Acierto direccional vs Albani et al.
ax = axs[1, 1]
grupos = ["Nuestro\nensamble 24h\n(TAPI diario)", "Albani et al. 2025\n(general)", "Albani et al. 2025\n(alta correlación\nENA-precio)"]
vals = [76.4, 50, 60]
colores = [BLUE, MUTED, ORANGE]
barras = ax.bar(grupos, vals, color=colores, width=0.55)
etiquetas_barras(ax, barras, "{:.0f}%", fontweight="bold")
ax.axhline(50, color=INK2, linestyle=(0, (3, 2)), linewidth=1)
ax.set_ylim(0, 90)
ax.set_ylabel("Acierto direccional (%)")
ax.set_title("Brasil — Albani et al. (2025)\nforward mensual", fontsize=11.5)
limpiar_ejes(ax)

fig.suptitle("Comparación con la literatura de mercados hidro-dominados\n(cada panel usa la métrica que reporta ESE paper — no son horizontes idénticos, ver texto)",
             fontsize=12.5, color=INK, y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig5_comparacion_papers.png", dpi=200, bbox_inches="tight")
plt.close()

# =====================================================================
# 6) Walk-forward por regimen ENSO (6 origenes) — MAE
# =====================================================================
w = pd.read_csv(RES / "walkforward_5origenes.csv") if (RES / "walkforward_5origenes.csv").exists() else None
r = pd.read_csv(RES / "rmae_walkforward_completo.csv")
fig, ax = plt.subplots(figsize=(10, 4.8))
origenes = [o.replace("Origen ", "O") + f"\n{reg}" for o, reg in zip(r["origen"], r["regimen"])]
modelos = [("Persistencia", MUTED), ("XGBoost", YELLOW), ("ARX+GARCH", VIOLET), ("N-BEATSx", BLUE), ("N-HiTS", AQUA)]
x = np.arange(len(origenes))
n = len(modelos)
w_bar = 0.15
for i, (m, c) in enumerate(modelos):
    vals = r[f"mae_{m}"]
    ax.bar(x + (i - n / 2 + 0.5) * w_bar, vals, w_bar, label=m, color=c)
ax.set_xticks(x)
ax.set_xticklabels(origenes, fontsize=8.5)
ax.set_ylabel("MAE (COP/kWh)")
ax.set_title("Validación walk-forward: error por régimen ENSO (6 orígenes temporales,\n2019-2026, cada uno con su propio entrenamiento y prueba)", fontsize=12, color=INK, pad=14)
limpiar_ejes(ax)
ax.legend(frameon=False, ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16))
plt.tight_layout()
plt.savefig(OUT / "fig6_walkforward_regimenes.png", dpi=200, bbox_inches="tight")
plt.close()

# =====================================================================
# 7) Techo teorico del 8% (24h)
# =====================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.4))
etapas = ["Modelo actual", "Objetivo\n\"10% tranquilo\"", "Techo teórico\n(pico perfecto)", "Objetivo\noriginal 8%"]
vals = [10.74, 10.0, 6.66, 8.0]
colores = [BLUE, AQUA, MUTED, RED]
barras = ax.bar(etapas, vals, color=colores, width=0.55)
etiquetas_barras(ax, barras, "{:.2f}%", fontweight="bold")
ax.set_ylim(0, 13.5)
ax.set_ylabel("MAPE (%)")
ax.set_title("¿Se puede llegar a 8%? El techo teórico (acertando PERFECTO\nen las 6 horas más difíciles del día) es 6.66%", fontsize=12, color=INK, pad=14)
limpiar_ejes(ax)
plt.tight_layout()
plt.savefig(OUT / "fig7_techo_8pct.png", dpi=200)
plt.close()

print("Graficos generados en", OUT)
for f in sorted(OUT.glob("*.png")):
    print(" -", f.name)
