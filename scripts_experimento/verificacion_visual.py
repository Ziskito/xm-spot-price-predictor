# -*- coding: utf-8 -*-
"""
Verificacion visual desde datos crudos: precio real contra prediccion, y el error.

Todo se calcula aqui mismo desde los archivos fuente, sin reusar ningun numero previo, para que
cada cifra del grafico sea comprobable de forma independiente.

Fuentes:
  - data/processed/resultados/combinador_optimo_mape.csv  (modelo actual: real + pred_smape)
  - data/processed/dataset_features_2026.csv              (precio de bolsa crudo, para el salto
                                                           de medianoche)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RES = RAIZ / "data/processed/resultados"
OUT = RAIZ / "scripts_experimento/informe_assets"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
BASE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10,
})


def limpiar(ax, quitar_y=False):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.spines["left"].set_color(BASE)
    if quitar_y:
        ax.set_yticks([])
    ax.tick_params(length=3)


# ============================================================
# CARGA DIRECTA DE FUENTES
# ============================================================
d = pd.read_csv(RES / "combinador_optimo_mape.csv", parse_dates=["fecha_hora"]).sort_values("fecha_hora")
d["hora"] = d["fecha_hora"].dt.hour
y = d["real"].to_numpy()
p = d["pred_smape"].to_numpy()
err = np.abs(y - p)
ape = err / y * 100

print("=" * 78)
print("VERIFICACION DESDE DATOS CRUDOS")
print("=" * 78)
print(f"Archivo: combinador_optimo_mape.csv")
print(f"Filas: {len(d):,}   Periodo: {d.fecha_hora.min()}  ->  {d.fecha_hora.max()}")
print(f"Precio real:      min={y.min():7.2f}  media={y.mean():7.2f}  max={y.max():7.2f}")
print(f"Precio predicho:  min={p.min():7.2f}  media={p.mean():7.2f}  max={p.max():7.2f}")
print()
print(f"MAE   = media(|real - pred|)              = {err.mean():.2f} COP/kWh")
print(f"MAPE  = media(|real - pred| / real) * 100 = {ape.mean():.2f} %")
print(f"R2    = 1 - SSres/SStot                   = {1 - ((y-p)**2).sum()/((y-y.mean())**2).sum():.4f}")
print(f"Correlacion real vs predicho              = {np.corrcoef(y, p)[0,1]:.4f}")

# salto de medianoche, desde el precio crudo
b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv",
                parse_dates=["fecha_hora"]).sort_values("fecha_hora").reset_index(drop=True)
b["hora"] = b["fecha_hora"].dt.hour
pr = b["precio_bolsa"].to_numpy()
salto = {}
for h in range(24):
    idx = np.where(b["hora"].to_numpy() == h)[0]
    idx = idx[idx >= 1]
    salto[h] = np.abs(pr[idx] - pr[idx - 1]).mean()
print()
print("Cambio medio de precio respecto a la hora anterior (precio crudo 2026):")
for h in [22, 23, 0, 1, 2]:
    print(f"   {h:02d}:00   {salto[h]:7.2f} COP/kWh")

fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.45, wspace=0.22)

# ============================================================
# 1) SERIE TEMPORAL: real vs predicho (2 semanas)
# ============================================================
ax = fig.add_subplot(gs[0, :])
ini = pd.Timestamp("2026-06-01")
fin = ini + pd.Timedelta(days=14)
m = (d.fecha_hora >= ini) & (d.fecha_hora < fin)
ax.plot(d.loc[m, "fecha_hora"], d.loc[m, "real"], color=INK, linewidth=1.8, label="Precio real")
ax.plot(d.loc[m, "fecha_hora"], d.loc[m, "pred_smape"], color=BLUE, linewidth=1.6,
        linestyle="--", label="Predicción del modelo")
mae_v = np.abs(d.loc[m, "real"] - d.loc[m, "pred_smape"]).mean()
ax.set_title(f"1. Precio real contra predicción — 2 semanas de junio 2026 (el mes más difícil del año)\n"
             f"MAE en esta ventana: {mae_v:.1f} COP/kWh",
             fontsize=12, color=INK, pad=12, loc="left")
ax.set_ylabel("COP/kWh")
ax.legend(frameon=False, fontsize=10)
limpiar(ax)

# ============================================================
# 2) DISPERSION real vs predicho
# ============================================================
ax = fig.add_subplot(gs[1, 0])
ax.scatter(y, p, s=3, alpha=0.18, color=BLUE, edgecolors="none")
lim = [min(y.min(), p.min()), max(y.max(), p.max())]
ax.plot(lim, lim, color=RED, linewidth=1.4, linestyle="--", label="predicción perfecta")
ax.set_xlabel("Precio real (COP/kWh)")
ax.set_ylabel("Precio predicho (COP/kWh)")
r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
ax.set_title(f"2. Las {len(d):,} horas de 2026, una por punto\nR² = {r2:.3f}   ·   "
             f"correlación = {np.corrcoef(y,p)[0,1]:.3f}", fontsize=11.5, color=INK, pad=12, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")
limpiar(ax)

# ============================================================
# 3) EL SALTO DE MEDIANOCHE (precio crudo, sin modelo)
# ============================================================
ax = fig.add_subplot(gs[1, 1])
horas = list(range(24))
vals = [salto[h] for h in horas]
colores = [RED if h == 0 else BLUE for h in horas]
ax.bar(horas, vals, color=colores, width=0.72)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hora del día")
ax.set_ylabel("COP/kWh")
ax.annotate(f"{salto[0]:.1f}", xy=(0, salto[0]), xytext=(2.2, salto[0] * 0.95),
            color=RED, fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax.annotate(f"{salto[1]:.1f}", xy=(1, salto[1]), xytext=(4.5, salto[1] + 12),
            color=INK2, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
ax.set_title("3. Cuánto cambia el precio respecto a la hora anterior\n"
             "(dato crudo del mercado, sin modelo de por medio)",
             fontsize=11.5, color=INK, pad=12, loc="left")
limpiar(ax)

# ============================================================
# 4) ERROR POR HORA (el grafico que viste), recalculado aqui
# ============================================================
ax = fig.add_subplot(gs[2, :])
g = d.groupby("hora").apply(lambda s: pd.Series({
    "MAPE": (np.abs(s["real"] - s["pred_smape"]) / s["real"]).mean() * 100,
    "MAE": np.abs(s["real"] - s["pred_smape"]).mean(),
}), include_groups=False).reset_index()
g["aporte_%"] = g["MAPE"] / g["MAPE"].sum() * 100
peores = set(g.nlargest(6, "aporte_%")["hora"])
aporte = g.nlargest(6, "aporte_%")["aporte_%"].sum()
colores = [RED if h in peores else BLUE for h in g["hora"]]
ax.bar(g["hora"], g["MAPE"], color=colores, width=0.72)
ax.axhline(ape.mean(), color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
ax.text(23.4, ape.mean() + 0.4, f"media global {ape.mean():.2f}%", color=MUTED, fontsize=9, ha="right")
ax.set_xticks(range(0, 24, 1))
ax.set_xlabel("Hora del día")
ax.set_ylabel("MAPE (%)")
ax.set_title(f"4. Error del modelo por hora del día — las 6 en rojo "
             f"({', '.join(str(h) for h in sorted(peores))}) concentran el {aporte:.1f}% del error total\n"
             f"(recalculado en este mismo script desde el archivo de predicciones)",
             fontsize=11.5, color=INK, pad=12, loc="left")
limpiar(ax)

fig.suptitle("Verificación directa del modelo de 24 horas — todo calculado desde los archivos fuente",
             fontsize=13.5, color=INK, y=0.985)
plt.savefig(OUT / "verificacion_visual.png", dpi=170, bbox_inches="tight")
plt.close()

print()
print("Error por hora (recalculado):")
print(g.round(2).to_string(index=False))
print()
print(f"6 horas peores: {sorted(peores)}  ->  {aporte:.1f}% del error total")
print(f"\nGuardado: {OUT / 'verificacion_visual.png'}")
