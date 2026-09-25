# -*- coding: utf-8 -*-
"""Resultados y figuras de OE3 (motor de decision) para el informe de avance ABET.

Todo se calcula con el codigo real de src/motor_decision.py sobre los contratos de
pronostico registrados en fuentes_pronostico.json, asi las cifras del informe son
exactamente las que produce el motor. Guarda las figuras en informe_oe3_assets/ y
los numeros en informe_oe3_assets/resultados_oe3.json (los lee
actualizar_informe_avance_oe3.py). Estilo igual al de informe_avance_resultados.py.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from motor_decision import (  # noqa: E402
    cargar_fuente_pronostico, comparar_metodos, comparar_metodos_estable,
    elegir_mejor_metodo, elegir_mejor_metodo_estable, evaluar_backtest, generar_senales,
)

OUT = Path(__file__).resolve().parent / "informe_oe3_assets"
OUT.mkdir(exist_ok=True)

AZUL, NARANJA, VERDE, ROJO, GRIS, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#898781", "#4a3aa7"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
                     "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
COLOR_METODO = {"fijo": GRIS, "rodante": AZUL, "banda": NARANJA, "hibrido": VERDE}

PRECIO_HIST = pd.read_csv(RAIZ / "data" / "processed" / "dataset_maestro_2019_2025.csv",
                          usecols=["precio_bolsa"])["precio_bolsa"].values
POTENCIA_KW = 100  # cliente de ejemplo, igual que en pruebas_precision_motor_decision_Rafa.ipynb

res = {"casos": {}, "potencia_kw": POTENCIA_KW}
bandas_por_h = {}

# ---------------------------------------------------------------------------
# 1) Seleccion por promedio (criterio anterior) y por estabilidad (criterio nuevo)
# ---------------------------------------------------------------------------
for h in ["24h", "72h"]:
    bandas, _ = cargar_fuente_pronostico(RAIZ, h)
    bandas_por_h[h] = bandas
    corte = bandas["fecha_hora"].min() + (bandas["fecha_hora"].max() - bandas["fecha_hora"].min()) / 2
    res["corte"] = str(corte.date())
    res["precio_medio_2026"] = float(bandas["real"].mean())
    for rol in ["generador", "comercializador"]:
        t_prom = comparar_metodos(bandas, rol, PRECIO_HIST)
        m_prom, _ = elegir_mejor_metodo(t_prom)
        t_est = comparar_metodos_estable(bandas, rol, PRECIO_HIST)
        m_est, estable = elegir_mejor_metodo_estable(t_est)
        full = t_prom.set_index("metodo")
        est = t_est.set_index("metodo")
        res["casos"][f"{h}|{rol}"] = {
            "metodo_promedio": m_prom,
            "metodo_estable": m_est,
            "es_estable": bool(estable),
            "metodos": {
                m: {
                    "ventaja_anio": None if np.isnan(full.loc[m, "ventaja_cop_kwh"]) else float(full.loc[m, "ventaja_cop_kwh"]),
                    "frec_anio": float(full.loc[m, "frecuencia_accion"]),
                    "ventaja_h1": None if np.isnan(est.loc[m, "ventaja_H1"]) else float(est.loc[m, "ventaja_H1"]),
                    "frec_h1": float(est.loc[m, "frecuencia_H1"]),
                    "ventaja_h2": None if np.isnan(est.loc[m, "ventaja_H2"]) else float(est.loc[m, "ventaja_H2"]),
                    "frec_h2": float(est.loc[m, "frecuencia_H2"]),
                    "estable": bool(est.loc[m, "estable"]),
                }
                for m in full.index
            },
        }

# ---------------------------------------------------------------------------
# 2) Figura: frecuencia de accion por mitad del periodo (la inestabilidad)
# ---------------------------------------------------------------------------
caso = res["casos"]["24h|generador"]["metodos"]
metodos = ["fijo", "rodante", "banda", "hibrido"]
x = np.arange(len(metodos))
h1 = [caso[m]["frec_h1"] * 100 for m in metodos]
h2 = [caso[m]["frec_h2"] * 100 for m in metodos]
fig, ax = plt.subplots(figsize=(8.2, 3.4))
b1 = ax.bar(x - 0.19, h1, 0.36, color="#9cc2ef", label=f"1.ª mitad (hasta {res['corte']})")
b2 = ax.bar(x + 0.19, h2, 0.36, color=AZUL, label="2.ª mitad")
ax.axhspan(10, 40, color=VERDE, alpha=0.10, lw=0)
ax.text(len(metodos) - 0.45, 41.5, "rango válido 10-40 %", color=VERDE, fontsize=9, ha="right")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.0f} %",
                ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(metodos)
ax.set_ylabel("Horas con acción (%)")
ax.set_ylim(0, max(h1 + h2) * 1.18 + 5)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "oe3_frecuencia_mitades.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3) Figura: barrido de percentiles (frontera ventaja vs. frecuencia)
# ---------------------------------------------------------------------------
bandas24 = bandas_por_h["24h"]
pares = [(10, 90), (15, 85), (20, 80), (25, 75), (30, 70), (35, 65)]
barrido = []
for pb, pa in pares:
    s = generar_senales(bandas24, "fijo", "generador", PRECIO_HIST, p_bajo=pb, p_alto=pa)
    bt = evaluar_backtest(bandas24, s, "generador")
    barrido.append({"par": f"{pb}/{pa}", "ventaja": bt["ventaja_cop_kwh"], "frec": bt["frecuencia_accion"] * 100})
res["barrido_percentiles"] = barrido
fig, ax = plt.subplots(figsize=(8.2, 3.4))
fx = [r["frec"] for r in barrido]
fy = [r["ventaja"] for r in barrido]
ax.plot(fx, fy, color=GRIS, lw=1.2, zorder=1)
for r in barrido:
    es_actual = r["par"] == "25/75"
    ax.scatter(r["frec"], r["ventaja"], s=70 if es_actual else 40, color=AZUL if es_actual else GRIS, zorder=2)
    ax.annotate(r["par"] + (" (actual)" if es_actual else ""), (r["frec"], r["ventaja"]),
                textcoords="offset points", xytext=(6, 6), fontsize=9,
                color=AZUL if es_actual else "#52514e", fontweight="bold" if es_actual else "normal")
ax.set_xlabel("Horas con acción (%)")
ax.set_ylabel("Ventaja (COP/kWh)")
fig.tight_layout()
fig.savefig(OUT / "oe3_barrido_percentiles.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4) Figura: ultimo mes, ganancia o perdida total en pesos por metodo
# ---------------------------------------------------------------------------
fin = bandas24["fecha_hora"].max()
ini = fin - pd.Timedelta(days=30)
mask = bandas24["fecha_hora"] > ini
res["ultimo_mes"] = {"inicio": str((ini + pd.Timedelta(hours=1)).date()), "fin": str(fin.date())}
sub = bandas24[mask]
res["ultimo_mes"]["cobertura"] = float(((sub["real"] >= sub["q10"]) & (sub["real"] <= sub["q90"])).mean() * 100)
res["ultimo_mes"]["mae"] = float((sub["real"] - sub["q50"]).abs().mean())
filas = {}
for m in metodos:
    s = generar_senales(bandas24, m, "generador", PRECIO_HIST)  # serie completa: continuidad de la ventana movil
    bt = evaluar_backtest(bandas24[mask], s[mask], "generador")
    v = bt["ventaja_cop_kwh"]
    filas[m] = {
        "ventaja": None if np.isnan(v) else float(v),
        "frec": bt["frecuencia_accion"] * 100,
        "horas": bt["horas_accion"],
        "cop": 0.0 if np.isnan(v) else float(v * bt["horas_accion"] * POTENCIA_KW),
    }
res["ultimo_mes"]["metodos"] = filas

fig, ax = plt.subplots(figsize=(8.2, 3.4))
vals = [filas[m]["cop"] / 1e6 for m in metodos]
bars = ax.bar(metodos, vals, color=[COLOR_METODO[m] for m in metodos], width=0.55)
ax.axhline(0, color="#222222", lw=0.8)
rng = max(vals) - min(min(vals), 0)
for b, m, v in zip(bars, metodos, vals):
    y = v + (rng * 0.04 if v >= 0 else -rng * 0.04)
    txt = f"{v:+.2f} M COP".replace(".", ",")
    ax.text(b.get_x() + b.get_width() / 2, y, txt, ha="center", va="bottom" if v >= 0 else "top",
            fontsize=10, fontweight="bold")
    ax.text(b.get_x() + b.get_width() / 2, -rng * 0.30 if v >= 0 else y - rng * 0.12,
            f"{filas[m]['frec']:.0f} % de las horas", ha="center", va="top", fontsize=8.5, color="#52514e")
ax.set_ylabel("Ganancia / pérdida (millones COP)")
ax.set_ylim(min(min(vals), 0) - rng * 0.45, max(vals) + rng * 0.2)
fig.tight_layout()
fig.savefig(OUT / "oe3_ultimo_mes_pesos.png")
plt.close(fig)

(OUT / "resultados_oe3.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))
