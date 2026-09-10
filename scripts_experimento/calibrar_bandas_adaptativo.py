# -*- coding: utf-8 -*-
"""
Calibracion conforme ADAPTATIVA de las bandas de N-BEATSx.

La calibracion conforme estatica (calibrar_bandas.py) fallo en generalizar: el margen fijo
calculado en ene-abr 2026 (precio medio 202) no sirvio para abr-ago 2026 (precio medio 594,
2.9x mas alto) -- el supuesto de intercambiabilidad de CQR se rompe cuando el propio periodo
de prueba cambia de regimen a medio camino.

Metodo adaptativo (similar en espiritu a Gibbs & Candes 2021, "Adaptive Conformal
Inference"): en vez de un margen fijo, se recalcula cada semana usando SOLO los scores de
conformidad de las semanas anteriores (ventana movil de 30 dias). Esto deja que el margen
crezca cuando el mercado se vuelve mas volatil, en vez de quedarse anclado al nivel de
volatilidad de cuando se calibro por primera vez.

Es tambien la forma realista de operar esto en produccion: el motor de decision (OE3)
recalibraria las bandas cada semana con los errores mas recientes, no una sola vez.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")
RUTA_CRUDO = RAIZ / "data" / "processed" / "resultados" / "pronostico_con_bandas_2026.csv"
RUTA_SALIDA = RAIZ / "data" / "processed" / "resultados" / "pronostico_con_bandas_2026_adaptativo.csv"

NIVEL_OBJETIVO = 0.80
VENTANA_DIAS = 30
DIAS_CALENTAMIENTO = 30  # primeros dias: sin suficiente historia, se usa el margen crudo del propio holdout inicial

df = pd.read_csv(RUTA_CRUDO, parse_dates=["fecha_hora"]).sort_values("fecha_hora").reset_index(drop=True)
df["score"] = np.maximum(df["q10"] - df["real"], df["real"] - df["q90"])
df["fecha"] = df["fecha_hora"].dt.normalize()

fechas_unicas = sorted(df["fecha"].unique())
margenes = {}

for fecha in fechas_unicas:
    ventana_ini = fecha - pd.Timedelta(days=VENTANA_DIAS)
    historia = df[(df["fecha"] < fecha) & (df["fecha"] >= ventana_ini)]
    if len(historia) < 24 * 7:  # menos de una semana de historia -> calentamiento
        margenes[fecha] = df[df["fecha"] < fecha]["score"].quantile(0.90) if len(df[df["fecha"] < fecha]) > 24 else 0.0
        continue
    n = len(historia)
    nivel_ajustado = min(np.ceil((n + 1) * NIVEL_OBJETIVO) / n, 1.0)
    margenes[fecha] = historia["score"].quantile(nivel_ajustado)

df["margen"] = df["fecha"].map(margenes).fillna(0.0)
df["q10_adapt"] = df["q10"] - df["margen"]
df["q90_adapt"] = df["q90"] + df["margen"]

# Evaluacion solo despues del calentamiento, para que sea honesto (el calentamiento no tiene margen fiable)
post_calentamiento = df[df["fecha"] >= fechas_unicas[0] + pd.Timedelta(days=DIAS_CALENTAMIENTO)]

cob_cruda = ((post_calentamiento["real"] >= post_calentamiento["q10"]) &
             (post_calentamiento["real"] <= post_calentamiento["q90"])).mean()
cob_adapt = ((post_calentamiento["real"] >= post_calentamiento["q10_adapt"]) &
             (post_calentamiento["real"] <= post_calentamiento["q90_adapt"])).mean()
ancho_crudo = (post_calentamiento["q90"] - post_calentamiento["q10"]).mean()
ancho_adapt = (post_calentamiento["q90_adapt"] - post_calentamiento["q10_adapt"]).mean()

print(f"Evaluando desde {post_calentamiento['fecha_hora'].min()} (tras {DIAS_CALENTAMIENTO}d de calentamiento)")
print(f"n = {len(post_calentamiento):,}\n")
print(f"Cobertura CRUDA:       {cob_cruda*100:.1f}%   ancho medio {ancho_crudo:.1f}")
print(f"Cobertura ADAPTATIVA:  {cob_adapt*100:.1f}%   ancho medio {ancho_adapt:.1f}")

# Cobertura por mes, para ver si se adapta de verdad al cambio de regimen
print("\nCobertura ADAPTATIVA por mes:")
post_calentamiento = post_calentamiento.copy()
post_calentamiento["mes"] = post_calentamiento["fecha_hora"].dt.to_period("M")
por_mes = post_calentamiento.groupby("mes").apply(
    lambda g: pd.Series({
        "cobertura_cruda": ((g["real"] >= g["q10"]) & (g["real"] <= g["q90"])).mean() * 100,
        "cobertura_adapt": ((g["real"] >= g["q10_adapt"]) & (g["real"] <= g["q90_adapt"])).mean() * 100,
        "precio_medio": g["real"].mean(),
        "ancho_adapt": (g["q90_adapt"] - g["q10_adapt"]).mean(),
    }), include_groups=False
)
print(por_mes.round(1).to_string())

salida = df[["fecha_hora", "real", "q50", "q10_adapt", "q90_adapt", "margen"]].rename(
    columns={"q10_adapt": "q10", "q90_adapt": "q90"})
salida.to_csv(RUTA_SALIDA, index=False)
print(f"\nContrato ADAPTATIVO para OE3 guardado en: {RUTA_SALIDA}")
