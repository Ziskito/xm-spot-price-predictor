# -*- coding: utf-8 -*-
"""
rMAE mensual de N-BEATSx en 2026 (Origen 6 del walk-forward), siguiendo la recomendacion de
Lago et al. (2021, Applied Energy): en vez de MAPE (que se distorsiona en precios bajos), usar
rMAE = MAE(modelo) / MAE(naive_7d), con naive_7d = precio_lag168h.

Un rMAE < 1 significa que el modelo le gana al naive semanal; > 1 significa que pierde contra
el naive semanal. Se compara mes a mes contra el MAPE ya calculado antes, para ver si el
"sesgo invertido" y los meses "peores" cambian de lectura.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

wf = pd.read_csv(RAIZ / "data/processed/resultados/walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
nb = wf[(wf["origen"] == "Origen 6") & (wf["modelo"] == "N-BEATSx")].copy()

feats = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
nb = nb.merge(feats[["fecha_hora", "precio_lag168h"]], on="fecha_hora", how="left")
nb = nb.dropna(subset=["precio_lag168h"])

nb["mes"] = nb["fecha_hora"].dt.to_period("M")
nb["err_abs_modelo"] = (nb["real"] - nb["prediccion"]).abs()
nb["err_abs_naive7d"] = (nb["real"] - nb["precio_lag168h"]).abs()
nb["ape"] = nb["err_abs_modelo"] / nb["real"] * 100

print(f"{'Mes':10s} {'MAE modelo':>10s} {'MAE naive7d':>12s} {'rMAE':>8s}  {'MAPE':>7s}   Lectura")
print("=" * 80)
filas = []
for mes, g in nb.groupby("mes"):
    mae_modelo = g["err_abs_modelo"].mean()
    mae_naive = g["err_abs_naive7d"].mean()
    rmae = mae_modelo / mae_naive
    mape = g["ape"].mean()
    lectura = "gana al naive 7d" if rmae < 1 else "PIERDE contra naive 7d"
    filas.append({"mes": str(mes), "mae_modelo": mae_modelo, "mae_naive7d": mae_naive, "rmae": rmae, "mape": mape})
    print(f"{str(mes):10s} {mae_modelo:10.2f} {mae_naive:12.2f} {rmae:8.3f}  {mape:6.2f}%   {lectura}")

print("\n" + "=" * 80)
tot_modelo = nb["err_abs_modelo"].mean()
tot_naive = nb["err_abs_naive7d"].mean()
print(f"TOTAL 2026  MAE modelo={tot_modelo:.2f}  MAE naive7d={tot_naive:.2f}  rMAE={tot_modelo/tot_naive:.3f}  MAPE={nb['ape'].mean():.2f}%")

df_out = pd.DataFrame(filas)
df_out.to_csv(RAIZ / "data" / "processed" / "resultados" / "rmae_mensual_nbeatsx_2026.csv", index=False)
print("\nGuardado: rmae_mensual_nbeatsx_2026.csv")

# --- Correlacion entre el ranking de MAPE y el ranking de rMAE: coinciden los meses "peores"? ---
corr = df_out["mape"].corr(df_out["rmae"])
print(f"\nCorrelacion MAPE vs rMAE entre meses: {corr:.3f}")
print("\nRanking por MAPE (peor a mejor):")
print(df_out.sort_values("mape", ascending=False)[["mes", "mape", "rmae"]].to_string(index=False))
print("\nRanking por rMAE (peor a mejor):")
print(df_out.sort_values("rmae", ascending=False)[["mes", "mape", "rmae"]].to_string(index=False))
