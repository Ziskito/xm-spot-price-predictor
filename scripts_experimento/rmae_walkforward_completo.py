# -*- coding: utf-8 -*-
"""
Extiende el rMAE (Lago et al. 2021) a la comparacion walk-forward completa (6 origenes x 5
modelos), no solo a N-BEATSx en 2026.

Nota metodologica importante: cuando se comparan modelos en el MISMO periodo de test (p.ej. la
tabla consolidada de 2026), rMAE = MAE(modelo) / MAE(naive_7d) es solo un reescalado por una
constante -- no puede cambiar el ranking entre modelos (el propio paper lo señala explicitamente).
Donde SI aporta es comparando ENTRE origenes/regimenes distintos: cada origen tiene su propia
dificultad base (el naive funciona peor en regimenes volatiles), y el MAE crudo no lo controla --
dos modelos con la "misma ventaja relativa" sobre el naive pueden verse muy distintos en MAE crudo
solo porque un origen es intrinsecamente mas dificil que otro.

Por eso este script calcula rMAE POR ORIGEN, no solo para el holdout unico de 2026 -- para ver si
la ventaja relativa de N-BEATSx/N-HiTS sobre el naive semanal es estable entre regimenes, mas alla
de la significancia binaria que ya da Diebold-Mariano.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

wf = pd.read_csv(RAIZ / "data/processed/resultados/walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])

a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
feats = pd.concat([a, b], ignore_index=True)[["fecha_hora", "precio_lag168h"]].drop_duplicates("fecha_hora")

wf = wf.merge(feats, on="fecha_hora", how="left")

regimenes = {
    "Origen 1": "La Nina (inicio)", "Origen 2": "La Nina (continuacion)",
    "Origen 3": "La Nina (triple-dip)", "Origen 4": "El Nino (fuerte)",
    "Origen 5": "El Nino (pico)", "Origen 6": "El Nino 2026 (neutral->fuerte)",
}

resultados = []
for origen in sorted(wf["origen"].unique()):
    datos_o = wf[wf["origen"] == origen].dropna(subset=["precio_lag168h"])
    naive = datos_o.drop_duplicates("fecha_hora")
    mae_naive = (naive["real"] - naive["precio_lag168h"]).abs().mean()

    fila = {"origen": origen, "regimen": regimenes.get(origen, ""), "mae_naive7d": mae_naive}
    for modelo in ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"]:
        m = datos_o[datos_o["modelo"] == modelo]
        if len(m) == 0:
            continue
        mae_modelo = (m["real"] - m["prediccion"]).abs().mean()
        fila[f"mae_{modelo}"] = mae_modelo
        fila[f"rmae_{modelo}"] = mae_modelo / mae_naive
    resultados.append(fila)

df_res = pd.DataFrame(resultados)
df_res.to_csv(RAIZ / "data" / "processed" / "resultados" / "rmae_walkforward_completo.csv", index=False)

print("=" * 100)
print("MAE del naive-7d por origen (proxy de 'que tan dificil es el regimen')")
print("=" * 100)
print(df_res[["origen", "regimen", "mae_naive7d"]].to_string(index=False))

print("\n" + "=" * 100)
print("rMAE por origen y modelo (MAE modelo / MAE naive-7d) -- <1 = le gana al naive semanal")
print("=" * 100)
cols_rmae = ["origen"] + [c for c in df_res.columns if c.startswith("rmae_")]
print(df_res[cols_rmae].round(3).to_string(index=False))

print("\n" + "=" * 100)
print("Promedio y desviacion estandar del rMAE entre los 6 origenes, por modelo")
print("(desviacion baja = ventaja relativa ESTABLE entre regimenes, no solo 'significativa')")
print("=" * 100)
for modelo in ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"]:
    col = f"rmae_{modelo}"
    if col in df_res.columns:
        print(f"{modelo:12s} rMAE medio={df_res[col].mean():.3f}  desv.std={df_res[col].std():.3f}  "
              f"min={df_res[col].min():.3f}  max={df_res[col].max():.3f}")

print("\nGuardado: rmae_walkforward_completo.csv")
