# -*- coding: utf-8 -*-
"""
Idea #4 de la lista inspirada en los papers colombianos: precision por tercil (bajo/medio/alto),
inspirado en el paper de AES Colombia (Herrera-Mejia et al. 2025) revisado hoy. Analisis puro
sobre las predicciones YA guardadas del ensemble de stacking (mejor resultado del proyecto,
MAE 44.51 / MAPE 12.05%) -- no requiere reentrenar nada.

Complementa Corr-f/MHD/MPD (Maciejowska et al.) que ya usamos: para una regla de decision de
compra/venta, saber si el modelo acierta el tercil (bajo/medio/alto) del precio real suele ser
mas accionable que el error puntual exacto.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

df = pd.read_csv(RAIZ / "data/processed/resultados/stacking_ensamble_cv_2026.csv", parse_dates=["fecha_hora"])

# --- Terciles definidos sobre el precio REAL del propio periodo evaluado (2026) ---
terciles = df["real"].quantile([1/3, 2/3]).values
print(f"Cortes de tercil (precio real 2026): bajo < {terciles[0]:.2f} <= medio < {terciles[1]:.2f} <= alto")

def a_tercil(x, cortes):
    return np.where(x < cortes[0], "bajo", np.where(x < cortes[1], "medio", "alto"))

df["tercil_real"] = a_tercil(df["real"].values, terciles)
df["tercil_pred"] = a_tercil(df["pred_ensamble"].values, terciles)

acc_exacta = (df["tercil_real"] == df["tercil_pred"]).mean() * 100
print(f"\nPrecision por tercil (exacta): {acc_exacta:.2f}%")

# --- Matriz de confusion ---
orden = ["bajo", "medio", "alto"]
matriz = pd.crosstab(df["tercil_real"], df["tercil_pred"]).reindex(index=orden, columns=orden, fill_value=0)
print("\nMatriz de confusion (filas=real, columnas=predicho):")
print(matriz)

# --- Precision "adyacente" (bajo vs alto es un error grave, bajo vs medio es leve) ---
mapa_num = {"bajo": 0, "medio": 1, "alto": 2}
dist = (df["tercil_real"].map(mapa_num) - df["tercil_pred"].map(mapa_num)).abs()
pct_exacto = (dist == 0).mean() * 100
pct_adyacente = (dist == 1).mean() * 100
pct_grave = (dist == 2).mean() * 100
print(f"\nExacto: {pct_exacto:.2f}%  |  Adyacente (1 tercil de distancia): {pct_adyacente:.2f}%  |  "
      f"Error grave (bajo<->alto): {pct_grave:.2f}%")

# --- Comparacion contra el mejor modelo individual (N-BEATSx) para referencia ---
crudas = pd.read_csv(RAIZ / "data/processed/resultados/walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
nbeatsx = crudas[(crudas["origen"] == "Origen 6") & (crudas["modelo"] == "N-BEATSx")].set_index("fecha_hora")["prediccion"]
df_i = df.set_index("fecha_hora")
merged = df_i.join(nbeatsx.rename("pred_nbeatsx"), how="inner")
merged["tercil_pred_nbeatsx"] = a_tercil(merged["pred_nbeatsx"].values, terciles)
acc_nbeatsx = (merged["tercil_real"] == merged["tercil_pred_nbeatsx"]).mean() * 100
print(f"\nReferencia -- N-BEATSx individual, precision por tercil: {acc_nbeatsx:.2f}%")
print(f"Ensamble de stacking, precision por tercil: {acc_exacta:.2f}%")

df.to_csv(RAIZ / "data" / "processed" / "resultados" / "tercil_accuracy_ensamble.csv", index=False)
print("\nGuardado: tercil_accuracy_ensamble.csv")
