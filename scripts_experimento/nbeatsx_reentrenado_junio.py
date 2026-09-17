# -*- coding: utf-8 -*-
"""
Idea nueva #2: en vez de un modelo con pesos congelados en 2025-12-31 (lo que hace el walk-forward
actual -- Origen 6 SI ve el historial real de precios hora a hora al rodar hacia adelante, via su
ventana de entrada de 168h, pero los PESOS de la red nunca se actualizan), se reentrena N-BEATSx
con corte justo antes de junio (2026-06-01), incorporando ya enero-mayo 2026 -- meses que ya
mostraban la aceleracion del ONI -- para ver si eso mejora la prediccion de junio especificamente.

Es la prueba directa de la hipotesis: "un modelo que ve mas cerca en el tiempo el inicio de la
transicion de regimen, predice mejor la transicion misma?" En vez de reentrenar mes a mes (caro),
un solo reentrenamiento con corte justo antes del mes problematico, comparado contra el mismo mes
ya evaluado por el modelo congelado (walkforward_predicciones_crudas.csv, Origen 6, junio).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import time

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
df_completo = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)

CORTE = "2026-06-01"
FIN_TEST = "2026-06-30"

hist_exog_dl = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
futr_exog_continua_dl = ["oni"]
futr_exog_ya_escalada_dl = ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
                             "es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h",
                             "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]
futr_exog_dl = futr_exog_continua_dl + futr_exog_ya_escalada_dl
INPUT_SIZE_DL = 168
MAX_STEPS_DL = 1000

train = df_completo[df_completo["fecha_hora"] < CORTE].dropna(subset=hist_exog_dl + futr_exog_dl + ["precio_bolsa"]).copy()
test = df_completo[(df_completo["fecha_hora"] >= CORTE) & (df_completo["fecha_hora"] <= FIN_TEST)].copy()
print(f"Entrenamiento: {len(train)} filas (hasta {train['fecha_hora'].max()})")
print(f"Prueba: {len(test)} filas ({test['fecha_hora'].min()} a {test['fecha_hora'].max()})")

df_origen = pd.concat([train, test], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)
mascara_train = df_origen["fecha_hora"] < CORTE

df_nf = df_origen[["fecha_hora", "precio_bolsa"] + hist_exog_dl + futr_exog_dl].copy()
for col in hist_exog_dl + futr_exog_continua_dl:
    mu, sigma = df_nf.loc[mascara_train, col].mean(), df_nf.loc[mascara_train, col].std()
    df_nf[col] = (df_nf[col] - mu) / sigma

df_nf["unique_id"] = "precio_bolsa"
df_nf = df_nf.rename(columns={"fecha_hora": "ds", "precio_bolsa": "y"})
df_nf = df_nf[["unique_id", "ds", "y"] + hist_exog_dl + futr_exog_dl]

n_test = int((~mascara_train).sum())
n_windows = n_test // 24
print(f"Ventanas de prueba (dias de junio): {n_windows}")

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx

m_nbeatsx = NBEATSx(h=24, input_size=INPUT_SIZE_DL, hist_exog_list=hist_exog_dl, futr_exog_list=futr_exog_dl,
                      max_steps=MAX_STEPS_DL, val_check_steps=100, random_seed=42, enable_progress_bar=False)
nf = NeuralForecast(models=[m_nbeatsx], freq="h")

t0 = time.time()
cv = nf.cross_validation(df=df_nf, n_windows=n_windows, step_size=24)
print(f"Entrenamiento completo en {time.time()-t0:.0f}s")

y_real = cv["y"].values
y_pred = cv["NBEATSx"].values
error = y_real - y_pred
mae = np.abs(error).mean()
mape = (np.abs(error) / y_real).mean() * 100
rmse = np.sqrt((error ** 2).mean())

print(f"\n{'='*70}\nN-BEATSx reentrenado con corte 2026-06-01 -- junio 2026\n{'='*70}")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"\nReferencia -- N-BEATSx con pesos congelados en 2025-12-31 (walk-forward Origen 6, junio): MAE 91.29 | MAPE 19.78%")

pd.DataFrame({"fecha_hora": cv["ds"], "real": y_real, "prediccion": y_pred}).to_csv(
    RAIZ / "data" / "processed" / "resultados" / "nbeatsx_reentrenado_junio.csv", index=False
)
print("\nGuardado: nbeatsx_reentrenado_junio.csv")
