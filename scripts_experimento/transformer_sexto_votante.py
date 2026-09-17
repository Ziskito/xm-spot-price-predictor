# -*- coding: utf-8 -*-
"""
Idea #2 de la lista inspirada en los papers colombianos: sumar un modelo tipo Transformer como
sexto votante del ensemble de stacking, inspirado en el T2V-TE de Cantillo-Luna et al. (Energies
2023). No se busca que le gane a N-BEATSx solo -- la logica es la misma que ya gano hoy con el
stacking: la diversidad entre familias (no el acierto individual) es lo que mejora el ensamble.
Un Transformer tiene un sesgo de aprendizaje distinto (atencion global vs. la descomposicion
basis-expansion de N-BEATS), asi que sus errores deberian estar menos correlacionados con los
de N-BEATSx/N-HiTS que XGBoost lo esta (que el meta-modelo NNLS ya casi descarto, peso ~0.004).

Se usa VanillaTransformer de neuralforecast (misma familia de API que N-BEATSx/N-HiTS, mismas
exogenas, mismo input_size=168h), entrenado UNA vez sobre Origen 6 (2026) con la misma
metodologia de walk-forward que los demas modelos del proyecto.
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

columnas_excluir = ["fecha_hora", "precio_bolsa", "demanda", "generacion", "anio", "mes", "hora", "dia_semana", "dia_anio"]
columnas_features = [c for c in df_completo.columns if c not in columnas_excluir]

CORTE = "2026-01-01"
train = df_completo[df_completo["fecha_hora"] < CORTE].dropna(subset=columnas_features).copy()
test = df_completo[df_completo["fecha_hora"] >= CORTE].copy()
print(f"Entrenamiento: {len(train)} filas | Prueba (Origen 6, 2026): {len(test)} filas")

hist_exog_dl = ["volumen_embalses", "aportes_hidricos", "demanda_lag24h"]
futr_exog_continua_dl = ["oni"]
futr_exog_ya_escalada_dl = ["hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
                             "es_festivo", "festivo_lag24h", "festivo_lag48h", "festivo_lag72h",
                             "festivo_lag168h", "mismatch_festivo_24h", "mismatch_festivo_168h"]
futr_exog_dl = futr_exog_continua_dl + futr_exog_ya_escalada_dl
INPUT_SIZE_DL = 168
MAX_STEPS_DL = 1000

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
print(f"Ventanas de prueba (dias de 2026): {n_windows}")

from neuralforecast import NeuralForecast
from neuralforecast.models import TFT

# VanillaTransformer no soporta hist_exog_list (solo futr_exog) -- TFT (Temporal Fusion
# Transformer) si, y es literalmente la arquitectura disenada para mezclar historico + futuro
# conocido + atencion, mas apropiada para este caso que forzar VanillaTransformer sin
# demanda_lag24h/volumen_embalses/aportes_hidricos.
modelo = TFT(
    h=24, input_size=INPUT_SIZE_DL, hist_exog_list=hist_exog_dl, futr_exog_list=futr_exog_dl,
    max_steps=MAX_STEPS_DL, val_check_steps=100, random_seed=42, enable_progress_bar=False,
    hidden_size=64, n_head=4, dropout=0.1,
)
nf = NeuralForecast(models=[modelo], freq="h")

t0 = time.time()
cv = nf.cross_validation(df=df_nf, n_windows=n_windows, step_size=24)
print(f"Entrenamiento completo en {time.time()-t0:.0f}s")

y_real = cv["y"].values
y_pred = cv["TFT"].values
error = y_real - y_pred
mae = np.abs(error).mean()
mape = (np.abs(error) / y_real).mean() * 100
rmse = np.sqrt((error ** 2).mean())

print(f"\n{'='*70}\nTFT (Temporal Fusion Transformer) -- holdout 2026 (Origen 6)\n{'='*70}")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print("\nReferencia -- N-BEATSx: MAE 46.82 | N-HiTS: MAE 46.44 | XGBoost: MAE 61.39")
print("Referencia -- Ensamble stacking actual (5 modelos): MAE 44.51 | MAPE 12.05%")

pd.DataFrame({"fecha_hora": cv["ds"], "real": y_real, "prediccion": y_pred}).to_csv(
    RAIZ / "data" / "processed" / "resultados" / "transformer_2026.csv", index=False
)
print("\nGuardado: transformer_2026.csv")

# ================== Re-armar el ensamble de stacking con 6 modelos ==================
print(f"\n{'='*70}\nRe-ajustando el ensamble de stacking con Transformer como 6to votante\n{'='*70}")

from scipy.optimize import nnls

MODELOS_5 = ["Persistencia", "XGBoost", "ARX+GARCH", "N-BEATSx", "N-HiTS"]
crudas = pd.read_csv(RAIZ / "data/processed/resultados/walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
datos_o6 = crudas[crudas["origen"] == "Origen 6"]
pivot = datos_o6.pivot_table(index="fecha_hora", columns="modelo", values="prediccion")
real_o6 = datos_o6.groupby("fecha_hora")["real"].first()
pivot = pivot.join(real_o6).dropna()

transf = pd.DataFrame({"fecha_hora": cv["ds"], "Transformer": y_pred}).set_index("fecha_hora")
pivot6 = pivot.join(transf, how="inner")
MODELOS_6 = MODELOS_5 + ["Transformer"]
pivot6 = pivot6.dropna(subset=MODELOS_6 + ["real"])
print(f"Filas disponibles con los 6 modelos alineados: {len(pivot6)}")

pivot6["dia"] = pivot6.index.normalize()
dias = np.sort(pivot6["dia"].unique())
rng = np.random.default_rng(42)
dias_shuffled = rng.permutation(dias)
pliegues = np.array_split(dias_shuffled, 5)

predicciones_oos_5 = []
predicciones_oos_6 = []
for dias_test in pliegues:
    dias_train = np.setdiff1d(dias, dias_test)
    train_p = pivot6[pivot6["dia"].isin(dias_train)]
    test_p = pivot6[pivot6["dia"].isin(dias_test)]

    pesos_5, _ = nnls(train_p[MODELOS_5].values, train_p["real"].values)
    pesos_5n = pesos_5 / pesos_5.sum() if pesos_5.sum() > 0 else pesos_5
    pred_5 = test_p[MODELOS_5].values @ pesos_5n

    pesos_6, _ = nnls(train_p[MODELOS_6].values, train_p["real"].values)
    pesos_6n = pesos_6 / pesos_6.sum() if pesos_6.sum() > 0 else pesos_6
    pred_6 = test_p[MODELOS_6].values @ pesos_6n

    for fh, real_v, p5, p6 in zip(test_p.index, test_p["real"].values, pred_5, pred_6):
        predicciones_oos_5.append({"fecha_hora": fh, "real": real_v, "pred": p5})
        predicciones_oos_6.append({"fecha_hora": fh, "real": real_v, "pred": p6})

df_5 = pd.DataFrame(predicciones_oos_5)
df_6 = pd.DataFrame(predicciones_oos_6)

mae_5 = (df_5["real"] - df_5["pred"]).abs().mean()
mape_5 = ((df_5["real"] - df_5["pred"]).abs() / df_5["real"]).mean() * 100
mae_6 = (df_6["real"] - df_6["pred"]).abs().mean()
mape_6 = ((df_6["real"] - df_6["pred"]).abs() / df_6["real"]).mean() * 100

print(f"\nEnsamble 5 modelos (sin Transformer), mismo subconjunto alineado: MAE {mae_5:.2f}  MAPE {mape_5:.2f}%")
print(f"Ensamble 6 modelos (CON Transformer):                              MAE {mae_6:.2f}  MAPE {mape_6:.2f}%")

import statsmodels.api as sm
e5 = (df_5["real"] - df_5["pred"]).abs().values
e6 = (df_6["real"] - df_6["pred"]).abs().values
d = e5 - e6
r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 23})
print(f"\nDM (6 modelos vs 5 modelos): t={r.tvalues[0]:.3f}  p={r.pvalues[0]:.5f}  "
      f"{'6 MODELOS GANA (sig.)' if r.pvalues[0]<0.05 and r.tvalues[0]>0 else 'sin diferencia / pierde'}")

df_6.to_csv(RAIZ / "data" / "processed" / "resultados" / "stacking_ensamble_6modelos_cv_2026.csv", index=False)
print("\nGuardado: stacking_ensamble_6modelos_cv_2026.csv")
