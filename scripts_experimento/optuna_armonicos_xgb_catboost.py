# -*- coding: utf-8 -*-
"""
Idea #1 (busqueda Bayesiana de hiperparametros, inspirada en el paper T2V-TE de Energies que
afino TODOS sus baselines con busqueda Bayesiana) + Idea #3 (armonicos adicionales de calendario,
inspirado en la representacion temporal explicita de Time2Vec -- en vez de aprenderla, se agregan
2do y 3er armonico fijos de hora-del-dia y dia-del-anio, mas baratos de probar).

Se afinan XGBoost y CatBoost (los dos modelos "rapidos" de arbol que alimentan el ensemble de
stacking) con Optuna, sobre las features actuales + los armonicos extra. Validacion: train hasta
2024, validacion en 2025 (mismo criterio de siempre en el proyecto: 2025 se reserva para elegir
hiperparametros, nunca se toca 2026). Se reentrena con el mejor set de hiperparametros sobre TODO
2019-2025 y se evalua en el holdout real de 2026, comparando contra los XGBoost/CatBoost actuales.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import optuna
import xgboost as xgb
from catboost import CatBoostRegressor
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

a = pd.read_csv(RAIZ / "data/processed/dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
b = pd.read_csv(RAIZ / "data/processed/dataset_features_2026.csv", parse_dates=["fecha_hora"])
df_completo = pd.concat([a, b], ignore_index=True).sort_values("fecha_hora").reset_index(drop=True)

# --- Armonicos adicionales (2do y 3er orden), idea #3 ---
def agregar_armonicos(df):
    df = df.copy()
    df["hora_sin2"] = np.sin(4 * np.pi * df["hora"] / 24)
    df["hora_cos2"] = np.cos(4 * np.pi * df["hora"] / 24)
    df["hora_sin3"] = np.sin(6 * np.pi * df["hora"] / 24)
    df["hora_cos3"] = np.cos(6 * np.pi * df["hora"] / 24)
    df["dia_anio_sin2"] = np.sin(4 * np.pi * df["dia_anio"] / 365.25)
    df["dia_anio_cos2"] = np.cos(4 * np.pi * df["dia_anio"] / 365.25)
    df["dia_anio_sin3"] = np.sin(6 * np.pi * df["dia_anio"] / 365.25)
    df["dia_anio_cos3"] = np.cos(6 * np.pi * df["dia_anio"] / 365.25)
    return df

df_completo = agregar_armonicos(df_completo)

columnas_excluir = ["fecha_hora", "precio_bolsa", "demanda", "generacion", "anio", "mes", "hora", "dia_semana", "dia_anio"]
columnas_features = [c for c in df_completo.columns if c not in columnas_excluir]
print(f"Total features (incluyendo {8} armonicos extra): {len(columnas_features)}")

train_val = df_completo[df_completo["fecha_hora"] < "2026-01-01"].dropna(subset=columnas_features)
test = df_completo[df_completo["fecha_hora"] >= "2026-01-01"].dropna(subset=columnas_features)

train = train_val[train_val["fecha_hora"] < "2025-01-01"]
val = train_val[train_val["fecha_hora"] >= "2025-01-01"]

X_train, y_train = train[columnas_features], np.log(train["precio_bolsa"])
X_val, y_val_real = val[columnas_features], val["precio_bolsa"].values
X_test, y_test_real = test[columnas_features], test["precio_bolsa"].values
X_trainval, y_trainval = train_val[columnas_features], np.log(train_val["precio_bolsa"])

print(f"Train: {len(train)}  Val (2025): {len(val)}  Test (2026): {len(test)}")

N_TRIALS = 30

# ============================== XGBoost ==============================
def objetivo_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
    }
    m = xgb.XGBRegressor(**params)
    m.fit(X_train, y_train)
    pred = np.exp(m.predict(X_val))
    return np.abs(y_val_real - pred).mean()

print("\nAfinando XGBoost con Optuna...")
estudio_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
estudio_xgb.optimize(objetivo_xgb, n_trials=N_TRIALS, show_progress_bar=False)
print(f"Mejor MAE validacion (2025): {estudio_xgb.best_value:.2f}")
print(f"Mejores params: {estudio_xgb.best_params}")

modelo_xgb_final = xgb.XGBRegressor(**estudio_xgb.best_params, random_state=42)
modelo_xgb_final.fit(X_trainval, y_trainval)
pred_xgb_test = np.exp(modelo_xgb_final.predict(X_test))
mae_xgb = np.abs(y_test_real - pred_xgb_test).mean()
mape_xgb = (np.abs(y_test_real - pred_xgb_test) / y_test_real).mean() * 100
print(f"\nXGBoost afinado -- holdout 2026: MAE {mae_xgb:.2f}  MAPE {mape_xgb:.2f}%")
print("Referencia -- XGBoost sin afinar: MAE 61.15 | MAPE 15.89%")

# ============================== CatBoost ==============================
def objetivo_cb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "loss_function": "MAE",
        "random_seed": 42,
        "verbose": False,
        "boosting_type": "Ordered",
    }
    m = CatBoostRegressor(**params)
    m.fit(X_train, y_train)
    pred = np.exp(m.predict(X_val))
    return np.abs(y_val_real - pred).mean()

print("\nAfinando CatBoost con Optuna...")
estudio_cb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
estudio_cb.optimize(objetivo_cb, n_trials=N_TRIALS, show_progress_bar=False)
print(f"Mejor MAE validacion (2025): {estudio_cb.best_value:.2f}")
print(f"Mejores params: {estudio_cb.best_params}")

modelo_cb_final = CatBoostRegressor(**estudio_cb.best_params, loss_function="MAE", random_seed=42,
                                      verbose=False, boosting_type="Ordered")
modelo_cb_final.fit(X_trainval, y_trainval)
pred_cb_test = np.exp(modelo_cb_final.predict(X_test))
mae_cb = np.abs(y_test_real - pred_cb_test).mean()
mape_cb = (np.abs(y_test_real - pred_cb_test) / y_test_real).mean() * 100
print(f"\nCatBoost afinado -- holdout 2026: MAE {mae_cb:.2f}  MAPE {mape_cb:.2f}%")
print("Referencia -- CatBoost sin afinar (ordered boosting default): MAE 57.38 | MAPE 15.19%")

# --- Pruebas de significancia vs versiones sin afinar ---
crudas = pd.read_csv(RAIZ / "data/processed/resultados/walkforward_predicciones_crudas.csv", parse_dates=["fecha_hora"])
xgb_o6 = crudas[(crudas["origen"] == "Origen 6") & (crudas["modelo"] == "XGBoost")].set_index("fecha_hora")["prediccion"]
cb_actual = pd.read_csv(RAIZ / "data/processed/resultados/catboost_2026.csv", parse_dates=["fecha_hora"]).set_index("fecha_hora")["prediccion"]

df_xgb_new = pd.DataFrame({"fecha_hora": test["fecha_hora"].values, "real": y_test_real, "pred_new": pred_xgb_test}).set_index("fecha_hora")
merged_xgb = df_xgb_new.join(xgb_o6.rename("pred_old"), how="inner")
d = np.abs(merged_xgb["real"] - merged_xgb["pred_old"]) - np.abs(merged_xgb["real"] - merged_xgb["pred_new"])
r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 23})
print(f"\nDM (XGBoost afinado vs sin afinar): t={r.tvalues[0]:.3f}  p={r.pvalues[0]:.5f}  "
      f"{'GANA (sig.)' if r.pvalues[0]<0.05 and r.tvalues[0]>0 else 'sin diferencia / pierde'}")

df_cb_new = pd.DataFrame({"fecha_hora": test["fecha_hora"].values, "real": y_test_real, "pred_new": pred_cb_test}).set_index("fecha_hora")
merged_cb = df_cb_new.join(cb_actual.rename("pred_old"), how="inner")
d2 = np.abs(merged_cb["real"] - merged_cb["pred_old"]) - np.abs(merged_cb["real"] - merged_cb["pred_new"])
r2 = sm.OLS(d2, np.ones_like(d2)).fit(cov_type="HAC", cov_kwds={"maxlags": 23})
print(f"DM (CatBoost afinado vs sin afinar): t={r2.tvalues[0]:.3f}  p={r2.pvalues[0]:.5f}  "
      f"{'GANA (sig.)' if r2.pvalues[0]<0.05 and r2.tvalues[0]>0 else 'sin diferencia / pierde'}")

pd.DataFrame({"fecha_hora": test["fecha_hora"].values, "real": y_test_real,
              "pred_xgb_afinado": pred_xgb_test, "pred_catboost_afinado": pred_cb_test}).to_csv(
    RAIZ / "data" / "processed" / "resultados" / "optuna_armonicos_xgb_catboost_2026.csv", index=False
)
print("\nGuardado: optuna_armonicos_xgb_catboost_2026.csv")
