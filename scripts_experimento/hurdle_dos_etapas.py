# -*- coding: utf-8 -*-
"""
Punto 3 de la lista de alternativas: modelo de dos etapas (hurdle), inspirado en el paper de AES
Colombia (Herrera-Mejia et al. 2025) que revisamos hoy -- en vez de pedirle a un solo modelo
que maneje tanto los precios normales como los picos con los mismos parametros, se separa el
problema en dos:
  1) Un CLASIFICADOR (XGBoost) predice si la hora va a estar en "regimen alto" (precio > umbral,
     el percentil 80 de precios de TRAIN) usando solo variables conocidas de antemano (hidrologia,
     ONI, calendario, festivos, precio rezagado) -- nunca el precio real de esa hora.
  2) Dos REGRESORES (XGBoost) especializados, uno entrenado SOLO con horas "altas" de train y otro
     SOLO con horas "bajas" -- en vez de uno solo que tiene que aprender ambos comportamientos con
     los mismos parametros.

En la prediccion de 2026, la clasificacion decide a cual de los dos regresores enrutar cada hora.
Ataca directamente el sesgo invertido en los picos que encontramos hoy (sobreestima en precios
bajos, subestima en los altos) -- si un regresor "vive" solo en el regimen alto, no tiene que
promediar su comportamiento con el regimen bajo.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import statsmodels.api as sm
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

RAIZ = Path(r"C:\Users\mgdbj\xm-spot-price-predictor")

df_train = pd.read_csv(RAIZ / "data" / "processed" / "dataset_features_2019_2025.csv", parse_dates=["fecha_hora"])
df_test = pd.read_csv(RAIZ / "data" / "processed" / "dataset_features_2026.csv", parse_dates=["fecha_hora"])

columnas_excluir = ["fecha_hora", "precio_bolsa", "demanda", "generacion", "anio", "mes", "hora", "dia_semana", "dia_anio"]
columnas_features = [c for c in df_train.columns if c not in columnas_excluir]

# --- Umbral de "regimen alto": percentil 80 de precio en TRAIN, fijo (no se recalcula con 2026) ---
umbral = df_train["precio_bolsa"].quantile(0.80)
print(f"Umbral de regimen alto (percentil 80 de train): {umbral:.2f} COP/kWh")

df_train["es_alto"] = (df_train["precio_bolsa"] > umbral).astype(int)
df_test["es_alto_real"] = (df_test["precio_bolsa"] > umbral).astype(int)
print(f"Proporcion de horas altas en train: {df_train['es_alto'].mean()*100:.1f}%")
print(f"Proporcion de horas altas en test (2026): {df_test['es_alto_real'].mean()*100:.1f}%")

# --- Etapa 1: clasificador ---
X_train_clf = df_train[columnas_features]
y_train_clf = df_train["es_alto"]
clf = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.02,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         scale_pos_weight=(1 - y_train_clf.mean()) / y_train_clf.mean())
clf.fit(X_train_clf, y_train_clf)

X_test_clf = df_test[columnas_features]
prob_alto = clf.predict_proba(X_test_clf)[:, 1]
pred_alto = (prob_alto > 0.5).astype(int)

acc = accuracy_score(df_test["es_alto_real"], pred_alto)
f1 = f1_score(df_test["es_alto_real"], pred_alto)
print(f"\nClasificador -- Accuracy: {acc:.3f}  F1: {f1:.3f}")
print(f"Horas clasificadas como 'alto' en 2026: {pred_alto.sum()} de {len(pred_alto)} "
      f"({pred_alto.mean()*100:.1f}%)")

# --- Etapa 2: dos regresores especializados ---
train_alto = df_train[df_train["es_alto"] == 1]
train_bajo = df_train[df_train["es_alto"] == 0]
print(f"\nFilas de entrenamiento -- regimen alto: {len(train_alto)}  |  regimen bajo: {len(train_bajo)}")

reg_alto = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.01,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
reg_alto.fit(train_alto[columnas_features], np.log(train_alto["precio_bolsa"]))

reg_bajo = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.01,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
reg_bajo.fit(train_bajo[columnas_features], np.log(train_bajo["precio_bolsa"]))

# --- Enrutar cada hora de test al regresor que indique el clasificador ---
pred_reg_alto = np.exp(reg_alto.predict(X_test_clf))
pred_reg_bajo = np.exp(reg_bajo.predict(X_test_clf))
pred_hurdle = np.where(pred_alto == 1, pred_reg_alto, pred_reg_bajo)

y_test_real = df_test["precio_bolsa"].values
error = y_test_real - pred_hurdle
mae = np.abs(error).mean()
rmse = np.sqrt((error ** 2).mean())
mape = (np.abs(error) / y_test_real).mean() * 100

print(f"\n{'='*70}\nModelo de dos etapas (hurdle) -- holdout 2026\n{'='*70}")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# --- Diagnostico: sesgo por decil de precio, mismo formato que el resto de la sesion ---
df_diag = pd.DataFrame({"real": y_test_real, "pred": pred_hurdle})
df_diag["decil"] = pd.qcut(df_diag["real"], 10, labels=[f"D{i+1}" for i in range(10)], duplicates="drop")
df_diag["sesgo_pct"] = (df_diag["real"] - df_diag["pred"]) / df_diag["real"] * 100
print("\nSesgo por decil de precio (hurdle):")
print(df_diag.groupby("decil", observed=True)["sesgo_pct"].mean().round(2))

# --- Referencia: XGBoost normal (un solo modelo, mismas features) ---
X_train_normal = df_train[columnas_features]
reg_normal = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.01,
                                subsample=0.8, colsample_bytree=0.8, random_state=42)
reg_normal.fit(X_train_normal, np.log(df_train["precio_bolsa"]))
pred_normal = np.exp(reg_normal.predict(X_test_clf))
mae_normal = np.abs(y_test_real - pred_normal).mean()
mape_normal = (np.abs(y_test_real - pred_normal) / y_test_real).mean() * 100
print(f"\nReferencia -- XGBoost normal (un solo modelo): MAE {mae_normal:.2f}  MAPE {mape_normal:.2f}%")
print(f"Referencia -- ARX+GARCH: MAE 55.76 | MAPE 15.43%")
print(f"Referencia -- Ensamble stacking (NNLS, CV 5 pliegues): MAE 44.51 | MAPE 12.05%")

# --- Prueba de significancia: hurdle vs XGBoost normal ---
d = np.abs(y_test_real - pred_normal) - np.abs(y_test_real - pred_hurdle)
m_dm = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 23})
print(f"\nDM (hurdle vs XGBoost normal): t={m_dm.tvalues[0]:.3f}  p={m_dm.pvalues[0]:.5f}  "
      f"{'HURDLE GANA (sig.)' if m_dm.pvalues[0]<0.05 and m_dm.tvalues[0]>0 else 'sin diferencia / pierde'}")

pd.DataFrame({"fecha_hora": df_test["fecha_hora"], "real": y_test_real, "prediccion": pred_hurdle,
              "prob_alto": prob_alto, "clasificado_alto": pred_alto}).to_csv(
    RAIZ / "data" / "processed" / "resultados" / "hurdle_dos_etapas_2026.csv", index=False
)
print("\nGuardado: hurdle_dos_etapas_2026.csv")
